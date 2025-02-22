import os
import openai
import anthropic
from openai import OpenAI
from anthropic import Anthropic
import json
import re
from tqdm import tqdm, trange

import time
from datasets import load_dataset
import argparse

import numpy as np

EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()

def check_equality(sampler, expr1: str, expr2: str):
    prompt = EQUALITY_TEMPLATE % {"expression1": expr1, "expression2": expr2}
    response = sampler(dict(content=prompt, role="user"))
    return 1*(response.lower().strip() == "yes")

def extract_answer(response):
    patterns = [
        r"(?i)Final Answer\s*:\s*.*",
        r"(?i)Answer\s*:\s*.*",
        r"(?i)Ans\s*:\s*.*",
        r"(?i)Ans\s*:\s*.*",
        r"(?i)An\s*:\s*.*",
        r"(?i)An\s*:\s*.*",
    ]
    extracted_answer = None
    for pattern in patterns:
        match = re.search(pattern, response, flags = re.DOTALL)

        if match:
            extracted_answer = match.group(0).replace('(','').replace(')','')
            break

    return extracted_answer

def score_response(sampler, response, correct_answer):
    extracted_answer = extract_answer(response)
    score = check_equality(sampler, correct_answer, extracted_answer)

    return extracted_answer, score

def load_math():
    dataset = load_dataset("HuggingFaceH4/MATH-500", 'default')
    test_df = dataset["test"]
    return test_df

def format_example(question, cot_content='Think step by step before answering.'):

    example = '''Answer the following question. {}
    
Question: {}

The last line of your response should be of the following format: 'Answer: $ANSWER' (without quotes) where ANSWER is your final answer.
'''.format(cot_content, question)
    
    return example

def run_eval(client, provider, model, cot, run_id, file_to_dump, verify_api_key, subjects = ['math'], seed = 91, n_subset = 500, max_tokens = 3000):

    print("Running MATH500 on {}-{} with CoT prompt: {}.".format(provider, model, run_id))
    print("Prompt ", cot, "\n")

    verifier = lambda x: OpenAI(api_key=verify_api_key).chat.completions.create(
            model='gpt-4o-mini-2024-07-18',
            messages=[x],
            temperature=0,
            max_tokens=1000,
        ).choices[0].message.content

    # Load Data
    test_df = load_math()

    # Load all questions from subjects
    questions = [q for q in test_df]
    # rng = np.random.default_rng(seed = seed)
    # rng.shuffle(questions)
    
    prompts = [format_example(q['problem'], cot_content=cot) for q in questions][:n_subset]
    corrects = [q['answer'] for q in questions][:n_subset]

    print(prompts[0])
    
    result = {
        "model": model,
        "run_id": run_id,
        "corrects": corrects,
        "prompts": prompts,
        "grading": [],
        "sampled_texts": [],
        "token_count": [],
        "answers": []
    }

    for i in trange(len(prompts)):
        prompt = prompts[i]
        correct_letter = corrects[i]

        trial = 0
        success = False
        while not success:
            if provider in ['openai', 'groq', 'openrouter', 'together']:
                try:
                    message = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=max_tokens,
                    )
                    success = True
                    response = message.choices[0].message.content
                    token_count = message.usage.completion_tokens

                except openai.RateLimitError as e:
                    print(e)
                    exception_backoff = 2**trial  # exponential back off
                    print(
                        f"{provider} rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    )
                    time.sleep(exception_backoff)
                    trial += 1
            elif provider == 'anthropic':
                try:
                    message = client.messages.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=max_tokens,
                    )
                    success = True
                    response = message.content[0].text
                    token_count = message.usage.output_tokens
                except anthropic.InternalServerError as e:
                    print(e)
                    exception_backoff = 2**trial  # exponential back off
                    print(
                        f"{provider} overloaded so wait and retry {trial} after {exception_backoff} sec",
                    )
                    time.sleep(exception_backoff)
                    trial += 1
                except anthropic.RateLimitError as e:
                    print(e)
                    exception_backoff = 2**trial  # exponential back off
                    print(
                        f"{provider} rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    )
                    time.sleep(exception_backoff)
                    trial += 1
            else:
                print('Not a valid provider')

        result["sampled_texts"].append(response)
        extracted_answer, score = score_response(verifier, response, correct_letter)
        
        result["grading"].append(score)
        result["token_count"].append(token_count)
        result["answers"].append(extracted_answer)

        if (i + 1) % 10 == 0:
            with open(file_to_dump, "w") as file:
                json.dump(result, file)

    with open(file_to_dump, "w") as file:
        json.dump(result, file)

def main():
    parser = argparse.ArgumentParser(
        description="Run MATH500 evaluation on a specified model."
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        help="Provider of the model (e.g., 'anthropic', 'openai').",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to evaluate."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=2000, help="Max tokens."
    )
    parser.add_argument(
        "--subject", type=str, nargs="+", help="Subject.", default = 'math'
    )
    parser.add_argument(
        "--prompt_ids",
        type=str,
        default = "",
        help="List of prompts to test.",
    )
    parser.add_argument(
        "--single_prompt_id",
        type=str,
        default = "",
        help="List of prompts to test.",
    )

    parser.add_argument(
        "--redo",
        type=bool,
        default = False,
        help="List of prompts to test.",
    )

    # Parse inputs
    args = parser.parse_args()
    prompt_ids = args.prompt_ids
    provider = args.provider
    model = args.model
    max_tokens = args.max_tokens
    single_prompt_id = args.single_prompt_id
    redo = args.redo

    # Load dictionary of prompts
    with open('./cot_prompts.json', "r") as file:
        cot_prompts = json.load(file)

    if single_prompt_id == "":
        with open(f'./{prompt_ids}.json', "r") as file:
            cot_prompts = json.load(file)
            prompt_ids = cot_prompts.keys()
    else:
        prompt_ids = [single_prompt_id]

    print('prompt_ids ', prompt_ids)
    
    # Load provider API key
    with open(f"./keys/{provider}.secret", "r") as f:
        api_key = f.read().strip()

    with open("./keys/openai.secret", "r") as f:
        verify_api_key = f.read().strip()

    if provider == 'anthropic':
        client = Anthropic(api_key=api_key)
    elif provider == 'openai':
        client = OpenAI(api_key=api_key)
    elif provider == 'openrouter':
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    elif provider == 'together':
        client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=api_key,
        )

    # Run evals
    for prompt_id in prompt_ids:
        # Get prompt and output path
        cot_prompt = cot_prompts[prompt_id]
        output_path = './results/' + "-".join([provider, str(prompt_id), model.replace('/', '-'), "math500"]) + ".json"

        if redo or not os.path.isfile(output_path):
            # Run eval
            run_eval(client, provider, model, cot_prompt, prompt_id, output_path, verify_api_key = verify_api_key, max_tokens = max_tokens)

if __name__ == "__main__":

    main()