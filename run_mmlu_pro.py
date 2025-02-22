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

def score_response(response, correct_answer):
    patterns = [
        r"(?i)Answer\s*:\s*(\([A-J]\))",
        r"(?i)Answer\s*:\s*([A-J])",
        r"(?i)Ans\s*:\s*(\([A-J]\))",
        r"(?i)Ans\s*:\s*([A-J])",
        r"(?i)An\s*:\s*(\([A-J]\))",
        r"(?i)An\s*:\s*([A-J])",
    ]
    extracted_answer = None
    for pattern in patterns:
        match = re.search(pattern, response)

        if match:
            extracted_answer = match.group(1).replace('(','').replace(')','')
            break

    score = 1.0 if extracted_answer == correct_answer else 0.0
    return extracted_answer, score



def preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    res = {}
    for each in res_df:
        if each["category"] not in res:
            res[each["category"]] = []
        res[each["category"]].append(each)
    return res

def load_mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df

def format_example(question, options, cot_content='Think step by step before answering.'):
    # if cot_content == "":
    #     cot_content = "Let's think step by step."
    # if cot_content.startswith("A: "):
    #     cot_content = cot_content[3:]

    example = '''Answer the following multiple choice question. {}
    
Question: {}

Options:
    
'''.format(cot_content, question)
    
    # example = "Question: {}\nOptions: "
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += "{}) {}\n".format(choice_map[i], opt)

    example += f"The last line of your response should be of the following format: 'Answer: ($LETTER)' (without quotes) where LETTER is one of ABCDEFGHIJ."
    # Chain of thought
    return example


def run_eval(client, provider, model, cot, run_id, file_to_dump, subjects = ['math'], seed = 91, n_subset = 500, max_tokens = 3000):

    print("Running MMLU-Pro on {}-{} with CoT prompt: {}.".format(provider, model, run_id))
    print("Prompt ", cot, "\n")

    # Load Data
    test_df, _ = load_mmlu_pro()

    # Load all questions from subjects
    questions = [q for subject in subjects for q in test_df[subject]]
    rng = np.random.default_rng(seed = seed)
    rng.shuffle(questions)
    
    prompts = [format_example(q['question'], q['options'], cot_content=cot) for q in questions][:n_subset]
    corrects = [q['answer'] for q in questions][:n_subset]
    
    result = {
        "model": model,
        "run_id": run_id,
        "corrects": corrects,
        "prompts": prompts,
        "grading": [],
        "sampled_texts": [],
        "token_count": [],
        "answers": [],
        # "answer_probs": []
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
                        # logprobs = True
                    )
                    success = True
                    response = message.choices[0].message.content
                    token_count = message.usage.completion_tokens
                    # answer_probs = np.exp(message.choices[0].logprobs.content[-2].logprob)

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
        extracted_answer, score = score_response(response, correct_letter)
        
        result["grading"].append(score)
        result["token_count"].append(token_count)
        result["answers"].append(extracted_answer)
        # result["answer_probs"].append(answer_probs)

        if (i + 1) % 10 == 0:
            with open(file_to_dump, "w") as file:
                json.dump(result, file)

    with open(file_to_dump, "w") as file:
        json.dump(result, file)

def main():
    parser = argparse.ArgumentParser(
        description="Run MMLU Pro evaluation on a specified model."
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
        output_path = './results/' + "-".join([provider, str(prompt_id), model.replace('/', '-'), "mmlu-pro"]) + ".json"
        
        # Run eval
        if redo or not os.path.isfile(output_path):
            run_eval(client, provider, model, cot_prompt, prompt_id, output_path, max_tokens = max_tokens)

if __name__ == "__main__":

    main()