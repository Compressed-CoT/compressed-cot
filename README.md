# Token Complexity

**Token Complexity** computes a fine-grained measure of reasoning problem difficulty for 
large language models (LLMs),
given a dataset of Chain-of-Thought prompts with varying response lengths. Token complexity
is the minimum number of Chain-of-Thought output tokens required to solve a reasoning task, and we show in this work that this measure is universal across qualitatively different chains-of-thought.

We calculate token complexities for MMLU-Pro Math, GSM8K, and MATH-500. The data is available at
> <https://huggingface.co/datasets/cot-compression/TokenComplexity>

## Features

Token complexity is a useful notion for researchers to study the reasoning capabilities of LLMs.
- **Fine-grained measure of difficulty**: Token complexity offers a fine-grained notion of problem difficulty that is tailored to the LLM, rather than coarse classifications of 'easy' or 'hard'.
- **Assessing adaptive response length**: By looking at correlations of response length with token complexity, we can assess whether the LLM chooses the response length *adaptively*, using shorter responses for easier questions.
- **Improving the trade-off between response length and accuracy**: By predicting token complexity,
one can tailor response length precisely to the problem, reducing inference costs.

## File Structure

The following files are involved in reproducing the results in the paper.
- 📂 **`token_complexity.py`** – Core script for analyzing token complexity.
- 📂 **`plots.py`** – Reproduces plots in the paper.
- 📂 **`tables.py`** – Reproduces tables in the paper.
- 📂 **`cot_prompts.json`** – Chain-of-Thought prompts used to induce response length variation.

In order to generate the dataset from scratch, run the following scripts
- 📂 **`run_gsm8k.py`** – Evaluates LLMs under multiple prompts for the GSM8K dataset.
- 📂 **`run_math.py`** – Evaluates LLMs under multiple prompts for the MATH-500 dataset.
- 📂 **`run_mmlu_pro.py`** – Evaluates LLMs under multiple prompts for a subset of questions in MMLU-Pro Math.

## Installation

Ensure you have Python installed, then install the necessary dependencies:

```
python3 -m pip install -r requirements.txt
```

To immediately reproduce the results in the paper, run
```
python3 tables.py
python3 plots.py
```