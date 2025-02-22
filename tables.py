import numpy as np
import json
from collections import defaultdict
from token_complexity import token_complexity
from datasets import load_dataset
import scipy

if __name__ == "__main__":

    # Table
    benches = [
                'math500', 
                'gsm8k', 
                'mmlu'
    ]

    models = [
            ('openai', 'gpt-4o-2024-11-20'),
            ('openai', 'gpt-4o-mini-2024-07-18'),
            ('anthropic', 'claude-3-5-sonnet-20241022'),
            ('anthropic', 'claude-3-5-haiku-20241022'),
            ('together', 'meta-llama-Llama-3.3-70B-Instruct-Turbo'),
    ]

    table_results = defaultdict(dict)
    correlations = {}

    # Loop over benchmarks and models
    for bench in benches:
        dataset = load_dataset("cot-compression/TokenComplexity", split = bench)
        correlations[bench] = {}
        for provider, model in models:
            correlations[bench][model] = {}

            data_dict = {a['prompt_id']: a for a in dataset[model]}

            all_grades = []
            all_token_counts = []
            
            # Collect scores and token counts for all prompts
            for key, item in data_dict.items():
                all_grades.append(item['grading'])
                all_token_counts.append(item['token_count'])

            token_reqs, accuracies, trivials = token_complexity(all_grades, all_token_counts)

            table_results[bench][model] = {}
            table_results[bench][model]['num_prompts'] = len(all_grades)
            
            # Table: Record accuracy of the optimal token threshold classifier
            table_results[bench][model]['accuracy'] = np.mean(accuracies)

            discrep = []

            # Table: Record discrepancy between the accuracy of each prompt
            # vs the predicted accuracy under the token complexity hypothesis
            for key, item in data_dict.items():
                y = np.mean(item['grading'])
                token_req_pred = np.mean(1.*(item['token_count'] >= np.array(token_reqs)))
                discrep.append(np.abs(token_req_pred - y) / token_req_pred) 

            table_results[bench][model]['discrep'] = np.mean(discrep)

            default_cot_results = data_dict['DefaultCoT']
            table_results[bench][model]['default_cot_len'] = np.mean(default_cot_results['token_count'])

            concise_cot_results = data_dict['BeConcise']
            table_results[bench][model]['concise_cot_len'] = np.mean(concise_cot_results['token_count'])

            # Table: Lower limit
            table_results[bench][model]['T_star'] = np.mean([a for a in token_reqs if a < float('inf')])

            table_results[bench][model]['concise_token_reduction'] = np.mean(default_cot_results['token_count']) / np.mean(concise_cot_results['token_count'])
            table_results[bench][model]['optimal_reduction'] = np.mean(default_cot_results['token_count']) / np.mean([a for a in token_reqs if a < float('inf')])

            
            # Get correlations with true token complexities
            for key, item in data_dict.items():

                tokens_and_complexities = list(zip(item['token_count'], token_reqs))
                tokens_no_inf = np.array([x for x,y in tokens_and_complexities if y < float('inf')])
                reqs_no_inf = np.array([y for _,y in tokens_and_complexities if y < float('inf')])
                
                if np.isnan(scipy.stats.kendalltau(tokens_no_inf, reqs_no_inf).statistic):
                    correlations[bench][model][key] = None
                else:
                    correlations[bench][model][key] = scipy.stats.kendalltau(tokens_no_inf, reqs_no_inf).statistic
            

    with open(f'./tables.json', 'w') as file:
        json.dump(dict(table_results), file)

    with open(f'./correlations.json', 'w') as file:
        json.dump(correlations, file)
