import numpy as np
import matplotlib.pyplot as plt
import os
from token_complexity import token_complexity
from datasets import load_dataset

if __name__ == "__main__":
    
    code_labels = {'NoCoT': 'no_cot',
                    'DefaultCoT': 'default_cot',
                    'BeConcise': 'be_concise',
                    'BulletPoints': 'bullet_points',
                    'OnlyNumbers': 'only_numbers',
                    'NoSpaces': 'no_spaces',
                    'NoProperGrammar': 'no_proper_grammar',
                    'AbbreviateWords': 'abbreviate_words',
                    'WordLimit(1)': 'word_limit_1',
                    'WordLimit(5)': 'word_limit_5',
                    'WordLimit(10)': 'word_limit_10',
                    'WordLimit(15)': 'word_limit_15',
                    'WordLimit(25)': 'word_limit_25',
                    'WordLimit(50)': 'word_limit_50',
                    'WordLimit(100)': 'word_limit_100',
                    'CharLimit(10)': 'char_limit_10',
                    'CharLimit(50)': 'char_limit_50',
                    'CharLimit(100)': 'char_limit_100',
                    'CharLimit(500)': 'char_limit_500',
                    'TokenLimit(10)': 'token_limit_10',
                    'TokenLimit(50)': 'token_limit_50',
                    'TokenLimit(100)': 'token_limit_100',
                    'TokenLimit(500)': 'token_limit_500',
                    'StepLimit(1)': 'step_limit_1',
                    'StepLimit(2)': 'step_limit_2',
                    'StepLimit(5)': 'step_limit_5',
                    'ChineseCoT': 'chinese_cot',
                    'ChineseCoT(10)': 'chinese_char_limit_10',
                    'ChineseCoT(50)': 'chinese_char_limit_50',
                    'ChineseCoT(100)': 'chinese_char_limit_100',
                    'ChineseCoT(500)': 'chinese_char_limit_500'}

    pretty_bench = {
        'mmlu': 'MMLU-Pro Math',
        'gsm8k': 'GSM8K',
        'math500': 'MATH-500',
    }

    pretty_models = {
                'gpt-4o-2024-11-20': 'GPT-4o',
                'gpt-4o-mini-2024-07-18': 'GPT-4o-mini',
                'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet',
                'claude-3-5-haiku-20241022': 'Claude 3.5 Sonnet',
                'meta-llama-Llama-3.3-70B-Instruct-Turbo': 'Llama 3.3 70b',
            }

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

    markers = {
        'chinese_char_limit': 'p',   # separate from "char_limit"
        'chinese_cot':        'o',   # separate from "default_cot" or "cot"
        'word_limit':         '.',
        'char_limit':         'h',
        'token_limit':        '^',
        'step_limit':         '>',
        'letter_limit':       '<',
        'be_concise':         's',
        'default_cot':        '*',
        'no_cot':             'd',
        'no_complete_sentences': 'X',
        'only_numbers':       'D',
        'no_proper_grammar':  'd',
        'abbreviate_words':   '+',
        'bullet_points':      'x',
        'no_spaces':          'v'
    }

    fixed_colors = {
        'chinese_char_limit': 'deepskyblue',
        'chinese_cot': 'darkblue',
        
        'word_limit': 'darkorange',
        'char_limit': 'yellowgreen',
        'token_limit': 'blue',
        'step_limit': 'lightsteelblue',
        'letter_limit': 'gray',
        
        'no_cot': 'limegreen',
        'only_numbers': 'gray',
        'be_concise': 'pink',
        'default_cot': 'black',
        'only_numbers': 'green',
        'no_spaces': 'magenta',
        'no_proper_grammar': 'blueviolet'
    }

    discrete_bins = {
        'word_limit':   [1, 5, 10, 15, 25, 50, 100],
        'char_limit':   [1, 10, 20, 50, 100, 500],
        'token_limit':  [1, 10, 20, 50, 100, 500],
        'step_limit':   [1, 2, 5],
        'letter_limit': [2, 3, 5],
        'chinese_char_limit': [10, 50, 100, 500]  # if you want numeric scaling for Chinese versions
    }

    marker_size_ranges = {
        'word_limit':   {'min_val': 1, 'max_val': 100, 'min_size': 200,  'max_size': 500},
        'char_limit':   {'min_val': 1, 'max_val': 500, 'min_size': 50,   'max_size': 300},
        'token_limit':  {'min_val': 1, 'max_val': 500, 'min_size': 100,  'max_size': 300},
        'step_limit':   {'min_val': 1, 'max_val': 5,   'min_size': 100,  'max_size': 300},
        'letter_limit': {'min_val': 2, 'max_val': 5,   'min_size': 100,  'max_size': 300},
        'chinese_char_limit': {'min_val': 10, 'max_val': 500, 'min_size': 100, 'max_size': 300}
    }

    plot_ylim_bounds = {
        'mmlu': (0.3, 0.9),
        'gsm8k': (0.2, 1.0),
        'math500': (0.2, 0.85),
    }

    # Fallback logic for unknown categories
    unknown_marker_cycle = ['^', 'v', 'D', '*', 'H', '8']
    unknown_category_markers = {}
    color_cycle = plt.cm.tab10.colors
    unknown_category_colors = {}

    def parse_numeric_value(cot):
        """Extract numeric part from prompt string, e.g. 'word_limit10' -> 10."""
        try:
            return int(''.join(filter(str.isdigit, cot)))
        except ValueError:
            return None

    def get_marker_size(category, numeric_value, exponent=1):
        """Map numeric_value to a size, based on discrete bins."""
        if category not in marker_size_ranges:
            return 300  # fallback size
        info = marker_size_ranges[category]
        min_val, max_val = info['min_val'], info['max_val']
        min_size, max_size = info['min_size'], info['max_size']
        if numeric_value is None:
            numeric_value = min_val
        bins = discrete_bins.get(category, [])
        if not bins:
            return 300
        clamped_val = max(min_val, min(numeric_value, max_val))
        differences = [abs(b - clamped_val) for b in bins]
        index = differences.index(min(differences))
        t = index / (len(bins) - 1) if (len(bins) - 1) > 0 else 0
        t = t ** exponent  # non-linear scaling
        return min_size + t * (max_size - min_size)

    def get_fixed_color(category):
        if category in fixed_colors:
            return fixed_colors[category]
        return 'gray'

    def get_marker_shape(category):
        if category in markers:
            return markers[category]
        if category not in unknown_category_markers:
            idx = len(unknown_category_markers) % len(unknown_marker_cycle)
            unknown_category_markers[category] = unknown_marker_cycle[idx]
        return unknown_category_markers[category]
    
    def compute_pareto_frontier(xs, ys, minimize_x=True, maximize_y=True, extend_to_max_x=True):
        """
        Computes a 2D Pareto frontier for points (xs, ys).
        By default:
        - x is a 'cost' we want to minimize
        - y is a 'benefit' we want to maximize
        
        If extend_to_max_x=True, after computing the frontier,
        we add a final horizontal line extending to the global max x
        if that max x is larger than the last frontier point.
        """
        points = sorted(zip(xs, ys), key=lambda t: (t[0], -t[1]))
        frontier = []
        best_y = -np.inf if maximize_y else np.inf

        # Single pass to pick frontier points
        for x_val, y_val in points:
            if maximize_y:
                if y_val > best_y:
                    frontier.append((x_val, y_val))
                    best_y = y_val
            else:
                if y_val < best_y:
                    frontier.append((x_val, y_val))
                    best_y = y_val

        # Optionally extend horizontally to the max X
        if extend_to_max_x and frontier:
            global_max_x = max(xs)
            last_x_frontier = frontier[-1][0]
            last_y_frontier = frontier[-1][1]
            if global_max_x > last_x_frontier:
                frontier.append((global_max_x, last_y_frontier))

        return frontier

    # Loop over all benchmarks and models
    for bench in benches:
        dataset = load_dataset("cot-compression/TokenComplexity", split = bench)
        for provider, model in models:

            data_dict = {a['prompt_id']: a for a in dataset[model]}
            
            all_grades = []
            all_token_counts = []
            max_avg_token_count = 0

            # for cot in view_prompts:
            #     if os.path.isfile(f'./results/{provider}-{cot}-{model}-{bench}.json'):
            #         with open(f'./results/{provider}-{cot}-{model}-{bench}.json', 'r') as file:
            #             cot_results = json.load(file)

            #         max_avg_token_count = max(max_avg_token_count, np.mean(cot_results['token_count']))
            #         all_grades.append(cot_results['grading'])
            #         all_token_counts.append(cot_results['token_count'])

            for key, item in data_dict.items():
                max_avg_token_count = max(max_avg_token_count, np.mean(item['token_count']))
                all_grades.append(item['grading'])
                all_token_counts.append(item['token_count'])

            num_qs = len(all_grades[0])
            token_reqs, accuracies, trivials = token_complexity(all_grades, all_token_counts)

            sorted_token_counts = sorted(token_reqs)
            token_resid_costs = np.array([min(token_reqs)*(len(token_reqs) - i - 1) for i in range(len(token_reqs))])

            xlim = int(max_avg_token_count)

            # Optimal accuracy for an average budget of j tokens, if we know token complexities
            upper_bound = [np.sum(1*(np.cumsum(sorted_token_counts) + token_resid_costs <= j*num_qs))/ num_qs for j in range(0,xlim)]

            # Accuracy if we use same tokens for each question
            constant_bound = [np.sum(1*(np.array(sorted_token_counts) <= j))/ num_qs for j in range(0,xlim)]

            # --------------------------------------------------------------------------
            # THRESHOLD PLOTS
            # --------------------------------------------------------------------------
            j = 24
            plt.figure(figsize=(8, 5))
            # Loop over each prompt index j
            for i, cot in enumerate(data_dict.keys()):
                
                token_count = all_token_counts[i][j]  # x
                grade = all_grades[i][j]              # y
                cot = code_labels[cot]

                # Determine category from prefix
                matched_category = None
                for cat in markers.keys():
                    if cot == cat or cot.startswith(cat + "_"):
                        matched_category = cat
                        break
                if matched_category is None:
                    matched_category = cot  # fallback

                numeric_val = parse_numeric_value(cot)
                shape = get_marker_shape(matched_category)
                color = get_fixed_color(matched_category)
                size = get_marker_size(matched_category, numeric_val)

                # Plot one data point
                plt.scatter(
                    token_count, grade,
                    s=size,
                    marker=shape,
                    color=color,
                    alpha=0.8,
                    edgecolors='black'
                )

            # Draw the vertical line if within range
            line_x = token_reqs[j]
            plt.axvline(line_x, color='red', linestyle='--')

            # Build the label string in parts to avoid f-string issues with backslashes.
            label_str = f'$\\leftarrow \\hat{{\\tau}}_{{{j}}} = {line_x}$'

            # Capture the text artist in a variable.
            text_obj = plt.text(
                line_x + 5,
                0.5,  # y-level
                label_str,
                color='red',
                fontsize=14,
                ha='left',
                va='center'
            )

            plt.xlabel('Token Length of Chain-of-Thought', fontsize=11)
            plt.ylabel('Correct', fontsize=11)
            plt.title(f'{model}: Performance on Question {j} across Different Prompts', fontsize=11)

            # Instead of tight_layout (which is causing issues), adjust margins manually.
            plt.subplots_adjust(left=0.15, right=0.95, top=0.9, bottom=0.15)

            save_dir = f'./plots/{model}'
            os.makedirs(save_dir, exist_ok=True)
            save_path = f'{save_dir}/{pretty_bench[bench]}_question_{j}.png'
            plt.savefig(save_path, dpi = 300)
            # plt.show()
            plt.close()

            # --------------------------------------------------------------------------
            # TOKEN COMPLEXITY PREDICTIONS
            # --------------------------------------------------------------------------
            accuracies = []

            for cot, item in data_dict.items():
                cot_results = item
                cot = code_labels[cot]

                # Compute the means and error bars
                x = np.mean(cot_results['token_count'])
                y = np.mean(cot_results['grading'])

                x_err = 2 * np.std(cot_results['token_count']) / np.sqrt(len(cot_results['token_count']))
                y_err = 2 * np.std(cot_results['grading']) / np.sqrt(len(cot_results['grading']))

                # Determine the category by checking if cot starts with one of the marker keys.
                matched_category = None
                for cat in markers.keys():
                    if cot == cat or cot.startswith(cat + "_"):
                        matched_category = cat
                        break
                if matched_category is None:
                    matched_category = cot  # fallback if no category match is found

                # Parse any numeric value from the prompt (e.g., "word_limit10" -> 10)
                numeric_val = parse_numeric_value(cot)

                # Get marker shape, color, and size based on the category and numeric value
                shape = get_marker_shape(matched_category)
                color = get_fixed_color(matched_category)
                size = get_marker_size(matched_category, numeric_val)

                # Compute predicted accuracy (token hypothesis)
                token_req_pred = np.mean(1. * (np.array(cot_results['token_count']) >= np.array(token_reqs)))

                # Plot the data point with error bars using the customized marker settings.
                # Note: We removed the label parameter so these won't appear in the legend.
                plt.scatter(token_req_pred, y, s=size, marker=shape,
                            color=color, alpha=0.8, edgecolors='black', zorder=3)
                plt.errorbar(token_req_pred, y, xerr=0, yerr=y_err, ecolor='gainsboro', fmt='none', zorder=1)

                accuracies.append(token_req_pred)

            # Plot the diagonal reference line (y = x) and give it a label.
            acc_grid = np.arange(np.min(accuracies), np.max(accuracies) + 0.01, 0.01)
            plt.plot(acc_grid, acc_grid, label='$y=x$', color='red', linestyle='--', zorder=2)
            #plt.figure(figsize=(6,4))
            plt.title(f'{pretty_bench[bench]}: {model}', fontsize=15)
            plt.ylabel('Actual accuracy', fontsize=15)
            plt.xlabel('Predicted accuracy from token complexity', fontsize=15)
            # Place the legend in the upper left corner inside the plot.
            plt.legend(loc='upper left',fontsize=14)
            plt.xticks(fontsize = 15)
            plt.yticks(fontsize = 15)
            plt.tight_layout()

            save_path = f'{save_dir}/{bench}_{model}_token_complexity.png'
            plt.savefig(save_path, dpi=300)
            plt.close()

            # --------------------------------------------------------------------------
            # PARETO CURVE
            # --------------------------------------------------------------------------

            fig, ax = plt.subplots(figsize=(9, 6))
            legend_entries = {}
            all_x = []
            all_y = []

            for cot, item in data_dict.items():
                cot_results = item
                cot = code_labels[cot]
                
                # Compute the average token count (x) and average grading (y)
                x = np.mean(cot_results['token_count'])
                y = np.mean(cot_results['grading'])
                
                # 2-sigma error bars
                x_err = 2 * np.std(cot_results['token_count']) / np.sqrt(len(cot_results['token_count']))
                y_err = 2 * np.std(cot_results['grading']) / np.sqrt(len(cot_results['grading']))
                
                all_x.append(x)
                all_y.append(y)
                
                # ---------------------------------------------
                # Use prefix-based detection
                matched_category = None
                for cat in markers.keys():
                    if cot == cat or cot.startswith(cat + "_"):
                        matched_category = cat
                        break
                
                if matched_category is None:
                    # Fallback if no recognized prefix
                    matched_category = cot
                
                numeric_val = parse_numeric_value(cot)
                marker_shape = get_marker_shape(matched_category)
                color = get_fixed_color(matched_category)
                size = get_marker_size(matched_category, numeric_val)
                
                # We'll label the legend by the *exact prompt name*:
                legend_key = cot
                
                # Plot the data point (zorder=3 to ensure markers are above the Pareto line)
                sc = ax.scatter(
                    x, y, s=size,
                    marker=marker_shape, color=color,
                    edgecolor='black', alpha=0.9,
                    zorder=3
                )
                
                # Add error bars with the same color as the scatter (zorder=2 or 3 as well)
                eb = ax.errorbar(
                    x, y,
                    xerr=x_err, yerr=y_err,
                    fmt='none', 
                    ecolor=color,  # match the marker color
                    elinewidth=1.5, capsize=3, alpha=0.3,
                    zorder=3
                )
                for line in eb[2]:
                    line.set_linestyle('-')
                
                if legend_key not in legend_entries:
                    legend_entries[legend_key] = sc

            # ------------- PARETO FRONT -------------
            pareto_front = compute_pareto_frontier(
                all_x, all_y,
                minimize_x=True,  # treat x as cost (lower = better)
                maximize_y=True,  # treat y as benefit (higher = better)
                extend_to_max_x=True
            )
            if pareto_front:
                px, py = zip(*pareto_front)
                # Plot the frontier behind other elements (zorder=1)
                pf_line, = ax.plot(
                    px, py,
                    color='crimson',
                    linestyle='--',
                    linewidth=3,
                    label='Pareto Frontier',
                    zorder=1
                )
                legend_entries['pareto_frontier'] = pf_line

            # ----------- FINISH PLOT -----------
            ax.set_title(f'{pretty_bench[bench]}: {model}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Average Number of Tokens', fontsize=14)
            ax.set_ylabel('Accuracy', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.5)

            # ___________ Upper and Lower bounds __________

            # Plot the bounds and capture their line objects
            ub_line, = ax.plot(
                upper_bound,
                label='Oracle Upperbound',
                color='orchid',
                linestyle='solid',
                linewidth=3,
                alpha=0.3,
                zorder=2
            )
            cb_line, = ax.plot(
                constant_bound,
                label='Non-adaptive Baseline',
                color='royalblue',
                linestyle='solid',
                linewidth=3,
                alpha=0.3,
                zorder=2
            )

            # Create a separate legend for these two lines in the upper left corner.
            legend_bounds = ax.legend(handles=[ub_line, cb_line, pf_line], loc='lower right', fontsize=18)
            ax.add_artist(legend_bounds)

            #################################
            #######LEGEND BOUNDS

            ax.set_ylim(plot_ylim_bounds[bench][0], plot_ylim_bounds[bench][1])

            #################################


            plt.tight_layout()


            # Save the main figure (make sure directory exists or change path)
            save_path = f'{save_dir}/{model}-{bench}-main.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()