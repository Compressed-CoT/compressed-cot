import numpy as np


def token_complexity(all_grades, all_token_counts):
    """
    Computes the token complexity for a set of n questions on the results 
    of a set of K prompts (for a fixed LLM on a fixed dataset).
    
    The token complexity of a question is defined as the minimum token count required
    for an LLM prompt to correctly answer the question. This is estimated by training
    a simple threshold classifier that predicts whether a prompt successfully solves
    the problem (based on whether it )
    
    Parameters:
    -----------
    all_grades : list of lists [K x n]
        A list where each element is a list containing the grade (1 or 0) of a prompt on 
        all the n questions
    
    all_token_counts : list of lists (or NumPy arrays)
        A list where each element is a list containing the token counts (int) of a prompt on 
        all the n questions
    
    Returns:
    --------
    token_reqs : list
        The estimated token complexity for each question. If all prompts are correct, the
        minimum token count is returned. If all prompts are incorrect, the value is set to infinity.
    
    accuracies : list
        The highest achieved classification accuracy for each question using a token
        threshold classifier.
    
    trivials : list
        A list of flags indicating trivial cases (1 if all prompts are correct or incorrect,
        0 otherwise).
    
    Method:
    -------
    1. If all prompts for a question are correct, set token complexity to the minimum token count.
    2. If all prompts are incorrect, set token complexity to infinity.
    3. Otherwise:
       - Sort prompts in ascending order of token count.
       - For each token threshold, classify prompts as correct (if tokens >= threshold) or incorrect.
       - Compute classification accuracy for each threshold.
       - Select the token count that maximizes accuracy as the token complexity.
    """

    
    all_grades = np.stack(all_grades).T
    all_token_counts = np.stack(all_token_counts).T
    
    num_qs = len(all_grades)
    token_reqs = []
    accuracies = []
    trivials = []
    correlations = []
    
    
    for i in range(num_qs):
        '''
            If all prompts get it correct, token complexity is set to minimum
            If all prompts get incorrect, token complexity is infinite
            
            For situations in the middle, we iterate over all prompts in ascending order of token count.
            For each prompt k (and token count t_k), we consider a threshold classifier set at t_k that predicts
            that all prompts with tokens >= t_k get the problem correct (grades = 1) and all prompts with tokens
            less than t_k get it wrong. We then compute classification accuracy with the real results.
            
            Token complexity is the t_k that achieves the highest classification accuracy.
        '''
        if all(all_grades[i] == 1.):
            token_req = min(all_token_counts[i])
            token_reqs.append(token_req)
            accuracies.append(1.)
            trivials.append(1.)
        elif all(all_grades[i] == 0.):
            token_req = float('inf')
            token_reqs.append(token_req)
            accuracies.append(1.)
            trivials.append(1.)
        else:
            grades = list(all_grades[i].astype('int'))
            token_count = list(all_token_counts[i])
        
            grades_tokens = [a for a in zip(grades, token_count)]
            grades_tokens.sort(key = lambda x: x[1])
        
            grades_sorted = np.array([x for x,_ in grades_tokens])
            tokens_sorted = [y for _,y in grades_tokens]
    
            max_correct = 0
            max_j = 0
            for j in range(len(grades_sorted) + 1):
                num_1s = sum(grades_sorted[j:])
                num_0s = sum(1-grades_sorted[:j])
                correct = num_1s + num_0s
        
                if correct > max_correct:
                    max_j = j
                    max_correct = correct

            if max_j == len(grades_sorted):
                thresh = float('inf')
            else:
                thresh = tokens_sorted[max_j]
        
            token_reqs.append(thresh)
            accuracies.append(max_correct / len(grades_sorted))
            trivials.append(0.)

    return token_reqs, accuracies, trivials
        
