import pickle
import numpy as np
from lp_tokenizer.lp_functions import deterministic_rounding, biased_rounding,probabilistic_rounding


def jaccard_distance(a, b):
    inter = len(set(a) & set(b))
    union = len(set(a) | set(b))
    return inter / union if union > 0 else 0.0


def jaccard_distance_different_rounding(vocab_size,raw_tokens):
    with open(raw_tokens, "rb") as f:
        tokens = pickle.load(f)
  
    num_special_chars=len(tokens["special_tokens"])

    det_tokens=deterministic_rounding(tokens["possible_tokens"],tokens["unique_chars"],vocab_size-num_special_chars)
    bias_tokens=biased_rounding(tokens["possible_tokens"],tokens["unique_chars"],vocab_size-num_special_chars)
    prob_tokens=probabilistic_rounding(tokens["possible_tokens"],tokens["unique_chars"],vocab_size-num_special_chars)    
    tokens_ones = [token.token for token in tokens["possible_tokens"] if token.lp_value >= 0.99]
    
    det_tokens  = list(set(det_tokens+tokens["special_tokens"]))
    bias_tokens = list(set(bias_tokens+tokens["special_tokens"]))
    prob_tokens = list(set(prob_tokens+tokens["special_tokens"]))
    tokens_ones = list(set(tokens_ones+tokens["unique_chars"]+tokens["special_tokens"]))
        
    return {"all_ones":tokens_ones,"det":det_tokens,"bias":bias_tokens,"prob":prob_tokens}



if __name__ == "__main__":
    
    VOCAB_SIZE=32768

    token_sets=[]
    for i in range(5):
        raw_tokens=f"sampled_lp_tokens/lp_tokens_{VOCAB_SIZE}_{i}.pkl"
        token_sets.append(jaccard_distance_different_rounding(VOCAB_SIZE,raw_tokens))
    

    n = 5
    keys=["all_ones","det","bias","prob"]

    dist_matrix_ones = np.zeros((n-1, n-1))
    dist_matrix_det = np.zeros((n-1, n-1))
    dist_matrix_bias = np.zeros((n-1, n-1))
    dist_matrix_prob = np.zeros((n-1, n-1))

    dist_matrices=[dist_matrix_ones,dist_matrix_det,dist_matrix_bias,dist_matrix_prob]

    for i in range(5):
        for j in range(i+1,5):
            for k in range(len(keys)):
                set_a=token_sets[i][keys[k]]
                set_b=token_sets[j][keys[k]]
                
                d=jaccard_distance(set_a,set_b)
                dist_matrices[k][i][j-1]=d
                
    
    for dist_matrix in dist_matrices:
        print(dist_matrix)