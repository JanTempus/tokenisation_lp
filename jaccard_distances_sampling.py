import json
import os
import pickle

import numpy as np

from sampling_jaccard import (
    jaccard_score,
    pairwise_jaccard_by_length,
    plot_length_conditioned_jaccard,
)
from lp_tokenizer.lp_functions import (
    biased_rounding,
    deterministic_rounding,
    probabilistic_rounding,
)


def jaccard_distance(a, b):
    return jaccard_score(a, b)


def jaccard_distance_different_rounding(vocab_size,raw_tokens):
    with open(raw_tokens, "rb") as f:
        tokens = pickle.load(f)
  
    num_special_chars=len(tokens["special_tokens"])

    det_tokens=deterministic_rounding(tokens["possible_tokens"],tokens["unique_chars"],vocab_size-num_special_chars)
    bias_tokens=biased_rounding(tokens["possible_tokens"],tokens["unique_chars"],vocab_size-num_special_chars)
    prob_tokens=probabilistic_rounding(tokens["possible_tokens"],tokens["unique_chars"],vocab_size-num_special_chars)    
    tokens_ones = [token.token for token in tokens["possible_tokens"] if token.lp_value >= 0.99]
    
    det_tokens = list(set(det_tokens))
    bias_tokens = list(set(bias_tokens))
    prob_tokens = list(set(prob_tokens))
    tokens_ones = list(set(tokens_ones + tokens["unique_chars"]))
        
    return {
        "all_ones": tokens_ones,
        "det": det_tokens,
        "bias": bias_tokens,
        "prob": prob_tokens,
    }



if __name__ == "__main__":
    
    VOCAB_SIZE=32768

    token_sets=[]
    for i in range(5):
        raw_tokens=f"sampled_lp_tokens/lp_tokens_{VOCAB_SIZE}_{i}.pkl"
        token_sets.append(jaccard_distance_different_rounding(VOCAB_SIZE,raw_tokens))
    

    n = 5
    keys=["all_ones","det","bias","prob"]
    length_conditioned_results = {
        f"lp_{key}": pairwise_jaccard_by_length(
            [sample_tokens[key] for sample_tokens in token_sets]
        )
        for key in keys
    }

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

    output_dir = os.path.dirname(
        os.path.abspath(f"sampled_lp_tokens/lp_tokens_{VOCAB_SIZE}_0.pkl")
    )
    json_path = os.path.join(
        output_dir, f"jaccard_by_token_length_{VOCAB_SIZE}.json"
    )
    with open(json_path, "w") as f:
        json.dump(
            {
                "vocab_size": VOCAB_SIZE,
                "by_token_length": length_conditioned_results,
            },
            f,
            indent=2,
        )

    plot_path = os.path.join(
        output_dir, f"jaccard_by_token_length_{VOCAB_SIZE}.png"
    )
    plot_length_conditioned_jaccard(
        length_conditioned_results,
        plot_path,
        title=f"LP Jaccard by stored token length (vocab size {VOCAB_SIZE})",
    )
    print(f"Saved length-conditioned results to {json_path}")
    print(f"Saved length-conditioned plot to {plot_path}")
