import pickle
import numpy as np
from lp_tokenizer.lp_functions import deterministic_rounding, biased_rounding,probabilistic_rounding


from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import AutoTokenizer
import json
from pathlib import Path
import os
from transformers import PreTrainedTokenizerFast

from tokenizers.models import Unigram


def bytelevel_vocab_to_tokenizer(vocab, save_dir: str,rnd_scheme:str):
    """
    Load a vocab.json (byte-level tokens -> IDs) into a Hugging Face compatible tokenizer.

    Args:
        vocab_path (str): Path to the vocab.json file.

    Returns:
        PreTrainedTokenizerFast: Hugging Face compatible tokenizer.
    """

    pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000"
                                            )

    vocab_size=len(vocab)
    
    #vocab_dict # {token: id}

    # --- Convert to Unigram format: [(token, score), ...] ---
    unigram_vocab = [(token, -1.0) for token in vocab]

    # --- Create Unigram tokenizer ---
    tokenizer = Tokenizer(Unigram(unigram_vocab))

    # --- Attach pre-tokenizer (replace with your own if different) ---
    tokenizer.pre_tokenizer = pretokenizer.backend_tokenizer.pre_tokenizer
    special_tokens = {
        "unk_token": "[UNK]",           # unknown token
        "eos_token": "[EOS]",  # end-of-sequence token
        "pad_token":"[PAD]",
        "cls_token":"[CLS]",
        "sep_token":"[SEP]",
        "mask_token":"[MASK]"
    }
    tokenizer.add_special_tokens(list(special_tokens.values()))
    
    save_path = os.path.join(save_dir, f"lp_{vocab_size}_{rnd_scheme}")
    os.makedirs(save_path, exist_ok=True)
   
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                             **special_tokens
                                            )

    fast_tokenizer.save_pretrained(save_path)
    print(f"Saved Hugging Face–compatible tokenizer at {save_path}")



def round_vocabs(vocab_size,raw_tokens):
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




if __name__ == '__main__':
    save_dir="rounded_tokenizers"
    vocab_sizes=[1024,2048,4096,8192,16384,32768,65536,131072]
    raw_vocab_path="/local/home/jtempus/tokenisation_lp/rounding_vocabs"
    for vocab_size in vocab_sizes:
        vocabs_name=f"lp_tokens_{vocab_size}.pkl"
        raw_tokens=os.path.join(raw_vocab_path,vocabs_name)
        vocabs=round_vocabs(vocab_size,raw_tokens)
        bytelevel_vocab_to_tokenizer(vocabs["all_ones"],save_dir,"all_ones")
        bytelevel_vocab_to_tokenizer(vocabs["det"],save_dir,"det")
        bytelevel_vocab_to_tokenizer(vocabs["bias"],save_dir,"bias")
        bytelevel_vocab_to_tokenizer(vocabs["prob"],save_dir,"prob")
    
