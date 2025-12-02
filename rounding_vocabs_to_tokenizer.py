import pickle
import numpy as np
from lp_tokenizer.lp_functions import deterministic_rounding, biased_rounding,probabilistic_rounding
import lp_tokenizer.lp_tokenizer as LP_TOK
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import AutoTokenizer
import json
from pathlib import Path
import os
from transformers import PreTrainedTokenizerFast
from tokenizers.models import Unigram
from datasets import Dataset, load_dataset
from tqdm import tqdm


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
    
    
    #vocab_dict # {token: id}

    # --- Convert to Unigram format: [(token, score), ...] ---
    unigram_vocab = [(token, -1.0) for token in vocab]

    # --- Create Unigram tokenizer ---
    tokenizer = Tokenizer(Unigram(unigram_vocab))

    # --- Attach pre-tokenizer (replace with your own if different) ---
    tokenizer.pre_tokenizer = pretokenizer.backend_tokenizer.pre_tokenizer
    special_tokens = {
        "unk_token": "UNKtokenbehere",           # unknown token
        "eos_token": "EOStokenbehere",  # end-of-sequence token
        "pad_token": "PADtokenbehere",
        "cls_token": "CLStokenbehere",
        "sep_token": "SEPtokenbehere",
        "mask_token":"MASKtokenbehere"
    }
    tokenizer.add_special_tokens(list(special_tokens.values()))


    vocab_size=len(vocab)+len(special_tokens)
    save_path = os.path.join(save_dir, f"lp_{vocab_size}_{rnd_scheme}")
    os.makedirs(save_path, exist_ok=True)
   
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                             **special_tokens
                                            )
    print("Number of tokens in fast_tokenizer:", len(fast_tokenizer))
    fast_tokenizer.save_pretrained(save_path)
    print(f"Saved Hugging Face–compatible tokenizer at {save_path}")



def round_vocabs(vocab_size,raw_tokens,unique_chars):
    with open(raw_tokens, "rb") as f:
        tokens = pickle.load(f)
  
    num_special_chars=len(tokens["special_tokens"])+1
    print(f"Num special characters: {num_special_chars}")

    det_tokens=deterministic_rounding(tokens["possible_tokens"],unique_chars,vocab_size-num_special_chars)
    bias_tokens=biased_rounding(tokens["possible_tokens"],unique_chars,vocab_size-num_special_chars)
    prob_tokens=probabilistic_rounding(tokens["possible_tokens"],unique_chars,vocab_size-num_special_chars)    
    tokens_ones = [token.token for token in tokens["possible_tokens"] if token.lp_value >= 0.99]
    
    det_tokens  = list(set(det_tokens))
    bias_tokens = list(set(bias_tokens))
    prob_tokens = list(set(prob_tokens))
    tokens_ones = list(set(tokens_ones+unique_chars))
        
    
    return {"all_ones":tokens_ones,"det":det_tokens,"bias":bias_tokens,"prob":prob_tokens}


pretokenizer = AutoTokenizer.from_pretrained(
    "EleutherAI/pythia-70m-deduped",
    revision="step3000",
    cache_dir="./pythia-70m-deduped/step3000",
)

# ---- Batch processing function ----
def extract_unique_chars(batch):
    """
    batch["text"] is a list[str]
    returns a dictionary of lists, since HF cannot store sets natively
    """
    batch_chars = set()

    for text in batch["text"]:
        words_with_offsets = pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        tokens = [tok for tok, _ in words_with_offsets]
        for tok in tokens:
            batch_chars.update(tok)

    return {"unique_chars": list(batch_chars)}


# ---- Run .map with batching ----
def get_all_unique_chars(dataset, text_column="text", batch_size=10000, num_proc=16):
    """
    dataset: HuggingFace Dataset object
    text_column: name of the text column
    batch_size: size per map batch
    num_proc: number of CPU processes
    """
    # Map returns a dataset with a new column "unique_chars",
    # each entry is a *list* of characters for that batch.
    mapped = dataset.map(
        extract_unique_chars,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Extracting chars with pretokenizer"
    )

    # Merge all sets
    final_chars = set()
    for batch_list in tqdm(mapped["unique_chars"], desc="Merging characters"):
        final_chars.update(batch_list)

    return sorted(final_chars)




if __name__ == '__main__':
    save_dir="rounded_tokenizers_fixed"
    dataset_url="pietrolesci/finewebedu-20B"
    
    dataset=load_dataset(dataset_url)['train']
    unique_chars=get_all_unique_chars(dataset)


    vocab_sizes=[1024,2048,4096,8192,16384,32768,65536,131072]
    raw_vocab_path="/local/home/jtempus/tokenisation_lp/rounding_vocabs"


    for vocab_size in vocab_sizes:
        vocabs_name=f"lp_tokens_{vocab_size}.pkl"
        raw_tokens=os.path.join(raw_vocab_path,vocabs_name)
        vocabs=round_vocabs(vocab_size,raw_tokens,unique_chars)
        bytelevel_vocab_to_tokenizer(vocabs["all_ones"],save_dir,"all_ones")
        bytelevel_vocab_to_tokenizer(vocabs["det"],save_dir,"det")
        bytelevel_vocab_to_tokenizer(vocabs["bias"],save_dir,"bias")
        bytelevel_vocab_to_tokenizer(vocabs["prob"],save_dir,"prob")
    
