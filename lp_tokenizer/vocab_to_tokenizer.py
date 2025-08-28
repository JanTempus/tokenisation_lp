from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizer, AutoTokenizer
import json
from pathlib import Path
import os
from transformers import PreTrainedTokenizerFast


from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.pre_tokenizers import ByteLevel  # or replace with your pretokenizer
import json


def bytelevel_vocab_to_tokenizer(vocab_path: str,vocab_size:int, raw_dataset_path,save_dir: str):
    """
    Load a vocab.json (byte-level tokens -> IDs) into a Hugging Face compatible tokenizer.

    Args:
        vocab_path (str): Path to the vocab.json file.

    Returns:
        PreTrainedTokenizerFast: Hugging Face compatible tokenizer.
    """

    pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            )

    vocab_path = Path(vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found at {vocab_path}")

    # --- Load vocab dict ---
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)   # {token: id}

    # --- Convert to Unigram format: [(token, score), ...] ---
    unigram_vocab = [(token, -1.0) for token in vocab_dict.keys()]

    # --- Create Unigram tokenizer ---
    tokenizer = Tokenizer(Unigram(unigram_vocab))

    # --- Attach pre-tokenizer (replace with your own if different) ---
    tokenizer.pre_tokenizer = pretokenizer.backend_tokenizer.pre_tokenizer


    save_path = os.path.join(save_dir, f"lp_{vocab_size}_{raw_dataset_path}")
    os.makedirs(save_path, exist_ok=True)
   
    fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    fast_tokenizer.save_pretrained(save_path)
    print(f"Saved Hugging Face–compatible tokenizer at {save_path}")


if __name__ == '__main__':
    
    datasetname="finewebedu"
    dataset_path="finewebedu_data"
    vocab_sizes=[2048,4096,8192,16384,32768,65536,131072]
    for vocab_size in vocab_sizes:
        vocab=f"new_vocab/vocab_finewebedu_data_0_{vocab_size}.json"

        save_dir="tokenizers_lp"
        bytelevel_vocab_to_tokenizer(vocab,vocab_size,dataset_path, save_dir)
    