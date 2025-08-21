from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizer, AutoTokenizer
import json
from pathlib import Path
import os

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

    # Load vocab
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Create WordLevel model (with unk token)
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))

    # Set ByteLevel pre-tokenizer and decoder
    tokenizer.pre_tokenizer = pretokenizer.backend_tokenizer.pre_tokenizer
    tokenizer.decoder = ByteLevelDecoder()

    # Wrap with Hugging Face interface
    hf_tokenizer = PreTrainedTokenizer(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        eos_token="<|endoftext|>"
    )

    save_path = os.path.join(save_dir, f"lp_{vocab_size}_{raw_dataset_path}")
    os.makedirs(save_path, exist_ok=True)
    hf_tokenizer.save_pretrained(save_path)
    print(f"Saved Hugging Face–compatible tokenizer at {save_path}")


if __name__ == '__main__':
    
    datasetname="finewebedu"
    dataset_url="pietrolesci/finewebedu-20B"
    dataset_path="finewebedu_data"


vocab_4096="/home/jantempus/Desktop/Projects/NLP/tokenisation_lp/lp_tokenizer/vocabs/vocab_finewebedu_data_0_4096.json"
vocab_8192="/home/jantempus/Desktop/Projects/NLP/tokenisation_lp/lp_tokenizer/vocabs/vocab_finewebedu_data_0_8192.json"
vocab_32768="/home/jantempus/Desktop/Projects/NLP/tokenisation_lp/lp_tokenizer/vocabs/vocab_finewebedu_data_0_32768.json"

save_dir="tokenizers"
bytelevel_vocab_to_tokenizer(vocab_4096,4096,dataset_path, save_dir)
bytelevel_vocab_to_tokenizer(vocab_8192,8192,dataset_path, save_dir)
bytelevel_vocab_to_tokenizer(vocab_32768,32768,dataset_path, save_dir)