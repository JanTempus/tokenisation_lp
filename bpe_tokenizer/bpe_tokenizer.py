from datasets import load_dataset, load_from_disk
from tokenizers import Regex, Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import os
from tokenizers.pre_tokenizers import ByteLevel, Sequence, Split
from transformers import PreTrainedTokenizerFast, AutoTokenizer


PRETOKENIZER_MODE = os.environ.get("PRETOKENIZER_MODE", "nanochat").strip().lower()
_APERTUS_SPLIT_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n/]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
_NANOCHAT_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

_SPLIT_PATTERNS = {
    "apertus": _APERTUS_SPLIT_PATTERN,
    "nanochat": _NANOCHAT_SPLIT_PATTERN,
}


def build_pretokenizer(mode):
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m-deduped",
        revision="step3000",
        cache_dir="./pythia-70m-deduped/step3000",
    )

    if mode == "pythia":
        return tokenizer

    if mode == "split_bytelevel":
        tokenizer.backend_tokenizer.pre_tokenizer = Sequence(
            [ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)]
        )
        return tokenizer

    if mode in _SPLIT_PATTERNS:
        tokenizer.backend_tokenizer.pre_tokenizer = Sequence(
            [
                Split(
                    pattern=Regex(_SPLIT_PATTERNS[mode]),
                    behavior="isolated",
                    invert=False,
                ),
                ByteLevel(
                    add_prefix_space=False,
                    trim_offsets=True,
                    use_regex=False,
                ),
            ]
        )
        return tokenizer

    raise ValueError(
        f"Unsupported PRETOKENIZER_MODE='{mode}'. "
        f"Expected one of: pythia, split_bytelevel, apertus, nanochat"
    )


PRETOKENIZER = build_pretokenizer(PRETOKENIZER_MODE)


# Full ByteLevel alphabet: 256 byte-level chars. Passed as initial_alphabet
# to BpeTrainer so every byte is guaranteed present in the final vocab
# regardless of what the training corpus happened to contain.
BYTE_LEVEL_ALPHABET = list(ByteLevel.alphabet())


SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>",       # user messages
    "<|user_end|>",
    "<|assistant_start|>",  # assistant messages
    "<|assistant_end|>",
    "<|python_start|>",     # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>",     # python REPL outputs back to assistant
    "<|output_end|>",
    # unk fallback
    "<|unk|>",
]

BOS_TOKEN = "<|bos|>"
UNK_TOKEN = "<|unk|>"


def train_bpe_tokenizer(vocab_size:int,dataset,save_dir: str):

    # Create training corpus
    dataset_size=len(dataset)
    corpus = [dataset[i]["text"] for i in range(dataset_size)]


    # Build tokenizer from scratch with BPE model
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    # Keep pretokenization consistent with the LP training / rounding pipeline
    tokenizer.pre_tokenizer = PRETOKENIZER.backend_tokenizer.pre_tokenizer

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=BYTE_LEVEL_ALPHABET,
    )

    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # Wrap for HF compatibility
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS_TOKEN,
        unk_token=UNK_TOKEN,
        additional_special_tokens=[t for t in SPECIAL_TOKENS if t not in (BOS_TOKEN, UNK_TOKEN)],
    )
    
    # Save in HF format
    save_path = os.path.join(save_dir, f"bpe_{vocab_size}")
    os.makedirs(save_path, exist_ok=True)
    hf_tokenizer.save_pretrained(save_path)
    print(f"Saved Hugging Face–compatible tokenizer at {save_path}")

    return hf_tokenizer



if __name__ == '__main__':
    
    dataset_url="pietrolesci/finewebedu-20B"
    
    save_dir = "bpe_tokenizers_new"
    vocab_sizes = [1024,2048,4096,8192,16384,32768,65536,131072]
    dataset_size = 60000
    dataset=load_dataset(dataset_url)['train'].select(range(dataset_size))
    for vs in vocab_sizes:
        train_bpe_tokenizer(vs, dataset, save_dir)
