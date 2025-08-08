from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import os
from tokenizers.pre_tokenizers import ByteLevel

def train_bpe_tokenizer(vocab_size:int,data_set_size_div: int = 2, raw_dataset_path: str = "tinystories_data"):
    # Load dataset
    if os.path.exists(raw_dataset_path):
        print("Loading dataset from disk...")
        TinyStories = load_from_disk(raw_dataset_path)
    else:
        print("Downloading dataset...")
        TinyStories = load_dataset("roneneldan/TinyStories")
        TinyStories.save_to_disk(raw_dataset_path)

    # Create training corpus
    corpus = []
    data_set_size = int(len(TinyStories['train'])/data_set_size_div)

    data_set_size=1
    for i in range(data_set_size):
        corpus.append(TinyStories['train'][i]['text'])

    print("Created the corpus")

    # Build tokenizer from scratch with BPE model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Match EleutherAI/pythia-70m pretokenizer
    tokenizer.pre_tokenizer = ByteLevel()

    # Special tokens (GPTNeoX-style)
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    tokenizer.train_from_iterator(corpus, trainer=trainer)

    print("Tokenizer training complete")
    return tokenizer


