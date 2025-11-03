from datasets import load_dataset, load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import os
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast, AutoTokenizer


def train_bpe_tokenizer(vocab_size:int,dataset_size:int, raw_dataset_path: str,dataset_url,save_dir: str):
    # Load dataset
    # if os.path.exists(raw_dataset_path):
    #     print("Loading dataset from disk...")
    #     dataset = load_from_disk(raw_dataset_path)
    # else:
    #     print("Downloading dataset...")
    #     dataset = load_dataset(dataset_url)
    #     dataset.save_to_disk(raw_dataset_path)
    dataset=load_dataset(dataset_url)
    pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                              revision="step3000",
                              cache_dir="./pythia-70m-deduped/step3000",
                                            )

    # Create training corpus

    corpus = [dataset["train"][i]["text"] for i in range(dataset_size)]

    print("Created the corpus")

    # Build tokenizer from scratch with BPE model
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Match EleutherAI/pythia-70m pretokenizer
    tokenizer.pre_tokenizer = pretokenizer.backend_tokenizer.pre_tokenizer

    # Special tokens (GPTNeoX-style)
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]","endoftextbehere"]
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    tokenizer.train_from_iterator(corpus, trainer=trainer)

    # Wrap for HF compatibility
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="endoftextbehere",
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    
    # Save in HF format
    save_path = os.path.join(save_dir, f"bpe_{vocab_size}_finewebedu")
    os.makedirs(save_path, exist_ok=True)
    hf_tokenizer.save_pretrained(save_path)
    print(f"Saved Hugging Face–compatible tokenizer at {save_path}")

    return hf_tokenizer



if __name__ == '__main__':
    
    datasetname="finewebedu"
    dataset_url="pietrolesci/finewebedu-20B"
    dataset_path="/local/home/jtempus/token_lp/tokenisation_lp/lp_tokenizer/finewebedu_data"
    
    save_dir = "bpe_tokenizers"
    vocab_sizes = [1024,2048,4096,8192,16384,32768,65536,131072]
    dataset_size = 65536

    for vs in vocab_sizes:
        train_bpe_tokenizer(vs, dataset_size, dataset_path, dataset_url, save_dir)
