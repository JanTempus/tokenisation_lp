from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

def total_chars_in_dataset(dataset_path: str, dataset_size: int) -> int:
    """
    Loads a Hugging Face dataset, pre-tokenizes it using EleutherAI/pythia-70m-deduped
    tokenizer, and returns the total number of characters across the texts.
    """
    # Load dataset
    dataset = load_from_disk(dataset_path)['train'].select(range(dataset_size))
    
    # Load pre-tokenizer
    pretokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m-deduped",
        revision="step3000",
        cache_dir="./pythia-70m-deduped/step3000",
    )
    
    total_chars = 0
    
    for row in tqdm(dataset, desc="Pre-tokenizing dataset"):
        # pretokenizer can do pre_tokenize_str or encode depending on usage
        pre_tokens = pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(row["text"])
        # Sum lengths of all pre-tokenized pieces
        total_chars += sum(len(piece) for piece, _ in pre_tokens)
    
    return total_chars

# Example usage
dataset_path = "finewebedu_data"
dataset_size = 65536
total_chars = total_chars_in_dataset(dataset_path, dataset_size)
print(f"Total characters in dataset: {total_chars}")
