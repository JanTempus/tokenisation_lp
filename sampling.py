import random
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets
from bpe_tokenizer.bpe_tokenizer import train_bpe_tokenizer
from lp_tokenizer.train_tokenizer import train_lp_tokenizer
dataset_url = "pietrolesci/finewebedu-20B"
NUM_PROC = 16
BATCH_SIZE = 1000
VOCAB_SIZE=32768

def make_pretokenize_and_sample_fn(p: float, t: int, seed: int = 42):
    """
    Returns a HuggingFace `.map`-compatible function that:
      - Pretokenizes text using the tokenizer's pre_tokenizer
      - For each batch, creates t independently sampled variants
      - Each token is kept with probability p independently
      - Returns t separate sampled text lists: sampled_text_0, ..., sampled_text_{t-1}

    Parameters
    ----------
    p : float
        Probability of keeping each pretokenized token independently
    t : int
        Number of independent samples per text
    seed : int
        Random seed for reproducibility
    """

    pretrained_tok = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m-deduped",
        revision="step3000",
        cache_dir="./pythia-70m-deduped/step3000",
    )

    def _fn(batch):
        sampled_texts_per_t = [[] for _ in range(t)]

        for text in batch["text"]:
            pre_toks = [tok for tok, _ in pretrained_tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)]

            # Generate t independent samples
            for i in range(t):
                rnd = random.Random(seed + i + hash(text) % (10**6))  # make per-sample seed reproducible
                sampled_toks = [tok for tok in pre_toks if rnd.random() < p]
                sampled_texts_per_t[i].append(" ".join(sampled_toks))

        # Return dictionary with sampled_text_0, sampled_text_1, ..., sampled_text_{t-1}
        return {f"sampled_text_{i}": sampled_texts_per_t[i] for i in range(t)}

    return _fn


if __name__ == "__main__":
    # Load dataset
    ds = load_dataset(dataset_url, split="train").select_columns(["id", "text"])

    # Build map function
    NUM_SAMPLES=5  # number of independent samples per text
    map_fn = make_pretokenize_and_sample_fn(
        p=0.0001,
        t=NUM_SAMPLES,
        seed=42
    )

    # Apply map (parallel safe)
    sampled_ds = ds.map(
        map_fn,
        batched=True,
        batch_size=BATCH_SIZE,
        desc="Sampling multiple datasets",
        num_proc=NUM_PROC,
    )

    # Split out into t separate datasets and remove empty samples
    sampled_datasets = []
    for i in range(NUM_SAMPLES):
        ds_i = (
            sampled_ds
            .select_columns(["id", f"sampled_text_{i}"])
            .rename_columns({f"sampled_text_{i}": "text"})
            .filter(lambda x: len(x["text"].strip()) > 0)
        )
        sampled_datasets.append(ds_i)

    for i in range(NUM_SAMPLES):
        save_dir="sampled_lp_tokens/"
        train_lp_tokenizer(sampled_datasets[i],VOCAB_SIZE,i,save_dir)
        #save_dir=f"sampled_bpe_tokenizer"
        #train_bpe_tokenizer(32768,sampled_datasets[i],save_dir,i)




