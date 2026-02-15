from lp_tokenizer.lp_tokenizer import Tokenizer
from transformers import Autotokenizer
from datasets import load_dataset
import pickle
import os


num_proc=16
batch_size=10000
pretokenizer=AutoTokenizer.from_pretrained("EleutherAI/pythia-70m-deduped",
                                            revision="step3000"
                                            )


def get_unique_chars_batch(batch):
    unique_chars = set()

    for text in batch["text"]:
        words_with_offsets = pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        tokens = [word for word, _ in words_with_offsets]
        for token in tokens:
            unique_chars.update(token)

    return {"unique_chars": [list(unique_chars)]}


def train_lp_tokenizer(dataset,unique_chars,vocab_size,save_dir):
    dataset_size=len(dataset)
    corpus_all = [dataset[i]["text"] for i in range(dataset_size)]

    tokenizer=Tokenizer(corpus=corpus_all,
                    vocab_size=vocab_size,
                    unique_chars=unique_chars,
                    unk_token="[UNK]",
                    eos_token="[EOS]",
                    pad_token="[PAD]",
                    cls_token="[CLS]",
                    sep_token="[SEP]",
                    mask_token="[MASK]")
    tokens=tokenizer.make_vocab()
    file_name=os.path.join(save_dir,f"lp_tokens_{vocab_size}.pkl")
    os.makedirs(save_dir, exist_ok=True)
    with open(file_name, "wb") as f:
        pickle.dump(tokens, f)



if __name__== "__main__":
    dataset_url="pietrolesci/finewebedu-20B"
    dataset=load_dataset(dataset_url)['train']
        
    char_chunks = dataset.map(
        get_unique_chars_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Finding unique characters",
    )

    unique_chars = sorted(set().union(*map(set, char_chunks["unique_chars"])))

    
    vocab_size = [1024,2048,4096,8192,16384,32768,65536,131072]
    dataset_size = 60000
    dataset=load_dataset(dataset_url)['train'].select(range(dataset_size))
    save_dir="rounding_vocabs/"
    for vs in vocab_size:
       train_lp_tokenizer(dataset, unique_chars, vs, save_dir)
