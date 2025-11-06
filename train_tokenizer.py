from lp_tokenizer.lp_tokenizer import Tokenizer

from datasets import load_dataset
import pickle
import os


def train_lp_tokenizer(dataset,vocab_size,save_dir):
    dataset_size=len(dataset)
    corpus = [dataset[i]["text"] for i in range(dataset_size)]


    tokenizer=Tokenizer(corpus=corpus,
                    vocab_size=vocab_size,
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
    vocab_size = [1024,2048,4096,8192,16384,32768,65536,131072]
    dataset_size = 60000
    dataset=load_dataset(dataset_url)['train'].select(range(dataset_size))
    save_dir="rounding_vocabs/"
    for vs in vocab_size:
        train_lp_tokenizer(dataset,vs,save_dir)