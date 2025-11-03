import os
import json
from datasets import  load_dataset,Dataset
from transformers import AutoTokenizer


# --- Config ---
dataset_url="pietrolesci/finewebedu-20B"
lp_tokenizer_path = "/local/home/jtempus/tokenisation_lp/lp_tokenizer/tokenizers_lp/lp_32768_finewebedu_data"
bpe_tokenizer_path= "/local/home/jtempus/tokenisation_lp/bpe_tokenizer/bpe_tokenizers/bpe_32768_finewebedu"


dataset=load_dataset(dataset_url)["train"][-1]["text"]

#print(dataset)

lp_tokenizer=AutoTokenizer.from_pretrained(lp_tokenizer_path, local_files_only=True)
bpe_tokenizer=AutoTokenizer.from_pretrained(bpe_tokenizer_path,local_files_only=True)

encoded_lp = lp_tokenizer(dataset)
tokens_lp = [lp_tokenizer.convert_ids_to_tokens(i) for i in encoded_lp["input_ids"]]

encoded_bpe=bpe_tokenizer(dataset)
tokens_bpe=[bpe_tokenizer.convert_ids_to_tokens(i) for i in encoded_bpe["input_ids"]]

print("=== LP tokens ===")
print(len(tokens_lp))
print("=== BPE tokens ===")
print(len(tokens_bpe))