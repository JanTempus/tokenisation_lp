import lp_tokenizer
from datasets import  load_from_disk
import pickle

dataset_path = "tinystories_data"

tokenizer=lp_tokenizer.Tokenizer()

tokenizer.load_and_prepare_dataset()

tokenizer.get_unique_chars()

tokenizer.make_vocab()

TinyStories = load_from_disk(dataset_path)

corpus=TinyStories['train'][1]['text']
tokenized_data= tokenizer.tokenize_data_set([corpus] )
print(tokenized_data)

datasetname="finewebedu"
dataset_url="pietrolesci/finewebedu-20B"
dataset_path="finewebedu_data"
