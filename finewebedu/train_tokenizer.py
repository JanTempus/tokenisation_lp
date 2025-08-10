
from tokenisation_lp.lp_tokenizer import lp_tokenizer

from datasets import  load_from_disk,load_dataset
import pickle
import os

"pietrolesci/finewebedu-20B"

datasetname="TinyStories"
dataset_url="roneneldan/TinyStories"
dataset_path="tinystories_data"

tokenizer=lp_tokenizer.Tokenizer(dataset_url, dataset_path, 3,300)

tokenizer.make_vocab()



dataset=load_from_disk(dataset_path)

corpus=[]

for i in range(1,2):
    corpus.append(dataset['train'][i]['text'])

vocab=tokenizer.get_vocab()

tokenized_data=tokenizer.encode(corpus,vocab)

print(tokenized_data)


