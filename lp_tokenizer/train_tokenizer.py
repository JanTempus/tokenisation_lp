
from lp_tokenizer import Tokenizer

from datasets import  load_from_disk,load_dataset
import pickle
import os



# datasetname="tinystories"
# dataset_url="roneneldan/TinyStories"
# dataset_path="tinystories_data"


datasetname="finewebedu"
dataset_url="pietrolesci/finewebedu-20B"
dataset_path="finewebedu_data"

tokenizer=Tokenizer(dataset_url, dataset_path, 1000000,32000)

tokenizer.make_vocab()



# dataset=load_from_disk(dataset_path)

# corpus=[]

# for i in range(4,8):
#     corpus.append(dataset['train'][i]['text'])

# vocab=tokenizer.get_vocab()

# tokenized_data=tokenizer.encode(corpus,vocab)
# print(tokenized_data)


