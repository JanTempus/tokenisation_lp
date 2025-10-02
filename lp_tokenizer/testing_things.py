from lp_tokenizer import Tokenizer
from datasets import  load_from_disk,load_dataset, Dataset

datasetname="finewebedu"
dataset_url="pietrolesci/finewebedu-20B"
dataset_path="tokenized_dataset/train"



dataset=load_from_disk(dataset_path)

print(dataset[0])

