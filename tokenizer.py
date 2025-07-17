from transformers import AutoTokenizer
from collections import OrderedDict,defaultdict
from setup_lp import create_vocab,tokenize
import numpy as np
import os
import pickle
import helper_functions as hf
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


class Tokenizer:
    vocab: OrderedDict
    pretokenizer: AutoTokenizer
    max_token_count:int
    dataset_path:str
    unique_chars_path:str
    original_corpus_size:int
    tokenized_corpus_size:int
    data_set_size:int
    

    def __init__(self):
        self.pretokenizer=AutoTokenizer.from_pretrained(
                                                        "EleutherAI/pythia-70m-deduped",
                                                        revision="step3000",
                                                        cache_dir="./pythia-70m-deduped/step3000",
                                                        )
        self.max_token_count=35000
        self.vocab=None
        self.dataset_path="strings_with_frequency.npz" # Has 2 values inputStrings and inputStringsfrequencies
        self.unique_chars_path="unique_characters.npz"
        self.original_corpus_size=0
        self.tokenized_corpus_size=0
        self.data_set_size = 2119719

    def make_vocab(self):
        data = np.load(self.dataset_path)
        input_strings=data["input_strings" ]
        input_strings_frequencies=data["input_strings_frequencies"]

        # Load .npz and extract correctly
        unique_chars_npz = np.load(self.unique_chars_path, allow_pickle=True)
        unique_chars = list(unique_chars_npz["unique_chars"].item()) 
        tokens=create_vocab(input_strings,input_strings_frequencies,self.max_token_count)
        all_tokens = tokens + unique_chars
        if "[UNK]" not in all_tokens:
             all_tokens.append("[UNK]")

        vocab = OrderedDict((token, idx) for idx, token in enumerate(all_tokens))

        self.vocab=vocab


    def load_and_prepare_dataset(self,data_set_size_div:int=2,raw_dataset_path:str = "tinystories_data"):
        if os.path.exists(raw_dataset_path):
            print("Loading dataset from disk...")
            TinyStories = load_from_disk(raw_dataset_path)
        else:
            print("Downloading dataset...")
            TinyStories = load_dataset("roneneldan/TinyStories")
            TinyStories.save_to_disk(raw_dataset_path)

        corpus=[]

        data_set_size=len(TinyStories['train'])

        data_set_size = int(data_set_size/data_set_size_div)

        for i in range(data_set_size):
            corpus.append(TinyStories['train'][i]['text'])

        word_freqs = defaultdict(int)

        print("created the corpus")

        file_name= "word_freqs_"+str(data_set_size)+".pkl"


        if os.path.exists(file_name):
            print("Loading word frequencies from pickle...")
            with open(file_name, "rb") as f:
                word_freqs = pickle.load(f)

        else:
            print("Started working on pre tokenization")
            word_freqs = defaultdict(int)
            
            for i, text in tqdm(enumerate(corpus), total=len(corpus)):
                words_with_offsets = self.pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
                new_words = [word for word, offset in words_with_offsets]
                for word in new_words:
                    word_freqs[word] += 1
            print("Finished working on pre tokenization")
            with open(file_name, "wb") as f:
                pickle.dump(word_freqs, f)
            print("Saved word frequencies to disk.")


        unique_chars = set()
        for word in word_freqs:
            unique_chars.update(word)

        input_strings=list(word_freqs.keys())
        input_strings_frequencies=list(word_freqs.values())

        np.savez(self.dataset_path, input_strings=np.array(input_strings), input_strings_frequencies=np.array(input_strings_frequencies))


    def tokenize_data_set(self,corpus_path:str):

        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)


        print("Started working on pre tokenization")
        word_freqs = defaultdict(int)
        
        for i, text in tqdm(enumerate(corpus), total=len(corpus)):
            words_with_offsets = self.pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                word_freqs[word] += 1
      
        print("We have finished pretokenizing")

        unique_chars = set()
        for word in word_freqs:
            unique_chars.update(word)

        input_strings=list(word_freqs.keys())
        edges_list_weight=list(word_freqs.values())

        num_strings=len(input_strings)

        edges_list=[]
        num_vertices=[]

        for i in range(num_strings):
            string_len=len(input_strings[i])
            edges_list.append(hf.get_strings_from_vocab(input_strings[i],self.vocab) )
            num_vertices.append(string_len+1)

        tokenized_data=tokenize(edges_list,edges_list_weight,num_vertices )

        return tokenized_data


    def get_unique_chars(self):

        corpus=[]

        file_name= "word_freqs_"+str(self.data_set_size)+".pkl"


        if os.path.exists(file_name):
            print("Loading word frequencies from pickle...")
            with open(file_name, "rb") as f:
                word_freqs = pickle.load(f)

        else:
            print("Started working on pre tokenization")
            word_freqs = defaultdict(int)
            
            for i, text in tqdm(enumerate(corpus), total=len(corpus)):
                words_with_offsets = self.pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
                new_words = [word for word, offset in words_with_offsets]
                for word in new_words:
                    word_freqs[word] += 1
            print("Finished working on pre tokenization")
            with open(file_name, "wb") as f:
                pickle.dump(word_freqs, f)
            print("Saved word frequencies to disk.")

        unique_chars = set()
        for word in word_freqs:
            unique_chars.update(word)

        # Convert to NumPy array (dtype=object since it's strings)
        unique_chars_array = np.array(unique_chars, dtype=object)

        # Save to .npz file
        np.savez_compressed("unique_characters.npz", unique_chars=unique_chars_array)

        print(f"Saved {len(unique_chars)} unique characters to 'unique_characters.npz'")




            

            

