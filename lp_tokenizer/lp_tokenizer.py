from transformers import AutoTokenizer
from collections import OrderedDict,defaultdict
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from lp_functions import create_vocab,tokenize, deterministic_rounding,probabilistic_rounding
from datastructures import tokenInstance
import numpy as np
import os
import pickle
import json
import helper_functions as hf
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm


class Tokenizer:
    vocab: OrderedDict
    pretokenizer: AutoTokenizer
    saved_dataset_path:str
    dataset_size:int
    max_dataset_size:int
    dataset_url:str
    vocab_size:int

    

    def __init__(self,dataset_url,saved_dataset_path,dataset_size,vocab_size,vocab=None):
        self.pretokenizer=AutoTokenizer.from_pretrained(
                                                        "EleutherAI/pythia-70m-deduped",
                                                        revision="step3000",
                                                        cache_dir="./pythia-70m-deduped/step3000",
                                                        )
        self.vocab=vocab
        self.saved_dataset_path=saved_dataset_path
        self.vocab_size=vocab_size
        self.dataset_size = dataset_size
        self.dataset_url= dataset_url
        self.debug=False


    def make_vocab(self,save_vocab:bool=True):

        if self.dataset_url is None and self.dataset_path is None:
            raise ValueError("Must include either dataset_url or dataset_path")
       
        if os.path.exists(self.saved_dataset_path):
            dataset=load_from_disk(self.saved_dataset_path)
        else:
            dataset=load_dataset(self.dataset_url)
            dataset.save_to_disk(self.saved_dataset_path)
    
        self.max_dataset_size=len(dataset['train'])
        input_strings,  input_strings_frequencies = self.pretokenize_and_prepare_dataset(self.dataset_size,dataset)

        unique_chars = self.get_unique_chars(dataset,1000000)
      
        lp_budget=self.vocab_size-len(unique_chars)-2 # Minus 2 for the special tokens unknown and end of text
        
        if lp_budget <= 0:
            raise ValueError("Vocab size is too small, entire vocab already unique characters")


        possible_tokens=create_vocab(input_strings,input_strings_frequencies,lp_budget)
        
        # Change this depending on what behaviour one would like
        # Minus 2 as we add two special tokens
        all_tokens=deterministic_rounding(possible_tokens,unique_chars,self.vocab_size-2)
        if "[UNK]" not in all_tokens:
             all_tokens.append("[UNK]")

        if "<|endoftext|>" not in all_tokens:
            all_tokens.append("<|endoftext|>")   

        assert(len(all_tokens)==self.vocab_size)

        vocab = OrderedDict((token, idx) for idx, token in enumerate(all_tokens))
   
        if save_vocab:
            vocab_length=len(all_tokens)
            file_name= os.path.join("vocab_" + self.saved_dataset_path + f"{vocab_length}.json")
            with open(file_name, "w") as f:
                json.dump(vocab, f)
        self.vocab=vocab


    def pretokenize_and_prepare_dataset(self, dataset_size,dataset):
        base_name = f"word_freqs_testing{self.saved_dataset_path}{dataset_size}"
        strings_file = base_name + "_strings.npy"
        freqs_file = base_name + "_freqs.npy"
    
        if os.path.exists(strings_file) and os.path.exists(freqs_file):
            print("Loading .npy files")
            input_strings = np.load(strings_file, allow_pickle=True).tolist()
            input_strings_frequencies = np.load(freqs_file).tolist()
            
        else:
            print("Creating input strings and input strings freq")
            corpus=[]

            for i in range(dataset_size):
                corpus.append(dataset['train'][i]['text'])

            word_freqs = defaultdict(int)
            
            for i, text in tqdm(enumerate(corpus), total=len(corpus), desc="Pretokenizing"):
                words_with_offsets = self.pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
                new_words = [word for word, offset in words_with_offsets]
                for word in new_words:
                    word_freqs[word] += 1

            unique_chars = set()
            for word in word_freqs:
                unique_chars.update(word)

            input_strings=list(word_freqs.keys())
            input_strings_frequencies=list(word_freqs.values())

            # Save as .npy for faster reloads
            np.save(strings_file, np.array(input_strings, dtype=object),allow_pickle=True)
            np.save(freqs_file, np.array(input_strings_frequencies, dtype=np.int64))

        return input_strings, input_strings_frequencies

    def encode(self,corpus:list[str],vocab):
     
        input_strings=[]
        for i, text in tqdm(enumerate(corpus), total=len(corpus, desc="Pretokenizing")):
            words_with_offsets = self.pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            input_strings+=new_words
       
    
     
        num_strings=len(input_strings)

        edges_list=[]
        num_vertices=[]

        unk_token_instance=[tokenInstance(
                    token="[UNK]",
                    start=0,
                    end=1,
                    token_index=vocab.get("[UNK]", -1)
                )]

        for i in range(num_strings):
            string_len=len(input_strings[i])
            edges=hf.get_strings_from_vocab(input_strings[i],vocab)
            if(edges != []):
                edges_list.append(edges )
                num_vertices.append(string_len+1)
            else:
                edges_list.append(unk_token_instance)
                num_vertices.append(2)
            
        
        edges_list_weight=np.ones(len(edges_list),dtype=float)
        tokenized_data=tokenize(edges_list,edges_list_weight,num_vertices )

        
        flat_tokens = []
        for sublist in tokenized_data:
            flat_tokens.extend(sublist)
        return flat_tokens
      
                    
            
    def get_unique_chars(self, dataset,dataset_size):
        """
        Collect unique characters from the pretokenized dataset.
        Uses pre_tokenize_str to get tokens, then collects all unique characters from those tokens.
        """

        file_name = f"unique_chars{self.saved_dataset_path}{dataset_size}.npy"

        if os.path.exists(file_name):
            # Load directly from .npy
            unique_chars = np.load(file_name, allow_pickle=True).tolist()
        else:
            unique_chars = set()

            for i in tqdm(range(dataset_size), desc="Getting Unique characters"):
                text = dataset['train'][i]['text']
                words_with_offsets = self.pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
                tokens = [word for word, _ in words_with_offsets]
                for token in tokens:
                    unique_chars.update(token)

            unique_chars = sorted(unique_chars)
            np.save(file_name, np.array(unique_chars, dtype=object))

        return unique_chars

    
    def get_vocab(self):
        return self.vocab
      
