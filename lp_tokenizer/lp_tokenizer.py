from transformers import AutoTokenizer
from collections import OrderedDict,defaultdict
from lp_functions import create_vocab,tokenize, deterministic_rounding,probabilistic_rounding,fill_missing_edges_with_unk
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
    unk_token:str
    eot_token:str

    

    def __init__(self,dataset_url=None,saved_dataset_path=None,
                 dataset_size=0,max_dataset_size=0,
                 vocab_size=0,vocab=None,
                 unk_token=None,eot_token=None,pretokenizer=None):
        
        if pretokenizer is None:    
            self.pretokenizer=AutoTokenizer.from_pretrained(
                                                            "EleutherAI/pythia-70m-deduped",
                                                            revision="step3000",
                                                            cache_dir="./pythia-70m-deduped/step3000",
                                            )
        else:
            self.pretokenizer=pretokenizer                                                                                                                
        self.vocab=vocab
        self.saved_dataset_path=saved_dataset_path
        self.vocab_size=vocab_size
        self.dataset_size = dataset_size
        self.max_dataset_size=max_dataset_size
        self.dataset_url= dataset_url
        self.debug=False
        self.unk_token=unk_token
        self.eot_token=eot_token     

    def make_vocab(self,save_vocab:bool=True,input_strings=None,input_strings_frequencies=None,unique_chars=None):

        if self.dataset_url is None and self.saved_dataset_path is None:
            raise ValueError("Must include either dataset_url or dataset_path")
       
        if self.saved_dataset_path is not None:
            if os.path.exists(self.saved_dataset_path):
                dataset=load_from_disk(self.saved_dataset_path)
            else:
                dataset=load_dataset(self.dataset_url)
                dataset.save_to_disk(self.saved_dataset_path)


        if self.max_dataset_size == 0:
            self.max_dataset_size=len(dataset['train'])


        if input_strings is None:
            input_strings,  input_strings_frequencies = self.pretokenize_and_prepare_dataset(self.dataset_size,dataset)


        if unique_chars is None:
            unique_chars = self.get_unique_chars(dataset,self.max_dataset_size)
      
        special_char_count=0
        special_tokens=[]

        if self.unk_token is None:
            special_tokens.append("[UNK]")
            self.unk_token="[UNK]"
            special_char_count+=1

        if self.eot_token is None:
            special_tokens.append("<|endoftext|>")
            self.eot_token="<|endoftext|>" 
            special_char_count+=1

        lp_budget=self.vocab_size-len(unique_chars)-special_char_count # Minus 2 for the special tokens unknown and end of text
        
        if lp_budget <= 0:
            raise ValueError("Vocab size is too small, entire vocab already unique characters")


        possible_tokens=create_vocab(input_strings,input_strings_frequencies,lp_budget)
        
        # Change this depending on what behaviour one would like
        # Minus special_char_count as we add two special tokens

        
        rounded_tokens=deterministic_rounding(possible_tokens,unique_chars,self.vocab_size-special_char_count)

        all_tokens=special_tokens+rounded_tokens
             
              
        if len(all_tokens) != self.vocab_size:
            print(f"number of tokens {len(all_tokens)}, vocab size {self.vocab_size} all tokens {len(possible_tokens)}")
            assert(len(all_tokens)==self.vocab_size)

        vocab = OrderedDict((token, idx) for idx, token in enumerate(all_tokens))
   
            
        if save_vocab:
            vocab_length = len(all_tokens)
            file_name = f"vocab_{self.saved_dataset_path}_{self.dataset_size}_{vocab_length}.json"

            folder_name = "vocabs"
            os.makedirs(folder_name, exist_ok=True)

            file_path = os.path.join(folder_name, file_name)
            with open(file_path, "w") as f:
                json.dump(vocab, f)
        self.vocab=vocab

    
    def pretokenize_and_prepare_dataset(self, dataset_size,dataset,input_strings=None, save:bool=True):
        base_name = f"word_freqs_testing{self.saved_dataset_path}{dataset_size}"
        strings_file = base_name + "_strings.npy"
        freqs_file = base_name + "_freqs.npy"
    
        if os.path.exists(strings_file) and os.path.exists(freqs_file):
            print("Loading .npy files")
            input_strings = np.load(strings_file, allow_pickle=True).tolist()
            input_strings_frequencies = np.load(freqs_file).tolist()
            
        else:
            corpus=[]

            for i in tqdm(range(dataset_size),desc="Appending text to the corpus"):
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
           
            if save:
            # Save as .npy for faster reloads
                np.save(strings_file, np.array(input_strings, dtype=object),allow_pickle=True)
                np.save(freqs_file, np.array(input_strings_frequencies, dtype=np.int64))

        return input_strings, input_strings_frequencies

    def encode(self,corpus:list[str], vocab, just_size:bool=False):
        
        if self.unk_token is None:
            raise KeyError("Please assign a token to the unkown token")

        input_strings=[]
        #for i, text in tqdm(enumerate(corpus), total=len(corpus), desc="Pretokenizing"):

        for text in corpus:
            words_with_offsets = self.pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            input_strings+=new_words
    
    
     
        num_strings=len(input_strings)

        edges_list=[]
        num_vertices=[]

        #unk_id=vocab[self.unk_token]

        # for i in tqdm(range(num_strings), desc="Processing strings"):
        for i in range(num_strings):
            string_len=len(input_strings[i])
            edges=hf.get_strings_from_vocab(input_strings[i],vocab)
            #edges_corrected=fill_missing_edges_with_unk(edges,string_len+1,self.unk_token,unk_id)
            edges_list.append(edges)
            num_vertices.append(string_len+1)
        
        edges_list_weight=np.ones(len(edges_list),dtype=float)
        tokenized_data=tokenize(edges_list,edges_list_weight,num_vertices)

     
        return tokenized_data
      
                    
            
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
      
