from transformers import AutoTokenizer
from collections import OrderedDict,defaultdict
from lp_functions import create_vocab,tokenize, deterministic_rounding,probabilistic_rounding,fill_missing_edges_with_unk, shortest_tokenization_path
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
from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map  # or thread_map
import csv


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


        possible_tokens=create_vocab(input_strings,input_strings_frequencies,lp_budget,self.vocab_size)
        
        # Change this depending on what behaviour one would like
        # Minus special_char_count as we add two special tokens
        self.vocab_size=min(len(possible_tokens)+len(unique_chars)-special_char_count,self.vocab_size)
        
        rounded_tokens=deterministic_rounding(possible_tokens,unique_chars,self.vocab_size-special_char_count)

        all_tokens=special_tokens+rounded_tokens
             
              
        if len(all_tokens) != self.vocab_size:
            print(f"number of tokens {len(all_tokens)}, vocab size {self.vocab_size} all tokens {len(possible_tokens)} rounded tokens {len(rounded_tokens)} ")
            assert(len(all_tokens)==self.vocab_size)

        vocab = OrderedDict((token, idx) for idx, token in enumerate(all_tokens))
   
            
        if save_vocab:
            vocab_length = len(all_tokens)
            file_name = f"vocab_{self.saved_dataset_path}_{vocab_length}.json"

            folder_name = "vocabs_new"
            os.makedirs(folder_name, exist_ok=True)

            file_path = os.path.join(folder_name, file_name)
            with open(file_path, "w") as f:
                json.dump(vocab, f)
        self.vocab=vocab


    def generate_vocab_nonzero(self,input_strings,input_strings_frequencies,unique_chars):
      
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


        possible_tokens=create_vocab(input_strings,input_strings_frequencies,lp_budget,self.vocab_size)

        tokens_flat=[token.token for token in possible_tokens]
        
        return tokens_flat



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

            input_strings=list(word_freqs.keys())
            input_strings_frequencies=list(word_freqs.values())
           
            if save:
            # Save as .npy for faster reloads
                np.save(strings_file, np.array(input_strings, dtype=object),allow_pickle=True)
                np.save(freqs_file, np.array(input_strings_frequencies, dtype=np.int64))

        return input_strings, input_strings_frequencies

    def encode(self,corpus:list[str], vocab):
        if self.unk_token is None:
            raise KeyError("Please assign a token to the unkown token")

        input_strings=[]
        words_with_offsets=self.pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(corpus)
        input_strings= [word for word, offset in words_with_offsets]

     
        num_strings=len(input_strings)

        edges_list=[]
        num_vertices=[]


        # for i in tqdm(range(num_strings), desc="Processing strings"):
        for i in range(num_strings):
            string_len=len(input_strings[i])
            edges=hf.get_strings_from_vocab(input_strings[i],vocab)
            #edges_corrected=fill_missing_edges_with_unk(edges,string_len+1,self.unk_token,0)#0 is the unkown ID
            if len(edges)>0: 
                edges_list.append(edges)
                num_vertices.append(string_len+1)
            
        edges_list_weight=np.ones(len(edges_list),dtype=float)
        tokenized_data=tokenize(edges_list,edges_list_weight,num_vertices)

     
        return tokenized_data
      


    def encode_combinatorial(self,corpus:list[str], vocab):
        if self.unk_token is None:
            raise KeyError("Please assign a token to the unkown token")

        input_strings=[]
        words_with_offsets=self.pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(corpus)
        input_strings= [word for word, offset in words_with_offsets]

     
        num_strings=len(input_strings)

        edges_list=[]
        num_vertices=[]


        # for i in tqdm(range(num_strings), desc="Processing strings"):
        for i in range(num_strings):
            string_len=len(input_strings[i])
            edges=hf.get_strings_from_vocab(input_strings[i],vocab)
            #edges_corrected=fill_missing_edges_with_unk(edges,string_len+1,self.unk_token,0)#0 is the unkown ID
            if len(edges)>0: 
                edges_list.append(edges)
                num_vertices.append(string_len+1)
            
       
        tokenized_data=shortest_tokenization_path(edges_list,num_vertices)

     
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



    def extract_unique_from_text(args):
        text, pretokenizer = args
        words_with_offsets = pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        tokens = [word for word, _ in words_with_offsets]
        chars = set()
        for token in tokens:
            chars.update(token)
        return chars


    def get_unique_chars_parallel(self, dataset, dataset_size, pretokenizer, num_proc=4):
        """
        Collect unique characters from the pretokenized dataset in parallel
        using tqdm.contrib.concurrent's process_map.
        """

        file_name = f"unique_chars{self.saved_dataset_path}{dataset_size}.npy"

        if os.path.exists(file_name):
            unique_chars = np.load(file_name, allow_pickle=True).tolist()
        else:
            texts = [dataset['train'][i]['text'] for i in range(dataset_size)]

            results = process_map(
                self.extract_unique_from_text,
                [(t, pretokenizer) for t in texts],
                max_workers=num_proc,
                chunksize=1,
                desc="Getting Unique characters (parallel)"
            )

            unique_chars = set()
            for chars in results:
                unique_chars.update(chars)

            unique_chars = sorted(unique_chars)
            np.save(file_name, np.array(unique_chars, dtype=object))

        return unique_chars




    def get_vocab(self):
        return self.vocab


    def check_number_edges(self,inputStringList: list[str],input_strings_freq,
                        minTokenCount: int = 1):
        
        numStrings = len(inputStringList)

        edgesList = []
        tokensList = []
        freeEdgesList = []
        numVertices = []

        for i in tqdm(range(numStrings), desc="Converting data to graph format"):
            stringLen = len(inputStringList[i])
            edgesList.append(hf.get_all_nonFree_substrings(inputStringList[i]))
            tokensList.append(hf.get_tokens(inputStringList[i]))
            freeEdgesList.append(hf.get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen + 1)
        
        # Flatten all tokens and remove duplicates
        tokens = list(set([item for sublist in tokensList for item in sublist]))
        

        hf.update_token_instance_counts(tokens,input_strings_freq,edgesList)
        # Filter tokens by minTokenCount
        tokens_to_keep = [token for token in tokens if token.token_instance_count > minTokenCount]
        keep_set = set(t.token for t in tokens_to_keep)

        # Filter edges by the tokens we keep
        filtered_edgesList = [
            [token for token in sublist if token.token in keep_set]
            for sublist in edgesList
        ]

        # Compute total number of edges
        total_edges = sum(len(sublist) for sublist in filtered_edgesList)
        # Compute total number of tokens
        total_tokens = len(tokens_to_keep)

        print(f"Total edges {total_edges}, total tokens {total_tokens}" )