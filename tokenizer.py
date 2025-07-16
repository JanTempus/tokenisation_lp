from transformers import AutoTokenizer
from collections import OrderedDict,defaultdict
from setup_lp import create_vocab
import numpy as np
import os
import pickle
import helper_functions as hf

class Tokenizer:
    vocab: OrderedDict
    pretokenizer: AutoTokenizer
    max_token_count:int
    dataset_path:str
    unique_chars_path:str
    original_corpus_size:int
    tokenized_corpus_size:int



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

    def make_vocab(self):
        data = np.load(self.dataset_path)
        input_strings=data["inputStrings" ]
        input_strings_frequencies=data["inputStringsfrequencies"]

        # Load .npz and extract correctly
        unique_chars_npz = np.load(self.unique_chars_path, allow_pickle=True)
        unique_chars = list(unique_chars_npz["unique_chars"].item()) 
        tokens=create_vocab(input_strings,input_strings_frequencies,self.max_token_count)
        vocab = OrderedDict((token, idx) for idx, token in enumerate(all_tokens))
        all_tokens = tokens + unique_chars

        # Ensure there's an unknown token
        if "[UNK]" not in vocab:
            vocab["[UNK]"] = len(vocab)

        self.vocab=vocab

    def load_vocab(self, vocab:OrderedDict):
        self.vocab=vocab

    def tokenize_data_set(self,corpus_path:str):

        with open(corpus_path, "rb") as f:
            corpus = pickle.load(f)


        print("Started working on pre tokenization")
        word_freqs = defaultdict(int)
        
        for i, text in enumerate(corpus):
            words_with_offsets = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
            new_words = [word for word, offset in words_with_offsets]
            for word in new_words:
                word_freqs[word] += 1
      

        unique_chars = set()
        for word in word_freqs:
            unique_chars.update(word)

        inputStrings=list(word_freqs.keys())
        inputStringsfrequencies=list(word_freqs.values())






    

    

