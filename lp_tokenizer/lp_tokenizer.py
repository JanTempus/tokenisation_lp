from transformers import AutoTokenizer
from datasets import Dataset, Features, Sequence, Value
from tokenizers.pre_tokenizers import ByteLevel
from collections import OrderedDict,defaultdict
from lp_tokenizer.lp_functions import (
    create_vocab,
    create_vocab_cuopt,
    prepare_cuopt_model,
    solve_vocab_on_model,
    tokenize,
    deterministic_rounding,
    probabilistic_rounding,
    fill_missing_edges_with_unk,
)
from lp_tokenizer.datastructures import tokenInstance
import numpy as np
import os
import pickle
import json
import lp_tokenizer.helper_functions as hf
import matplotlib.pyplot as plt
import pickle
from concurrent.futures import ProcessPoolExecutor
import csv
import time


BYTE_LEVEL_ALPHABET = sorted(ByteLevel.alphabet())


def _pretokenize_batch(batch, indices, pretokenizer):
    word_freqs = defaultdict(int)
    empty_token_count = 0
    empty_token_text_count = 0
    empty_token_indices = []
    empty_token_previews = []

    for text_idx, text in zip(indices, batch["text"]):
        if not isinstance(text, str) or not text:
            continue

        words_with_offsets = (
            pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        )
        saw_empty_token = False
        for word, _ in words_with_offsets:
            if word == "":
                empty_token_count += 1
                saw_empty_token = True
                continue
            word_freqs[word] += 1

        if saw_empty_token:
            empty_token_text_count += 1
            if len(empty_token_indices) < 5:
                empty_token_indices.append(text_idx)
                empty_token_previews.append(text[:80].replace("\n", "\\n"))

    return {
        "tokens": [list(word_freqs.keys())],
        "frequencies": [list(word_freqs.values())],
        "empty_token_count": [empty_token_count],
        "empty_token_text_count": [empty_token_text_count],
        "empty_token_indices": [empty_token_indices],
        "empty_token_previews": [empty_token_previews],
    }


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

    

    def __init__(self,
                 corpus,
                 vocab_size,
                 special_tokens,
                 unique_chars=None,
                 pretokenizer=None):

        if pretokenizer is None:
            self.pretokenizer=AutoTokenizer.from_pretrained(
                                                            "EleutherAI/pythia-70m-deduped",
                                                            revision="step3000",
                                                            cache_dir="./pythia-70m-deduped/step3000",
                                            )
        else:
            self.pretokenizer=pretokenizer

        self.unique_chars = (
            list(BYTE_LEVEL_ALPHABET)
            if unique_chars is None
            else list(unique_chars)
        )
        self.corpus=corpus
        self.vocab_size=vocab_size
        self.special_tokens_list=special_tokens
        self.debug=False
      
       
    def make_vocab(self):

        if self.corpus is None:
            raise ValueError("Must include a corpus")

        pretoken_dataset = self.pretokenize_and_prepare_corpus(self.corpus)

        special_tokens = list(self.special_tokens_list)
        lp_budget = self.vocab_size - len(self.unique_chars) - len(special_tokens)

        if lp_budget <= 0:
            raise ValueError("Vocab size is too small, entire vocab already unique characters")

        possible_tokens = create_vocab(input_strings, input_strings_frequencies, lp_budget, self.vocab_size)
        
       

        # Change this depending on what behaviour one would like
        # Minus special_char_count as we add two special tokens
      
        return {"possible_tokens": possible_tokens,"unique_chars":self.unique_chars,"special_tokens":special_tokens}
        # self.vocab_size=min(len(possible_tokens)+len(unique_chars)-special_char_count,self.vocab_size)  
        
        # rounded_tokens=deterministic_rounding(possible_tokens,unique_chars,self.vocab_size-special_char_count)

        # all_tokens=special_tokens+rounded_tokens
             
              
        # if len(all_tokens) != self.vocab_size:
        #     print(f"number of tokens {len(all_tokens)}, vocab size {self.vocab_size} all tokens {len(possible_tokens)} rounded tokens {len(rounded_tokens)} ")
        #     assert(len(all_tokens)==self.vocab_size)

        # vocab = OrderedDict((token, idx) for idx, token in enumerate(all_tokens))
   
        # self.vocab=vocab


    def make_vocab_cuopt(self, solver_parameters=None, verbose: bool = True,
                         morphology_rho: float = 0.0, celex_dir: str = None,
                         unmatched_report_path: str = None):

        if self.corpus is None:
            raise ValueError("Must include a corpus")

        pretoken_dataset = self.pretokenize_and_prepare_corpus(self.corpus)
        input_strings = pretoken_dataset["pretoken"]
        input_strings_frequencies = pretoken_dataset["frequency"]

        special_tokens = list(self.special_tokens_list)
        lp_budget = self.vocab_size - len(self.unique_chars) - len(special_tokens)
        if lp_budget <= 0:
            raise ValueError("Vocab size is too small, entire vocab already unique characters")

        possible_tokens = create_vocab_cuopt(
            inputStringList=None,
            inputStringFreq=None,
            numAllowedTokens=lp_budget,
            vocab_size=self.vocab_size,
            pretoken_dataset=pretoken_dataset,
            morphology_rho=morphology_rho,
            celex_dir=celex_dir,
            unmatched_report_path=unmatched_report_path,
        )

        return {"possible_tokens": possible_tokens, "unique_chars": self.unique_chars, "special_tokens": special_tokens}


    def prepare_cuopt_model(self, verbose: bool = True,
                            morphology_rho: float = 0.0,
                            celex_dir: str = None,
                            unmatched_report_path: str = None):
        if self.corpus is None:
            raise ValueError("Must include a corpus")

        total_start = time.perf_counter()
        print("[pipeline] Starting corpus preparation and cuOpt model construction")
        pretoken_dataset = self.pretokenize_and_prepare_corpus(self.corpus)
        print(
            f"[pipeline] Corpus preparation returned "
            f"{len(pretoken_dataset):,} unique pre-tokens after "
            f"{time.perf_counter() - total_start:.1f}s"
        )

        model_start = time.perf_counter()
        print("[pipeline] Starting LP data and cuOpt model construction")
        self._cuopt_model = prepare_cuopt_model(
            pretoken_dataset=pretoken_dataset,
            verbose=verbose,
            morphology_rho=morphology_rho,
            celex_dir=celex_dir,
            unmatched_report_path=unmatched_report_path,
        )
        print(
            f"[pipeline] cuOpt model construction finished in "
            f"{time.perf_counter() - model_start:.1f}s "
            f"({time.perf_counter() - total_start:.1f}s total)"
        )
        return self._cuopt_model


    def solve_for_vocab_size(self, vocab_size: int,
                             solver_parameters=None, verbose: bool = True):
        if not hasattr(self, "_cuopt_model") or self._cuopt_model is None:
            raise RuntimeError("Call prepare_cuopt_model() before solve_for_vocab_size().")

        special_tokens = list(self.special_tokens_list)
        lp_budget = vocab_size - len(self.unique_chars) - len(special_tokens)
        if lp_budget <= 0:
            raise ValueError(
                f"Vocab size {vocab_size} too small: unique_chars={len(self.unique_chars)} "
                f"+ special_tokens={len(special_tokens)} already exceeds budget."
            )

        print(f"[solve_for_vocab_size] vocab_size={vocab_size} lp_budget={lp_budget}")

        result = solve_vocab_on_model(
            self._cuopt_model,
            numAllowedTokens=lp_budget,
            solver_parameters=solver_parameters,
            verbose=verbose,
        )

        return {
            "possible_tokens": result["possible_tokens"],
            "unique_chars": self.unique_chars,
            "special_tokens": special_tokens,
            "x_values": result["x_values"],
        }


    def generate_vocab_nonzero(self,input_strings,input_strings_frequencies,unique_chars):
      
        special_char_count=0
        special_tokens=[]

        if self.unk_token is None:
            special_tokens.append("[UNK]")
            self.unk_token="[UNK]"
            special_char_count+=1

        if self.eot_token is None:
            special_tokens.append("[EOS]")
            self.eot_token="[EOS]" 
            special_char_count+=1

        lp_budget=self.vocab_size-len(unique_chars)-special_char_count # Minus 2 for the special tokens unknown and end of text
        
        if lp_budget <= 0:
            raise ValueError("Vocab size is too small, entire vocab already unique characters")


        possible_tokens=create_vocab(input_strings,input_strings_frequencies,lp_budget,self.vocab_size)

        tokens_flat=[token.token for token in possible_tokens]
        
        return tokens_flat


    def pretokenize_and_prepare_corpus(self, corpus):
        total_start = time.perf_counter()
        if isinstance(corpus, Dataset):
            corpus_dataset = corpus
        else:
            corpus_dataset = Dataset.from_dict({"text": list(corpus)})

        if "text" not in corpus_dataset.column_names:
            raise ValueError("Corpus dataset must contain a 'text' column")

        word_freqs = defaultdict(int)
        empty_token_count = 0
        empty_token_text_count = 0
        empty_token_examples = []

        if len(corpus_dataset) > 0:
            batch_size = int(os.environ.get("BATCH_SIZE", "10000"))
            num_proc = int(os.environ.get("NUM_PROC", "16"))
            aggregate_features = Features(
                {
                    "tokens": Sequence(Value("string")),
                    "frequencies": Sequence(Value("int64")),
                    "empty_token_count": Value("int64"),
                    "empty_token_text_count": Value("int64"),
                    "empty_token_indices": Sequence(Value("int64")),
                    "empty_token_previews": Sequence(Value("string")),
                }
            )
            print(
                f"[pretokenize] Starting worker map: "
                f"rows={len(corpus_dataset):,}, num_proc={num_proc}, "
                f"batch_size={batch_size:,}"
            )
            map_start = time.perf_counter()
            aggregates = corpus_dataset.map(
                _pretokenize_batch,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                with_indices=True,
                fn_kwargs={"pretokenizer": self.pretokenizer},
                remove_columns=corpus_dataset.column_names,
                features=aggregate_features,
                desc="Pretokenizing corpus",
            )
            print(
                f"[pretokenize] Worker map finished in "
                f"{time.perf_counter() - map_start:.1f}s; "
                f"partial frequency tables={len(aggregates):,}"
            )

            merge_start = time.perf_counter()
            print("[pretokenize] Merging partial frequency tables")
            for tokens, frequencies in zip(
                aggregates["tokens"], aggregates["frequencies"]
            ):
                for word, frequency in zip(tokens, frequencies):
                    word_freqs[word] += frequency
            print(
                f"[pretokenize] Frequency merge finished in "
                f"{time.perf_counter() - merge_start:.1f}s; "
                f"unique pre-tokens={len(word_freqs):,}"
            )

            empty_token_count = sum(aggregates["empty_token_count"])
            empty_token_text_count = sum(aggregates["empty_token_text_count"])
            empty_token_examples = sorted(
                (
                    (text_idx, preview)
                    for indices, previews in zip(
                        aggregates["empty_token_indices"],
                        aggregates["empty_token_previews"],
                    )
                    for text_idx, preview in zip(indices, previews)
                ),
                key=lambda example: example[0],
            )[:5]

        input_strings = list(word_freqs.keys())
        input_strings_frequencies = list(word_freqs.values())
        if empty_token_count > 0:
            print(
                f"[WARN] Found {empty_token_count} empty pretokenized strings "
                f"across {empty_token_text_count} corpus entries."
            )
            for text_idx, preview in empty_token_examples:
                print(f"[WARN] Empty-token example at corpus index {text_idx}: '{preview}'")
            if os.environ.get("FAIL_ON_EMPTY_PRETOKENIZED_STRINGS", "0") == "1":
                raise ValueError(
                    "Empty pretokenized strings detected. "
                    "Set FAIL_ON_EMPTY_PRETOKENIZED_STRINGS=0 to continue."
                )
        pretoken_dataset = Dataset.from_dict(
            {
                "pretoken": input_strings,
                "frequency": input_strings_frequencies,
            },
            features=Features(
                {
                    "pretoken": Value("string"),
                    "frequency": Value("int64"),
                }
            ),
        )
        print(
            f"[pretokenize] Corpus preparation finished in "
            f"{time.perf_counter() - total_start:.1f}s; "
            f"unique pre-tokens={len(pretoken_dataset):,}; "
            f"total pre-token occurrences={sum(input_strings_frequencies):,}"
        )

        return pretoken_dataset

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

            for i in range(dataset_size):
                corpus.append(dataset['train'][i]['text'])

            word_freqs = defaultdict(int)
            empty_token_count = 0
            empty_token_text_count = 0
            empty_token_examples = []
            
            for i, text in enumerate(corpus):
                words_with_offsets = self.pretokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
                saw_empty_token = False
                for word, _ in words_with_offsets:
                    if word == "":
                        empty_token_count += 1
                        saw_empty_token = True
                        continue
                    word_freqs[word] += 1
                if saw_empty_token:
                    empty_token_text_count += 1
                    if len(empty_token_examples) < 5:
                        text_preview = text[:80].replace("\n", "\\n")
                        empty_token_examples.append((i, text_preview))

            input_strings=list(word_freqs.keys())
            input_strings_frequencies=list(word_freqs.values())

            if empty_token_count > 0:
                print(
                    f"[WARN] Found {empty_token_count} empty pretokenized strings "
                    f"across {empty_token_text_count} dataset entries."
                )
                for text_idx, preview in empty_token_examples:
                    print(f"[WARN] Empty-token example at dataset index {text_idx}: '{preview}'")
                if os.environ.get("FAIL_ON_EMPTY_PRETOKENIZED_STRINGS", "0") == "1":
                    raise ValueError(
                        "Empty pretokenized strings detected. "
                        "Set FAIL_ON_EMPTY_PRETOKENIZED_STRINGS=0 to continue."
                    )
           
            if save:
            # Save as .npy for faster reloads
                np.save(strings_file, np.array(input_strings, dtype=object),allow_pickle=True)
                np.save(freqs_file, np.array(input_strings_frequencies, dtype=np.int64))
        print("pretokenize_and_prepare_dataset finished")

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
      
    def get_vocab(self):
        return self.vocab


    def check_number_edges(self,inputStringList: list[str],input_strings_freq,
                        minTokenCount: int = 1):
        
        numStrings = len(inputStringList)

        edgesList = []
        tokensList = []
        freeEdgesList = []
        numVertices = []

        for i in range(numStrings):
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
        print("check_number_edges finished")
