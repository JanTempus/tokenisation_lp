"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.

This module is derived from and extends the minBPE project by Andrej Karpathy
(https://github.com/karpathy/minbpe), which is MIT licensed.
"""

# need to remove the . to run the tests at the bottom
from .util import frombytes, tobytes

import regex as re

from typing import Tuple, Optional, TextIO

# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer

# TODO: convert to proper tests rather than code below

# the old form used for justbperegex.py
def get_stats_simple(tokens : list[bytes],
              counts : Optional[dict[Tuple[bytes,bytes],int]] = None,
              multiplier : int = 1,
              ) -> dict[Tuple[bytes,bytes],int]:
    """
    Given a list of bytes object tokens,
    return a dictionary of counts of consecutive pairs
    Example: [b'a', b'b', b'c', b'a', b'b'] -> {(b'a', b'b'): 2,
                                                (b'b', b'c'): 1,
                                                (b'c', b'a'): 1}
    Optionally allows an existing dictionary of counts to update
    """
    counts = {} if counts is None else counts
    prev_pair = None
    for pair in zip(tokens, tokens[1:]): # iterate consecutive elements
        if not (pair[0] == pair[1] and pair == prev_pair):
            counts[pair] = counts.get(pair, 0) + multiplier
            prev_pair = pair
        else:
            prev_pair = None # do the next one
    return counts

def get_stats(tokens : list[bytes], 
              counts : Optional[dict[Tuple[bytes,bytes],int]] = None,
              unique_counts : Optional[dict[Tuple[bytes,bytes],int]] = None,  # like IDF
              multiplier : int = 1,  # cnt or -cnt, or 1 for tests
              unique_multiplier : int = 1  # is +1 or -1
              ) -> Tuple[dict[Tuple[bytes,bytes],int], dict[Tuple[bytes,bytes],int]]:
    """
    Given a list of bytes object tokens, 
    return a dictionary of counts of consecutive pairs
    Example: [b'a', b'b', b'c', b'a', b'b'] -> {(b'a', b'b'): 2, 
                                                (b'b', b'c'): 1, 
                                                (b'c', b'a'): 1}
    Optionally allows an existing dictionary of counts to update
    """
    # deduplicate in case pair appears multiple times
    unique_pairs = set()

    counts = {} if counts is None else counts
    prev_pair = None
    for pair in zip(tokens, tokens[1:]): # iterate consecutive elements
        if not (pair[0] == pair[1] and pair == prev_pair):
            counts[pair] = counts.get(pair, 0) + multiplier
            unique_pairs.add(pair)
            prev_pair = pair
        else:
            prev_pair = None # do the next one

    unique_counts = {} if unique_counts is None else unique_counts
    for pair in unique_pairs:
        unique_counts[pair] = unique_counts.get(pair, 0) + unique_multiplier    # these appear in this chunk
        
    return counts, unique_counts


def min_merge(tokens : list[bytes], 
              merges : dict[Tuple[bytes,bytes],int]) -> Optional[Tuple[bytes,bytes]]:
    """
    Given a list of bytes object tokens, 
    and a dictionary of merges, return the pair with the lowest index, 
    or None if none are found
    """
    min_pair = None # returns None if nothing to merge 
    min_cnt = len(merges) + 1  # initialize to a value larger than any index

    for pair in zip(tokens, tokens[1:]): # iterate consecutive elements
        if pair in merges and merges[pair] < min_cnt:
            min_cnt = merges[pair]
            min_pair = pair
    
    return min_pair

# TODO: write tests
def merge(tokens : list[bytes], pair : Tuple[bytes,bytes]) -> Tuple[list[bytes],int]:
    """
    In the list of tokens, replace all consecutive occurrences
    of pair (t1,t2) with the combined token t1+t2
    Example: tokens=[b'a', b'b', b'c', b'a', b'b'], 
    pair=(b'a', b'b') -> [b'ab', b'c', b'ab']
    will have max_count merges found, 
    unless there are the pair elements are runs of the same
    """
    newtokens = []
    i = 0
    merge_cnt = 0
    left, right = pair
    merged = left + right
    while i < len(tokens):
        # if not at the very last position AND the pair matches, replace it
        if tokens[i] == left and i < len(tokens) - 1 and tokens[i+1] == right:
            newtokens.append(merged)
            merge_cnt += 1
            i += 2
        else:
            newtokens.append(tokens[i])
            i += 1
    
    # did we find what we expected?
    # note that in the case of (b'2', b'2') and a string with '222222'
    # we may count more merges then we can execute since we really 
    # can only execute non-overlapping counts
    # ignore if doing multiple documents
    # we should have done enough checks before calling that we never get 0 here
    assert merge_cnt > 0

    return newtokens, merge_cnt

# write the dict, first the number of them, then each one 
# the keys are either bytes, or a pair (bytes,bytes), depending 
# on ispair
def _write_sorted_dict(d : dict, f : TextIO, ispair : bool, isstr : bool) -> None:
        
    # write the size, so we don't need to care about the indices being continuous
    f.write(f"{len(d)}\n")
    sortedd = [(idx, k) for k, idx in d.items()]
    if ispair:
        for idx, (tok1,tok2) in sortedd:
            if isstr:
                f.write(f"{idx} {tok1} {tok2}\n")
            else:
                f.write(f"{idx} {frombytes(tok1)} {frombytes(tok2)}\n")
    else:
        for idx, tok in sortedd:
            if isstr:
                f.write(f"{idx} {tok}\n")
            else:
                f.write(f"{idx} {frombytes(tok)}\n")


# read the dict 
def _read_sorted_dict(f : TextIO, ispair : bool, isstr : bool) -> dict:
        
    # read the size
    n = int(f.readline().rstrip("\n") )

    d = {}
    if ispair:
        for i in range(n):
            line = f.readline().rstrip("\n").split(" ")
            assert len(line) == 3, f"expected 3 fields: {line} on line {i} of {n}"
            idx, tok1, tok2 = line 
            idx = int(idx)
            if not isstr:
                tok1 = tobytes(tok1)
                tok2 = tobytes(tok2)
            d[(tok1,tok2)] = idx
    else:
        for i in range(n):
            line = f.readline().rstrip("\n").split(" ")
            assert len(line) == 2, f"expected 2 fields: {line} on line {i} of {n}"
            idx, tok = line 
            idx = int(idx)
            if not isstr:
                tok = tobytes(tok)
            d[tok] = idx

    return d


# -----------------------------------------------------------------------------
# the base Tokenizer class

class UniformTokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        # the list of merges, with the value giving the order
        # merge of (b1,b2) will give b1+b2
        self.merges = {} # (bytes,bytes) -> int
        # pre-tokenization pattern
        self.pattern = "" # str
        # extra tokens stored at the end
        # lets require that these are strings rather than bytes
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.inv_special_tokens = {} # int -> str
        self.vocab = {} # bytes -> int
        self.inv_vocab = {} # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'wt') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("uniformbpe v1\n")
            f.write(f"{self.pattern}\n")
            _write_sorted_dict(self.vocab, f, ispair=False, isstr=False)
            _write_sorted_dict(self.special_tokens, f, ispair=False, isstr=True) # these are strings
            _write_sorted_dict(self.merges, f, ispair=True, isstr=False)


    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'rt', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "uniformbpe v1"

            # read the pattern
            self.pattern = f.readline().strip()
            # Compile the loaded pattern
            self.compiled_pattern = re.compile(self.pattern)

            self.vocab = _read_sorted_dict(f, ispair=False, isstr=False)
            self.special_tokens = _read_sorted_dict(f, ispair=False, isstr=True)  # these are strings
            self.merges  = _read_sorted_dict(f, ispair=True, isstr=False)

            # compute the inverses 
            self.inv_special_tokens = { v : k for (k,v) in self.special_tokens.items()}
            self.inv_vocab = { v:k for (k,v) in self.vocab.items()}


if __name__ == "__main__":

    # TODO: move to proper tests
    print('testing')

    tokens=[b'a', b'b', b'c', b'a', b'b']
    pair=(b'a', b'b')
    m, merge_cnt = merge(tokens, pair)
    assert m == [b'ab', b'c', b'ab']
    assert merge_cnt == 2


    tokens = [b'a', b'b', b'c', b'a', b'b'] 

    counts, unique_counts = get_stats(tokens)
    correct =  {(b'a', b'b'): 2, 
                (b'b', b'c'): 1, 
                (b'c', b'a'): 1}
    unique_correct =  {(b'a', b'b'): 1, 
                       (b'b', b'c'): 1, 
                       (b'c', b'a'): 1}
    
    assert counts == correct
    assert unique_counts == unique_correct

    def single_tally(tokens):
        counts = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        return counts

    # what if there are runs of the same token
    tokens = [b'a', b'a', b'a', b'a', b'a', b'b', b'c', b'a', b'b'] 
    print("1:", tokens)
    pair=(b'a', b'a')
    m, merge_cnt = merge(tokens, pair)
    # note we only get non-overlapping counts on b'aa'
    # [b'aa', b'aa', b'a', b'b', b'c', b'a', b'b']  

    # some duplicate a's
    counts, unique_counts = get_stats(tokens)
    print("2:", counts)
    correct = {(b'a', b'a'): 2, (b'a', b'b'): 2, (b'b', b'c'): 1, (b'c', b'a'): 1}
    unique_correct = {(b'a', b'a'): 1, (b'a', b'b'): 1, (b'b', b'c'): 1, (b'c', b'a'): 1}
    assert counts == correct
    assert unique_counts == unique_correct

    # no duplicates
    tokens = [b'a', b'b', b'c', b'd', b'e'] 
    counts, unique_counts = get_stats(tokens)
    print("3:", counts)
    correct = {(b'a', b'b'): 1, (b'b', b'c'): 1, (b'c', b'd'): 1, (b'd', b'e'): 1}
    unique_correct = {(b'a', b'b'): 1, (b'b', b'c'): 1, (b'c', b'd'): 1, (b'd', b'e'): 1}
    assert correct == counts 
    assert unique_counts == unique_correct

    # duplicates in the middle 
    tokens = [b'b', b'c', b'a', b'a', b'a', b'b', b'c'] 
    counts, unique_counts = get_stats(tokens)
    print("4:", counts)
    correct = {(b'b', b'c'): 2, (b'c', b'a'): 1, (b'a', b'a'): 1, (b'a', b'b'): 1}
    unique_correct = {(b'b', b'c'): 1, (b'c', b'a'): 1, (b'a', b'a'): 1, (b'a', b'b'): 1}
    assert correct == counts 
    assert unique_counts == unique_correct

    # duplicates at the end
    tokens = [b'b', b'c', b'd', b'a', b'a', b'a'] 
    counts, unique_counts = get_stats(tokens)
    print("5:", counts)
    correct = {(b'b', b'c'): 1, (b'c', b'd'): 1, (b'd', b'a'): 1, (b'a', b'a'): 1}
    unique_correct = {(b'b', b'c'): 1, (b'c', b'd'): 1, (b'd', b'a'): 1, (b'a', b'a'): 1}
    assert correct == counts 
    assert unique_counts == unique_correct

    # even number of pairs
    tokens = [b'b', b'c', b'd', b'a', b'a', b'a', b'a'] 
    counts, unique_counts = get_stats(tokens)
    print("6:", counts)
    correct =  {(b'b', b'c'): 1, (b'c', b'd'): 1, (b'd', b'a'): 1, (b'a', b'a'): 2}
    unique_correct =  {(b'b', b'c'): 1, (b'c', b'd'): 1, (b'd', b'a'): 1, (b'a', b'a'): 1}
    assert correct == counts 
    assert unique_counts == unique_correct

   # two runs 
    tokens = [b'b', b'a', b'a', b'a', b'b', b'c', b'a', b'a', b'a', b'a', b'a', b'e'] 
    counts, unique_counts = get_stats(tokens)
    print("7:", counts)
    correct = {(b'b', b'a'): 1, (b'a', b'a'): 3, (b'a', b'b'): 1, (b'b', b'c'): 1, (b'c', b'a'): 1, (b'a', b'e'): 1}
    unique_correct = {(b'b', b'a'): 1, (b'a', b'a'): 1, (b'a', b'b'): 1, (b'b', b'c'): 1, (b'c', b'a'): 1, (b'a', b'e'): 1}
    assert correct == counts 
    assert unique_counts == unique_correct