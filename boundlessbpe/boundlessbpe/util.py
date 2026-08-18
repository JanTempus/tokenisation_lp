from pathlib import Path
import os
import random
from typing import TextIO, Optional

# do the same encoding as Huggingface bytelevel pretokenization
# see byte_level.rs for the original

# fn bytes_char() -> HashMap<u8, char> {
#     let mut bs: Vec<u8> = vec![];
#     bs.extend(b'!'..=b'~');
#     bs.extend(b'\xA1'..=b'\xAC');
#     bs.extend(b'\xAE'..=b'\xFF');

#     let mut cs: Vec<u32> = bs.iter().map(|i| *i as u32).collect();
#     let mut n = 0;

#     for b in 0..=255u8 {
#         if !bs.contains(&b) {
#             bs.push(b);
#             cs.push(u32::pow(2, 8) + n);
#             n += 1;
#         }
#     }

#     bs.into_iter()
#         .zip(cs)
#         .map(|(f, t)| (f, unsafe { std::char::from_u32_unchecked(t) }))
#         .collect()
# }

def bytes_char():
    bs = []
    bs.extend(range(ord('!'), ord('~') + 1))
    bs.extend(range(0xA1, 0xAC + 1))
    bs.extend(range(0xAE, 0xFF + 1))

    # these map to the same character
    cs = [b for b in bs]
    n = 0

    # which are invalid chars and have to use a mapping
    added = []

    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            added.append((bytes([b]), chr(2 ** 8 + n)))
            n += 1

    result = {bytes([f]): chr(t) for f, t in zip(bs, cs)}

    return result, added

byte_map, added = bytes_char()

inv_byte_map = { v : k for k, v in byte_map.items() }

def tobytes(s : str) -> bytes:
    return b"".join([inv_byte_map[c] for c in s])

# encode a bytestring
def frombytes(bs : bytes) -> str:
        return "".join([byte_map[bytes([b])] for b in bs])

# convert from a hex string to bytes,
# like in a .vocab file
def fromhex(hex : str) -> bytes:
    return bytes.fromhex(hex)

# given a bytes object b, get a hex encoded string
def tohex(b : bytes) -> str:
    return b.hex()

def make_dir_if_not_exists(directory_path : str) -> None:

    # Create a Path object for the directory
    directory = Path(directory_path)

    # Check if the directory exists, and if not, create it
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

# TODO: update these

# dump the vocab to a file, encoded as characters here
# no special tokens are added
# are saved in same order by index, so should preserve order
def write_vocab(vocab : dict[int, bytes], 
                filename : str):
    vocab_size = len(vocab)

    # write these in increasing index order
    # so same as any previous order
    byindex = sorted([(idx,token) for token,idx in vocab.items()])

    with open(filename, 'w') as f:
        for _, token in byindex:
            f.write(token.hex() + '\n')

# read our hex formatted vocab file
# return a list of bytes objects
# input file has one vocab word per line,
# each hex encoded
def load_vocab(vocab_filepath : str):

    if not os.path.exists(vocab_filepath):
        raise FileNotFoundError(f'Missing vocab file: {vocab_filepath}')

    with open(vocab_filepath) as vocab_file:
        # fromhex ignores whitespace from \n at end
        initial_vocab = [bytes.fromhex(token) for token in vocab_file.readlines()]

    return initial_vocab


def fix_random_seed(random_seed : int) -> None:
    random.seed(random_seed)

def verify_all_bytes(vocab : dict[int, bytes]) -> None:
    for i in range(256):
        b = bytes([i])
        if b not in vocab:
            print("missing byte", b)
        assert b in vocab


# are the dicts equal?
# if not print some diagnostics before dying
def verify_dicts(d1 : dict, d2 : dict) -> None:

    if d1 != d2:
        print(len(d1), len(d2))
        joint_keys = set(d1.keys()) | set(d2.keys())
        for tok in joint_keys:
            cnt = d1.get(tok, None)
            sc = d2.get(tok, None)
            if sc != cnt:
                print("verify_dicts:", tok, cnt, sc)
        assert False


def is_all_digits(byte_data : bytes) -> bool:
    return all(b in b'0123456789' for b in byte_data)

# allow ' _ space and letters
def can_supermerge(byte_data : bytes) -> bool:
    return all(b in b" _'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for b in byte_data)

def frombytespair(pair):
    left, right = pair
    return (frombytes(left), frombytes(right))



# write the dict, first the number of them, then each one 
# the keys are either bytes, or a pair (bytes,bytes), depending 
# on ispair
# this is now backwards as they values aren't always going to be unique
# but the indices are 
def _write_sorted_dict_intkey(d : dict, f : TextIO, ispair : bool, isstr : bool) -> None:
        
    # write the size, so we don't need to care about the indices being continuous
    f.write(f"{len(d)}\n")
    sortedd = [(idx, val) for idx, val  in d.items()]
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
def _read_sorted_dict_intkey(f : TextIO, ispair : bool, isstr : bool) -> dict:
        
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
            d[idx] = (tok1,tok2)
    else:
        for i in range(n):
            line = f.readline().rstrip("\n").split(" ")
            assert len(line) == 2, f"expected 2 fields: {line} on line {i} of {n}"
            idx, tok = line 
            idx = int(idx)
            if not isstr:
                tok = tobytes(tok)
            d[idx] = tok

    return d


# find all occurences of bad_token in the list, and replace with the individual bytes
# retuns the new tokens, and the number of deletions (can be 0)
# replacement_pair should combine to form bad_token 

# TODO: now will need to track initial words in superwords

def blow_up(lst : list[bytes], bad_token : bytes, parts : list[bytes]):

    new_tokens = []
    deletions = 0

    for tok in lst:
        if tok == bad_token:
            # either single bytes or a pair of bytes
            new_tokens.extend(parts) 
            deletions += 1
        else:
            new_tokens.append(tok)
    return new_tokens, deletions

# delete the token from out list
# asserts bad_token was in the list at least once
# since otherwise we should have skipped tokens
def delete(tokens : list[bytes], bad_token : bytes):
    before = len(tokens)
    # delete all occurences
    tokens = [t for t in tokens if t != bad_token]
    # should be in here
    if len(tokens) == before:
        print("debug 1:", frombytes(bad_token), before)

    assert len(tokens) < before
    return tokens