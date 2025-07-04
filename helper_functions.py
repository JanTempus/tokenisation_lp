from datastructures import tokenInstance, possibleToken
import json

def get_all_nonFree_substrings_upto_len_t(inputString: str, maxTokenLength: int) -> list[tokenInstance]:
    substrings = []
    n = len(inputString)
    maxTokenLength=min(n,maxTokenLength)
    for length in range(2, maxTokenLength + 1):
        for i in range(n - length + 1):
            substrings.append(tokenInstance(inputString[i:i+length], i, i+length) )
    return substrings

def get_all_nonFree_substrings(inputString: str) ->list[tokenInstance]:
    substrings = []
    maxTokenLength=len(inputString)
    for length in range(2, maxTokenLength + 1):
        for i in range(maxTokenLength - length + 1):
            substrings.append(tokenInstance(inputString[i:i+length], i, i+length) )
    return substrings


def get_tokens(inputString: str) -> list[possibleToken]:
    substrings = []
    maxTokenLength=len(inputString)
    for length in range(2, maxTokenLength + 1):
        for i in range(maxTokenLength - length + 1):
            substrings.append(possibleToken(inputString[i:i+length]) )
    return list(set(substrings))


def get_tokens_upto_len_t(inputString: str, maxTokenLength: int) -> list[possibleToken]:
    substrings = []
    n = len(inputString)
    maxTokenLength=min(n,maxTokenLength)
    for length in range(2, maxTokenLength + 1):
        for i in range(n - length + 1):
            substrings.append(possibleToken(inputString[i:i+length]) )
    return list(set(substrings))

def get_all_free_substrings(inputString: str) -> list[tokenInstance]:
    substrings = []
    for i in range(len(inputString) ):
        substrings.append(tokenInstance(inputString[i:i+1], i, i+1) )
    return substrings

def find_corresponding_token(fixedString,tokenSet )->tokenInstance:
    tokenIndex=-1

    for i in range(len(tokenSet)):
        if(tokenSet[i].token==fixedString):
            tokenIndex =i
            break

    if(tokenIndex==-1):
        raise ValueError("Corresponding token not in set. This not good" )
        
    return tokenIndex


def count_tokens_with_max_frequency(tokens: list[possibleToken], k: int) -> int:
    """
    Returns the number of tokens in `tokens` with `token_instance_count` ≤ k.

    Args:
        tokens (list[possibleToken]): List of token objects.
        k (int): Maximum frequency threshold.

    Returns:
        int: Number of tokens with token_instance_count ≤ k.
    """
    return sum(1 for token in tokens if token.token_instance_count <= k)