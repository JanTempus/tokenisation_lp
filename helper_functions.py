from datastructures import tokenInstance, possibleToken
from collections import defaultdict
import json,pickle

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


def print_top_tokens_by_instance_count(tokens: list[tokenInstance], top_n: int = 10):
    """
    Prints the top N tokens with the highest token_instance_count.

    Args:
        tokens (list[possibleToken]): List of token objects with updated counts.
        top_n (int): Number of top tokens to display.
    """
    # Sort tokens by token_instance_count descending
    sorted_tokens = sorted(tokens, key=lambda t: t.token_instance_count, reverse=True)

    print(f"Top {top_n} tokens by instance count:")
    for token in sorted_tokens[:top_n]:
        print(f"{token.token}: {token.token_instance_count}")


def bucket_token_instance_counts(tokens: list[possibleToken], bucket_size: int = 1000) -> dict:
    """
    Groups tokens into buckets based on token_instance_count and counts how many tokens fall into each bucket.

    Args:
        tokens (list[possibleToken]): List of token objects with token_instance_count.
        bucket_size (int): Size of each bucket (default is 100).

    Returns:
        dict: Mapping from bucket_start -> count of tokens in that bucket.
              For example, 200 → number of tokens with count in [200, 299]
    """
    bucket_counts = defaultdict(int)

    for token in tokens:
        bucket_start = (token.token_instance_count // bucket_size) * bucket_size
        bucket_counts[bucket_start] += 1

    # Sort the result by bucket start
    return dict(sorted(bucket_counts.items()))


def save_bucket_counts_pickle(bucket_counts: dict, filename: str):
    """
    Saves bucket counts to a pickle file for efficient storage.

    Args:
        bucket_counts (dict): Dictionary of bucket_start → token count.
        filename (str): Output .pkl filename.
    """
    with open(filename, 'wb') as f:
        pickle.dump(bucket_counts, f)


def extendFreeEdges(
    edgesList: list[list['tokenInstance']], 
    Acceptedtokens: list['possibleToken'], 
    freeEdgesList: list[list['tokenInstance']]
) -> tuple[list[list['tokenInstance']], list[list['tokenInstance']]]:
    """
    Moves accepted tokens from edgesList into freeEdgesList.
    
    Args:
        edgesList: List of lists of tokenInstance objects (non-free edges).
        Acceptedtokens: List of possibleToken objects to be moved to free list.
        freeEdgesList: Existing list of lists of tokenInstance objects (free edges).
        
    Returns:
        Tuple of two new lists:
            - new_edgesList: with tokenInstances not in Acceptedtokens
            - new_freeEdgesList: with tokenInstances moved to free list
    """
    # Convert accepted tokens to a set of strings for fast lookup
    accepted_token_set = set(t.token for t in Acceptedtokens)

    new_edgesList = []
    new_freeEdgesList = []

    for edge_row, free_row in zip(edgesList, freeEdgesList):
        new_edge_row = []
        new_free_row = list(free_row)  # copy existing free edges
        
        for token_instance in edge_row:
            if token_instance.token in accepted_token_set:
                new_free_row.append(token_instance)
            else:
                new_edge_row.append(token_instance)

        new_edgesList.append(new_edge_row)
        new_freeEdgesList.append(new_free_row)

    return new_edgesList, new_freeEdgesList



def update_token_instance_counts(tokens: list[tokenInstance],stringFreq:list[int], tokensList: list[list[possibleToken]]):
    """
    Updates the `token_instance_count` field for each `possibleToken` in `tokens`,
    based on how many times it appears in `tokensList`.

    Args:
        tokens (list[possibleToken]): Unique list of token objects.
        tokensList (list[list[possibleToken]]): Nested lists of token objects per string.
    """
    # Count occurrences of each token string
    freq_map = defaultdict(int)
    numStrings=len(tokensList)
    for i in range(numStrings):
        for token in tokensList[i]:
            freq_map[token.token] += stringFreq[i]

    # Update the count in each unique possibleToken
    for token in tokens:
        token.token_instance_count = freq_map[token.token]



def get_token_instance_count(token_string: str, tokens: list[possibleToken]) -> int:
    """
    Returns the token_instance_count for a given token string from the list of token instances.

    Args:
        token_string (str): The token string to search for.
        tokens (list[tokenInstance]): List of tokenInstance objects.

    Returns:
        int: The instance count for the token, or 0 if not found.
    """
    print("Hello")
    for token in tokens:
        if token.token == token_string:
            return token.token_instance_count
    return -1  # Token not found