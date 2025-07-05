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
