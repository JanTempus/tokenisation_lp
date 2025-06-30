from matrix_construction import TokenInstance, PossibleToken
import json

def get_all_nonFree_substrings_upto_len_t(inputString: str, maxTokenLength: int) -> list[TokenInstance]:
    substrings = []
    n = len(inputString)
    maxTokenLength=min(n,maxTokenLength)
    for length in range(2, maxTokenLength + 1):
        for i in range(n - length + 1):
            substrings.append(TokenInstance(i, i+length, inputString[i:i+length],0.0) )
    return substrings

def get_tokens_upto_len_t(inputString: str, maxTokenLength: int) -> list[PossibleToken]:
    substrings = []
    n = len(inputString)
    maxTokenLength=min(n,maxTokenLength)
    for length in range(2, maxTokenLength + 1):
        for i in range(n - length + 1):
            substrings.append(PossibleToken(inputString[i:i+length],0.0) )
    return list(set(substrings))

def get_all_free_substrings(inputString: str) -> list[TokenInstance]:
    substrings = []
    for i in range(len(inputString) ):
        substrings.append(TokenInstance( i, i+1,inputString[i:i+1],0.0) )
    return substrings

def find_corresponding_token(fixedString: TokenInstance,tokenSet )->TokenInstance:
    tokenIndex=-1

    for i in range(len(tokenSet)):
        if(tokenSet[i].token==fixedString):
            tokenIndex =i
            break

    if(tokenIndex==-1):
        raise ValueError("Corresponding token not in set. This not good" )
        
    return tokenIndex
