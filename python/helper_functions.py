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


def save_lp_data(
    edgesList,
    edgeListWeight,
    tokens,
    freeEdgesList,
    numVerticesList,
    file_path
):
    data = {
        "edges_list": [[t.to_dict() for t in lst] for lst in edgesList],
        "edge_list_weight": edgeListWeight,
        "tokens": [t.to_dict() for t in tokens],
        "free_edges_list": [[t.to_dict() for t in lst] for lst in freeEdgesList],
        "num_vertices_list": numVerticesList
    }

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
