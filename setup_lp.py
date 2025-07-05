import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import pickle
from collections import defaultdict

from datastructures import tokenInstance, possibleToken
from helper_functions import get_all_nonFree_substrings_upto_len_t, get_tokens_upto_len_t, get_tokens, get_all_free_substrings, get_all_nonFree_substrings

def setup_LP_tokenization(edgesList: list[list[tokenInstance]] , 
            edgeListWeight:list[int] , 
            tokens: list[possibleToken], 
            freeEdgesList: list[list[tokenInstance]], 
            numVerticesList:list[int]):
    
    numStrings = len(edgesList)
    if numStrings != len(freeEdgesList):
        raise ValueError

    numTokens = len(tokens)
    token_index_map = {t.token: i for i, t in enumerate(tokens)}

  

    # Data holders for constructing big sparse matrices in COO format
    A_rows, A_cols, A_data = [], [], []
    B_rows, B_cols, B_data = [], [], []
    M_rows, M_cols, M_data = [], [], []

    BigbVector_parts = []
    BigFreewVector_parts = []
    BigNonFreewVector_parts = []

    A_row_offset = 0
    B_row_offset = 0
    M_row_offset = 0
    A_col_offset = 0
    B_col_offset = 0

    for i in range(numStrings):
        edges = edgesList[i]
        freeEdges = freeEdgesList[i]
        numEdges = len(edges)
        numFreeEdges = len(freeEdges)
        numVertices = numVerticesList[i]

        # Flow constraints (A) for non-free edges
        for idx, edge in enumerate(edges):
            A_rows.append(edge.start + A_row_offset)
            A_cols.append(idx + A_col_offset)
            A_data.append(1)

            A_rows.append(edge.end + A_row_offset)
            A_cols.append(idx + A_col_offset)
            A_data.append(-1)

        # Flow constraints (B) for free edges
        for idx, edge in enumerate(freeEdges):
            B_rows.append(edge.start + B_row_offset)
            B_cols.append(idx + B_col_offset)
            B_data.append(1)

            B_rows.append(edge.end + B_row_offset)
            B_cols.append(idx + B_col_offset)
            B_data.append(-1)

        # Token preservation matrix (M)
        for j, edge in enumerate(edges):
            tokenIndex = token_index_map[edge.token]
            M_rows.append(j + M_row_offset)
            M_cols.append(tokenIndex)
            M_data.append(1)

        # b vector
        b = np.zeros(numVertices, dtype=int)
        b[0] = 1
        b[numVertices - 1] = -1
        BigbVector_parts.append(b)

        # weights
        wnonFree = np.full(numEdges, edgeListWeight[i])
        wFree = np.full(numFreeEdges, edgeListWeight[i])
        BigNonFreewVector_parts.append(wnonFree)
        BigFreewVector_parts.append(wFree)

        # Update offsets
        A_row_offset += numVertices
        B_row_offset += numVertices
        A_col_offset += numEdges
        B_col_offset += numFreeEdges
        M_row_offset += numEdges

    # Construct final sparse matrices
    BigAConstraint = sp.coo_matrix((A_data, (A_rows, A_cols)), shape=(A_row_offset, A_col_offset)).tocsr()
    BigBConstraint = sp.coo_matrix((B_data, (B_rows, B_cols)), shape=(B_row_offset, B_col_offset)).tocsr()
    BigMConstraint = sp.coo_matrix((M_data, (M_rows, M_cols)), shape=(M_row_offset, numTokens)).tocsr()
    
    BigbVector = np.hstack(BigbVector_parts)
    BigFreewVector = np.hstack(BigFreewVector_parts)
    BigNonFreewVector = np.hstack(BigNonFreewVector_parts)
    tokensCap = np.ones(numTokens, dtype=float)

    
  

    f=cp.Variable(A_col_offset,nonneg=True )
    g=cp.Variable(B_col_offset,nonneg=True)
    t=cp.Variable(numTokens,nonneg=True)
    numAllowedTokens = cp.Parameter(nonneg=True)

    constraints=[BigAConstraint@f+ BigBConstraint@g==BigbVector,
                 f <= BigMConstraint @ t,
                 cp.sum(t)<=numAllowedTokens,
                 t <=tokensCap]   


    objective=cp.Minimize(BigNonFreewVector.T@f +BigFreewVector.T@g)

    problem = cp.Problem(objective, constraints)

    return problem


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

def save_bucket_counts_pickle(bucket_counts: dict, filename: str):
    """
    Saves bucket counts to a pickle file for efficient storage.

    Args:
        bucket_counts (dict): Dictionary of bucket_start → token count.
        filename (str): Output .pkl filename.
    """
    with open(filename, 'wb') as f:
        pickle.dump(bucket_counts, f)


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


def create_instance(inputStringList: list[str],inputStringFreq:list[int], maxTokenLength: int ):
    
    numStrings=len(inputStringList)
 
    all_tokens=True

    edgesList=[]
    tokensList=[]
    freeEdgesList=[]
    numVertices=[]


    if all_tokens:  
        for i in range(numStrings):
            stringLen=len(inputStringList[i])
            edgesList.append(get_all_nonFree_substrings(inputStringList[i]) )
            tokensList.append(get_tokens(inputStringList[i]))
            freeEdgesList.append(get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen+1)

        print("Finished preparing data")
        
        tokens=tokensList[0]

        tokens=list(set([item for sublist in tokensList for item in sublist] ))
       

    else:
        for i in range(numStrings):
            stringLen=len(inputStringList[i])
            edgesList.append(get_all_nonFree_substrings_upto_len_t(inputStringList[i],maxTokenLength) )
            tokensList.append(get_tokens_upto_len_t(inputStringList[i],maxTokenLength))
            freeEdgesList.append(get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen+1)

        print("Finished preparing data")
        
        tokens=tokensList[0]

        tokens=list(set([item for sublist in tokensList for item in sublist] ))
    update_token_instance_counts(tokens,inputStringFreq,edgesList)
    print(len(tokens))
    # bucketed_counts = bucket_token_instance_counts(tokens)

    # for bucket_start, count in bucketed_counts.items():
    #     print(f"{bucket_start}-{bucket_start + 999}: {count} tokens")

    #save_bucket_counts_pickle(bucketed_counts, "bucketed_counts.pkl")
    # k = 20

    # tokens_to_remove =[token for token in tokens if token.token_instance_count <= k]
    # remove_set = set(t.token for t in tokens_to_remove)
    # edges_before=sum(len(sublist) for sublist in edgesList)
    # print(f"number of edges before: {edges_before}  ")
    # for sublist_idx, sublist in enumerate(edgesList):
    #     # Filter sublist to only keep tokens NOT in remove_set
    #     edgesList[sublist_idx] = [token for token in sublist if token.token not in remove_set]

    # edges_after=sum(len(sublist) for sublist in edgesList)

    # print(f"number of edges after: {edges_after}")
   
    
# inputStrings=["world","hello","hello"]

# create_instance(inputStrings,[1,1,1],5)