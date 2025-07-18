import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import pickle
from collections import defaultdict
import time

from datastructures import tokenInstance, possibleToken

import helper_functions as hf
#from helper_functions import get_all_nonFree_substrings_upto_len_t, get_tokens_upto_len_t, get_tokens, get_all_free_substrings, get_all_nonFree_substrings

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

def tokenize(edgesList: list[list[tokenInstance]] , 
            edgeListWeight:list[int] , 
            numVerticesList:list[int]):
    
    numStrings = len(edgesList)

    A_rows, A_cols, A_data = [], [], []
   
    BigbVector_parts = []
    BigNonFreewVector_parts = []

    A_row_offset = 0
    A_col_offset = 0

    for i in range(numStrings):
        edges = edgesList[i]
        numEdges = len(edges)
        numVertices = numVerticesList[i]

        # Flow constraints (A) for non-free edges
        for idx, edge in enumerate(edges):
            A_rows.append(edge.start + A_row_offset)
            A_cols.append(idx + A_col_offset)
            A_data.append(1)

            A_rows.append(edge.end + A_row_offset)
            A_cols.append(idx + A_col_offset)
            A_data.append(-1)

        # b vector
        b = np.zeros(numVertices, dtype=int)
        b[0] = 1
        b[numVertices - 1] = -1
        BigbVector_parts.append(b)

        # weights
        wnonFree = np.full(numEdges, edgeListWeight[i])
        BigNonFreewVector_parts.append(wnonFree)

        # Update offsets
        A_row_offset += numVertices
        A_col_offset += numEdges
    # Construct final sparse matrices
    BigAConstraint = sp.coo_matrix((A_data, (A_rows, A_cols)), shape=(A_row_offset, A_col_offset)).tocsr()
   
    
    BigbVector = np.hstack(BigbVector_parts)
    BigNonFreewVector = np.hstack(BigNonFreewVector_parts)  

    f=cp.Variable(A_col_offset,nonneg=True )

    constraints=[BigAConstraint@f==BigbVector]   


    objective=cp.Minimize(BigNonFreewVector.T@f)

    problem = cp.Problem(objective, constraints)
    start = time.time()
    problem.solve(solver=cp.GLOP)
    end=time.time()
    #print(f"Took {end - start:.4f} seconds")

    #print(f"The compression is now {problem.value}")

    flow_values = f.value   
    shortest_paths = []
    offset = 0
    for i in range(numStrings):
        edges = edgesList[i]
        numEdges = len(edges)
        flows = flow_values[offset:offset+numEdges]
        used_edges = [edges[j].token for j in range(numEdges) if flows[j] > 1e-6]  # tolerance for numerical noise
        shortest_paths.append(used_edges)
        offset += numEdges

    return shortest_paths


def create_vocab(inputStringList: list[str],
                    inputStringFreq:list[int],
                    numAllowedTokens:int, 
                    minTokenCount:int=1,  
                    maxTokenLength: int=5, 
                    all_tokens:bool=True ):
    
    numStrings=len(inputStringList)

    edgesList=[]
    tokensList=[]
    freeEdgesList=[]
    numVertices=[]


    if all_tokens:  
        for i in range(numStrings):
            stringLen=len(inputStringList[i])
            edgesList.append(hf.get_all_nonFree_substrings(inputStringList[i]) )
            tokensList.append(hf.get_tokens(inputStringList[i]))
            freeEdgesList.append(hf.get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen+1)
        
        tokens=tokensList[0]
        tokens=list(set([item for sublist in tokensList for item in sublist] ))

    else:
        for i in range(numStrings):
            stringLen=len(inputStringList[i])
            edgesList.append(hf.get_all_nonFree_substrings_upto_len_t(inputStringList[i],maxTokenLength) )
            tokensList.append(hf.get_tokens_upto_len_t(inputStringList[i],maxTokenLength))
            freeEdgesList.append(hf.get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen+1)

       
        tokens=tokensList[0]
        tokens=list(set([item for sublist in tokensList for item in sublist] ))
    


    print("Finished preparing data")

    hf.update_token_instance_counts(tokens,inputStringFreq,edgesList)
    print("Total number of tokens " ,len(tokens))
    tokens_to_keep = [token for token in tokens if token.token_instance_count > minTokenCount]
    print("Total number of tokens kept:", len(tokens_to_keep))

    # Create a set of valid token strings
    keep_set = set(t.token for t in tokens_to_keep)

    # Count edges before filtering
    edges_before = sum(len(sublist) for sublist in edgesList)
    print(f"Number of edges before: {edges_before}")

    # Create a new edgesList that only contains tokens in keep_set
    filtered_edgesList = [
        [token for token in sublist if token.token in keep_set]
        for sublist in edgesList
    ]

    # Count edges after filtering
    edges_after = sum(len(sublist) for sublist in filtered_edgesList)
    print(f"Number of edges after: {edges_after}")

    lpProblem=setup_LP_tokenization(filtered_edgesList,inputStringFreq,tokens_to_keep , freeEdgesList,numVertices)

    numAllowedTokensParam = lpProblem.parameters()[0]
    numAllowedTokensParam.value = numAllowedTokens

    start = time.time()
    lpProblem.solve(solver=cp.PDLP,solver_opts={"eps_optimal_absolute": 1.0e-8})
    end=time.time()
    print(f"The first iteration took {end - start:.4f} seconds")

    lpVariables=lpProblem.variables()
   
    tVar=lpVariables[2].value
    
    chosenTokens=[]
    nonZeroTokenCount=0
    chosenTokensCount=0
    for i in range(len(tokens_to_keep)):
        tokens_to_keep[i].lpValue=tVar[i]
        if tokens_to_keep[i].lpValue>0.0:
            nonZeroTokenCount+=1
        if tokens_to_keep[i].lpValue>0.99:
            chosenTokens.append(tokens_to_keep[i])
            chosenTokensCount+=1

    newEdges,newFreeEdges=hf.extendFreeEdges(edgesList,chosenTokens,freeEdgesList)
    
    chosenTokensStrings=[token.token for token in chosenTokens]

    print(f"We have selected {chosenTokensCount} tokens out of {numAllowedTokens}")
    print( f"The number of non zero tokens is {nonZeroTokenCount}  which is {(nonZeroTokenCount/numAllowedTokens)} percent")


    return chosenTokensStrings

   


    
# inputStrings=["world","hello","hello"]

# create_instance(inputStrings,[1,1,1],5)