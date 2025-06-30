from matrix_construction import TokenInstance, PossibleToken,make_matrix, make_matrix_parallel
from helper_functions_rust import get_all_free_substrings,get_tokens_upto_len_t,get_all_nonFree_substrings_upto_len_t
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import matrix_construction as mc
import time


def setup_and_solve_rust_seq(edgesList: list[list[TokenInstance]] , 
            edgeListWeight:list[int] , 
            tokens: list[PossibleToken], 
            freeEdgesList: list[list[TokenInstance]], 
            numVerticesList:list[int],
            edgeCount:int,
            freeEdgeCount:int):
     
    print("starting with the rust function sequential")
    
    start=time.time()
    a_raw, b_raw, m_raw, b_vec, w_nonfree, w_free = make_matrix(edgesList,edgeListWeight,tokens,freeEdgesList,numVerticesList )
    
    end = time.time()

    print(f"Rust took {end - start:.4f} seconds")
    A = sp.csr_matrix((a_raw[0], a_raw[1], a_raw[2]), shape=(a_raw[3], a_raw[4]))
    B = sp.csr_matrix((b_raw[0], b_raw[1], b_raw[2]), shape=(b_raw[3], b_raw[4]))
    M = sp.csr_matrix((m_raw[0], m_raw[1], m_raw[2]), shape=(m_raw[3], m_raw[4]))
    
    w_nonfree_np = np.array(w_nonfree, dtype=np.int64)
    w_free_np = np.array(w_free, dtype=np.int64)

    
    numTokens=len(tokens)

    tokensCap=np.ones(numTokens,dtype=float)
    
    f=cp.Variable(edgeCount,nonneg=True )
    g=cp.Variable(freeEdgeCount,nonneg=True)
    t=cp.Variable(numTokens,nonneg=True)
    numAllowedTokens = cp.Parameter(nonneg=True)

    constraints=[A@f+ B@g==b_vec,
                 f <= M @ t,
                 cp.sum(t)<=numAllowedTokens,
                 t <=tokensCap]   


    objective=cp.Minimize(w_nonfree_np.T@f +w_free_np.T@g)

    problem = cp.Problem(objective, constraints)

def setup_and_solve_rust_parralell(edgesList: list[list[TokenInstance]] , 
            edgeListWeight:list[int] , 
            tokens: list[PossibleToken], 
            freeEdgesList: list[list[TokenInstance]], 
            numVerticesList:list[int],
            edgeCount:int,
            freeEdgeCount:int):
     
    print("starting with the rust function parallel")

    a_raw, b_raw, m_raw, b_vec, w_nonfree, w_free = make_matrix_parallel(edgesList,edgeListWeight,tokens,freeEdgesList,numVerticesList )

    print("Finished with the rust function parallel")
    A = sp.csr_matrix((a_raw[0], a_raw[1], a_raw[2]), shape=(a_raw[3], a_raw[4]))
    B = sp.csr_matrix((b_raw[0], b_raw[1], b_raw[2]), shape=(b_raw[3], b_raw[4]))
    M = sp.csr_matrix((m_raw[0], m_raw[1], m_raw[2]), shape=(m_raw[3], m_raw[4]))
    
    w_nonfree_np = np.array(w_nonfree, dtype=np.int64)
    w_free_np = np.array(w_free, dtype=np.int64)

    
    numTokens=len(tokens)

    tokensCap=np.ones(numTokens,dtype=float)
    
    f=cp.Variable(edgeCount,nonneg=True )
    g=cp.Variable(freeEdgeCount,nonneg=True)
    t=cp.Variable(numTokens,nonneg=True)
    numAllowedTokens = cp.Parameter(nonneg=True)

    constraints=[A@f+ B@g==b_vec,
                 f <= M @ t,
                 cp.sum(t)<=numAllowedTokens,
                 t <=tokensCap]   


    objective=cp.Minimize(w_nonfree_np.T@f +w_free_np.T@g)

    problem = cp.Problem(objective, constraints)


def CreateInstanceAndSolve(inputStringList: list[str],inputStringFreq:list[int], maxTokenLength: int ):
    
    numStrings=len(inputStringList)
 

    edgesList=[]
    tokensList=[]
    freeEdgesList=[]
    numVertices=[]
    for i in range(numStrings):
        stringLen=len(inputStringList[i])
        edgesList.append(get_all_nonFree_substrings_upto_len_t(inputStringList[i],maxTokenLength) )
        tokensList.append(get_tokens_upto_len_t(inputStringList[i],maxTokenLength))
        freeEdgesList.append(get_all_free_substrings(inputStringList[i]))
        numVertices.append(stringLen+1)

    non_free_edge_count=0
    free_edge_count=0
    for i in range(numStrings):
        non_free_edge_count+=len(edgesList[i])
        free_edge_count+=len(freeEdgesList[i])

    tokens=tokensList[0]

    tokens=list(set([item for sublist in tokensList for item in sublist] ))
    
    print("Finished preparing data from rust")

    lp_problem=setup_and_solve_rust_seq(edgesList,inputStringFreq,tokens,freeEdgesList,numVertices,non_free_edge_count,free_edge_count )
    #lp_problem=setup_and_solve_rust_parralell(edgesList,inputStringFreq,tokens,freeEdgesList,numVertices,non_free_edge_count,free_edge_count )
    # numAllowedTokensParam = lp_problem.parameters()[0]
    # numAllowedTokensParam.value = 2
    # lp_problem.solve(solver=cp.GLOP,verbose=True)



    # return problem



# inputStrings=["world","hello"]

# # CreateInstanceAndSolve(inputStrings,[1,1],5)