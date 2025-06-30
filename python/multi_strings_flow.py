import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import resource
import pickle
import json
import time

class tokenInstance:
    token: str
    start: int
    end: int
    lpValue:float

    def __init__(self,token,start,end):
        self.token=token
        self.start=start
        self.end=end
        self.lpValue=float(-1)
    
    def __eq__(self, other):
        if not isinstance(other,possibleToken):
            return False
        return self.token==other.token
    
    def __hash__(self):
        return hash(self.token)

    def __str__(self):
        return f"{self.start,self.end,self.token,self.lpValue}"

    def __repr__(self):
        return self.__str__()
    def to_dict(self):
        return {
            "token": self.token,
            "start": self.start,
            "end": self.end,
            "lp_value": self.lpValue
        }

class possibleToken:
    token:str
    lpValue:float

    def __init__(self,token):
        self.token=token
        self.lpValue=float(-1)

    def __eq__(self, other):
        if not isinstance(other,possibleToken):
            return False
        return self.token==other.token

    def __hash__(self):
        return hash(self.token)
    
    def __str__(self):
        return f"{self.token, self.lpValue}"

    def __repr__(self):
        return self.__str__()
    def to_dict(self):
        return {
            "token": self.token,
            "lp_value": self.lpValue
        }


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

def find_corresponding_token(fixedString: tokenInstance,tokenSet )->tokenInstance:
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



def SolveLPVec(edgesList: list[list[tokenInstance]] , 
            edgeListWeight:list[int] , 
            tokens: list[possibleToken], 
            freeEdgesList: list[list[tokenInstance]], 
            numVerticesList:list[int]):
    
    numStrings=len(edgesList)
    
    if(numStrings != len(freeEdgesList) ):
        raise ValueError

    AMatrices=[]
    BMatrices=[]
    MMatrices=[]
    bVectors=[]
    freewVectors=[]
    nonfreewVectors=[]
    numTokens=len(tokens)


    print("started working on the LP")
    start = time.time()
    edgeCount=0
    freeEdgeCount=0
    for i in range(numStrings):
        edges=edgesList[i]
        freeEdges=freeEdgesList[i]
        tokens=tokens

        numEdges=len(edges)
        edgeCount+=numEdges
        
        numFreeEdges=len(freeEdges)
        freeEdgeCount+=numFreeEdges

        numVertices=numVerticesList[i]

        #Create the flow preservation matrix for the non free edges for the ith string
        A= sp.lil_matrix((numVertices, numEdges))
        for idx, edge in enumerate(edges):
            A[edge.start, idx]=1
            A[edge.end,idx]=-1

        
        #Create the flow preservation matrix for the free edges for the ith string
        B=sp.lil_matrix((numVertices, numFreeEdges))
        for idx, edge in enumerate(freeEdges):
            B[edge.start, idx]=1
            B[edge.end,idx]=-1


        #Create the token preservation matrix for the ith string
        M = sp.lil_matrix((numEdges, numTokens))
        for j, edge in enumerate(edges):
            tokenIndex = find_corresponding_token(edge.token, tokens)
            M[j, tokenIndex] = 1

       
        AMatrices.append(A)
        BMatrices.append(B)
        MMatrices.append(M)


        b=np.zeros(numVertices,dtype=int)
        b[0]=1
        b[numVertices-1]=-1 


        bVectors.append(b)

        wnonFree=np.full(numEdges,edgeListWeight[i])
        wFree=np.full(numFreeEdges,edgeListWeight[i])
        nonfreewVectors.append(wnonFree)
        freewVectors.append(wFree)
    end = time.time()

    print(f"python settin up took {end - start:.4f} seconds")

    BigAConstraint=sp.block_diag(AMatrices)
    BigBConstraint=sp.block_diag(BMatrices)
    BigMConstraint=sp.vstack(MMatrices)
    BigbVector=np.hstack(bVectors)
    BigFreewVector=np.hstack(freewVectors)
    BigNonFreewVector=np.hstack(nonfreewVectors)

    tokensCap=np.ones(numTokens,dtype=float)

    BigAConstraint=BigAConstraint.tocsr()
    BigBConstraint=BigBConstraint.tocsr()
    BigMConstraint=BigMConstraint.tocsr()

    
    f=cp.Variable(edgeCount,nonneg=True )
    g=cp.Variable(freeEdgeCount,nonneg=True)
    t=cp.Variable(numTokens,nonneg=True)
    numAllowedTokens = cp.Parameter(nonneg=True)

    constraints=[BigAConstraint@f+ BigBConstraint@g==BigbVector,
                 f <= BigMConstraint @ t,
                 cp.sum(t)<=numAllowedTokens,
                 t <=tokensCap]   


    objective=cp.Minimize(BigNonFreewVector.T@f +BigFreewVector.T@g)

    problem = cp.Problem(objective, constraints)

    return problem



def SolveLP(edgesList: list[list[tokenInstance]] , 
            edgeListWeight:list[int] , 
            tokens: list[possibleToken], 
            freeEdgesList: list[list[tokenInstance]], 
            numVerticesList:list[int], 
            numAllowedTokens:int)->tuple[list[list[tokenInstance]],list[list[tokenInstance]],list[possibleToken]]:
    
    numStrings=len(edgesList)
    
    if(numStrings != len(freeEdgesList) ):
        raise ValueError

    constraints=[]

    f_sum=0
    g_sum=0
    numTokens=len(tokens)
    t=cp.Variable(numTokens,nonneg=True)

    for i in range(numStrings):
        edges=edgesList[i]
        freeEdges=freeEdgesList[i]
        tokens=tokens

        numEdges=len(edges)
        
        numFreeEdges=len(freeEdges)
        numVertices=numVerticesList[i]

        f=cp.Variable(numEdges,nonneg=True)
        g=cp.Variable(numFreeEdges,nonneg=True)
        
       

        A= sp.lil_matrix((numVertices, numEdges))
        for idx, edge in enumerate(edges):
            A[edge.start, idx]=1
            A[edge.end,idx]=-1

       
        B=sp.lil_matrix((numVertices, numFreeEdges))
        
        
        for idx, edge in enumerate(freeEdges):
            B[edge.start, idx]=1
            B[edge.end,idx]=-1


        A=A.tocsr()
        B=B.tocsr()
        constraints.append(A[0] @ f+B[0]@g == 1)                            
        constraints.append(A[numVertices - 1] @ f +B[numVertices-1]@g== -1)             
        
        constraints.append(A[1:-1] @ f + B[1:-1] @ g == 0)
        
        M = sp.lil_matrix((numEdges, numTokens))
        for j, edge in enumerate(edges):
            tokenIndex = find_corresponding_token(edge.token, tokens)
            M[j, tokenIndex] = 1

        M = M.tocsr()


        constraints.append(f <= M @ t)

        constraints.append(cp.sum(t)==numAllowedTokens )
      
        f_sum+=cp.sum(f)*edgeListWeight[i]
        g_sum+=cp.sum(g)*edgeListWeight[i]

       


    objective=cp.Minimize(f_sum +g_sum)

    problem = cp.Problem(objective, constraints)
    problem.solve()
    #print("Optimal value:", problem.value)


"""
This function takes in the free edges, the non free edges and tokens we include in the free edges. 
tl;dr moves chosen tokens from nonFree to Free edges list depending on whether we have "selected" this edge.
Only moves edges from non free to free.
For every tokenInstance in edgesList it checks whether it is contained in accepted tokens, and if it is appends it to free edges,
otherwise it puts it back in the nonFreeEdgesList.
"""
def extendFreeEdges(edgesList: list[list[tokenInstance]] , 
            Acceptedtokens: list[possibleToken], 
            freeEdgesList: list[list[tokenInstance]]
            )->tuple[list[tokenInstance],list[tokenInstance]]:
    

    numStrings=len(edgesList)

    newFreeEdgestList=[]
    newNonFreeEdgesList=[]
    for i in range(numStrings):
        edges=edgesList[i]
        newFreeEdges=freeEdgesList[i]
        newNonFreeEdges=[]
        for edge in edges:
            if edge in Acceptedtokens:
                newFreeEdges.append(edge)
            else:
                newNonFreeEdges.append(edge)


        newFreeEdgestList.append(newFreeEdges)
        newNonFreeEdgesList.append(newNonFreeEdges)

    return newNonFreeEdgesList, newFreeEdgestList
    
def shortestPath(edgeListWeight:list[int] , 
            edgesList: list[list[tokenInstance]], 
            numVerticesList:list[int])->int:
    numStrings=len(edgesList)
    

    AMatrices=[]
    bVectors=[]
    weightVectors=[]


    edgeCount=0
    for i in range(numStrings):
        edges=edgesList[i]
        
        numEdges=len(edges)
        edgeCount+=numEdges

        numVertices=numVerticesList[i]

        #Create the flow preservation matrix for the non free edges for the ith string
        A= sp.lil_matrix((numVertices, numEdges))
        for idx, edge in enumerate(edges):
            A[edge.start, idx]=1
            A[edge.end,idx]=-1
       
        AMatrices.append(A)
        
        b=np.zeros(numVertices,dtype=int)
        b[0]=1
        b[numVertices-1]=-1 

        bVectors.append(b)

        weights=np.full(numEdges,edgeListWeight[i])
        weightVectors.append(weights)
    

    BigAConstraint=sp.block_diag(AMatrices)
    BigbVector=np.hstack(bVectors)
    BigweightVector=np.hstack(weightVectors)
    BigAConstraint=BigAConstraint.tocsr()

    f=cp.Variable(edgeCount,nonneg=True )

    constraints=[BigAConstraint@f == BigbVector]   


    objective=cp.Minimize(BigweightVector.T@f)

    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=True)
    return problem.value



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

    print("Finished preparing data")
    
    tokens=tokensList[0]

    tokens=list(set([item for sublist in tokensList for item in sublist] ))
    

    # save_lp_data(edgesList,inputStringFreq,tokens,freeEdgesList,numVertices,"lp_data.json" )
    print("Finished preparing tokens")


    lpProblem = SolveLPVec(edgesList,inputStringFreq,tokens,freeEdgesList,numVertices)

   

    # with open("lp_problem.pkl", "wb") as f:
    #     pickle.dump(lpProblem, f)

    # with open("tokens.pkl", "wb") as f:
    #     pickle.dump(tokens, f)
    # numAllowedTokensParam = lpProblem.parameters()[0]
    # numAllowedTokensParam.value = numAllowedTokens
    # lpProblem.solve(solver=cp.GLOP, max_iters=5000, verbose=True)
    # lpVariables=lpProblem.variables()
    
    # # fVar=lpVariables[0].value
    # # gVar=lpVariables[1].value
    # tVar=lpVariables[2].value
    # for i in range(len(tokens)):
    #     tokens[i].lpValue=tVar[i]

    # length_sorted_tokens=sorted(tokens, key=lambda t: len(t.token), reverse=True)
    # sorted_tokens=sorted(tokens, key=lambda t: t.lpValue, reverse=True)
    # print(length_sorted_tokens[0])
    # print(sorted_tokens[0:numAllowedTokens+2])

    



#inputStrings=["One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp.", " Lily wanted to share the needle with her mom, so she could sew a button on her shirt."]
# inputStrings=["world","hello"]

# CreateInstanceAndSolve(inputStrings,[1,1],5)