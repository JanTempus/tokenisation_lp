import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import resource


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




def SolveLPVectorized(edgesList: list[list[tokenInstance]] , 
            edgeListWeight:list[int] , 
            tokens: list[tokenInstance], 
            freeEdgesList: list[list[tokenInstance]], 
            totalVertices:int, 
            numAllowedTokens:int)->tuple[list[list[tokenInstance]],list[list[tokenInstance]],list[possibleToken]]:

    numStrings=len(edgesList)
    
    if(numStrings != len(freeEdgesList) ):
        raise ValueError
    
    totalNumEdges=sum([len(edges) for edges in edgesList ] )
    totalNumFreeEdges=sum(len(edges) for edges in freeEdgesList )
    A= sp.lil_matrix((totalVertices, totalNumEdges))
    B= sp.lil_matrix((totalVertices,totalNumFreeEdges))




def SolveLPVec(edgesList: list[list[tokenInstance]] , 
            edgeListWeight:list[int] , 
            tokens: list[tokenInstance], 
            freeEdgesList: list[list[tokenInstance]], 
            numVerticesList:list[int], 
            numAllowedTokens:int)->tuple[list[list[tokenInstance]],list[list[tokenInstance]],list[possibleToken]]:
    
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


        #Create the demand vector for the ith string
        b=np.zeros(numVertices,dtype=int)
        b[0]=1
        b[numVertices-1]=-1 # print(f"{BigwVector.shape = }" )
    # print(f"{BigbVector.shape = }" )
    # print(f"{BigAConstraint.shape= }")
    # print(f"{BigBConstraint.shape= }")
    # print(f"{BigMConstraint.shape= }")

        bVectors.append(b)

        wnonFree=np.full(numEdges,edgeListWeight[i])
        wFree=np.full(numFreeEdges,edgeListWeight[i])
        nonfreewVectors.append(wnonFree)
        freewVectors.append(wFree)
    

    BigAConstraint=sp.block_diag(AMatrices)
    BigBConstraint=sp.block_diag(BMatrices)
    BigMConstraint=sp.vstack(MMatrices)
    BigbVector=np.hstack(bVectors)
    BigFreewVector=np.hstack(freewVectors)
    BigNonFreewVector=np.hstack(nonfreewVectors)

    BigAConstraint=BigAConstraint.tocsr()
    BigBConstraint=BigBConstraint.tocsr()
    BigMConstraint=BigMConstraint.tocsr()


    f=cp.Variable(edgeCount,nonneg=True )
    g=cp.Variable(freeEdgeCount,nonneg=True)
    t=cp.Variable(numTokens,nonneg=True)

    constraints=[BigAConstraint@f+ BigBConstraint@g==BigbVector,
                 f <= BigMConstraint @ t,
                 cp.sum(t)==numAllowedTokens  ]


    objective=cp.Minimize(BigNonFreewVector.T@f +BigFreewVector.T@g)

    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=True)
    #print("Optimal value:", problem.value)

    usage = resource.getrusage(resource.RUSAGE_SELF)

    with open("solve_log_vectorized.txt", "a") as f:
        solve_time = problem.solver_stats.solve_time
        f.write("Max tokens 5, num Allwed tokens 5 \n" )
        f.write("Corpus size: \n" )
        f.write(f"Solve time: {solve_time:.4f} seconds\n" if solve_time is not None else "Solve time: N/A\n")
        f.write(f"Peak memory usage: {usage.ru_maxrss / 1024:.2f} MB\n")
        f.write("-" * 40 + "\n")  # separator




def SolveLP(edgesList: list[list[tokenInstance]] , 
            edgeListWeight:list[int] , 
            tokens: list[tokenInstance], 
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





def CreateInstanceAndSolve(inputStringList: list[str],inputStringFreq:list[int], maxTokenLength: int, numAllowedTokens:int ):
    
    numStrings=len(inputStringList)
 

    edgesList=[]
    tokensList=[]
    freeEdgesList=[]
    numVertices=[]
    for i in range(numStrings):

        stringLen=len(inputStringList[i])

        maxTokenLength=min(stringLen,maxTokenLength)
        edgesList.append(get_all_nonFree_substrings_upto_len_t(inputStringList[i],maxTokenLength) )
        tokensList.append(get_tokens_upto_len_t(inputStringList[i],maxTokenLength))
        freeEdgesList.append(get_all_free_substrings(inputStringList[i]))
        numVertices.append(stringLen+1)

    
    tokens=tokensList[0]
    for i in range(1,numStrings):
        tokens=list(set(tokens+tokensList[i] ))
    SolveLPVec(edgesList,inputStringFreq,tokens,freeEdgesList,numVertices, numAllowedTokens)
    # SolveLP(edgesList,inputStringFreq,tokens,freeEdgesList,numVertices, numAllowedTokens)

# inputStrings=["One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp.", " Lily wanted to share the needle with her mom, so she could sew a button on her shirt."]
# inputStrings=["hello", "world"]

# CreateInstanceAndSolve(inputStrings,[1,1],2,3)