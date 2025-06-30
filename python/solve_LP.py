import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import resource
import pickle
from multi_strings_flow import possibleToken
import argparse


def solve(numAllowedTokens:int):

    with open("lp_problem.pkl", "rb") as f:
        lpProblem = pickle.load(f)

    with open("tokens.pkl", "rb") as g:
        tokens=pickle.load(g)


    print("We we now have the lp and tokens")
  
    numAllowedTokensParam = lpProblem.parameters()[0]
    numAllowedTokensParam.value = numAllowedTokens
    lpProblem.solve(solver=cp.GLOP,verbose=True)
    lpVariables=lpProblem.variables()
   

    
    # fVar=lpVariables[0].value
    # gVar=lpVariables[1].value
    tVar=lpVariables[2].value

    # if allBoolean:  
    #     print("For ", numTokens, " everything is integral " )
    
    for i in range(len(tokens)):
        tokens[i].lpValue=tVar[i]


    sorted_tokens=sorted(tokens, key=lambda t: t.lpValue, reverse=True)
    print(sorted_tokens[0:(2*numAllowedTokens)])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi_strings_flow with variable arguments.")
    parser.add_argument("arg1", type=int, help="First integer argument for CreateInstanceAndSolve")
    #parser.add_argument("arg2", type=int, help="Second integer argument for CreateInstanceAndSolve")
    args = parser.parse_args()

solve(args.arg1)