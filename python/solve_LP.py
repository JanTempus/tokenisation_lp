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

    for numTokens in range(2, numAllowedTokens):
        numAllowedTokensParam = lpProblem.parameters()[0]
        numAllowedTokensParam.value = numTokens
        lpProblem.solve(solver=cp.GLOP)
        lpVariables=lpProblem.variables()
        
        # fVar=lpVariables[0].value
        # gVar=lpVariables[1].value
        tVar=lpVariables[2].value

        allBoolean=True

        print("Current working on: ", numTokens)
        for num in tVar:
            if abs(num-0)>0.0001 :
                print(num)
            elif abs(num-1.0)>0.0001:
                 print(num)
       
        if allBoolean:  
            print("For ", numTokens, " everything is integral " )
        
        for i in range(len(tokens)):
            tokens[i].lpValue=tVar[i]

        
        length_sorted_tokens=sorted(tokens, key=lambda t: len(t.token), reverse=True)
        sorted_tokens=sorted(tokens, key=lambda t: t.lpValue, reverse=True)
        print(sorted_tokens[0:(2*numAllowedTokens)])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi_strings_flow with variable arguments.")
    parser.add_argument("arg1", type=int, help="First integer argument for CreateInstanceAndSolve")
    #parser.add_argument("arg2", type=int, help="Second integer argument for CreateInstanceAndSolve")
    args = parser.parse_args()

solve(args.arg1)