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

    numAllowedTokensParam = lpProblem.parameters()[0]
    numAllowedTokensParam.value = numAllowedTokens
    lpProblem.solve(solver=cp.GLOP, max_iters=5000, verbose=True)
    lpVariables=lpProblem.variables()
    
    # fVar=lpVariables[0].value
    # gVar=lpVariables[1].value
    tVar=lpVariables[2].value
    for i in range(len(tokens)):
        tokens[i].lpValue=tVar[i]

    length_sorted_tokens=sorted(tokens, key=lambda t: len(t.token), reverse=True)
    sorted_tokens=sorted(tokens, key=lambda t: t.lpValue, reverse=True)
    print(length_sorted_tokens[0])
    print(sorted_tokens[0:numAllowedTokens+2])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi_strings_flow with variable arguments.")
    parser.add_argument("arg1", type=int, help="First integer argument for CreateInstanceAndSolve")
    #parser.add_argument("arg2", type=int, help="Second integer argument for CreateInstanceAndSolve")
    args = parser.parse_args()

solve(args.arg1)