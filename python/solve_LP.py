import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import resource
import pickle
from multi_strings_flow import possibleToken
import argparse
import matplotlib.pyplot as plt
from collections import Counter


def save_lp_value_plot(tokens, filename, title):
    lp_values = [t.lpValue for t in tokens]
    counts = Counter(lp_values)
    values = sorted(counts.keys())
    frequencies = [counts[v] for v in values]

    plt.figure(figsize=(6, 3))
    plt.bar(values, frequencies, width=0.1, color='lightgreen', edgecolor='black')
    plt.yscale('log')  # log-scale y-axis
    plt.xlabel("lpValue", fontsize=10)
    plt.ylabel("Log Frequency", fontsize=10)
    plt.title(title, fontsize=12)
    plt.yticks(fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout(pad=1.0)

    # 🔽 Save the plot to a file
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


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
        if tokens[i].lpValue<1.0 and tokens[i].lpValue>0.0:
            print(tokens[i].token)


    #sorted_tokens=sorted(tokens, key=lambda t: t.lpValue, reverse=True)
    # filename="lp_value_distribution"+str(numAllowedTokens)+".png"
    # title="Log-Scaled LP Value Frequency Plot for "+str(numAllowedTokens)
    # save_lp_value_plot(tokens,filename,title)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi_strings_flow with variable arguments.")
    parser.add_argument("arg1", type=int, help="First integer argument for CreateInstanceAndSolve")
    #parser.add_argument("arg2", type=int, help="Second integer argument for CreateInstanceAndSolve")
    args = parser.parse_args()

solve(args.arg1)