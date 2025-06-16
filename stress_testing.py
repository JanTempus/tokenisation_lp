from datasets import load_dataset
import multi_strings_flow as msf
from collections import defaultdict
import time
import numpy as np
import argparse



data = np.load("strings_with_frequency.npz")
inputStrings=data["inputStrings" ]
inputStringsfrequencies=data["inputStringsfrequencies"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi_strings_flow with variable arguments.")
    parser.add_argument("arg1", type=int, help="First integer argument for CreateInstanceAndSolve")
    #parser.add_argument("arg2", type=int, help="Second integer argument for CreateInstanceAndSolve")
    args = parser.parse_args()

msf.CreateInstanceAndSolve(inputStrings,inputStringsfrequencies,4,args.arg1)