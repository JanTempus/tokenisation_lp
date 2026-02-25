import cvxpy as cp
import numpy as np
import scipy.sparse as sp
import pickle
from collections import defaultdict
import time
from numpy.typing import NDArray
import random
import csv
import psutil
import os
import threading
import matplotlib.pyplot as plt
#import cudf
#import cugraph

from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_METHOD,
    CUOPT_PDLP_SOLVER_MODE,
)

from cuopt.linear_programming.solver_settings import (
    SolverSettings,
    SolverMethod,
    PDLPSolverMode,
)


from lp_tokenizer.datastructures import tokenInstance, possibleToken
import lp_tokenizer.helper_functions as hf


def prepare_vocab_lp_data(inputStringList: list[str],
                          inputStringFreq: list[int],
                          minTokenCount: int = 1,
                          maxTokenLength: int = 5,
                          all_tokens: bool = True):
    numStrings = len(inputStringList)

    edgesList = []
    tokensList = []
    freeEdgesList = []
    numVertices = []

    if all_tokens:
        for i in range(numStrings):
            stringLen = len(inputStringList[i])
            edgesList.append(hf.get_all_nonFree_substrings(inputStringList[i]))
            tokensList.append(hf.get_tokens(inputStringList[i]))
            freeEdgesList.append(hf.get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen + 1)
    else:
        for i in range(numStrings):
            stringLen = len(inputStringList[i])
            edgesList.append(hf.get_all_nonFree_substrings_upto_len_t(inputStringList[i], maxTokenLength))
            tokensList.append(hf.get_tokens_upto_len_t(inputStringList[i], maxTokenLength))
            freeEdgesList.append(hf.get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen + 1)

    tokens = list(set([item for sublist in tokensList for item in sublist]))
    hf.update_token_instance_counts(tokens, inputStringFreq, edgesList)
    tokens_to_keep = [token for token in tokens if token.token_instance_count > minTokenCount]
    keep_set = set(t.token for t in tokens_to_keep)

    filtered_edgesList = [
        [token for token in sublist if token.token in keep_set]
        for sublist in edgesList
    ]

    return filtered_edgesList, freeEdgesList, numVertices, tokens_to_keep


def build_lp_blocks(edgesList: list[list[tokenInstance]],
                    edgeListWeight: list[int],
                    tokens: list[possibleToken],
                    freeEdgesList: list[list[tokenInstance]],
                    numVerticesList: list[int]):
    numStrings = len(edgesList)
    if numStrings != len(freeEdgesList):
        raise ValueError("edgesList and freeEdgesList must have the same length.")
    if numStrings != len(edgeListWeight):
        raise ValueError("edgeListWeight must have one entry per string.")
    if numStrings != len(numVerticesList):
        raise ValueError("numVerticesList must have one entry per string.")

    numTokens = len(tokens)
    token_index_map = {t.token: i for i, t in enumerate(tokens)}

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

        for idx, edge in enumerate(edges):
            A_rows.append(edge.start + A_row_offset)
            A_cols.append(idx + A_col_offset)
            A_data.append(1)

            A_rows.append(edge.end + A_row_offset)
            A_cols.append(idx + A_col_offset)
            A_data.append(-1)

        for idx, edge in enumerate(freeEdges):
            B_rows.append(edge.start + B_row_offset)
            B_cols.append(idx + B_col_offset)
            B_data.append(1)

            B_rows.append(edge.end + B_row_offset)
            B_cols.append(idx + B_col_offset)
            B_data.append(-1)

        for j, edge in enumerate(edges):
            tokenIndex = token_index_map[edge.token]
            M_rows.append(j + M_row_offset)
            M_cols.append(tokenIndex)
            M_data.append(1)

        b = np.zeros(numVertices, dtype=float)
        b[0] = 1.0
        b[numVertices - 1] = -1.0
        BigbVector_parts.append(b)

        wnonFree = np.full(numEdges, float(edgeListWeight[i]), dtype=float)
        wFree = np.full(numFreeEdges, float(edgeListWeight[i]), dtype=float)
        BigNonFreewVector_parts.append(wnonFree)
        BigFreewVector_parts.append(wFree)

        A_row_offset += numVertices
        B_row_offset += numVertices
        A_col_offset += numEdges
        B_col_offset += numFreeEdges
        M_row_offset += numEdges

    BigAConstraint = sp.coo_matrix(
        (A_data, (A_rows, A_cols)),
        shape=(A_row_offset, A_col_offset),
        dtype=float,
    ).tocsr()
    BigBConstraint = sp.coo_matrix(
        (B_data, (B_rows, B_cols)),
        shape=(B_row_offset, B_col_offset),
        dtype=float,
    ).tocsr()
    BigMConstraint = sp.coo_matrix(
        (M_data, (M_rows, M_cols)),
        shape=(M_row_offset, numTokens),
        dtype=float,
    ).tocsr()

    BigbVector = np.hstack(BigbVector_parts) if BigbVector_parts else np.array([], dtype=float)
    BigFreewVector = np.hstack(BigFreewVector_parts) if BigFreewVector_parts else np.array([], dtype=float)
    BigNonFreewVector = np.hstack(BigNonFreewVector_parts) if BigNonFreewVector_parts else np.array([], dtype=float)
    tokensCap = np.ones(numTokens, dtype=float)

    return {
        "BigAConstraint": BigAConstraint,
        "BigBConstraint": BigBConstraint,
        "BigMConstraint": BigMConstraint,
        "BigbVector": BigbVector,
        "BigFreewVector": BigFreewVector,
        "BigNonFreewVector": BigNonFreewVector,
        "tokensCap": tokensCap,
        "numNonFreeEdges": A_col_offset,
        "numFreeEdges": B_col_offset,
        "numTokens": numTokens,
    }


def build_cuopt_standard_form(lp_blocks, numAllowedTokens: int):
    BigAConstraint = lp_blocks["BigAConstraint"]
    BigBConstraint = lp_blocks["BigBConstraint"]
    BigMConstraint = lp_blocks["BigMConstraint"]
    BigbVector = lp_blocks["BigbVector"]
    BigFreewVector = lp_blocks["BigFreewVector"]
    BigNonFreewVector = lp_blocks["BigNonFreewVector"]

    num_f = lp_blocks["numNonFreeEdges"]
    num_g = lp_blocks["numFreeEdges"]
    num_t = lp_blocks["numTokens"]
    num_x = num_f + num_g + num_t

    zeros_eq_t = sp.csr_matrix((BigAConstraint.shape[0], num_t), dtype=float)
    A_eq = sp.hstack([BigAConstraint, BigBConstraint, zeros_eq_t], format="csr")
    b_eq = BigbVector.astype(float)

    eye_f = sp.identity(num_f, format="csr", dtype=float)
    zeros_fg = sp.csr_matrix((num_f, num_g), dtype=float)
    A_ub_flow = sp.hstack([eye_f, zeros_fg, -BigMConstraint], format="csr")
    b_ub_flow = np.zeros(num_f, dtype=float)

    if num_t > 0:
        sum_t_data = np.ones(num_t, dtype=float)
        sum_t_row = np.zeros(num_t, dtype=int)
        sum_t_col = np.arange(num_f + num_g, num_x, dtype=int)
        A_ub_budget = sp.coo_matrix(
            (sum_t_data, (sum_t_row, sum_t_col)),
            shape=(1, num_x),
            dtype=float,
        ).tocsr()
    else:
        A_ub_budget = sp.csr_matrix((1, num_x), dtype=float)
    b_ub_budget = np.array([float(numAllowedTokens)], dtype=float)

    A_ub = sp.vstack([A_ub_flow, A_ub_budget], format="csr")
    b_ub = np.hstack([b_ub_flow, b_ub_budget])

    c = np.hstack([BigNonFreewVector, BigFreewVector, np.zeros(num_t, dtype=float)])
    lower_bounds = np.zeros(num_x, dtype=float)
    upper_bounds = np.full(num_x, 1.0, dtype=float)
    upper_bounds[num_f + num_g:] = 1.0

    return {
        "A_eq": A_eq,
        "b_eq": b_eq,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "c": c,
        "lb": lower_bounds,
        "ub": upper_bounds,
        "num_f": num_f,
        "num_g": num_g,
        "num_t": num_t,
    }


def _build_linear_expression(variables, coeff_indices, coeff_values):
    expr = 0.0
    for col_idx, value in zip(coeff_indices, coeff_values):
        expr += float(value) * variables[int(col_idx)]
    return expr


def _iter_csr_rows(matrix: sp.csr_matrix):
    indptr = matrix.indptr
    indices = matrix.indices
    data = matrix.data
    for row_idx in range(matrix.shape[0]):
        start = indptr[row_idx]
        end = indptr[row_idx + 1]
        yield row_idx, indices[start:end], data[start:end]


def _get_var_value(variable_obj):
    if hasattr(variable_obj, "getValue"):
        return variable_obj.getValue()
    return variable_obj.Value


def solve_lp_direct_cuopt(cuopt_lp_data, solver_parameters=None, verbose: bool = True):
    try:
        from cuopt.linear_programming.problem import MINIMIZE, Problem
    except ImportError:
        from cuopt.linear_programming.problem import Problem, sense
        MINIMIZE = sense.MINIMIZE
    from cuopt.linear_programming.solver_settings import SolverSettings

    A_eq = cuopt_lp_data["A_eq"]
    b_eq = cuopt_lp_data["b_eq"]
    A_ub = cuopt_lp_data["A_ub"]
    b_ub = cuopt_lp_data["b_ub"]
    c = cuopt_lp_data["c"]
    lb = cuopt_lp_data["lb"]
    ub = cuopt_lp_data["ub"]
    num_f = cuopt_lp_data["num_f"]
    num_g = cuopt_lp_data["num_g"]
    num_t = cuopt_lp_data["num_t"]

    problem = Problem("tokenizer_lp_cuopt")
    variables = []

    if verbose:
        print(f"Creating {len(c)} LP variables in cuOpt")
    for idx in range(len(c)):
        var = problem.addVariable(
            lb=float(lb[idx]),
            ub=float(ub[idx]),
            obj=float(c[idx]),
            name=f"x_{idx}",
        )
        variables.append(var)

    if verbose:
        print(f"Adding {A_eq.shape[0]} equality constraints to cuOpt")
    for row_idx, cols, vals in _iter_csr_rows(A_eq):
        rhs = float(b_eq[row_idx])
        if len(cols) == 0:
            if abs(rhs) > 1e-12:
                raise ValueError("Inconsistent empty equality row encountered while building cuOpt model.")
            continue
        expr = _build_linear_expression(variables, cols, vals)
        problem.addConstraint(expr == rhs, f"eq_{row_idx}")

    if verbose:
        print(f"Adding {A_ub.shape[0]} inequality constraints to cuOpt")
    for row_idx, cols, vals in _iter_csr_rows(A_ub):
        rhs = float(b_ub[row_idx])
        if len(cols) == 0:
            if rhs < -1e-12:
                raise ValueError("Inconsistent empty inequality row encountered while building cuOpt model.")
            continue
        expr = _build_linear_expression(variables, cols, vals)
        problem.addConstraint(expr <= rhs, f"ub_{row_idx}")

    # Objective coefficients are attached to variables via addVariable(obj=...).
    objective_expr = 0.0 * variables[0] if variables else 0.0
    problem.setObjective(objective_expr, MINIMIZE)
    settings = SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)
    settings.set_parameter(CUOPT_PDLP_SOLVER_MODE, PDLPSolverMode.Stable2)
    if solver_parameters:
        for param_name, param_value in solver_parameters.items():
            settings.set_parameter(param_name, param_value)

    start = time.time()
    problem.solve(settings)
    end = time.time()

    status_obj = getattr(problem, "Status", getattr(problem, "status", None))
    status_name = getattr(status_obj, "name", str(status_obj))
    solve_time = getattr(problem, "SolveTime", getattr(problem, "solve_time", end - start))

    t_offset = num_f + num_g
    t_values = np.array([float(_get_var_value(variables[t_offset + i])) for i in range(num_t)], dtype=float)

    return {
        "status_name": status_name,
        "solve_time": solve_time,
        "wall_time": end - start,
        "t_values": t_values,
    }

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
    print("setup_LP_tokenization finished")

    return problem


def tokenize(edgesList: list[list[tokenInstance]] , 
            edgeListWeight:list[int] , 
            numVerticesList:list[int],
            just_size:bool=False):
    
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

    problem.solve(solver=cp.GLOP)
    #problem.solve(solver=cp.CUOPT)
    flow_values = f.value 
    shortest_paths = []
    offset = 0

    if flow_values is not None:
        if not just_size:
            for i in range(numStrings):
                edges = edgesList[i]
                numEdges = len(edges)
                flows = flow_values[offset:offset+numEdges]
                used_edges = [edges[j].token_index for j in range(numEdges) if flows[j] > 1e-6]  # tolerance for numerical noise
                shortest_paths.append(used_edges)
                offset += numEdges

               
            flat_tokens = []
            for sublist in shortest_paths:
                flat_tokens.extend(sublist)
            return flat_tokens
        else:
            return f.value
    else:
      raise ValueError("Cannot represent data")


def create_vocab(inputStringList: list[str],
                 inputStringFreq: list[int],
                 numAllowedTokens: int, 
                 vocab_size:int,
                 minTokenCount: int = 1,  
                 maxTokenLength: int = 5, 
                 all_tokens: bool = True):

    numStrings = len(inputStringList)

    edgesList = []
    tokensList = []
    freeEdgesList = []
    numVertices = []

    if all_tokens:  
        for i in range(numStrings):
            stringLen = len(inputStringList[i])
            edgesList.append(hf.get_all_nonFree_substrings(inputStringList[i]))
            tokensList.append(hf.get_tokens(inputStringList[i]))
            freeEdgesList.append(hf.get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen + 1)
    else:
        for i in range(numStrings):
            stringLen = len(inputStringList[i])
            edgesList.append(hf.get_all_nonFree_substrings_upto_len_t(inputStringList[i], maxTokenLength))
            tokensList.append(hf.get_tokens_upto_len_t(inputStringList[i], maxTokenLength))
            freeEdgesList.append(hf.get_all_free_substrings(inputStringList[i]))
            numVertices.append(stringLen + 1)

    tokens = list(set([item for sublist in tokensList for item in sublist]))
    hf.update_token_instance_counts(tokens, inputStringFreq, edgesList)
    tokens_to_keep = [token for token in tokens if token.token_instance_count > minTokenCount]
    keep_set = set(t.token for t in tokens_to_keep)

    filtered_edgesList = [
        [token for token in sublist if token.token in keep_set]
        for sublist in edgesList
    ]

    lpProblem = setup_LP_tokenization(filtered_edgesList, inputStringFreq, tokens_to_keep, freeEdgesList, numVertices)
    numAllowedTokensParam = lpProblem.parameters()[0]
    numAllowedTokensParam.value = numAllowedTokens

    # # --- Memory tracking setup ---
    # process = psutil.Process(os.getpid())
    # memory_samples = []
    # timestamps = []
    # stop_flag = False

    # def track_memory(interval=0.05):
    #     start_time = time.time()
    #     while not stop_flag:
    #         mem = process.memory_info().rss / (1024**2)  # in MB
    #         memory_samples.append(mem)
    #         timestamps.append(time.time() - start_time)
    #         time.sleep(interval)

    # tracker_thread = threading.Thread(target=track_memory, daemon=True)
    # tracker_thread.start()
    # # --- End memory tracking setup ---

    start = time.time()
    lpProblem.solve(solver=cp.CUOPT,verbose=True)
    # lpProblem.solve(
    #     solver=cp.PDLP,
    #     verbose=True,
    #     solver_opts={
    #         "eps_optimal_absolute": 1.0e-6,
    #         "num_threads": 8,
    #         "num_shards": 32
    #     }
    # )
    end = time.time()

    internal_time=lpProblem.solver_stats.solve_time
    my_time= end - start
    output_file="computation_time.csv"
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Interal Time {internal_time}"])
        writer.writerow([f"My Time {my_time}"])


    # Stop memory tracking
    stop_flag = True
    # tracker_thread.join()

    #print(f"The LP solve took {my_time:.4f} seconds")
    # print(f"Peak memory: {max(memory_samples):.2f} MB, Average memory: {sum(memory_samples)/len(memory_samples):.2f} MB")

    # # Save memory usage plot
    # plt.figure(figsize=(10, 5))
    # plt.plot(timestamps, memory_samples, label="RSS Memory (MB)")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Memory (MB)")
    # plt.title("Memory Usage During LP Solve")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"lp_memory_usage_{vocab_size}.png")
    # print(f"Memory usage plot saved to lp_memory_usage_{vocab_size}.png")

    lpVariables = lpProblem.variables()
    tVar = lpVariables[2].value

    possibleTokens = []
    for i in range(len(tokens_to_keep)):
        if tVar[i] > 0.0:
            nonZeroToken = possibleToken(
                tokens_to_keep[i].get_token(),
                tVar[i],
                tokens_to_keep[i].get_count(),
                tokens_to_keep[i].get_index()
            )
            possibleTokens.append(nonZeroToken)
    print("create_vocab finished")

    return possibleTokens


def create_vocab_cuopt(inputStringList: list[str],
                       inputStringFreq: list[int],
                       numAllowedTokens: int,
                       vocab_size: int,
                       minTokenCount: int = 1,
                       maxTokenLength: int = 5,
                       all_tokens: bool = True,
                       solver_parameters=None,
                       verbose: bool = True):
    filtered_edgesList, freeEdgesList, numVertices, tokens_to_keep = prepare_vocab_lp_data(
        inputStringList=inputStringList,
        inputStringFreq=inputStringFreq,
        minTokenCount=minTokenCount,
        maxTokenLength=maxTokenLength,
        all_tokens=all_tokens,
    )

    lp_blocks = build_lp_blocks(
        edgesList=filtered_edgesList,
        edgeListWeight=inputStringFreq,
        tokens=tokens_to_keep,
        freeEdgesList=freeEdgesList,
        numVerticesList=numVertices,
    )
    cuopt_lp_data = build_cuopt_standard_form(lp_blocks, numAllowedTokens=numAllowedTokens)

    try:
        solve_output = solve_lp_direct_cuopt(
            cuopt_lp_data=cuopt_lp_data,
            solver_parameters=solver_parameters,
            verbose=verbose,
        )
    except ImportError as import_error:
        raise ImportError(
            "Direct cuOpt LP solve requested, but cuOpt Python modules are not available in this environment."
        ) from import_error

    status_name = solve_output["status_name"]
    if status_name is not None:
        normalized_status = str(status_name).lower()
        if "optimal" not in normalized_status and "feasible" not in normalized_status:
            raise RuntimeError(f"cuOpt solve did not return an optimal/feasible status. Status={status_name}")

    internal_time = solve_output["solve_time"]
    wall_time = solve_output["wall_time"]
    output_file = "computation_time.csv"
    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Interal Time {internal_time}"])
        writer.writerow([f"My Time {wall_time}"])

    tVar = solve_output["t_values"]

    possibleTokens = []
    for i in range(len(tokens_to_keep)):
        if tVar[i] > 0.0:
            nonZeroToken = possibleToken(
                tokens_to_keep[i].get_token(),
                tVar[i],
                tokens_to_keep[i].get_count(),
                tokens_to_keep[i].get_index(),
            )
            possibleTokens.append(nonZeroToken)
    print("create_vocab_cuopt finished")

    return possibleTokens


def create_vocab_old(inputStringList: list[str],
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
    



    hf.update_token_instance_counts(tokens,inputStringFreq,edgesList)

    tokens_to_keep = [token for token in tokens if token.token_instance_count > minTokenCount]

    # Create a set of valid token strings
    keep_set = set(t.token for t in tokens_to_keep)


    # Create a new edgesList that only contains tokens in keep_set
    filtered_edgesList = [
        [token for token in sublist if token.token in keep_set]
        for sublist in edgesList
    ]


    lpProblem=setup_LP_tokenization(filtered_edgesList,inputStringFreq,tokens_to_keep , freeEdgesList,numVertices)

    numAllowedTokensParam = lpProblem.parameters()[0]
    numAllowedTokensParam.value = numAllowedTokens

    start = time.time()
    #lpProblem.solve(solver=cp.GLOP)
    lpProblem.solve(
    solver=cp.PDLP,
    verbose=True,
    solver_opts={
        "eps_optimal_absolute": 1.0e-6,
        "num_threads": 8,
        "num_shards": 32
                 }
    )
    end=time.time()
    print(f"The first iteration took {end - start:.4f} seconds")

    lpVariables=lpProblem.variables()
   
    tVar=lpVariables[2].value
    
    
    possibleTokens=[]
    for i in range(len(tokens_to_keep)):
        if(tVar[i]>0.0):
            nonZeroToken=possibleToken(tokens_to_keep[i].get_token(),
                                       tVar[i],
                                       tokens_to_keep[i].get_count(),
                                       tokens_to_keep[i].get_index()  )

            
            possibleTokens.append(nonZeroToken)
    print("create_vocab_old finished")
    
    return possibleTokens



def deterministic_rounding(possible_tokens:list[possibleToken],unique_chars:list[str] ,vocab_size:int):
    if(vocab_size<len(unique_chars)):
        raise(ValueError( "Number of unique characters is greater than vocab size "))
    sorted_tokens=sorted(possible_tokens, key=lambda obj: obj.lp_value, reverse=True)

    tokens_to_choose=vocab_size-len(unique_chars)

    chosen_tokens=[token.token for token in sorted_tokens[0:tokens_to_choose]]

    tokens=list(set(unique_chars+chosen_tokens))


    return tokens


def biased_rounding(possible_tokens:list[possibleToken],unique_chars:list[str] ,vocab_size:int):
    if(vocab_size<len(unique_chars)):
        raise(ValueError( "Number of unique characters is greater than vocab size "))

    tokens_to_consider=[token for token in possible_tokens if token.lp_value>0]
    sorted_tokens=sorted(tokens_to_consider, key=lambda obj: obj.lp_value/len(obj.token), reverse=True)

    tokens_to_choose=vocab_size-len(unique_chars)
    chosen_tokens=[token.token for token in sorted_tokens[0:tokens_to_choose]]

    tokens=list(set(unique_chars+chosen_tokens))


    return tokens

def probabilistic_rounding(possible_tokens: list, unique_chars: list[str], vocab_size: int):
    if vocab_size < len(unique_chars):
        raise ValueError("Number of unique characters is greater than vocab size.")

    # Tokens that are always taken
    always_taking = [token.token for token in possible_tokens if token.lp_value > 0.99]

    # All candidate tokens (excluding those already taken)
    candidate_tokens = [token for token in possible_tokens 
                        if token.token not in always_taking]

    # If there are not enough tokens to sample, raise error
    remaining_budget = vocab_size - len(unique_chars)
    if len(always_taking) > remaining_budget:
        raise ValueError("Too many always-taking tokens to fit in vocabulary.")

    # Adjust remaining budget
    remaining_budget -= len(always_taking)

    # Get tokens and their associated probabilities
    token_list = [token.token for token in candidate_tokens]
    lp_values = np.array([token.lp_value for token in candidate_tokens])

    if len(lp_values) == 0 and remaining_budget > 0:
        raise ValueError("No available tokens to sample from.")
        
    probabilities = lp_values / lp_values.sum()

    # Sample without replacement
    sampled_tokens = list(np.random.choice(token_list, size=remaining_budget, replace=False, p=probabilities))

    # Final vocabulary
    final_vocab = list(set(unique_chars) | set(always_taking) | set(sampled_tokens))

    # Sanity check
    if len(final_vocab) != vocab_size:
        raise ValueError(f"Final vocabulary size {len(final_vocab)} does not match expected size {vocab_size}.")

    return final_vocab


def fill_missing_edges_with_unk(edges: list[tokenInstance], num_vertices: int, unk_token:str,unk_id: int ):
   
    # Keep only direct edges i -> i+1
    direct_edges = {(e.start, e.end): e for e in edges if e.end == e.start + 1}

    
    result_edges = edges

    # Walk through consecutive vertices
    for i in range(num_vertices - 1):
        if (i, i + 1) in direct_edges:
            # Keep the original edge
            result_edges.append(direct_edges[(i, i + 1)])
        else:
            # Insert UNK edge for missing step
            unk_edge = tokenInstance(
                token=unk_token,
                start=i,
                end=i + 1,
                token_index=unk_id
            )
            result_edges.append(unk_edge)

    return result_edges


    
