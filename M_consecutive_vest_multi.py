import time
import os
import numpy as np
import pandas as pd
from math import ceil
from multiprocessing import Pool

def path_sim(args):
    S0, rfr, vol, term, strike, threshold, m_days, seed = args
    np.random.seed(seed)  # Ensure different seed for each process
    dt = 1/252
    steps = ceil(term / dt)

    # Generate random values for the path
    rand = np.random.standard_normal(steps)
    path = S0 * np.exp(np.cumsum((rfr - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * rand))
    path = np.insert(path, 0, S0)

    path_df = pd.Series(path)
    indicator = np.any(path_df.rolling(m_days, min_periods=0).min() >= threshold)
    
    end_stk = path[-1]
    intrinsic_value = 0
    if indicator:
        intrinsic_value = max(end_stk - strike, 0)
    
    return intrinsic_value, end_stk


def anti_path_sim(args):
    S0, rfr, vol, term, strike, threshold, m_days, seed = args
    np.random.seed(seed)  # Ensure different seed for each process
    dt = 1/252
    steps = ceil(term / dt)

    # Generate random values for the first path
    rand_1 = np.random.standard_normal(steps)
    path_1 = S0 * np.exp(np.cumsum((rfr - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * rand_1))
    path_1 = np.insert(path_1, 0, S0)

    # Use the negative of the random values for the antithetic path
    rand_2 = -rand_1
    path_2 = S0 * np.exp(np.cumsum((rfr - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * rand_2))
    path_2 = np.insert(path_2, 0, S0)

    # Analyze the first path
    path_df_1 = pd.Series(path_1)
    indicator_1 = np.any(path_df_1.rolling(m_days, min_periods=0).min() >= threshold)
    end_stk_1 = path_1[-1]
    intrinsic_value_1 = max(end_stk_1 - strike, 0) if indicator_1 else 0

    # Analyze the antithetic path
    path_df_2 = pd.Series(path_2)
    indicator_2 = np.any(path_df_2.rolling(m_days, min_periods=0).min() >= threshold)
    end_stk_2 = path_2[-1]
    intrinsic_value_2 = max(end_stk_2 - strike, 0) if indicator_2 else 0

    # Averaging the results from both paths
    avg_intrinsic_value = (intrinsic_value_1 + intrinsic_value_2) / 2
    avg_end_stk = (end_stk_1 + end_stk_2) / 2
    
    return avg_intrinsic_value, avg_end_stk

def gen_paths_multiprocessing(S0, rfr, vol, term, iterations, strike, threshold, m_days, num_processes):
    args = [(S0, rfr, vol, term, strike, threshold, m_days, 1 + i) for i in range(iterations)]
    
    with Pool(num_processes) as pool:
        # results = pool.map(path_sim, args)
        results = pool.map(anti_path_sim, args)

    fv, end_stks = zip(*results)

    return list(fv), list(end_stks)

# Initialize parameters
S0 = 2.61
strike = 0.01
rfr = 0.0402
vol = 1.30
term = 5.00
iterations = 2_000_000
threshold = 8.72
m_days = 20
num_processes = int(os.cpu_count() * 2/3)  # Adjust this based on your machine's capabilities

if __name__ == '__main__':
    
    # start time
    start = time.time()
    
    # Run the function with multiprocessing
    fv, end_stks = gen_paths_multiprocessing(S0, rfr, vol, term, iterations, strike, threshold, m_days, num_processes)
    
    # Calculate average present value and average ending stock price
    df = np.exp(-rfr * term)
    pv_arr = [value * df for value in fv]
    std_dev = np.std(pv_arr)
    std_err = std_dev / np.sqrt(iterations)
    avg_end_stk = np.mean(end_stks)

    # calculate expected ending stock price and compare with average
    exp_end_stk = S0 * np.exp(rfr * term)
    diff = avg_end_stk - exp_end_stk
    diff_pct = (avg_end_stk / exp_end_stk - 1) * 100
    print(f"\nAverage value: {np.mean(pv_arr):.8f}")
    print(f"{avg_end_stk:.2f}, {exp_end_stk:.2f}, {diff:.2f}, {diff_pct:.2f}%")
    print(f"standard error: {std_err:.8f}")
    
    # output end time and iterations per second
    print(f"\nIterations: {iterations:,}")
    print(f"\nTime taken (seconds): {time.time() - start:.3f}")
    print(f"Iterations per second: {iterations / (time.time() - start):,.0f}")
    
    
    # export the results to a csv file
    results_df = pd.DataFrame({'Present Value': pv_arr, 'Ending Stock Price': end_stks})
    results_df.to_csv(f'results {iterations}.csv', index=False)
    
