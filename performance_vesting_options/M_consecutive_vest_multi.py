"""
This code uses a Monte Carlo simulation to price a stock option with a market vesting condition. The simulation generates random paths for the stock price using geometric Brownian motion and checks if the stock price hits the threshold within a specified number of days. The simulation can be run using a single path method or an antithetic method to generate two paths for each iteration.
"""

import time
import os
import numpy as np
import pandas as pd
from math import ceil
from multiprocessing import Pool
import scipy.stats as stats

#  Single path is generated for each iteration
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
    intrinsic_value = max(end_stk - strike, 0) if indicator else 0
    # if indicator:
    #     intrinsic_value = max(end_stk - strike, 0)
    
    return intrinsic_value, end_stk


# Uses the antithetic method to generate two paths for each iteration
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


# Generate paths using multiprocessing
def gen_paths_multiprocessing(S0, rfr, vol, term, iterations, strike, threshold, m_days, num_processes, use_antithetic):
    args = [(S0, rfr, vol, term, strike, threshold, m_days, 1 + i) for i in range(iterations)]
    
    with Pool(num_processes) as pool:
        if use_antithetic:
            results = pool.map(anti_path_sim, args)
        else:
            results = pool.map(path_sim, args)

    fv, end_stks = zip(*results)

    return list(fv), list(end_stks)


# Output the results
def output_results(fv, end_stks, rfr, term, iterations, units, S0, run_time):
      
        #------ Statistics from Simulation ------#
        # Calculate average present value and average ending stock price
        df = np.exp(-rfr * term)
        pv_arr = [value * df for value in fv]
        std_dev = np.std(pv_arr)
        std_err = std_dev / np.sqrt(iterations)
        avg_end_stk = np.mean(end_stks)

        #------ Expected Ending vs Simulated ------# 
        # Calculate expected ending stock price and compare with average
        exp_end_stk = S0 * np.exp(rfr * term)
        diff = avg_end_stk - exp_end_stk
        diff_pct = (avg_end_stk / exp_end_stk - 1) * 100
        avg_value = np.mean(pv_arr)
        avg_total_value = avg_value * units

        # Print the results
        print(f"\nAverage Value (per unit): {avg_value:.4f}")
        if units > 1: print(f"Average Value (total): {avg_total_value:,.0f}")
        print(f"\nAvg Ending Stock Price: {avg_end_stk:.3f}") 
        print(f"Expected Ending Stock Price: {exp_end_stk:.3f}")
        print(f"Diff: {diff:.3f} \nDiff(%): {diff_pct:.3f}%")
        print(f"\nStandard error: {std_err:.6f}")

        #------ Calculate Confidence Intervals ------#
        # Calculate 95% confidence interval
        z_score = stats.norm.ppf(0.975)  # two-tailed test: 1 - (0.05 / 2)
        margin_error = z_score * std_err # margin of error

        # 95% Confidence Interval: Per Unit
        low_bound = avg_value - margin_error
        up_bound = avg_value + margin_error
        print(f"95% Confidence Interval (per unit): ({low_bound:.4f} - {up_bound:.4f}) Range: {up_bound - low_bound:.4f}")

        # 95% Confidence Interval: Total
        if units > 1:
            low_bound = low_bound * units
            up_bound = up_bound * units
            print(f"95% Confidence Interval (total): ({low_bound:,.2f} - {up_bound:,.2f}) Range: {up_bound - low_bound:,.2f}")

        # Output run time statistics
        print(f"\nIterations: {iterations:,}")
        print(f"\nTime taken (seconds): {run_time:.3f}")
        print(f"Iterations per second: {iterations / (run_time):,.0f}\n")  
    
        # Output to csv file
        if output_to_csv:
            results_df = pd.DataFrame({'Present Value': pv_arr, 'Ending Stock Price': end_stks})
            results_df.to_csv(f'results {iterations}.csv', index=False)
    

if __name__ == '__main__':

    # Initialize parameters
    S0 = 5.00
    strike = 0.01
    rfr = 0.05
    vol = 0.50
    term = 1.00
    iterations = 100_000
    threshold = 10.00
    m_days = 20
    units = 1
    use_antithetic=True
    
    # Set to True to output results to a csv file
    output_to_csv = False
    
    # Set the number of processes to use in multiprocessing implementation
    num_processes = int(os.cpu_count() * 3/4)  # Adjust this based on your machine's capabilities

    # Run the function with multiprocessing
    start = time.time() # start time
    fv, end_stks = gen_paths_multiprocessing(S0, rfr, vol, term, iterations, strike, threshold, m_days, num_processes, use_antithetic)
    run_time = time.time() - start # time to run simulation
    
    # Call the output_results function
    output_results(fv, end_stks, rfr, term, iterations, units, S0, run_time)
    

    
