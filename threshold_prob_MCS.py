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
    rand = np.random.standard_normal(steps) # creates a numpy array of random values

    # Creates a numpy array of stock prices using geometric brownian motion and random numbers from rand
    path = S0 * np.exp(np.cumsum((rfr - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * rand)) 
    path = np.insert(path, 0, S0)  # Insert the initial stock price at the beginning of the path

    path_df = pd.Series(path)
    indicator = np.any(path_df.rolling(m_days, min_periods=0).min() >= threshold)
    
    end_stk = path[-1]
    hit_indicator = 0
    if indicator:
        hit_indicator = 1
    
    return hit_indicator, end_stk


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
def gen_paths_multiprocessing(S0, rfr, vol, term, iterations, strike, threshold, m_days, num_processes):
    args = [(S0, rfr, vol, term, strike, threshold, m_days, 1 + i) for i in range(iterations)]
    
    with Pool(num_processes) as pool:
        results = pool.map(path_sim, args)          # Use this line for single path method
        # results = pool.map(anti_path_sim, args)   # Use this line for antithetic method

    hits, end_stks = zip(*results)

    return list(hits), list(end_stks)


# Output the results
def output_results(fv, end_stks, rfr, term, iterations, units, S0, run_time):
        
        print(f"Hit Count: {hit_list.count(1)}")
        print(f"Hit Probability: {hit_list.count(1) / len(hit_list) * 100}")
        
        # Calculate average ending stock price
        avg_end_stk = np.mean(end_stks)

        # Calculate expected ending stock price and compare with average
        exp_end_stk = S0 * np.exp(rfr * term)
        diff = avg_end_stk - exp_end_stk
        diff_pct = (avg_end_stk / exp_end_stk - 1) * 100

        
        # Print the results
        print(f"\nAvg Ending Stock Price: {avg_end_stk:.3f} \nExpected Ending Stock Price: {exp_end_stk:.3f} \nDiff: {diff:.3f} \nDiff(%): {diff_pct:.3f}%")

        # Output timing / run statistics
        print(f"\nIterations: {iterations:,}")
        print(f"\nTime taken (seconds): {run_time:.3f}")
        print(f"Iterations per second: {iterations / (run_time):,.0f}\n")      
    
        if output_to_csv:
            # Export the results to a csv file
            # results_df = pd.DataFrame({'Present Value': pv_arr, 'Ending Stock Price': end_stks})
            # results_df.to_csv(f'results {iterations}.csv', index=False)
            pass
    

if __name__ == '__main__':

    # Initialize parameters
    S0 = 10.00
    strike = 0.00
    rfr = 0.05
    vol = 0.70
    term = 1.00
    iterations = 100_000
    threshold = 12.00
    m_days = 1
    units = 1
    
    # Set to True to output results to a csv file
    output_to_csv = False
    
    # Set the number of processes to use in multiprocessing implementation
    num_processes = int(os.cpu_count() * 2/3)  # Adjust this based on your machine's capabilities

    # Run the function with multiprocessing
    start = time.time() # start time
    hit_list, end_stks = gen_paths_multiprocessing(S0, rfr, vol, term, iterations, strike, threshold, m_days, num_processes)
    run_time = time.time() - start # time to run simulation
    
    # Call the output_results function
    output_results(hit_list, end_stks, rfr, term, iterations, units, S0, run_time)    
