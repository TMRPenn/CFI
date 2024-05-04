import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
from bivariate_norm_dist import bivariate_normal_cdf

def two_asset_call(S1, S2, X1, X2, T, rf, vol1, vol2, rho):
    """
    Computes the price of a Two-Asset Correlation Call Option.
    
    Parameters:
    S1 (float): Spot price of the first asset
    S2 (float): Spot price of the second asset
    X1 (float): Strike price for the first asset condition
    X2 (float): Strike price for the payoff condition
    T (float): Time to maturity (in years)
    rf (float): Risk-free interest rate (continuous)
    vol1 (float): Volatility of the first asset
    vol2 (float): Volatility of the second asset
    rho (float): Correlation coefficient between the returns on the two assets
    
    Returns:
    float: Price of the Two-Asset Correlation Call Option
    """

    # Calculate y1 and y2
    y1 = (np.log(S1 / X1) + (rf - vol1**2 / 2) * T) / (vol1 * np.sqrt(T))
    y2 = (np.log(S2 / X2) + (rf - vol2**2 / 2) * T) / (vol2 * np.sqrt(T))

    #M(a, b, rho) is the cumulative bivariate normal distribution with limits a, b, and correlation rho. 

    # M1 = M(y2 + vol2 * sqrt(T), y1 + rho * vol2 * sqrt(T); rho)
    a = y2 + vol2 * np.sqrt(T)
    b = y1 + rho * vol2 * np.sqrt(T)
    M1 = bivariate_normal_cdf(a, b, rho)

    # M2 = M(y2, y1; rho)
    M2 = bivariate_normal_cdf(y2, y1, rho)

    # Calculate the Two-Asset Call price
    call_price = S2 * M1 - X2 * np.exp(-rf * T) * M2

    return call_price



# Example usage of the two_asset_call function
call_test_1 = two_asset_call(S1=52, S2=65, X1=50, X2=70, T=0.5, 
                                rf=0.10, vol1=0.20, vol2=0.30, rho=0.75)
print(f"Haug Book: \nCall price: {call_test_1:.4f}, Expected: 4.7073, Diff: {call_test_1 - 4.7073:.4f} \n")

# Additional tests with provided data can be added here
call_test_2 = two_asset_call(S1=65, S2=70, X1=60, X2=72, T=0.25, 
                                rf=0.06, vol1=0.25, vol2=0.30, rho=0.80)
print(f"Java Book: \nCall price: {call_test_2:.4f}, Expected: 3.735, Diff: {call_test_2 - 3.735:.4f} \n")

call_test_3 = two_asset_call(S1=52, S2=65, X1=50, X2=70, T=0.5, 
                                rf=0.10, vol1=0.20, vol2=0.30, rho=0.75)
print(f"Multi-Asset Options: \nCall price: {call_test_3:.4f}, Expected: 4.7073, Diff: {call_test_3 - 4.7073:.4f} \n")

