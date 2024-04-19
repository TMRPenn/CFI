import numpy as np
from scipy.stats import norm

def gap_call(S, X1, X2, T, rf, div_yield, vol):
    """
    Computes the price of a Gap Call Option based on the Reiner and Rubinstein formula.
    
    Parameters:
    S (float): Spot price of the underlying asset
    X1 (float): Strike price of the option
    X2 (float): Strike price for the payoff
    T (float): Time to maturity (in years)
    r (float): Risk-free interest rate (continuous)
    div_yield (float): dividend yield
    vol (float): Volatility of the underlying asset
    
    Returns:
    float: Price of the Gap Call Option
    """
    b = rf - div_yield
    
    # Calculate d1 and d2
    d1 = (np.log(S / X1) + (b + vol**2 / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    # Calculate the Gap Call price
    gap_call_price = S * np.exp((b-rf) * T) * norm.cdf(d1) - X2 * np.exp(-rf * T) * norm.cdf(d2)
    
    return gap_call_price

def gap_put(S, X1, X2, T, rf, div_yield, vol):
    """
    Computes the price of a Gap Put Option based on the Reiner and Rubinstein formula.
    
    Parameters:
    S (float): Spot price of the underlying asset
    X1 (float): Strike price of the option
    X2 (float): Strike price for the payoff
    T (float): Time to maturity (in years)
    r (float): Risk-free interest rate  (continuous)
    b (float): Cost of carry
    vol (float): Volatility of the underlying asset
    
    Returns:
    float: Price of the Gap Put Option
    """
    b = rf - div_yield
    
    # Calculate d1 and d2
    d1 = (np.log(S / X1) + (b + vol**2 / 2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    # Calculate the Gap Put price
    gap_put_price = X2 * np.exp(-rf * T) * norm.cdf(-d2) - S * np.exp((b-rf) * T) * norm.cdf(-d1)
    
    return gap_put_price

if __name__ == "__main__":
    # Example usage:
    print(gap_call(S=50, X1=50, X2=57, T=0.5, rf=0.09, div_yield=0, vol=0.2))
    print(gap_put(S=100, X1=100, X2=95, T=1, rf=0.05, div_yield=0, vol=0.2))
