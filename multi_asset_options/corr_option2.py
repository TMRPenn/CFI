# STILL NEEDS TO BE TESTED

"""
From Multi-Asset Options - A Numerical Study

exact_price_European_two_asset_corr_call computes the exact price
of an European two-asset correlation call option.

Input parameters are the initial asset prices S_0, volatilities sigma,
risk-free interest rate r, correlation rho, time until maturity T, and
strike prices K.

Returns the exact price as option-price.

Note: Uses a help-function bivariate_normal_pdf which returns the
probability density function (pdf) of the bivariate normal
distribution with parameters x, y and rho.
"""

import numpy as np
from scipy import integrate

def bivariate_normal_pdf(x, y, rho):
    return (1/(2*np.pi*np.sqrt(1-np.power(rho,2)))*np.exp(-1/(2*(1-np.power(rho,2)))*(np.power(x,2)-2*rho*x*y+np.power(y,2))))

def exact_price_European_two_asset_corr_call(S_0, sigma, r, rho, T, K):
    y_1 = (np.log(S_0[0]/K[0]) + T*(r-np.power(sigma[0], 2)/2))/(sigma[0]*np.sqrt(T))
    y_2 = (np.log(S_0[1]/K[1]) + T*(r-np.power(sigma[1], 2)/2))/(sigma[1]*np.sqrt(T))
    M1 = integrate.nquad(bivariate_normal_pdf, [[-np.inf, y_1+rho*sigma[1]*np.sqrt(T)], [y_2+sigma[1]*np.sqrt(T), np.inf]], args=([rho]))
    M2 = integrate.nquad(bivariate_normal_pdf, [[-np.inf, y_1], [-np.inf, y_2]], args=([rho]))
    option_price = S_0[1]*M1[0] - K[1]*np.exp(-r*T)*M2[0]
    return option_price

if __name__ == '__main__':
    S_0 = [52, 65]
    K = [50, 70]
    sigma = [0.2, 0.3]
    r = 0.10
    rho = 0.75
    T = 0.5
    
    print(exact_price_European_two_asset_corr_call(S_0, sigma, r, rho, T, K))