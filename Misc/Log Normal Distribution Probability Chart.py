import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Given parameters
S_0 = 10  # Initial stock price
X = 11    # Strike price
T = 1     # Time to maturity in years
rf = 0.05 # Risk-free rate
volatility = 0.20 # Volatility

# Calculate d2 for the lognormal distribution
d1 = (np.log(S_0 / X) + (rf + 0.5 * volatility**2) * T) / (volatility * np.sqrt(T))
d2 = d1 - volatility * np.sqrt(T)

# Range of possible final stock prices
S_T = np.linspace(0, S_0 * 3, 1000)
pdf = (1 / (S_T * volatility * np.sqrt(2 * np.pi * T))) * np.exp(-(np.log(S_T / S_0) - (rf - 0.5 * volatility**2) * T)**2 / (2 * volatility**2 * T))

# Plot the lognormal distribution
plt.figure(figsize=(10, 6))
plt.plot(S_T, pdf, label='Lognormal Distribution')

# Shade the area where S_T >= X
plt.fill_between(S_T, pdf, where=(S_T >= X), color='skyblue', alpha=0.5, label=f'P(S_T ≥ X)')

# Mark S_0 and X
plt.axvline(X, color='red', linestyle='--', label=f'X = ${X}')
# plt.axvline(S_0, color='green', linestyle='--', label=f'S_0 = ${S_0}')

plt.title('Lognormal Distribution with Shaded Area for P(S_T >= X)')
plt.xlabel('Stock Price at Maturity (S_T)')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()