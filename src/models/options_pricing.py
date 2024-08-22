import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call

# Example usage
S = 100  # Current stock price
K = 100  # Strike price
T = 1    # Time to expiration (in years)
r = 0.05 # Risk-free rate
sigma = 0.2 # Volatility

option_price = black_scholes(S, K, T, r, sigma)
print(f"The call option price is: {option_price:.2f}")