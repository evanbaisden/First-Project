import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def monte_carlo_simulation(S0, mu, sigma, T, N, M):
    """
    Perform Monte Carlo simulation for asset price prediction.
    
    Args:
    S0 (float): Initial stock price
    mu (float): Expected return (annualized)
    sigma (float): Volatility (annualized)
    T (float): Time period (in years)
    N (int): Number of time steps
    M (int): Number of simulations
    
    Returns:
    numpy.ndarray: Matrix of simulated asset prices
    """
    dt = T/N
    S = np.zeros((N+1, M))
    S[0] = S0
    
    rng = np.random.default_rng()  # Create a new Generator instance
    for t in range(1, N+1):
        z = rng.standard_normal(M)
        S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    return S

def plot_simulations(S, title):
    """
    Plot the results of Monte Carlo simulations.
    
    Args:
    S (numpy.ndarray): Matrix of simulated asset prices
    title (str): Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(S)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel('Asset Price')
    plt.show()

# Example usage
S0 = 100  # Initial stock price
mu = 0.1  # Expected return (10%)
sigma = 0.2  # Volatility (20%)
T = 1  # Time period (1 year)
N = 252  # Number of trading days
M = 1000  # Number of simulations

simulated_prices = monte_carlo_simulation(S0, mu, sigma, T, N, M)
plot_simulations(simulated_prices, 'Monte Carlo Simulation of Asset Prices')

# Calculate and print some statistics
final_prices = simulated_prices[-1]
mean_price = np.mean(final_prices)
std_price = np.std(final_prices)
confidence_interval = norm.interval(0.95, loc=mean_price, scale=std_price/np.sqrt(M))

print(f"Expected price after {T} year: ${mean_price:.2f}")
print(f"95% Confidence Interval: (${confidence_interval[0]:.2f}, ${confidence_interval[1]:.2f})")
