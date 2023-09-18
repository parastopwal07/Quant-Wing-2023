# Montecarlo
import yfinance as yf
import numpy as np

# Set stock symbols and fetch data
stock_symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS"]
start_date = "2022-01-01"
end_date = "2023-01-01"

stock_data = yf.download(stock_symbols, start=start_date, end=end_date)["Adj Close"]

# Calculate daily returns
returns = stock_data.pct_change().dropna()

# Calculate portfolio returns
weights = np.array([0.4, 0.4, 0.2]) 
portfolio_returns = np.dot(returns, weights)

# Monte Carlo simulation parameters
num_simulations = 100000  # number of simulations or iterations is high to get accurate model
confidence_level = 0.95  #standard value

# Generate simulated portfolio returns
simulated_returns = np.random.choice(portfolio_returns, size=(num_simulations, len(portfolio_returns)), replace=True) #replacement allowed
#creating a 2D array from portfolio_returns having row as the num_simulations and columns as returns for each day from start to end date
simulated_portfolio_returns = np.sum(simulated_returns, axis=1)
#sums up the return value for all days in a particular row and gives a 1D array, axis=1 implies horizontal operations

# Calculate VaR using Monte Carlo simulation
sorted_simulated_returns = np.sort(simulated_portfolio_returns) #sorting in ascending order
var_monte_carlo = -sorted_simulated_returns[int(num_simulations * (1 - confidence_level))] #it gives maximum index below which the value of stock falling has more than '1-confidence level' confidence
#-ve to get a postive VaR at the end, here to get index we convert the value of multiplication into int

print("VaR using Monte Carlo Simulation Approach:", var_monte_carlo)