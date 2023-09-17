import yfinance as yf
import numpy as np
import pandas as pd

# Set stock symbols and fetch data
stock_symbols = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS"] 
start_date = "2022-01-01"
end_date = "2023-01-01"

stock_data = yf.download(stock_symbols, start=start_date, end=end_date)["Adj Close"]

# Calculate daily returns
returns = stock_data.pct_change().dropna()

# Calculate portfolio returns
weights = np.array([0.3, 0.4, 0.3])  #assuming weights of the stocks in the portfolio
portfolio_returns = np.dot(returns, weights) #.dots is used to perform matrix/ array multiplicatio, here weight with the corresponding return df is multiplied


# Variance-covariance approach
portfolio_mean = np.mean(portfolio_returns) #gets mean 
portfolio_std = np.std(portfolio_returns)#gets standard deviation

confidence_level = 0.95  #assume 95% confidence level
z_score = np.abs(np.percentile(portfolio_returns, 100 * (1 - confidence_level))) #.abs gives absolute value, .percentile returns the percentile value
#here it returns the absolute of percentile value below which the stock is expected to fall with the given confidence level

var_variance_covariance = -portfolio_mean + (portfolio_std * z_score) #formula to calculate the VaR 

print("VaR using Variance-Covariance Approach:", var_variance_covariance)


# Historical simulation approach
sorted_returns = np.sort(portfolio_returns) #sorts the returns in ascending order
var_historical = -sorted_returns[int(len(sorted_returns) * (1 - confidence_level))] #gives the return value corresponding to the given confidence level
#here - is used because VaR is generally a positive number, the int is the index in the sorted_returns array
print("VaR using Historical Simulation Approach:", var_historical)