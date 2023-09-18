import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats #used for finding linear regression parameters later

# Set the stock symbol and fetch data
stock_symbol = "ADANIGREEN.NS" 
nifty_symbol = "^NSEI" #nifty index symbol
start_date = "2022-01-01"
end_date = "2023-01-01"

stock_data = yf.download(stock_symbol, start=start_date, end=end_date) #here stock data is initialised as a pandas data frame
nifty_data = yf.download(nifty_symbol, start=start_date, end=end_date) #similarly for nifty data

# Calculate daily returns
stock_data['Stock Daily Return'] = stock_data['Adj Close'].pct_change() #creates a new column in the dataframe that stores the percentage change in the current and previous adjusted closing price of the stock
nifty_data['NIFTY Daily Return'] = nifty_data['Adj Close'].pct_change() #similarly for the nifty index price

# Merge stock and NIFTY data
merged_data = pd.concat([stock_data['Stock Daily Return'], nifty_data['NIFTY Daily Return']], axis=1).dropna() #combines the two columns of dataframes side by side(as axis=1)
#dropna is pandas method used to drop any row that has a missing value so as to not include it in our analysis

# Calculate cumulative return
stock_cumulative_return = (stock_data['Adj Close'][-1] / stock_data['Adj Close'][0]) - 1 #compares the value of stock at the end and start of the specified period and then subtract by 1 to get change wrt to intial value
nifty_cumulative_return = (nifty_data['Adj Close'][-1] / nifty_data['Adj Close'][0]) - 1

# Plot cumulative returns
plt.figure(figsize=(13, 7)) #13 width and 7 height of the plot
plt.plot(stock_data.index, stock_data['Adj Close'], label="Stock's Closing Price") #.index gives the date index for x axis, and adjusted closing price as y axis
plt.plot(nifty_data.index, nifty_data['Adj Close'], label="NIFTY Index Closing Price")
plt.legend()        
plt.title("Stock vs NIFTY Index - Closing Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(merged_data['NIFTY Daily Return'], merged_data['Stock Daily Return']) # applied on merged data to get a meaningful regression between the two dataframes
#beta is slope or the sensitivity of the stock relative to the nifty index
#alpha is intercept or the excess return compared to the expected as calculated from beta
#r value is correlation coefficient and shows the relationship between the stock and nifty daily returns. positive shows direct relationship and magnitude shows the intensity
#p value associated with null hypothesis that slope or beta is 0, small p shows it contradicts null so slope is not 0 or the relationship is statistically significant
#std error is standard error in the estimate of beta or slope

print("Beta (Slope):", slope)
print("Alpha (Intercept):", intercept)
print("R-squared (R value):", r_value**2) #printing R squared value, it explains the percentage of dependent variable explained by independent variable

# Calculate expected return using CAPM
risk_free_rate = 0.06  # assumed as 6 percent
market_risk_premium = 0.08  # assumed 8 percent
beta = slope

expected_return = risk_free_rate + (beta * market_risk_premium) #according to the capital asset pricing model

print("Expected Return:", expected_return)

# Plot linear regression
plt.figure(figsize=(13, 7))
plt.scatter(merged_data['NIFTY Daily Return'], merged_data['Stock Daily Return'], alpha=0.7) #gives scatter plot, aplha is transparency value that allows us to see overalapping points in a better way, 1 being opaque and 0 being completely transparent
plt.plot(merged_data['NIFTY Daily Return'], slope * merged_data['NIFTY Daily Return'] + intercept, label='Linear Regression')
plt.legend()
plt.title("Stock vs NIFTY Daily Returns")
plt.xlabel("NIFTY Daily Return")
plt.ylabel("Stock Daily Return")
plt.grid(True)
plt.show()