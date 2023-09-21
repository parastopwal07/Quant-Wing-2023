import yfinance as yf
import pandas as pd #data manipulation library used especailly for organised data structures
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
ticker = "RELIANCE.NS" 
start_date = "2022-01-01"
end_date = "2023-01-01"
short_window = 12 #defining parameters necessary for calculating technical indicators
long_window = 26 #time over which long exponential moving average is calculated
signal_window = 9 #time for calculating signal from macd
bollinger_window = 20 #calculate bollinger based on most recent 20 periods of stock data
bollinger_std = 2 #standard deviations required to get upper and lower bollinger band

# Fetch stock data
stock_data = yf.download(ticker, start=start_date, end=end_date)["Adj Close"] #gets data using adj close that is closing price adjusted for dividends or stock splits coming up

# Calculate short-term and long-term moving averages
short_ma = stock_data.rolling(window=short_window).mean() #calculating short moving average using rolling method that gives us a moving window over short window and applies mean
long_ma = stock_data.rolling(window=long_window).mean()

# Calculate MACD and Signal Line
ema_short = stock_data.ewm(span=short_window, adjust=False).mean() #adjust false to avoid bias
ema_long = stock_data.ewm(span=long_window, adjust=False).mean() #ewm is another method to calculate exponential moving averages
macd = ema_short - ema_long #definiton of macd
signal = macd.ewm(span=signal_window, adjust=False).mean() #mean of macd over signal window gives signal line

# Calculate Bollinger Bands
rolling_mean = stock_data.rolling(window=bollinger_window).mean() #gives middle bollinger band
rolling_std = stock_data.rolling(window=bollinger_window).std() #gives rolling deviation for upper and lower band
upper_band = rolling_mean + (rolling_std * bollinger_std) #getting upper and lower band
lower_band = rolling_mean - (rolling_std * bollinger_std)

# Initialize trading signals
signals = pd.DataFrame(index=stock_data.index) #creating dataframe signal to store data and creates rows same as there are in stock data
signals["Signal"] = 0.0 #dataframe is a 2D table here it creates a column and initialises it to 0

# Generate signals based on MACD crossovers and Bollinger Bands
signals["Signal"][(macd > signal) & (stock_data > lower_band)] = 1.0 #checks the buy and sell condition 
signals["Signal"][(macd < signal) & (stock_data < upper_band)] = -1.0 #1 is buy and -1 is sell

# Calculate positions and returns
signals["Position"] = signals["Signal"].diff() #creates new column in df and stores difference in two successive rows of signal
signals["Returns"] = signals["Position"].shift(1) * stock_data.pct_change() #new column that gets return by multiplying stock data percentage change with the previous signal
#shift by 1 to match return and signal columns

# Calculate cumulative returns
cumulative_returns = (1 + signals["Returns"]).cumprod() #+1 to get the total return to the return column data then apply cumprod method that calculates the returns cumulatively

# Calculate Win Percentage and Loss Percentage
winning_trades = len(signals[signals["Returns"] > 0]) #gives only signal values that have return > 0, signals[boolean] is counted only when it is true
losing_trades = len(signals[signals["Returns"] < 0]) #similarly for loss
total_trades = winning_trades + losing_trades
win_percentage = (winning_trades / total_trades) * 100
loss_percentage = (losing_trades / total_trades) * 100

# Calculate Average Annual Returns
start_price = stock_data.iloc[0]
end_price = stock_data.iloc[-1]
total_return = (end_price / start_price - 1) * 100 #-1 to get relative to the initial price
num_years = len(stock_data) / 252  # Assuming 252 trading days in a year
average_annual_return = total_return / num_years

# Calculate Maximum Drawdown
cumulative_returns_max = cumulative_returns.cummax() #gives max cum 
drawdown = (cumulative_returns - cumulative_returns_max) / cumulative_returns_max # calculates cum max - current cum for each point
max_drawdown = drawdown.min() * 100  # Convert to percentage #gets min of the drawdown

# Print the results
print("Win Percentage:", win_percentage, "%")
print("Loss Percentage:", loss_percentage, "%")
print("Average Annual Returns:", average_annual_return, "%")
print("Maximum Drawdown:", max_drawdown, "%")

# Plot results
plt.figure(figsize=(13, 7))
plt.plot(cumulative_returns, label="Cumulative Returns") #plots cumulative return
plt.plot(stock_data / stock_data.iloc[0], label="Relative Returns")# plots relative price of stock wrt to intial price
plt.legend()
plt.title("MACD and Bollinger Band Backtest")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.show()