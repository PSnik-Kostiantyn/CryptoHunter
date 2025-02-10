import pandas as pd
import ta
from ta.trend import MACD
from ta.momentum import RSIIndicator

df = pd.read_csv('../datasets/formatted_data.csv')

df['SMA_20'] = df['Close'].rolling(window=20).mean()

df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['RSI_7'] = ta.momentum.RSIIndicator(df['Close'], window=7).rsi()

macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
df['MACD_14'] = macd.macd()

df = df.dropna()

df.to_csv('technical_indicators.csv', index=False)

print("Ready!")
