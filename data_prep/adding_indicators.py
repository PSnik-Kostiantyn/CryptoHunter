import pandas as pd
import ta
from ta.trend import MACD
from ta.momentum import RSIIndicator

df = pd.read_csv('../datasets/btc_1h_data_2018_to_2025.csv')

df['Open time'] = pd.to_datetime(df['Open time']).astype(int) // 10**9

df = df.drop(columns=['Ignore'], errors='ignore')

score_df = pd.read_csv('../CryptoHunter/analyzed_news_for_work.csv')
score_df = score_df.rename(columns={'timestamp': 'Open time', 'score': 'Score'})

df = df.merge(score_df[['Open time', 'Score']], on='Open time', how='left')

df = df.dropna(subset=['Score'])

df['SMA_20'] = df['Close'].rolling(window=20).mean()

df['RSI_14'] = RSIIndicator(df['Close'], window=14).rsi()
df['RSI_7'] = RSIIndicator(df['Close'], window=7).rsi()

macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
df['MACD_14'] = macd.macd()

df = df.dropna()

df.to_csv('../datasets/BTC_ready.csv', index=False)

print("Ready!")
