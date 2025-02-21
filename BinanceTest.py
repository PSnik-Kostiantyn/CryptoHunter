import requests

url = "https://api.binance.com/api/v3/klines"
params = {
    "symbol": "BTCUSDT",
    "interval": "1h",
    "limit": 12
}

response = requests.get(url, params=params)
data = response.json()

for candle in data:
    open_time, open_price, high, low, close, volume, *_ = candle
    print(f"Time: {open_time}, Open: {open_price}, High: {high}, Low: {low}, Close: {close}, Volume: {volume}")
