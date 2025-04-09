import requests
import pandas as pd
import numpy as np
import time
from ta.momentum import RSIIndicator
from ta.trend import MACD
import os

def round_to_hour(ts):
    return ((ts + 3599) // 3600) * 3600


def update_btc_dataset():
    dataset_path = '../datasets/BTC_ready_res.csv'
    print("Завантаження існуючих даних...")
    df = pd.read_csv(dataset_path)

    if df.empty:
        print("Файл порожній або невалідний.")
        return

    last_timestamp = int(df["Close time"].max())
    print("Останній запис у датасеті (Unix):", last_timestamp)

    now_unix = int(time.time())
    now_hour_unix = now_unix - (now_unix % 3600)

    if now_hour_unix <= last_timestamp:
        print("Дані вже актуальні.")
        return

    print("Починається оновлення...")
    limit = 2500
    added_rows = []
    requests_sent = 0
    current_time = last_timestamp + 3600

    while current_time <= now_hour_unix and requests_sent < limit:
        start_time_unix = current_time * 1000
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "startTime": start_time_unix,
            "limit": 1
        }
        try:
            response = requests.get(url, params=params)
            candle_data = response.json()
            if not candle_data:
                print("Немає нових даних на:", current_time)
                break

            candle = candle_data[0]
            open_time = int(candle[0]) // 1000
            close_time = round_to_hour(int(candle[6]) // 1000)

            score = analyze_news_by_data(open_time)
            requests_sent += 1

            row = {
                "Open time": open_time,
                "Open": float(candle[1]),
                "High": float(candle[2]),
                "Low": float(candle[3]),
                "Close": float(candle[4]),
                "Volume": float(candle[5]),
                "Close time": close_time,
                "Quote asset volume": float(candle[7]),
                "Number of trades": int(candle[8]),
                "Taker buy base asset volume": float(candle[9]),
                "Taker buy quote asset volume": float(candle[10]),
                "Score": score
            }
            added_rows.append(row)
            print(f"Додано: {close_time}")
            current_time += 3600
            time.sleep(0.1)

        except Exception as e:
            print("Помилка під час запиту або обробки:", e)
            break

    if not added_rows:
        print("Жодного нового рядка не додано.")
        return

    new_df = pd.DataFrame(added_rows)
    print(f"Додано {len(new_df)} нових рядків.")

    full_df = pd.concat([df, new_df], ignore_index=True)
    full_df = full_df.sort_values("Close time")

    full_df["SMA_20"] = full_df["Close"].rolling(window=20).mean()
    full_df["RSI_14"] = RSIIndicator(full_df["Close"], window=14).rsi()
    full_df["RSI_7"] = RSIIndicator(full_df["Close"], window=7).rsi()

    macd = MACD(full_df["Close"], window_slow=26, window_fast=12, window_sign=9)
    full_df["MACD_14"] = macd.macd()

    full_df = full_df.dropna()
    full_df.to_csv(dataset_path, index=False)
    print("Оновлено BTC_ready_res.csv")

def analyze_news_by_data(unix_timestamp):
    # TODO: імплементувати реальний аналіз новин
    return np.random.uniform(-1, 1)

if __name__ == "__main__":
    update_btc_dataset()
