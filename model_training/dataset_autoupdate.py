import re

import pandas as pd
import requests
import time
from datetime import datetime, timedelta, timezone
import json
from ta.momentum import RSIIndicator
from ta.trend import MACD

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={GEMINI_API_KEY}"


def round_to_hour(ts):
    return ((ts + 3599) // 3600) * 3600

def update_btc_dataset():
    dataset_path = '../datasets/BTC_ready.csv'
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
    limit = 2000
    added_rows = []
    requests_sent = 0
    current_time = last_timestamp

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
    print("Оновлено BTC_ready.csv")

    # train_model_on_new_data(full_df)

def get_news_for_certain_hour(unix_timestamp):
    target_time = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
    next_hour = target_time + timedelta(hours=1)

    try:
        with open('../datasets/news.json', 'r', encoding='utf-8') as f:
            news_dataset = json.load(f)
    except Exception as e:
        print(f"Помилка при читанні файлу: {e}")
        return None

    matching_news = []
    for news in news_dataset:
        try:
            published_at = datetime.fromisoformat(news['published_at'].replace('Z', '+00:00'))
            if target_time <= published_at < next_hour:
                summary = news.get('summary', '')
                content = news.get('content', '')
                matching_news.append(f"{summary}\n{content}")
        except Exception as e:
            print(f"Помилка обробки новини: {e}")

    return '\n\n'.join(matching_news) if matching_news else None

def analyze_market_reaction(news_text, gemini_api=GEMINI_API_URL):
    request_payload = {
        "contents": [{
            "parts": [{"text": (
                f"Output only the number of the score. Based on this news: {news_text} "
                "analyze the sentiment and provide a score from 0 to 1, where:\n"
                "0 means extremely, extremely negative,\n"
                "0.1 means very negative,\n"
                "0.9 means very positive, and 0.5 is neutral\n"
                "1 means extremely, extremely positive for Bitcoin.\n"
                "Any value between 0 and 1 is possible to capture the nuances of the sentiment. "
                "Output only the number."
            )}]
        }]
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(gemini_api, headers=headers, data=json.dumps(request_payload))

    if response.status_code == 200:
        result = response.json()
        output_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        if re.fullmatch(r"0(\.\d+)?|1(\.0+)?", output_text):
            return float(output_text)
        else:
            return analyze_market_reaction(news_text)
    else:
        print(response, {response.text})
        return None

def get_previous_score(unix_timestamp, dataset_path='../datasets/BTC_ready.csv'):
    df = pd.read_csv(dataset_path)
    df["Open time"] = pd.to_datetime(df["Open time"], unit='s')
    df["timestamp"] = df["Open time"].astype(int) // 10**9
    df = df[df["timestamp"] < unix_timestamp]
    if df.empty:
        return 0.5
    return df.sort_values("timestamp").iloc[-1]["Score"]

def analyze_news_by_data(unix_timestamp):
    news_items = get_news_for_certain_hour(unix_timestamp)
    if news_items is None:
        return get_previous_score(unix_timestamp)

    score = analyze_market_reaction(news_items)
    if score is None:
        time.sleep(2)
        score = analyze_market_reaction(news_items, gemini_api=f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={GEMINI_API_KEY}")
    if score is None:
        return get_previous_score(unix_timestamp)
    return score

if __name__ == "__main__":
    update_btc_dataset()
