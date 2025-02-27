import os
import requests
import json
import csv
import re
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
NEWS_FILE = "news.json"
OUTPUT_CSV = "analyzed_news.csv"
MAX_REQUESTS = 50


def load_news():
    try:
        with open(NEWS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_news(news_data):
    with open(NEWS_FILE, "w", encoding="utf-8") as file:
        json.dump(news_data, file, ensure_ascii=False, indent=4)


def get_last_processed_hour():
    try:
        with open(OUTPUT_CSV, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            last_row = None
            for row in reader:
                last_row = row
            return int(last_row["timestamp"]) if last_row else None
    except (FileNotFoundError, KeyError, ValueError):
        return None


def analyze_market_reaction(news_text, gemini_api = GEMINI_API_URL):
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
        # print("Error while getting news data")
        # print(response.text)
        return None


def process_news():
    start_time = time.time()
    news_data = load_news()
    if not news_data:
        print("Немає новин для обробки.")
        request_count = MAX_REQUESTS
        return

    last_processed_hour = get_last_processed_hour()
    news_by_hour = defaultdict(list)
    for news in news_data:
        dt = datetime.strptime(news["published_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        unix_timestamp = int(dt.timestamp())
        hour_key = unix_timestamp - (unix_timestamp % 3600) + 3600
        if last_processed_hour is None or hour_key > last_processed_hour:
            news_by_hour[hour_key].append(news)

    if not news_by_hour:
        print("Немає нових новин після останньої обробленої години.")
        return

    analyzed_results = []
    last_valid_score = 0.5
    request_count = 0
    previous_hour_score = last_valid_score

    start_hour = min(news_by_hour.keys(), default=last_processed_hour or 0)
    end_hour = max(news_by_hour.keys(), default=start_hour)

    for hour in range(start_hour, end_hour + 3600, 3600):
        dt_hour = datetime.fromtimestamp(hour, tz=timezone.utc)
        if hour in news_by_hour:
            combined_text = " ".join(news["summary"] for news in news_by_hour[hour])
            score = analyze_market_reaction(combined_text)
            request_count += 1
            if score is not None:
                last_valid_score = score
            else:
                time.sleep(2)
                print("_________________________________")
                score = analyze_market_reaction(combined_text, gemini_api = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={GEMINI_API_KEY}")
        else:
            score = previous_hour_score

        if score is None:
            score = last_valid_score

        previous_hour_score = score

        short_text = combined_text.strip()[:40] if hour in news_by_hour else "Немає новин"
        print(f"Година {dt_hour}: score = {score}, новина: {short_text}")

        analyzed_results.append({
            "timestamp": hour,
            "date": dt_hour.strftime("%Y-%m-%d %H:%M:%S"),
            "score": score
        })

        if request_count >= MAX_REQUESTS:
            print("Досягнуто максимальну кількість запитів. Зупинка програми.")
            break

    with open(OUTPUT_CSV, "a", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["timestamp", "date", "score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        for row in analyzed_results:
            writer.writerow(row)

    end_time = time.time()
    execution_time = (end_time - start_time) / 60
    print(f"Обробку завершено. Результати збережено у CSV. \nЧас виконання: {execution_time:.2f} хвилин.")


if __name__ == "__main__":
    process_news()
