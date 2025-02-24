import requests
import json
import csv
import re
from datetime import datetime, timezone
from collections import defaultdict

GEMINI_API_KEY = "AIzaSyCmfYo2PTrYX9u7stQM2DNlIupoSSLxcsI"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
NEWS_FILE = "news.json"
OUTPUT_CSV = "analyzed_news.csv"
MAX_REQUESTS = 12000


def load_news():
    try:
        with open(NEWS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def analyze_market_reaction(news_text):
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
    response = requests.post(GEMINI_API_URL, headers=headers, data=json.dumps(request_payload))

    if response.status_code == 200:
        result = response.json()
        output_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        if re.fullmatch(r"0(\.\d+)?|1(\.0+)?", output_text):
            return float(output_text)
        else:
            analyze_market_reaction(news_text)
    else:
        return None


def process_news():
    news_data = load_news()
    if not news_data:
        print("Немає новин для обробки.")
        return

    news_by_hour = defaultdict(list)
    for news in news_data:
        dt = datetime.strptime(news["published_at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        unix_timestamp = int(dt.timestamp())
        hour_key = unix_timestamp - (unix_timestamp % 3600)
        news_by_hour[hour_key].append(news)

    all_hours = sorted(news_by_hour.keys())
    start_hour = all_hours[0]
    end_hour = all_hours[-1]
    full_hours = []
    current = start_hour
    while current <= end_hour:
        full_hours.append(current)
        current += 3600

    analyzed_results = []
    last_valid_score = None
    request_count = 0

    for hour in full_hours:
        dt_hour = datetime.fromtimestamp(hour, tz=timezone.utc)
        if hour in news_by_hour:
            combined_text = " ".join(news["summary"] for news in news_by_hour[hour])
            score = analyze_market_reaction(combined_text)
            request_count += 1
            if score is not None:
                last_valid_score = score
            else:
                score = last_valid_score if last_valid_score is not None else 0
            short_text = combined_text.strip()[:40]
            print(f"Година {dt_hour}: score = {score}, новина: {short_text}")
        else:
            print(f"Немає новин для години {dt_hour}, встановлюю коефіцієнт 0.5")
            score = 0.55
            print(f"Година {dt_hour}: score = {score}")

        analyzed_results.append({
            "timestamp": hour,
            "date": dt_hour.strftime("%Y-%m-%d %H:%M:%S"),
            "score": score
        })

        if request_count >= MAX_REQUESTS:
            print("Досягнуто максимальну кількість запитів. Зупинка програми.")
            break

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["timestamp", "date", "score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in analyzed_results:
            writer.writerow(row)

    print("Обробку завершено. Результати збережено у CSV.")


if __name__ == "__main__":
    process_news()
