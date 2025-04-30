import os
import requests
import re
import json

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key={GEMINI_API_KEY}"

def analyze_market_reaction(news_text):
    request_payload = {
        "contents": [{
            "parts": [{"text": (
                    f"Output only the number of the score. Based on this news: {news_text} "
                    "analyze the sentiment and provide a score from 0 to 1, where:\n"
                    "0 means extremely, extremely negative,\n"
                    "0.1 means very negative,\n"
                    "0.9 means very positive, and 0.5 is neutral\n"
                    "1 means extremely, extremely positive for Bitcoin growth.\n"
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
            return output_text
        else:
            return "Invalid response format"
    else:
        return f"Error: {response.status_code}, {response.text}"


news_samples = [
    "Bitcoin hits a new all-time high, surpassing $100,000 for the first time in history.",
    "Regulatory concerns emerge as the SEC hints at stricter rules for cryptocurrency exchanges.",
    "A major institutional investor announces a $10 billion Bitcoin purchase, signaling strong market confidence.",
    "Hackers steal $500 million worth of Bitcoin from a major exchange, shaking investor trust.",
    "El Salvador announces further Bitcoin adoption plans, making it a key part of its financial system.",
    "IMF warns that cryptocurrencies could destabilize global financial markets.",
    "A leading financial institution declares Bitcoin as the best-performing asset of the decade.",
    "New Bitcoin ETF receives approval, opening doors for institutional investors.",
    "China reiterates its ban on Bitcoin mining, causing a temporary price dip.",
    "Ethereum's latest update sparks concerns about its impact on Bitcoin's market dominance.",
    "Elon Musk stole the whole 1 trillion bitcoin supply and now Bitcoin is about to crash as the whole ecosystem."
]

for i, news_text in enumerate(news_samples, start=1):
    result = analyze_market_reaction(news_text)
    print(f"News {i}:\nScore: {result}\n")
