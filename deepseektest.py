from openai import OpenAI
import re


def analyze_market_reaction(news_text):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

    while True:
        request = f"<important>Output only the number of the score.</important>. Based on this news {news_text} analyze the reaction and provide a score from 0 to 1 where 0 is extremely negative for Bitcoin and 1 is extremely positive. <important>Output only the number</important>"

        response = client.chat.completions.create(
            model="deepseek/deepseek-r1-distill-llama-70b:free",
            messages=[
                {"role": "user", "content": request}
            ]
        )

        result = response.choices[0].message.content.strip()
        if re.fullmatch(r"0(\.\d+)?|1(\.0+)?", result):
            return result

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
    "Elon Musk stole the whole 1 trillion bitcoin supply and now Bitcoin is about to crush as the whole eco system"
]

for i, news_text in enumerate(news_samples, start=1):
    result = analyze_market_reaction(news_text)
    print(f"News {i}:\nScore: {result}\n")
