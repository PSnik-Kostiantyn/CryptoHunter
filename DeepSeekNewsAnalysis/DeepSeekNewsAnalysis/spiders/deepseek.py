import json
from typing import Any, Iterable

import scrapy
from scrapy import Request
from scrapy.http import Response
from scrapy.utils.project import get_project_settings

from ..items import NewsScoreItem
from ..utils import retry_on_exception

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


class DeepseekSpider(scrapy.Spider):
    name = "deepseek"
    allowed_domains = ["openrouter.ai"]

    custom_settings = {
        "ITEM_PIPELINES": {
            "DeepSeekNewsAnalysis.pipelines.NewsScorePipeline": 300
        }
    }

    def start_requests(self) -> Iterable[Request]:
        base_url = "https://openrouter.ai/api/v1/chat/completions"
        for news_piece in news_samples:
            prompt = get_project_settings().get("DEEPSEEK_PROMPT")(news_piece)
            yield Request(method="POST", url=base_url, callback=self.parse, body=json.dumps(dict(
                model="deepseek/deepseek-r1-distill-llama-70b:free",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )), meta={"news_piece": news_piece})

    @retry_on_exception(10)
    def parse(self, response: Response, **kwargs: Any):
        news_score = NewsScoreItem()
        news_score["message"] = response.json().get('choices')[0].get("message").get("content")
        news_score["news_item"] = response.meta.get("news_piece")
        yield news_score
