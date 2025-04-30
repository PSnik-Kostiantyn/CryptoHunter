from datetime import datetime, UTC
from time import strptime
from typing import Any
from urllib.parse import urlparse, parse_qs

import pandas as pd
import scrapy
from scrapy import Request
from scrapy.http import Response

from ..items import NewsItem
from ..utils import get_iso_date


class NewsSpider(scrapy.Spider):
    name = "news"
    allowed_domains = ["coinfi.com"]
    base_url = "https://www.coinfi.com/api/news?coinSlugs%5B%5D=bitcoin&trending=&"
    custom_settings = {
        "ITEM_PIPELINES": {
            "NewsScrapper.pipelines.NewsPipeline": 300
        }
    }

    def __init__(self, *args, dataset_path="../datasets/news.json", **kwargs) -> None:
        super().__init__(*args, dataset_path=dataset_path, **kwargs)
        df = pd.read_json(dataset_path)
        self.start_urls = [
            f"{self.base_url}publishedSince={df['published_at'].max().to_pydatetime().strftime("%Y-%m-%dT%H:%M:%SZ")}&publishedUntil={datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}"
        ]

    def parse(self, response: Response, **kwargs: Any):
        data = response.json()
        url = response.url
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)

        published_since = get_iso_date(query_params.get('publishedSince')[0])
        published_until = get_iso_date(query_params.get('publishedUntil')[0])

        if published_since >= published_until:
            return

        latest_date = None

        for news_piece in data.get("payload"):
            news_item = NewsItem()
            latest_date = get_iso_date(news_piece.get("feed_item_published_at"))
            news_item["published_at"] = latest_date.strftime("%Y-%m-%dT%H:%M:%SZ")
            news_item["title"] = news_piece.get("title")
            news_item["summary"] = news_piece.get("summary")
            news_item["content"] = news_piece.get("content")
            yield news_item

        if latest_date is not None:
            yield Request(
                url=f"{self.base_url}publishedSince={published_since.strftime("%Y-%m-%dT%H:%M:%SZ")}&publishedUntil={latest_date.strftime("%Y-%m-%dT%H:%M:%SZ")}",
                callback=self.parse
            )
