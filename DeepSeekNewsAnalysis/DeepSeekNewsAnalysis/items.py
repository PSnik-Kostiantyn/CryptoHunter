# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class NewsItem(scrapy.Item):
    published_at = scrapy.Field()
    title = scrapy.Field()
    summary = scrapy.Field()
    content = scrapy.Field()


class NewsScoreItem(scrapy.Item):
    score = scrapy.Field()
    message = scrapy.Field()
    news_item = scrapy.Field()  # Should be NewsItem
