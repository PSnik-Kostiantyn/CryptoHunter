# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import re

# useful for handling different item types with a single interface
from itemadapter import ItemAdapter

from .utils import remove_html_tags


class NewsScorePipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        message = adapter.get("message")
        numbers = re.findall(r'0\.(?:\d{1,2})\b', message)
        item["score"] = float(numbers[0]) if numbers else None
        return item


class NewsPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        summary_value = adapter.get("summary")
        content_value = adapter.get("content")
        item["summary"] = remove_html_tags(summary_value).strip()
        item["content"] = remove_html_tags(content_value).strip()
        return item
