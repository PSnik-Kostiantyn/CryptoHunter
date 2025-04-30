# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html
import json
import os
import re

# useful for handling different item types with a single interface
from itemadapter import ItemAdapter

from .utils import remove_html_tags


class NewsPipeline:
    def __init__(self):
        self.items = []

    def open_spider(self, spider):
        if os.path.exists(spider.dataset_path):
            with open(spider.dataset_path, 'r', encoding='utf-8') as f:
                try:
                    self.items = json.load(f)
                except json.JSONDecodeError:
                    self.items = []

    def close_spider(self, spider):
        with open(spider.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        summary_value = adapter.get("summary")
        content_value = adapter.get("content")
        item["summary"] = remove_html_tags(summary_value).strip()
        item["content"] = remove_html_tags(content_value).strip()
        self.items.append(dict(item))
        return item
