## Prerequisites

- [Poetry](https://python-poetry.org/)
- python >= 3.12

## Usage

Install all dependencies using poetry:

```bash
    poetry install
```

To run spiders execute next command:

```bash
    poetry run scrapy crawl <spider_name> -O <filename>
```

Available spiders:

- news: Scrape all news from 2023-01-01 to 2025-01-01.
- deepseek: Use openrouter for sentiment analysis of news (currently not working).

Filename can be next: `data.json`. If you wish to get a csv format, use next params when running spider:
`-O data.csv -t csv`
