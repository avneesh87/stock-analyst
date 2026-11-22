from urllib.parse import urlparse


NEWS_PUBLISHERS = {
    "moneycontrol.com": "Moneycontrol",
    "reuters.com": "Reuters",
    "economictimes.indiatimes.com": "Economic Times",
    "livemint.com": "Mint",
    "cnbc.com": "CNBC",
    "cnbctv18.com": "CNBC-TV18",
    "business-standard.com": "Business Standard",
}


def financial_news_publisher(source_url, article_url):
    host = urlparse(source_url or article_url).hostname or ""
    for domain, name in NEWS_PUBLISHERS.items():
        if host == domain or host.endswith("." + domain):
            return name
    return None
