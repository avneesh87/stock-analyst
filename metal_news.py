from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import re

import feedparser
import requests

from analytics import StockDataError
from news_sources import NEWS_PUBLISHERS, financial_news_publisher


def potential_price_drivers(title):
    """Tag headline topics, not confirmed causes or direction of price moves."""
    topics = (
        ("Rates / inflation", r"\b(fed|federal reserve|rbi|repo rates?|interest rates?|yields?|inflation|cpi)\b"),
        ("USD / currency", r"\b(dollar|usd|rupee|inr|currency|forex)\b"),
        ("Geopolitics / trade", r"\b(war|conflict|tariffs?|sanctions?|geopolitic\w*)\b"),
        ("Demand / supply", r"\b(demand|supply|imports?|mining|industrial|solar|jewel\w*)\b"),
        ("Indian taxes / policy", r"\b(gst|dut(y|ies)|budget|rbi)\b"),
    )
    return [label for label, pattern in topics if re.search(pattern, title, re.IGNORECASE)]


def is_gold_silver_market_news(title):
    if not re.search(r"\b(gold|silver)\b", title, re.IGNORECASE):
        return False
    if re.search(r"\b(silver lining|gold medals?|silver medals?)\b", title, re.IGNORECASE):
        return False
    return bool(re.search(
        r"\b(prices?|rates?|futures?|bullion|etfs?|ounces?|demand|supply|"
        r"imports?|mining|jewel\w*|invest\w*|trading|mcx|comex|"
        r"rall\w*|climb\w*|ris\w*|fall\w*|drop\w*|surge\w*|slump\w*)\b",
        title, re.IGNORECASE,
    ))


def is_central_bank_rate_news(title):
    bank = re.search(
        r"\b(fed|federal reserve|rbi|reserve bank of india)\b", title, re.IGNORECASE
    )
    rate = re.search(r"\b(rates?|repo|monetary policy)\b", title, re.IGNORECASE)
    decision = re.search(
        r"\b(cut\w*|hik\w*|rais\w*|lower\w*|hold\w*|held|unchanged|"
        r"pause\w*|decision\w*|reduc\w*|eas\w*|tighten\w*)\b",
        title, re.IGNORECASE,
    )
    return bool(bank and rate and decision)


def parse_metal_news(content, now):
    feed = feedparser.parse(content)
    if feed.bozo and not feed.entries:
        raise StockDataError("The metals news feed could not be parsed.")
    items = []
    skipped = 0
    seen = set()
    for entry in feed.entries:
        title = entry.get("title", "").strip()
        link = entry.get("link", "")
        try:
            published = parsedate_to_datetime(entry.get("published", ""))
        except (TypeError, ValueError):
            skipped += 1
            continue
        if published.tzinfo is None:
            skipped += 1
            continue
        if not now - timedelta(days=7) <= published <= now:
            continue
        if not title or not link.startswith(("https://", "http://")):
            skipped += 1
            continue
        source_url = entry.get("source", {}).get("href", "")
        publisher = financial_news_publisher(source_url, link)
        if publisher is None:
            continue
        rate_news = is_central_bank_rate_news(title)
        if not is_gold_silver_market_news(title) and not rate_news:
            continue
        if link in seen:
            continue
        seen.add(link)
        items.append({
            "title": title,
            "link": link,
            "published": published,
            "publisher": publisher,
            "drivers": potential_price_drivers(title),
            "rate_decision": rate_news,
        })
    items.sort(key=lambda item: (
        not item["rate_decision"],
        item["publisher"] != "Moneycontrol", -item["published"].timestamp(),
    ))
    return {"items": items[:8], "fetched_at": now, "skipped": skipped}


def fetch_metal_news(currency):
    queries = {
        "USD": '(intitle:gold OR intitle:silver) (prices OR futures) (Fed OR dollar OR inflation OR demand OR war) when:7d',
        "INR": '(intitle:gold OR intitle:silver) (India OR rupee OR MCX) (prices OR demand OR duty) when:7d',
    }
    if currency not in queries:
        raise StockDataError(f"Unsupported metals-news currency: {currency}")
    rate_query = (
        '(intitle:Fed OR intitle:"Federal Reserve" OR intitle:RBI OR '
        'intitle:"Reserve Bank of India") '
        '("rate cut" OR "rate hike" OR "rates unchanged" OR "holds rates" OR "repo rate decision")'
    )
    publisher_query = "(" + " OR ".join(f"site:{domain}" for domain in NEWS_PUBLISHERS) + ")"
    try:
        response = requests.get(
            "https://news.google.com/rss/search",
            params={
                "q": f"(({queries[currency]}) OR ({rate_query})) when:7d {publisher_query}",
                "hl": "en-IN", "gl": "IN", "ceid": "IN:en",
            },
            timeout=15,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise StockDataError(f"Could not retrieve {currency} metals news: {exc}") from exc
    return parse_metal_news(response.content, datetime.now(timezone.utc))
