import re

import requests
from bs4 import BeautifulSoup

from analytics import StockDataError


GROWW_ETF_URLS = {
    "TATAGOLD.NS": "https://groww.in/etfs/tata-gold-exchange-traded-fund",
    "GOLDBEES.NS": "https://groww.in/etfs/reliance-etf-gold-bees",
    "SILVERBEES.NS": (
        "https://groww.in/etfs/"
        "nippon-life-india-asset-management-ltd-nippon-india-silver-etf"
    ),
}


def parse_groww_etf_details(html, symbol):
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find(["h1"])
    text = " ".join(soup.get_text(" ", strip=True).split())
    identity = text.split("Fundamentals", 1)[0]
    if heading is None or not re.search(
        rf"\b{re.escape(symbol.removesuffix('.NS'))}\b", identity
    ):
        raise StockDataError("Groww did not return the requested ETF page.")
    section = re.search(r"\bFundamentals\b(.*?)(?:Category returns|About|Similar ETFs|$)", text)
    if section is None:
        raise StockDataError("Groww ETF fundamentals are unavailable.")
    text = section.group(1)
    metrics = {}
    for label in ("Expense ratio", "Tracking error"):
        match = re.search(rf"{label}\s+([\d.]+)\s*%", text)
        if match:
            metrics[label] = f"{float(match.group(1)):.2f}%"
    match = re.search(r"AUM\s+₹\s*([\d,.]+)\s*Cr\b", text)
    if match:
        metrics["AUM"] = f"₹{float(match.group(1).replace(',', '')):,.2f} Cr"
    match = re.search(r"Avg\.\s*volume\s*\(1M\)\s+([\d,]+)\b", text)
    if match:
        metrics["Avg. volume"] = (
            f"{float(match.group(1).replace(',', '')) / 100_000:,.2f} Lac"
        )
    if not metrics:
        raise StockDataError("Groww did not provide usable ETF fundamentals.")
    return metrics


def fetch_groww_etf_details(symbol):
    url = GROWW_ETF_URLS[symbol]
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return parse_groww_etf_details(response.text, symbol), url
