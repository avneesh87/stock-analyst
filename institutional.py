from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
import math
from urllib.parse import quote
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import pandas as pd
import requests

from analytics import StockDataError, get_nifty50_universe


NIFTY500_URLS = (
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
)


@dataclass
class InstitutionalReport:
    stocks: pd.DataFrame
    universe: str
    total: int
    checked_at: datetime
    warnings: list[str]


def parse_shareholding(html: str, today: date) -> dict:
    table = BeautifulSoup(html, "html.parser").select_one("#quarterly-shp table")
    if table is None:
        raise StockDataError("Quarterly institutional holdings are unavailable.")
    periods = []
    for cell in table.select("thead th")[1:]:
        try:
            period = datetime.strptime(cell.get_text(strip=True), "%b %Y").date()
        except ValueError as exc:
            raise StockDataError("Unrecognized shareholding reporting period.") from exc
        periods.append(period)
    if len(periods) < 2 or periods != sorted(set(periods)):
        raise StockDataError("Two ordered quarterly holdings are required.")
    previous, latest = periods[-2:]
    gap = (latest.year - previous.year) * 12 + latest.month - previous.month
    if (gap != 3 or latest.month not in (3, 6, 9, 12)
            or latest >= today.replace(day=1) or (today - latest).days > 180):
        raise StockDataError("Recent consecutive completed quarters are unavailable.")
    holdings = {}
    for row in table.select("tbody tr"):
        cells = row.find_all("td", recursive=False)
        if not cells:
            continue
        label = cells[0].get_text(" ", strip=True).replace("+", "").strip()
        if label not in ("FIIs", "DIIs"):
            continue
        if len(cells) != len(periods) + 1:
            raise StockDataError("Institutional holdings have missing reporting columns.")
        try:
            values = [float(cell.get_text(strip=True).removesuffix("%")) for cell in cells[-2:]]
        except ValueError as exc:
            raise StockDataError("Institutional holding percentages are missing.") from exc
        if not all(math.isfinite(value) and 0 <= value <= 100 for value in values):
            raise StockDataError("Invalid institutional holding percentages.")
        group = "FII" if label == "FIIs" else "DII"
        holdings[f"{group} previous (%)"] = values[0]
        holdings[f"{group} latest (%)"] = values[1]
        holdings[f"{group} change (pp)"] = round(values[1] - values[0], 4)
    if "FII change (pp)" not in holdings or "DII change (pp)" not in holdings:
        raise StockDataError("Both FII and DII holdings are required.")
    return {"Previous quarter": previous, "Latest quarter": latest, **holdings}


def rank_institutional_stocks(stocks: pd.DataFrame, mode: str = "FII+DII") -> pd.DataFrame:
    if stocks.empty:
        return stocks.copy()
    ranked = stocks[stocks["Latest quarter"] == stocks["Latest quarter"].max()].copy()
    ranked["Combined change (pp)"] = (
        ranked["FII change (pp)"] + ranked["DII change (pp)"]
    ).round(4)
    if mode in ("FII", "DII"):
        metric = f"{mode} change (pp)"
        ranked = ranked[ranked[metric] > 0]
    else:
        metric = "Combined change (pp)"
        ranked = ranked[(ranked["FII change (pp)"] > 0) & (ranked["DII change (pp)"] > 0)]
    ranked = ranked.sort_values([metric, "Symbol"], ascending=[False, True]).head(20)
    ranked.insert(0, "Rank", range(1, len(ranked) + 1))
    return ranked.reset_index(drop=True)


def _company_holdings(company: dict, today: date) -> dict | None:
    symbol = str(company["Symbol"]).strip()
    url = f"https://www.screener.in/company/{quote(symbol, safe='')}/"
    try:
        response = requests.get(url, timeout=12)
        response.raise_for_status()
        holdings = parse_shareholding(response.text, today)
    except (requests.RequestException, StockDataError):
        return None
    return {
        "Symbol": symbol, "Company": company["Company Name"],
        **holdings, "Source": url + "#shareholding",
    }


def screen_institutional_buying() -> InstitutionalReport:
    warnings = []
    universe = None
    for url in NIFTY500_URLS:
        try:
            response = requests.get(url, timeout=12)
            response.raise_for_status()
            candidate = pd.read_csv(StringIO(response.text))
            candidate = candidate[["Symbol", "Company Name"]].dropna().drop_duplicates("Symbol")
            if len(candidate) < 400:
                continue
            universe = candidate
            break
        except (requests.RequestException, ValueError, KeyError):
            continue
    source = "Nifty 500 (official constituent list)"
    if universe is None:
        universe, origin = get_nifty50_universe()
        source = f"Nifty 50 fallback ({origin})"
        warnings.append("Nifty 500 constituents were unavailable. Coverage is limited to Nifty 50, not the whole market.")
    checked_at = datetime.now(ZoneInfo("Asia/Kolkata"))
    records = universe.to_dict("records")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(_company_holdings, records, [checked_at.date()] * len(records)))
    stocks = pd.DataFrame([result for result in results if result is not None])
    missing = len(records) - len(stocks)
    if missing:
        warnings.append(f"{missing} of {len(records)} companies had unavailable, incomplete, or stale holdings. Rankings cover successful responses only.")
    if not stocks.empty:
        latest_count = int((stocks["Latest quarter"] == stocks["Latest quarter"].max()).sum())
        if latest_count != len(stocks):
            warnings.append(f"Only {latest_count} companies have the newest available quarter. Older quarters are excluded from rankings; filings may still be arriving.")
    return InstitutionalReport(stocks, source, len(records), checked_at, warnings)