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
NIFTY500_MIRROR_URL = (
    "https://raw.githubusercontent.com/pkjmesra/PKScreener/main/"
    "results/Indices/ind_nifty500list.csv"
)
NIFTY500_MIRROR_HISTORY_URL = (
    "https://api.github.com/repos/pkjmesra/PKScreener/commits"
    "?path=results%2FIndices%2Find_nifty500list.csv&per_page=1"
)
NIFTY500_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/csv,*/*",
    "Referer": "https://www.nseindia.com/",
}


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


def _company_holdings(company: dict, today: date, session: requests.Session) -> dict | None:
    symbol = str(company["Symbol"]).strip()
    url = f"https://www.screener.in/company/{quote(symbol, safe='')}/"
    try:
        response = session.get(url, timeout=12)
        response.raise_for_status()
        holdings = parse_shareholding(response.text, today)
    except (requests.RequestException, StockDataError):
        return None
    return {
        "Symbol": symbol, "Company": company["Company Name"],
        **holdings, "Source": url + "#shareholding",
    }


def _scan_holdings_batch(companies: list[dict], today: date) -> list[dict | None]:
    with requests.Session() as session:
        return [_company_holdings(company, today, session) for company in companies]


def _nifty500_candidate(url: str) -> pd.DataFrame:
    response = requests.get(url, headers=NIFTY500_HEADERS, timeout=12)
    response.raise_for_status()
    candidate = pd.read_csv(StringIO(response.text), dtype=str)
    candidate = candidate[["Symbol", "Company Name"]].dropna().copy()
    for column in ("Symbol", "Company Name"):
        candidate[column] = candidate[column].str.strip()
    candidate["Symbol"] = candidate["Symbol"].str.upper()
    candidate = candidate[
        (candidate["Symbol"] != "") & (candidate["Company Name"] != "")
    ].drop_duplicates("Symbol")
    if not 400 <= len(candidate) <= 550:
        raise StockDataError(f"Nifty 500 list has an unexpected number of constituents: {len(candidate)}.")
    return candidate


def _mirror_last_updated() -> str:
    try:
        response = requests.get(NIFTY500_MIRROR_HISTORY_URL, timeout=12)
        response.raise_for_status()
        commits = response.json()
        timestamp = commits[0]["commit"]["committer"]["date"]
        if not isinstance(timestamp, str):
            raise ValueError("Invalid mirror update timestamp")
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).strftime("%d %b %Y")
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return "unavailable"


def screen_institutional_buying() -> InstitutionalReport:
    warnings = []
    universe = None
    source = "Nifty 500 (official constituent list)"
    for url in NIFTY500_URLS:
        try:
            universe = _nifty500_candidate(url)
            break
        except (requests.RequestException, ValueError, KeyError, StockDataError):
            continue
    if universe is None:
        try:
            universe = _nifty500_candidate(NIFTY500_MIRROR_URL)
        except (requests.RequestException, ValueError, KeyError, StockDataError):
            pass
        else:
            source = "Nifty 500 (unofficial PKScreener GitHub mirror)"
            warnings.append(
                "Official Nifty 500 lists were unavailable. The unofficial mirror may be "
                "outdated or inaccurate; its last file update is "
                f"{_mirror_last_updated()}. Current index membership is not verified."
            )
    if universe is None:
        universe, origin = get_nifty50_universe()
        source = f"Nifty 50 fallback ({origin})"
        warnings.append("Nifty 500 constituents were unavailable. Coverage is limited to Nifty 50, not the whole market.")
    checked_at = datetime.now(ZoneInfo("Asia/Kolkata"))
    records = universe.to_dict("records")
    batches = [records[start:start + 20] for start in range(0, len(records), 20)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        batch_results = executor.map(_scan_holdings_batch, batches, [checked_at.date()] * len(batches))
        results = [result for batch in batch_results for result in batch]
    stocks = pd.DataFrame([result for result in results if result is not None])
    missing = len(records) - len(stocks)
    if missing:
        warnings.append(f"{missing} of {len(records)} companies had unavailable, incomplete, or stale holdings. Rankings cover successful responses only.")
    if not stocks.empty:
        latest_count = int((stocks["Latest quarter"] == stocks["Latest quarter"].max()).sum())
        if latest_count != len(stocks):
            warnings.append(f"Only {latest_count} companies have the newest available quarter. Older quarters are excluded from rankings; filings may still be arriving.")
    return InstitutionalReport(stocks, source, len(records), checked_at, warnings)