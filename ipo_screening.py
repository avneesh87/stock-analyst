from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
import re
from zoneinfo import ZoneInfo
from urllib.parse import urlparse

from bs4 import BeautifulSoup
import pandas as pd
import requests

from analytics import StockDataError


IPO_LIST_URL = "https://webnodejs.chittorgarh.com/cloud/index/ipo-read/mainline"
IPO_SOURCE_URL = (
    "https://www.chittorgarh.com/report/ipo-in-india-list-main-board-sme/82/mainboard/"
)
INDIA_TIME = ZoneInfo("Asia/Kolkata")


@dataclass
class IPOAssessment:
    verdict: str
    strengths: list[str]
    risks: list[str]
    metrics: dict[str, float | None]


@dataclass
class GMPQuote:
    premium: float | None
    quoted_at: str
    source_url: str
    note: str = ""
    percentage: float | None = None


@dataclass
class OpenIPO:
    name: str
    url: str
    opens: date | None
    closes: date | None
    price_band: str
    lot_size: str
    sale_type: str
    financials: pd.DataFrame
    assessment: IPOAssessment
    status: str = "Open"
    sector: str = "Unavailable"
    sector_peer_pe: float | None = None
    sector_peer_count: int = 0
    gmp: GMPQuote | None = None
    gmp_url: str | None = None
    allotment_date: date | None = None


@dataclass
class IPOReport:
    issues: list[OpenIPO]
    warnings: list[str]
    checked_at: datetime


def _number(text: str) -> float | None:
    text = text.strip().replace(",", "").replace("\u2212", "-").removesuffix("%")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?|\(\d+(?:\.\d+)?\)", text):
        return None
    negative = text.startswith("(") and text.endswith(")")
    value = float(text.strip("()"))
    if negative:
        value = -value
    return value if math.isfinite(value) else None


def _period(text: str) -> date | None:
    for pattern in (
        "%d %b %Y", "%b %d, %Y", "%d %B %Y",
        "%a, %b %d, %Y", "%A, %B %d, %Y",
    ):
        try:
            return datetime.strptime(text.strip(), pattern).date()
        except ValueError:
            continue
    return None


def _subscription_dates(
    opening: str, closing: str, today: date, *, upcoming: bool = False,
) -> tuple[date, date] | None:
    # Feed dates omit the year; try adjacent years to handle December/January.
    for year in (today.year - 1, today.year, today.year + 1):
        start = _period(f"{opening} {year}")
        end = _period(f"{closing} {year}")
        if start is None or end is None:
            continue
        if end < start:
            end = _period(f"{closing} {year + 1}")
        if end is None:
            continue
        eligible = 0 < (start - today).days <= 183 if upcoming else start <= today <= end
        if eligible and (end - start).days <= 14:
            return start, end
    return None


def assess_ipo_fundamentals(
    financials: pd.DataFrame,
    market_cap_crore: float | None,
    *,
    sector: str | None = None,
    peer_pe: float | None = None,
) -> IPOAssessment:
    strengths: list[str] = []
    risks: list[str] = []
    metrics: dict[str, float | None] = {
        "Income growth %": None,
        "PAT growth %": None,
        "PAT margin %": None,
        "ROE proxy %": None,
        "Debt / equity": None,
        "Annual P/E": None,
    }
    annual = [
        column for column in financials.columns
        if isinstance(column, date) and (column.month, column.day) == (3, 31)
    ]
    annual.sort(reverse=True)
    if not annual:
        return IPOAssessment(
            "CAUTION", [], ["Two consecutive March year-end financials are required."],
            metrics,
        )

    latest = annual[0]
    previous = annual[1] if len(annual) > 1 and latest.year - annual[1].year == 1 else None
    if previous is None:
        risks.append("Two consecutive March year-end financials are required for growth checks.")

    def value(label: str, period: date | None) -> float | None:
        if period is None or label not in financials.index:
            return None
        raw = financials.at[label, period]
        if pd.isna(raw):
            return None
        return float(raw)

    income = value("Total Income", latest)
    old_income = value("Total Income", previous)
    profit = value("Profit After Tax", latest)
    old_profit = value("Profit After Tax", previous)
    equity = value("NET Worth", latest)
    debt = value("Total Borrowing", latest)
    if income is not None and old_income is not None and old_income > 0:
        metrics["Income growth %"] = (income / old_income - 1) * 100
    if profit is not None and old_profit is not None and old_profit > 0:
        metrics["PAT growth %"] = (profit / old_profit - 1) * 100
    if income is not None and income > 0 and profit is not None:
        metrics["PAT margin %"] = profit / income * 100
    if equity is not None and equity > 0:
        if profit is not None:
            metrics["ROE proxy %"] = profit / equity * 100
        if debt is not None and debt >= 0:
            metrics["Debt / equity"] = debt / equity
    if (
        market_cap_crore is not None and market_cap_crore > 0
        and profit is not None and profit > 0
    ):
        metrics["Annual P/E"] = market_cap_crore / profit

    financial_sector = sector is not None and any(
        word in sector.lower()
        for word in ("bank", "finance", "financial", "insurance", "asset management", "nbfc")
    )
    checks = (
        ("Income growth %", lambda v: v > 0, "Annual income is growing.", "Annual income is flat or declining."),
        ("PAT growth %", lambda v: v > 0, "Annual profit is growing.", "Annual profit is flat or declining."),
        ("PAT margin %", lambda v: v >= 10, "PAT margin is at least 10%.", "PAT margin is below 10%."),
        ("ROE proxy %", lambda v: v >= 15, "ROE proxy is at least 15%.", "ROE proxy is below 15%."),
        ("Debt / equity", lambda v: v <= 0.5, "Debt/equity is at most 0.5x.", "Debt/equity exceeds 0.5x."),
        (
            "Annual P/E",
            lambda v: 0 < v <= (peer_pe if peer_pe is not None else 25),
            "Annual P/E is at or below the recent mainboard sector IPO peer median."
            if peer_pe is not None else "Annual P/E is at most 25x.",
            "Annual P/E exceeds the recent mainboard sector IPO peer median."
            if peer_pe is not None else "Annual P/E exceeds 25x.",
        ),
    )
    failed = 0
    missing = False
    for label, passes, good, bad in checks:
        if financial_sector and label == "Debt / equity":
            risks.append(
                "Financial-sector business: the generic debt/equity threshold is "
                "not used. Sector-specific capital, asset quality or fee-income "
                "metrics need separate review."
            )
            missing = True
            continue
        metric = metrics[label]
        if metric is None:
            risks.append(f"{label} cannot be verified from available data.")
            missing = True
        elif passes(metric):
            strengths.append(good)
        else:
            risks.append(bad)
            failed += 1

    loss = profit is not None and profit <= 0
    nonpositive_equity = equity is not None and equity <= 0
    latest_period = max(financials.columns)
    interim_profit = value("Profit After Tax", latest_period)
    interim_loss = latest_period > latest and interim_profit is not None and interim_profit < 0
    interim_missing = latest_period > latest and interim_profit is None
    if loss:
        risks.append("Latest full-year profit is non-positive.")
    if nonpositive_equity:
        risks.append("Latest full-year net worth is non-positive.")
    if interim_loss:
        risks.append("The more recent interim period reports a loss; it is not annualized.")
    if interim_missing:
        risks.append("Profit for the more recent interim period could not be verified.")

    if loss or nonpositive_equity or failed >= 3:
        verdict = "AVOID"
    elif missing or failed or interim_loss or interim_missing:
        verdict = "CAUTION"
    else:
        verdict = "APPLY CANDIDATE"
    return IPOAssessment(verdict, strengths, risks, metrics)


def _sector_context(soup: BeautifulSoup) -> tuple[str, float | None, int]:
    for heading in soup.find_all(["h2", "h3"]):
        text = heading.get_text(" ", strip=True)
        prefix = "Recently Listed IPOs in "
        if not text.startswith(prefix):
            continue
        sector = text[len(prefix):].strip()
        table = heading.parent.find("table")
        if table is None:
            return sector, None, 0
        rows = [
            [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
            for row in table.find_all("tr")
        ]
        if not rows or "PE Ratio" not in rows[0] or "Issue Type" not in rows[0]:
            return sector, None, 0
        pe_index = rows[0].index("PE Ratio")
        type_index = rows[0].index("Issue Type")
        values = []
        for row in rows[1:]:
            if len(row) <= max(pe_index, type_index) or row[type_index] != "Mainboard":
                continue
            value = _number(row[pe_index])
            if value is not None and value > 0:
                values.append(value)
        return sector, float(pd.Series(values).median()) if values else None, len(values)
    return "Unavailable", None, 0


def parse_gmp_quote(html: str, source_url: str) -> GMPQuote:
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        rows = [
            [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
            for row in table.find_all("tr")
        ]
        if not rows or "GMP Date" not in rows[0] or "GMP" not in rows[0]:
            continue
        date_index, gmp_index = rows[0].index("GMP Date"), rows[0].index("GMP")
        if len(rows) < 2 or len(rows[1]) <= max(date_index, gmp_index):
            break
        raw = rows[1][gmp_index].replace("\u2212", "-")
        match = re.match(r"^\s*([+-]?)\s*(?:\u20b9|INR|Rs\.?)\s*([+-]?\d[\d,]*(?:\.\d+)?)", raw)
        if match:
            value = _number(match.group(2))
            if value is not None:
                if match.group(1) == "-":
                    value = -abs(value)
                percent_match = re.search(r"\(\s*([+-]?\d+(?:\.\d+)?)\s*%\s*\)", raw)
                percentage = _number(percent_match.group(1)) if percent_match else None
                return GMPQuote(
                    value, rows[1][date_index], source_url,
                    percentage=percentage,
                )
        break
    return GMPQuote(
        None, "Unavailable", source_url,
        "No readable GMP quote was found; the source may not have published a quote or its layout changed.",
    )


def parse_ipo_detail(
    html: str, name: str, url: str, opens: date | None, closes: date | None,
    *, reference_date: date | None = None,
) -> OpenIPO:
    reference_date = reference_date or opens or datetime.now(INDIA_TIME).date()
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="financialTable")
    if table is None:
        raise StockDataError("Company financial table is missing or the source layout changed.")
    rows = [
        [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        for row in table.find_all("tr")
    ]
    if not rows or not rows[0] or rows[0][0] != "Period Ended":
        raise StockDataError("Financial period headers could not be recognized.")
    parsed_periods = [_period(text) for text in rows[0][1:]]
    if not parsed_periods or any(period is None for period in parsed_periods):
        raise StockDataError("Financial period dates could not be parsed.")
    periods = [period for period in parsed_periods if period is not None]
    if len(set(periods)) != len(periods):
        raise StockDataError("Financial period dates are duplicated.")
    if max(periods) > reference_date:
        raise StockDataError("Reported financial periods are later than the assessment date.")
    if "crore" not in table.get_text(" ", strip=True).lower():
        raise StockDataError("Financial units are not confirmed as INR crore.")

    values = {}
    labels = {"Total Income", "Profit After Tax", "NET Worth", "Total Borrowing"}
    for row in rows[1:]:
        if row and row[0] in labels:
            if len(row) != len(periods) + 1:
                raise StockDataError(f"Financial values are misaligned for {row[0]}.")
            values[row[0]] = [_number(text) for text in row[1:]]
    financials = pd.DataFrame.from_dict(values, orient="index", columns=periods)
    if financials.empty:
        raise StockDataError("No recognized company financial rows were found.")

    details = {}
    market_cap = None
    for source_table in soup.find_all("table"):
        for row in source_table.find_all("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
            if len(cells) == 2:
                details[cells[0]] = cells[1]
            if len(cells) == 3 and cells[0] == "Market Cap at Offer Price":
                match = re.fullmatch(r"[^\d]*([\d,]+(?:\.\d+)?)\s*Cr\.?", cells[2])
                if match:
                    market_cap = _number(match.group(1))

    listing = details.get("Listing At", "")
    if "SME" in listing.upper() or not re.search(r"\b(BSE|NSE)\b", listing):
        raise StockDataError("Mainboard listing could not be verified on the detail page.")
    allotment_date = None
    for label in ("Basis of Allotment", "Allotment Date", "Allotment"):
        allotment_date = _period(details.get(label, ""))
        if allotment_date is not None:
            break
    if allotment_date is None:
        for row in soup.select("li"):
            cells = row.find_all("span", recursive=False)
            if len(cells) == 2 and cells[0].get_text(" ", strip=True) == "Allotment":
                allotment_date = _period(cells[1].get_text(" ", strip=True))
                if allotment_date is not None:
                    break
    sector, peer_pe, peer_count = _sector_context(soup)
    assessment = assess_ipo_fundamentals(
        financials, market_cap, sector=sector, peer_pe=peer_pe,
    )
    if peer_pe is None:
        assessment.risks.append(
            "Sector valuation is unverified: no positive mainboard sector-peer "
            "P/E sample was available. The 25x P/E check is only a generic reference."
        )
        if assessment.verdict == "APPLY CANDIDATE":
            assessment.verdict = "CAUTION"
    gmp_url = None
    for link in soup.find_all("a", href=True):
        candidate = str(link["href"])
        parsed = urlparse(candidate)
        if (
            parsed.scheme == "https"
            and parsed.netloc == "www.investorgain.com"
            and parsed.path.startswith("/chr-gmp/")
        ):
            gmp_url = candidate
            break
    annual_periods = [period for period in periods if (period.month, period.day) == (3, 31)]
    if annual_periods and (reference_date - max(annual_periods)).days > 550:
        assessment.risks.append("The latest full-year financials are more than 550 days old.")
        if assessment.verdict == "APPLY CANDIDATE":
            assessment.verdict = "CAUTION"
    return OpenIPO(
        name=name, url=url, opens=opens, closes=closes,
        price_band=details.get("Price Band", details.get("Issue Price", "Unavailable")),
        lot_size=details.get("Lot Size", "Unavailable"),
        sale_type=details.get("Sale Type", "Unavailable"),
        financials=financials,
        assessment=assessment,
        sector=sector,
        sector_peer_pe=peer_pe,
        sector_peer_count=peer_count,
        gmp_url=gmp_url,
        allotment_date=allotment_date,
    )


def _fetch(url: str) -> requests.Response:
    try:
        response = requests.get(
            url, timeout=30,
            headers={"User-Agent": "StockResearchDashboard/1.0", "Accept": "application/json,text/html"},
        )
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        raise StockDataError(f"Chittorgarh data request failed: {exc}") from exc


def screen_open_mainboard_ipos(
    today: date | None = None, *, include_upcoming: bool = False,
) -> IPOReport:
    checked_at = datetime.now(INDIA_TIME)
    today = today or checked_at.date()
    response = _fetch(IPO_LIST_URL)
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise StockDataError("Chittorgarh returned invalid IPO feed JSON.") from exc
    if (
        not isinstance(payload, dict) or payload.get("msg") != 1
        or not isinstance(payload.get("ipoList"), list)
    ):
        raise StockDataError("Chittorgarh IPO feed format was not recognized.")

    issues = []
    warnings = []
    seen = set()
    for item in payload["ipoList"]:
        if not isinstance(item, dict) or not isinstance(item.get("ipo_status"), str):
            raise StockDataError("Chittorgarh IPO feed contains an invalid status row.")
        status = item["ipo_status"].strip().lower()
        if status == "current":
            status = "open"
        if status != "open" and not (include_upcoming and status == "upcoming"):
            continue
        name = str(item.get("company_name") or "Unnamed IPO")
        if item.get("issue_type") != "IPO" or item.get("issue_withdraw") not in (0, "0", False):
            warnings.append(f"{name}: excluded because issue type or withdrawal status is not eligible.")
            continue
        dates = _subscription_dates(
            str(item.get("issue_open_date", "")),
            str(item.get("issue_close_date", "")), today,
            upcoming=status == "upcoming",
        )
        if dates is None:
            if status == "open":
                warnings.append(f"{name}: open status could not be confirmed against subscription dates.")
                continue
            warnings.append(
                f"{name}: upcoming dates are unconfirmed. The feed omits years; "
                "only dates within the next 183 days are inferred."
            )
        issue_id = str(item.get("id", ""))
        slug = str(item.get("urlrewrite_folder_name", ""))
        if not issue_id.isdigit() or not re.fullmatch(r"[a-z0-9-]+", slug):
            warnings.append(f"{name}: source detail URL is invalid.")
            continue
        if issue_id in seen:
            continue
        seen.add(issue_id)
        url = f"https://www.chittorgarh.com/ipo/{slug}/{issue_id}/"
        try:
            issue = parse_ipo_detail(
                _fetch(url).text, name, url,
                *(dates if dates is not None else (None, None)),
                reference_date=today,
            )
        except StockDataError as exc:
            warnings.append(f"{name}: assessment unavailable: {exc}")
        else:
            if issue.gmp_url:
                try:
                    response = _fetch(issue.gmp_url)
                    issue.gmp = parse_gmp_quote(response.text, response.url)
                except StockDataError as exc:
                    issue.gmp = GMPQuote(
                        None, "Unavailable", issue.gmp_url, str(exc),
                    )
            else:
                issue.gmp = GMPQuote(
                    None, "Unavailable", issue.url,
                    "No linked InvestorGain GMP source was found on the IPO page.",
                )
            issue.status = "Upcoming" if status == "upcoming" else "Open"
            if status == "upcoming":
                issue.assessment.risks.append(
                    "Upcoming IPO: not yet open for application. Dates, pricing "
                    "and the assessment are provisional; refresh when it opens."
                )
                if dates is None and issue.assessment.verdict == "APPLY CANDIDATE":
                    issue.assessment.verdict = "CAUTION"
            issues.append(issue)
    return IPOReport(issues, warnings, checked_at)
