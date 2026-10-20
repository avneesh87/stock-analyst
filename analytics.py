from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from io import StringIO
import math
import re
from urllib.parse import quote, urljoin
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import feedparser
import pandas as pd
import requests
import yfinance as yf
from curl_cffi.requests.exceptions import RequestException as CurlRequestException
from yfinance.exceptions import YFException


NSE_NIFTY_50_URL = (
    "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
)
NSE_EQUITY_LIST_URL = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
)

FALLBACK_NIFTY_50 = [
    ("ADANIENT", "Adani Enterprises", "Diversified"),
    ("ADANIPORTS", "Adani Ports", "Infrastructure"),
    ("APOLLOHOSP", "Apollo Hospitals", "Healthcare"),
    ("ASIANPAINT", "Asian Paints", "Consumer"),
    ("AXISBANK", "Axis Bank", "Financial Services"),
    ("BAJAJ-AUTO", "Bajaj Auto", "Automobile"),
    ("BAJFINANCE", "Bajaj Finance", "Financial Services"),
    ("BAJAJFINSV", "Bajaj Finserv", "Financial Services"),
    ("BEL", "Bharat Electronics", "Industrials"),
    ("BHARTIARTL", "Bharti Airtel", "Telecommunication"),
    ("CIPLA", "Cipla", "Healthcare"),
    ("COALINDIA", "Coal India", "Energy"),
    ("DRREDDY", "Dr. Reddy's Laboratories", "Healthcare"),
    ("EICHERMOT", "Eicher Motors", "Automobile"),
    ("ETERNAL", "Eternal", "Consumer Services"),
    ("GRASIM", "Grasim Industries", "Materials"),
    ("HCLTECH", "HCL Technologies", "Information Technology"),
    ("HDFCBANK", "HDFC Bank", "Financial Services"),
    ("HDFCLIFE", "HDFC Life", "Financial Services"),
    ("HEROMOTOCO", "Hero MotoCorp", "Automobile"),
    ("HINDALCO", "Hindalco Industries", "Materials"),
    ("HINDUNILVR", "Hindustan Unilever", "Consumer"),
    ("ICICIBANK", "ICICI Bank", "Financial Services"),
    ("INDUSINDBK", "IndusInd Bank", "Financial Services"),
    ("INFY", "Infosys", "Information Technology"),
    ("ITC", "ITC", "Consumer"),
    ("JIOFIN", "Jio Financial Services", "Financial Services"),
    ("JSWSTEEL", "JSW Steel", "Materials"),
    ("KOTAKBANK", "Kotak Mahindra Bank", "Financial Services"),
    ("LT", "Larsen & Toubro", "Industrials"),
    ("M&M", "Mahindra & Mahindra", "Automobile"),
    ("MARUTI", "Maruti Suzuki", "Automobile"),
    ("MAXHEALTH", "Max Healthcare", "Healthcare"),
    ("NESTLEIND", "Nestle India", "Consumer"),
    ("NTPC", "NTPC", "Energy"),
    ("ONGC", "ONGC", "Energy"),
    ("POWERGRID", "Power Grid Corporation", "Utilities"),
    ("RELIANCE", "Reliance Industries", "Diversified"),
    ("SBILIFE", "SBI Life", "Financial Services"),
    ("SBIN", "State Bank of India", "Financial Services"),
    ("SHRIRAMFIN", "Shriram Finance", "Financial Services"),
    ("SUNPHARMA", "Sun Pharmaceutical", "Healthcare"),
    ("TATACONSUM", "Tata Consumer Products", "Consumer"),
    ("TATAMOTORS", "Tata Motors", "Automobile"),
    ("TATASTEEL", "Tata Steel", "Materials"),
    ("TCS", "Tata Consultancy Services", "Information Technology"),
    ("TECHM", "Tech Mahindra", "Information Technology"),
    ("TITAN", "Titan Company", "Consumer"),
    ("TRENT", "Trent", "Consumer"),
    ("ULTRACEMCO", "UltraTech Cement", "Materials"),
    ("WIPRO", "Wipro", "Information Technology"),
]

POSITIVE_WORDS = {
    "beat",
    "beats",
    "buy",
    "growth",
    "gain",
    "gains",
    "higher",
    "jump",
    "profit",
    "record",
    "rise",
    "rises",
    "strong",
    "surge",
    "upgrade",
    "wins",
}
NEGATIVE_WORDS = {
    "avoid",
    "cut",
    "decline",
    "downgrade",
    "drop",
    "falls",
    "fraud",
    "loss",
    "lower",
    "miss",
    "risk",
    "sell",
    "slump",
    "weak",
}


class StockDataError(RuntimeError):
    pass


def to_crore(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value) / 10_000_000


def normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if symbol and "." not in symbol and not symbol.startswith("^"):
        return f"{symbol}.NS"
    return symbol


def get_nifty50_universe() -> tuple[pd.DataFrame, str]:
    try:
        response = requests.get(NSE_NIFTY_50_URL, timeout=12)
        response.raise_for_status()
        universe = pd.read_csv(StringIO(response.text))
        universe = universe[["Symbol", "Company Name", "Industry"]].copy()
        return universe, "NSE"
    except (requests.RequestException, KeyError, pd.errors.ParserError):
        universe = pd.DataFrame(
            FALLBACK_NIFTY_50,
            columns=["Symbol", "Company Name", "Industry"],
        )
        return universe, "built-in Nifty 50 fallback"


def get_all_nse_equities() -> tuple[pd.DataFrame, str]:
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        response = requests.get(
            NSE_EQUITY_LIST_URL,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        universe = pd.read_csv(StringIO(response.text))
        universe.columns = [str(column).strip() for column in universe.columns]
        universe = universe[universe["SERIES"].astype(str).str.strip() == "EQ"]
        universe = universe[["SYMBOL", "NAME OF COMPANY"]].rename(
            columns={
                "SYMBOL": "Symbol",
                "NAME OF COMPANY": "Company Name",
            }
        )
        universe["Symbol"] = universe["Symbol"].astype(str).str.strip()
        universe["Company Name"] = (
            universe["Company Name"].astype(str).str.strip()
        )
        universe["Industry"] = "N/A"
    except (
        requests.RequestException,
        KeyError,
        pd.errors.ParserError,
    ) as exc:
        raise StockDataError(
            f"Could not retrieve the official NSE equity universe: {exc}"
        ) from exc

    if universe.empty:
        raise StockDataError("The official NSE equity universe was empty.")
    return universe.reset_index(drop=True), "Official NSE equity master"


def get_news(symbol: str, limit: int = 8) -> list[dict[str, str]]:
    query = symbol.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query}+india+stock"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise StockDataError(f"Could not retrieve news: {exc}") from exc

    feed = feedparser.parse(response.content)
    return [
        {
            "title": entry.get("title", "Untitled"),
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
        }
        for entry in feed.entries[:limit]
    ]


def headline_sentiment(title: str) -> tuple[str, int]:
    words = {
        word.strip(".,:;!?()[]{}\"'").lower()
        for word in title.split()
    }
    score = len(words & POSITIVE_WORDS) - len(words & NEGATIVE_WORDS)
    if score > 0:
        return "Positive", score
    if score < 0:
        return "Negative", score
    return "Neutral", 0


def news_sentiment(news: list[dict[str, str]]) -> dict[str, object]:
    details = []
    total = 0
    for item in news:
        label, score = headline_sentiment(item["title"])
        total += score
        details.append({**item, "sentiment": label, "sentiment_score": score})

    if total > 0:
        overall = "Positive"
    elif total < 0:
        overall = "Negative"
    else:
        overall = "Neutral"
    return {"overall": overall, "score": total, "headlines": details}


def _statement_values(statement: pd.DataFrame, row_name: str) -> list[float | None]:
    if statement.empty or row_name not in statement.index:
        return []

    values = []
    for raw_value in statement.sort_index(axis=1, ascending=False).loc[row_name].tolist():
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            values.append(None)
        else:
            values.append(value if math.isfinite(value) else None)
    return values


def _ttm_income(
    statement: pd.DataFrame, annual: pd.DataFrame | None = None,
) -> tuple[float | None, str | None, str]:
    statement = statement.sort_index(axis=1, ascending=False)
    row = (
        "Net Income Common Stockholders"
        if "Net Income Common Stockholders" in statement.index
        else "Net Income"
    )
    periods = pd.DatetimeIndex(statement.columns[:4])
    values = _statement_values(statement.iloc[:, :4], row)
    valid_quarters = (
        len(periods) == 4 and len(values) == 4
        and not periods.has_duplicates
        and all(value is not None for value in values)
        and all(70 <= (periods[i] - periods[i + 1]).days <= 110 for i in range(3))
    )
    total = sum(value for value in values if value is not None) if valid_quarters else None
    basis = f"Market cap / four-quarter {row}"
    if total is None and annual is not None and not annual.empty and len(periods):
        annual = annual.sort_index(axis=1, ascending=False)
        annual_values = _statement_values(annual, row)
        year_end = pd.Timestamp(annual.columns[0])
        latest = periods[0]
        # FY + current YTD - prior YTD is equivalent to four consecutive quarters.
        current_ytd = []
        prior_ytd = []
        ytd_end = None
        for offset in (3, 6, 9):
            expected = year_end + pd.DateOffset(months=offset)
            if expected > latest + pd.Timedelta(days=7):
                break
            ytd_end = expected
            for target, output in (
                (expected, current_ytd),
                (expected - pd.DateOffset(years=1), prior_ytd),
            ):
                matches = [
                    column for column in statement.columns
                    if abs((pd.Timestamp(column) - target).days) <= 7
                ]
                matched_values = _statement_values(statement[matches], row)
                output.append(matched_values[0] if len(matched_values) == 1 else None)
        if (
            annual_values and annual_values[0] is not None
            and current_ytd and all(value is not None for value in current_ytd + prior_ytd)
            and ytd_end is not None and abs((ytd_end - latest).days) <= 7
        ):
            total = annual_values[0] + sum(
                value for value in current_ytd if value is not None
            ) - sum(value for value in prior_ytd if value is not None)
            basis = f"Market cap / {row} (latest FY + current YTD - prior YTD)"
        elif annual_values and annual_values[0] is not None and latest == year_end:
            total = annual_values[0]
            basis = f"Market cap / latest fiscal-year {row} (FY end equals TTM end)"
    if total is None:
        return None, None, (
            "TTM P/E unavailable: missing/non-consecutive quarterly profits "
            "and insufficient annual/YTD data."
        )
    period = str(periods[0].date())
    if total <= 0:
        return None, period, "TTM P/E not meaningful: trailing four-quarter earnings are non-positive."
    return total, period, f"{basis}; quarter ended {period}."


def _financial_currencies(
    stock: yf.Ticker, symbol: str,
) -> tuple[dict[tuple[str, str], tuple[str, float]], str]:
    metrics = [
        "annualTotalRevenue", "annualNetIncome", "annualNetIncomeCommonStockholders",
        "quarterlyNetIncome", "quarterlyNetIncomeCommonStockholders",
        "annualStockholdersEquity", "annualTotalDebt", "annualFreeCashFlow",
    ]
    now = pd.Timestamp.now(tz="UTC")
    try:
        response = stock.session.get(
            "https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/"
            f"finance/timeseries/{quote(symbol, safe='')}",
            params={
                "symbol": symbol, "type": ",".join(metrics),
                "period1": int((now - pd.DateOffset(years=6)).timestamp()),
                "period2": int(now.timestamp()),
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()["timeseries"]
        if payload.get("error") or not isinstance(payload.get("result"), list):
            raise ValueError("missing financial currency series")
        currencies = {}
        for series in payload["result"]:
            for metric in metrics:
                for item in series.get(metric, []):
                    currency = item.get("currencyCode")
                    value = float(item["reportedValue"]["raw"])
                    if isinstance(currency, str) and re.fullmatch(r"[A-Z]{3}", currency) and math.isfinite(value):
                        currencies[(metric, item["asOfDate"])] = (currency, value)
        return currencies, ""
    except (CurlRequestException, KeyError, TypeError, ValueError) as exc:
        return {}, f"Financial currency verification failed: {exc}. Currency-dependent values withheld."


def get_stock_data(symbol: str) -> dict[str, float | str | None]:
    symbol = normalize_symbol(symbol)
    try:
        stock = yf.Ticker(symbol)
        history = stock.history(period="5d", auto_adjust=False)
        fast_info = dict(stock.fast_info)
        income = stock.income_stmt
        quarterly_income = stock.quarterly_income_stmt
        balance = stock.balance_sheet
        cash_flow = stock.cashflow
    except (CurlRequestException, YFException, KeyError, TypeError, ValueError) as exc:
        raise StockDataError(f"Yahoo Finance could not load {symbol}: {exc}") from exc

    if history.empty:
        raise StockDataError(f"No online market data was found for '{symbol}'.")

    currencies, currency_note = _financial_currencies(stock, symbol)

    def reporting_currency(frame: pd.DataFrame, row: str, frequency: str, count: int) -> str | None:
        frame = frame.sort_index(axis=1, ascending=False).iloc[:, :count]
        values = _statement_values(frame, row)
        found = set()
        for column, value in zip(frame.columns, values):
            if value is None:
                continue
            metadata = currencies.get((frequency + row.replace(" ", ""), str(column.date())))
            if metadata is None or not math.isclose(metadata[1], value, rel_tol=1e-9, abs_tol=0.01):
                return None
            found.add(metadata[0])
        return next(iter(found)) if len(found) == 1 else None

    income = income.sort_index(axis=1, ascending=False)
    if not income.empty:
        periods = income.columns
        balance = balance.reindex(columns=periods)
        cash_flow = cash_flow.reindex(columns=periods)
    revenue = _statement_values(income, "Total Revenue")
    profit_row = (
        "Net Income Common Stockholders"
        if "Net Income Common Stockholders" in income.index else "Net Income"
    )
    net_income = _statement_values(income, profit_row)
    equity = _statement_values(balance, "Stockholders Equity")
    debt = _statement_values(balance, "Total Debt")
    free_cash_flow = _statement_values(cash_flow, "Free Cash Flow")
    market_cap = fast_info.get("marketCap")
    latest_revenue = revenue[0] if revenue else None
    latest_income = net_income[0] if net_income else None
    latest_equity = equity[0] if equity else None
    ttm_income, ttm_period, pe_note = _ttm_income(quarterly_income, income)
    income_currency = reporting_currency(income, profit_row, "annual", 1)
    revenue_currency = reporting_currency(income, "Total Revenue", "annual", 1)
    growth_currency = reporting_currency(income, "Total Revenue", "annual", 2)
    equity_currency = reporting_currency(balance, "Stockholders Equity", "annual", 1)
    debt_currency = reporting_currency(balance, "Total Debt", "annual", 1)
    cash_currency = reporting_currency(cash_flow, "Free Cash Flow", "annual", 1)
    quarterly_profit_row = (
        "Net Income Common Stockholders"
        if "Net Income Common Stockholders" in quarterly_income.index else "Net Income"
    )
    ttm_currency = reporting_currency(
        quarterly_income, quarterly_profit_row, "quarterly", len(quarterly_income.columns),
    )
    if not income.empty and ttm_currency != income_currency:
        ttm_currency = None
    quote_currency = fast_info.get("currency")
    same_currency = isinstance(quote_currency, str) and ttm_currency == quote_currency
    if not same_currency:
        pe_note = (
            f"TTM P/E withheld: quote currency {quote_currency or 'unverified'}, "
            f"earnings currency {ttm_currency or 'unverified'}. No assumed FX conversion."
        )
    notes = [currency_note] if currency_note else []
    if not same_currency:
        notes.append(pe_note)
    if not income_currency or not growth_currency or not equity_currency or not debt_currency or not cash_currency:
        notes.append("Some statement currencies could not be verified; affected financial values are unavailable.")
    if cash_currency and cash_currency != "INR":
        notes.append(f"Free cash flow is reported in {cash_currency}, not INR.")
    valid_market_cap = (
        isinstance(market_cap, (int, float))
        and math.isfinite(market_cap) and market_cap > 0
    )
    if not valid_market_cap:
        pe_note = "TTM P/E unavailable: valid market capitalization is missing."
    consecutive_years = (
        len(income.columns) >= 2
        and 330 <= (income.columns[0] - income.columns[1]).days <= 400
    )

    return {
        "Company": symbol,
        "Price Date": str(history.index[-1].date()) if isinstance(history.index, pd.DatetimeIndex) else None,
        "Quote Currency": quote_currency,
        "Financial Currency": income_currency,
        "Revenue Currency": growth_currency,
        "Free Cash Flow Currency": cash_currency,
        "Data Note": " ".join(notes),
        "Financial Period": str(income.columns[0].date()) if not income.empty else None,
        "Price": float(history["Close"].iloc[-1]),
        "Market Cap": market_cap,
        "Market Cap (Cr)": to_crore(market_cap),
        "PE": (
            market_cap / ttm_income
            if valid_market_cap and same_currency and ttm_income is not None else None
        ),
        "TTM Period": ttm_period,
        "PE Note": pe_note,
        "Annual PE": (
            market_cap / latest_income
            if valid_market_cap and income_currency == quote_currency
            and income_currency is not None and latest_income and latest_income > 0
            else None
        ),
        "ROE": (
            latest_income / latest_equity
            if income_currency is not None and income_currency == equity_currency
            and latest_income is not None and latest_equity is not None and latest_equity > 0
            else None
        ),
        "DebtEquity": (
            debt[0] / latest_equity * 100
            if debt_currency is not None and debt_currency == equity_currency
            and debt and debt[0] is not None and latest_equity is not None and latest_equity > 0
            else None
        ),
        "RevenueGrowth": (
            (revenue[0] / revenue[1]) - 1
            if growth_currency is not None and consecutive_years and len(revenue) > 1 and revenue[0] is not None
            and revenue[1] is not None and revenue[1] > 0
            else None
        ),
        "ProfitMargin": (
            latest_income / latest_revenue
            if income_currency is not None and income_currency == revenue_currency
            and latest_income is not None and latest_revenue is not None and latest_revenue > 0
            else None
        ),
        "FreeCashFlow": free_cash_flow[0] if cash_currency and free_cash_flow else None,
    }


def score_stock(data: dict[str, float | str | None]) -> int:
    score = 0
    checks = (
        (data["ROE"], lambda value: value > 0.15),
        (data["DebtEquity"], lambda value: value < 50),
        (data["RevenueGrowth"], lambda value: value > 0.15),
        (data["ProfitMargin"], lambda value: value > 0.10),
        (data["PE"], lambda value: 0 < value < 40),
    )
    for value, predicate in checks:
        if isinstance(value, (int, float)) and predicate(value):
            score += 20
    return score


def build_recommendation(
    data: dict[str, float | str | None],
    fundamental_score: int,
    technicals: dict[str, float | bool | None],
    sentiment: dict[str, object],
) -> dict[str, object]:
    problems = []
    debt_equity = data.get("DebtEquity")
    roe = data.get("ROE")
    revenue_growth = data.get("RevenueGrowth")
    free_cash_flow = data.get("FreeCashFlow")

    if isinstance(debt_equity, (int, float)) and debt_equity > 100:
        problems.append("High Debt")
    if isinstance(roe, (int, float)) and roe < 0.10:
        problems.append("Low ROE")
    if isinstance(revenue_growth, (int, float)) and revenue_growth < 0:
        problems.append("Declining Revenue")
    if isinstance(free_cash_flow, (int, float)) and free_cash_flow < 0:
        problems.append("Negative Cash Flow")

    technical_score = technicals.get("Technical Score")
    if not isinstance(technical_score, (int, float)):
        technical_score = 50
    sentiment_score = {
        "Positive": 70,
        "Neutral": 50,
        "Negative": 30,
    }.get(str(sentiment.get("overall")), 50)
    composite_score = (
        fundamental_score * 0.50
        + technical_score * 0.35
        + sentiment_score * 0.15
    )

    if len(problems) >= 2 or composite_score < 40:
        verdict = "AVOID"
        reason = (
            "Financial strength is weak because "
            + ", ".join(problem.lower() for problem in problems)
            + "."
            if problems
            else "Fundamental, technical, and sentiment signals are weak."
        )
    elif composite_score >= 70 and not problems:
        verdict = "BUY"
        reason = (
            "Fundamental strength is supported by technical momentum and "
            "non-negative news sentiment."
        )
    else:
        verdict = "WATCH"
        reason = (
            "The signals are mixed; monitor the identified risks and wait for "
            "stronger confirmation."
        )

    if verdict == "BUY":
        agreement = sum(
            (
                fundamental_score >= 60,
                technical_score >= 50,
                sentiment_score >= 50,
            )
        )
    elif verdict == "AVOID":
        agreement = sum(
            (
                fundamental_score <= 40,
                technical_score <= 40,
                sentiment_score <= 30,
            )
        )
        agreement = max(agreement, min(len(problems), 3))
    else:
        agreement = 1 + int(40 <= composite_score < 70)

    evidence_fields = ("ROE", "DebtEquity", "RevenueGrowth", "FreeCashFlow")
    missing_evidence = sum(data.get(field) is None for field in evidence_fields)
    confidence = 6.25 + agreement * 0.75 - missing_evidence * 0.25
    confidence = max(5.0, min(9.5, round(confidence * 2) / 2))

    return {
        "verdict": verdict,
        "confidence": confidence,
        "problems": problems,
        "reason": reason,
        "composite_score": round(composite_score, 1),
    }


def get_price_history(
    symbol: str, period: str = "1y", *, auto_adjust: bool = False,
) -> pd.DataFrame:
    symbol = normalize_symbol(symbol)
    try:
        history = yf.Ticker(symbol).history(period=period, auto_adjust=auto_adjust)
    except (CurlRequestException, YFException, KeyError, TypeError, ValueError) as exc:
        raise StockDataError(f"Could not retrieve price history for {symbol}: {exc}") from exc
    if history.empty:
        raise StockDataError(f"No price history was found for '{symbol}'.")
    return history


def _calendar_return(
    close: pd.Series, months: int, *, as_of: date | None = None,
) -> float | None:
    close = close.sort_index()
    dates = close.index.normalize()
    anchor = pd.Timestamp(as_of, tz=dates.tz) if as_of is not None else dates[-1]
    target = anchor - pd.DateOffset(months=months)
    earlier = close.loc[dates <= target]
    previous = float(earlier.iloc[-1]) if not earlier.empty else None
    current = float(close.iloc[-1])
    if (
        previous is None or not math.isfinite(previous) or previous <= 0
        or not math.isfinite(current) or current <= 0
    ):
        return None
    return (current / previous - 1) * 100


def calculate_technicals(history: pd.DataFrame) -> pd.DataFrame:
    if isinstance(history.index, pd.DatetimeIndex):
        history = history.sort_index()
    result = history[["Close"]].copy()
    close = result["Close"]
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    # Seed Wilder smoothing with the first 14 changes, then recurse.
    average_gain = gains.copy()
    average_loss = losses.copy()
    if len(close) > 14:
        average_gain.iloc[14] = gains.iloc[1:15].mean()
        average_loss.iloc[14] = losses.iloc[1:15].mean()
    average_gain.iloc[:14] = float("nan")
    average_loss.iloc[:14] = float("nan")
    average_gain = average_gain.ewm(alpha=1 / 14, adjust=False, ignore_na=True).mean()
    average_loss = average_loss.ewm(alpha=1 / 14, adjust=False, ignore_na=True).mean()
    relative_strength = average_gain / average_loss.replace(0, float("nan"))
    result["RSI"] = 100 - (100 / (1 + relative_strength))
    result.loc[(average_loss == 0) & (average_gain > 0), "RSI"] = 100.0
    result.loc[(average_loss == 0) & (average_gain == 0), "RSI"] = 50.0
    result["MACD"] = close.ewm(span=12, adjust=False).mean() - close.ewm(
        span=26, adjust=False
    ).mean()
    result["MACD Signal"] = result["MACD"].ewm(span=9, adjust=False).mean()
    result["200 DMA"] = close.rolling(200).mean()
    previous_close = close.shift(1)
    result["20D Support"] = previous_close.rolling(20).min()
    result["20D Resistance"] = previous_close.rolling(20).max()
    result["60D Support"] = previous_close.rolling(60).min()
    result["60D Resistance"] = previous_close.rolling(60).max()
    return result


def technical_snapshot(history: pd.DataFrame) -> dict[str, float | bool | None]:
    technicals = calculate_technicals(history)
    latest = technicals.iloc[-1]
    close = float(latest["Close"])
    rsi = float(latest["RSI"]) if pd.notna(latest["RSI"]) else None
    macd = float(latest["MACD"]) if pd.notna(latest["MACD"]) else None
    signal = (
        float(latest["MACD Signal"])
        if pd.notna(latest["MACD Signal"])
        else None
    )
    dma_200 = float(latest["200 DMA"]) if pd.notna(latest["200 DMA"]) else None
    support = (
        float(latest["20D Support"])
        if pd.notna(latest["20D Support"])
        else None
    )
    resistance = (
        float(latest["20D Resistance"])
        if pd.notna(latest["20D Resistance"])
        else None
    )
    support_60d = (
        float(latest["60D Support"])
        if pd.notna(latest["60D Support"])
        else None
    )
    resistance_60d = (
        float(latest["60D Resistance"])
        if pd.notna(latest["60D Resistance"])
        else None
    )
    if isinstance(technicals.index, pd.DatetimeIndex):
        six_month_return = _calendar_return(technicals["Close"], 6)
    else:
        old_close = float(technicals["Close"].iloc[-127]) if len(technicals) > 126 else None
        six_month_return = (close / old_close - 1) * 100 if old_close is not None and old_close > 0 else None

    score = 0
    if dma_200 is not None and close > dma_200:
        score += 30
    if macd is not None and signal is not None and macd > signal:
        score += 25
    if rsi is not None and 40 <= rsi <= 70:
        score += 20
    if six_month_return is not None and six_month_return > 0:
        score += 25

    return {
        "Price": close,
        "RSI": rsi,
        "MACD": macd,
        "MACD Signal": signal,
        "200 DMA": dma_200,
        "20D Support": support,
        "20D Resistance": resistance,
        "60D Support": support_60d,
        "60D Resistance": resistance_60d,
        "Above 200 DMA": dma_200 is not None and close > dma_200,
        "6M Return %": six_month_return,
        "Technical Score": score,
    }


def technical_entry_zone(
    technicals: dict[str, float | bool | None],
) -> dict[str, float | str | None]:
    price = technicals.get("Price")
    support_20d = technicals.get("20D Support")
    support_60d = technicals.get("60D Support")
    resistance = technicals.get("20D Resistance")
    rsi = technicals.get("RSI")
    dma_200 = technicals.get("200 DMA")

    def finite_number(value: object) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    if (
        not finite_number(price)
        or not finite_number(support_20d)
        or not finite_number(support_60d)
        or price <= 0
        or support_20d <= 0
        or support_60d <= 0
    ):
        return {
            "low": None,
            "high": None,
            "status": "Insufficient data",
            "reason": (
                "A valid price and both 20-day and 60-day support levels "
                "are required."
            ),
        }

    support_low = min(support_20d, support_60d)
    support_high = max(support_20d, support_60d)
    if (support_high - support_low) / price > 0.05:
        return {
            "low": None,
            "high": None,
            "status": "Wait — 20D and 60D support levels diverge",
            "reason": (
                "Short- and medium-term support are more than 5% apart; "
                "wait for clearer support confirmation."
            ),
        }

    if finite_number(resistance) and resistance <= support_low:
        return {
            "low": None,
            "high": None,
            "status": "Insufficient data",
            "reason": "The calculated support and resistance levels are invalid.",
        }

    high = support_high * 1.02
    if finite_number(resistance):
        high = min(high, resistance)

    if finite_number(rsi) and rsi > 70:
        status = "Wait — RSI is overbought"
        reason = "Avoid chasing; consider the zone only after momentum cools."
    elif price < support_20d:
        status = "Wait — price is below 20-day support"
        reason = "Wait for price to reclaim recent support before considering an entry."
    elif price > high:
        status = "Wait for a pullback"
        reason = "Current price is above the support-based entry zone."
    elif finite_number(dma_200) and price < dma_200:
        status = "Caution — price is below the 200-day average"
        reason = "The long-term trend is not confirmed; use extra caution."
    else:
        status = "Price is in the potential entry zone"
        reason = (
            "The zone uses 20-day and 60-day support levels with a 2% buffer."
        )

    return {
        "low": support_low,
        "high": high,
        "status": status,
        "reason": reason,
    }


def get_industry_pe(symbol: str) -> tuple[float | None, str]:
    normalized_symbol = normalize_symbol(symbol)
    base_symbol = normalized_symbol.removesuffix(".NS")
    if not normalized_symbol.endswith(".NS") or not re.fullmatch(r"[A-Z0-9&_-]+", base_symbol):
        return None, "Industry P/E lookup supports NSE equity symbols only."
    origin = "https://www.screener.in"

    def load_page(url: str) -> BeautifulSoup:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise StockDataError(f"Industry P/E source could not load {url}: {exc}") from exc
        return BeautifulSoup(response.text, "html.parser")

    company = load_page(f"{origin}/company/{quote(base_symbol, safe='')}/consolidated/")
    classifications = [
        link for link in company.select('#peers a[href^="/market/"]')
        if re.fullmatch(r"/market/(?:IN\d+/)+", link["href"])
    ]
    if not classifications:
        raise StockDataError("Industry classification unavailable from Screener; no Nifty 50 fallback used.")
    classification = max(classifications, key=lambda link: link["href"].count("/"))
    industry = classification.get_text(" ", strip=True)
    industry_url = origin + classification["href"]
    next_url = industry_url
    visited = set()
    peer_values = {}
    excluded = 0
    expected_pages = None
    while next_url:
        if next_url in visited:
            raise StockDataError("Industry P/E pagination repeated a page; incomplete benchmark withheld.")
        visited.add(next_url)
        page = load_page(next_url)
        pagination = re.search(
            r"Showing page\s+(\d+)\s+of\s+(\d+)", page.get_text(" ", strip=True),
            re.IGNORECASE,
        )
        if pagination is None:
            raise StockDataError("Industry page count unavailable; completeness cannot be verified.")
        page_number, page_count = map(int, pagination.groups())
        if expected_pages is None:
            expected_pages = page_count
        if page_count != expected_pages or page_number != len(visited):
            raise StockDataError("Industry pagination changed during lookup; benchmark withheld.")
        table = page.select_one("table")
        if table is None:
            raise StockDataError("Industry P/E table unavailable; incomplete benchmark withheld.")
        headers = [cell.get_text(" ", strip=True) for cell in table.select("tr")[0].select("th")]
        if "P/E" not in headers or "Company" not in headers:
            raise StockDataError("Industry P/E table columns changed; benchmark unavailable.")
        pe_index = headers.index("P/E")
        company_index = headers.index("Company")
        for row in table.select("tr"):
            cells = row.find_all("td", recursive=False)
            if not cells:
                continue
            if len(cells) <= max(pe_index, company_index):
                raise StockDataError("Industry P/E row is incomplete; benchmark withheld.")
            link = cells[company_index].select_one('a[href^="/company/"]')
            if link is None:
                raise StockDataError("Industry P/E company identifier missing; benchmark withheld.")
            peer_symbol = link["href"].split("/")[2]
            if peer_symbol == base_symbol:
                continue
            raw_value = cells[pe_index].get_text(strip=True).replace(",", "")
            try:
                value = float(raw_value)
            except ValueError:
                if raw_value not in ("", "-", "—", "N/A"):
                    raise StockDataError(f"Unrecognized industry P/E value: {raw_value}")
                excluded += 1
                continue
            if math.isfinite(value) and value > 0:
                peer_values[peer_symbol] = value
            else:
                excluded += 1
        next_link = page.find("a", string=re.compile(r"^\s*Next\s*$"))
        if next_link is None:
            next_link = next(
                (link for link in page.select("a[href]") if link.get_text(" ", strip=True) == "Next"),
                None,
            )
        next_url = urljoin(industry_url, next_link["href"]) if next_link else ""
        if next_url and not next_url.startswith(industry_url + "?"):
            raise StockDataError("Unexpected industry pagination URL; benchmark withheld.")
    if len(visited) != expected_pages:
        raise StockDataError("Industry pagination ended early; incomplete benchmark withheld.")
    if not peer_values:
        return None, f"No positive P/E data available for other {industry} companies. Source: {industry_url}"
    return float(pd.Series(peer_values).median()), (
        f"Median of {len(peer_values)} available {industry} peers across all "
        f"{len(visited)} Screener industry page(s), not limited to Nifty 50; "
        f"analyzed stock excluded, {excluded} missing/non-positive P/Es excluded. "
        "Source-reported TTM P/Es may differ in timing, consolidation and share basis "
        f"from Yahoo earnings; not a market-cap-weighted ratio. Source: {industry_url}"
    )


def get_peer_comparison(symbol: str, max_peers: int = 5) -> tuple[pd.DataFrame, str]:
    symbol = normalize_symbol(symbol)
    base_symbol = symbol.removesuffix(".NS")
    universe, source = get_nifty50_universe()
    match = universe[universe["Symbol"] == base_symbol]
    if match.empty:
        raise StockDataError(f"{symbol} is not in the Nifty 50 peer universe.")

    industry = match.iloc[0]["Industry"]
    peers = universe[universe["Industry"] == industry]["Symbol"].tolist()
    selected = [base_symbol] + [peer for peer in peers if peer != base_symbol]
    selected = selected[: max_peers + 1]

    rows = []
    for peer in selected:
        peer_symbol = normalize_symbol(peer)
        try:
            history = get_price_history(peer_symbol)
            snapshot = technical_snapshot(history)
            market_cap = dict(yf.Ticker(peer_symbol).fast_info).get("marketCap")
        except (StockDataError, CurlRequestException, YFException, KeyError, ValueError):
            continue
        rows.append(
            {
                "Symbol": peer_symbol,
                "Industry": industry,
                "Price": snapshot["Price"],
                "Market Cap (Cr)": to_crore(market_cap),
                "RSI": snapshot["RSI"],
                "6M Return %": snapshot["6M Return %"],
                "Above 200 DMA": snapshot["Above 200 DMA"],
                "Technical Score": snapshot["Technical Score"],
            }
        )
    if not rows:
        raise StockDataError("No online peer data could be retrieved.")
    return pd.DataFrame(rows), source


def _technical_rows_from_prices(
    prices: pd.DataFrame | pd.Series,
    symbols: list[str],
) -> list[dict[str, object]]:
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=symbols[0])

    rows = []
    for symbol in symbols:
        if symbol not in prices.columns:
            continue
        close = prices[symbol].dropna()
        if len(close) < 30:
            continue
        snapshot = technical_snapshot(pd.DataFrame({"Close": close}))
        rows.append(
            {
                "Symbol": symbol,
                "Price": snapshot["Price"],
                "RSI": snapshot["RSI"],
                "6M Return %": snapshot["6M Return %"],
                "Above 200 DMA": snapshot["Above 200 DMA"],
                "Technical Score": snapshot["Technical Score"],
            }
        )
    return rows


def _download_technical_universe(
    universe: pd.DataFrame,
    *,
    batch_size: int,
) -> pd.DataFrame:
    symbols = [normalize_symbol(symbol) for symbol in universe["Symbol"].tolist()]
    rows = []
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        try:
            prices = yf.download(
                batch,
                period="1y",
                auto_adjust=False,
                progress=False,
                threads=True,
                timeout=20,
            )["Close"]
        except (CurlRequestException, YFException, KeyError, ValueError):
            continue
        rows.extend(_technical_rows_from_prices(prices, batch))

    if not rows:
        raise StockDataError("The online screen returned no usable stock history.")
    return pd.DataFrame(rows).sort_values(
        ["Technical Score", "6M Return %"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _technical_universe() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    universe, source = get_nifty50_universe()
    ranking = _download_technical_universe(universe, batch_size=100)
    return ranking, universe, source


def _all_nse_technical_universe() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    universe, source = get_all_nse_equities()
    try:
        ranking = _download_technical_universe(universe, batch_size=150)
    except StockDataError as exc:
        raise StockDataError(f"Could not screen all NSE equities: {exc}") from exc
    return ranking, universe, source


def rank_top_stocks(limit: int = 10) -> tuple[pd.DataFrame, str]:
    ranking, _, source = _technical_universe()
    return ranking.head(limit).reset_index(drop=True), source


def meets_quality_rules(
    *,
    market_cap_crore: float | None,
    technical_score: float,
    above_200_dma: bool,
    sales_growth: float | None,
    profit_growth: float | None,
    latest_profit: float | None,
    minimum_market_cap_crore: float = 500,
) -> bool:
    return (
        market_cap_crore is not None
        and market_cap_crore > minimum_market_cap_crore
        and technical_score >= 80
        and above_200_dma
        and sales_growth is not None
        and sales_growth > 0
        and profit_growth is not None
        and profit_growth > 0
        and latest_profit is not None
        and latest_profit > 0
    )


def _quality_fundamentals(
    technical_row: dict[str, object],
    company_lookup: dict[str, tuple[str, str]],
    minimum_market_cap_crore: float,
) -> tuple[dict[str, object] | None, str | None]:
    symbol = str(technical_row["Symbol"])
    try:
        stock = yf.Ticker(symbol)
        statement = stock.quarterly_income_stmt
        market_cap_crore = to_crore(dict(stock.fast_info).get("marketCap"))
    except (CurlRequestException, YFException, KeyError, TypeError, ValueError) as exc:
        return None, f"{symbol}: {exc}"

    sales = _statement_values(statement, "Total Revenue")
    profits = _statement_values(statement, "Net Income")
    if (
        len(sales) < 2 or len(profits) < 2
        or any(value is None for value in sales[:2] + profits[:2])
    ):
        return None, f"{symbol}: fewer than two quarterly statements"
    periods = statement.sort_index(axis=1, ascending=False).columns
    if not 70 <= (periods[0] - periods[1]).days <= 110:
        return None, f"{symbol}: latest reported periods are not consecutive quarters"

    sales_growth = (sales[0] / sales[1] - 1) * 100 if sales[1] else None
    profit_growth = (profits[0] - profits[1]) / abs(profits[1]) * 100 if profits[1] else None
    technical_score = float(technical_row["Technical Score"])
    above_200_dma = bool(technical_row["Above 200 DMA"])
    latest_profit = profits[0]
    if not meets_quality_rules(
        market_cap_crore=market_cap_crore,
        technical_score=technical_score,
        above_200_dma=above_200_dma,
        sales_growth=sales_growth,
        profit_growth=profit_growth,
        latest_profit=latest_profit,
        minimum_market_cap_crore=minimum_market_cap_crore,
    ):
        return None, None

    base_symbol = symbol.removesuffix(".NS")
    company, industry = company_lookup.get(
        base_symbol,
        (base_symbol, "Unknown"),
    )
    return (
        {
            "Symbol": symbol,
            "Company": company,
            "Industry": industry,
            "Price": technical_row["Price"],
            "Market Cap (Cr)": market_cap_crore,
            "Sales QoQ %": sales_growth,
            "Profit QoQ %": profit_growth,
            "RSI": technical_row["RSI"],
            "6M Return %": technical_row["6M Return %"],
            "Technical Score": technical_score,
        },
        None,
    )


def screen_quality_stocks(
    minimum_market_cap_crore: float = 500,
) -> tuple[pd.DataFrame, str, dict[str, int]]:
    technicals, universe, source = _all_nse_technical_universe()
    technical_candidates = technicals[
        (technicals["Technical Score"] >= 80)
        & (technicals["Above 200 DMA"])
    ]
    company_lookup = {
        str(row["Symbol"]): (
            str(row["Company Name"]),
            str(row["Industry"]),
        )
        for row in universe.to_dict("records")
    }

    rows = []
    errors = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(
                _quality_fundamentals,
                row,
                company_lookup,
                minimum_market_cap_crore,
            )
            for row in technical_candidates.to_dict("records")
        ]
        for future in as_completed(futures):
            row, error = future.result()
            if row is not None:
                rows.append(row)
            if error is not None:
                errors.append(error)

    columns = [
        "Symbol",
        "Company",
        "Industry",
        "Price",
        "Market Cap (Cr)",
        "Sales QoQ %",
        "Profit QoQ %",
        "RSI",
        "6M Return %",
        "Technical Score",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if not result.empty:
        result = result.sort_values(
            ["Technical Score", "Profit QoQ %", "Sales QoQ %"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    stats = {
        "universe": len(universe),
        "price_history_available": len(technicals),
        "technical_candidates": len(technical_candidates),
        "matches": len(result),
        "quarterly_unavailable": len(errors),
    }
    return result, source, stats


def value_portfolio(holdings: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for holding in holdings:
        symbol = normalize_symbol(str(holding["symbol"]))
        try:
            history = get_price_history(symbol, period="5d", auto_adjust=False)
        except StockDataError:
            continue
        price = float(history["Close"].iloc[-1])
        quantity = float(holding["quantity"])
        average_price = float(holding["average_price"])
        invested = quantity * average_price
        current_value = quantity * price
        rows.append(
            {
                "Symbol": symbol,
                "Quantity": quantity,
                "Average Price": average_price,
                "Current Price": price,
                "Invested": invested,
                "Current Value": current_value,
                "P/L": current_value - invested,
                "P/L %": ((current_value / invested) - 1) * 100 if invested else None,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        total = result["Current Value"].sum()
        result["Allocation %"] = (
            result["Current Value"] / total * 100 if total else 0
        )
    return result


def watchlist_price_performance(
    history: pd.DataFrame, *, as_of: date | None = None,
) -> dict[str, float | None]:
    if history.empty or not isinstance(history.index, pd.DatetimeIndex):
        raise StockDataError("Dated price history is required to calculate price performance.")
    history = history.sort_index()
    as_of = as_of or datetime.now(ZoneInfo("Asia/Kolkata")).date()
    price = float(history["Close"].iloc[-1])
    highs = history["High"] if "High" in history else history["Close"]
    result = {
        "1M Return %": _calendar_return(history["Close"], 1),
        "6M Return %": _calendar_return(history["Close"], 6, as_of=as_of),
        "1Y Return %": _calendar_return(history["Close"], 12, as_of=as_of),
    }
    valid = highs.loc[highs.notna() & (highs > 0) & (highs < float("inf"))]
    high = float(valid.max()) if not valid.empty else None
    result["From All-Time High %"] = (
        (price / high - 1) * 100
        if high is not None and math.isfinite(price) and price > 0
        else None
    )
    return result


def build_watchlist_report(
    symbols: list[str], *, as_of: date | None = None,
) -> pd.DataFrame:
    as_of = as_of or datetime.now(ZoneInfo("Asia/Kolkata")).date()
    rows = []
    for symbol in symbols:
        normalized = normalize_symbol(symbol)
        try:
            history = get_price_history(normalized, period="max")
            snapshot = technical_snapshot(history)
            history = history.sort_index()
            performance = watchlist_price_performance(history, as_of=as_of)
            target = pd.Timestamp(as_of) - pd.DateOffset(years=1)
            reference = history.loc[
                history.index.normalize().date <= target.date()
            ]
            six_month_target = pd.Timestamp(as_of) - pd.DateOffset(months=6)
            six_month_reference = history.loc[
                history.index.normalize().date <= six_month_target.date()
            ]
        except StockDataError:
            continue
        rows.append(
            {
                "Symbol": normalized,
                "Price": snapshot["Price"],
                "Price Date": history.index[-1].date().isoformat(),
                "6M Reference Date": (
                    six_month_reference.index[-1].date().isoformat()
                    if not six_month_reference.empty else None
                ),
                "1Y Reference Date": (
                    reference.index[-1].date().isoformat() if not reference.empty else None
                ),
                "20D Support": snapshot["20D Support"],
                "60D Support": snapshot["60D Support"],
                **performance,
                "RSI": snapshot["RSI"],
                "Above 200 DMA": snapshot["Above 200 DMA"],
                "Technical Score": snapshot["Technical Score"],
            }
        )
    return pd.DataFrame(rows)
