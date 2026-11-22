import json
import os
import re
from datetime import datetime, timedelta
from functools import partial
from threading import Thread

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import truststore
import yfinance as yf
from bs4 import BeautifulSoup
from streamlit.errors import StreamlitSecretNotFoundError

from analytics import (
    StockDataError,
    build_recommendation,
    build_watchlist_report,
    calculate_alpha_beta,
    calculate_period_returns,
    calculate_technicals,
    get_industry_pe,
    get_news,
    get_peer_comparison,
    get_price_history,
    get_stock_data,
    get_stock_news,
    news_sentiment,
    normalize_symbol,
    rank_top_stocks,
    score_stock,
    screen_quality_stocks,
    technical_entry_zone,
    technical_snapshot,
    to_crore,
    value_portfolio,
)
from ipo_screening import (
    INDIA_TIME,
    IPO_SOURCE_URL,
    screen_open_mainboard_ipos,
)
from institutional import rank_institutional_stocks, screen_institutional_buying
from metal_history import parse_city_history, parse_ibja_history
from metal_news import fetch_metal_news
from etf_details import fetch_groww_etf_details
from notifications import (
    NotificationError,
    format_report,
    send_email,
    send_telegram,
)
from storage import (
    add_to_watchlist,
    begin_quality_screen_refresh,
    complete_quality_screen_refresh,
    fail_quality_screen_refresh,
    get_quality_screen_refresh,
    initialize_database,
    list_holdings,
    list_watchlist,
    load_quality_screen_cache,
    remove_from_watchlist,
    remove_holding,
    save_quality_screen_cache,
    upsert_holding,
)


truststore.inject_into_ssl()
st.set_page_config(
    page_title="AI Indian Stock Analyst",
    page_icon="📈",
    layout="wide",
)
initialize_database()


class AIServiceError(RuntimeError):
    pass


@st.cache_data(ttl=900, show_spinner=False)
def cached_stock_data(symbol: str):
    """Cache currency-verified TTM P/E and period-aligned financial ratios."""
    return get_stock_data(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_news(symbol: str):
    return get_news(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_stock_news(symbol: str):
    return get_stock_news(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_history(symbol: str, period: str = "1y"):
    """Cache unadjusted closes for price-only returns and chart levels."""
    return get_price_history(symbol, period=period)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_peers(symbol: str):
    """Cache peers using unadjusted price-only calendar returns."""
    return get_peer_comparison(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_industry_pe(symbol: str):
    """Cache full-industry Screener TTM peer medians, not Nifty 50 subsets."""
    return get_industry_pe(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_open_ipos(today):
    return screen_open_mainboard_ipos(today, include_upcoming=True)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_institutional_buying():
    return screen_institutional_buying()


@st.cache_data(ttl=900, show_spinner=False)
def cached_metal_prices():
    return fetch_metal_prices()


@st.cache_data(ttl=900, show_spinner=False)
def cached_etf_fund_details(symbol: str):
    return fetch_etf_fund_details(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_metal_news(currency: str):
    return fetch_metal_news(currency)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_top_stocks():
    """Cache rankings using unadjusted price-only calendar momentum."""
    return rank_top_stocks()


@st.cache_data(ttl=900, show_spinner=False)
def cached_watchlist_report(symbols: tuple[str, ...], as_of):
    """Cache today-anchored 6M/1Y returns with actual reference dates."""
    return build_watchlist_report(list(symbols), as_of=as_of)


def run_quality_screen_refresh():
    try:
        companies, source, stats = screen_quality_stocks(
            minimum_market_cap_crore=500
        )
        save_quality_screen_cache(companies, source, stats)
    except StockDataError as exc:
        fail_quality_screen_refresh(str(exc))
    else:
        complete_quality_screen_refresh()


def load_quality_result_into_session():
    cached_result = load_quality_screen_cache(max_age_hours=24)
    if cached_result is None:
        return False
    companies, source, stats, generated_at = cached_result
    st.session_state["quality_companies"] = companies
    st.session_state["quality_source"] = source
    st.session_state["quality_stats"] = stats
    st.session_state["quality_generated_at"] = generated_at
    return True


@st.fragment(run_every="5s")
def render_quality_refresh_status():
    refresh = get_quality_screen_refresh()
    if refresh is None:
        return
    if refresh["status"] == "running":
        st.info(
            "A fresh all-NSE scan is running in the background. "
            "The cached table below remains available."
        )
        return
    if refresh["status"] == "failed":
        st.error(f"Background refresh failed: {refresh['error']}")
        return

    cached_result = load_quality_screen_cache(max_age_hours=24)
    if cached_result is None:
        return
    _, _, _, generated_at = cached_result
    if st.session_state.get("quality_generated_at") != generated_at:
        load_quality_result_into_session()
        st.rerun()


def analyze_with_ai(
    stock_name,
    data,
    sentiment,
    technicals,
    score,
    recommendation,
):
    market_cap_crore = data.get("Market Cap (Cr)")
    if market_cap_crore is None:
        market_cap_crore = to_crore(data.get("Market Cap"))

    def rounded(value):
        return round(value, 2) if isinstance(value, float) else value

    ai_facts = {
        "price_inr": rounded(data["Price"]),
        "market_cap_crore": rounded(market_cap_crore),
        "pe_ttm": rounded(data["PE"]),
        "ttm_quarter_ended": data.get("TTM Period"),
        "pe_basis": data.get("PE Note"),
        "financial_period": data.get("Financial Period"),
        "financial_currency": data.get("Financial Currency"),
        "revenue_growth_currency": data.get("Revenue Currency"),
        "financial_data_note": data.get("Data Note"),
        "free_cash_flow": rounded(data.get("FreeCashFlow")),
        "free_cash_flow_currency": data.get("Free Cash Flow Currency"),
        "roe_year_end_equity_percent": (
            rounded(data["ROE"] * 100) if data["ROE"] is not None else None
        ),
        "debt_equity_percent": rounded(data["DebtEquity"]),
        "revenue_growth_percent": (
            rounded(data["RevenueGrowth"] * 100)
            if data["RevenueGrowth"] is not None
            else None
        ),
        "profit_margin_percent": (
            rounded(data["ProfitMargin"] * 100)
            if data["ProfitMargin"] is not None
            else None
        ),
    }
    ai_technicals = {
        key: rounded(value) for key, value in technicals.items()
    }
    prompt = f"""
    Analyze this Indian stock using only the supplied facts.

    Stock: {stock_name}
    Fundamentals: {ai_facts}
    News sentiment: {sentiment["overall"]} ({sentiment["score"]})
    Technical indicators: {ai_technicals}
    Fundamental score: {score}/100
    Deterministic verdict: {recommendation}

    Provide all four sections:
    1. Strengths
    2. Risks
    3. Outlook
    4. Recommendation

    Keep the response under 180 words. The recommendation must exactly match
    the supplied deterministic verdict and reason. Do not suggest a different
    verdict. Do not invent facts or omit the final recommendation.
    This is not financial advice.
    """

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "").strip()
    if not ollama_base_url:
        try:
            ollama_base_url = st.secrets.get("OLLAMA_BASE_URL", "")
        except StreamlitSecretNotFoundError:
            pass
    ollama_base_url = (
        ollama_base_url.strip().rstrip("/") or "http://localhost:11434"
    )
    try:
        with requests.post(
            f"{ollama_base_url}/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": 300, "temperature": 0.2},
            },
            stream=True,
            timeout=300,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                result = json.loads(line)
                if result.get("error"):
                    raise AIServiceError(f"Ollama request failed: {result['error']}")
                if result.get("response"):
                    yield result["response"]
    except requests.ConnectionError as exc:
        raise AIServiceError(
            f"Cannot connect to Ollama at {ollama_base_url}. Set OLLAMA_BASE_URL "
            "to an Ollama server reachable from this Streamlit deployment and "
            "ensure llama3.2:1b is installed there."
        ) from exc
    except requests.RequestException as exc:
        raise AIServiceError(f"Ollama request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AIServiceError("Ollama returned an invalid response.") from exc


def format_percent(value):
    return "N/A" if value is None else f"{value * 100:.2f}%"


def format_decimal(value):
    return "N/A" if value is None else f"{value:,.2f}"


def format_crore(value):
    return "N/A" if value is None else f"₹{value:,.2f} Cr"


def format_cash_flow(data):
    value = data.get("FreeCashFlow")
    currency = data.get("Free Cash Flow Currency")
    if value is None or not currency:
        return "N/A"
    if currency == "INR":
        return format_crore(to_crore(value))
    return f"{currency} {value / 1_000_000:,.2f} million"


def ensure_support_resistance(chart_data):
    chart_data = chart_data.copy()
    if "20D Support" not in chart_data.columns:
        previous_close = chart_data["Close"].shift(1)
        chart_data["20D Support"] = previous_close.rolling(20).min()
    if "20D Resistance" not in chart_data.columns:
        previous_close = chart_data["Close"].shift(1)
        chart_data["20D Resistance"] = previous_close.rolling(20).max()
    if "60D Support" not in chart_data.columns:
        previous_close = chart_data["Close"].shift(1)
        chart_data["60D Support"] = previous_close.rolling(60).min()
    if "60D Resistance" not in chart_data.columns:
        previous_close = chart_data["Close"].shift(1)
        chart_data["60D Resistance"] = previous_close.rolling(60).max()
    return chart_data


def technical_value(technicals, key):
    return technicals.get(key)


def build_price_chart(history, chart_data, chart_type):
    figure = go.Figure()
    if chart_type == "Candlestick":
        figure.add_trace(
            go.Candlestick(
                x=history.index,
                open=history["Open"],
                high=history["High"],
                low=history["Low"],
                close=history["Close"],
                name="Price",
                increasing_line_color="#16a34a",
                decreasing_line_color="#dc2626",
            )
        )
    else:
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data["Close"],
                name="Close",
                line={"color": "#2563eb", "width": 2},
            )
        )

    overlays = (
        ("200 DMA", "#f59e0b", "dot"),
        ("20D Support", "#16a34a", "dash"),
        ("20D Resistance", "#dc2626", "dash"),
        ("60D Support", "#7c3aed", "dashdot"),
    )
    for column, color, dash in overlays:
        figure.add_trace(
            go.Scatter(
                x=chart_data.index,
                y=chart_data[column],
                name=column,
                line={"color": color, "width": 1.5, "dash": dash},
            )
        )

    figure.update_layout(
        height=520,
        margin={"l": 10, "r": 10, "t": 35, "b": 10},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0},
        xaxis={"rangeslider": {"visible": False}},
        yaxis={"title": "Price (₹)", "tickformat": ",.2f"},
    )
    return figure


def render_notification_sidebar():
    with st.sidebar:
        st.header("Notification credentials")
        st.caption("Used only in this browser session and never saved.")
        with st.expander("Email (SMTP)"):
            smtp_host = st.text_input("SMTP host", "smtp.gmail.com")
            smtp_port = st.number_input(
                "SMTP port",
                min_value=1,
                max_value=65535,
                value=465,
            )
            email_user = st.text_input("Email username")
            email_password = st.text_input("Email password", type="password")
            email_to = st.text_input("Recipient email")
        with st.expander("Telegram"):
            telegram_token = st.text_input("Bot token", type="password")
            telegram_chat_id = st.text_input("Chat ID")
    return {
        "smtp_host": smtp_host,
        "smtp_port": int(smtp_port),
        "email_user": email_user,
        "email_password": email_password,
        "email_to": email_to,
        "telegram_token": telegram_token,
        "telegram_chat_id": telegram_chat_id,
    }


def select_stock_for_analysis(table, table_key):
    rows = st.session_state[table_key]["selection"]["rows"]
    if not rows:
        return
    symbol = normalize_symbol(table.iloc[rows[0]]["Symbol"])
    st.session_state["analysis_symbol"] = symbol
    st.session_state["main_tab"] = "Stock Analysis"
    st.session_state["analysis_requested"] = True
    st.session_state.pop("analysis", None)
    st.session_state["stock_table_version"] = (
        st.session_state.get("stock_table_version", 0) + 1
    )


def render_stock_table(table, formats, key):
    table_key = f"{key}-{st.session_state.get('stock_table_version', 0)}"
    st.caption("Click a row to open that stock in Stock Analysis.")
    st.dataframe(
        table.style.format(formats, na_rep="N/A"),
        hide_index=True,
        use_container_width=True,
        key=table_key,
        selection_mode="single-row",
        on_select=partial(select_stock_for_analysis, table, table_key),
    )


def render_stock_news_sentiment(symbol, sentiment, error=None):
    st.caption(
        f"News matching {symbol}, from Moneycontrol, Reuters, Economic Times, Mint, "
        "CNBC/CNBC-TV18 and Business Standard. Retrieved via Google News RSS; "
        "last 30 days. Sentiment is a keyword-based headline summary, not investment advice."
    )
    if error:
        st.warning(error)
    elif not sentiment["headlines"]:
        st.info("No recent company-matched headlines from the selected publishers are available.")
    else:
        st.metric("Aggregate headline sentiment", sentiment["overall"])
        headline_frame = pd.DataFrame(sentiment["headlines"]).reindex(
            columns=["title", "publisher", "published", "sentiment", "link"]
        )
        st.dataframe(
            headline_frame,
            column_config={"link": st.column_config.LinkColumn("Article", display_text="Read article")},
            hide_index=True,
            use_container_width=True,
        )


def render_analysis_company_news(symbol):
    st.subheader(f"Company News - {symbol}")
    if st.button("Load company news", key=f"analysis-company-news-{symbol}"):
        try:
            with st.spinner(f"Loading company news for {symbol}..."):
                news = cached_stock_news(symbol)
        except StockDataError as exc:
            st.session_state["analysis_company_news"] = {
                "symbol": symbol, "sentiment": news_sentiment([]), "error": str(exc),
            }
        else:
            st.session_state["analysis_company_news"] = {
                "symbol": symbol, "sentiment": news_sentiment(news), "error": None,
            }
    result = st.session_state.get("analysis_company_news")
    if result is None or result["symbol"] != symbol:
        st.info(f"Select Load company news for {symbol}. News is cached up to 15 minutes.")
        return
    render_stock_news_sentiment(symbol, result["sentiment"], result["error"])


def render_analysis_tab():
    st.subheader("Stock research")
    st.session_state.setdefault("analysis_symbol", "TCS.NS")
    analysis_requested = st.session_state.pop("analysis_requested", False)
    with st.form("stock-analysis-form"):
        input_column, benchmark_column, risk_free_column, button_column = st.columns(
            [3, 2, 2, 1]
        )
        symbol_input = input_column.text_input(
            "NSE/Yahoo symbol",
            key="analysis_symbol",
            help="The .NS suffix is added automatically when omitted.",
        )
        benchmark_symbol = benchmark_column.text_input(
            "Market benchmark",
            value="^NSEI",
            key="analysis_benchmark",
            help="Yahoo Finance symbol for the benchmark, e.g. ^NSEI.",
        ).strip().upper()
        risk_free_rate = risk_free_column.number_input(
            "Annual risk-free rate (%)",
            min_value=0.0,
            max_value=30.0,
            value=6.5,
            step=0.25,
            key="analysis_risk_free_rate",
        ) / 100
        analyze = button_column.form_submit_button(
            "Analyze",
            type="primary",
            use_container_width=True,
        )

    if analyze or analysis_requested:
        symbol = normalize_symbol(symbol_input)
        if not symbol:
            st.error("Enter a stock symbol.")
            return
        if not benchmark_symbol:
            st.error("Enter a market benchmark symbol.")
            return
        try:
            with st.spinner("Loading live fundamentals, news, and price history..."):
                data = cached_stock_data(symbol)
                news = cached_news(symbol)
                history = cached_history(symbol)
                try:
                    stock_period_returns = calculate_period_returns(
                        cached_history(symbol, period="5y")
                    )
                    stock_period_returns_error = None
                except StockDataError as exc:
                    stock_period_returns = None
                    stock_period_returns_error = str(exc)
                try:
                    benchmark_history = cached_history(benchmark_symbol)
                    risk_metrics = calculate_alpha_beta(
                        history, benchmark_history, risk_free_rate,
                    )
                    risk_metrics_error = None
                except StockDataError as exc:
                    risk_metrics = None
                    risk_metrics_error = str(exc)
                try:
                    benchmark_period_returns = calculate_period_returns(
                        cached_history(benchmark_symbol, period="5y")
                    )
                    benchmark_period_returns_error = None
                except StockDataError as exc:
                    benchmark_period_returns = None
                    benchmark_period_returns_error = str(exc)
                sentiment = news_sentiment(news)
                technicals = technical_snapshot(history)
                try:
                    industry_pe, industry_pe_note = cached_industry_pe(symbol)
                except StockDataError as exc:
                    industry_pe = None
                    industry_pe_note = f"Industry P/E unavailable: {exc}"
        except StockDataError as exc:
            st.error(str(exc))
            return
        st.session_state["analysis"] = {
            "symbol": symbol,
            "data": data,
            "news": news,
            "sentiment": sentiment,
            "history": history,
            "benchmark_symbol": benchmark_symbol,
            "risk_metrics": risk_metrics,
            "risk_metrics_error": risk_metrics_error,
            "risk_free_rate": risk_free_rate,
            "stock_period_returns": stock_period_returns,
            "stock_period_returns_error": stock_period_returns_error,
            "benchmark_period_returns": benchmark_period_returns,
            "benchmark_period_returns_error": benchmark_period_returns_error,
            "technicals": technicals,
            "industry_pe": industry_pe,
            "industry_pe_note": industry_pe_note,
            "score": score_stock(data),
        }

    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("Enter a symbol and select Analyze.")
        return

    symbol = analysis["symbol"]
    data = analysis["data"]
    technicals = analysis["technicals"]
    score = analysis["score"]
    sentiment = analysis["sentiment"]
    market_cap_crore = data.get("Market Cap (Cr)")
    if market_cap_crore is None:
        market_cap_crore = to_crore(data.get("Market Cap"))
    recommendation = build_recommendation(
        data,
        score,
        technicals,
        sentiment,
    )

    st.markdown(f"### {symbol}")
    st.metric("Price", f"₹{data['Price']:,.2f}")
    stock_pe_column, industry_pe_column = st.columns(2)
    stock_pe_column.metric(
        "Stock P/E (TTM)", format_decimal(data["PE"]), help=data.get("PE Note"),
    )
    industry_pe_column.metric(
        "Industry P/E (TTM peer median)",
        format_decimal(analysis["industry_pe"]),
        help=analysis["industry_pe_note"],
    )
    st.caption(
        analysis["industry_pe_note"]
    )
    st.caption(
        f"Annual financial period: {data.get('Financial Period') or 'Unavailable'}. "
        f"Latest price date: {data.get('Price Date') or 'Unavailable'}. "
        f"Statement currency: {data.get('Financial Currency') or 'Unverified'}. "
        f"Stock P/E: {data.get('PE Note', 'Unavailable')}. "
        "TTM uses four consecutive quarters or FY + current YTD - prior YTD; "
        "annual profit alone is valid only when the FY end equals the TTM end. "
        "ROE uses annual shareholder profit / year-end equity (not average equity); "
        "debt/equity is shown as a percentage: 50% = 0.5x. "
        "Revenue growth is annual YoY in the reported currency, not necessarily INR. "
        "Profit margin uses Yahoo Total Revenue (for banks, this may be net revenue); "
        "debt includes Yahoo Total Debt, which can differ from website borrowings. "
        "Prices are latest available daily closes, "
        "not guaranteed real-time quotes."
    )
    if data["PE"] is None:
        st.warning(data.get("PE Note") or "TTM P/E unavailable.")
    if data.get("Data Note"):
        st.warning(data["Data Note"])
    metric_columns = st.columns(4)
    metric_columns[0].metric("Fundamental Score", f"{score}/100")
    metric_columns[1].metric("ROE", format_percent(data["ROE"]))
    metric_columns[2].metric(
        "RSI",
        format_decimal(technicals["RSI"]),
    )
    metric_columns[3].metric("News Sentiment", sentiment["overall"])

    st.markdown("#### Recommendation")
    verdict_column, confidence_column = st.columns(2)
    verdict_column.metric("Verdict", recommendation["verdict"])
    confidence_column.metric(
        "Confidence",
        f"{recommendation['confidence']:.2f}/10",
    )
    st.caption(
        "Scores and confidence are rule-based heuristics, not calibrated "
        "probabilities of returns. Charts and technicals use unadjusted daily "
        "prices and Wilder RSI(14). Returns exclude dividends: latest close / "
        "close on or before the calendar lookback date - 1. They are not live "
        "intraday returns; Yahoo may still normalize historical prices for splits."
    )
    st.markdown("**Problems**")
    if recommendation["problems"]:
        for problem in recommendation["problems"]:
            st.markdown(f"❌ {problem}")
    else:
        st.markdown("✅ No major financial red flags detected")
    st.markdown(f"**Reason:** {recommendation['reason']}")

    entry_zone = technical_entry_zone(technicals)
    st.markdown("#### Technical entry zone")
    if entry_zone["low"] is None:
        st.info(entry_zone["reason"])
    else:
        entry_low, entry_high = st.columns(2)
        entry_low.metric("Potential entry from", f"₹{entry_zone['low']:,.2f}")
        entry_high.metric("Potential entry to", f"₹{entry_zone['high']:,.2f}")
        st.info(f"**{entry_zone['status']}** — {entry_zone['reason']}")
    st.caption(
        "Uses 20-day and 60-day support confluence with a 2% buffer, checked "
        "against RSI and the 200-day average. If the support levels diverge "
        "by more than 5%, the app withholds an entry zone. This is a technical "
        "reference, not financial advice."
    )

    action_column, _ = st.columns([1, 4])
    if action_column.button("Add to watchlist", key=f"watch-{symbol}"):
        add_to_watchlist(symbol)
        st.success(f"{symbol} added to the watchlist.")

    fundamental_tab, technical_tab, risk_tab, news_tab, company_news_tab, ai_tab = st.tabs(
        [
            "Fundamentals", "Technicals", "Risk & Return", "News sentiment",
            "Company News", "AI recommendation",
        ]
    )
    with fundamental_tab:
        fundamentals = pd.DataFrame(
            [
                {
                    "Metric": "Market Cap",
                    "Value": format_crore(market_cap_crore),
                },
                {"Metric": "P/E (TTM)", "Value": format_decimal(data["PE"])},
                {"Metric": "P/E (latest annual, comparison only)", "Value": format_decimal(data.get("Annual PE"))},
                {
                    "Metric": "Industry P/E (TTM peer median)",
                    "Value": format_decimal(analysis["industry_pe"]),
                },
                {"Metric": "ROE (annual / year-end equity)", "Value": format_percent(data["ROE"])},
                {
                    "Metric": "Debt / Equity (%)",
                    "Value": format_decimal(data["DebtEquity"]),
                },
                {
                    "Metric": f"Revenue Growth (annual YoY, {data.get('Revenue Currency') or 'currency unverified'})",
                    "Value": format_percent(data["RevenueGrowth"]),
                },
                {
                    "Metric": "Profit Margin (annual)",
                    "Value": format_percent(data["ProfitMargin"]),
                },
                {
                    "Metric": "Free Cash Flow (annual)",
                    "Value": format_cash_flow(data),
                },
            ]
        )
        st.dataframe(fundamentals, hide_index=True, use_container_width=True)
        st.caption(analysis["industry_pe_note"])

    with technical_tab:
        history = analysis["history"]
        chart_data = ensure_support_resistance(calculate_technicals(history))
        st.caption(
            "Daily MACD(12,26,9): EMA(12) - EMA(26); signal = EMA(9) of MACD. "
            "EMAs use alpha = 2/(period + 1), seeded from the first available "
            "close (signal from the first MACD). Values are price units, not "
            "percentages. Different history lengths, EMA seeds, intraday prices "
            "or adjusted prices can differ from other platforms."
        )
        st.caption(
            "Support and resistance use the lowest and highest closes from "
            "the previous 20 and 60 trading sessions."
        )
        chart_type = st.radio(
            "Price chart type",
            ["Line", "Candlestick"],
            horizontal=True,
            key=f"price-chart-type-{symbol}",
        )
        st.plotly_chart(
            build_price_chart(history, chart_data, chart_type),
            use_container_width=True,
            key=f"price-chart-{symbol}",
        )
        macd_column, rsi_column = st.columns(2)
        macd_column.line_chart(chart_data[["MACD", "MACD Signal"]])
        rsi_column.line_chart(chart_data[["RSI"]])
        technical_rows = [
            {"Metric": "Price", "Value": f"₹{technicals['Price']:,.2f}"},
            {"Metric": "RSI", "Value": format_decimal(technicals["RSI"])},
            {"Metric": "MACD", "Value": format_decimal(technicals["MACD"])},
            {
                "Metric": "MACD Signal",
                "Value": format_decimal(technicals["MACD Signal"]),
            },
            {
                "Metric": "200 DMA",
                "Value": (
                    "N/A"
                    if technicals["200 DMA"] is None
                    else f"₹{technicals['200 DMA']:,.2f}"
                ),
            },
            {
                "Metric": "20D Support",
                "Value": (
                    "N/A"
                    if technical_value(technicals, "20D Support") is None
                    else f"₹{technical_value(technicals, '20D Support'):,.2f}"
                ),
            },
            {
                "Metric": "20D Resistance",
                "Value": (
                    "N/A"
                    if technical_value(technicals, "20D Resistance") is None
                    else f"₹{technical_value(technicals, '20D Resistance'):,.2f}"
                ),
            },
            {
                "Metric": "60D Support",
                "Value": (
                    "N/A"
                    if technical_value(technicals, "60D Support") is None
                    else f"₹{technical_value(technicals, '60D Support'):,.2f}"
                ),
            },
            {
                "Metric": "Above 200 DMA",
                "Value": "Yes" if technicals["Above 200 DMA"] else "No",
            },
            {
                "Metric": "6M Return",
                "Value": (
                    "N/A"
                    if technicals["6M Return %"] is None
                    else f"{technicals['6M Return %']:.2f}%"
                ),
            },
            {
                "Metric": "Technical Score",
                "Value": f"{technicals['Technical Score']}/100",
            },
        ]
        st.dataframe(
            pd.DataFrame(technical_rows),
            hide_index=True,
            use_container_width=True,
        )

    with risk_tab:
        st.markdown("#### Alpha and beta")
        st.markdown("#### Trailing annualized returns")
        stock_returns = analysis.get("stock_period_returns")
        benchmark_returns = analysis.get("benchmark_period_returns")
        return_rows = []
        for period, label in (
            ("1Y", "1 Year"),
            ("3Y", "3 Years"),
            ("5Y", "5 Years"),
        ):
            return_rows.append(
                {
                    "Period": label,
                    f"{symbol} CAGR": format_percent(
                        stock_returns.get(period) if stock_returns else None
                    ),
                    f"{analysis.get('benchmark_symbol', '^NSEI')} CAGR": format_percent(
                        benchmark_returns.get(period)
                        if benchmark_returns
                        else None
                    ),
                }
            )
        st.dataframe(
            pd.DataFrame(return_rows),
            hide_index=True,
            use_container_width=True,
        )
        if analysis.get("stock_period_returns_error"):
            st.warning(
                "Stock trailing returns unavailable: "
                f"{analysis['stock_period_returns_error']}"
            )
        if analysis.get("benchmark_period_returns_error"):
            st.warning(
                "Benchmark trailing returns unavailable: "
                f"{analysis['benchmark_period_returns_error']}"
            )
        st.caption(
            "Trailing compound annual growth rates (CAGR), using unadjusted "
            "closes from the latest available price to the last close on or "
            "before each calendar-year lookback date. Returns exclude dividends; "
            "periods without enough price history are shown as N/A."
        )
        risk_metrics = analysis.get("risk_metrics")
        risk_metrics_error = analysis.get("risk_metrics_error")
        if risk_metrics_error:
            st.warning(
                f"Risk metrics unavailable: {risk_metrics_error}"
            )
        elif "risk_metrics" not in analysis:
            st.info("Select Analyze again to calculate alpha and beta.")
        elif risk_metrics is None:
            st.info(
                "Risk metrics are unavailable because the aligned price "
                "history is too short or the benchmark has no variation."
            )
        else:
            risk_columns = st.columns(3)
            risk_columns[0].metric(
                f"Beta (vs {analysis.get('benchmark_symbol', '^NSEI')})",
                format_decimal(risk_metrics["beta"]),
            )
            risk_columns[1].metric(
                "Annualized alpha",
                format_percent(risk_metrics["alpha"]),
            )
            risk_columns[2].metric(
                "CAPM expected return",
                format_percent(risk_metrics["expected_return"]),
            )
            st.caption(
                f"Annualized actual return: "
                f"{format_percent(risk_metrics['actual_return'])}; "
                f"annualized benchmark return: "
                f"{format_percent(risk_metrics['market_return'])}; "
                f"risk-free rate: "
                f"{analysis.get('risk_free_rate', 0.065):.2%}; "
                f"{risk_metrics['observations']} aligned daily returns. "
                "Beta is covariance(stock, market) / variance(market), "
                "using daily close-to-close price returns. Alpha is actual "
                "annualized return minus the CAPM expected annual return. "
                "Price returns exclude dividends; estimates use the available "
                "history (normally one year) and are not investment advice."
            )

    with news_tab:
        st.metric("Aggregate headline sentiment", sentiment["overall"])
        headline_frame = pd.DataFrame(sentiment["headlines"])
        st.dataframe(
            headline_frame[["title", "published", "sentiment"]],
            hide_index=True,
            use_container_width=True,
        )

    with company_news_tab:
        render_analysis_company_news(symbol)

    with ai_tab:
        st.caption("Generated with Ollama llama3.2:1b.")
        try:
            with st.spinner("Generating analysis...", show_time=True):
                st.write_stream(
                    analyze_with_ai(
                        symbol,
                        data,
                        sentiment,
                        technicals,
                        score,
                        recommendation,
                    )
                )
        except AIServiceError as exc:
            st.warning(str(exc))


def render_peers_tab():
    st.subheader("Nifty peer comparison")
    symbol = normalize_symbol(st.text_input("Base symbol", "TCS", key="peer-symbol"))
    if st.button("Compare peers", type="primary"):
        try:
            with st.spinner("Loading online peer market data..."):
                peers, source = cached_peers(symbol)
        except StockDataError as exc:
            st.error(str(exc))
            return
        st.caption(f"Peer universe source: {source}")
        st.dataframe(
            peers.style.format(
                {
                    "Price": "₹{:,.2f}",
                    "Market Cap (Cr)": "₹{:,.2f} Cr",
                    "RSI": "{:.2f}",
                    "6M Return %": "{:.2f}%",
                },
                na_rep="N/A",
            ),
            hide_index=True,
            use_container_width=True,
        )


def render_portfolio_tab():
    st.subheader("Portfolio analysis")
    with st.form("holding-form"):
        columns = st.columns(3)
        symbol = normalize_symbol(columns[0].text_input("Symbol", "TCS"))
        quantity = columns[1].number_input(
            "Quantity",
            min_value=0.0001,
            value=1.0,
        )
        average_price = columns[2].number_input(
            "Average buy price",
            min_value=0.0,
            value=1000.0,
        )
        if st.form_submit_button("Add or update holding", type="primary"):
            upsert_holding(symbol, quantity, average_price)
            st.success(f"Saved {symbol}.")

    holdings = list_holdings()
    if not holdings:
        st.info("Add a holding to begin portfolio analysis.")
        return

    try:
        with st.spinner("Valuing portfolio with live prices..."):
            valuation = value_portfolio(holdings)
    except StockDataError as exc:
        st.error(str(exc))
        return

    if valuation.empty:
        st.warning("No live prices were available for the saved holdings.")
        return

    total_invested = valuation["Invested"].sum()
    st.caption(
        "Valuation uses unadjusted latest daily closes; P/L excludes fees, taxes "
        "and dividends. Percentage P/L is undefined (N/A) when invested cost is zero."
    )
    total_value = valuation["Current Value"].sum()
    metrics = st.columns(3)
    metrics[0].metric("Invested", f"₹{total_invested:,.2f}")
    metrics[1].metric("Current value", f"₹{total_value:,.2f}")
    metrics[2].metric(
        "Total P/L",
        f"₹{total_value - total_invested:,.2f}",
        f"{((total_value / total_invested) - 1) * 100:.2f}%"
        if total_invested
        else None,
    )
    st.dataframe(
        valuation.style.format(
            {
                "Quantity": "{:,.2f}",
                "Average Price": "₹{:,.2f}",
                "Current Price": "₹{:,.2f}",
                "Invested": "₹{:,.2f}",
                "Current Value": "₹{:,.2f}",
                "P/L": "₹{:,.2f}",
                "P/L %": "{:.2f}%",
                "Allocation %": "{:.2f}%",
            },
            na_rep="N/A",
        ),
        hide_index=True,
        use_container_width=True,
    )
    remove_symbol = st.selectbox(
        "Remove holding",
        [holding["symbol"] for holding in holdings],
    )
    if st.button("Remove selected holding"):
        remove_holding(remove_symbol)
        st.rerun()


def render_watchlist_tab(credentials):
    if st.session_state.get("main_tab", "Stock Analysis") != "Watchlist & Alerts":
        return

    st.subheader("Watchlist and on-demand alerts")
    add_column, button_column = st.columns([4, 1])
    symbol = normalize_symbol(
        add_column.text_input("Add symbol", "INFY", key="watchlist-symbol")
    )
    if button_column.button("Add", use_container_width=True):
        add_to_watchlist(symbol)
        st.rerun()

    symbols = list_watchlist()
    if not symbols:
        st.info("Add stocks to build a watchlist report.")
        return

    try:
        with st.spinner("Building report from live online prices..."):
            report = cached_watchlist_report(
                tuple(symbols), datetime.now(INDIA_TIME).date(),
            )
    except StockDataError as exc:
        st.error(str(exc))
        return

    st.caption(
        "20D and 60D support are the lowest closing prices over the previous "
        "20 and 60 trading sessions, excluding the current session. "
        "1M, 6M and 1Y returns use (latest close / earlier close - 1) "
        "× 100. 1M looks back from the latest price date; 6M and 1Y look back from "
        "today's India date. The session on or before each target date is used. "
        "Price Date, 6M Reference Date and 1Y Reference Date show the actual sessions; when Price "
        "Date is older than today, 6M/1Y mix the latest available quote with "
        "today's anniversary reference. "
        "Positive means a gain; negative means a loss. All-time-high distance "
        "uses (current close / high - 1) × 100 over all available unadjusted Yahoo "
        "history, including the latest session; closing highs are used if daily "
        "High data is absent. Dividend adjustments are not included. "
        "N/A means there is not enough price history."
    )
    render_stock_table(
        report,
        {
            "Price": "₹{:,.2f}",
            "20D Support": "₹{:,.2f}",
            "60D Support": "₹{:,.2f}",
            "1M Return %": "{:+.2f}%",
            "1Y Return %": "{:+.2f}%",
            "From All-Time High %": "{:+.2f}%",
            "RSI": "{:.2f}",
            "6M Return %": "{:.2f}%",
        },
        key="watchlist-table",
    )
    report_text = format_report(report)
    with st.expander("Alert preview"):
        st.code(report_text)

    email_column, telegram_column = st.columns(2)
    if email_column.button("Send email now", use_container_width=True):
        try:
            send_email(
                smtp_host=credentials["smtp_host"],
                smtp_port=credentials["smtp_port"],
                username=credentials["email_user"],
                password=credentials["email_password"],
                recipient=credentials["email_to"],
                report_text=report_text,
            )
        except NotificationError as exc:
            st.error(str(exc))
        else:
            st.success("Watchlist email sent.")

    if telegram_column.button("Send Telegram now", use_container_width=True):
        try:
            send_telegram(
                credentials["telegram_token"],
                credentials["telegram_chat_id"],
                report_text,
            )
        except NotificationError as exc:
            st.error(str(exc))
        else:
            st.success("Telegram alert sent.")

    remove_symbol = st.selectbox("Remove from watchlist", symbols)
    if st.button("Remove selected stock"):
        remove_from_watchlist(remove_symbol)
        cached_watchlist_report.clear()
        st.rerun()


def render_top_picks_tab():
    st.subheader("Top 10 Nifty technical picks")
    st.caption(
        "Ranks the online Nifty universe using six-month momentum, RSI, MACD, "
        "and the 200-day moving average. This is not financial advice."
    )
    if st.button("Generate Top 10", type="primary"):
        try:
            with st.spinner("Screening live online Nifty price history..."):
                picks, source = cached_top_stocks()
        except StockDataError as exc:
            st.error(str(exc))
            return
        st.session_state["top_picks"] = picks
        st.session_state["top_picks_source"] = source

    picks = st.session_state.get("top_picks")
    if picks is None:
        return
    st.caption(f"Universe source: {st.session_state['top_picks_source']}")
    render_stock_table(
        picks,
        {
            "Price": "₹{:,.2f}",
            "RSI": "{:.2f}",
            "6M Return %": "{:.2f}%",
        },
        key="top-picks-table",
    )


def render_quality_screen_tab():
    st.subheader("Strong companies across all NSE-listed equities")
    st.caption(
        "Scans the official NSE equity master, not only the Nifty 50. Rules: "
        "market cap above ₹500 Cr, technical score of at least 80, price above "
        "the 200 DMA, positive latest quarterly sales growth, positive latest "
        "quarterly profit growth, and positive latest-quarter profit. A match "
        "is a research candidate, not a guaranteed buy recommendation."
    )
    st.caption(
        "Quarterly growth compares consecutive reported quarters, not YoY. "
        "Profit growth uses (latest PAT - prior PAT) / abs(prior PAT); "
        "a zero prior PAT cannot produce a percentage. Cached results must "
        "be refreshed to reflect calculation changes."
    )
    if "quality_companies" not in st.session_state:
        load_quality_result_into_session()

    if st.button("Refresh All NSE Screen (slow)", type="primary"):
        if begin_quality_screen_refresh():
            Thread(
                target=run_quality_screen_refresh,
                daemon=True,
                name="quality-screen-refresh",
            ).start()
            st.success(
                "Refresh started in the background. You can continue using "
                "the dashboard and the cached results below."
            )
        else:
            st.info("An all-NSE refresh is already running.")

    refresh = get_quality_screen_refresh()
    if refresh is not None and refresh["status"] == "running":
        render_quality_refresh_status()
    elif refresh is not None and refresh["status"] == "failed":
        st.error(f"Background refresh failed: {refresh['error']}")

    companies = st.session_state.get("quality_companies")
    if companies is None:
        return

    stats = st.session_state["quality_stats"]
    metric_columns = st.columns(5)
    metric_columns[0].metric("NSE equities", stats["universe"])
    metric_columns[1].metric(
        "Price history found",
        stats.get("price_history_available", stats["universe"]),
    )
    metric_columns[2].metric(
        "Technically strong",
        stats["technical_candidates"],
    )
    metric_columns[3].metric("Final matches", stats["matches"])
    metric_columns[4].metric(
        "Quarterly unavailable",
        stats.get("quarterly_unavailable", stats.get("unavailable", 0)),
    )
    st.caption(f"Universe source: {st.session_state['quality_source']}")
    generated_at = st.session_state.get("quality_generated_at")
    if generated_at:
        st.caption(
            f"Cached result generated at {generated_at}. "
            "Results are reused for up to 24 hours."
        )

    if companies.empty:
        st.info("No companies currently meet every quality-screen rule.")
        return

    render_stock_table(
        companies,
        {
            "Price": "₹{:,.2f}",
            "Market Cap (Cr)": "₹{:,.2f} Cr",
            "Sales QoQ %": "{:.2f}%",
            "Profit QoQ %": "{:.2f}%",
            "RSI": "{:.2f}",
            "6M Return %": "{:.2f}%",
            "Technical Score": "{:.2f}",
        },
        key="quality-screen-table",
    )


def render_ipo_screen_tab():
    st.subheader("Open and upcoming mainboard IPOs: fundamentals-only screening")
    st.caption(
        "Source: Chittorgarh. Includes open issues within their subscription dates "
        "in India and upcoming issues. Upcoming IPOs are not yet open for application; "
        "their assessments are provisional. SME and closed issues are excluded. "
        "Sector context is included in the fundamental assessment. GMP is shown "
        "separately and does not affect the verdict; subscription demand and "
        "technical indicators are not used."
    )
    st.link_button("View Chittorgarh mainboard IPOs", IPO_SOURCE_URL)
    if st.button("Refresh IPOs", type="primary"):
        cached_open_ipos.clear()
        st.session_state["ipo_screen_requested"] = True
    if not st.session_state.get("ipo_screen_requested"):
        st.info("Select Refresh IPOs to retrieve open and upcoming mainboard issues.")
        return

    try:
        with st.spinner("Loading open/upcoming IPOs and company financials from Chittorgarh..."):
            report = cached_open_ipos(datetime.now(INDIA_TIME).date())
    except StockDataError as exc:
        st.error(str(exc))
        return

    st.caption(
        f"Source checked at {report.checked_at.strftime('%d %b %Y %H:%M IST')}. "
        "Results are cached for up to 15 minutes; refresh before deciding to apply. "
        "Open dates do not confirm broker-specific intraday application cutoffs."
    )
    for warning in report.warnings:
        st.warning(warning)
    if not report.issues:
        if report.warnings:
            st.info("No complete assessments are available. See the source warnings above.")
        else:
            st.info("No open or upcoming mainboard IPOs are currently available.")
        return

    st.markdown("### Open and upcoming IPO summary")
    summary = pd.DataFrame([
        {
            "Company name": issue.name,
            "Status": issue.status,
            "GMP (%)": issue.gmp.percentage if issue.gmp is not None else None,
            "GMP (INR)": issue.gmp.premium if issue.gmp is not None else None,
            "Price band": issue.price_band,
            "Opening date": issue.opens.strftime("%d %b %Y") if issue.opens else "Unconfirmed",
            "Closing date": issue.closes.strftime("%d %b %Y") if issue.closes else "Unconfirmed",
            "Allotment date": (
                issue.allotment_date.strftime("%d %b %Y")
                if issue.allotment_date else "Unavailable"
            ),
        }
        for issue in report.issues
    ])
    st.dataframe(
        summary.style.format(
            {"GMP (INR)": "₹{:+,.2f}", "GMP (%)": "{:+.2f}%"},
            na_rep="Unavailable",
        ).map(
            lambda value: (
                "" if pd.isna(value) or value == 0
                else "color: green" if value > 0 else "color: red"
            ),
            subset=["GMP (INR)", "GMP (%)"],
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "GMP is an unofficial premium per share, not the IPO offer price. "
        "Green = positive, red = negative; zero and unavailable quotes are neutral. "
        "GMP percentages are source-reported. Allotment dates are tentative; "
        "unpublished or unreadable dates are shown as unavailable."
    )

    st.info(
        "APPLY CANDIDATE means all six screening checks passed, not a guarantee "
        "or personalized advice. CAUTION means mixed or incomplete evidence. "
        "AVOID means a full-year loss, non-positive net worth, or at least three "
        "failed checks. Sector valuation uses available recent mainboard IPO "
        "peers, not an industry-wide or current traded P/E. Small samples and "
        "different business models limit comparability."
    )
    with st.expander("Screening rules and limitations"):
        st.write(
            "Requires two consecutive March year-end financials. Annual income "
            "and profit must grow; PAT margin must be at least 10%; ROE proxy "
            "(annual PAT / year-end net worth) at least 15%; debt/equity at most "
            "0.5x for non-financial businesses; and annual P/E at or below the "
            "available sector IPO peer median. Without peer data, 25x is only a "
            "generic reference and cannot produce Apply Candidate. Financial "
            "businesses require separate sector-specific review, not the generic "
            "debt threshold. Annual P/E uses post-IPO market "
            "capitalization at the offer price divided by latest annual PAT. "
            "Interim losses prevent an Apply Candidate result; interim profits "
            "are never annualized or compared against full years. Missing metrics "
            "and annual financials older than 550 days prevent an Apply Candidate "
            "result. These are screening heuristics, not validated return forecasts."
        )
        st.write(
            "Cash-flow quality, related-party transactions, litigation, business "
            "concentration and use of proceeds are not fully assessed. Sector "
            "peer P/Es are historical offer valuations from Chittorgarh's recently "
            "listed IPO table; only positive mainboard P/Es are used. Read the RHP "
            "and consult a SEBI-registered adviser "
            "before investing. No listing-gain prediction is made."
        )

    for issue in report.issues:
        with st.expander(
            f"{issue.name} | {issue.status} | {issue.assessment.verdict}", expanded=False,
        ):
            st.write(f"Status: {issue.status}")
            if issue.opens is not None and issue.closes is not None:
                st.write(
                    f"Subscription: {issue.opens:%d %b %Y} to {issue.closes:%d %b %Y}"
                )
            else:
                st.write("Subscription dates: Unconfirmed")
            if issue.status == "Upcoming":
                st.info("Not yet open for application. Recheck dates and pricing before applying.")
            st.write(f"Price band: {issue.price_band} | Lot size: {issue.lot_size}")
            st.write(f"Sale type: {issue.sale_type}")
            st.write(f"Sector / industry: {issue.sector}")
            st.write(
                "Recent mainboard sector IPO peer P/E median: "
                f"{format_decimal(issue.sector_peer_pe)} "
                f"({issue.sector_peer_count} positive-P/E peers)"
            )
            st.caption("Historical IPO offer P/Es, not the current industry P/E.")
            st.markdown("**Grey-market premium (GMP)**")
            if issue.gmp is None or issue.gmp.premium is None:
                st.write("GMP: Unavailable")
                if issue.gmp is not None:
                    st.caption(issue.gmp.note)
            else:
                premium = issue.gmp.premium
                text = f"GMP: ₹{premium:+,.2f} per share"
                if issue.gmp.percentage is not None:
                    text += f" ({issue.gmp.percentage:+.2f}%)"
                else:
                    text += " (% unavailable)"
                if premium > 0:
                    st.markdown(f":green[**{text}**]")
                elif premium < 0:
                    st.markdown(f":red[**{text}**]")
                else:
                    st.markdown(f"**{text} (no premium)**")
                st.caption(f"Source quote time (as reported): {issue.gmp.quoted_at}")
            if issue.gmp is not None:
                st.link_button("GMP source", issue.gmp.source_url)
            st.caption(
                "GMP is an unofficial, unregulated premium over the offer price; "
                "the percentage is as reported by the source, "
                "not a guaranteed listing price. Quotes may be stale or volatile; "
                "a green GMP does not mean the IPO is fundamentally good."
            )
            st.link_button("IPO details and prospectus links", issue.url)
            st.markdown(f"**Assessment: {issue.assessment.verdict}**")
            if issue.assessment.strengths:
                st.markdown("**Strengths**")
                for strength in issue.assessment.strengths:
                    st.write(f"- {strength}")
            if issue.assessment.risks:
                st.markdown("**Risks / missing evidence**")
                for risk in issue.assessment.risks:
                    st.write(f"- {risk}")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Metric": key, "Value": format_decimal(value)}
                        for key, value in issue.assessment.metrics.items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            financials = issue.financials.copy()
            financials.columns = [
                period.strftime("%d %b %Y") for period in financials.columns
            ]
            st.caption("Reported financials in INR crore; interim periods are shown separately.")
            st.dataframe(
                financials.style.format("{:,.2f}", na_rep="Unavailable"),
                width="stretch",
            )


def fetch_indian_bullion_spot():
    """Fetch daily Indian bullion benchmark rates from IBJA."""
    response = requests.get("https://ibjarates.com", timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    gold_rates = {}
    for purity in ("999", "916", "750"):
        element = soup.find(id=f"GoldRatesCompare{purity}")
        if element is not None:
            value = element.get_text(strip=True).replace(",", "")
            if value.isdigit():
                gold_rates[purity] = float(value)

    silver_element = soup.select_one('td[data-label="Silver 999"]')
    silver_per_gram = None
    if silver_element is not None:
        value = silver_element.get_text(strip=True).replace(",", "")
        if value.isdigit():
            silver_per_gram = float(value) / 1000

    if "999" not in gold_rates or silver_per_gram is None:
        raise StockDataError("Indian spot bullion prices are currently unavailable.")

    return {
        "gold_24k_per_10g": gold_rates["999"] * 10,
        "gold_22k_per_10g": gold_rates.get("916", gold_rates["999"] * 916 / 999) * 10,
        "silver_per_10g": silver_per_gram * 10,
        "history": parse_ibja_history(soup, datetime.now(INDIA_TIME).date()),
    }


def fetch_indian_city_market(city):
    """Fetch indicative retail market rates for an Indian city."""
    city_slug = city.lower()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }
    response = requests.get(
        f"https://www.goodreturns.in/gold-rates/{city_slug}.html",
        timeout=20,
        headers=headers,
    )
    response.raise_for_status()
    gold_soup = BeautifulSoup(response.text, "html.parser")
    gold_text = gold_soup.get_text(" ", strip=True)
    gold_match = re.search(
        rf"gold price in {city}.*?₹([\d,]+).*?₹([\d,]+)",
        gold_text,
        re.IGNORECASE | re.DOTALL,
    )
    if gold_match is None:
        gold_match = re.search(
            rf"gold price.*?{city}.*?₹([\d,]+).*?₹([\d,]+)",
            gold_text,
            re.IGNORECASE | re.DOTALL,
        )

    response = requests.get(
        f"https://www.goodreturns.in/silver-rates/{city_slug}.html",
        timeout=20,
        headers=headers,
    )
    response.raise_for_status()
    silver_soup = BeautifulSoup(response.text, "html.parser")
    silver_text = silver_soup.get_text(" ", strip=True)
    silver_match = re.search(
        rf"(?:silver|price).+?{city}.+?₹([\d,]+)\s+per gram",
        silver_text,
        re.IGNORECASE | re.DOTALL,
    )
    if gold_match is None or silver_match is None:
        raise StockDataError(f"{city} market bullion prices are currently unavailable.")

    gold_24k, gold_22k = (
        float(value.replace(",", "")) * 10 for value in gold_match.groups()
    )
    return {
        "gold_24k_per_10g": gold_24k,
        "gold_22k_per_10g": gold_22k,
        "silver_per_10g": float(silver_match.group(1).replace(",", "")) * 10,
        "history": parse_city_history(gold_soup, silver_soup, datetime.now(INDIA_TIME).date()),
    }


METAL_ETF_SYMBOLS = {
    "Tata Gold ETF": "TATAGOLD.NS",
    "Gold BeES": "GOLDBEES.NS",
    "Silver BeES": "SILVERBEES.NS",
}


def _numeric_info_value(info, keys):
    for key in keys:
        try:
            value = float(info.get(key))
        except (TypeError, ValueError):
            continue
        if pd.notna(value):
            return value
    return None


def fetch_etf_fund_details(symbol):
    """Preserve Yahoo fund details and fill missing fields from Groww."""
    warnings = []
    try:
        info = yf.Ticker(symbol).info
    except Exception as exc:
        warnings.append(f"Yahoo Finance ETF metadata failed: {exc}")
        info = {}
    if not info:
        warnings.append("Yahoo Finance returned no ETF metadata; trying Groww.")

    assets = _numeric_info_value(info, ("totalAssets",))
    currency = info.get("currency")
    if assets is None:
        aum = "Unavailable from Yahoo Finance"
    elif currency == "INR":
        aum = f"₹{assets / 10_000_000:,.2f} Cr"
    elif currency:
        aum = f"{assets:,.2f} {currency}"
    else:
        aum = f"{assets:,.2f} (currency unavailable)"

    expense_ratio = _numeric_info_value(
        info, ("annualReportExpenseRatio", "netExpenseRatio")
    )
    expense_ratio = (
        f"{expense_ratio * 100:.2f}%"
        if expense_ratio is not None
        else "Unavailable from Yahoo Finance"
    )

    average_volume = _numeric_info_value(
        info, ("averageDailyVolume3Month", "averageVolume", "averageDailyVolume10Day")
    )
    average_volume = (
        f"{average_volume / 100_000:,.2f} Lac"
        if average_volume is not None
        else "Unavailable from Yahoo Finance"
    )

    tracking_error = _numeric_info_value(info, ("trackingError",))
    tracking_error = (
        f"{tracking_error * 100:.2f}%"
        if tracking_error is not None
        else "Not provided by Yahoo Finance"
    )
    metrics = {
        "AUM": aum,
        "Expense ratio": expense_ratio,
        "Avg. volume": average_volume,
        "Tracking error": tracking_error,
    }
    missing = [
        label for label, value in metrics.items()
        if value.startswith(("Unavailable", "Not provided"))
    ]
    sources = {
        label: "Yahoo Finance" for label in metrics if label not in missing
    }
    if missing:
        try:
            fallback, url = fetch_groww_etf_details(symbol)
        except (StockDataError, requests.RequestException) as exc:
            warnings.append(f"Groww ETF fundamentals failed: {exc}")
        else:
            for label in missing:
                if label in fallback:
                    metrics[label] = fallback[label]
                    sources[label] = f"Groww ({url})"
    for label in missing:
        if label not in sources:
            metrics[label] = "Unavailable"
            warnings.append(f"{label} is unavailable from the checked sources.")
    return {
        "metrics": metrics,
        "sources": sources,
        "warnings": warnings,
        "fetched_at": datetime.now(INDIA_TIME),
    }


def fetch_gold_silver_etf_prices():
    """Fetch recent daily closing prices for popular Indian gold and silver ETFs."""
    etfs = {}
    errors = {}
    for name, symbol in METAL_ETF_SYMBOLS.items():
        try:
            history = yf.Ticker(symbol).history(period="1mo", interval="1d")
            closes = history["Close"].dropna() if "Close" in history else pd.Series(dtype=float)
            if closes.empty:
                raise StockDataError(f"{name} closing prices are currently unavailable.")
            etfs[name] = {
                "symbol": symbol,
                "price": float(closes.iloc[-1]),
                "history": closes.tail(15),
                "latest_session": closes.index[-1],
            }
        except Exception as exc:
            errors[name] = str(exc) or exc.__class__.__name__
    return etfs, errors


def fetch_metal_prices():
    """Fetch recent COMEX gold and silver futures plus the USD/INR rate."""
    symbols = {"Gold": "GC=F", "Silver": "SI=F", "USD/INR": "INR=X"}
    result = {}
    try:
        for name, symbol in symbols.items():
            ticker = yf.Ticker(symbol)
            history = ticker.history(period="1mo", interval="1d")
            closes = history["Close"].dropna() if "Close" in history else pd.Series(dtype=float)
            if closes.empty:
                raise StockDataError(f"{name} price is currently unavailable.")

            live_price = None
            try:
                fast = ticker.fast_info
                for key in ("last_price", "regularMarketPrice"):
                    if key in fast and fast[key] is not None:
                        live_price = float(fast[key])
                        break
            except Exception:
                live_price = None

            price = live_price if live_price is not None else float(closes.iloc[-1])
            previous = float(closes.iloc[-2]) if len(closes) > 1 else price
            change_pct = ((price - previous) / previous * 100) if previous else 0.0
            latest_close = closes.index[-1]
            result[name] = {
                "price": price,
                "change_pct": change_pct,
                "history": closes.tail(15),
                "latest_session": latest_close,
                "is_live_quote": live_price is not None,
            }
    except Exception as exc:
        if isinstance(exc, StockDataError):
            raise
        raise StockDataError(f"Gold and silver prices could not be loaded: {exc}") from exc

    usd_inr = result["USD/INR"]["price"]
    ounce_to_10g = 10 / 31.1034768
    result["Gold"]["inr_10g"] = result["Gold"]["price"] * usd_inr * ounce_to_10g
    result["Silver"]["inr_10g"] = result["Silver"]["price"] * usd_inr * ounce_to_10g
    result["USD/INR"]["inr_10g"] = None
    try:
        result["Indian"] = fetch_indian_bullion_spot()
    except (StockDataError, requests.RequestException) as exc:
        result["Indian"] = None
        result["Indian error"] = str(exc)
    result["City"] = {}
    for city in ("Hyderabad", "Delhi"):
        try:
            result["City"][city] = fetch_indian_city_market(city)
        except (StockDataError, requests.RequestException) as exc:
            result["City"][city] = None
            result[f"{city} error"] = str(exc)
    result["ETFs"], result["ETF errors"] = fetch_gold_silver_etf_prices()
    result["fetched_at"] = datetime.now(INDIA_TIME)
    return result


def render_metal_news(currency):
    st.subheader("Recent news & potential price drivers")
    st.caption(
        "Gold/silver market and Fed/RBI rate-decision headlines from the last 7 days. "
        "Topic tags identify possible drivers, "
        "not verified causes of the displayed price change or buy/sell signals."
    )
    try:
        with st.spinner(f"Loading {currency} gold and silver news..."):
            news = cached_metal_news(currency)
    except StockDataError as exc:
        st.warning(str(exc))
        return
    st.caption(
        "Publishers: Moneycontrol, Reuters, Economic Times, Mint, CNBC/CNBC-TV18 "
        "and Business Standard. Headlines via Google News RSS; rate decisions prioritized, "
        "then Moneycontrol. "
        "News fetched: "
        + news["fetched_at"].astimezone(INDIA_TIME).strftime("%d %b %Y, %I:%M:%S %p %Z")
    )
    if news["skipped"]:
        st.caption(f"Skipped {news['skipped']} headlines with missing or invalid dates/links.")
    if not news["items"]:
        st.info("No recent matching gold/silver or Fed/RBI rate-decision headlines were returned.")
        return
    for index, item in enumerate(news["items"]):
        st.write(item["title"])
        published = item["published"].astimezone(INDIA_TIME).strftime("%d %b %Y, %I:%M %p %Z")
        st.caption(f"{item['publisher']} | Published: {published}")
        if item["drivers"]:
            st.caption("Potential drivers: " + ", ".join(item["drivers"]))
        st.link_button(
            "Read article", item["link"], key=f"metal-news-{currency}-{index}"
        )


def render_metal_prices_tab():
    st.subheader("Gold & Silver Price")
    st.caption(
        "USD prices show COMEX futures. INR prices show Indian city-market "
        "rates or IBJA bullion benchmarks per 10 grams."
    )
    if st.button("Refresh gold & silver prices", type="primary"):
        cached_metal_prices.clear()
        cached_etf_fund_details.clear()
        cached_metal_news.clear()
        st.session_state["metal_prices_requested"] = True
    if not st.session_state.get("metal_prices_requested"):
        st.info("Select Refresh gold & silver prices to load the latest market prices.")
        return
    try:
        with st.spinner("Loading gold and silver market prices..."):
            prices = cached_metal_prices()
    except StockDataError as exc:
        st.error(str(exc))
        return

    gold = prices["Gold"]
    silver = prices["Silver"]
    fetched_at = prices["fetched_at"].strftime("%d %b %Y, %I:%M:%S %p %Z")
    usd_tab, inr_tab, etf_tab = st.tabs(["USD Prices", "INR Prices", "Gold/Silver ETFs"])

    with usd_tab:
        st.caption(f"Price data fetched: {fetched_at}")
        quote_dates = []
        for name in ("Gold", "Silver", "USD/INR"):
            session = prices[name]["latest_session"]
            quote_dates.append(session.strftime("%d %b %Y"))
        st.caption(
            "Latest available market session: "
            f"Gold {quote_dates[0]}, Silver {quote_dates[1]}, USD/INR {quote_dates[2]} "
            "(prices use live Yahoo/Fast Quote values where available)."
        )
        gold_col, silver_col, forex_col = st.columns(3)
        gold_col.metric(
            "Gold",
            f"${gold['price']:,.2f}/oz",
            f"{gold['change_pct']:+.2f}%",
        )
        silver_col.metric(
            "Silver",
            f"${silver['price']:,.2f}/oz",
            f"{silver['change_pct']:+.2f}%",
        )
        forex_col.metric(
            "USD / INR",
            f"₹{prices['USD/INR']['price']:,.2f}",
            f"{prices['USD/INR']['change_pct']:+.2f}%",
        )

        gold_chart = go.Figure()
        gold_chart.add_trace(go.Scatter(
            x=gold["history"].index,
            y=gold["history"].values,
            mode="lines+markers",
            name="Gold (USD/oz)",
        ))
        gold_chart.update_layout(
            title="Gold - USD",
            xaxis_title="Date",
            yaxis_title="Gold (USD per troy ounce)",
            height=420,
        )
        st.plotly_chart(gold_chart, width="stretch")

        silver_chart = go.Figure()
        silver_chart.add_trace(go.Scatter(
            x=silver["history"].index,
            y=silver["history"].values,
            mode="lines+markers",
            name="Silver (USD/oz)",
        ))
        silver_chart.update_layout(
            title="Silver - USD",
            xaxis_title="Date",
            yaxis_title="Silver (USD per troy ounce)",
            height=420,
        )
        st.plotly_chart(silver_chart, width="stretch")
        render_metal_news("USD")

    with inr_tab:
        st.caption(f"Price data fetched: {fetched_at}")
        market = st.radio(
            "Indian market",
            ["Hyderabad", "Delhi", "IBJA Benchmark"],
            horizontal=True,
            key="indian_metal_market",
        )
        if market == "IBJA Benchmark":
            indian = prices.get("Indian")
        else:
            indian = prices.get("City", {}).get(market)

        if indian is None:
            source_error = (
                prices.get("Indian error")
                if market == "IBJA Benchmark"
                else prices.get(f"{market} error", f"{market} rates are unavailable")
            )
            st.warning(f"{market} prices are unavailable: {source_error}.")
        else:
            gold_24k_col, gold_22k_col, silver_col = st.columns(3)
            gold_24k_col.metric("Gold - 24K", f"₹{indian['gold_24k_per_10g']:,.2f}/10g")
            gold_22k_col.metric("Gold - 22K", f"₹{indian['gold_22k_per_10g']:,.2f}/10g")
            silver_col.metric("Silver - 999", f"₹{indian['silver_per_10g']:,.2f}/10g")
            if market == "IBJA Benchmark":
                st.caption(
                    "IBJA Indian bullion benchmark rates. Gold: 24K (999), 22K (916). "
                    "Silver: 999. GST and making charges are excluded."
                )
            else:
                st.caption(
                    f"Indicative {market} city-market retail rates for today. "
                    "Rates exclude GST, TCS, making charges and dealer levies."
                )

            history = indian["history"]
            if history.empty or history.drop(columns="Date").dropna(how="all").empty:
                st.warning(f"No published {market} history is available for the last 15 calendar days.")
            else:
                missing = []
                today = datetime.now(INDIA_TIME).date()
                for name, title, axis_title in (
                    ("24K Gold", "24K Gold", "Gold (₹ per 10g)"),
                    ("Silver", "Silver", "Silver (₹ per 10g)"),
                ):
                    points = history.dropna(subset=[name])
                    if len(points) < 15:
                        missing.append(f"{name}: {len(points)}/15")
                    if points.empty:
                        continue
                    chart = go.Figure()
                    chart.add_trace(go.Scatter(
                        x=points["Date"].tolist(), y=points[name].tolist(), mode="lines+markers",
                        name=name,
                    ))
                    chart.update_layout(
                        title=f"{market} - {title} - Last 15 Calendar Days",
                        xaxis_title="Date",
                        xaxis=dict(range=[today - timedelta(days=14), today]),
                        yaxis_title=axis_title,
                        height=420,
                    )
                    st.plotly_chart(chart, width="stretch")
                if missing:
                    st.caption(
                        "Published observations in the last 15 calendar days: "
                        + ", ".join(missing)
                        + ". Missing dates are not estimated; lines connect available observations."
                    )
                if market == "IBJA Benchmark":
                    st.caption("Chart uses IBJA's published PM benchmark rates, excluding GST.")
        render_metal_news("INR")
    with etf_tab:
        st.caption(
            f"ETF market closing prices from Yahoo Finance. Price data fetched: {fetched_at}"
        )
        etfs = prices["ETFs"]
        etf_names = tuple(METAL_ETF_SYMBOLS)
        selected_etf = st.radio(
            "Select ETF",
            etf_names,
            horizontal=True,
            key="selected_etf_details",
        )
        etf = etfs.get(selected_etf)
        if etf is None:
            st.metric(selected_etf, "Unavailable")
            st.warning(f"{selected_etf} is unavailable: {prices['ETF errors'][selected_etf]}")
        else:
            st.metric(selected_etf, f"₹{etf['price']:,.2f}")
            st.caption(f"Latest close: {etf['latest_session'].strftime('%d %b %Y')}")

        st.divider()
        st.subheader("ETF Fund Details")
        try:
            with st.spinner(f"Loading {selected_etf} fund details..."):
                details = cached_etf_fund_details(METAL_ETF_SYMBOLS[selected_etf])
        except StockDataError as exc:
            st.warning(str(exc))
        else:
            sources = list(dict.fromkeys(details["sources"].values()))
            if sources:
                st.caption("Source: " + "; ".join(sources))
            detail_columns = st.columns(4)
            for column, (label, value) in zip(detail_columns, details["metrics"].items()):
                column.metric(label, value)
            for warning in details["warnings"]:
                st.warning(warning)
            st.caption(
                f"Fund details fetched: {details['fetched_at'].strftime('%d %b %Y, %I:%M:%S %p %Z')}. "
                "Fundamentals reporting date is not supplied by these sources. Groww average "
                "volume uses 1 month; Yahoo uses its available average-volume period. "
                "Missing values are not estimated."
            )
        if etf is not None:
            chart = go.Figure()
            chart.add_trace(go.Scatter(
                x=etf["history"].index,
                y=etf["history"].values,
                mode="lines+markers",
                name=selected_etf,
            ))
            chart.update_layout(
                title=f"{selected_etf} ({etf['symbol']}) - Last 15 Trading Sessions",
                xaxis_title="Date",
                yaxis_title="ETF market price (₹ per unit)",
                height=420,
            )
            st.plotly_chart(chart, width="stretch")
    st.caption(
        "Sources: Yahoo Finance futures and ETF quotes (USD and ETF tabs), IBJA benchmark "
        "and Good Returns Hyderabad/Delhi city rates (INR tab). Cached up to 15 minutes."
    )


def render_institutional_tab():
    st.subheader("Top 20 stocks with increasing FII/DII holdings")
    st.caption(
        "Latest reported quarterly shareholdings from Screener, covering Nifty 500 "
        "where available. Changes are percentage points (pp), not daily purchases "
        "or transaction values. Ownership changes can also reflect dilution or "
        "reclassification and are not proof of open-market buying."
    )
    mode = st.radio(
        "Institutional investors", ["FII", "DII", "FII+DII"],
        index=2, horizontal=True, key="institutional_mode",
    )
    if st.button("Refresh FII/DII data", type="primary"):
        cached_institutional_buying.clear()
        st.session_state["institutional_requested"] = True
    if not st.session_state.get("institutional_requested"):
        st.info("Select Refresh FII/DII data to load recent disclosures. A full scan can take several minutes.")
        return
    try:
        with st.spinner("Checking recent institutional shareholding disclosures..."):
            report = cached_institutional_buying()
    except StockDataError as exc:
        st.error(str(exc))
        return
    st.caption(
        f"Coverage: {report.universe}. Valid reports: {len(report.stocks)}/{report.total}. "
        f"Checked {report.checked_at.strftime('%d %b %Y %H:%M IST')}. Cached up to 1 hour."
    )
    for warning in report.warnings:
        st.warning(warning)
    ranked = rank_institutional_stocks(report.stocks, mode)
    if ranked.empty:
        st.info("No qualifying stocks are available for this filter in the latest reporting quarter.")
        return
    st.caption(
        "Ranked by combined holding increase, or the selected investor's change. "
        "FII+DII requires increases from each group. All ranked stocks use the same latest quarter. "
        "Fewer than 20 rows means fewer qualifying stocks were available."
    )
    for column in ("Previous quarter", "Latest quarter"):
        ranked[column] = ranked[column].map(lambda value: value.strftime("%b %Y"))
    st.dataframe(
        ranked, hide_index=True, width="stretch",
        column_config={
            "Source": st.column_config.LinkColumn("Source", display_text="Shareholding"),
            **{
                column: st.column_config.NumberColumn(column, format="%.2f")
                for column in ranked.columns if column.endswith(("(%)", "(pp)"))
            },
        },
    )
    st.download_button(
        "Download FII/DII results", ranked.to_csv(index=False),
        file_name="fii_dii_top20.csv", mime="text/csv",
    )


st.title("📈 AI Indian Stock Analyst")
st.caption(
    "Live online research, sentiment, technicals, peers, portfolio, "
    "watchlist alerts, Nifty screening, quarterly quality filtering, and open/upcoming IPOs."
)

notification_credentials = render_notification_sidebar()
(
    analysis_tab,
    peers_tab,
    portfolio_tab,
    watchlist_tab,
    top_picks_tab,
    quality_screen_tab,
    ipo_screen_tab,
    institutional_tab,
    metal_prices_tab,
) = st.tabs(
    [
        "Stock Analysis",
        "Peer Comparison",
        "Portfolio",
        "Watchlist & Alerts",
        "Top 10 Picks",
        "Quality Screen",
        "IPO Screening",
        "FII/DII buying stock",
        "Gold & Silver Price",
    ],
    key="main_tab",
    on_change="rerun",
)

with analysis_tab:
    render_analysis_tab()
with peers_tab:
    render_peers_tab()
with portfolio_tab:
    render_portfolio_tab()
with watchlist_tab:
    render_watchlist_tab(notification_credentials)
with top_picks_tab:
    render_top_picks_tab()
with quality_screen_tab:
    render_quality_screen_tab()
with ipo_screen_tab:
    render_ipo_screen_tab()
with institutional_tab:
    render_institutional_tab()
with metal_prices_tab:
    render_metal_prices_tab()
