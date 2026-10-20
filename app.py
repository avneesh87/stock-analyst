import json
from datetime import datetime
from threading import Thread

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import truststore

from analytics import (
    StockDataError,
    build_recommendation,
    build_watchlist_report,
    calculate_technicals,
    get_industry_pe,
    get_news,
    get_peer_comparison,
    get_price_history,
    get_stock_data,
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
    return get_stock_data(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_news(symbol: str):
    return get_news(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_history(symbol: str):
    return get_price_history(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_peers(symbol: str):
    return get_peer_comparison(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_industry_pe(symbol: str):
    return get_industry_pe(symbol)


@st.cache_data(ttl=900, show_spinner=False)
def cached_open_ipos(today):
    return screen_open_mainboard_ipos(today, include_upcoming=True)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_top_stocks():
    return rank_top_stocks()


@st.cache_data(ttl=900, show_spinner=False)
def cached_watchlist_report(symbols: tuple[str, ...]):
    return build_watchlist_report(list(symbols))


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
        "pe": rounded(data["PE"]),
        "roe_percent": (
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

    try:
        with requests.post(
            "http://localhost:11434/api/generate",
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
            "Ollama is not running. Start Ollama and ensure llama3.2:1b is installed."
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


def render_analysis_tab():
    st.subheader("Stock research")
    input_column, button_column = st.columns([4, 1])
    symbol_input = input_column.text_input(
        "NSE/Yahoo symbol",
        "TCS.NS",
        help="The .NS suffix is added automatically when omitted.",
    )
    analyze = button_column.button(
        "Analyze",
        type="primary",
        use_container_width=True,
    )

    if analyze:
        symbol = normalize_symbol(symbol_input)
        if not symbol:
            st.error("Enter a stock symbol.")
            return
        try:
            with st.spinner("Loading live fundamentals, news, and price history..."):
                data = cached_stock_data(symbol)
                news = cached_news(symbol)
                history = cached_history(symbol)
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
    stock_pe_column.metric("Stock P/E", format_decimal(data["PE"]))
    industry_pe_column.metric(
        "Industry P/E",
        format_decimal(analysis["industry_pe"]),
        help=analysis["industry_pe_note"],
    )
    st.caption(
        "Industry P/E is an available Nifty 50 industry-peer median, not a "
        f"full-market industry average. {analysis['industry_pe_note']}"
    )
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

    fundamental_tab, technical_tab, news_tab, ai_tab = st.tabs(
        ["Fundamentals", "Technicals", "News sentiment", "AI recommendation"]
    )
    with fundamental_tab:
        fundamentals = pd.DataFrame(
            [
                {
                    "Metric": "Market Cap",
                    "Value": format_crore(market_cap_crore),
                },
                {"Metric": "PE", "Value": format_decimal(data["PE"])},
                {
                    "Metric": "Industry P/E (Nifty 50 peer median)",
                    "Value": format_decimal(analysis["industry_pe"]),
                },
                {"Metric": "ROE", "Value": format_percent(data["ROE"])},
                {
                    "Metric": "Debt / Equity",
                    "Value": format_decimal(data["DebtEquity"]),
                },
                {
                    "Metric": "Revenue Growth",
                    "Value": format_percent(data["RevenueGrowth"]),
                },
                {
                    "Metric": "Profit Margin",
                    "Value": format_percent(data["ProfitMargin"]),
                },
                {
                    "Metric": "Free Cash Flow",
                    "Value": format_crore(to_crore(data["FreeCashFlow"])),
                },
            ]
        )
        st.dataframe(fundamentals, hide_index=True, use_container_width=True)
        st.caption(analysis["industry_pe_note"])

    with technical_tab:
        history = analysis["history"]
        chart_data = ensure_support_resistance(calculate_technicals(history))
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

    with news_tab:
        st.metric("Aggregate headline sentiment", sentiment["overall"])
        headline_frame = pd.DataFrame(sentiment["headlines"])
        st.dataframe(
            headline_frame[["title", "published", "sentiment"]],
            hide_index=True,
            use_container_width=True,
        )

    with ai_tab:
        st.caption("Generated locally with Ollama llama3.2:1b.")
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
            }
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
            report = cached_watchlist_report(tuple(symbols))
    except StockDataError as exc:
        st.error(str(exc))
        return

    st.dataframe(
        report.style.format(
            {
                "Price": "₹{:,.2f}",
                "RSI": "{:.2f}",
                "6M Return %": "{:.2f}%",
            },
            na_rep="N/A",
        ),
        hide_index=True,
        use_container_width=True,
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
    st.dataframe(
        picks.style.format(
            {
                "Price": "₹{:,.2f}",
                "RSI": "{:.2f}",
                "6M Return %": "{:.2f}%",
            },
            na_rep="N/A",
        ),
        hide_index=True,
        use_container_width=True,
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

    st.dataframe(
        companies.style.format(
            {
                "Price": "₹{:,.2f}",
                "Market Cap (Cr)": "₹{:,.2f} Cr",
                "Sales QoQ %": "{:.2f}%",
                "Profit QoQ %": "{:.2f}%",
                "RSI": "{:.2f}",
                "6M Return %": "{:.2f}%",
                "Technical Score": "{:.2f}",
            },
            na_rep="N/A",
        ),
        hide_index=True,
        use_container_width=True,
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
            f"{issue.name} | {issue.status} | {issue.assessment.verdict}", expanded=True,
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
) = st.tabs(
    [
        "Stock Analysis",
        "Peer Comparison",
        "Portfolio",
        "Watchlist & Alerts",
        "Top 10 Picks",
        "Quality Screen",
        "IPO Screening",
    ]
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
