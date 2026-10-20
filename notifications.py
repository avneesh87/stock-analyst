from __future__ import annotations

from email.message import EmailMessage
import smtplib

import pandas as pd
import requests


class NotificationError(RuntimeError):
    pass


def format_report(report: pd.DataFrame) -> str:
    if report.empty:
        return "No watchlist market data is currently available."

    lines = ["Indian Stock Watchlist", ""]
    for row in report.to_dict("records"):
        trend = "Above" if row["Above 200 DMA"] else "Below"
        support_values = []
        for label in ("20D Support", "60D Support"):
            value = row.get(label)
            formatted = "N/A" if pd.isna(value) else f"INR {value:.2f}"
            support_values.append(f"{label} {formatted}")
        support_text = " | ".join(support_values)
        high_values = []
        for label in ("1M Return %", "1Y Return %", "From All-Time High %"):
            value = row.get(label)
            formatted = "N/A" if pd.isna(value) else f"{value:+.2f}%"
            high_values.append(f"{label.removesuffix(' %')} {formatted}")
        high_text = " | ".join(high_values)
        rsi_text = "N/A" if pd.isna(row["RSI"]) else f"{row['RSI']:.2f}"
        return_text = "N/A" if pd.isna(row["6M Return %"]) else f"{row['6M Return %']:.2f}%"
        lines.append(
            f"{row['Symbol']}: INR {row['Price']:.2f} | "
            f"Price date {row.get('Price Date', 'N/A')} | "
            f"6M reference {row.get('6M Reference Date') or 'N/A'} | "
            f"1Y reference {row.get('1Y Reference Date') or 'N/A'} | "
            f"{support_text} | "
            f"{high_text} | "
            f"RSI {rsi_text} | 6M {return_text} | "
            f"{trend} 200 DMA | Score {row['Technical Score']}/100"
        )
    lines.append("")
    lines.append(
        "Price returns exclude dividends: latest close / reference close - 1. "
        "1M looks back from the latest price date; 6M/1Y from today's India date. "
        "If the latest quote is older than today, 6M/1Y span less than their named periods. "
        "All-time-high distance uses "
        "available unadjusted Yahoo history."
    )
    lines.append("Generated from live online market data. Not financial advice.")
    return "\n".join(lines)


def send_email(
    *,
    smtp_host: str,
    smtp_port: int,
    username: str,
    password: str,
    recipient: str,
    report_text: str,
) -> None:
    if not all((smtp_host, username, password, recipient)):
        raise NotificationError("Complete all email credential fields.")

    message = EmailMessage()
    message["Subject"] = "Indian Stock Watchlist Report"
    message["From"] = username
    message["To"] = recipient
    message.set_content(report_text)

    try:
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
                server.login(username, password)
                server.send_message(message)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(message)
    except (OSError, smtplib.SMTPException) as exc:
        raise NotificationError(f"Email delivery failed: {exc}") from exc


def send_telegram(bot_token: str, chat_id: str, report_text: str) -> None:
    if not bot_token or not chat_id:
        raise NotificationError("Enter both Telegram bot token and chat ID.")

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    try:
        response = requests.post(
            url,
            json={"chat_id": chat_id, "text": report_text},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, requests.JSONDecodeError) as exc:
        raise NotificationError(f"Telegram delivery failed: {exc}") from exc

    if not payload.get("ok"):
        raise NotificationError(
            f"Telegram rejected the message: {payload.get('description', payload)}"
        )
