from datetime import datetime, timedelta
import re

import pandas as pd


HISTORY_COLUMNS = ["Date", "24K Gold", "22K Gold", "Silver"]
HISTORY_DAYS = 15


def _price(text):
    match = re.match(r"^\s*₹?\s*([\d,]+(?:\.\d+)?)", text)
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    return value if value > 0 else None


def _in_last_15_days(day, today):
    return today - timedelta(days=HISTORY_DAYS - 1) <= day <= today


def _table_with_headers(soup, headers):
    for table in soup.find_all("table"):
        row = table.find("tr")
        if row is None:
            continue
        values = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if values[:len(headers)] == headers:
            return table
    return None


def parse_city_history(gold_soup, silver_soup, today):
    """Parse city gold per gram and silver per 10 grams for the last 15 days."""
    records = {}
    for soup, headers, fields in (
        (gold_soup, ["Date", "24K", "22K"], (("24K Gold", 1, 10), ("22K Gold", 2, 10))),
        (silver_soup, ["Date", "10 gram", "100 gram", "1 Kg"], (("Silver", 1, 1),)),
    ):
        table = _table_with_headers(soup, headers)
        if table is None:
            continue
        for row in table.find_all("tr")[1:]:
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
            if len(cells) < len(headers):
                continue
            try:
                day = datetime.strptime(cells[0], "%b %d, %Y").date()
            except ValueError:
                continue
            if not _in_last_15_days(day, today):
                continue
            record = records.setdefault(day, {"Date": day})
            for name, position, multiplier in fields:
                value = _price(cells[position])
                if value is not None:
                    record[name] = value * multiplier
    return pd.DataFrame(list(records.values()), columns=HISTORY_COLUMNS).sort_values("Date")


def parse_ibja_history(soup, today):
    """Parse IBJA's published PM benchmark rates without inventing closed-day data."""
    section = soup.select_one("#tab-pm")
    if section is None:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    table = _table_with_headers(
        section, ["", "999", "995", "916", "750", "585", "Silver 999", "Platinum 999"]
    )
    records = []
    if table is not None:
        for row in table.find_all("tr")[1:]:
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
            if len(cells) < 8:
                continue
            try:
                day = datetime.strptime(cells[0], "%d/%m/%Y").date()
            except ValueError:
                continue
            if not _in_last_15_days(day, today):
                continue
            record = {"Date": day}
            for name, position, divisor in (
                ("24K Gold", 1, 1), ("22K Gold", 3, 1),
                ("Silver", 6, 100),
            ):
                value = _price(cells[position])
                if value is not None:
                    record[name] = value / divisor
            records.append(record)
    return pd.DataFrame(records, columns=HISTORY_COLUMNS).sort_values("Date")
