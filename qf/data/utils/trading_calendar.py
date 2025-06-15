from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd
import pandas_market_calendars as mcal


def get_trading_days(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    exchange: str = "NYSE",
) -> pd.DatetimeIndex:
    """
    Get a DatetimeIndex containing only trading days for the specified date range.

    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        exchange: Stock exchange to use for calendar (default: 'NYSE')

    Returns:
        pd.DatetimeIndex containing only trading days
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    # Get the calendar for the specified exchange
    calendar = mcal.get_calendar(exchange)

    # Get trading days
    trading_days = calendar.valid_days(start_date=start_date, end_date=end_date)

    return trading_days


def is_trading_day(date: Union[str, datetime], exchange: str = "NYSE") -> bool:
    """
    Check if a given date is a trading day.

    Args:
        date: Date to check
        exchange: Stock exchange to use for calendar (default: 'NYSE')

    Returns:
        bool: True if the date is a trading day, False otherwise
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    calendar = mcal.get_calendar(exchange)
    return date in calendar.valid_days(start_date=date, end_date=date)


def get_next_trading_day(
    date: Union[str, datetime], exchange: str = "NYSE"
) -> pd.Timestamp:
    """
    Get the next trading day after the given date.

    Args:
        date: Reference date
        exchange: Stock exchange to use for calendar (default: 'NYSE')

    Returns:
        pd.Timestamp: Next trading day
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    calendar = mcal.get_calendar(exchange)
    next_day = date + timedelta(days=1)
    trading_days = calendar.valid_days(
        start_date=next_day, end_date=next_day + timedelta(days=10)
    )

    if len(trading_days) > 0:
        return trading_days[0]
    else:
        raise ValueError(f"No trading days found after {date}")


def get_previous_trading_day(
    date: Union[str, datetime], exchange: str = "NYSE"
) -> pd.Timestamp:
    """
    Get the previous trading day before the given date.

    Args:
        date: Reference date
        exchange: Stock exchange to use for calendar (default: 'NYSE')

    Returns:
        pd.Timestamp: Previous trading day
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    calendar = mcal.get_calendar(exchange)
    prev_day = date - timedelta(days=1)
    trading_days = calendar.valid_days(
        start_date=prev_day - timedelta(days=10), end_date=prev_day
    )

    if len(trading_days) > 0:
        return trading_days[-1]
    else:
        raise ValueError(f"No trading days found before {date}")
