"""Rebalance scheduling on the ASX session calendar.

The preferred rebalance weekday is configurable (``weekly`` + ``monday`` by
default). If the preferred weekday is not a trading session (public
holiday), the rebalance shifts FORWARD to the next actual session — the
portfolio can only be constructed on a day the market traded.

Rebalance dates earlier than the first date with available OOS predictions
are skipped: no signal, no portfolio (never padded with in-sample data).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from asx_quant.observability.logging import get_logger

logger = get_logger("portfolio.schedule")

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4,
}
_FREQUENCIES = ("weekly", "fortnightly", "monthly")


def rebalance_dates(
    sessions: pd.DatetimeIndex,
    frequency: str,
    weekday: str,
    first_available_signal: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Actual-session rebalance dates, preferred weekday shifted forward.

    Args:
        sessions: Sorted ASX sessions (the feature calendar).
        frequency: weekly | fortnightly | monthly.
        weekday: Preferred weekday name (e.g. ``monday``).
        first_available_signal: Earliest session with OOS predictions;
            rebalances before it are skipped.

    Returns:
        Sorted rebalance dates (a subset of ``sessions``).
    """
    if frequency not in _FREQUENCIES:
        raise ValueError(
            f"Unknown rebalance frequency '{frequency}'. "
            f"Valid: {', '.join(_FREQUENCIES)}."
        )
    if weekday not in _WEEKDAYS:
        raise ValueError(
            f"Unknown rebalance weekday '{weekday}'. "
            f"Valid: {', '.join(_WEEKDAYS)}."
        )
    step_weeks = {"weekly": 1, "fortnightly": 2, "monthly": 4}[frequency]
    preferred = _WEEKDAYS[weekday]

    # Every preferred-weekday calendar day in range, thinned by frequency,
    # then shifted forward to the next ACTUAL session (holiday Monday ->
    # Tuesday session). Forward shifts can collide with the next candidate
    # (multi-day holiday); keep the first occurrence.
    start = sessions[0].normalize()
    # First candidate = preferred weekday of the week containing the first
    # session (an early-week holiday Monday shifts forward to the session
    # that opens the calendar — it must not be skipped).
    day = start - dt.timedelta(days=(start.weekday() - preferred) % 7)
    end = sessions[-1].normalize()
    calendar_days = [day + dt.timedelta(weeks=k)
                     for k in range(0, ((end - day).days // 7) + 1)
                     if day + dt.timedelta(weeks=k) <= end]
    calendar_days = calendar_days[::step_weeks]

    sessions_index = pd.DatetimeIndex(sessions)
    candidates: list[pd.Timestamp] = []
    seen: set[pd.Timestamp] = set()
    for candidate in calendar_days:
        pos = int(sessions_index.searchsorted(candidate, side="left"))
        if pos >= len(sessions_index):
            continue
        session = sessions_index[pos]
        if session not in seen:
            seen.add(session)
            candidates.append(session)

    return [d for d in candidates if d >= first_available_signal]


def next_session(sessions: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp | None:
    """First session strictly after ``date`` (the execution-effective day)."""
    pos = int(sessions.searchsorted(date, side="right"))
    return sessions[pos] if pos < len(sessions) else None
