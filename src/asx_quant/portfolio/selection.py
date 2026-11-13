"""Eligibility filters and signal ranking for portfolio selection.

At each rebalance date a ticker is eligible only when ALL of:

1. An out-of-sample prediction exists for that (ticker, date) from the
   configured model, produced by a fold whose training ended STRICTLY
   before the rebalance date (no look-ahead — machine-checked);
2. The liquidity flag at that date is 1 (when ``require_liquidity_flag``);
3. The ticker has at least ``min_history_sessions`` sessions of feature
   history at the rebalance date.

Selection ranks eligible tickers by the configured signal, descending, and
takes the top ``n_holdings``. Fewer eligible names mean a smaller
portfolio with a larger cash residual — never padded with ineligible
stocks.
"""

from __future__ import annotations

import pandas as pd

from asx_quant.observability.logging import get_logger

logger = get_logger("portfolio.selection")


def load_signal(
    predictions: pd.DataFrame,
    model_name: str,
    signal: str,
) -> pd.DataFrame:
    """OOS signal frame for the configured model.

    Args:
        predictions: ``gold/predictions.parquet`` (long format).
        model_name: Configured model (``portfolio.selection.prediction_source.model``).
        signal: Configured signal name; its prefix must agree with the
            model's kind (``predicted_return_*`` -> regressor,
            ``prob_*`` -> classifier).

    Returns:
        Frame with ticker, date, prediction, trained_through.
    """
    if model_name not in set(predictions["model_name"].unique()):
        raise ValueError(
            f"Model '{model_name}' has no predictions in "
            "gold/predictions.parquet. Run the 'train' stage or update "
            "portfolio.selection.prediction_source.model."
        )
    kind = predictions.loc[
        predictions["model_name"] == model_name, "model_kind"
    ].iloc[0]
    if signal.startswith("predicted_return") and kind != "regressor":
        raise ValueError(
            f"Signal '{signal}' needs a regressor, but model '{model_name}' "
            f"is a {kind}."
        )
    if signal.startswith("prob") and kind != "classifier":
        raise ValueError(
            f"Signal '{signal}' needs a classifier, but model '{model_name}' "
            f"is a {kind}."
        )
    one = predictions[predictions["model_name"] == model_name][
        ["ticker", "date", "prediction", "trained_through", "fold_id"]
    ].copy()
    one = one.rename(columns={"prediction": "signal"})
    return one


def history_sessions_per_row(features: pd.DataFrame) -> pd.Series:
    """Cumulative session count per ticker (aligned to the frame index).

    ``features`` must be sorted by (ticker, date) — the gold build order.
    Row value = 1 + number of earlier sessions for the same ticker.
    """
    ordered = features.sort_values(["ticker", "date"], kind="stable")
    counts = ordered.groupby("ticker").cumcount() + 1
    return counts.reindex(features.index)


def eligible_universe(
    signals: pd.DataFrame,
    features: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    require_liquidity_flag: bool = True,
    min_history_sessions: int = 250,
    history: pd.Series | None = None,
) -> pd.DataFrame:
    """Eligible candidates at one rebalance date, with their signals.

    Args:
        signals: Frame from :func:`load_signal`.
        features: Gold features frame (for liquidity/history lookups).
        rebalance_date: The session on which the portfolio is constructed.
        require_liquidity_flag: Require ``liquid_flag == 1``.
        min_history_sessions: Minimum feature-history sessions per ticker.
        history: Optional precomputed per-row session counts
            (:func:`history_sessions_per_row`); computed once per build for
            speed instead of per date.

    Returns:
        Frame (ticker, signal, trained_through, fold_id, history_sessions)
        of eligible tickers, unranked. Empty when the date has no
        predictions.
    """
    day_signals = signals[signals["date"] == rebalance_date]
    if day_signals.empty:
        return pd.DataFrame(
            columns=["ticker", "signal", "trained_through", "fold_id",
                     "history_sessions"])

    # Look-ahead guard: the fold producing this prediction ended strictly
    # before the rebalance date.
    day_signals = day_signals[
        pd.to_datetime(day_signals["trained_through"]) < rebalance_date
    ]

    day_features = features[features["date"] == rebalance_date]
    if day_features.empty:
        return pd.DataFrame(
            columns=["ticker", "signal", "trained_through", "fold_id",
                     "history_sessions"])

    merged = day_signals.merge(
        day_features[["ticker", "liquid_flag"]],
        on="ticker", how="inner", validate="one_to_one",
    )
    if require_liquidity_flag:
        merged = merged[merged["liquid_flag"] == 1.0]

    if history is None:
        history = history_sessions_per_row(features)
    history_at_date = history.loc[day_features.index]
    merged["history_sessions"] = merged["ticker"].map(
        pd.Series(history_at_date.to_numpy(),
                  index=day_features["ticker"].to_numpy()))
    merged = merged[merged["history_sessions"] >= min_history_sessions]

    return merged[["ticker", "signal", "trained_through", "fold_id",
                   "history_sessions"]].reset_index(drop=True)


def rank_candidates(
    eligible: pd.DataFrame,
    n_holdings: int,
) -> pd.DataFrame:
    """Top-N by descending signal with explicit ranks (1 = strongest)."""
    if eligible.empty:
        return eligible.assign(rank=pd.Series(dtype="int64"))
    ranked = eligible.sort_values(
        ["signal", "ticker"], ascending=[False, True], kind="stable"
    ).head(n_holdings).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    return ranked
