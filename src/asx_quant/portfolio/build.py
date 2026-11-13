"""Portfolio construction orchestrator: OOS predictions -> target weights.

Pipeline (per rebalance date, all information available AT that session):

1. Load the configured model's OOS signal frame from
   ``gold/predictions.parquet`` and the feature columns needed for
   eligibility (``liquid_flag``) and vol-adjusted weighting
   (``volatility_20d``).
2. Build the rebalance schedule on the ASX session calendar (weekly
   Mondays, shifted forward to actual sessions, never before the first
   OOS prediction date).
3. Filter the eligible universe (liquidity flag, minimum history,
   prediction existence with ``trained_through`` strictly before the
   rebalance date — the look-ahead guard).
4. Rank by signal, take top N, apply the configured weighting with the
   max-stock cap; residual stays in cash (``prefer_cash_over_violation``).
5. Run quality rules; write ``gold/target_portfolios.parquet`` (long
   format: one row per rebalance-date/ticker) plus a per-rebalance summary
   JSON (cash weight, turnover, holdings). The Phase 7 backtester
   reconstructs cash as ``1 - sum(weights)`` and simulates execution
   (next-session-open, costs, dividends).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from asx_quant.config.loader import load_yaml
from asx_quant.exceptions import StorageError
from asx_quant.observability.logging import get_logger
from asx_quant.portfolio import checks as portfolio_checks
from asx_quant.portfolio.schedule import next_session, rebalance_dates
from asx_quant.portfolio.selection import (
    eligible_universe,
    history_sessions_per_row,
    load_signal,
    rank_candidates,
)
from asx_quant.portfolio.weights import build_weights, turnover_vs_previous
from asx_quant.quality.checks import QualityReport
from asx_quant.storage.parquet import read_parquet
from asx_quant.storage.paths import DataLake

logger = get_logger("portfolio.build")

PORTFOLIO_COLUMNS = (
    "rebalance_date", "effective_date", "ticker", "signal", "rank",
    "target_weight", "model_name", "model_version", "signal_name",
    "feature_set_version", "trained_through",
)


@dataclass
class PortfolioBuildResult:
    """Paths and quality outcomes of one portfolio-construction run."""

    portfolios_path: Path
    summary_path: Path
    report: QualityReport | None
    n_rebalances: int
    n_rows: int
    avg_turnover: float


def build_portfolios(
    lake: DataLake,
    settings,  # noqa: ANN001
    registry,  # noqa: ANN001
    with_quality: bool = True,
) -> PortfolioBuildResult:
    """Construct the configured target portfolios into Gold datasets."""
    cfg = load_yaml("portfolio").get("portfolio", {})
    selection = cfg.get("selection", {})
    weights_cfg = cfg.get("weights", {})
    rebal_cfg = cfg.get("rebalancing", {})

    signal_name = str(selection.get("signal", "predicted_return_3d"))
    source = selection.get("prediction_source", {})
    model_name = str(source.get("model", "hist_gbt_reg"))
    model_version = ""

    # ---- inputs ------------------------------------------------------------
    predictions_path = lake.gold_dataset_path(
        str(source.get("dataset", "predictions")))
    if not Path(predictions_path).exists():
        raise StorageError(
            "gold/predictions.parquet not found; run the 'train' stage first."
        )
    predictions = read_parquet(predictions_path)
    model_version = str(predictions["model_version"].iloc[0]) \
        if not predictions.empty else ""
    signals = load_signal(predictions, model_name, signal_name)
    if signals.empty:
        raise StorageError(
            f"No predictions for model '{model_name}'; run the 'train' stage."
        )
    feature_set_version = str(predictions["feature_set_version"].iloc[0])

    features = read_parquet(lake.gold_dataset_path("features"))
    vol_col = "volatility_20d" if "volatility_20d" in features.columns else None

    sessions = pd.DatetimeIndex(sorted(features["date"].unique()))
    first_signal = pd.Timestamp(signals["date"].min())
    schedule = rebalance_dates(
        sessions,
        frequency=str(rebal_cfg.get("frequency", "weekly")),
        weekday=str(rebal_cfg.get("weekday", "monday")),
        first_available_signal=first_signal,
    )
    if not schedule:
        raise StorageError(
            "No rebalance dates available after the first OOS prediction "
            "date; check the rebalancing config and prediction coverage."
        )
    logger.info("Rebalance schedule: %d dates, %s -> %s",
                len(schedule), schedule[0].date(), schedule[-1].date())

    n_holdings = int(selection.get("n_holdings", 20))
    require_liquidity = bool(selection.get("require_liquidity_flag", True))
    min_history = int(cfg.get("constraints", {}).get("min_history_sessions", 250))
    method = str(weights_cfg.get("method", "equal_weight"))

    history = history_sessions_per_row(features)

    # ---- construct per rebalance -------------------------------------------
    weight_parts: list[pd.DataFrame] = []
    rebalance_records: list[dict] = []
    previous_weights = pd.DataFrame(columns=["ticker", "target_weight"])

    for date in schedule:
        eligible = eligible_universe(
            signals, features, date,
            require_liquidity_flag=require_liquidity,
            min_history_sessions=min_history,
            history=history,
        )
        ranked = rank_candidates(eligible, n_holdings)
        if vol_col is not None and not ranked.empty:
            day_features = features.loc[features["date"] == date,
                                        ["ticker", vol_col]]
            ranked = ranked.merge(
                day_features.rename(columns={vol_col: "volatility_20d"}),
                on="ticker", how="left", validate="one_to_one",
            )
        weighted, cash = build_weights(ranked, method, weights_cfg)

        effective = next_session(sessions, date)
        if not weighted.empty:
            part = weighted[[
                "ticker", "signal", "rank", "target_weight",
                "trained_through",
            ]].copy()
            part.insert(0, "rebalance_date", date)
            part.insert(1, "effective_date", effective)
            part["model_name"] = model_name
            part["model_version"] = model_version
            part["signal_name"] = signal_name
            part["feature_set_version"] = feature_set_version
            weight_parts.append(part)

        day_weights = weighted[["ticker", "target_weight"]]
        turnover = turnover_vs_previous(day_weights, previous_weights)
        previous_weights = day_weights

        rebalance_records.append({
            "rebalance_date": str(date.date()),
            "effective_date": str(effective.date()) if effective is not None
            else None,
            "n_holdings": int(len(weighted)),
            "n_holdings_target": n_holdings,
            "cash_weight": round(float(cash), 10),
            "turnover": round(turnover, 6),
            "is_session": bool(date in set(sessions)),
            "all_liquid": bool(
                require_liquidity is False or len(weighted) == 0 or (
                    weighted["ticker"].isin(
                        features.loc[
                            (features["date"] == date)
                            & (features["liquid_flag"] == 1.0), "ticker"]
                    ).all()
                )
            ),
            "min_history_sessions": (
                float(weighted["history_sessions"].min())
                if "history_sessions" in weighted and len(weighted)
                else float("inf")
            ),
            "method": method,
        })

    portfolios = (
        pd.concat(weight_parts, ignore_index=True) if weight_parts
        else pd.DataFrame(columns=list(PORTFOLIO_COLUMNS))
    )

    # ---- quality + persist ---------------------------------------------------
    report = None
    if with_quality:
        report = portfolio_checks.run_portfolio_quality_checks(
            portfolios, rebalance_records,
            max_stock_weight=float(weights_cfg.get("max_stock_weight", 1.0)),
            max_holdings=int(cfg.get("constraints", {}).get("max_holdings",
                                                            n_holdings)),
            min_history_sessions=min_history,
            require_liquidity_flag=require_liquidity,
            sector_rule_implemented=False,
        )
        report.save(lake.report_path("portfolio_quality", ".json"))
        report.assert_passed()
        logger.info("Portfolio quality: %s (%d warnings)",
                    report.to_dict()["status"], report.warn_count())

    portfolios_path = lake.gold_dataset_path("target_portfolios")
    portfolios.to_parquet(portfolios_path, index=False)

    turnovers = [r["turnover"] for r in rebalance_records]
    summary = {
        "run_id": registry.run_id,
        "portfolio_id": str(cfg.get("portfolio_id", "ml_long_only_v1")),
        "model_name": model_name,
        "model_version": model_version,
        "signal_name": signal_name,
        "weight_method": method,
        "feature_set_version": feature_set_version,
        "n_rebalances": len(rebalance_records),
        "n_weight_rows": len(portfolios),
        "first_rebalance": rebalance_records[0]["rebalance_date"],
        "last_rebalance": rebalance_records[-1]["rebalance_date"],
        "avg_holdings": round(
            sum(r["n_holdings"] for r in rebalance_records)
            / len(rebalance_records), 2),
        "avg_cash_weight": round(
            sum(r["cash_weight"] for r in rebalance_records)
            / len(rebalance_records), 4),
        "avg_turnover": round(sum(turnovers) / len(turnovers), 4),
        "per_rebalance": rebalance_records,
    }
    summary_path = lake.report_path("portfolio_summary", ".json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str),
                            encoding="utf-8")

    return PortfolioBuildResult(
        portfolios_path=portfolios_path,
        summary_path=summary_path,
        report=report,
        n_rebalances=len(rebalance_records),
        n_rows=len(portfolios),
        avg_turnover=summary["avg_turnover"],
    )
