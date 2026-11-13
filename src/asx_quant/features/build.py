"""Gold feature-set orchestrator: Silver -> gold/features + target_labels.

Pipeline (all steps backward-looking at session ``t``):

1. Load Silver equity prices; split universe vs benchmark tickers.
2. Ticker-level features (returns, MAs, volatility, technicals, volume).
3. Benchmark features broadcast by date; breadth; beta; cross-sectional
   percentile ranks.
4. Point-in-time macro matrix joined on ``publication_date <= date``.
5. Point-in-time short-interest features joined on
   ``publication_date <= date``.
6. Target labels (future returns) — the only forward-looking columns,
   written to a SEPARATE dataset with availability dates for purging.
7. Quality rules; write gold parquet datasets; register DuckDB views.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from asx_quant.config.loader import load_yaml
from asx_quant.exceptions import StorageError
from asx_quant.features import checks as feature_checks
from asx_quant.features import market as market_features
from asx_quant.features import price as price_features
from asx_quant.features.base import ensure_sorted
from asx_quant.features.macro_features import (
    add_derived_macro_features,
    build_macro_pit_matrix,
)
from asx_quant.features.short_features import compute_short_pit_features
from asx_quant.features.targets import compute_target_labels
from asx_quant.observability.logging import get_logger
from asx_quant.quality.checks import QualityReport
from asx_quant.storage.parquet import ParquetWriter, read_parquet
from asx_quant.storage.paths import DataLake

logger = get_logger("features.build")

TARGETS_COLUMNS = (
    "ticker", "date", "future_return_3d", "direction_3d",
    "label_available_date_3d", "future_return_10d", "direction_10d",
    "label_available_date_10d",
)


@dataclass
class FeatureBuildResult:
    """Paths and quality outcomes of one feature-set build."""

    features_path: Path
    targets_path: Path
    report: QualityReport
    feature_set_version: str
    n_rows: int
    n_features: int


def _load_macro_series_ids(features_cfg: dict) -> list[str]:
    macro_cfg = features_cfg.get("features", {}).get("macro", {})
    return [
        s["id"] for s in macro_cfg.get("series", [])
        if isinstance(s, dict) and "id" in s
    ]


def build_feature_set(
    lake: DataLake,
    settings,  # noqa: ANN001
    registry,  # noqa: ANN001
    with_quality: bool = True,
) -> FeatureBuildResult:
    """Compute the full configured feature set into Gold datasets."""
    features_cfg = load_yaml("features")
    version = str(features_cfg.get("feature_set_version", "v0"))
    cfg = features_cfg.get("features", {})
    targets_cfg = features_cfg.get("targets", {})

    prices_path = lake.silver_dataset_path("equity_prices")
    if not prices_path.exists():
        raise StorageError(
            "Silver equity_prices not found; run 'transform' first."
        )
    silver = ensure_sorted(read_parquet(prices_path))

    # Restrict to ONE benchmark series in priority order (^AXJO, IDX_AXJO,
    # VAS.AX). Mixing e.g. the index level (~9000 pts) with an ETF price
    # (~AUD 45) would fabricate enormous returns at every series switch.
    bench_ticker = market_features.pick_benchmark_ticker(silver)
    if bench_ticker is None:
        raise StorageError(
            "No benchmark (^AXJO/VAS.AX) rows in silver prices; cannot build "
            "market features."
        )
    logger.info("Benchmark series: %s", bench_ticker)
    benchmark = silver.loc[
        silver["ticker"] == bench_ticker
    ].reset_index(drop=True)
    # The universe excludes every benchmark ticker (the index itself AND
    # the listed ETF proxy) — neither is a tradable portfolio constituent.
    universe = silver.loc[
        ~silver["ticker"].isin(market_features.BENCHMARK_TICKERS)
    ].reset_index(drop=True)

    calendar_dates = pd.DatetimeIndex(sorted(universe["date"].unique()))
    logger.info("Building features: %d universe rows, %d sessions",
                len(universe), len(calendar_dates))

    # ---- 2. ticker-level features ---------------------------------------
    # daily_return is carried through for beta/volatility inputs and dropped
    # from the final table (return_1d already covers it).
    frames: list[pd.DataFrame] = [
        universe[["ticker", "date", "daily_return"]]
    ]

    if cfg.get("returns", {}).get("enabled", True):
        windows = list(cfg["returns"].get("windows_sessions", []))
        frames.append(price_features.compute_return_features(universe, windows))

    if cfg.get("moving_averages", {}).get("enabled", True):
        ma_cfg = cfg["moving_averages"]
        frames.append(price_features.compute_moving_average_features(
            universe,
            windows=list(ma_cfg.get("windows_sessions", [20])),
            slope_windows=list(ma_cfg.get("slope_windows_sessions", [])),
        ))

    if cfg.get("volatility", {}).get("enabled", True):
        vol_cfg = cfg["volatility"]
        frames.append(price_features.compute_volatility_features(
            universe,
            windows=list(vol_cfg.get("windows_sessions", [20])),
            atr_window=int(vol_cfg.get("atr_window", 14)),
            drawdown_window=int(vol_cfg.get("drawdown_window", 252)),
        ))

    if cfg.get("technical", {}).get("enabled", True):
        tech_cfg = cfg["technical"]
        frames.append(price_features.compute_technical_features(
            universe,
            rsi_window=int(tech_cfg.get("rsi_window", 14)),
            macd=dict(tech_cfg.get("macd", {"fast": 12, "slow": 26,
                                            "signal": 9})),
            bollinger=dict(tech_cfg.get("bollinger",
                                        {"window": 20, "n_std": 2.0})),
            stochastic=dict(tech_cfg.get("stochastic",
                                         {"window": 14, "smooth": 3})),
            adx_window=int(tech_cfg.get("adx_window", 14)),
            recent_high_low_window=int(
                tech_cfg.get("recent_high_low_window", 63)),
        ))

    if cfg.get("volume", {}).get("enabled", True):
        vol_cfg = cfg["volume"]
        frames.append(price_features.compute_volume_features(
            universe,
            windows=list(vol_cfg.get("windows_sessions", [5, 20])),
            min_avg_dollar_turnover_aud=float(
                vol_cfg.get("min_avg_dollar_turnover_aud", 250000)),
        ))

    features = pd.concat(frames, axis=1)

    # ---- 3. market-level features ----------------------------------------
    bench_cfg = cfg.get("market", {})
    bench_frame = market_features.compute_benchmark_features(
        benchmark, list(bench_cfg.get("regime_windows", [20, 100])))
    features = features.merge(bench_frame, on="date", how="left")

    if bench_cfg.get("enabled", True):
        # Breadth needs the per-ticker SMA ratios computed above.
        breadth = market_features.compute_breadth_features(
            features, list(bench_cfg.get("breadth_ma_windows", [20, 50])))
        features = features.merge(breadth, on="date", how="left")

    momentum_window = int(cfg.get("momentum", {}).get(
        "momentum_percentile_window", 63))
    ret20 = "return_20d"
    if ret20 in features.columns:
        # Time-series percentile of the 20d return over a trailing
        # momentum_percentile_window of that stock's own sessions.
        features["momentum_ts_pctile_63d"] = features.groupby(
            "ticker", sort=False
        )[ret20].transform(
            lambda s: s.rolling(momentum_window).rank(pct=True)
        )
    else:
        features["momentum_ts_pctile_63d"] = float("nan")

    if "return_20d" in features.columns and "bench_return_20d" \
            in features.columns:
        features["momentum_rel_bench_20d"] = (
            features["return_20d"] - features["bench_return_20d"]
        )
    else:
        features["momentum_rel_bench_20d"] = float("nan")
    if "return_60d" in features.columns and "bench_return_20d" \
            in features.columns:
        features["momentum_rel_bench_63d"] = (
            features["return_60d"] - features["bench_return_20d"]
        )
    else:
        features["momentum_rel_bench_63d"] = float("nan")

    xsec = market_features.compute_cross_sectional_features(
        features, momentum_window=20)
    features = pd.concat([features, xsec], axis=1)

    if cfg.get("volatility", {}).get("beta_window"):
        bench_ret = bench_frame.set_index("date")["bench_return_1d"]
        features["beta_252d"] = market_features.compute_beta_feature(
            features, bench_ret,
            beta_window=int(cfg["volatility"]["beta_window"]),
        )

    # ---- 4. point-in-time macro ------------------------------------------
    macro_ids = _load_macro_series_ids(features_cfg)
    macro_path = lake.silver_dataset_path("macro_observations")
    if macro_path.exists() and macro_ids:
        macro_matrix = build_macro_pit_matrix(
            read_parquet(macro_path), calendar_dates, macro_ids)
        macro_matrix = add_derived_macro_features(macro_matrix)
        features = features.merge(
            macro_matrix.reset_index(), on="date", how="left")
    else:
        logger.warning("Macro silver absent or unconfigured — macro "
                       "features omitted this build.")

    # ---- 5. point-in-time short interest ---------------------------------
    short_path = lake.silver_dataset_path("short_positions")
    if short_path.exists():
        short_feats = compute_short_pit_features(
            read_parquet(short_path), features[["ticker", "date"]])
        features = pd.concat([features, short_feats], axis=1)
    else:
        logger.warning("Short-position silver absent — short-interest "
                       "features omitted this build.")

    # ---- 6. targets (separate dataset) -----------------------------------
    def _horizon(cfg_key: str, default: str) -> int:
        raw = str(targets_cfg.get(cfg_key, default))  # e.g. 'future_return_3d'
        digits = "".join(ch for ch in raw.split("_")[-1] if ch.isdigit())
        return int(digits) if digits else int(default)

    horizons = (
        _horizon("primary", "3d"),
        _horizon("secondary", "10d"),
    )
    labels = compute_target_labels(
        universe,
        horizons=horizons,
        direction_threshold=float(targets_cfg.get("direction_threshold", 0.0)),
    )
    targets = pd.concat([universe[["ticker", "date"]], labels], axis=1)
    targets = targets[list(TARGETS_COLUMNS)]

    features["feature_set_version"] = version

    # ---- 7. quality + write ----------------------------------------------
    feature_cols = [
        c for c in features.columns
        if c not in {"ticker", "date", "feature_set_version", "daily_return"}
    ]
    features = features.drop(columns=["daily_return"])
    report = feature_checks.run_feature_quality_checks(
        features, targets, config_version=version) if with_quality else None
    if report is not None:
        report.save(lake.report_path("data_quality_features", ".json"))
        report.assert_passed()
        logger.info("Feature quality: %s (%d warnings)",
                    report.to_dict()["status"], report.warn_count())

    writer = ParquetWriter(settings)
    features_path = lake.gold_dataset_path("features")
    writer.write(features, features_path, run_id=registry.run_id,
                 overwrite=True, dataset_name="features")
    targets_path = lake.gold_dataset_path("target_labels")
    writer.write(targets, targets_path, run_id=registry.run_id,
                 overwrite=True, dataset_name="target_labels")

    _register_gold_views(settings, lake, features_path, targets_path)

    return FeatureBuildResult(
        features_path=features_path,
        targets_path=targets_path,
        report=report,
        feature_set_version=version,
        n_rows=len(features),
        n_features=len(feature_cols),
    )


def _register_gold_views(
    settings,  # noqa: ANN001
    lake: DataLake,
    features_path: Path,
    targets_path: Path,
) -> None:
    from asx_quant.storage.duckdb import get_warehouse

    warehouse = get_warehouse(settings)
    try:
        warehouse.connect()
        warehouse.register_view("gold", "features", features_path,
                                layer="gold")
        warehouse.register_view("gold", "target_labels", targets_path,
                                layer="gold")
    finally:
        warehouse.close()
