"""Phase 4 tests: feature engineering, point-in-time joins, targets, checks.

All offline — synthetic price/macro/short frames and a temporary lake for
the end-to-end build. Leakage invariants are asserted directly:

* ticker-level features recomputed on a TRUNCATED frame must equal the
  full-frame values at the truncation session (no future data);
* macro/short features must be NaN before the value's publication_date;
* labels must reference only strictly-later sessions and live in their own
  dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from asx_quant.exceptions import DataQualityError
from asx_quant.features.base import ensure_sorted
from asx_quant.features.build import build_feature_set
from asx_quant.features.checks import run_feature_quality_checks
from asx_quant.features.macro_features import build_macro_pit_matrix
from asx_quant.features.market import (
    compute_benchmark_features,
    compute_beta_feature,
    compute_breadth_features,
    compute_cross_sectional_features,
)
from asx_quant.features.price import (
    _rsi,
    compute_moving_average_features,
    compute_return_features,
    compute_technical_features,
    compute_volatility_features,
    compute_volume_features,
)
from asx_quant.features.short_features import compute_short_pit_features
from asx_quant.features.targets import compute_target_labels
from asx_quant.observability.runs import RunRegistry
from asx_quant.storage.paths import DataLake

# ------------------------------------------------------------ synthetic data
SESSIONS = pd.bdate_range("2023-01-02", periods=320)
TICKERS = ["CBA.AX", "BHP.AX", "WES.AX"]
BENCH = "IDX_AXJO"

RNG = np.random.default_rng(42)


def _synthetic_prices() -> pd.DataFrame:
    """Deterministic synthetic universe: 3 tickers + benchmark."""
    frames = []
    bench_ret = pd.Series(
        RNG.normal(0.0004, 0.008, size=len(SESSIONS)), index=SESSIONS
    )
    for i, ticker in enumerate([BENCH, *TICKERS]):
        if ticker == BENCH:
            ret = bench_ret
        else:
            beta = 0.6 + 0.4 * i  # 0.6, 1.0, 1.4
            ret = beta * bench_ret + RNG.normal(0.0001, 0.006,
                                                size=len(SESSIONS))
        close = pd.Series(
            10.0 * np.exp(ret.cumsum()), index=SESSIONS
        )
        high = close * (1 + RNG.uniform(0.001, 0.02, len(SESSIONS)))
        low = close * (1 - RNG.uniform(0.001, 0.02, len(SESSIONS)))
        frames.append(pd.DataFrame({
            "ticker": ticker,
            "date": SESSIONS,
            "open_price": close * (1 - RNG.uniform(0, 0.01, len(SESSIONS))),
            "high_price": high,
            "low_price": low,
            "close_price": close,
            "adjusted_close": close,  # no distributions in synthetic data
            "volume": RNG.integers(100_000, 2_000_000, len(SESSIONS)),
        }))
    prices = pd.concat(frames, ignore_index=True)
    prices["daily_return"] = prices.groupby("ticker")["adjusted_close"].pct_change()
    return ensure_sorted(prices)


PRICES = _synthetic_prices()


def _cfg_technical() -> dict:
    return dict(rsi_window=14,
                macd={"fast": 12, "slow": 26, "signal": 9},
                bollinger={"window": 20, "n_std": 2.0},
                stochastic={"window": 14, "smooth": 3},
                adx_window=14,
                recent_high_low_window=63)


# ----------------------------------------------------- price-derived features
def test_return_features_known_values() -> None:
    one = PRICES[PRICES["ticker"] == "CBA.AX"].reset_index(drop=True)
    out = compute_return_features(one, [1, 3])
    assert np.isclose(out["return_1d"].iloc[10],
                      one["adjusted_close"].iloc[10]
                      / one["adjusted_close"].iloc[9] - 1)
    assert np.isclose(out["return_3d"].iloc[10],
                      one["adjusted_close"].iloc[10]
                      / one["adjusted_close"].iloc[7] - 1)
    assert out["return_3d"].iloc[:3].isna().all()  # warm-up is NaN


def test_moving_average_features() -> None:
    one = PRICES[PRICES["ticker"] == "BHP.AX"].reset_index(drop=True)
    out = compute_moving_average_features(one, [5, 20], slope_windows=[5, 20])
    expected_sma = one["adjusted_close"].rolling(20).mean()
    assert np.isclose(out["price_to_sma_20"].iloc[30],
                      one["adjusted_close"].iloc[30] / expected_sma.iloc[30])
    assert out["price_to_sma_20"].iloc[:19].isna().all()
    assert set(out["sma_cross_5_20"].dropna().unique()) <= {-1.0, 0.0, 1.0}
    assert out["sma_20_slope_5d"].iloc[:24].isna().all()


def test_volatility_features_bounds() -> None:
    one = PRICES[PRICES["ticker"] == "WES.AX"].reset_index(drop=True)
    out = compute_volatility_features(one, [5, 20], atr_window=14,
                                      drawdown_window=252)
    vols = out["volatility_20d"].dropna()
    assert (vols > 0).all() and (vols < 2.0).all()
    assert out["atr_14"].dropna().gt(0).all()
    dd = out["drawdown_252d"].dropna()
    assert (dd <= 0).all() and (dd > -1).all()


def test_rsi_bounds_and_pure_trend_edges() -> None:
    up = pd.Series(np.linspace(1.0, 2.0, 40))
    down = pd.Series(np.linspace(2.0, 1.0, 40))
    assert _rsi(up, 14).iloc[-1] == 100.0
    assert _rsi(down, 14).iloc[-1] == 0.0
    tech = compute_technical_features(
        PRICES[PRICES["ticker"] == "CBA.AX"].reset_index(drop=True),
        **_cfg_technical())
    rsi = tech["rsi_14"].dropna()
    assert rsi.between(0, 100).all()
    stoch = tech[["stoch_k_14", "stoch_d_14"]].dropna()
    assert ((stoch >= 0) & (stoch <= 100)).all().all()
    assert tech["bollinger_pctb"].dropna().notna().all()
    assert tech["adx_14"].dropna().ge(0).all()
    assert tech["pct_from_63d_high"].dropna().le(0).all()
    assert tech["pct_from_63d_low"].dropna().ge(0).all()


def test_volume_features_and_liquidity_flag() -> None:
    one = PRICES[PRICES["ticker"] == "CBA.AX"].reset_index(drop=True)
    out = compute_volume_features(one, [5, 20],
                                  min_avg_dollar_turnover_aud=250_000)
    assert out["volume_ratio_20d"].dropna().gt(0).all()
    assert out["liquid_flag"].dropna().eq(1.0).all()  # synthetic volume high


# ------------------------------------------------- market / cross-sectional
def test_benchmark_features_and_breadth() -> None:
    bench = PRICES[PRICES["ticker"] == BENCH].reset_index(drop=True)
    bf = compute_benchmark_features(bench, regime_windows=[20, 100])
    assert bf["bench_return_1d"].iloc[1] == pytest.approx(
        bench["adjusted_close"].iloc[1] / bench["adjusted_close"].iloc[0] - 1)
    assert bf["bench_close_to_sma_20"].iloc[:19].isna().all()

    uni = PRICES[PRICES["ticker"].isin(TICKERS)].reset_index(drop=True)
    sma = compute_moving_average_features(uni, [20], slope_windows=[])
    uni = pd.concat([uni, sma], axis=1)
    breadth = compute_breadth_features(uni, ma_windows=[20])
    assert breadth["breadth_above_sma_20"].dropna().between(0, 1).all()
    # exactly 3 tickers -> breadth values on a 1/3 grid
    vals = breadth["breadth_above_sma_20"].dropna().round(6)
    assert set(vals.unique()) <= {round(k / 3, 6) for k in range(4)}


def test_benchmark_rejects_mixed_ticker_series() -> None:
    """Regression: two benchmark series at different price levels MUST NOT
    be silently interleaved — that fabricated +/-80x returns and half-NaN
    columns in the first real build."""
    bench = PRICES[PRICES["ticker"] == BENCH].reset_index(drop=True)
    # A second "benchmark" at ETF-like price levels (~45), same sessions.
    proxy = bench.assign(
        ticker="VAS.AX", adjusted_close=45.0 + bench["adjusted_close"] * 0.001
    )
    mixed = pd.concat([bench, proxy], ignore_index=True)

    with pytest.raises(ValueError, match="single benchmark series"):
        compute_benchmark_features(mixed, regime_windows=[20, 100])

    # The single picked series is safe: one row per date, values equal to
    # that series' own pct_change, physically bounded.
    picked = mixed[mixed["ticker"] == "IDX_AXJO"]
    bf = compute_benchmark_features(picked, regime_windows=[20])
    expected = bench.sort_values("date")["adjusted_close"].pct_change()
    got = bf["bench_return_1d"]
    assert len(bf) == len(bench)
    np.testing.assert_allclose(
        got.to_numpy(), expected.to_numpy(), atol=1e-12, rtol=1e-9)
    assert got.abs().max() < 0.15


def test_pick_benchmark_ticker_priority() -> None:
    from asx_quant.features.market import pick_benchmark_ticker

    frame = pd.DataFrame({"ticker": ["VAS.AX", "IDX_AXJO"]})
    assert pick_benchmark_ticker(frame) == "IDX_AXJO"
    assert pick_benchmark_ticker(pd.DataFrame({"ticker": ["VAS.AX"]})) == \
        "VAS.AX"
    assert pick_benchmark_ticker(pd.DataFrame({"ticker": ["CBA.AX"]})) is None


def test_beta_recovers_known_factor() -> None:
    # Synthetic betas: CBA 1.0, BHP 1.4, WES 1.8 (enumerate starts at bench).
    uni = PRICES[PRICES["ticker"].isin(TICKERS)].reset_index(drop=True)
    bench = PRICES[PRICES["ticker"] == BENCH].reset_index(drop=True)
    bench_ret = bench.set_index("date")["daily_return"]
    beta = compute_beta_feature(uni, bench_ret, beta_window=252)
    uni = uni.assign(_beta=beta.to_numpy())
    last = uni.groupby("ticker").tail(1).set_index("ticker")["_beta"]
    assert last["CBA.AX"] == pytest.approx(1.0, abs=0.25)
    assert last["BHP.AX"] == pytest.approx(1.4, abs=0.25)
    assert last["WES.AX"] == pytest.approx(1.8, abs=0.25)


def test_cross_sectional_percentiles() -> None:
    uni = PRICES[PRICES["ticker"].isin(TICKERS)].reset_index(drop=True)
    rets = compute_return_features(uni, [20])
    uni = pd.concat([uni, rets], axis=1)
    xsec = compute_cross_sectional_features(uni, momentum_window=20)
    pct = xsec["momentum_xsec_pctile_20d"].dropna()
    assert pct.between(0, 1).all()
    # 3-ticker ranks: each session's ranks must be exactly {1/3, 2/3, 1}
    for _, group in uni[uni["return_20d"].notna()].groupby("date"):
        ranks = sorted(xsec.loc[group.index,
                                "momentum_xsec_pctile_20d"].to_numpy())
        assert ranks == pytest.approx([1 / 3, 2 / 3, 1.0])


# ------------------------------------------------------ point-in-time joins
def test_macro_pit_join_never_before_publication() -> None:
    calendar = pd.bdate_range("2024-02-01", periods=90)
    macro = pd.DataFrame({
        "series_name": ["abs_cpi", "abs_cpi"],
        "observation_date": pd.to_datetime(["2024-01-31", "2024-04-30"]),
        "publication_date": pd.to_datetime(["2024-03-06", "2024-06-05"]),
        "value": [4.1, 3.8],
    })
    matrix = build_macro_pit_matrix(macro, calendar, ["abs_cpi"])
    col = matrix["macro_abs_cpi"]
    # Observation period Jan-31: value MUST be NaN before publication.
    assert col.loc[:pd.Timestamp("2024-03-05")].isna().all()
    assert col.loc[pd.Timestamp("2024-03-06"):pd.Timestamp("2024-06-04")].eq(
        4.1).all()
    assert col.loc[pd.Timestamp("2024-06-05"):].eq(3.8).all()


def test_short_pit_join_never_before_publication() -> None:
    dates = pd.bdate_range("2024-02-01", periods=8)
    universe = pd.DataFrame({
        "ticker": ["CBA.AX"] * len(dates),
        "date": dates,
    })
    positions = pd.DataFrame({
        "ticker": ["CBA.AX", "CBA.AX"],
        "trade_date": [dates[1], dates[5]],
        "publication_date": [dates[2], dates[6]],  # published AFTER trade date
        "reported_short_positions": [100.0, 250.0],
        "pct_of_product_in_issue": [5.0, 6.0],
    })
    out = compute_short_pit_features(positions, universe, change_window=5)
    # trade dates BEFORE publication must be NaN (leakage guard)
    assert out["short_pct"].iloc[0] != out["short_pct"].iloc[0] or True
    assert pd.isna(out["short_pct"].iloc[0])
    assert pd.isna(out["short_pct"].iloc[1])  # trade date d1, published d2
    assert out["short_pct"].iloc[2] == 5.0
    assert out["short_pct"].iloc[5] == 5.0   # stale fill until next pub
    assert out["short_pct"].iloc[6] == 6.0


# ---------------------------------------------------------------- targets
def test_target_labels_and_availability_dates() -> None:
    one = PRICES[PRICES["ticker"] == "CBA.AX"].reset_index(drop=True)
    labels = compute_target_labels(one, horizons=(3, 10))
    close = one["adjusted_close"]
    assert labels["future_return_3d"].iloc[10] == pytest.approx(
        close.iloc[13] / close.iloc[10] - 1)
    assert labels["future_return_3d"].iloc[-3:].isna().all()
    assert labels["label_available_date_3d"].iloc[10] == one["date"].iloc[13]
    dir3 = labels["direction_3d"].dropna()
    ret3 = labels["future_return_3d"].dropna()
    assert ((ret3 > 0) == (dir3 == 1.0)).all()


# ------------------------------------------------------------ quality rules
def _features_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["CBA.AX"] * 3,
        "date": pd.bdate_range("2024-01-02", periods=3),
        "return_1d": [0.01, -0.02, 0.03],
        "rsi_14": [55.0, 48.0, 61.0],
        "feature_set_version": ["v0.1"] * 3,
    })


def _targets_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=3)
    return pd.DataFrame({
        "ticker": ["CBA.AX"] * 3,
        "date": dates,
        "future_return_3d": [0.05, -0.01, np.nan],
        "direction_3d": [1.0, 0.0, np.nan],
        "label_available_date_3d": dates + pd.Timedelta(days=7),
        "future_return_10d": [np.nan, np.nan, np.nan],
        "direction_10d": [np.nan, np.nan, np.nan],
        "label_available_date_10d": [pd.NaT] * 3,
    })


def test_feature_quality_clean_passes() -> None:
    report = run_feature_quality_checks(_features_frame(), _targets_frame(),
                                        config_version="v0.1")
    assert report.passed, [r.name for r in report.errors if not r.passed]


def test_feature_quality_label_column_in_features_is_error() -> None:
    bad = _features_frame().assign(future_return_3d=[0.1, 0.2, 0.3])
    report = run_feature_quality_checks(bad, _targets_frame(),
                                        config_version="v0.1")
    rule = next(r for r in report.results
                if r.name == "no_label_columns_in_features")
    assert rule.severity == "error" and not rule.passed
    with pytest.raises(DataQualityError):
        report.assert_passed()


def test_feature_quality_direction_mismatch_is_error() -> None:
    targets = _targets_frame()
    targets.loc[0, "direction_3d"] = 0.0  # return 0.05 but direction 0
    report = run_feature_quality_checks(_features_frame(), targets,
                                        config_version="v0.1")
    rule = next(r for r in report.results
                if r.name == "direction_matches_return_3d")
    assert rule.severity == "error" and not rule.passed


def test_feature_quality_availability_before_session_is_error() -> None:
    targets = _targets_frame()
    targets.loc[0, "label_available_date_3d"] = targets.loc[0, "date"]
    report = run_feature_quality_checks(_features_frame(), targets,
                                        config_version="v0.1")
    rule = next(r for r in report.results
                if r.name == "label_available_after_session")
    assert rule.severity == "error" and not rule.passed


def test_feature_quality_inf_values_are_error() -> None:
    bad = _features_frame()
    bad.loc[0, "rsi_14"] = float("inf")
    report = run_feature_quality_checks(bad, _targets_frame(),
                                        config_version="v0.1")
    rule = next(r for r in report.results if r.name == "feature_values_finite")
    assert rule.severity == "error" and not rule.passed


# --------------------------------------------- truncation leakage regression
def test_price_features_truncation_identical() -> None:
    """Features at session k must not change when later data is removed."""
    uni = PRICES[PRICES["ticker"].isin(TICKERS)].reset_index(drop=True)
    full = pd.concat([
        compute_return_features(uni, [1, 20, 60]),
        compute_moving_average_features(uni, [5, 20], slope_windows=[5, 20]),
        compute_volatility_features(uni, [5, 20], 14, 252),
        compute_technical_features(uni, **_cfg_technical()),
        compute_volume_features(uni, [5, 20], 250_000),
    ], axis=1)

    cutoff = SESSIONS[200]
    truncated = uni[uni["date"] <= cutoff].reset_index(drop=True)
    part = pd.concat([
        compute_return_features(truncated, [1, 20, 60]),
        compute_moving_average_features(truncated, [5, 20],
                                        slope_windows=[5, 20]),
        compute_volatility_features(truncated, [5, 20], 14, 252),
        compute_technical_features(truncated, **_cfg_technical()),
        compute_volume_features(truncated, [5, 20], 250_000),
    ], axis=1)

    mask = uni["date"] == cutoff
    cols = [c for c in full.columns if c in part.columns]
    for col in cols:
        for ticker in TICKERS:
            ai = uni.index[mask & (uni["ticker"] == ticker)][0]
            bi = truncated.index[(truncated["date"] == cutoff)
                                 & (truncated["ticker"] == ticker)][0]
            av, bv = full.loc[ai, col], part.loc[bi, col]
            if pd.isna(av) and pd.isna(bv):
                continue
            assert av == pytest.approx(bv), f"{col} differs for {ticker}"


# ------------------------------------------------------------- e2e gold build
class _S:
    parquet_compression = "zstd"
    parquet_compression_level = 3


def _write_silver(lake: DataLake, name: str, frame: pd.DataFrame) -> None:
    from asx_quant.storage.parquet import ParquetWriter

    path = lake.silver_dataset_path(name)
    ParquetWriter(_S()).write(frame, path, run_id="test_run",
                              dataset_name=name)


def test_build_feature_set_end_to_end(tmp_path, monkeypatch) -> None:
    from asx_quant.config.loader import load_settings

    monkeypatch.setattr("asx_quant.features.build._register_gold_views",
                        lambda *a, **k: None)

    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    registry = RunRegistry.start(lake, config_version="v0.1")

    prices = PRICES.copy()
    _write_silver(lake, "equity_prices", prices)

    # macro: one CPI observation published mid-history (leakage-visible)
    mid = SESSIONS[150] + pd.Timedelta(days=1)
    macro = pd.DataFrame({
        "series_name": ["abs_cpi"],
        "source_series_id": ["GCPIAGYP"],
        "observation_date": [SESSIONS[100]],
        "value": [4.2],
        "publication_date": [mid],
        "publication_date_method": ["estimated_publication_lag"],
        "potentially_revised": [True],
        "frequency": ["quarterly"],
        "unit": ["per_cent_yoy"],
        "source": ["abs"],
        "source_table": ["G1"],
    })
    _write_silver(lake, "macro_observations", macro)

    # short positions: published strictly after trade date
    short = pd.DataFrame({
        "ticker": [f"{t}.AX" if not t.endswith(".AX") else t
                   for t in ["CBA", "BHP", "WES"]],
        "code": ["CBA", "BHP", "WES"],
        "trade_date": [SESSIONS[100]] * 3,
        "publication_date": [SESSIONS[101]] * 3,
        "reported_short_positions": [1000.0, 2000.0, 3000.0],
        "pct_of_product_in_issue": [1.0, 2.0, 3.0],
        "report_date": [SESSIONS[101]] * 3,
        "publication_date_method": ["estimated_from_report_calendar"] * 3,
    })
    _write_silver(lake, "short_positions", short)

    settings = load_settings()
    result = build_feature_set(lake, settings, registry)

    features = pd.read_parquet(result.features_path)
    targets = pd.read_parquet(result.targets_path)

    assert result.feature_set_version == "v0.1"
    assert result.n_features > 40
    assert not targets.empty

    # leakage invariants on the built dataset
    assert not [c for c in features.columns
                if c.startswith(("future_", "direction_"))]
    assert (features["feature_set_version"] == "v0.1").all()

    # macro: NaN strictly before the publication date
    macro_col = features["macro_abs_cpi"]
    before = features[features["date"] < mid]
    assert macro_col.loc[before.index].isna().all()
    after = features[features["date"] >= mid]
    assert macro_col.loc[after.index].notna().all()

    # short: NaN on the trade date, available from publication
    trade_rows = features[features["date"] == SESSIONS[100]]
    assert trade_rows["short_pct"].isna().all()
    pub_rows = features[features["date"] >= SESSIONS[101]]
    assert pub_rows["short_pct"].notna().all()

    # targets: availability strictly >= 3 sessions after the feature session
    labelled = targets[targets["label_available_date_3d"].notna()]
    sessions = pd.DatetimeIndex(sorted(targets["date"].unique()))
    pos_avail = sessions.searchsorted(
        pd.DatetimeIndex(labelled["label_available_date_3d"]))
    pos_own = sessions.searchsorted(pd.DatetimeIndex(labelled["date"]))
    assert (pos_avail - pos_own >= 3).all()
    assert not targets["ticker"].str.startswith("IDX_").any()
