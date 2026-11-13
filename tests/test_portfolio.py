"""Phase 6 tests: eligibility, ranking, weighting, schedule, orchestrator.

All offline — synthetic predictions/features frames and a temporary lake
for the end-to-end build. Platform invariants are asserted directly:

* long-only weights, max-stock cap, gross exposure + cash = 1;
* the look-ahead guard (trained_through strictly before rebalance date);
* eligibility (liquidity flag, min history) is never bypassed to fill slots;
* rebalance dates are actual sessions; holiday Mondays shift forward.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from asx_quant.portfolio import checks as portfolio_checks
from asx_quant.portfolio.schedule import next_session, rebalance_dates
from asx_quant.portfolio.selection import (
    eligible_universe,
    history_sessions_per_row,
    load_signal,
    rank_candidates,
)
from asx_quant.portfolio.weights import (
    apply_weight_cap,
    build_weights,
    enforce_min_weight,
    inverse_volatility_weights,
    turnover_vs_previous,
)

# ------------------------------------------------------------ synthetic data
SESSIONS = pd.bdate_range("2024-01-01", periods=260)  # 1 trading year
TICKERS = [f"{c}.AX" for c in
           ["CBA", "BHP", "WES", "ANZ", "TLS", "WOW", "RIO", "WBC",
            "MQG", "SUN", "PMC", "S32", "ORI", "QBE", "BXB", "STO",
            "ALL", "TWE", "MFG", "COL", "NAB", "GMG", "JHX", "AMCOR",
            "ASX", "BKL"]]
SIGNAL_START = SESSIONS[20]


def _features_frame() -> pd.DataFrame:
    """Every ticker trades every session; half are liquid after session 100."""
    rows = []
    rng = np.random.default_rng(3)
    for ticker in TICKERS:
        liquid = 1.0 if TICKERS.index(ticker) % 2 == 0 else np.nan
        for i, date in enumerate(SESSIONS):
            rows.append({
                "ticker": ticker,
                "date": date,
                "liquid_flag": liquid if i >= 100 else np.nan,
                "volatility_20d": abs(rng.normal(0.25, 0.05)),
            })
    return pd.DataFrame(rows)


def _predictions_frame() -> pd.DataFrame:
    """OOS predictions from the first SIGNAL_START session, all tickers."""
    rows = []
    rng = np.random.default_rng(11)
    for date in SESSIONS[20:]:
        for ticker in TICKERS:
            rows.append({
                "ticker": ticker,
                "date": date,
                "model_name": "hist_gbt_reg",
                "model_kind": "regressor",
                "model_version": "v0.1",
                "feature_set_version": "v0.1",
                "fold_id": 0,
                "trained_through": SESSIONS[19],  # one static fold
                "prediction": rng.normal(0.002, 0.01),
            })
    return pd.DataFrame(rows)


FEATURES = _features_frame()
PREDICTIONS = _predictions_frame()
HISTORY = history_sessions_per_row(FEATURES.sort_values(["ticker", "date"],
                                                        kind="stable")
                                   .reset_index(drop=True)
                                   .pipe(lambda f: f.assign(
                                       _idx=f.index))
                                   ).reindex(FEATURES.index)


# ----------------------------------------------------------------- schedule
def test_rebalance_dates_weekly_mondays() -> None:
    dates = rebalance_dates(SESSIONS, "weekly", "monday", SIGNAL_START)
    assert dates, "schedule must produce dates"
    # all dates are actual sessions on (or shifted from) Mondays
    assert all(d in set(SESSIONS) for d in dates)
    assert all(d >= SIGNAL_START for d in dates)
    # weekly spacing of ~5-7 sessions; no duplicates
    assert len(dates) == len(set(dates))
    gaps = [(b - a).days for a, b in zip(dates, dates[1:], strict=False)]
    assert all(2 <= g <= 12 for g in gaps)


def test_rebalance_dates_holiday_monday_shifts_forward() -> None:
    # remove one Monday from the session calendar (public holiday)
    monday = next(d for d in SESSIONS if d.weekday() == 0)
    reduced = SESSIONS[SESSIONS != monday]
    dates = rebalance_dates(reduced, "weekly", "monday", reduced[0])
    assert monday not in dates
    assert monday + dt.timedelta(days=1) in dates  # shifted to Tuesday


def test_rebalance_dates_respect_first_signal() -> None:
    dates = rebalance_dates(SESSIONS, "weekly", "monday", SESSIONS[100])
    assert min(dates) >= SESSIONS[100]


def test_rebalance_dates_invalid_frequency_raises() -> None:
    with pytest.raises(ValueError, match="frequency"):
        rebalance_dates(SESSIONS, "daily", "monday", SIGNAL_START)
    with pytest.raises(ValueError, match="weekday"):
        rebalance_dates(SESSIONS, "weekly", "sunday", SIGNAL_START)


def test_next_session() -> None:
    assert next_session(SESSIONS, SESSIONS[10]) == SESSIONS[11]
    assert next_session(SESSIONS, SESSIONS[-1]) is None


# ---------------------------------------------------------------- selection
def test_load_signal_validates_kind() -> None:
    signals = load_signal(PREDICTIONS, "hist_gbt_reg", "predicted_return_3d")
    assert set(signals.columns) >= {"ticker", "date", "signal",
                                    "trained_through"}
    with pytest.raises(ValueError, match="regressor"):
        load_signal(PREDICTIONS, "hist_gbt_reg", "prob_up_3d")
    with pytest.raises(ValueError, match="no predictions"):
        load_signal(PREDICTIONS, "no_such_model", "predicted_return_3d")


def test_eligibility_filters() -> None:
    date = SESSIONS[150]
    signals = load_signal(PREDICTIONS, "hist_gbt_reg", "predicted_return_3d")
    eligible = eligible_universe(
        signals, FEATURES, date, require_liquidity_flag=True,
        min_history_sessions=100, history=HISTORY,
    )
    # only liquid (even-index) tickers are eligible
    eligible_names = set(eligible["ticker"])
    for i, ticker in enumerate(TICKERS):
        if i % 2 == 1:
            assert ticker not in eligible_names
        else:
            assert ticker in eligible_names
    # history filter: every eligible ticker has enough sessions
    assert (eligible["history_sessions"] >= 100).all()
    # look-ahead guard holds
    assert (pd.to_datetime(eligible["trained_through"]) < date).all()


def test_eligibility_excludes_lookahead_predictions() -> None:
    date = SESSIONS[150]
    signals = load_signal(PREDICTIONS, "hist_gbt_reg", "predicted_return_3d")
    leaky = signals.copy()
    leaky["trained_through"] = date  # fold trained ON the rebalance date
    eligible = eligible_universe(leaky, FEATURES, date, history=HISTORY)
    assert eligible.empty


def test_eligibility_empty_when_no_predictions() -> None:
    signals = load_signal(PREDICTIONS, "hist_gbt_reg", "predicted_return_3d")
    eligible = eligible_universe(signals, FEATURES, SESSIONS[5],  # pre-signal
                                 history=HISTORY)
    assert eligible.empty


def test_rank_candidates_top_n_and_ranks() -> None:
    date = SESSIONS[150]
    signals = load_signal(PREDICTIONS, "hist_gbt_reg", "predicted_return_3d")
    eligible = eligible_universe(signals, FEATURES, date,
                                 min_history_sessions=100, history=HISTORY)
    ranked = rank_candidates(eligible, n_holdings=5)
    assert len(ranked) == 5
    assert list(ranked["rank"]) == [1, 2, 3, 4, 5]
    assert ranked["signal"].is_monotonic_decreasing
    # more than n_holdings eligible -> strictly the strongest signals win
    strongest = set(eligible.nlargest(5, "signal")["ticker"])
    assert set(ranked["ticker"]) == strongest


# ------------------------------------------------------------------ weights
def test_equal_weights_under_and_over_cap() -> None:
    raw = pd.Series([0.05] * 20)          # cap 0.10 -> untouched
    weights, cash = apply_weight_cap(raw, 0.10, prefer_cash=True)
    assert cash == pytest.approx(0.0)
    assert (weights == 0.05).all()

    raw8 = pd.Series([0.125] * 8)         # 1/8 > cap -> clamped, cash rest
    weights8, cash8 = apply_weight_cap(raw8, 0.10, prefer_cash=True)
    assert (weights8 == 0.10).all()
    assert cash8 == pytest.approx(0.20)


def test_weight_cap_water_filling_when_not_prefer_cash() -> None:
    raw = pd.Series([0.30, 0.30, 0.20, 0.10, 0.10])
    weights, cash = apply_weight_cap(raw, 0.25, prefer_cash=False)
    assert cash == pytest.approx(0.0, abs=1e-9)
    assert (weights <= 0.25 + 1e-9).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    # uncapped names absorb the excess proportionally to headroom
    assert weights.loc[2] > 0.20
    # fully-saturated portfolio: everyone at cap, excess has nowhere to go
    raw2 = pd.Series([0.50] * 4)
    weights2, cash2 = apply_weight_cap(raw2, 0.15, prefer_cash=False)
    assert (weights2 == 0.15).all()
    assert cash2 == pytest.approx(0.40)


def test_inverse_volatility_weights() -> None:
    vols = pd.Series([0.10, 0.20, 0.40])
    raw = inverse_volatility_weights(vols)
    assert raw.sum() == pytest.approx(1.0)
    # lower volatility -> larger weight
    assert raw.iloc[0] > raw.iloc[1] > raw.iloc[2]
    assert raw.iloc[0] / raw.iloc[2] == pytest.approx(4.0)


def test_enforce_min_weight_drops_dust() -> None:
    weights = pd.Series([0.90, 0.05, 0.05])
    out = enforce_min_weight(weights, 0.10)
    assert len(out) == 1 and out.iloc[0] == pytest.approx(1.0)
    # never silently empties the portfolio
    tiny = pd.Series([0.01, 0.01])
    assert len(enforce_min_weight(tiny, 0.10)) == 2


def test_build_weights_equal_and_vol_adjusted() -> None:
    ranked = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(4)],
        "rank": [1, 2, 3, 4],
        "volatility_20d": [0.10, 0.20, 0.40, 0.80],
    })
    frame, cash = build_weights(
        ranked, "equal_weight", {"max_stock_weight": 0.10}, )
    assert cash == pytest.approx(0.60)
    assert (frame["target_weight"] == 0.10).all()

    frame2, cash2 = build_weights(
        ranked, "volatility_adjusted",
        {"max_stock_weight": 0.40,
         "vol_adjust": {"min_weight": 0.05}},
    )
    # T0's raw inverse-vol weight is 0.533 > cap 0.40 -> clamped, residual
    # to cash (prefer_cash_over_violation defaults to True).
    assert cash2 == pytest.approx(1.0 - (0.40 + 0.266666 + 0.133333 + 0.066667),
                                  abs=1e-4)
    assert (frame2["target_weight"] <= 0.40 + 1e-9).all()
    w = frame2.set_index("ticker")["target_weight"]
    assert w["T0"] > w["T3"]


def test_turnover_metric() -> None:
    current = pd.DataFrame({"ticker": ["A", "B"], "target_weight": [0.5, 0.5]})
    previous = pd.DataFrame({"ticker": ["A", "C"], "target_weight": [0.5, 0.5]})
    # A kept (0), B bought (+0.5), C sold (-0.5) -> 0.5 * 1.0 = 0.5
    assert turnover_vs_previous(current, previous) == pytest.approx(0.5)
    assert turnover_vs_previous(current, current) == pytest.approx(0.0)
    assert turnover_vs_previous(current, pd.DataFrame(
        columns=["ticker", "target_weight"])) == pytest.approx(0.5)


# ------------------------------------------------------------------- checks
def _portfolios_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "rebalance_date": [pd.Timestamp("2024-06-03")] * 3,
        "ticker": ["A.AX", "B.AX", "C.AX"],
        "target_weight": [0.05, 0.05, 0.05],
        "trained_through": [pd.Timestamp("2024-05-30")] * 3,
    })


def _records(cash: float = 0.85, n: int = 3) -> list[dict]:
    return [{
        "rebalance_date": "2024-06-03",
        "n_holdings": n, "n_holdings_target": 20,
        "cash_weight": cash, "turnover": 0.5,
        "is_session": True, "all_liquid": True,
        "min_history_sessions": 260.0, "method": "equal_weight",
    }]


def test_portfolio_quality_clean_passes() -> None:
    report = portfolio_checks.run_portfolio_quality_checks(
        _portfolios_frame(), _records(), max_stock_weight=0.10,
        max_holdings=20, min_history_sessions=250,
        require_liquidity_flag=True)
    assert report.passed, [r.name for r in report.errors if not r.passed]
    # the sector rule is a documented warn, always present in v0.1
    warn = next(r for r in report.results if r.name == "sector_weight_rule_skipped")
    assert warn.severity == "warn" and not warn.passed


@pytest.mark.parametrize("mutate,rule", [
    (lambda p: p.assign(target_weight=[0.05, 0.05, -0.01]),
     "long_only_weights"),
    (lambda p: p.assign(target_weight=[0.20, 0.05, 0.05]),
     "max_stock_weight_respected"),
    (lambda p: p.assign(trained_through=pd.Timestamp("2024-06-03")),
     "no_lookahead_signals"),
])
def test_portfolio_quality_row_violations(mutate, rule: str) -> None:
    report = portfolio_checks.run_portfolio_quality_checks(
        mutate(_portfolios_frame()), _records(), max_stock_weight=0.10,
        max_holdings=20, min_history_sessions=250,
        require_liquidity_flag=True)
    target = next(r for r in report.results if r.name == rule)
    assert target.severity == "error" and not target.passed


def test_portfolio_quality_record_violations() -> None:
    # gross exposure != 1
    report = portfolio_checks.run_portfolio_quality_checks(
        _portfolios_frame(), _records(cash=0.80), max_stock_weight=0.10,
        max_holdings=20, min_history_sessions=250,
        require_liquidity_flag=True)
    rule = next(r for r in report.results
                if r.name == "gross_exposure_including_cash_is_one")
    assert rule.severity == "error" and not rule.passed

    # too many holdings
    report2 = portfolio_checks.run_portfolio_quality_checks(
        _portfolios_frame(), _records(n=25), max_stock_weight=0.10,
        max_holdings=20, min_history_sessions=250,
        require_liquidity_flag=True)
    rule2 = next(r for r in report2.results if r.name == "max_holdings_respected")
    assert rule2.severity == "error" and not rule2.passed

    # illiquid selection
    records = _records()
    records[0]["all_liquid"] = False
    report3 = portfolio_checks.run_portfolio_quality_checks(
        _portfolios_frame(), records, max_stock_weight=0.10,
        max_holdings=20, min_history_sessions=250,
        require_liquidity_flag=True)
    rule3 = next(r for r in report3.results if r.name == "selected_names_liquid")
    assert rule3.severity == "error" and not rule3.passed

    # short history
    records4 = _records()
    records4[0]["min_history_sessions"] = 100.0
    report4 = portfolio_checks.run_portfolio_quality_checks(
        _portfolios_frame(), records4, max_stock_weight=0.10,
        max_holdings=20, min_history_sessions=250,
        require_liquidity_flag=True)
    rule4 = next(r for r in report4.results if r.name == "min_history_respected")
    assert rule4.severity == "error" and not rule4.passed


def test_portfolio_quality_empty_is_error() -> None:
    report = portfolio_checks.run_portfolio_quality_checks(
        pd.DataFrame(columns=["rebalance_date", "ticker", "target_weight",
                              "trained_through"]),
        [], max_stock_weight=0.10, max_holdings=20,
        min_history_sessions=250, require_liquidity_flag=True)
    assert not next(r for r in report.results
                    if r.name == "non_empty_portfolios").passed


# ------------------------------------------------------- e2e orchestrator
class _S:
    parquet_compression = "zstd"
    parquet_compression_level = 3


def test_build_portfolios_end_to_end(tmp_path, monkeypatch) -> None:
    from asx_quant.observability.runs import RunRegistry
    from asx_quant.storage.paths import DataLake

    cfg = {
        "portfolio": {
            "portfolio_id": "test_v1",
            "selection": {
                "n_holdings": 5,
                "signal": "predicted_return_3d",
                "prediction_source": {"dataset": "predictions",
                                      "model": "hist_gbt_reg"},
                "require_liquidity_flag": True,
            },
            "rebalancing": {"frequency": "weekly", "weekday": "monday"},
            "weights": {"method": "equal_weight", "max_stock_weight": 0.10,
                        "prefer_cash_over_violation": True},
            "constraints": {"min_history_sessions": 100},
        },
    }
    monkeypatch.setattr("asx_quant.portfolio.build.load_yaml",
                        lambda name: cfg)

    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    registry = RunRegistry.start(lake, config_version="test")

    FEATURES.to_parquet(lake.gold_dataset_path("features"), index=False)
    PREDICTIONS.to_parquet(lake.gold_dataset_path("predictions"), index=False)

    from asx_quant.portfolio.build import build_portfolios

    result = build_portfolios(lake, _S(), registry)
    assert result.n_rebalances > 40  # weekly over ~1 year
    assert result.n_rows > 0

    portfolios = pd.read_parquet(result.portfolios_path)
    assert not portfolios.duplicated(
        subset=["rebalance_date", "ticker"]).any()
    assert (portfolios["target_weight"] >= 0).all()
    assert (portfolios["target_weight"] <= 0.10 + 1e-9).all()
    assert (portfolios["model_name"] == "hist_gbt_reg").all()
    assert (portfolios["feature_set_version"] == "v0.1").all()
    # every selected name was liquid at the rebalance date
    liquid_at = FEATURES[FEATURES["liquid_flag"] == 1.0]
    pairs = set(zip(liquid_at["ticker"], liquid_at["date"], strict=True))
    assert all((r.ticker, r.rebalance_date) in pairs
               for r in portfolios.itertuples())
    # look-ahead guard on the written dataset
    assert (portfolios["trained_through"] < portfolios["rebalance_date"]).all()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["portfolio_id"] == "test_v1"
    assert summary["weight_method"] == "equal_weight"
    assert summary["n_rebalances"] == result.n_rebalances
    # gross exposure including cash is 1 everywhere
    for rec in summary["per_rebalance"]:
        day = portfolios[portfolios["rebalance_date"]
                         == pd.Timestamp(rec["rebalance_date"])]
        assert day["target_weight"].sum() + rec["cash_weight"] == \
            pytest.approx(1.0, abs=1e-6)

    quality = json.loads(
        (tmp_path / "data/reports/portfolio_quality.json").read_text(
            encoding="utf-8"))
    assert quality["status"] == "pass"


def test_build_portfolios_rejects_missing_predictions(tmp_path,
                                                      monkeypatch) -> None:
    from asx_quant.observability.runs import RunRegistry
    from asx_quant.storage.paths import DataLake

    cfg = json.loads(json.dumps({
        "portfolio": {
            "selection": {"n_holdings": 5, "signal": "predicted_return_3d",
                          "prediction_source": {"dataset": "predictions",
                                                "model": "ghost_model"},
                          "require_liquidity_flag": True},
            "rebalancing": {"frequency": "weekly", "weekday": "monday"},
            "weights": {"method": "equal_weight",
                        "max_stock_weight": 0.10},
            "constraints": {"min_history_sessions": 100},
        },
    }))
    monkeypatch.setattr("asx_quant.portfolio.build.load_yaml",
                        lambda name: cfg)

    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    registry = RunRegistry.start(lake, config_version="test")
    FEATURES.to_parquet(lake.gold_dataset_path("features"), index=False)
    PREDICTIONS.to_parquet(lake.gold_dataset_path("predictions"), index=False)

    from asx_quant.portfolio.build import build_portfolios

    with pytest.raises(ValueError, match="no predictions"):
        build_portfolios(lake, _S(), registry)


def test_portfolio_config_file_is_valid() -> None:
    """The real portfolio.yaml parses and is internally consistent."""
    from asx_quant.config.loader import load_yaml

    cfg = load_yaml("portfolio")["portfolio"]
    assert cfg["direction"] == "long_only"
    selection = cfg["selection"]
    assert selection["n_holdings"] == 20
    assert selection["signal"].startswith(("predicted_return", "prob"))
    assert selection["prediction_source"]["model"]
    weights = cfg["weights"]
    assert weights["max_stock_weight"] <= 1.0
    assert weights["method"] in ("equal_weight", "volatility_adjusted")
    assert cfg["rebalancing"]["frequency"] in ("weekly", "fortnightly",
                                               "monthly")
    assert cfg["constraints"]["min_history_sessions"] > 0
