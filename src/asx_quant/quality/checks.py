"""Automated data-quality rule engine.

Every check is measured and reported — never silently discarded. Checks are
grouped by severity:

* ``error``   — structurally invalid data; fails the pipeline stage.
* ``warn``    — data is usable but coverage/anomalies must be documented.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from asx_quant.exceptions import DataQualityError
from asx_quant.observability.logging import get_logger

logger = get_logger("quality")


@dataclass
class RuleResult:
    """Outcome of one quality rule."""

    name: str
    severity: str  # error | warn
    passed: bool
    failures: int = 0
    sample: list[str] = field(default_factory=list)
    detail: str = ""


@dataclass
class QualityReport:
    """Aggregated report of all rule results for one dataset."""

    dataset: str
    results: list[RuleResult] = field(default_factory=list)

    @property
    def errors(self) -> list[RuleResult]:
        return [r for r in self.results if r.severity == "error"]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.errors)

    def warn_count(self) -> int:
        return sum(1 for r in self.results if r.severity == "warn" and not r.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "status": "pass" if self.passed else "fail",
            "warn_failures": self.warn_count(),
            "rules": [asdict(r) for r in self.results],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str),
                        encoding="utf-8")
        return path

    def assert_passed(self) -> None:
        """Raise :class:`DataQualityError` if any error-severity rule failed."""
        failed = [r for r in self.errors if not r.passed]
        if failed:
            names = ", ".join(f.name for f in failed)
            raise DataQualityError(
                f"Data-quality check failed for '{self.dataset}': {names}. "
                "See the saved quality report for samples."
            )


def _sample_failures(mask: pd.Series, keys: pd.DataFrame, limit: int = 10) -> list[str]:
    """Build readable failure samples from a boolean failure mask."""
    if not mask.any():
        return []
    idx = mask[mask].index[:limit]
    return [f"{keys.loc[i, 'ticker']}@{keys.loc[i, 'date']}" for i in idx]


def run_price_quality_checks(
    prices: pd.DataFrame,
    dataset_name: str = "bronze_prices",
    quality_cfg: dict[str, Any] | None = None,
) -> QualityReport:
    """Run all price quality rules on a standardised price frame.

    Expected columns: ``ticker, date, open, high, low, close, adj_close,
    volume`` (post :func:`asx_quant.transformations.standardise_bronze`).
    Thresholds come from ``config/settings.yaml`` (``quality`` section).
    """
    cfg = quality_cfg or {}
    price_min = float(cfg.get("price_bounds", {}).get("min", 0.001))
    price_max = float(cfg.get("price_bounds", {}).get("max", 100000.0))
    extreme_threshold = float(cfg.get("extreme_return_threshold", 0.5))
    report = QualityReport(dataset=dataset_name)
    keys = prices[["ticker", "date"]] if not prices.empty else pd.DataFrame(
        {"ticker": [], "date": []}
    )

    def _rule(name: str, severity: str, passed: bool, failures: int = 0,
              sample: list[str] | None = None, detail: str = "") -> None:
        report.results.append(RuleResult(
            name=name, severity=severity, passed=passed,
            failures=int(failures), sample=sample or [], detail=detail,
        ))

    # ---- error-severity structural rules ---------------------------------
    dup_mask = prices.duplicated(subset=["ticker", "date"], keep=False) \
        if not prices.empty else pd.Series(dtype=bool)
    _rule("no_duplicate_security_date", "error", not dup_mask.any(),
          dup_mask.sum(), _sample_failures(dup_mask, keys),
          "Duplicate (ticker, date) records must be resolved before Silver.")

    if prices.empty:
        _rule("non_empty_dataset", "error", False, 0,
              detail="Price dataset is empty.")
    else:
        _rule("non_empty_dataset", "error", True, 0)

    if not prices.empty:
        bad_price = (
            prices[["open", "high", "low", "close", "adj_close"]]
            .le(price_min).any(axis=1)
            | prices[["open", "high", "low", "close", "adj_close"]]
            .ge(price_max).any(axis=1)
        )
        bad_price &= prices[["open", "high", "low", "close"]].notna().all(axis=1)
        _rule("prices_positive_and_bounded", "error", not bad_price.any(),
              bad_price.sum(), _sample_failures(bad_price, keys))

        bad_high_low = prices["high"] < prices["low"]
        _rule("high_not_below_low", "error", not bad_high_low.any(),
              bad_high_low.sum(), _sample_failures(bad_high_low, keys))

        bad_volume = prices["volume"] < 0
        _rule("volume_non_negative", "error", not bad_volume.any(),
              bad_volume.sum(), _sample_failures(bad_volume, keys))

        bad_dates = prices["date"].isna()
        _rule("trading_dates_valid", "error", not bad_dates.any(),
              bad_dates.sum(), _sample_failures(bad_dates, keys))

        # ---- warn-severity coverage/anomaly rules -------------------------
        ret = prices.groupby("ticker")["adj_close"].pct_change()
        extreme = ret.abs() >= extreme_threshold
        extreme &= prices["adj_close"].notna()
        _rule("extreme_single_session_returns", "warn", not extreme.any(),
              extreme.sum(), _sample_failures(extreme, keys),
              "Possible corporate actions or data errors; flagged for review.")

        # A session is a market day when a majority of active tickers traded;
        # whole-market holidays (where nobody traded) are excluded. A union
        # calendar cannot detect absences — this one can.
        n_tickers = prices["ticker"].nunique()
        per_date = prices.groupby("date")["ticker"].nunique()
        calendar = sorted(per_date[per_date >= max(1, 0.5 * n_tickers)].index)
        missing_total = 0
        worst: list[str] = []
        for ticker, group in prices.groupby("ticker"):
            first, last = group["date"].min(), group["date"].max()
            expected = [d for d in calendar if first <= d <= last]
            actual = set(group["date"])
            missing = len(set(expected) - actual)
            missing_total += missing
            if missing:
                worst.append(f"{ticker}:{missing}")
        _rule("missing_sessions_measured", "warn", missing_total == 0,
              missing_total, sorted(worst, key=lambda s: -int(s.split(":")[1]))[:10],
              "Missing sessions vs union calendar within each ticker's history "
              "window; measured and reported, never silently filled.")

        missing_prices = prices[["open", "high", "low", "close", "volume"]].isna().any(axis=1)
        _rule("missing_price_fields", "warn", not missing_prices.any(),
              missing_prices.sum(), _sample_failures(missing_prices, keys))
    return report


# ------------------------------------------------------------------- macro
def run_macro_quality_checks(
    macro: pd.DataFrame,
    dataset_name: str = "macro_observations",
) -> QualityReport:
    """Quality rules for macro observations (point-in-time leakage guard).

    Expected columns: ``series_name, observation_date, value, publication_date,
    publication_date_method``. The publication rule is the highest-priority
    check in the platform: no model may ever see a macro observation before
    its publication date, so a violation here is an ERROR, not a warning.
    """
    report = QualityReport(dataset=dataset_name)

    def _rule(name: str, severity: str, passed: bool, failures: int = 0,
              sample: list[str] | None = None, detail: str = "") -> None:
        report.results.append(RuleResult(
            name=name, severity=severity, passed=passed,
            failures=int(failures), sample=sample or [], detail=detail,
        ))

    if macro.empty:
        _rule("non_empty_dataset", "error", False, 0,
              detail="Macro observation dataset is empty.")
        return report
    _rule("non_empty_dataset", "error", True, 0)

    dup = macro.duplicated(subset=["series_name", "observation_date"],
                           keep=False)
    _rule("no_duplicate_series_observation", "error", not dup.any(),
          dup.sum(),
          [f"{macro.loc[i, 'series_name']}@{macro.loc[i, 'observation_date']}"
           for i in dup[dup].index[:10]])

    bad_dates = macro["observation_date"].isna() | macro["publication_date"].isna()
    _rule("dates_present", "error", not bad_dates.any(), bad_dates.sum())

    # THE leakage guard: publication must never precede observation.
    leaky = macro["publication_date"] < macro["observation_date"]
    _rule("publication_not_before_observation", "error", not leaky.any(),
          leaky.sum(),
          [f"{macro.loc[i, 'series_name']}: pub="
           f"{macro.loc[i, 'publication_date'].date()} "
           f"obs={macro.loc[i, 'observation_date'].date()}"
           for i in leaky[leaky].index[:10]],
          "Any macro observation published before its own observation date "
          "would allow look-ahead joins.")

    bad_values = macro["value"].isna() | ~pd.to_numeric(
        macro["value"], errors="coerce"
    ).notna()
    _rule("values_numeric", "error", not bad_values.any(), bad_values.sum())

    unflagged = ~macro["publication_date_method"].isin([
        "same_day_publication", "estimated_publication_lag",
        "announcement_date", "estimated_from_report_calendar",
    ])
    _rule("publication_method_flagged", "warn", not unflagged.any(),
          unflagged.sum(),
          detail="Every observation must state how its publication date was "
                 "determined (exact or estimated).")

    # Sparse-series coverage warning: a series with very few observations
    # over its own history may indicate a truncated source download.
    counts = macro.groupby("series_name")["observation_date"].count()
    sparse = counts[counts < 20]
    _rule("series_coverage", "warn", sparse.empty, int(sparse.sum()),
          [f"{name}:{count}" for name, count in sparse.items()],
          "Series with fewer than 20 observations in the ingested window.")
    return report


def run_short_position_quality_checks(
    positions: pd.DataFrame,
    dataset_name: str = "short_positions",
) -> QualityReport:
    """Quality rules for ASIC short positions (leakage guard + bounds)."""
    report = QualityReport(dataset=dataset_name)

    def _rule(name: str, severity: str, passed: bool, failures: int = 0,
              sample: list[str] | None = None, detail: str = "") -> None:
        report.results.append(RuleResult(
            name=name, severity=severity, passed=passed,
            failures=int(failures), sample=sample or [], detail=detail,
        ))

    if positions.empty:
        _rule("non_empty_dataset", "error", False, 0,
              detail="Short-position dataset is empty.")
        return report
    _rule("non_empty_dataset", "error", True, 0)

    dup = positions.duplicated(subset=["ticker", "trade_date"], keep=False)
    _rule("no_duplicate_ticker_trade_date", "error", not dup.any(),
          dup.sum(),
          [f"{positions.loc[i, 'ticker']}@{positions.loc[i, 'trade_date']}"
           for i in dup[dup].index[:10]])

    leaky = positions["publication_date"] < positions["trade_date"]
    _rule("publication_not_before_trade_date", "error", not leaky.any(),
          leaky.sum(),
          [f"{positions.loc[i, 'ticker']}: pub="
           f"{positions.loc[i, 'publication_date'].date()} "
           f"trade={positions.loc[i, 'trade_date'].date()}"
           for i in leaky[leaky].index[:10]],
          "Short positions must not be usable before their report date.")

    pct = positions["pct_of_product_in_issue"].dropna()
    negative = pct < 0
    _rule("pct_non_negative", "error", not negative.any(), negative.sum(),
          detail="A negative percent of product in issue is structurally "
                 "invalid.")
    # >100% occurs in genuine ASIC data for exchange-traded products (the
    # 'product in issue' denominator lags creations/redemptions for ETFs and
    # similar). Measured and disclosed, never discarded — see
    # docs/data_sources.md section 5.
    over_100 = pct > 100
    _rule("pct_within_bounds", "warn", not over_100.any(), over_100.sum(),
          detail="Percent of product in issue above 100 occurs in real ASIC "
                 "reports for ETPs; rows are kept and flagged, not dropped.")

    bad_pos = positions["reported_short_positions"].dropna() < 0
    _rule("positions_non_negative", "error", not bad_pos.any(),
          bad_pos.sum())
    return report
