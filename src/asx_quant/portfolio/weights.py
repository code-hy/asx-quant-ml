"""Portfolio weighting with long-only constraints.

Weight methods (``portfolio.weights.method``):

* ``equal_weight`` — 1/N among selected names.
* ``volatility_adjusted`` — inverse 20-session volatility, normalised to
  a gross exposure of 1, clamped to ``[min_weight, max_weight]`` with
  below-floor names dropped once (documented simplification: diagonal
  inverse-vol, no covariance estimate).

Constraint handling follows ``prefer_cash_over_violation``:

* ``true`` (default) — weights are clamped at ``max_stock_weight`` and the
  residual stays in CASH. The portfolio may be under-invested rather than
  violate a rule.
* ``false`` — capped weight is redistributed iteratively among uncapped
  names (water-filling) so gross exposure reaches 1.0 when possible.

Sector weighting (``max_sector_weight``) is NOT implemented: the platform
has no free point-in-time sector mapping (documented since Phase 4). The
build reports this as an explicit skipped rule — it is never silently
ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asx_quant.observability.logging import get_logger

logger = get_logger("portfolio.weights")

_TOLERANCE = 1e-9


def equal_weights(n_holdings: int) -> pd.Series:
    """Raw equal weights (pre-constraint): 1/N each."""
    if n_holdings <= 0:
        return pd.Series(dtype="float64")
    return pd.Series(1.0 / n_holdings,
                     index=pd.RangeIndex(n_holdings), name="target_weight")


def inverse_volatility_weights(volatility: pd.Series) -> pd.Series:
    """Raw inverse-volatility weights, normalised to sum 1."""
    vol = volatility.astype("float64").replace(0.0, np.nan)
    inv = 1.0 / vol
    valid = inv.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        raise ValueError(
            "No usable volatility values for volatility_adjusted weights."
        )
    return (valid / valid.sum()).rename("target_weight")


def apply_weight_cap(
    raw: pd.Series,
    max_stock_weight: float,
    prefer_cash: bool = True,
) -> tuple[pd.Series, float]:
    """Clamp weights at ``max_stock_weight``.

    Returns:
        ``(weights, cash_weight)``. With ``prefer_cash`` the residual
        ``1 - sum(weights)`` is returned as cash (may be 0). Without it,
        capped weight is redistributed water-filling style among uncapped
        names until no name exceeds the cap (residual then reflects only
        rounding/NaN slack).
    """
    if raw.empty:
        return pd.Series(dtype="float64", name="target_weight"), 1.0
    w = raw.astype("float64").clip(lower=0.0)

    if not prefer_cash:
        for _ in range(len(w) + 1):
            over = w > max_stock_weight + _TOLERANCE
            if not over.any():
                break
            excess = float((w[over] - max_stock_weight).sum())
            w[over] = max_stock_weight
            below = w < max_stock_weight - _TOLERANCE
            if not below.any() or excess <= _TOLERANCE:
                break
            # Distribute proportionally to HEADROOM, clamped so no
            # recipient crosses the cap (no oscillation; capped names are
            # never re-inflated).
            headroom = (max_stock_weight - w[below]).to_numpy()
            give = np.minimum(headroom, excess * headroom / headroom.sum())
            w[below] = w[below] + give
            excess -= float(give.sum())
            if excess <= _TOLERANCE:
                break
    else:
        w = w.clip(upper=max_stock_weight)

    cash = max(0.0, 1.0 - float(w.sum()))
    return w.rename("target_weight"), cash


def enforce_min_weight(
    weights: pd.Series,
    min_weight: float,
) -> pd.Series:
    """Drop names below ``min_weight`` and renormalise the remainder once."""
    if weights.empty or min_weight <= 0:
        return weights
    keep = weights[weights >= min_weight]
    if keep.empty:
        return weights  # never return an empty portfolio silently
    return (keep / keep.sum()).rename("target_weight")


def build_weights(
    ranked: pd.DataFrame,
    method: str,
    weights_cfg: dict,
) -> tuple[pd.DataFrame, float]:
    """Raw -> constrained target weights for one rebalance.

    Args:
        ranked: Frame from :func:`asx_quant.portfolio.selection.rank_candidates`
            (needs ``rank``; and ``volatility_20d`` for vol-adjusted).
        method: equal_weight | volatility_adjusted.
        weights_cfg: The ``portfolio.weights`` config section.

    Returns:
        ``(frame, cash_weight)`` where frame carries ``target_weight`` per
        selected ticker (input columns preserved).
    """
    cap = float(weights_cfg.get("max_stock_weight", 1.0))
    prefer_cash = bool(weights_cfg.get("prefer_cash_over_violation", True))

    if ranked.empty:
        return ranked.assign(target_weight=pd.Series(dtype="float64")), 1.0

    if method == "equal_weight":
        # Raw equal weight across the SELECTED names; the max-stock cap
        # (not n_holdings) decides the final per-name exposure, so a small
        # eligible set is clamped and the residual stays in cash.
        raw = pd.Series(
            1.0 / len(ranked),
            index=ranked.index, name="target_weight",
        )
    elif method == "volatility_adjusted":
        if "volatility_20d" not in ranked.columns:
            raise ValueError(
                "volatility_adjusted weights need the volatility_20d "
                "feature column joined into the ranked frame."
            )
        raw = inverse_volatility_weights(ranked["volatility_20d"])
        min_weight = float(
            weights_cfg.get("vol_adjust", {}).get("min_weight", 0.0))
        raw = enforce_min_weight(raw, min_weight)
        raw = raw.reindex(ranked.index)
    else:
        raise ValueError(
            f"Unknown weight method '{method}'. Valid: equal_weight, "
            "volatility_adjusted."
        )

    weights, cash = apply_weight_cap(raw, cap, prefer_cash=prefer_cash)
    out = ranked.copy()
    out["target_weight"] = weights
    return out, cash


def turnover_vs_previous(
    current: pd.DataFrame,
    previous: pd.DataFrame,
) -> float:
    """One-sided turnover: 0.5 * sum of absolute weight changes.

    Both frames are per-rebalance weight tables (ticker, target_weight).
    """
    if current.empty:
        return 0.0 if previous.empty else 1.0
    if previous.empty:
        return float(current["target_weight"].sum()) / 2.0
    merged = current[["ticker", "target_weight"]].merge(
        previous[["ticker", "target_weight"]],
        on="ticker", how="outer", suffixes=("_new", "_old"),
    ).fillna(0.0)
    return float(0.5 * (merged["target_weight_new"]
                        - merged["target_weight_old"]).abs().sum())
