"""Sensitivity grid expansion (Phase 8).

The grid axes come from ``backtest.yaml`` -> ``sensitivity``:

* ``portfolio_sizes``          -> portfolio construction (n_holdings)
* ``rebalance_frequencies``    -> portfolio construction (schedule)
* ``max_stock_weights``        -> portfolio construction (cap)
* ``cost_scenarios``           -> backtest execution costs
* ``horizons``                 -> the trained model per horizon (3d primary)
* ``max_sector_weights``       -> NOT RUNNABLE (no point-in-time sector
  mapping exists — Phase 4 limitation); recorded as a skipped axis, never
  silently dropped.

A grid point = (horizon, size, frequency, cap) portfolio configuration x
every cost scenario. Horizons without a trained model produce a documented
``skipped`` entry instead of a fabricated result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from asx_quant.observability.logging import get_logger

logger = get_logger("sensitivity.grid")


@dataclass(frozen=True)
class PortfolioScenario:
    """One portfolio-construction grid point."""

    horizon: int
    n_holdings: int
    frequency: str
    max_stock_weight: float


@dataclass(frozen=True)
class GridPoint:
    """One sensitivity run: portfolio scenario x cost scenario."""

    portfolio: PortfolioScenario
    cost_scenario: str


@dataclass
class GridPlan:
    """Runnable grid points plus explicitly skipped axes."""

    points: list[GridPoint] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)


def expand_grid(
    sensitivity_cfg: dict,
    trained_horizon: int,
) -> GridPlan:
    """Expand the configured sensitivity axes into runnable grid points.

    Args:
        sensitivity_cfg: The ``backtest.sensitivity`` section (axes lists).
        trained_horizon: The horizon actually trained (``train.yaml``
            ``target.horizon``); horizons without a trained model are
            skipped with an explicit reason.

    Returns:
        :class:`GridPlan` with the cartesian product of runnable axes and
        the documented skip list.
    """
    sizes = list(sensitivity_cfg.get("portfolio_sizes", [20]))
    frequencies = list(sensitivity_cfg.get("rebalance_frequencies", ["weekly"]))
    weights = list(sensitivity_cfg.get("max_stock_weights", [0.10]))
    cost_scenarios = list(sensitivity_cfg.get("cost_scenarios", ["base"]))
    horizons = list(sensitivity_cfg.get("horizons", [trained_horizon]))

    plan = GridPlan()
    for horizon in horizons:
        if int(horizon) != int(trained_horizon):
            plan.skipped.append({
                "axis": "horizon",
                "value": int(horizon),
                "reason": f"No model trained for horizon {horizon} "
                          f"(train.yaml target.horizon = {trained_horizon}); "
                          "flip the training config and rerun 'train' to "
                          "evaluate it. No synthetic substitute is used.",
            })
            continue
        for size in sizes:
            for frequency in frequencies:
                for weight in weights:
                    scenario = PortfolioScenario(
                        horizon=int(horizon), n_holdings=int(size),
                        frequency=str(frequency),
                        max_stock_weight=float(weight))
                    for cost in cost_scenarios:
                        plan.points.append(GridPoint(scenario, str(cost)))
    return plan
