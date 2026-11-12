"""Command-line interface for the asx-quant platform.

Command set (Phase 1 implements ``health`` and ``initialise``; later phases
implement the remaining stages — each prints a clear "not yet implemented"
message until then, so the contract is visible from day one):

    python -m asx_quant.cli health                 foundation self-check
    python -m asx_quant.cli initialise             create dirs, DuckDB, registry
    python -m asx_quant.cli ingest    --mode ...   (Phase 2+)
    python -m asx_quant.cli validate               (Phase 2+)
    python -m asx_quant.cli transform              (Phase 2+)
    python -m asx_quant.cli features               (Phase 4+)
    python -m asx_quant.cli train                  (Phase 5+)
    python -m asx_quant.cli predict                (Phase 6+)
    python -m asx_quant.cli portfolio              (Phase 7+)
    python -m asx_quant.cli backtest               (Phase 8+)
    python -m asx_quant.cli report                 (Phase 9+)
    python -m asx_quant.cli update    --mode ...   (Phase 9+)
"""

from __future__ import annotations

import argparse
import sys

from asx_quant import __version__
from asx_quant.config.loader import Settings, load_settings
from asx_quant.exceptions import AsxQuantError
from asx_quant.observability.logging import configure_logging, get_logger
from asx_quant.observability.runs import RunRegistry
from asx_quant.pipeline.health import run_health_check
from asx_quant.storage.duckdb import get_warehouse
from asx_quant.storage.paths import DataLake

logger = get_logger("cli")

#: Commands delivered in later phases; listed so users can see the contract.
_PHASED_COMMANDS = (
    "ingest", "validate", "transform", "features",
    "train", "predict", "portfolio", "backtest", "report", "update",
)


def _add_mode_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mode",
        choices=["historical", "incremental"],
        default="historical",
        help="Pipeline operating mode.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asx_quant",
        description=(
            "ASX Quantitative Intelligence & ML Portfolio Platform "
            "(research only — not financial advice)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"asx-quant {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Run the foundation health check.")
    subparsers.add_parser("initialise", help="Create directory skeleton, DuckDB warehouse and run registry.")

    for name in _PHASED_COMMANDS:
        sub = subparsers.add_parser(name, help=f"[Phase 2+] Not yet implemented: {name}")
        _add_mode_argument(sub)

    return parser


# ------------------------------------------------------------------- commands
def _cmd_health(settings: Settings) -> int:
    report = run_health_check(settings)
    print(report.summary())
    return 0 if report.passed else 1


def _cmd_initialise(settings: Settings) -> int:
    configure_logging()
    lake = DataLake.from_settings(settings)
    lake.ensure_all()
    logger.info("Directory skeleton ensured under '%s'", lake.root)

    warehouse = get_warehouse(settings)
    try:
        warehouse.connect()
        registry_rows = warehouse.query(
            "SELECT COUNT(*) AS n FROM meta.dataset_registry"
        ).iloc[0]["n"]
    finally:
        warehouse.close()

    registry = RunRegistry.start(lake, config_version=f"v{settings.project_version}")
    record = registry.begin_stage("initialise")
    registry.finish_stage(
        record,
        record_count=int(registry_rows),
        run_id=registry.run_id,
    )

    print(f"Initialised data lake at '{lake.root}'.")
    print(f"DuckDB warehouse ready at '{settings.duckdb_path}' (registered views: {int(registry_rows)}).")
    print(f"Run registry initialised with run '{registry.run_id}'.")
    return 0


def _cmd_not_implemented(name: str) -> int:
    print(
        f"Command '{name}' is planned for a later phase and is not implemented yet.\n"
        "See the phase roadmap in README.md — do not assume this command has run."
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "health":
        try:
            return _cmd_health(load_settings())
        except AsxQuantError as exc:
            print(f"Health check could not start: {exc}", file=sys.stderr)
            return 1

    try:
        settings = load_settings()
    except AsxQuantError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    if args.command == "initialise":
        return _cmd_initialise(settings)

    return _cmd_not_implemented(args.command)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
