"""Command-line interface for the asx-quant platform.

Command set:

    python -m asx_quant.cli health                 foundation self-check
    python -m asx_quant.cli initialise             create dirs, DuckDB, registry
    python -m asx_quant.cli ingest    --mode historical   universe + prices → Bronze
    python -m asx_quant.cli validate               data-quality checks on Bronze
    python -m asx_quant.cli transform              Bronze → Silver + coverage
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
    "features", "train", "predict", "portfolio", "backtest", "report", "update",
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
            "(research only - not financial advice)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"asx-quant {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Run the foundation health check.")
    subparsers.add_parser("initialise", help="Create directory skeleton, DuckDB warehouse and run registry.")

    ingest = subparsers.add_parser("ingest", help="Ingest universe + historical market data into Bronze.")
    _add_mode_argument(ingest)
    subparsers.add_parser("validate", help="Run data-quality checks on Bronze data.")
    subparsers.add_parser("transform", help="Transform Bronze into Silver datasets + coverage report.")

    for name in ("features", "train", "predict", "portfolio",
                 "backtest", "report", "update"):
        sub = subparsers.add_parser(name, help=f"[Phase 4+] Not yet implemented: {name}")
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
        "See the phase roadmap in README.md - do not assume this command has run."
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
    if args.command == "ingest":
        from asx_quant.pipeline.stages import run_ingest

        summary = run_ingest(settings, mode=args.mode)
        print(f"Ingestion complete (run {summary['run_id']}):")
        print(f"  universe source: {summary['universe_source']} ({summary['universe_size']} securities)")
        print(f"  tickers with data: {summary['tickers_written']}")
        print(f"  records written:  {summary['records_written']}")
        if summary["failed_tickers"]:
            print(f"  FAILED tickers:   {', '.join(summary['failed_tickers'])}")
        macro = summary["macro"]
        if macro.get("status") == "ok":
            print(f"  macro (RBA):      {macro['records']:,} observations, "
                  f"{macro['series']} series across {len(macro['tables'])} tables")
        else:
            print(f"  macro (RBA):      {macro.get('status', 'skipped')}")
        asic = summary["asic"]
        if asic.get("status") in ("ok", "partial"):
            print(f"  ASIC short pos:   {asic['records']:,} records, "
                  f"{asic['securities']} securities, years "
                  f"{asic['years'][0]}-{asic['years'][-1]}"
                  + (f" (FAILED: {', '.join(asic['failed_reports'])})"
                     if asic.get("failed_reports") else ""))
        else:
            print(f"  ASIC short pos:   {asic.get('status', 'skipped')}")
        return 0
    if args.command == "validate":
        from asx_quant.pipeline.stages import run_validate

        report = run_validate(settings)
        status = report.to_dict()["status"]
        print(f"Data-quality validation: {status.upper()} "
              f"({report.warn_count()} warning rules failed)")
        for rule in report.results:
            marker = "PASS" if rule.passed else "FAIL"
            print(f"  [{rule.severity[0].upper()}] {marker}  {rule.name}"
                  + (f" - {rule.failures} failures" if not rule.passed else ""))
        return 0 if report.passed else 1
    if args.command == "transform":
        from asx_quant.pipeline.stages import run_transform

        summary = run_transform(settings)
        print(f"Silver transform complete (run {summary['run_id']}):")
        print(f"  records: {summary['records']:,} across {summary['securities']} securities")
        print(f"  calendar sessions: {summary['calendar_sessions']}")
        print(f"  date range: {summary['first_date']} -> {summary['last_date']}")
        print(f"  quality status: {summary['quality_status']}")
        print(f"  macro silver: {summary['macro_status']}")
        print(f"  short-position silver: {summary['short_positions_status']}")
        return 0

    return _cmd_not_implemented(args.command)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
