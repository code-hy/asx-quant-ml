"""Phase 3 tests: RBA/ASIC ingestion, publication-date logic, quality rules.

All tests are offline — every network touch is replaced by injectable
``http_get`` functions returning fixture bytes.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from asx_quant.exceptions import DataQualityError, IngestionError
from asx_quant.ingestion.asic import (
    ingest_asic_reports,
    parse_ytd_pivot,
    pick_ytd_reports,
    ytd_report_url,
)
from asx_quant.ingestion.rba import (
    assign_publication_dates,
    build_macro_observations,
    load_series_specs,
    parse_cash_rate_html,
    parse_rba_csv,
)
from asx_quant.observability.runs import RunRegistry
from asx_quant.quality.checks import (
    run_macro_quality_checks,
    run_short_position_quality_checks,
)
from asx_quant.storage.paths import DataLake

# ---------------------------------------------------------------- fixtures
RBA_CSV = """F2 INTEREST RATES AND YIELDS - COMMONWEALTH GOVERNMENT SECURITIES
Title,Some Series,Another Series
Description,annual yields,annual yields
Frequency,Daily,Daily
Type,Original,Original
Units,Per cent,Per cent

Source,RBA,RBA
Publication date,02-Nov-2026,02-Nov-2026
Series ID,FCMYGBAG2D,FCMYGBAG10D
20-May-2013,3.10,4.20
21-May-2013,3.11,4.21
30/06/2013,3.15,4.25
31/12/2026,,          <- forward-looking placeholder row (no values)
"""

CASH_RATE_HTML = """<html><body>
<table>
<tr><th>Effective Date</th><th>Change% points</th><th>Cash rate target %</th></tr>
<tr><td>4 Nov 2026</td><td>+0.25</td><td>4.85</td></tr>
<tr><td>30 Sep 2026</td><td>+0.25</td><td>4.60</td></tr>
<tr><td>12 Aug 2026</td><td>0.00</td><td>4.35</td></tr>
<tr><td>20 May 2026</td><td>-0.25</td><td>4.35</td></tr>
<tr><td>19 Feb 2026</td><td>-0.25</td><td>4.60</td></tr>
</table></body></html>
"""

YTD_CSV = (
    ',Trade Date,6/11/2026,6/11/2026,5/11/2026,5/11/2026\n'
    'Product,Product Code,Reported Short Positions,'
    '% of Total Product in Issue Reported as Short Positions,'
    'Reported Short Positions,'
    '% of Total Product in Issue Reported as Short Positions\n'
    '3D ENERGI LTD ORDINARY,TDO,181029,0.03,-,-\n'
    '3P LEARNING LTD,3PL,-,-,11749,0.00\n'
    '"BIG CO, INC ORDINARY",BIG,555,0.10,444,0.09\n'
)

INDEX_JSON = [
    {"date": 20261106, "version": "001"},
    {"date": 20261105, "version": "002"},
    {"date": 20150701, "version": "001"},
    {"date": 20150630, "version": "001"},
]


def _fake_http_dispatch(url: str) -> bytes:
    """Route fixture bytes by URL: cash-rate page, index JSON, RBA CSV, YTD."""
    if "cash-rate" in url:
        return CASH_RATE_HTML.encode()
    if url.endswith("short-selling-data.json"):
        return json.dumps(INDEX_JSON).encode()
    if "/short-selling/RR" in url:
        return YTD_CSV.encode()
    return RBA_CSV.encode()  # any RBA statistical CSV


def _macro_frame(**overrides) -> pd.DataFrame:
    base = pd.DataFrame({
        "series_name": ["abs_cpi"] * 3,
        "source_series_id": ["GCPIAGYP"] * 3,
        "observation_date": pd.to_datetime(
            ["2024-03-31", "2024-06-30", "2024-09-30"]),
        "value": [3.6, 3.8, 2.8],
        "publication_date": pd.to_datetime(
            ["2024-05-05", "2024-08-04", "2024-11-04"]),
        "publication_date_method": ["estimated_publication_lag"] * 3,
        "potentially_revised": [True] * 3,
        "frequency": ["quarterly"] * 3,
        "unit": ["per_cent_yoy"] * 3,
        "source": ["abs"] * 3,
        "source_table": ["G1"] * 3,
    })
    for key, value in overrides.items():
        base[key] = value
    return base


# ------------------------------------------------------------- RBA parsing
def test_parse_rba_csv_mixed_dates_and_placeholder_rows() -> None:
    long = parse_rba_csv(RBA_CSV.encode())
    assert set(long["series_id"]) == {"FCMYGBAG2D", "FCMYGBAG10D"}
    # placeholder row (31/12/2026, empty values) must be dropped
    assert len(long) == 6
    assert long["observation_date"].max() == pd.Timestamp("2013-06-30")
    row = long[long["observation_date"] == pd.Timestamp("2013-06-30")]
    assert row[row["series_id"] == "FCMYGBAG2D"]["value"].iloc[0] == 3.15


def test_parse_rba_csv_rejects_unexpected_layout() -> None:
    with pytest.raises(IngestionError, match="Series ID"):
        parse_rba_csv(b"header,only\n1,2\n")


def test_publication_dates_daily_same_day() -> None:
    obs = pd.DataFrame({"observation_date": pd.to_datetime(["2024-01-02"])})
    out = assign_publication_dates(obs, "daily", lag_days=0)
    assert out["publication_date"].iloc[0] == out["observation_date"].iloc[0]
    assert out["publication_date_method"].iloc[0] == "same_day_publication"
    assert not out["potentially_revised"].iloc[0]


def test_publication_dates_monthly_lag_flagged() -> None:
    obs = pd.DataFrame({"observation_date": pd.to_datetime(["2024-05-31"])})
    out = assign_publication_dates(obs, "monthly", lag_days=15)
    assert out["publication_date"].iloc[0] == pd.Timestamp("2024-06-15")
    assert out["publication_date_method"].iloc[0] == "estimated_publication_lag"
    assert out["potentially_revised"].iloc[0]  # monthly series are revisable


def test_parse_cash_rate_html() -> None:
    parsed = parse_cash_rate_html(CASH_RATE_HTML.encode(), min_rows=2)
    assert len(parsed) == 5
    assert parsed["effective_date"].iloc[0] == pd.Timestamp("2026-11-04")
    assert parsed["cash_rate_target"].iloc[0] == 4.85


def test_build_macro_observations_offline() -> None:
    """Full macro build with fake HTTP: series, units, publication flags."""
    import pathlib

    from asx_quant.config.loader import load_settings, project_root

    settings = load_settings(project_root(pathlib.Path(__file__)))

    macro = build_macro_observations(settings, http_get=_fake_http_dispatch)
    names = set(macro["series_name"])
    assert "rba_cash_rate" in names
    assert "au_gov_yield_2y" in names
    cash = macro[macro["series_name"] == "rba_cash_rate"]
    # observation dated at the ANNOUNCEMENT (day before effective); the
    # effective date is one day later (kept in bronze for provenance).
    assert (cash["observation_date"] == cash["effective_date"]
            - pd.Timedelta(days=1)).all()
    assert (cash["publication_date"] == cash["observation_date"]).all()
    assert (cash["publication_date_method"] == "same_day_publication").all()
    daily = macro[macro["series_name"] == "au_gov_yield_2y"]
    assert (daily["publication_date"] == daily["observation_date"]).all()
    # leakage guard holds for every ingested row
    assert (macro["publication_date"] >= macro["observation_date"]).all()


# ------------------------------------------------------------ ASIC parsing
def test_parse_ytd_pivot_long_format() -> None:
    parsed = parse_ytd_pivot(YTD_CSV.encode())
    assert len(parsed) == 4  # '-' cells produce no records
    assert set(parsed["code"]) == {"TDO", "3PL", "BIG"}
    tdo = parsed[parsed["code"] == "TDO"].iloc[0]
    assert tdo["trade_date"] == pd.Timestamp("2026-11-06")
    assert tdo["reported_short_positions"] == 181029
    assert tdo["pct_of_product_in_issue"] == 0.03
    assert parsed["report_date"].iloc[0] == pd.Timestamp("2026-11-06")
    # quoted name containing the delimiter must not shift columns
    big = parsed[(parsed["code"] == "BIG")
                 & (parsed["trade_date"] == pd.Timestamp("2026-11-05"))].iloc[0]
    assert big["reported_short_positions"] == 444
    assert big["pct_of_product_in_issue"] == 0.09
    assert "BIG CO, INC ORDINARY" == big["product_name"]


# Pre-2022 vintage: UTF-16LE, tab-separated, ISO dates, 4-row header block.
# Dates sit at pair-start columns (2, 4, 6...) with an empty gap column
# between; values occupy (2+2i, 3+2i) like the current vintage.
def _old_row(fields: list[str]) -> str:
    return "\t".join(fields)


OLD_YTD_ROWS = [
    ["", "", "Trade Date"],
    ["", "", "2018-11-30", "", "2018-11-29", "", "2018-11-28"],
    ["", "", "Reported Short Positions",
     "% of Total Product in Issue Reported as Short Positions",
     "Reported Short Positions",
     "% of Total Product in Issue Reported as Short Positions",
     "Reported Short Positions",
     "% of Total Product in Issue Reported as Short Positions"],
    ["Product", " Product Code", ""],
    # Quoted product name CONTAINING the delimiter (real-world case:
    # "GI DYNAMICS\t INC CDI US PROHIBITED"); must not shift columns.
    ['"GI DYNAMICS\t INC CDI US PROHIBITED"', "GID ", "64000", "1.09",
     "", "", "", ""],
    # code, then pairs (pos, pct): 11-30 none, 11-29 none, 11-28 (93000,0.06)
    ["ZIPTEL LIMITED ORDINARY", "ZIP ", "", "", "", "", "93000", "0.06"],
    # 11-30 (181029, 0.03), 11-29 none, 11-28 none
    ["3D ENERGI LTD ORDINARY", "TDO", "181029", "0.03", "", "", "", ""],
]


def test_parse_ytd_pivot_old_utf16_vintage() -> None:
    content = "\r\n".join(_old_row(r) for r in OLD_YTD_ROWS).encode("utf-16")
    parsed = parse_ytd_pivot(content)
    assert len(parsed) == 3
    gid = parsed[parsed["code"] == "GID"].iloc[0]
    # the quoted name is unquoted by the CSV reader and kept whole
    assert "GI DYNAMICS" in gid["product_name"]
    assert "INC CDI US PROHIBITED" in gid["product_name"]
    assert gid["trade_date"] == pd.Timestamp("2018-11-30")
    assert gid["reported_short_positions"] == 64000
    ziptel = parsed[parsed["code"] == "ZIP"].iloc[0]
    assert ziptel["trade_date"] == pd.Timestamp("2018-11-28")
    assert ziptel["reported_short_positions"] == 93000
    assert ziptel["pct_of_product_in_issue"] == 0.06
    tdo = parsed[parsed["code"] == "TDO"].iloc[0]
    assert tdo["trade_date"] == pd.Timestamp("2018-11-30")
    assert tdo["pct_of_product_in_issue"] == 0.03
    assert (parsed["report_date"] == pd.Timestamp("2018-11-30")).all()


def test_pick_ytd_reports_latest_per_year() -> None:
    index = pd.DataFrame({
        "report_date": pd.to_datetime(
            ["2026-11-06", "2026-11-05", "2015-07-01", "2015-06-30"]),
        "version": ["001", "002", "001", "001"],
    })
    picks = pick_ytd_reports(index, start_year=2015)
    assert [p["year"] for p in picks] == [2015, 2026]
    assert picks[1]["report_date"] == pd.Timestamp("2026-11-06")
    # start_year filter drops earlier years
    picks_2020 = pick_ytd_reports(index, start_year=2020)
    assert [p["year"] for p in picks_2020] == [2026]


def test_ingest_asic_reports_publication_dates_offline() -> None:
    import pathlib

    from asx_quant.config.loader import load_settings, project_root

    project = project_root(pathlib.Path(__file__))
    settings = load_settings(project)
    settings = type(settings)(
        project_root_dir=project,
        raw={**settings.raw,
             "ingestion": {
                 **settings.raw["ingestion"],
                 "asic_short_positions": {
                     **settings.raw["ingestion"]["asic_short_positions"],
                     "start_year": 2015,
                     "batch_pause_seconds": 0.0,
                 },
             }},
    )
    index, positions, logs = ingest_asic_reports(settings, _fake_http_dispatch)
    assert set(positions["report_year"]) == {2015, 2026}
    assert (positions["publication_date"] >= positions["trade_date"]).all()
    trade_5 = positions[positions["trade_date"] == pd.Timestamp("2026-11-05")]
    assert trade_5["publication_date"].iloc[0] == pd.Timestamp("2026-11-05")
    assert (positions["publication_date_method"]
            == "estimated_from_report_calendar").all()
    failed = [r for r in logs if r.get("status") != "success"]
    assert failed == []


def test_ytd_report_url_pattern() -> None:
    url = ytd_report_url(pd.Timestamp("2026-11-06"), "001")
    assert url == ("https://download.asic.gov.au/short-selling/"
                   "RR20261106-001-SSDailyYTD.csv")


# ---------------------------------------------------------- quality checks
def test_macro_quality_clean_passes() -> None:
    report = run_macro_quality_checks(_macro_frame())
    assert report.passed, [r.name for r in report.errors if not r.passed]


def test_macro_quality_publication_before_observation_is_error() -> None:
    macro = _macro_frame()
    macro.loc[0, "publication_date"] = macro.loc[0, "observation_date"] \
        - pd.Timedelta(days=1)
    report = run_macro_quality_checks(macro)
    rule = next(r for r in report.results
                if r.name == "publication_not_before_observation")
    assert rule.severity == "error" and not rule.passed
    with pytest.raises(DataQualityError):
        report.assert_passed()


def test_macro_quality_duplicate_series_observation_is_error() -> None:
    macro = pd.concat([_macro_frame(), _macro_frame().tail(1)],
                      ignore_index=True)
    report = run_macro_quality_checks(macro)
    rule = next(r for r in report.results
                if r.name == "no_duplicate_series_observation")
    assert not rule.passed and rule.severity == "error"


def test_macro_quality_unflagged_method_is_warning() -> None:
    macro = _macro_frame(publication_date_method="unknown_method")
    report = run_macro_quality_checks(macro)
    rule = next(r for r in report.results
                if r.name == "publication_method_flagged")
    assert rule.severity == "warn" and not rule.passed
    assert report.passed  # warn never blocks


def test_short_position_quality_rules() -> None:
    good = pd.DataFrame({
        "ticker": ["CBA.AX"] * 2,
        "trade_date": pd.to_datetime(["2026-11-05", "2026-11-06"]),
        "reported_short_positions": [100.0, 120.0],
        "pct_of_product_in_issue": [0.01, 0.02],
        "publication_date": pd.to_datetime(["2026-11-05", "2026-11-06"]),
    })
    assert run_short_position_quality_checks(good).passed

    leaky = good.copy()
    leaky.loc[0, "publication_date"] = pd.Timestamp("2026-11-04")
    report = run_short_position_quality_checks(leaky)
    assert not next(r for r in report.results
                    if r.name == "publication_not_before_trade_date").passed

    over = good.copy()
    over.loc[0, "pct_of_product_in_issue"] = 150.0
    report = run_short_position_quality_checks(over)
    rule = next(r for r in report.results if r.name == "pct_within_bounds")
    assert rule.severity == "warn" and not rule.passed
    assert report.passed  # warn never blocks; rows kept and disclosed

    below_zero = good.copy()
    below_zero.loc[0, "pct_of_product_in_issue"] = -1.0
    report = run_short_position_quality_checks(below_zero)
    rule = next(r for r in report.results if r.name == "pct_non_negative")
    assert rule.severity == "error" and not rule.passed
    with pytest.raises(DataQualityError):
        report.assert_passed()

    dup = pd.concat([good, good.tail(1)], ignore_index=True)
    report = run_short_position_quality_checks(dup)
    assert not next(r for r in report.results
                    if r.name == "no_duplicate_ticker_trade_date").passed


# --------------------------------------------------- offline silver e2e
def _write_bronze_macro(lake: DataLake, macro: pd.DataFrame,
                        run_id: str) -> None:
    from asx_quant.storage.parquet import ParquetWriter

    class _S:  # minimal settings shim for the writer
        parquet_compression = "zstd"
        parquet_compression_level = 3

    for table_key, group in macro.groupby("source_table"):
        path = lake.bronze_batch_path("rba", str(table_key), run_id)
        ParquetWriter(_S()).write(group, path, run_id=run_id,
                                  dataset_name=f"bronze_rba_{table_key}")


def _write_bronze_asic(lake: DataLake, positions: pd.DataFrame,
                       run_id: str) -> None:
    from asx_quant.storage.parquet import ParquetWriter

    class _S:
        parquet_compression = "zstd"
        parquet_compression_level = 3

    for year, group in positions.groupby("report_year"):
        path = lake.bronze_batch_path("asic", f"ytd_{int(year)}", run_id)
        ParquetWriter(_S()).write(group, path, run_id=run_id,
                                  dataset_name=f"bronze_asic_{int(year)}")


def test_macro_and_short_silver_offline(tmp_path) -> None:
    from asx_quant.transformations.macro import (
        transform_macro_to_silver,
        transform_short_positions_to_silver,
    )

    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    registry = RunRegistry.start(lake, config_version="v0.1")

    class _S:
        parquet_compression = "zstd"
        parquet_compression_level = 3

    macro = _macro_frame()
    _write_bronze_macro(lake, macro, registry.run_id)
    macro_path, macro_report = transform_macro_to_silver(
        lake, _S(), registry)
    assert macro_report is not None and macro_report.passed
    silver = pd.read_parquet(macro_path)
    assert len(silver) == 3
    assert {"series_name", "observation_date", "publication_date"} <= \
        set(silver.columns)

    positions = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-11-05", "2026-11-06"]),
        "code": ["TDO", "TDO"],
        "product_name": ["3D ENERGI LTD ORDINARY"] * 2,
        "reported_short_positions": [100.0, 120.0],
        "pct_of_product_in_issue": [0.01, 0.02],
        "report_date": pd.to_datetime(["2026-11-05", "2026-11-06"]),
        "report_year": [2026, 2026],
        "publication_date": pd.to_datetime(["2026-11-05", "2026-11-06"]),
        "publication_date_method": ["estimated_from_report_calendar"] * 2,
    })
    _write_bronze_asic(lake, positions, registry.run_id)
    short_path, short_report = transform_short_positions_to_silver(
        lake, _S(), registry)
    assert short_report is not None and short_report.passed
    short_silver = pd.read_parquet(short_path)
    assert short_silver["ticker"].eq("TDO.AX").all()  # code -> standard ticker
    assert len(short_silver) == 2


def test_load_series_specs_from_real_config() -> None:
    import pathlib

    from asx_quant.config.loader import load_settings, project_root

    settings = load_settings(project_root(pathlib.Path(__file__)))
    specs = {s.name: s for s in load_series_specs(settings)}
    expected = {
        "au_gov_yield_2y", "au_gov_yield_3y",
        "au_gov_yield_10y", "aud_usd", "aud_usd_monthly", "abs_cpi",
        "abs_cpi_trimmed_mean", "abs_unemployment_rate",
        "rba_commodity_index",
    }
    assert expected <= set(specs)
    assert specs["abs_cpi"].publication_lag_days == 35
    assert specs["abs_unemployment_rate"].publication_lag_days == 15
    assert specs["abs_unemployment_rate"].original_publisher == "abs"


def test_macro_history_start_precedes_price_start() -> None:
    from asx_quant.ingestion.rba import macro_history_start

    class _Cfg:
        raw = {"ingestion": {"market_data": {"history_years": 10}}}

    start = macro_history_start(_Cfg())
    assert start <= date.today() - pd.Timedelta(days=365 * 10 - 1)
