"""Inspect securities without usable price data (development helper)."""
from asx_quant.config.loader import load_settings
from asx_quant.storage.duckdb import get_warehouse

wh = get_warehouse(load_settings())
with wh:
    no_data = wh.query(
        "SELECT ticker, company_name, data_quality_status "
        "FROM silver.security_master WHERE data_quality_status = 'no_price_data'"
    )
    print(no_data.to_string(index=False))
    benchmarks = wh.query(
        "SELECT DISTINCT ticker FROM silver.equity_prices "
        "WHERE ticker LIKE 'IDX%' OR ticker = 'VAS.AX'"
    )
    print(benchmarks.to_string(index=False))
