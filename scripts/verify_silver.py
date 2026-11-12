"""Quick warehouse verification (development helper, not part of the package)."""
from asx_quant.config.loader import load_settings
from asx_quant.storage.duckdb import get_warehouse

settings = load_settings()
wh = get_warehouse(settings)
with wh:
    print(wh.query(
        "SELECT dataset_name, record_count FROM meta.dataset_registry "
        "ORDER BY dataset_name"
    ).to_string(index=False))
    print(wh.query(
        "SELECT COUNT(*) AS rows, COUNT(DISTINCT ticker) AS tickers, "
        "MIN(date) AS first, MAX(date) AS last FROM silver.equity_prices"
    ).to_string(index=False))
    print(wh.query(
        "SELECT COUNT(*) AS sessions, MIN(calendar_date) AS first, "
        "MAX(calendar_date) AS last FROM silver.market_calendar"
    ).to_string(index=False))
    print(wh.query(
        "SELECT data_quality_status, COUNT(*) AS n FROM silver.security_master "
        "GROUP BY data_quality_status ORDER BY n DESC"
    ).to_string(index=False))
