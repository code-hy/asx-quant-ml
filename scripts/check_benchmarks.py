"""Check benchmark presence in silver (development helper)."""
from asx_quant.config.loader import load_settings
from asx_quant.storage.duckdb import get_warehouse

wh = get_warehouse(load_settings())
with wh:
    print(wh.query(
        "SELECT ticker, COUNT(*) AS n FROM silver.equity_prices "
        "GROUP BY ticker ORDER BY n DESC LIMIT 10"
    ).to_string(index=False))
    print("---")
    print(wh.query(
        "SELECT ticker, COUNT(*) AS n, MIN(date) AS first, MAX(date) AS last "
        "FROM silver.equity_prices "
        "WHERE ticker IN ('^AXJO', 'IDX_AXJO', 'VAS.AX') GROUP BY ticker"
    ).to_string(index=False))
