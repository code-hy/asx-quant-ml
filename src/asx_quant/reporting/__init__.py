"""Read-only reporting layer (Phase 9).

Modules:
    artifacts — safe loaders for lake artifacts (JSON reports, parquet
                tables) + the consolidated platform_status() snapshot;
                a missing artifact reads as None, never fabricated
"""
