"""Model training: purged walk-forward CV, evaluation and artifacts (Phase 5).

Modules:
    splits   — purged/embargoed walk-forward fold generation and masks
    dataset  — gold features + target labels -> training frame and matrix
    fit      — config-declared model factory (sklearn pipelines)
    cv       — walk-forward training/evaluation loop and metric aggregation
    checks   — model-quality rules (leakage invariants, prediction sanity)
    build    — orchestrator: config -> CV -> predictions + artifacts
"""
