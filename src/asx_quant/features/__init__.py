"""Feature engineering (Phase 4).

Computes the configured feature families from Silver data into Gold
datasets, with leakage controls enforced by construction and by tests:

* ticker-level features use only data at or before each session;
* macro and short-interest features join on ``publication_date <= session``
  (never the observation period);
* target labels are the ONLY columns allowed to reference future sessions,
  and they carry ``label_available_date`` for purge/embargo splitting.

The authoritative catalogue is ``docs/feature_catalogue.md``; WHICH features
are enabled and their windows come from ``config/features.yaml``.
"""

from asx_quant.features.build import build_feature_set

__all__ = ["build_feature_set"]
