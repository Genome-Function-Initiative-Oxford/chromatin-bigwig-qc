# E9 - Implementation and availability

Status: done (this repository is the implementation).

**Paper section:** Materials and Methods / Implementation and availability;
required Frontiers "Data Availability Statement" / "Supplemental Data"
sections.

## Ask

- Record software versions: LanceOtron, XGBoost, SHAP, pyBigWig, CATCH-UP
  (and whatever else ends up used across E1-E8).
- Package/pipeline release: GitHub repository + Zenodo DOI.
- Public accessions for every collection (GEO GSE124853, GSE125164, CATlas,
  10x Genomics, etc.) for the Data Availability Statement.
- Per-track quality scores released as supplementary data.

## Suggested starting point

- `pyproject.toml` already pins the direct Python dependencies; once E1-E8
  settle on their final tooling, freeze exact versions here
  (`pip freeze`/`pip list` after a full run) for the methods section.
- This repository *is* the release artefact referenced by the paper - a
  Zenodo DOI can be minted from a GitHub release/tag when ready.
- Per-track scores: `scripts/train_model.py`'s output model, applied back
  over `scripts/build_dataset.py`'s dataset via `bigwig_qc.model.load_model`,
  gives the full per-track score table to publish as Supplementary Data.
