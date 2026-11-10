#!/usr/bin/env python
"""CLI: sample ENCODE DNase-seq experiments across three audit-severity tiers
(E8's DNase-seq half - mouse ATAC-seq is out of scope, see README) and build
a manifest of (accession, bigwig URL, audit tier) ready for download.

Audit tiers are defined by ENCODE's own "SPOT score" audits (Signal Portion
Of Tags - the ENCODE analogue of FRiP/peak enrichment, i.e. the audit
category most directly comparable to what this pipeline's own quality score
measures), not generic/administrative audits (e.g. "unreplicated experiment",
INTERNAL_ACTION) which aren't about signal quality per se:
  - "error":   has an ERROR-level "...spot score..." audit
  - "warning": has a WARNING-level "...spot score..." audit, no ERROR
  - "clean":   no audits of any kind

Only experiments with exactly one released GRCh38 "read-depth normalized
signal" bigWig file are kept, since that is a human/hg38 signal track
directly compatible with this pipeline's existing (human-only) reference
sets - no new species reference data is needed for this half of E8.
"""
import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

import pandas as pd

ENCODE_BASE = "https://www.encodeproject.org"
SEARCH_URL = (
    f"{ENCODE_BASE}/search/?type=Experiment&assay_title=DNase-seq&status=released"
    "&assembly=GRCh38&format=json&limit=all&field=accession&field=audit"
)


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def _audit_tier(audit: dict) -> str:
    categories = [
        (severity, entry.get("category", ""))
        for severity in ("ERROR", "WARNING")
        for entry in audit.get(severity, [])
    ]
    if any(severity == "ERROR" and "spot score" in category for severity, category in categories):
        return "error"
    if any(severity == "WARNING" and "spot score" in category for severity, category in categories):
        return "warning"
    if not audit:
        return "clean"
    return "other"  # has audits, but not spot-score-related and not ERROR/WARNING-spot - excluded


def _bigwig_file_for(accession: str) -> dict | None:
    """Return one representative bigWig file for `accession`, or None if no
    matching file exists. Most ENCODE experiments have one "read-depth
    normalized signal" bigWig PER BIOLOGICAL REPLICATE (not a single
    experiment-level file), with no `preferred_default` flag distinguishing
    them - picking the lowest biological-replicate number (falling back to
    file accession for determinism) is an arbitrary but fixed, reproducible
    choice; which specific replicate is used doesn't matter here since this
    pipeline already treats each experiment as a single bigWig regardless of
    replicate structure.
    """
    detail = _get_json(f"{ENCODE_BASE}/experiments/{accession}/?format=json")
    candidates = [
        f
        for f in detail.get("files", [])
        if isinstance(f, dict)
        and f.get("file_format") == "bigWig"
        and f.get("output_type") == "read-depth normalized signal"
        and f.get("assembly") == "GRCh38"
        and f.get("status") == "released"
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda f: (min(f.get("biological_replicates") or [999]), f.get("accession", "")))
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-tier", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    search_results = _get_json(SEARCH_URL)["@graph"]
    by_tier = {"clean": [], "warning": [], "error": []}
    for item in search_results:
        tier = _audit_tier(item.get("audit", {}))
        if tier in by_tier:
            by_tier[tier].append(item["accession"])
    print({tier: len(accs) for tier, accs in by_tier.items()})

    rng = random.Random(args.random_state)
    rows = []
    for tier, accessions in by_tier.items():
        rng.shuffle(accessions)
        found = 0
        for accession in accessions:
            if found >= args.n_per_tier:
                break
            bigwig_file = _bigwig_file_for(accession)
            time.sleep(0.2)  # polite rate limit on the ENCODE API
            if bigwig_file is None:
                continue
            rows.append(
                {
                    "experiment_accession": accession,
                    "file_accession": bigwig_file["accession"],
                    "bigwig_url": ENCODE_BASE + bigwig_file["href"],
                    "audit_tier": tier,
                }
            )
            found += 1
        print(f"tier={tier}: found {found}/{args.n_per_tier}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
