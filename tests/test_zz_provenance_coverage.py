"""Runs last (file name): the provenance records that the jobs of the OTHER test modules
wrote while the default tier ran in this process.

The other modules run nearly every analysis job on fabricated data with planted effects.
Each of those runs went through the job's own ``provenance._prov_record`` call; the app
keeps the last records of the process in memory (``_prov_recent``) whatever cache
directory a test pointed it at. Here every one of them is checked for the required
fields, for being plain JSON and for a methods paragraph that renders, and the set of
kinds is compared with the kinds the default tier is known to exercise.

Skipped unless the whole default tier ran in this process (``pytest -k ...`` or a single
file would see only part of the records)."""
from __future__ import annotations

import json

import pytest

# kinds whose jobs the default tier runs for real (the others need R, Baysor / Proseg,
# Cellpose or a reference and are covered by `-m tools`, by the fabricated-call test of
# tests/test_provenance.py and by the source check there)
DEFAULT_TIER_KINDS = {
    "de", "pseudobulk", "proximity", "contact", "geneset", "density", "composition", "moran", "lr",
    "lrdecomp", "compartments", "spillover", "segqc", "coloc", "aggregates", "segfree", "banksy",
    "joint", "txfactor", "spatial", "niches", "imgseg", "patches", "save",
}
# not run for real in the default tier: annotation, baysor, proseg, split, spage (tools /
# R / a reference), umap, cellnest, and h5ad (its tests convert in a process of their own)


def _whole_tier_ran(request) -> bool:
    files = {item.fspath.basename for item in request.session.items}
    need = {"test_stats.py", "test_controls.py", "test_analysis.py", "test_aggregates.py", "test_banksy.py",
            "test_joint.py", "test_lrdecomp.py", "test_proximity.py", "test_segfree.py", "test_txfactor.py",
            "test_imgseg.py", "test_save_sdata.py", "test_spatial_analysis.py"}
    return need <= files and not request.config.getoption("-k")


def test_records_written_by_the_jobs_of_the_other_test_modules(xe, request):
    if not _whole_tier_ran(request):
        pytest.skip("needs a full run of the default tier in this process")
    recs = list(xe._prov_recent)
    assert recs, "no job of the test session wrote a provenance record"
    by_kind = {}
    for r in recs:
        by_kind.setdefault(r["kind"], []).append(r)
        assert all(k in r for k in xe._PROV_REQUIRED), (r["kind"], set(xe._PROV_REQUIRED) - set(r))
        assert json.loads(json.dumps(r, allow_nan=False)) == r
        assert r["kind"] in xe._PROV_KINDS and r["software"]["python"] and r["data"]["n_cells"] >= 0
        md = xe._prov_methods_md([r])
        assert f"## {r['title']}" in md and "could not be filled" not in md, r["kind"]
    print("\nprovenance records of this session by kind:",
          {k: len(v) for k, v in sorted(by_kind.items())})
    missing = DEFAULT_TIER_KINDS - set(by_kind)
    # _prov_recent keeps the last _PROV_RECENT_MAX records: a kind whose jobs ran early may
    # have been pushed out, so a missing kind only fails while the list is not full
    if len(recs) < xe._PROV_RECENT_MAX:
        assert not missing, f"no record from a real job run for: {sorted(missing)}"
    # the tests that state a unit state it in the record too
    for kind in ("de", "moran", "lr"):
        for r in by_kind.get(kind, []):
            if "Unit of the p-values" in r["status_line"]:
                assert r["unit"]["statement"] and r["unit"]["source"] == "status line"
