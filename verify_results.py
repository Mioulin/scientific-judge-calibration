"""Verify clustered judgement design and regenerate analysis in isolation."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from study import CONDITIONS, analyse


def verify(root=Path("results")):
    problems = [json.loads(s) for s in (root / "problems.jsonl").read_text().splitlines()]
    assert len(problems) == 64 and len({r["id"] for r in problems}) == 64
    mapping = {r["id"]: r for r in problems}
    splits = {
        s: {r["id"] for r in problems if r["judgement_split"] == s} for s in ["calibration", "test"]
    }
    assert len(splits["calibration"]) == len(splits["test"]) == 32 and not splits[
        "calibration"
    ].intersection(splits["test"])
    inputs = ["problems.jsonl"]
    count = 0
    for policy in ["baseline", "sft", "reward"]:
        fn = f"{policy}.jsonl"
        inputs.append(fn)
        rows = [json.loads(s) for s in (root / fn).read_text().splitlines()]
        assert len(rows) == 640
        expected = {
            (pid, c, o)
            for pid in mapping
            for c in CONDITIONS
            for o in ["good_first", "good_second"]
        }
        assert {(r["id"], r["condition"], r["order"]) for r in rows} == expected
        for r in rows:
            assert 0 <= r["prob_correct"] <= 1 and np.isfinite(r["prob_correct"])
            assert r["correct"] == (r["prob_correct"] > 0.5)
            assert r["judgement_split"] == mapping[r["id"]]["judgement_split"]
        count += len(rows)
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        for fn in inputs:
            (scratch / fn).write_bytes((root / fn).read_bytes())
        analyse(scratch)
        for fn in [
            "metrics.json",
            "temperature_calibration.json",
            "style_sensitivity.json",
            "reliability.json",
            "failure_examples.json",
        ]:
            assert json.loads((root / fn).read_text()) == json.loads((scratch / fn).read_text()), fn
        for fn in ["comparison.png", "reliability.png"]:
            assert (root / fn).read_bytes() == (scratch / fn).read_bytes(), fn
    manifest = {
        "plots_exactly_reproduced": True,
        "status": "COMPLETED",
        "verified_decisions": count,
        "calibration_test_disjoint": True,
        "analysis_exactly_recomputed": True,
        "input_sha256": {fn: hashlib.sha256((root / fn).read_bytes()).hexdigest() for fn in inputs},
    }
    (root / "verification.json").write_text(json.dumps(manifest, indent=2))
    print(f"Verified {count} decisions and reproduced all analysis tables.")


if __name__ == "__main__":
    verify()
