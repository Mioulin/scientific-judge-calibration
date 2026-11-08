"""Original controlled judge study. No paid API or private evaluation data."""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

CONDITIONS = [
    "neutral",
    "wrong_verbose",
    "correct_verbose",
    "wrong_confident",
    "correct_confident",
]
FAMILIES = [
    "rate",
    "mean",
    "difference",
    "weighted",
    "bayes",
    "missing_rate",
    "causality",
    "evidence",
]


def decorate(text, condition, correct):
    target = condition.startswith("correct") if condition != "neutral" else None
    if target is not None and target == correct:
        if condition.endswith("verbose"):
            return (
                text
                + "\nThis answer follows a careful scientific analysis. The reasoning has been checked systematically, with attention to the quantities, their interpretation, and the evidence supplied in the question."
            )
        return text + "\nI am absolutely certain that this answer is correct."
    return text


def brier(prob):
    return float(np.mean((np.asarray(prob) - 1) ** 2))


def cluster_ci(values, seed=902):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(len(values), size=(4000, len(values)))].mean(axis=1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def temperature_scale(prob, temperature):
    p = np.clip(np.asarray(prob), 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p)) / temperature
    return 1 / (1 + np.exp(-logit))


def fit_temperature(prob):
    candidates = np.geomspace(0.25, 4, 81)
    losses = [-np.mean(np.log(np.clip(temperature_scale(prob, t), 1e-9, 1))) for t in candidates]
    return float(candidates[np.argmin(losses)])


def run(lab_root, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lab = Path(lab_root).resolve()
    sys.path.insert(0, str(lab / "src"))
    os.chdir(lab)
    from scireason.data import SYSTEM
    from scireason.runtime import hardware, load
    from scireason.toolstudy import make_example, semantic_key

    rng = random.Random(773290)
    # Explicitly exclude all fresh-study numerical conditions.
    exclusions = set()
    for p in (lab / "data/toolstudy").glob("*.jsonl"):
        exclusions.update(semantic_key(json.loads(s)) for s in p.read_text().splitlines())
    problems = []
    for family in FAMILIES:
        for i in range(8):
            row = make_example(family, "test", i, rng)
            while semantic_key(row) in exclusions:
                row = make_example(family, "test", i, rng)
            exclusions.add(semantic_key(row))
            row["judgement_split"] = "calibration" if i < 4 else "test"
            problems.append(row)
    (output / "problems.jsonl").write_text("".join(json.dumps(r) + "\n" for r in problems))
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "status": "RUNNING",
                "n_problems": 64,
                "conditions": CONDITIONS,
                "seed": 773290,
                "hardware": hardware(),
                "lab_revision": __import__("subprocess")
                .check_output(["git", "rev-parse", "HEAD"], text=True)
                .strip(),
                "problems_sha256": hashlib.sha256(
                    (output / "problems.jsonl").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
    )
    for policy, adapter, reward in [
        ("baseline", None, False),
        ("sft", "checkpoints/sft/final", False),
        ("reward", "checkpoints/reward/final", True),
    ]:
        start = time.perf_counter()
        torch.manual_seed(42)
        model, tok, device = load(adapter, reward=reward)
        model.eval()
        records = []
        with (output / f"{policy}.jsonl").open("w") as f:
            for j, row in enumerate(problems):
                for condition in CONDITIONS:
                    good = decorate(row["chosen"], condition, True)
                    bad = decorate(row["rejected"], condition, False)
                    if reward:
                        scores = []
                        for response in [good, bad]:
                            chat = [
                                {"role": "system", "content": SYSTEM},
                                {"role": "user", "content": row["prompt"]},
                                {"role": "assistant", "content": response},
                            ]
                            inp = tok(
                                tok.apply_chat_template(chat, tokenize=False),
                                return_tensors="pt",
                            ).to(device)
                            with torch.inference_mode():
                                scores.append(float(model(**inp).logits[0, 0]))
                        margin = scores[0] - scores[1]
                    for order in ["good_first", "good_second"]:
                        if reward:
                            prob = float(torch.sigmoid(torch.tensor(margin)))
                        else:
                            a, b = (good, bad) if order == "good_first" else (bad, good)
                            prompt = f"Problem: {row['prompt']}\nResponse A: {a}\nResponse B: {b}\nWhich response is scientifically correct? Judge factual and numerical correctness, not style. Reply with A or B only."
                            chat = [{"role": "user", "content": prompt}]
                            text = tok.apply_chat_template(
                                chat, tokenize=False, add_generation_prompt=True
                            )
                            inp = tok(text, return_tensors="pt").to(device)
                            ids = [tok.encode(x, add_special_tokens=False) for x in ["A", "B"]]
                            assert all(len(x) == 1 for x in ids)
                            with torch.inference_mode():
                                logits = model(**inp).logits[0, -1, [x[0] for x in ids]].float()
                            pp = logits.softmax(0).cpu().numpy()
                            prob = float(pp[0 if order == "good_first" else 1])
                        assert np.isfinite(prob) and 0 <= prob <= 1, (
                            "Non-finite decision probability"
                        )
                        record = {
                            "id": row["id"],
                            "judgement_split": row["judgement_split"],
                            "family": row["family"],
                            "condition": condition,
                            "order": order,
                            "prob_correct": prob,
                            "correct": prob > 0.5,
                            "tie": prob == 0.5,
                        }
                        if reward:
                            record["margin"] = margin
                        records.append(record)
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                if j % 16 == 0:
                    print(policy, j, flush=True)
        (output / f"{policy}_runtime.json").write_text(
            json.dumps(
                {
                    "elapsed_seconds": time.perf_counter() - start,
                    "adapter": adapter,
                    "status": "COMPLETED",
                },
                indent=2,
            )
        )
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    analyse(output)


def analyse(output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplcache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = []
    for policy in ["baseline", "sft", "reward"]:
        rows = [json.loads(x) for x in (output / f"{policy}.jsonl").read_text().splitlines()]
        for condition in CONDITIONS:
            subset = [x for x in rows if x["condition"] == condition]
            groups = {i: [] for i in sorted({r["id"] for r in subset})}
            for r in subset:
                groups[r["id"]].append(r)
            per_problem = [np.mean([r["correct"] for r in group]) for group in groups.values()]
            flips = np.mean(
                [group[0]["correct"] != group[1]["correct"] for group in groups.values()]
            )
            first = [r for r in subset if r["order"] == "good_first"]
            second = [r for r in subset if r["order"] == "good_second"]
            table.append(
                {
                    "policy": policy,
                    "condition": condition,
                    "accuracy": float(np.mean(per_problem)),
                    "ci95": cluster_ci(per_problem),
                    "brier": brier([r["prob_correct"] for r in subset]),
                    "order_flip_rate": float(flips),
                    "good_first_minus_second": float(
                        np.mean([r["correct"] for r in first])
                        - np.mean([r["correct"] for r in second])
                    ),
                }
            )
    (output / "metrics.json").write_text(json.dumps(table, indent=2))
    calibration = []
    for policy in ["baseline", "sft", "reward"]:
        rr = [json.loads(x) for x in (output / f"{policy}.jsonl").read_text().splitlines()]
        cal = [
            r["prob_correct"]
            for r in rr
            if r["judgement_split"] == "calibration" and r["condition"] == "neutral"
        ]
        temp = fit_temperature(cal)
        for condition in CONDITIONS:
            probs = [
                r["prob_correct"]
                for r in rr
                if r["judgement_split"] == "test" and r["condition"] == condition
            ]
            scaled = temperature_scale(probs, temp)
            calibration.append(
                {
                    "policy": policy,
                    "condition": condition,
                    "temperature": temp,
                    "n_independent_test_problems": 32,
                    "raw_brier": brier(probs),
                    "scaled_brier": brier(scaled),
                    "raw_nll": float(-np.log(np.clip(probs, 1e-9, 1)).mean()),
                    "scaled_nll": float(-np.log(np.clip(scaled, 1e-9, 1)).mean()),
                }
            )
    (output / "temperature_calibration.json").write_text(json.dumps(calibration, indent=2))
    sensitivity = []
    reliability = []
    for policy in ["baseline", "sft", "reward"]:
        rr = [json.loads(x) for x in (output / f"{policy}.jsonl").read_text().splitlines()]
        bykey = {(r["id"], r["condition"], r["order"]): r for r in rr}
        for condition in CONDITIONS[1:]:
            deltas = []
            for pid in sorted({r["id"] for r in rr}):
                deltas.append(
                    np.mean(
                        [
                            float(bykey[pid, condition, o]["correct"])
                            - float(bykey[pid, "neutral", o]["correct"])
                            for o in ["good_first", "good_second"]
                        ]
                    )
                )
            sensitivity.append(
                {
                    "policy": policy,
                    "condition": condition,
                    "accuracy_delta_vs_neutral": float(np.mean(deltas)),
                    "cluster_ci95": cluster_ci(deltas),
                }
            )
        # Diagnostic reliability: condition/order repeats remain correlated.
        confidence = np.array([max(r["prob_correct"], 1 - r["prob_correct"]) for r in rr])
        accuracy = np.array([r["correct"] for r in rr])
        for lo, hi in zip([0.5, 0.6, 0.7, 0.8, 0.9], [0.6, 0.7, 0.8, 0.9, 1.000001]):
            select = (confidence >= lo) & (confidence < hi)
            if select.any():
                reliability.append(
                    {
                        "policy": policy,
                        "lower": lo,
                        "upper": min(hi, 1),
                        "n_repeated_decisions": int(select.sum()),
                        "mean_confidence": float(confidence[select].mean()),
                        "accuracy": float(accuracy[select].mean()),
                    }
                )
    (output / "style_sensitivity.json").write_text(json.dumps(sensitivity, indent=2))
    (output / "reliability.json").write_text(json.dumps(reliability, indent=2))
    fig, ax = plt.subplots(figsize=(10, 4))
    for k, policy in enumerate(["baseline", "sft", "reward"]):
        vals = [r["accuracy"] for r in table if r["policy"] == policy]
        ax.bar(np.arange(5) + (k - 1) * 0.25, vals, 0.25, label=policy)
    ax.set_xticks(np.arange(5), CONDITIONS, rotation=15)
    ax.set(ylim=(0, 1), ylabel="Pairwise correctness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Scientific judge calibration — executed study",
        "",
        "64 independent synthetic problems; eight families, five controlled style conditions, both response orders. 640 pair judgements per policy; only 64 independent problem clusters. Baseline and SFT share one Qwen family; reward model is a separate scalar head trained on synthetic preferences.",
        "",
        "| Policy | Condition | Accuracy | Brier | Order flips |",
        "|---|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['policy']} | {r['condition']} | {r['accuracy']:.3f} | {r['brier']:.3f} | {r['order_flip_rate']:.3f} |"
        for r in table
    ]
    lines += [
        "",
        "## Interpretation and limitations",
        "A/B probabilities are softmax-normalised over two next-token logits: conditional forced-choice preferences, not unconstrained response probabilities. Brier is computed against the objectively correct candidate and is a proper binary score. A separate temperature experiment fits T on 32 neutral calibration problems and evaluates on 32 disjoint problems under every style condition. T is selected from 81 log-spaced values in [0.25,4] by calibration NLL; positive T cannot improve classification accuracy. The raw 64-problem audit remains separate from this held-out calibration test. Small data and model mismatch prevent claims of established calibration. Reward probabilities are sigmoid reward margins, whose scale is not intrinsically calibrated. Reward order invariance is architectural, not an empirical discovery. No malformed generation rate is measured because decisions are constrained to A/B. Confidence claims are rhetorical perturbations, not elicited uncertainty. Labels grade the core numerical/evidence answer, not every appended rhetorical claim. For the reward model, appended prose also violates its strict-JSON training format: this is a joint style/format distribution shift, not an isolated verbosity effect. The correct-verbose control is retained to inspect that confound.",
        "Numerical answers have independent Decimal oracles. Evidence/abstention labels are template-defined. Synthetic preferences are not human annotations. Family-level generality, expert agreement, externally validated uncertainty calibration and a second model family are NOT RUN. Bootstraps resample problems, keeping conditions/orders together; systematic template error remains. New condition combinations can change conclusions.",
        "The SFT/reward checkpoints are the earlier lab checkpoints, not adapted using this test. New numerical instances exclude the fresh tool-study dataset; the shared families/templates remain an important overlap. No claims about proprietary reviewer tasks are made.",
    ]
    lines += [
        "",
        "## Held-out temperature calibration: neutral test",
        "| Policy | T | Raw Brier | Scaled Brier | Raw NLL | Scaled NLL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in calibration:
        if r["condition"] == "neutral":
            lines.append(
                f"| {r['policy']} | {r['temperature']:.3f} | {r['raw_brier']:.3f} | {r['scaled_brier']:.3f} | {r['raw_nll']:.3f} | {r['scaled_nll']:.3f} |"
            )
    lines += [
        "",
        "Temperature changes uncertainty only, not decision accuracy. Values at either search bound indicate a bounded optimum rather than a precisely identified temperature. All style-specific test scores remain in temperature_calibration.json.",
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    for policy in ["baseline", "sft", "reward"]:
        rs = [r for r in reliability if r["policy"] == policy]
        ax.plot(
            [r["mean_confidence"] for r in rs],
            [r["accuracy"] for r in rs],
            marker="o",
            label=policy,
        )
    ax.plot([0.5, 1], [0.5, 1], "k--")
    ax.set(
        xlim=(0.5, 1),
        ylim=(0, 1),
        xlabel="Conditional chosen-answer confidence",
        ylabel="Observed correctness",
        title="Raw descriptive reliability: repeated decisions",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "reliability.png", dpi=160)
    plt.close(fig)
    problems = {
        r["id"]: r
        for r in [json.loads(x) for x in (output / "problems.jsonl").read_text().splitlines()]
    }
    failures = []
    for policy in ["baseline", "sft", "reward"]:
        rr = [json.loads(x) for x in (output / f"{policy}.jsonl").read_text().splitlines()]
        selected = [r for r in rr if r["condition"] == "neutral" and not r["correct"]][:3]
        failures.extend(
            [
                {
                    "policy": policy,
                    "selection": "first three neutral incorrect records in deterministic file order",
                    **r,
                    "problem": problems[r["id"]],
                }
                for r in selected
            ]
        )
    (output / "failure_examples.json").write_text(json.dumps(failures, indent=2))
    (output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab-root", required=True)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    run(args.lab_root, args.output)
