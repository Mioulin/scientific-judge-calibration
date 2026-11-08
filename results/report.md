# Scientific judge calibration — executed study

64 independent synthetic problems; eight families, five controlled style conditions, both response orders. 640 pair judgements per policy; only 64 independent problem clusters. Baseline and SFT share one Qwen family; reward model is a separate scalar head trained on synthetic preferences.

| Policy | Condition | Accuracy | Brier | Order flips |
|---|---|---:|---:|---:|
| baseline | neutral | 0.500 | 0.479 | 1.000 |
| baseline | wrong_verbose | 0.500 | 0.482 | 1.000 |
| baseline | correct_verbose | 0.500 | 0.487 | 1.000 |
| baseline | wrong_confident | 0.500 | 0.494 | 1.000 |
| baseline | correct_confident | 0.500 | 0.494 | 1.000 |
| sft | neutral | 0.500 | 0.485 | 1.000 |
| sft | wrong_verbose | 0.500 | 0.471 | 1.000 |
| sft | correct_verbose | 0.500 | 0.491 | 1.000 |
| sft | wrong_confident | 0.500 | 0.495 | 1.000 |
| sft | correct_confident | 0.500 | 0.496 | 1.000 |
| reward | neutral | 0.781 | 0.166 | 0.000 |
| reward | wrong_verbose | 0.125 | 0.759 | 0.000 |
| reward | correct_verbose | 1.000 | 0.005 | 0.000 |
| reward | wrong_confident | 0.000 | 0.934 | 0.000 |
| reward | correct_confident | 1.000 | 0.000 | 0.000 |

## Interpretation and limitations
A/B probabilities are softmax-normalised over two next-token logits: conditional forced-choice preferences, not unconstrained response probabilities. Brier is computed against the objectively correct candidate and is a proper binary score. A separate temperature experiment fits T on 32 neutral calibration problems and evaluates on 32 disjoint problems under every style condition. T is selected from 81 log-spaced values in [0.25,4] by calibration NLL; positive T cannot improve classification accuracy. The raw 64-problem audit remains separate from this held-out calibration test. Small data and model mismatch prevent claims of established calibration. Reward probabilities are sigmoid reward margins, whose scale is not intrinsically calibrated. Reward order invariance is architectural, not an empirical discovery. No malformed generation rate is measured because decisions are constrained to A/B. Confidence claims are rhetorical perturbations, not elicited uncertainty. Labels grade the core numerical/evidence answer, not every appended rhetorical claim. For the reward model, appended prose also violates its strict-JSON training format: this is a joint style/format distribution shift, not an isolated verbosity effect. The correct-verbose control is retained to inspect that confound.
Numerical answers have independent Decimal oracles. Evidence/abstention labels are template-defined. Synthetic preferences are not human annotations. Family-level generality, expert agreement, externally validated uncertainty calibration and a second model family are NOT RUN. Bootstraps resample problems, keeping conditions/orders together; systematic template error remains. New condition combinations can change conclusions.
The SFT/reward checkpoints are the earlier lab checkpoints, not adapted using this test. The SFT adapter was trained to answer scientific questions, not specifically to judge response pairs; this is a transfer audit, not an optimised judge-training comparison. New numerical instances exclude the fresh tool-study dataset; the shared families/templates remain an important overlap. No claims about proprietary reviewer tasks are made.

## Held-out temperature calibration: neutral test
| Policy | T | Raw Brier | Scaled Brier | Raw NLL | Scaled NLL |
|---|---:|---:|---:|---:|---:|
| baseline | 4.000 | 0.479 | 0.312 | 2.121 | 0.840 |
| sft | 4.000 | 0.486 | 0.323 | 2.387 | 0.870 |
| reward | 1.189 | 0.152 | 0.153 | 0.456 | 0.460 |

An uninformative probability of 0.5 has Brier 0.25 and NLL 0.693; this is an essential calibration control. Lower Brier than the raw model alone does not imply beating this control. Temperature changes uncertainty only, not decision accuracy. Values at either search bound indicate a bounded optimum rather than a precisely identified temperature. All style-specific test scores remain in temperature_calibration.json.
