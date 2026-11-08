# Progress

COMPLETED: protocol implementation, 64 synthetic problems, 1,920 recorded decisions across three policies, held-out 32/32 temperature calibration, clustered intervals, style sensitivity, failure examples, report and plots.
COMPLETED: three unit tests in an isolated pinned Python 3.12 environment; independent re-analysis reproduces all analysis tables and both plot files exactly.
NOT RUN: expert annotation, another model family, new judge-specific fine-tuning.

The conditional A/B judges always selected A. Reward-model neutral accuracy was 78.1% but collapsed under the joint style/format perturbations. Do not claim a generally reliable judge or successful calibration.

Resume/reproduce: README commands. The companion lab's original SFT/reward checkpoints are required; their hashes and base-model revision are recorded in results/protocol.json. Public GitHub publication: COMPLETED after explicit final approval; agreed commit verified.

Publication URL: https://github.com/Mioulin/scientific-judge-calibration
Approved research commit: 5c1fc3c023f832b5fc5e89d096ade107988ab990
CI status is available in GitHub Actions; publication-status documentation may have a later commit.
