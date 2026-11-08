from study import brier, cluster_ci, decorate


def test_intervention_preserves_answer():
    assert decorate("42", "neutral", True) == "42"
    assert decorate("42", "wrong_verbose", True) == "42"
    assert decorate("41", "wrong_verbose", False).startswith("41\n")
    assert decorate("42", "correct_confident", True).startswith("42\n")


def test_scores_and_cluster_interval():
    assert brier([1, 1]) == 0
    assert brier([0, 0]) == 1
    assert cluster_ci([1, 1, 1]) == [1, 1]


def test_temperature_preserves_decisions():
    import numpy as np

    from study import fit_temperature, temperature_scale

    p = np.array([0.1, 0.3, 0.8, 0.95])
    for t in [0.25, 1, 4]:
        assert np.array_equal(temperature_scale(p, t) > 0.5, p > 0.5)
    assert 0.25 <= fit_temperature(p) <= 4
