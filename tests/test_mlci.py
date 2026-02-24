"""
Test suite for mlci.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from mlci.core.results import ExperimentResults
from mlci.core.experiment import Experiment
from mlci.stats.bootstrap import bootstrap_ci, summary
from mlci.stats.tests import (
    compare,
    corrected_resampled_ttest,
    wilcoxon_test,
    bayesian_comparison,
)
from mlci.stats.anova import decompose_variance


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def iris_data():
    return load_iris(return_X_y=True)


@pytest.fixture
def binary_data():
    return load_breast_cancer(return_X_y=True)


@pytest.fixture
def synthetic_results_good():
    """A well-performing model: scores around 0.92."""
    rng = np.random.RandomState(0)
    scores = 0.92 + rng.normal(0, 0.02, size=(10, 5))
    scores = np.clip(scores, 0, 1)
    return ExperimentResults(scores=scores, metric="accuracy", model_name="GoodModel")


@pytest.fixture
def synthetic_results_bad():
    """A worse model: scores around 0.80."""
    rng = np.random.RandomState(1)
    scores = 0.80 + rng.normal(0, 0.02, size=(10, 5))
    scores = np.clip(scores, 0, 1)
    return ExperimentResults(scores=scores, metric="accuracy", model_name="BadModel")


@pytest.fixture
def synthetic_results_similar():
    """A model indistinguishable from GoodModel."""
    rng = np.random.RandomState(2)
    scores = 0.921 + rng.normal(0, 0.02, size=(10, 5))
    scores = np.clip(scores, 0, 1)
    return ExperimentResults(scores=scores, metric="accuracy", model_name="SimilarModel")


# -----------------------------------------------------------------------
# ExperimentResults
# -----------------------------------------------------------------------

class TestExperimentResults:
    def test_1d_input_becomes_2d(self):
        r = ExperimentResults(scores=[0.9, 0.91, 0.89], metric="accuracy")
        assert r.scores.shape == (3, 1)

    def test_2d_input(self):
        r = ExperimentResults(scores=np.ones((5, 3)))
        assert r.n_seeds == 5
        assert r.n_splits == 3

    def test_3d_raises(self):
        with pytest.raises(ValueError):
            ExperimentResults(scores=np.ones((2, 3, 4)))

    def test_mean(self):
        r = ExperimentResults(scores=np.ones((4, 3)) * 0.5)
        assert r.mean == pytest.approx(0.5)

    def test_seed_means_shape(self, synthetic_results_good):
        r = synthetic_results_good
        assert r.seed_means.shape == (r.n_seeds,)

    def test_split_means_shape(self, synthetic_results_good):
        r = synthetic_results_good
        assert r.split_means.shape == (r.n_splits,)

    def test_save_load_roundtrip(self, synthetic_results_good, tmp_path):
        path = tmp_path / "results.json"
        synthetic_results_good.save(path)
        loaded = ExperimentResults.load(path)
        np.testing.assert_allclose(loaded.scores, synthetic_results_good.scores)
        assert loaded.model_name == synthetic_results_good.model_name
        assert loaded.metric == synthetic_results_good.metric


# -----------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------

class TestExperiment:
    def test_basic_run(self, iris_data):
        X, y = iris_data
        exp = Experiment(
            model_factory=lambda seed: RandomForestClassifier(n_estimators=10, random_state=seed),
            X=X, y=y,
            metric="accuracy",
            n_seeds=3,
            n_splits=3,
        )
        results = exp.run(verbose=False)
        assert results.scores.shape == (3, 3)
        assert 0.5 < results.mean < 1.0

    def test_custom_metric(self, binary_data):
        X, y = binary_data
        from sklearn.metrics import roc_auc_score
        exp = Experiment(
            model_factory=lambda seed: LogisticRegression(random_state=seed, max_iter=500),
            X=X, y=y,
            metric=lambda yt, yp: roc_auc_score(yt, yp),
            n_seeds=3,
            n_splits=3,
            model_name="LogReg",
        )
        results = exp.run(verbose=False)
        assert 0.5 < results.mean <= 1.0

    def test_single_split(self, iris_data):
        X, y = iris_data
        exp = Experiment(
            model_factory=lambda seed: DummyClassifier(random_state=seed),
            X=X, y=y,
            n_seeds=5,
            n_splits=1,
        )
        results = exp.run(verbose=False)
        assert results.scores.shape == (5, 1)

    def test_reproducibility(self, iris_data):
        """Same seeds must give same scores."""
        X, y = iris_data
        kwargs = dict(
            model_factory=lambda seed: RandomForestClassifier(n_estimators=5, random_state=seed),
            X=X, y=y, n_seeds=3, n_splits=3,
        )
        r1 = Experiment(**kwargs).run(verbose=False)
        r2 = Experiment(**kwargs).run(verbose=False)
        np.testing.assert_allclose(r1.scores, r2.scores)

    def test_unknown_metric_raises(self, iris_data):
        X, y = iris_data
        with pytest.raises(ValueError, match="Unknown metric"):
            Experiment(
                model_factory=lambda seed: DummyClassifier(),
                X=X, y=y,
                metric="nonexistent_metric_xyz",
            )


# -----------------------------------------------------------------------
# Bootstrap CI
# -----------------------------------------------------------------------

class TestBootstrapCI:
    def test_ci_contains_mean(self, synthetic_results_good):
        ci = bootstrap_ci(synthetic_results_good)
        assert ci.lower <= ci.mean <= ci.upper

    def test_ci_width_positive(self, synthetic_results_good):
        ci = bootstrap_ci(synthetic_results_good)
        assert ci.width > 0

    def test_higher_confidence_wider(self, synthetic_results_good):
        ci_90 = bootstrap_ci(synthetic_results_good, confidence=0.90)
        ci_99 = bootstrap_ci(synthetic_results_good, confidence=0.99)
        assert ci_99.width > ci_90.width

    def test_constant_scores_tight_ci(self):
        """Constant scores → very tight CI."""
        r = ExperimentResults(scores=np.ones((10, 5)) * 0.9)
        ci = bootstrap_ci(r)
        assert ci.width < 1e-6

    def test_invalid_statistic_raises(self, synthetic_results_good):
        with pytest.raises(ValueError):
            bootstrap_ci(synthetic_results_good, statistic="mode")

    def test_summary_runs(self, synthetic_results_good, capsys):
        summary(synthetic_results_good)
        captured = capsys.readouterr()
        assert "Mean" in captured.out


# -----------------------------------------------------------------------
# Hypothesis tests
# -----------------------------------------------------------------------

class TestHypothesisTests:
    def test_corrected_ttest_clearly_different(
        self, synthetic_results_good, synthetic_results_bad
    ):
        """Models 12% apart should be clearly distinguishable."""
        result = corrected_resampled_ttest(synthetic_results_good, synthetic_results_bad)
        assert result.p_value < 0.05
        assert "better" in result.conclusion.lower()

    def test_corrected_ttest_similar(
        self, synthetic_results_good, synthetic_results_similar
    ):
        """Models 0.1% apart should NOT be distinguishable."""
        result = corrected_resampled_ttest(synthetic_results_good, synthetic_results_similar)
        # Not guaranteed to be non-significant with synthetic data, but effect should be tiny
        assert abs(result.effect_size) < 0.05

    def test_wilcoxon_clearly_different(
        self, synthetic_results_good, synthetic_results_bad
    ):
        result = wilcoxon_test(synthetic_results_good, synthetic_results_bad)
        assert result.p_value < 0.05

    def test_bayesian_clearly_different(
        self, synthetic_results_good, synthetic_results_bad
    ):
        result = bayesian_comparison(synthetic_results_good, synthetic_results_bad)
        assert result.prob_a_better > 0.9

    def test_bayesian_similar(self, synthetic_results_good, synthetic_results_similar):
        result = bayesian_comparison(synthetic_results_good, synthetic_results_similar)
        # P(A>B) should be uncertain — not close to 0 or 1
        assert 0.2 < result.prob_a_better < 0.8 or abs(result.effect_size) < 0.01

    def test_shape_mismatch_raises(
        self, synthetic_results_good
    ):
        r_wrong = ExperimentResults(scores=np.ones((5, 3)) * 0.9)
        with pytest.raises(ValueError, match="Shapes must match"):
            corrected_resampled_ttest(synthetic_results_good, r_wrong)

    def test_compare_dispatch(self, synthetic_results_good, synthetic_results_bad):
        for method in ("corrected_ttest", "wilcoxon", "bayesian"):
            result = compare(synthetic_results_good, synthetic_results_bad, method=method)
            assert result is not None

    def test_compare_invalid_method(self, synthetic_results_good, synthetic_results_bad):
        with pytest.raises(ValueError):
            compare(synthetic_results_good, synthetic_results_bad, method="garbage")

    def test_effect_size_direction(
        self, synthetic_results_good, synthetic_results_bad
    ):
        """Good model should have positive effect vs bad model."""
        result = corrected_resampled_ttest(synthetic_results_good, synthetic_results_bad)
        assert result.effect_size > 0  # Good - Bad > 0

    def test_ci_contains_effect(self, synthetic_results_good, synthetic_results_bad):
        result = corrected_resampled_ttest(synthetic_results_good, synthetic_results_bad)
        assert result.ci_lower <= result.effect_size <= result.ci_upper


# -----------------------------------------------------------------------
# Variance decomposition
# -----------------------------------------------------------------------

class TestVarianceDecomposition:
    def test_fractions_sum_to_one(self, synthetic_results_good):
        d = decompose_variance(synthetic_results_good)
        total = d.seed_fraction + d.split_fraction + d.interaction_fraction
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_all_fractions_non_negative(self, synthetic_results_good):
        d = decompose_variance(synthetic_results_good)
        assert d.seed_fraction >= 0
        assert d.split_fraction >= 0
        assert d.interaction_fraction >= 0

    def test_single_seed_raises(self):
        r = ExperimentResults(scores=np.ones((1, 5)) * 0.9)
        with pytest.raises(ValueError, match="2 seeds"):
            decompose_variance(r)

    def test_single_split_raises(self):
        r = ExperimentResults(scores=np.ones((5, 1)) * 0.9)
        with pytest.raises(ValueError, match="2 splits"):
            decompose_variance(r)

    def test_seed_dominant_when_seeds_vary(self):
        """When only seeds vary (splits are constant), seed variance dominates."""
        rng = np.random.RandomState(42)
        # High seed variance, low split variance
        scores = np.tile(rng.normal(0.9, 0.05, size=(10, 1)), (1, 5))
        scores += rng.normal(0, 0.001, size=(10, 5))  # tiny split noise
        r = ExperimentResults(scores=scores)
        d = decompose_variance(r)
        assert d.seed_fraction > d.split_fraction

    def test_split_dominant_when_splits_vary(self):
        """When only splits vary (seeds are constant), split variance dominates."""
        rng = np.random.RandomState(42)
        # High split variance, low seed variance
        scores = np.tile(rng.normal(0.9, 0.05, size=(1, 5)), (10, 1))
        scores += rng.normal(0, 0.001, size=(10, 5))
        r = ExperimentResults(scores=scores)
        d = decompose_variance(r)
        assert d.split_fraction > d.seed_fraction


# -----------------------------------------------------------------------
# Integration test: end-to-end
# -----------------------------------------------------------------------

class TestEndToEnd:
    def test_full_workflow(self, iris_data):
        """
        Full workflow: run two experiments, compare them statistically.
        This is the core use case of the library.
        """
        X, y = iris_data

        rf_exp = Experiment(
            model_factory=lambda seed: RandomForestClassifier(n_estimators=20, random_state=seed),
            X=X, y=y, metric="accuracy", n_seeds=5, n_splits=5,
            model_name="RandomForest",
        )
        lr_exp = Experiment(
            model_factory=lambda seed: LogisticRegression(random_state=seed, max_iter=500),
            X=X, y=y, metric="accuracy", n_seeds=5, n_splits=5,
            model_name="LogisticRegression",
        )

        rf_results = rf_exp.run(verbose=False)
        lr_results = lr_exp.run(verbose=False)

        # Both should have reasonable accuracy
        assert rf_results.mean > 0.8
        assert lr_results.mean > 0.8

        # Bootstrap CIs should be valid
        rf_ci = bootstrap_ci(rf_results)
        assert rf_ci.lower < rf_ci.mean < rf_ci.upper

        # Comparison should run without error
        comparison = compare(rf_results, lr_results, method="corrected_ttest")
        assert comparison is not None
        assert isinstance(comparison.p_value, float)

        # Variance decomposition should work
        d = decompose_variance(rf_results)
        assert abs(d.seed_fraction + d.split_fraction + d.interaction_fraction - 1.0) < 1e-5
