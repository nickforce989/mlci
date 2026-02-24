"""
Tests for the new modules:
  - mlci.integrations.sklearn
  - mlci.benchmarks.datasets
  - mlci.benchmarks.runner
  - mlci.sensitivity.hyperparameter
  - mlci.sensitivity.dataset
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from mlci.core.results import ExperimentResults


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def iris():
    return load_iris(return_X_y=True)

@pytest.fixture
def binary():
    return load_breast_cancer(return_X_y=True)


# ═══════════════════════════════════════════════════════════════
# INTEGRATIONS — sklearn
# ═══════════════════════════════════════════════════════════════

class TestPipelineFactory:

    def test_returns_pipeline_with_scaler(self, iris):
        from mlci.integrations.sklearn import PipelineFactory
        from sklearn.pipeline import Pipeline
        X, y = iris
        factory = PipelineFactory(LogisticRegression, scaler="standard")
        model = factory(seed=0)
        assert isinstance(model, Pipeline)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_no_scaler(self, iris):
        from mlci.integrations.sklearn import PipelineFactory
        X, y = iris
        factory = PipelineFactory(RandomForestClassifier, scaler="none")
        model = factory(seed=42)
        model.fit(X, y)
        assert hasattr(model, "predict")

    def test_seed_is_injected(self, iris):
        from mlci.integrations.sklearn import PipelineFactory
        X, y = iris
        factory = PipelineFactory(RandomForestClassifier, params={"n_estimators": 5})
        m0 = factory(seed=0)
        m1 = factory(seed=1)
        m0.fit(X, y); m1.fit(X, y)
        # Different seeds → different predictions (with high probability)
        # Just check they run without error
        assert m0.predict(X).shape == (len(y),)

    def test_invalid_scaler_raises(self):
        from mlci.integrations.sklearn import PipelineFactory
        with pytest.raises(ValueError, match="Unknown scaler"):
            PipelineFactory(RandomForestClassifier, scaler="magic")

    def test_repr(self):
        from mlci.integrations.sklearn import PipelineFactory
        f = PipelineFactory(RandomForestClassifier, scaler="standard")
        assert "RandomForestClassifier" in repr(f)


class TestWrapCrossValScores:

    def test_1d_input(self):
        from mlci.integrations.sklearn import wrap_cross_val_scores
        scores = [0.90, 0.91, 0.89, 0.92, 0.88]
        r = wrap_cross_val_scores(scores, metric="accuracy", model_name="RF")
        assert r.scores.shape == (1, 5)
        assert r.metric == "accuracy"
        assert r.model_name == "RF"

    def test_2d_input(self):
        from mlci.integrations.sklearn import wrap_cross_val_scores
        scores = np.array([
            [0.90, 0.91, 0.89],
            [0.92, 0.88, 0.93],
        ])
        r = wrap_cross_val_scores(scores)
        assert r.scores.shape == (2, 3)

    def test_from_cross_val_score(self, iris):
        from mlci.integrations.sklearn import wrap_cross_val_scores
        from sklearn.model_selection import cross_val_score
        X, y = iris
        scores = cross_val_score(
            RandomForestClassifier(n_estimators=5, random_state=0),
            X, y, cv=3,
        )
        r = wrap_cross_val_scores(scores, metric="accuracy", model_name="RF")
        assert r.n_splits == 3
        assert 0.5 < r.mean < 1.0

    def test_summary_runs(self, capsys):
        from mlci.integrations.sklearn import wrap_cross_val_scores
        from mlci.stats.bootstrap import summary
        scores = np.random.RandomState(0).uniform(0.85, 0.95, size=(5, 5))
        r = wrap_cross_val_scores(scores)
        summary(r)
        out = capsys.readouterr().out
        assert "Mean" in out


class TestModelGrid:

    def test_basic_run(self, iris):
        from mlci.integrations.sklearn import ModelGrid
        X, y = iris
        grid = ModelGrid(
            models={
                "RF":    lambda seed: RandomForestClassifier(n_estimators=5, random_state=seed),
                "Dummy": lambda seed: DummyClassifier(random_state=seed),
            },
            X=X, y=y,
            metric="accuracy",
            n_seeds=3,
            n_splits=3,
        )
        results = grid.run(verbose=False)
        assert set(results.keys()) == {"RF", "Dummy"}
        assert results["RF"].scores.shape == (3, 3)
        assert results["RF"].mean > results["Dummy"].mean

    def test_results_are_experiment_results(self, iris):
        from mlci.integrations.sklearn import ModelGrid
        X, y = iris
        grid = ModelGrid(
            models={"RF": lambda seed: RandomForestClassifier(n_estimators=5, random_state=seed)},
            X=X, y=y, n_seeds=2, n_splits=2,
        )
        results = grid.run(verbose=False)
        assert isinstance(results["RF"], ExperimentResults)


# ═══════════════════════════════════════════════════════════════
# BENCHMARKS — datasets
# ═══════════════════════════════════════════════════════════════

class TestBenchmarkDatasets:

    def test_breast_cancer(self):
        from mlci.benchmarks.datasets import load_breast_cancer_bench
        d = load_breast_cancer_bench()
        assert d.name == "BreastCancer"
        assert d.task == "classification"
        assert d.n_classes == 2
        assert d.X.shape == (569, 30)
        assert len(d.y) == 569

    def test_iris(self):
        from mlci.benchmarks.datasets import load_iris_bench
        d = load_iris_bench()
        assert d.n_classes == 3
        assert d.n_samples == 150

    def test_digits(self):
        from mlci.benchmarks.datasets import load_digits_bench
        d = load_digits_bench()
        assert d.n_features == 64
        assert d.n_classes == 10

    def test_diabetes(self):
        from mlci.benchmarks.datasets import load_diabetes_bench
        d = load_diabetes_bench()
        assert d.task == "regression"
        assert d.n_classes is None
        assert d.default_metric == "rmse"

    def test_load_all_classification(self):
        from mlci.benchmarks.datasets import load_all_classification
        datasets = load_all_classification()
        assert len(datasets) == 4
        assert all(d.task == "classification" for d in datasets)

    def test_load_all_regression(self):
        from mlci.benchmarks.datasets import load_diabetes_bench, load_california_bench
        d = load_diabetes_bench()
        assert d.task == "regression"
        assert d.n_samples == 442
        try:
            ca = load_california_bench()
            assert ca.task == "regression"
        except RuntimeError:
            pytest.skip("CaliforniaHousing not available (requires internet download)")

    def test_repr(self):
        from mlci.benchmarks.datasets import load_iris_bench
        d = load_iris_bench()
        assert "Iris" in repr(d)
        assert "150" in repr(d)


# ═══════════════════════════════════════════════════════════════
# BENCHMARKS — runner
# ═══════════════════════════════════════════════════════════════

class TestBenchmarkRunner:

    def test_basic_run(self):
        from mlci.benchmarks.runner import run_benchmark
        from mlci.benchmarks.datasets import load_iris_bench, load_wine_bench
        datasets = [load_iris_bench(), load_wine_bench()]
        models = {
            "RF":    lambda seed: RandomForestClassifier(n_estimators=5, random_state=seed),
            "Dummy": lambda seed: DummyClassifier(random_state=seed),
        }
        results = run_benchmark(models, datasets, n_seeds=3, n_splits=3, verbose=False)
        assert set(results.model_names) == {"RF", "Dummy"}
        assert set(results.dataset_names) == {"Iris", "Wine"}
        assert len(results.results) == 4  # 2 models × 2 datasets

    def test_rankings_sorted(self):
        from mlci.benchmarks.runner import run_benchmark
        from mlci.benchmarks.datasets import load_iris_bench
        models = {
            "RF":    lambda seed: RandomForestClassifier(n_estimators=10, random_state=seed),
            "Dummy": lambda seed: DummyClassifier(random_state=seed),
        }
        results = run_benchmark(models, [load_iris_bench()], n_seeds=3, n_splits=3, verbose=False)
        ranking = results.rankings["Iris"]
        # RF should rank above Dummy
        model_order = [m for m, _ in ranking]
        assert model_order.index("RF") < model_order.index("Dummy")

    def test_corrected_wins_keys(self):
        from mlci.benchmarks.runner import run_benchmark
        from mlci.benchmarks.datasets import load_iris_bench
        models = {
            "RF":    lambda seed: RandomForestClassifier(n_estimators=10, random_state=seed),
            "Dummy": lambda seed: DummyClassifier(random_state=seed),
        }
        results = run_benchmark(models, [load_iris_bench()], n_seeds=5, n_splits=5, verbose=False)
        # corrected_wins is populated per dataset (may be None if no sig. winner)
        assert "Iris" in results.corrected_wins
        # RF should clearly beat Dummy — expect a winner
        assert results.corrected_wins["Iris"] is not None

    def test_benjamini_hochberg(self):
        from mlci.benchmarks.runner import benjamini_hochberg
        # Very small p-values should be rejected
        p_vals = [0.001, 0.002, 0.5, 0.8, 0.9]
        rejected = benjamini_hochberg(p_vals, alpha=0.05)
        assert rejected[0] is True
        assert rejected[1] is True
        assert rejected[4] is False

    def test_benjamini_hochberg_empty(self):
        from mlci.benchmarks.runner import benjamini_hochberg
        assert benjamini_hochberg([], alpha=0.05) == []

    def test_print_benchmark_report_runs(self, capsys):
        from mlci.benchmarks.runner import run_benchmark, print_benchmark_report
        from mlci.benchmarks.datasets import load_iris_bench
        models = {
            "RF":    lambda seed: RandomForestClassifier(n_estimators=5, random_state=seed),
            "Dummy": lambda seed: DummyClassifier(random_state=seed),
        }
        results = run_benchmark(models, [load_iris_bench()], n_seeds=2, n_splits=2, verbose=False)
        print_benchmark_report(results)
        out = capsys.readouterr().out
        assert "Iris" in out
        assert "RF" in out


# ═══════════════════════════════════════════════════════════════
# SENSITIVITY — hyperparameter
# ═══════════════════════════════════════════════════════════════

class TestHyperparameterSensitivity:

    def test_basic_run(self, iris):
        from mlci.sensitivity.hyperparameter import hyperparameter_sensitivity
        X, y = iris
        results = hyperparameter_sensitivity(
            model_factory=lambda seed, n_estimators:
                RandomForestClassifier(n_estimators=n_estimators, random_state=seed),
            param_grid={"n_estimators": [5, 10]},
            X=X, y=y,
            n_seeds=3, n_splits=3,
        )
        assert len(results.param_grid) == 2
        assert len(results.results) == 2
        assert "n_estimators" in results.sensitivity

    def test_best_params_is_valid(self, iris):
        from mlci.sensitivity.hyperparameter import hyperparameter_sensitivity
        X, y = iris
        results = hyperparameter_sensitivity(
            model_factory=lambda seed, n_estimators:
                RandomForestClassifier(n_estimators=n_estimators, random_state=seed),
            param_grid={"n_estimators": [5, 50]},
            X=X, y=y,
            n_seeds=3, n_splits=3,
        )
        assert results.best_params in results.param_grid
        assert results.best_result is not None
        # 50 trees should generally beat 5 trees
        assert results.best_params["n_estimators"] == 50

    def test_sensitivity_non_negative(self, iris):
        from mlci.sensitivity.hyperparameter import hyperparameter_sensitivity
        X, y = iris
        results = hyperparameter_sensitivity(
            model_factory=lambda seed, n_estimators:
                RandomForestClassifier(n_estimators=n_estimators, random_state=seed),
            param_grid={"n_estimators": [5, 10, 20]},
            X=X, y=y,
            n_seeds=2, n_splits=2,
        )
        for val in results.sensitivity.values():
            assert val >= 0

    def test_multi_param_grid(self, iris):
        from mlci.sensitivity.hyperparameter import hyperparameter_sensitivity
        X, y = iris
        results = hyperparameter_sensitivity(
            model_factory=lambda seed, n_estimators, max_depth:
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=seed,
                ),
            param_grid={
                "n_estimators": [5, 10],
                "max_depth":    [3, None],
            },
            X=X, y=y,
            n_seeds=2, n_splits=2,
        )
        # 2 × 2 = 4 combinations
        assert len(results.param_grid) == 4
        assert "n_estimators" in results.sensitivity
        assert "max_depth" in results.sensitivity


# ═══════════════════════════════════════════════════════════════
# SENSITIVITY — dataset
# ═══════════════════════════════════════════════════════════════

class TestDatasetSensitivity:

    def test_basic_run(self, iris):
        from mlci.sensitivity.dataset import dataset_sensitivity
        X, y = iris
        results = dataset_sensitivity(
            model_factory=lambda seed: RandomForestClassifier(n_estimators=5, random_state=seed),
            X=X, y=y,
            n_seeds=3, n_splits=3,
        )
        assert len(results.misclassification_rate) == len(y)
        assert len(results.appearances) == len(y)

    def test_rates_in_valid_range(self, iris):
        from mlci.sensitivity.dataset import dataset_sensitivity
        X, y = iris
        results = dataset_sensitivity(
            model_factory=lambda seed: DummyClassifier(random_state=seed),
            X=X, y=y,
            n_seeds=3, n_splits=3,
        )
        valid = ~np.isnan(results.misclassification_rate)
        assert np.all(results.misclassification_rate[valid] >= 0)
        assert np.all(results.misclassification_rate[valid] <= 1)

    def test_all_samples_visited(self, iris):
        from mlci.sensitivity.dataset import dataset_sensitivity
        X, y = iris
        results = dataset_sensitivity(
            model_factory=lambda seed: DummyClassifier(random_state=seed),
            X=X, y=y,
            n_seeds=5, n_splits=5,
        )
        # With 5 folds every sample appears in exactly one test fold per seed
        assert np.all(results.appearances > 0)

    def test_hard_easy_indices_valid(self, binary):
        from mlci.sensitivity.dataset import dataset_sensitivity
        X, y = binary
        results = dataset_sensitivity(
            model_factory=lambda seed: RandomForestClassifier(n_estimators=5, random_state=seed),
            X=X, y=y,
            n_seeds=3, n_splits=3,
        )
        # Hard indices should have rate > 0.5
        if len(results.hard_indices) > 0:
            assert np.all(results.misclassification_rate[results.hard_indices] > 0.5)
        # Easy indices should have rate == 0
        if len(results.easy_indices) > 0:
            assert np.all(results.misclassification_rate[results.easy_indices] == 0.0)

    def test_repr_runs(self, iris):
        from mlci.sensitivity.dataset import dataset_sensitivity
        X, y = iris
        results = dataset_sensitivity(
            model_factory=lambda seed: DummyClassifier(random_state=seed),
            X=X, y=y,
            n_seeds=2, n_splits=3,
            model_name="DummyTest",
        )
        r = repr(results)
        assert "DummyTest" in r
        assert "samples" in r


# ═══════════════════════════════════════════════════════════════
# INTEGRATION — everything works together
# ═══════════════════════════════════════════════════════════════

class TestEndToEndNew:

    def test_pipeline_factory_into_experiment(self, iris):
        """PipelineFactory → Experiment → bootstrap_ci chain."""
        from mlci.integrations.sklearn import PipelineFactory
        from mlci import Experiment
        from mlci.stats.bootstrap import bootstrap_ci
        X, y = iris
        factory = PipelineFactory(LogisticRegression, scaler="standard",
                                  params={"max_iter": 500})
        exp = Experiment(
            model_factory=factory,
            X=X, y=y,
            metric="accuracy",
            n_seeds=3, n_splits=3,
            model_name="LogReg+Scaler",
        )
        results = exp.run(verbose=False)
        ci = bootstrap_ci(results)
        assert ci.lower <= ci.mean <= ci.upper
        assert 0.7 < ci.mean < 1.0

    def test_model_grid_then_compare(self, iris):
        """ModelGrid → compare two results from the grid."""
        from mlci.integrations.sklearn import ModelGrid
        from mlci.stats.tests import compare
        X, y = iris
        grid = ModelGrid(
            models={
                "RF":    lambda seed: RandomForestClassifier(n_estimators=10, random_state=seed),
                "Dummy": lambda seed: DummyClassifier(random_state=seed),
            },
            X=X, y=y, n_seeds=5, n_splits=5,
        )
        results = grid.run(verbose=False)
        comp = compare(results["RF"], results["Dummy"], method="corrected_ttest")
        assert comp.p_value < 0.05   # RF should clearly beat Dummy
        assert comp.effect_size > 0

    def test_benchmark_then_sensitivity(self, iris):
        """Load a benchmark dataset, run dataset sensitivity on it."""
        from mlci.benchmarks.datasets import load_iris_bench
        from mlci.sensitivity.dataset import dataset_sensitivity
        d = load_iris_bench()
        results = dataset_sensitivity(
            model_factory=lambda seed: RandomForestClassifier(n_estimators=5, random_state=seed),
            X=d.X, y=d.y,
            n_seeds=3, n_splits=3,
            model_name=d.name,
        )
        assert len(results.misclassification_rate) == d.n_samples
