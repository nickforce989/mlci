from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from mlci.sensitivity.hyperparameter import hyperparameter_sensitivity

X, y = load_breast_cancer(return_X_y=True)

results = hyperparameter_sensitivity(
    model_factory=lambda seed, n_estimators, max_depth:
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
        ),
    param_grid={
        "n_estimators": [10, 50, 100, 200],
        "max_depth":    [None, 5, 10],
    },
    X=X, y=y,
    metric="accuracy",
    n_seeds=10,
    n_splits=5,
)

results.summary_table()