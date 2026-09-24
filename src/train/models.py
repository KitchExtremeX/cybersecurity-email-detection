"""Estimators compared by the training pipeline.

Every model exposes ``predict_proba`` so inference can report confidence.
The linear SVM is wrapped in probability calibration for that reason.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

MODEL_ORDER = ("svm", "logistic_regression", "decision_tree", "random_forest")


def build_models(seed: int) -> dict:
    """Return the four classifiers in comparison order."""
    svm = CalibratedClassifierCV(
        estimator=LinearSVC(
            class_weight="balanced",
            random_state=seed,
            max_iter=10000,
        ),
        cv=3,
        method="sigmoid",
    )
    models = {
        "svm": svm,
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
    }
    return {name: models[name] for name in MODEL_ORDER}
