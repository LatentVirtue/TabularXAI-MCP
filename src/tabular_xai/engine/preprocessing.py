"""Leak-free per-fold preprocessing: imputation, scaling, categorical encoding.

The trainer builds a FRESH transformer inside every fold (fit on the train
partition only, applied to the validation partition), so no test-fold
statistics ever leak into preprocessing.
"""
from __future__ import annotations

from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MAX_ONEHOT_CATEGORIES = 40


def build_column_transformer(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    for_nn: bool,
) -> ColumnTransformer:
    """ColumnTransformer: median-impute (+ StandardScaler for the NN path)
    for numeric columns; most-frequent-impute + one-hot (handle_unknown=ignore,
    capped cardinality) for categorical columns.
    """
    transformers = []
    if numeric_columns:
        steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if for_nn:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), list(numeric_columns)))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=2,
                                max_categories=MAX_ONEHOT_CATEGORIES,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                list(categorical_columns),
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def encoded_feature_names(prepared: ColumnTransformer) -> list[str]:
    """Encoded column names of a FITTED ColumnTransformer (e.g. 'numeric__x1',
    'categorical__cat_a_charlie')."""
    return [str(n) for n in prepared.get_feature_names_out()]
