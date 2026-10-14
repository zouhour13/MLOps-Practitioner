from typing import Any

import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the pickup/drop-off location feature."""
    features = df.copy()

    features["PU_DO"] = (
        features["PULocationID"].astype(int).astype(str)
        + "_"
        + features["DOLocationID"].astype(int).astype(str)
    )

    return features[["PU_DO", "trip_distance"]]


def vectorize_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    dv: DictVectorizer | None = None,
) -> tuple[Any, Any, DictVectorizer]:
    """Fit a vectorizer on training data and transform both datasets."""
    if dv is None:
        dv = DictVectorizer()

    X_train_dicts = X_train.to_dict(orient="records")
    X_val_dicts = X_val.to_dict(orient="records")

    X_train_vectorized = dv.fit_transform(X_train_dicts)
    X_val_vectorized = dv.transform(X_val_dicts)

    return X_train_vectorized, X_val_vectorized, dv
