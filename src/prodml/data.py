import pandas as pd
from sklearn.model_selection import train_test_split

from prodml.config import settings


def load_data() -> pd.DataFrame:
    """Load and clean the Green Taxi dataset."""
    df = pd.read_parquet(settings.data_path)

    df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])
    df["lpep_dropoff_datetime"] = pd.to_datetime(df["lpep_dropoff_datetime"])

    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df = df[
        (df["duration"] >= 1)
        & (df["duration"] <= 60)
        & (df["trip_distance"] > 0)
        & df["PULocationID"].notna()
        & df["DOLocationID"].notna()
    ].copy()

    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into training and validation sets."""
    features = ["PULocationID", "DOLocationID", "trip_distance"]
    target = "duration"

    X_train, X_val, y_train, y_val = train_test_split(
        df[features],
        df[target],
        test_size=settings.validation_size,
        random_state=settings.random_state,
    )

    return X_train, X_val, y_train, y_val
