import pandas as pd
import pytest

from prodml.features import create_features, vectorize_features


@pytest.mark.parametrize(
    ("pickup", "dropoff", "expected"),
    [
        (100, 180, "100_180"),
        (1, 2, "1_2"),
        (265, 265, "265_265"),
    ],
)
def test_create_features_builds_pu_do(
    pickup: int,
    dropoff: int,
    expected: str,
) -> None:
    df = pd.DataFrame(
        {
            "PULocationID": [pickup],
            "DOLocationID": [dropoff],
            "trip_distance": [2.5],
        }
    )

    result = create_features(df)

    assert result.loc[0, "PU_DO"] == expected
    assert result.loc[0, "trip_distance"] == 2.5
    assert list(result.columns) == ["PU_DO", "trip_distance"]


def test_create_features_does_not_modify_input() -> None:
    df = pd.DataFrame(
        {
            "PULocationID": [100],
            "DOLocationID": [180],
            "trip_distance": [2.5],
        }
    )
    original = df.copy(deep=True)

    create_features(df)

    pd.testing.assert_frame_equal(df, original)


def test_create_features_missing_location_column_raises() -> None:
    df = pd.DataFrame({"trip_distance": [2.5]})

    with pytest.raises(KeyError):
        create_features(df)


def test_vectorize_features_handles_unseen_pu_do() -> None:
    train = pd.DataFrame(
        {
            "PU_DO": ["100_180", "100_200"],
            "trip_distance": [2.5, 4.0],
        }
    )
    validation = pd.DataFrame(
        {
            "PU_DO": ["999_999"],
            "trip_distance": [3.0],
        }
    )

    X_train, X_val, dv = vectorize_features(train, validation)

    assert X_val.shape[0] == 1
    assert X_val.shape[1] == X_train.shape[1]
    assert dv.transform(validation.to_dict(orient="records")).shape == X_val.shape


def test_create_features_missing_location_id_raises() -> None:
    df = pd.DataFrame(
        {
            "PULocationID": [None],
            "DOLocationID": [180],
            "trip_distance": [2.5],
        }
    )

    with pytest.raises((ValueError, TypeError)):
        create_features(df)


def test_create_features_zero_distance_preserved() -> None:
    df = pd.DataFrame(
        {
            "PULocationID": [100],
            "DOLocationID": [180],
            "trip_distance": [0.0],
        }
    )

    result = create_features(df)

    # Feature engineering preserves the value; API validation
    # is responsible for rejecting zero distance.
    assert result.loc[0, "trip_distance"] == 0.0
