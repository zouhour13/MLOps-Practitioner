import logging
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import create_features, vectorize_features
from prodml.logging_conf import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Train, evaluate, and save the baseline model."""
    df = load_data()
    X_train, X_val, y_train, y_val = split_data(df)

    X_train = create_features(X_train)
    X_val = create_features(X_val)

    X_train_vectorized, X_val_vectorized, dv = vectorize_features(X_train, X_val)

    model = LinearRegression()
    model.fit(X_train_vectorized, y_train)

    predictions = model.predict(X_val_vectorized)

    mae = mean_absolute_error(y_val, predictions)
    rmse = root_mean_squared_error(y_val, predictions)

    logger.info("Validation MAE: %.3f minutes", mae)
    logger.info("Validation RMSE: %.3f minutes", rmse)

    settings.model_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "dv": dv,
        "model": model,
        "features": ["PU_DO", "trip_distance"],
    }

    with settings.model_path.open("wb") as model_file:
        pickle.dump(artifact, model_file)

    logger.info("Model saved to: %s", settings.model_path)


if __name__ == "__main__":
    configure_logging()
    main()
