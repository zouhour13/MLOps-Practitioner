from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, configurable through environment variables."""

    project_root: Path = Path(__file__).resolve().parents[2]
    data_filename: str = "green_tripdata_2021-01.parquet"
    model_filename: str = "model.pkl"
    validation_size: float = 0.2
    random_state: int = 42

    model_config = SettingsConfigDict(
        env_prefix="PRODML_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def data_path(self) -> Path:
        """Return the dataset path."""
        return self.project_root / "data" / self.data_filename

    @property
    def model_path(self) -> Path:
        """Return the trained model artifact path."""
        return self.project_root / "models" / self.model_filename


settings = Settings()
