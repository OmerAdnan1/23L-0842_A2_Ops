"""
Task 4.4 - All runtime settings come from environment variables.
"""
import os
import secrets
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    inference_mode: str = "local"                 # local | remote
    model_path: str = "models/champion.keras"
    model_name: str = "champion"
    threshold: float = 0.80
    database_url: str = "sqlite:///visionguard.db"
    max_upload_mb: int = 5
    model_api_url: str = ""                       # used when inference_mode=remote
    model_api_key: str = ""                       # sent to the model API
    api_key: str = ""                             # expected by the model API itself
    secret_key: str = field(default_factory=lambda: secrets.token_hex(16))

    @classmethod
    def from_env(cls):
        e = os.environ.get
        return cls(
            inference_mode=e("INFERENCE_MODE", "local"),
            model_path=e("MODEL_PATH", cls.model_path),
            model_name=e("MODEL_NAME", cls.model_name),
            threshold=float(e("THRESHOLD", cls.threshold)),
            database_url=e("DATABASE_URL", cls.database_url),
            max_upload_mb=int(e("MAX_UPLOAD_MB", cls.max_upload_mb)),
            model_api_url=e("MODEL_API_URL", ""),
            model_api_key=e("MODEL_API_KEY", ""),
            api_key=e("API_KEY", ""),
            secret_key=e("SECRET_KEY") or secrets.token_hex(16),
        )
