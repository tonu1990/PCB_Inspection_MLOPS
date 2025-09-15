"""
App settings loaded from environment variables.

We keep this simple (no external libs) to avoid surprises.
In Docker/Pi, you'll pass the same variables via an env file.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

@dataclass
class Settings:
    # Where the web server binds (the container will map ports anyway)
    host: str = os.getenv("APP_HOST")
    port: int = int(os.getenv("APP_PORT"))

    # Where the ONNX model will be mounted on the Pi
    model_path: str = os.getenv("MODEL_PATH")

    # if the model's first input has dynamic shape, use this as square size (e.g., 640)
    model_input_size: int = int(os.getenv("MODEL_INPUT_SIZE"))

    # General app metadata and logging
    app_name: str = os.getenv("APP_NAME")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

# A single global instance is fine for simple settings
settings = Settings()
