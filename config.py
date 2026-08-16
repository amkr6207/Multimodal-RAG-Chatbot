"""Application configuration validation."""

import os
from collections.abc import Mapping

from dotenv import load_dotenv

load_dotenv()


class ConfigurationError(ValueError):
    """Raised when required application configuration is missing or invalid."""


def validate_environment(environment: Mapping[str, str] | None = None) -> None:
    """Validate required settings without contacting external services."""
    values = environment if environment is not None else os.environ
    errors = []

    groq_api_key = values.get("GROQ_API_KEY", "").strip()
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        errors.append("GROQ_API_KEY is missing")

    mongodb_uri = values.get("MONGODB_ATLAS_CLUSTER_URI", "").strip()
    if not mongodb_uri:
        errors.append("MONGODB_ATLAS_CLUSTER_URI is missing")
    elif not mongodb_uri.startswith(("mongodb://", "mongodb+srv://")):
        errors.append(
            "MONGODB_ATLAS_CLUSTER_URI must start with mongodb:// or mongodb+srv://"
        )

    if errors:
        raise ConfigurationError("; ".join(errors))
