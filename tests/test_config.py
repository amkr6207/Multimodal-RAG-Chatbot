import pytest

from config import ConfigurationError, validate_environment


def test_validate_environment_accepts_required_values():
    validate_environment(
        {
            "GROQ_API_KEY": "test-groq-key",
            "MONGODB_ATLAS_CLUSTER_URI": "mongodb+srv://user:password@example.net/",
        }
    )


def test_validate_environment_reports_all_missing_values():
    with pytest.raises(ConfigurationError) as error:
        validate_environment({})

    message = str(error.value)
    assert "GROQ_API_KEY is missing" in message
    assert "MONGODB_ATLAS_CLUSTER_URI is missing" in message


def test_validate_environment_rejects_placeholder_key():
    with pytest.raises(ConfigurationError, match="GROQ_API_KEY is missing"):
        validate_environment(
            {
                "GROQ_API_KEY": "your_groq_api_key_here",
                "MONGODB_ATLAS_CLUSTER_URI": "mongodb://localhost:27017",
            }
        )


def test_validate_environment_rejects_invalid_mongodb_uri():
    with pytest.raises(ConfigurationError, match="must start with"):
        validate_environment(
            {
                "GROQ_API_KEY": "test-groq-key",
                "MONGODB_ATLAS_CLUSTER_URI": "https://example.net/database",
            }
        )
