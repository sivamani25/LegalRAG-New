"""M0 smoke tests: package structure, config integrity, and environment file.

All tests are fully offline — no network calls, no model downloads, no API calls.
These tests verify only that the project skeleton is correctly wired:
  - All sub-packages import without error.
  - config.yaml is present, valid YAML, and contains the required top-level keys.
  - .env.example is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# Resolve the project root relative to this test file so tests work regardless
# of the working directory pytest is invoked from.
PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Package import tests
# ---------------------------------------------------------------------------


class TestPackageImports:
    """Verify the src-layout package and all sub-packages are importable."""

    def test_top_level_package_importable(self):
        import legalrag  # noqa: F401

    def test_version_attribute_present(self):
        import legalrag

        assert hasattr(legalrag, "__version__")
        assert isinstance(legalrag.__version__, str)

    def test_ingest_subpackage_importable(self):
        import legalrag.ingest  # noqa: F401

    def test_retrieval_subpackage_importable(self):
        import legalrag.retrieval  # noqa: F401

    def test_generation_subpackage_importable(self):
        import legalrag.generation  # noqa: F401

    def test_eval_subpackage_importable(self):
        import legalrag.eval  # noqa: F401

    def test_app_subpackage_importable(self):
        import legalrag.app  # noqa: F401


# ---------------------------------------------------------------------------
# Config file tests
# ---------------------------------------------------------------------------


class TestConfigFile:
    """Verify config.yaml exists, is valid YAML, and has all required keys."""

    @pytest.fixture(scope="class")
    @classmethod
    def config(cls) -> dict:
        path = PROJECT_ROOT / "config.yaml"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_config_yaml_exists(self):
        assert (PROJECT_ROOT / "config.yaml").is_file(), "config.yaml not found at project root"

    def test_config_yaml_is_valid_yaml(self, config):
        assert isinstance(config, dict), "config.yaml did not parse as a mapping"

    def test_config_has_ingest_key(self, config):
        assert "ingest" in config

    def test_config_has_chunking_key(self, config):
        assert "chunking" in config

    def test_config_has_retrieval_key(self, config):
        assert "retrieval" in config

    def test_config_has_generation_key(self, config):
        assert "generation" in config

    def test_config_has_adaptive_key(self, config):
        assert "adaptive" in config

    def test_config_has_evaluation_key(self, config):
        assert "evaluation" in config

    def test_config_has_output_key(self, config):
        assert "output" in config

    def test_config_has_app_key(self, config):
        assert "app" in config

    def test_generation_provider_is_openrouter(self, config):
        """Generation must use OpenRouter, not the Anthropic SDK."""
        assert config["generation"]["provider"] == "openrouter"

    def test_sac_llm_disabled_by_default(self, config):
        """Default SAC mode must be the offline heuristic (no LLM, no network)."""
        assert config["chunking"]["sac_use_llm"] is False

    def test_embedding_model_is_config_value(self, config):
        """Embedding model must be a non-empty config value (never hardcoded)."""
        model = config["retrieval"]["embedding_model"]
        assert isinstance(model, str) and model.strip()


# ---------------------------------------------------------------------------
# Environment example file test
# ---------------------------------------------------------------------------


class TestEnvExample:
    def test_env_example_exists(self):
        assert (PROJECT_ROOT / ".env.example").is_file(), ".env.example not found"

    def test_env_example_documents_openrouter_key(self):
        text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        assert "OPENROUTER_API_KEY" in text

    def test_env_example_documents_ollama_url(self):
        """Ollama must be documented as optional."""
        text = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        assert "OLLAMA_BASE_URL" in text

def test_config_has_categories(config):
    assert isinstance(config.get("ingest", {}).get("categories"), list)
    assert len(config["ingest"]["categories"]) > 0

def test_config_has_chunking_presets(config):
    assert isinstance(config.get("chunking", {}).get("presets"), dict)
    assert "generic" in config["chunking"]["presets"]
