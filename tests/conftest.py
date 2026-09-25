import pytest
import yaml

@pytest.fixture
def config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
