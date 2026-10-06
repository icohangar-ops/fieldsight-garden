"""render.yaml matches the FieldSight deploy contract and the Render schema."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_URL = "https://render.com/schema/render.yaml.json"


def load_blueprint() -> dict:
    return yaml.safe_load((ROOT / "render.yaml").read_text())


def test_blueprint_contract():
    doc = load_blueprint()
    service = doc["services"][0]
    assert service["type"] == "web"
    assert service["runtime"] == "python"
    assert service["plan"] == "starter"
    assert service["healthCheckPath"] == "/health"
    assert service["branch"] == "main"
    env = {item["key"]: item for item in service["envVars"] if "key" in item}
    assert env["TINKER_API_KEY"]["sync"] is False
    assert "value" not in env["TINKER_API_KEY"]
    assert env["FIELDSIGHT_MODEL_PATH"]["value"].startswith("tinker://")
    assert "DATABASE_URL" in env
    database = doc["databases"][0]
    assert database["plan"] in {"free", "basic-256mb", "basic-1gb"}
    text = (ROOT / "render.yaml").read_text()
    assert "sk-" not in text
    assert "TINKER_API_KEY" in text


def test_blueprint_validates_against_render_schema():
    jsonschema = pytest_jsonschema()
    with urlopen(SCHEMA_URL, timeout=30) as response:
        schema = json.load(response)
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(load_blueprint()), key=lambda err: list(err.path))
    assert errors == [], "\n".join(err.message for err in errors)


def pytest_jsonschema():
    import jsonschema

    return jsonschema
