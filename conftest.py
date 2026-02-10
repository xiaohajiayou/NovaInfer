from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
PYTHON_SRC = ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))


def pytest_addoption(parser):
    parser.addoption(
        "--model-path",
        action="store",
        default=os.environ.get("MODEL_PATH", ""),
        help="Local model path used by tests marked with requires_model",
    )


def _resolved_model_path(config) -> str:
    raw = str(config.getoption("--model-path") or "").strip()
    if not raw:
        return ""
    p = Path(raw)
    return str(p) if p.exists() else ""


def pytest_collection_modifyitems(config, items):
    model_path = _resolved_model_path(config)
    if model_path:
        return
    skip_requires_model = pytest.mark.skip(reason="requires --model-path (or MODEL_PATH)")
    for item in items:
        if "requires_model" in item.keywords:
            item.add_marker(skip_requires_model)


@pytest.fixture(scope="session")
def model_path(pytestconfig) -> str:
    resolved = _resolved_model_path(pytestconfig)
    if not resolved:
        pytest.skip("requires --model-path (or MODEL_PATH)")
    return resolved


@pytest.fixture(scope="session")
def require_model_path(model_path: str) -> str:
    return model_path
