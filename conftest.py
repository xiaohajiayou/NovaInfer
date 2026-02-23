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
    parser.addoption(
        "--device",
        action="store",
        default=os.environ.get("TEST_DEVICE", "all"),
        choices=["all", "cpu", "nvidia"],
        help="Filter parametrized tests by device axis",
    )
    parser.addoption(
        "--layout",
        action="store",
        default=os.environ.get("TEST_LAYOUT", "all"),
        choices=["all", "slot", "block"],
        help="Filter parametrized tests by KV layout axis",
    )
    parser.addoption(
        "--backend",
        action="store",
        default=os.environ.get("TEST_BACKEND", "all"),
        choices=["all", "native", "cudnn"],
        help="Filter parametrized tests by backend axis",
    )


def _resolved_model_path(config) -> str:
    raw = str(config.getoption("--model-path") or "").strip()
    if not raw:
        return ""
    p = Path(raw)
    return str(p) if p.exists() else ""


def pytest_collection_modifyitems(config, items):
    model_path = _resolved_model_path(config)
    selected_device = str(config.getoption("--device") or "all").strip().lower()
    selected_layout = str(config.getoption("--layout") or "all").strip().lower()
    selected_backend = str(config.getoption("--backend") or "all").strip().lower()

    skip_requires_model = pytest.mark.skip(reason="requires --model-path (or MODEL_PATH)")
    skip_by_axis = pytest.mark.skip(reason="filtered out by --device/--layout/--backend")

    for item in items:
        if not model_path and "requires_model" in item.keywords:
            item.add_marker(skip_requires_model)

        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec is not None else {}

        param_device = params.get("ll_device", params.get("device"))
        param_layout = params.get("kv_layout", params.get("layout"))
        param_backend = params.get("backend")

        marker_device = item.get_closest_marker("test_device")
        marker_layout = item.get_closest_marker("test_layout")
        marker_backend = item.get_closest_marker("test_backend")

        if param_device is None and marker_device and marker_device.args:
            param_device = marker_device.args[0]
        if param_layout is None and marker_layout and marker_layout.args:
            param_layout = marker_layout.args[0]
        if param_backend is None and marker_backend and marker_backend.args:
            param_backend = marker_backend.args[0]

        if selected_device != "all" and param_device is not None:
            if str(param_device).strip().lower() != selected_device:
                item.add_marker(skip_by_axis)
                continue

        if selected_layout != "all" and param_layout is not None:
            if str(param_layout).strip().lower() != selected_layout:
                item.add_marker(skip_by_axis)
                continue

        if selected_backend != "all" and param_backend is not None:
            if str(param_backend).strip().lower() != selected_backend:
                item.add_marker(skip_by_axis)
                continue


@pytest.fixture(scope="session")
def model_path(pytestconfig) -> str:
    resolved = _resolved_model_path(pytestconfig)
    if not resolved:
        pytest.skip("requires --model-path (or MODEL_PATH)")
    return resolved


@pytest.fixture(scope="session")
def require_model_path(model_path: str) -> str:
    return model_path
