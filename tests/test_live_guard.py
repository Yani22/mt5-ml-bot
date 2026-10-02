"""The legacy live bot must not place real orders unless live trading was explicitly switched on."""
import ast
import importlib
import os

import pytest

MAIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py")


def _guard():
    return importlib.import_module("src.live_guard")


def _run_def():
    tree = ast.parse(open(MAIN).read())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")


def test_dry_run_needs_no_permission():
    assert _guard().require_live_permission(True, env={}) is None


@pytest.mark.parametrize("value", [None, "", "0", "no", "true", "yes", " 1", "11"])
def test_live_refused_unless_flag_is_exactly_1(value):
    g = _guard()
    env = {} if value is None else {"ALLOW_LIVE_TRADING": value}
    with pytest.raises(g.LiveTradingNotAllowed, match="ALLOW_LIVE_TRADING"):
        g.require_live_permission(False, env=env)


def test_live_allowed_with_explicit_flag():
    assert _guard().require_live_permission(False, env={"ALLOW_LIVE_TRADING": "1"}) is None


def test_guard_reads_the_process_environment_by_default(monkeypatch):
    g = _guard()
    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    with pytest.raises(g.LiveTradingNotAllowed):
        g.require_live_permission(False)
    monkeypatch.setenv("ALLOW_LIVE_TRADING", "1")
    assert g.require_live_permission(False) is None


def test_main_run_defaults_to_dry_run():
    args = _run_def().args
    names = [a.arg for a in args.args]
    default = args.defaults[names.index("dry_run") - (len(names) - len(args.defaults))]
    assert isinstance(default, ast.Constant) and default.value is True, "main.run() must default to dry_run=True"


def test_main_run_checks_permission_before_anything_else():
    body = [n for n in _run_def().body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]  # skip docstring
    first = body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call), "first statement of run() must be the guard call"
    call = first.value
    assert getattr(call.func, "id", getattr(call.func, "attr", None)) == "require_live_permission"
    assert [a.id for a in call.args] == ["dry_run"]
