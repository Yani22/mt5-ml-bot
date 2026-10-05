"""K10: a dry-run reconcile must not import MetaTrader5 (the package does not exist on Linux). K15: the live start never
invents an account equity; no usable value from the terminal means retry, not start on 100.0."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from src.execution import Execution
from src.live_performance_monitor import equity_from_account


def test_dry_run_reconcile_does_not_import_metatrader5(monkeypatch):
    monkeypatch.setitem(sys.modules, "MetaTrader5", None)  # `import MetaTrader5` now raises ImportError
    ex = object.__new__(Execution)
    ex.dry_run = True
    ex._close_simulated_positions = lambda: []
    assert ex.reconcile_open_positions_with_mt5() == []


def test_a_usable_equity_is_returned_as_a_float():
    assert equity_from_account(NS(equity=130)) == 130.0


@pytest.mark.parametrize("info", [None, NS(), NS(equity=None), NS(equity=0), NS(equity=-5), NS(equity=math.nan),
                                  NS(equity=math.inf), NS(equity="abc")])
def test_no_usable_equity_is_none(info):
    assert equity_from_account(info) is None


def test_main_uses_the_helper_and_has_no_made_up_equity():
    src = (Path(__file__).resolve().parents[1] / "main.py").read_text()
    assert "equity_from_account(account_info)" in src
    assert "else 100.0" not in src
