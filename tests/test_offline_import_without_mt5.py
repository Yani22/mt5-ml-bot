"""E: the offline tools (`trainer.py`, `backtester.py`, `tuner.py`) must import on a machine without the MetaTrader5 package
(Linux). `src/time_utils.py` and `src/mt5_client.py` used to import it unguarded. Every check runs in a subprocess because the
conftest stub would hide the missing package inside pytest. A call that really needs the terminal must still fail loudly, with
a message that names the package, not turn into a silent None or default."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NO_MT5 = "import sys; sys.modules['MetaTrader5'] = None\n"


def run(code):
    return subprocess.run([sys.executable, "-c", NO_MT5 + code], cwd=ROOT, capture_output=True, text=True, timeout=120)


def test_the_time_and_client_modules_import_without_the_package():
    out = run("import src.time_utils, src.mt5_client, src.utils, src.data_manager; print('ok')")
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-600:]


def test_string_timeframes_still_convert_without_the_package():
    out = run("from src.time_utils import timeframe_to_seconds as t; print(t('M5'), t('H1'), t('D1'))")
    assert out.returncode == 0 and out.stdout.split() == ["300", "3600", "86400"], out.stderr[-600:]


def test_an_mt5_constant_without_the_package_raises_import_error_naming_it():
    out = run("from src.time_utils import timeframe_to_mt5_timeframe as f\n"
              "try:\n    f('M5')\nexcept ImportError as e:\n    print('IE', 'MetaTrader5' in str(e))\n")
    assert out.returncode == 0 and out.stdout.split() == ["IE", "True"], out.stdout + out.stderr[-600:]


def test_an_integer_timeframe_without_the_package_raises_import_error_not_a_wrong_answer():
    out = run("from src.time_utils import timeframe_to_seconds as t\n"
              "try:\n    t(5)\nexcept ImportError:\n    print('IE')\n")
    assert out.returncode == 0 and out.stdout.strip() == "IE", out.stdout + out.stderr[-600:]


def test_client_calls_and_constants_without_the_package_raise_import_error():
    out = run("from src.mt5_client import MT5Client\n"
              "import src.mt5_client as m\n"
              "for f in (lambda: m.mt5.initialize(), lambda: MT5Client.ORDER_TYPE_BUY):\n"
              "    try:\n        f()\n    except ImportError as e:\n        print('IE', 'MetaTrader5' in str(e))\n")
    assert out.returncode == 0 and out.stdout.split() == ["IE", "True", "IE", "True"], out.stdout + out.stderr[-600:]


def test_a_dunder_lookup_on_the_missing_package_is_an_attribute_error():
    """copy, inspect and mock probe dunder names with a default; only AttributeError lets them carry on."""
    out = run("import src.mt5_client as m, copy\n"
              "print(hasattr(m.mt5, '__wrapped__'), copy.copy(m.mt5) is not None)")
    assert out.returncode == 0 and out.stdout.split() == ["False", "True"], out.stdout + out.stderr[-600:]


def test_the_offline_tools_import_without_the_package():
    out = run("import trainer, backtester, tuner; print('ok')")
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-600:]
