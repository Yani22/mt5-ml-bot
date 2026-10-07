"""A typo in `timeframe` used to run on M1 bars (`timeframe_to_mt5_timeframe` fell back to M1) and `Cfg` never checked the value."""
import pytest

import src.data as data_module
from src.config import Cfg, ConfigError
from src.time_utils import timeframe_to_mt5_timeframe, timeframe_to_seconds


def load(tmp_path, timeframe):
    path = tmp_path / "c.yaml"
    path.write_text(f"timeframe: {timeframe}\n")
    return Cfg.from_yaml(str(path))


@pytest.mark.parametrize("tf", ["M1", "M5", "M15", "M30", "H1", "H4", "D1"])
def test_the_supported_timeframes_load(tmp_path, tf):
    assert load(tmp_path, tf).timeframe == tf


@pytest.mark.parametrize("tf", ["M7", "5M", "m5", "H2", "X1"])
def test_an_unknown_timeframe_stops_start_up(tmp_path, tf):
    with pytest.raises(ConfigError, match="timeframe"):
        load(tmp_path, tf)


def test_an_unknown_string_is_an_error_not_m1():
    with pytest.raises(ValueError, match="M7"):
        timeframe_to_mt5_timeframe("M7")


def test_a_month_is_not_read_as_minutes():
    assert timeframe_to_seconds("MN1") == 30 * 24 * 3600
    assert timeframe_to_seconds("M5") == 300


def test_the_uncalled_fetch_bars_is_gone():
    assert not hasattr(data_module, "fetch_bars") and hasattr(data_module, "merge_features_labels")
