"""K16: `risk.transaction_cost_pips` was read by nothing since fix 28 (costs come from `trading_costs`). The field is gone, so a
config that still sets it stops with a ConfigError instead of looking like a live setting; the tracked config does not set it."""
from pathlib import Path

import pytest

from src.config import Cfg, ConfigError, RiskCfg

ROOT = Path(__file__).resolve().parent.parent


def test_the_risk_block_has_no_unused_cost_field():
    assert not hasattr(RiskCfg(), "transaction_cost_pips")


def test_a_config_that_still_sets_it_stops_with_a_named_error(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("risk:\n  transaction_cost_pips: 1.5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="transaction_cost_pips"):
        Cfg.from_yaml(str(path))


def test_the_tracked_config_still_loads():
    Cfg.from_yaml(str(ROOT / "config.yaml"))
