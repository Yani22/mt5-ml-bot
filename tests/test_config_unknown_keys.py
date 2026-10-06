"""An unknown key in a config block used to make `Cfg.from_yaml` log one warning and reset the WHOLE block to its defaults:
`risk: {risk_per_trad: 0.01}` quietly ran every other risk setting on defaults too. A bot that places orders must not start on a
config it did not understand, so an unknown key now raises `ConfigError` naming the block, the key and the closest known key."""
from pathlib import Path

import pytest

from src.config import Cfg, ConfigError, RiskCfg, WatchdogCfg

ROOT = Path(__file__).resolve().parent.parent


def load(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return Cfg.from_yaml(str(path))


def test_an_unknown_risk_key_stops_the_load_and_names_the_block_the_key_and_the_likely_fix(tmp_path):
    with pytest.raises(ConfigError) as err:
        load(tmp_path, "risk:\n  risk_per_trad: 0.01\n  max_portfolio_risk: 0.03\n")
    message = str(err.value)
    assert "risk" in message and "risk_per_trad" in message and "risk_per_trade" in message


@pytest.mark.parametrize("block, name", [
    ("thompson_sampling:\n  enabeld: true\n", "thompson_sampling"),
    ("watchdog:\n  enabeld: true\n", "watchdog"),
    ("fetch:\n  initial_fetch_bar: 100\n", "fetch"),
    ("monitoring:\n  telegram_tokn: x\n", "monitoring"),
    ("backtesting:\n  initial_equty: 100\n", "backtesting"),
    ("asymmetric_compounding:\n  enabeld: true\n", "asymmetric_compounding"),
    ("context_features:\n  mta:\n    timefrane: H1\n", "context_features.mta"),
    ("trading_costs:\n  defaults:\n    spred_pips: 1.0\n", "trading_costs.defaults"),
    ("trading_costs:\n  defualts: {}\n", "trading_costs"),
])
def test_every_block_refuses_an_unknown_key(tmp_path, block, name):
    with pytest.raises(ConfigError, match=name.replace(".", r"\.")):
        load(tmp_path, block)


def test_a_known_key_still_applies(tmp_path):
    cfg = load(tmp_path, "risk:\n  risk_per_trade: 0.01\n")
    assert cfg.risk.risk_per_trade == 0.01


def test_a_missing_block_still_gets_its_defaults(tmp_path):
    cfg = load(tmp_path, "symbols: [EURUSD]\n")
    assert cfg.risk == RiskCfg() and cfg.watchdog == WatchdogCfg()


def test_the_real_config_yaml_loads_without_an_error():
    """Pins the user's actual file: a block that fell back to defaults silently before would now stop the bot."""
    Cfg.from_yaml(str(ROOT / "config.yaml"))
