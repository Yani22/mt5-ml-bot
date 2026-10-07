"""The `features:` block of config.yaml is the untuned setting: with no tuned-params file every path builds its features from it
(they used to build `FeatureCfg()` defaults, so `ema_slow: 200` in the YAML was never applied). A tuned file overrides single keys."""
import os
import re

from src.config import Cfg, FeatureCfg
from src.features import resolve_feature_cfg

ROOT = os.path.join(os.path.dirname(__file__), "..")


def cfg_with(**features):
    cfg = Cfg()
    cfg.features = FeatureCfg(**features)
    return cfg


def test_without_tuned_params_the_configured_features_are_used():
    cfg = cfg_with(ema_slow=200, rsi_period=14, roc_lags=[1, 3, 5])
    got = resolve_feature_cfg(cfg, None)
    assert (got.ema_slow, got.rsi_period, got.roc_lags) == (200, 14, [1, 3, 5])
    assert got is not cfg.features                       # a copy: nobody edits the shared config through it


def test_tuned_keys_override_and_the_rest_stays_configured():
    cfg = cfg_with(ema_slow=200, rsi_period=14)
    got = resolve_feature_cfg(cfg, {"ema_slow": 55, "adx_period": 10})
    assert (got.ema_slow, got.adx_period, got.rsi_period) == (55, 10, 14)


def test_an_empty_tuned_block_changes_nothing():
    cfg = cfg_with(ema_slow=200)
    assert resolve_feature_cfg(cfg, {}) == resolve_feature_cfg(cfg, None) == cfg.features


def test_the_shipped_config_gives_its_slow_ema():
    cfg = Cfg.from_yaml(os.path.join(ROOT, "config.yaml"))
    assert resolve_feature_cfg(cfg, None).ema_slow == cfg.features.ema_slow == 200


def test_no_path_builds_a_default_feature_config_any_more():
    for path in ("main.py", "src/symbol_processor.py", "trainer.py", "backtester.py", "tuner.py"):
        with open(os.path.join(ROOT, path)) as handle:
            calls = [line for line in handle if re.search(r"\bFeatureCfg\(", line) and not line.lstrip().startswith("#")]
        assert not calls, f"{path} still constructs FeatureCfg directly: {calls}"
