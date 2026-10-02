from src.config import Cfg
from src.ensemble import Ensemble


def _lgbm_params(ens):
    return ens.members["lgbm"]._pipe.named_steps["clf"].get_params()


def test_untuned_model_uses_config_defaults_not_range_minimums():
    cfg = Cfg.from_yaml("config.yaml")
    params = _lgbm_params(Ensemble(cfg))
    assert params["n_estimators"] == 200 and params["learning_rate"] == 0.05
    assert params["subsample"] == 0.8 and params["colsample_bytree"] == 0.8


def test_tuned_params_override_defaults():
    cfg = Cfg.from_yaml("config.yaml")
    params = _lgbm_params(Ensemble(cfg, model_params={"lgbm": {"n_estimators": 321}}))
    assert params["n_estimators"] == 321 and params["learning_rate"] == 0.05


def test_tune_ranges_are_valid_and_separate_from_defaults():
    cfg = Cfg.from_yaml("config.yaml")
    for m in cfg.models:
        assert set(m["defaults"]) <= set(m["tune"]), "every default should have a tuning range"
        for k, v in m["defaults"].items():
            assert not isinstance(v, list), f"default {k} must be a single value"
