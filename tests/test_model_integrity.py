import pytest

from src.ensemble import Ensemble
from src.model_integrity import MANIFEST, ModelIntegrityError, sign, verify


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("MODEL_SIGNING_KEY", "test-key-1")


@pytest.fixture
def model_dir(tmp_path):
    (tmp_path / "lgbm").mkdir()
    (tmp_path / "lgbm" / "model.lgbm").write_bytes(b"weights")
    (tmp_path / "ensemble_meta.pkl").write_bytes(b"meta")
    return tmp_path


def test_signed_dir_verifies(model_dir):
    sign(str(model_dir))
    verify(str(model_dir))


def test_unsigned_dir_is_refused(model_dir):
    with pytest.raises(ModelIntegrityError, match="unsigned"):
        verify(str(model_dir))


@pytest.mark.parametrize("tamper", ["edit", "add", "delete"])
def test_tampering_is_detected(model_dir, tamper):
    sign(str(model_dir))
    if tamper == "edit":
        (model_dir / "lgbm" / "model.lgbm").write_bytes(b"evil")
    elif tamper == "add":
        (model_dir / "stacker.pkl").write_bytes(b"evil")
    else:
        (model_dir / "ensemble_meta.pkl").unlink()
    with pytest.raises(ModelIntegrityError):
        verify(str(model_dir))


def test_manifest_forged_without_key_is_refused(model_dir, monkeypatch):
    # An attacker can recompute hashes, but cannot produce the HMAC without our key.
    monkeypatch.setenv("MODEL_SIGNING_KEY", "attacker-key")
    sign(str(model_dir))
    monkeypatch.setenv("MODEL_SIGNING_KEY", "test-key-1")
    with pytest.raises(ModelIntegrityError):
        verify(str(model_dir))


def test_ensemble_load_refuses_unsigned_directory(model_dir):
    with pytest.raises(ModelIntegrityError):
        Ensemble.load(str(model_dir), cfg=None)
    assert (model_dir / MANIFEST).exists() is False
