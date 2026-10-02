"""Signs saved model directories so a pickled file that was swapped, edited or downloaded from elsewhere is refused.

pickle.load can run arbitrary code, so every model directory gets a MANIFEST.json holding the SHA-256 of each file
plus an HMAC over that list. The HMAC key stays on this machine (env MODEL_SIGNING_KEY, or the git-ignored
.model_signing_key file created on first save), so someone who can only supply model files cannot forge a valid one.

Models trained elsewhere (e.g. Colab) fail verification until you inspect them and run:
    python -m src.model_integrity sign <model_dir>
"""
import hashlib
import hmac
import json
import os
import secrets
import sys

MANIFEST = "MANIFEST.json"
KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".model_signing_key")


class ModelIntegrityError(Exception):
    pass


def _key(create: bool) -> bytes:
    env = os.getenv("MODEL_SIGNING_KEY")
    if env:
        return env.encode()
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            return f.read().strip()
    if not create:
        raise ModelIntegrityError("No signing key found; models cannot be verified. Retrain or sign them locally.")
    key = secrets.token_hex(32).encode()
    fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    return key


def _hash_files(path: str) -> dict:
    files = {}
    for root, _, names in os.walk(path):
        for name in names:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            if rel == MANIFEST:
                continue
            h = hashlib.sha256()
            with open(full, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            files[rel] = h.hexdigest()
    return files


def _mac(key: bytes, files: dict) -> str:
    return hmac.new(key, json.dumps(files, sort_keys=True).encode(), hashlib.sha256).hexdigest()


def sign(path: str) -> None:
    files = _hash_files(path)
    manifest = {"files": files, "hmac": _mac(_key(create=True), files)}
    with open(os.path.join(path, MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def verify(path: str) -> None:
    """Raises ModelIntegrityError unless every file in `path` matches a manifest signed with our key."""
    manifest_path = os.path.join(path, MANIFEST)
    if not os.path.exists(manifest_path):
        raise ModelIntegrityError(f"{path}: no {MANIFEST} (unsigned model; retrain or sign it after inspecting it)")
    with open(manifest_path) as f:
        manifest = json.load(f)
    files = _hash_files(path)
    if not hmac.compare_digest(manifest.get("hmac", ""), _mac(_key(create=False), files)) or manifest.get("files") != files:
        raise ModelIntegrityError(f"{path}: files do not match the signed manifest (modified, added, missing, or signed with another key)")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "sign":
        sign(sys.argv[2])
        print(f"Signed {sys.argv[2]}")
    else:
        print("usage: python -m src.model_integrity sign <model_dir>")
        sys.exit(2)
