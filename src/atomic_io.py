"""Writes that a crash cannot leave half done, and a way to set a bad file aside."""
import json
import os
import tempfile
import time


def atomic_write_text(path: str, text: str) -> None:
    """Writes `text` to a temp file in the target's directory, flushes it to disk, then replaces `path` in one step. The directory is
    created when missing. On any error the old file stays as it was and no temp file is left."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=directory)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def atomic_write_json(path: str, obj, **dump_kwargs) -> None:
    """`json.dump(obj, ...)` through `atomic_write_text`; the object is serialised first, so a value JSON cannot hold raises before
    the file is touched."""
    atomic_write_text(path, json.dumps(obj, **dump_kwargs))


def quarantine(path: str):
    """Moves an unreadable file to `<path>.corrupt` (or `.corrupt.<time>` when that exists) and returns the new path, or None when
    there is no such file. Its bytes are kept: the next save must not overwrite the only evidence of what went wrong."""
    if not os.path.exists(path):
        return None
    target = path + ".corrupt"
    if os.path.exists(target):
        target = f"{target}.{int(time.time())}"
    os.replace(path, target)
    return target
