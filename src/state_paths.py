"""Keeps dry-run learning state apart from live learning state."""
import os

DRY_RUN_SUFFIX = "_dryrun"


def mode_state_path(path: str, dry_run: bool) -> str:
    """Live keeps the configured path; dry-run gets a `_dryrun` sibling (idempotent)."""
    if not dry_run:
        return path
    root, ext = os.path.splitext(path)
    if root.endswith(DRY_RUN_SUFFIX):
        return path
    return f"{root}{DRY_RUN_SUFFIX}{ext}"


def apply_mode_state_paths(cfg, dry_run: bool) -> None:
    """Points the bandit and monitor state files at the files for this mode."""
    ts = cfg.thompson_sampling
    ts.state_file = mode_state_path(ts.state_file, dry_run)
    mon = cfg.monitoring
    mon.monitor_state_file = mode_state_path(mon.monitor_state_file, dry_run)
