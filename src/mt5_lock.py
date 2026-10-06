"""One lock for every call into the MetaTrader5 package in this process.

The package keeps one connection per process, and `last_error()` is process-wide, so a call from another thread can overwrite
the error of a failed one. Symbol threads, the 5 s reconcile loop and the bar fetches all call it, so each `mt5.*` call is made
with this lock held (B14). It has no MetaTrader5 import: the offline tools load modules that take it on Linux.

Rules: a leaf lock. Only the `mt5.*` call (and the `last_error()` read that belongs to it) runs inside; never a sleep, a loop,
a notifier call or another lock. Allowed order: `_state_lock` then this lock, `cache_lock` then this lock; never the reverse.
A hung call stalls every thread that needs MT5 until it returns."""
import threading

api_lock = threading.RLock()
