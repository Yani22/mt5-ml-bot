"""Stand-in for the MetaTrader5 module on a machine that does not have the package (Linux).

It has no MetaTrader5 import. `src/time_utils.py` and `src/mt5_client.py` bind it as `mt5` when the import fails, so the offline
tools (trainer, backtester, tuner) can import them. Any real use fails loudly: reading a name (`mt5.initialize`,
`mt5.TIMEFRAME_M5`) raises ImportError that names the package. Dunder names raise AttributeError, which is what `copy`,
`inspect` and `unittest.mock` expect when they probe an object.
"""


class MissingMT5:
    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise ImportError(
            f"MetaTrader5.{name} was used, but the MetaTrader5 package is not installed here. It has no Linux wheel: the live "
            "path runs in the Windows Python under Wine (see AUDIT.md).")
