"""Symbol facts for a backtest that has no MT5 terminal (Linux, csv data): what `RiskManager` reads from `symbol_info`.

Only FX pairs on a USD account can be derived: a USD-quoted pair (EURUSD) is worth `tick_size * contract` per tick; a USD-based
pair (USDJPY) is that divided by the price. Anything else raises, so a backtest never trades on a made-up contract."""
from types import SimpleNamespace

CONTRACT = 100_000


class BacktestSymbolClient:
    def __init__(self, prices=None):
        self.prices = dict(prices or {})

    def update_price(self, symbol, price):
        self.prices[symbol] = float(price)

    def symbol_info(self, symbol):
        name = symbol.rstrip("#")
        if len(name) != 6 or not name.isalpha() or "USD" not in (name[:3], name[3:]):
            raise ValueError(f"{symbol}: cannot derive symbol info without an MT5 terminal (FX pairs with USD only)")
        quote = name[3:]
        digits = 3 if quote == "JPY" else 5
        point = 10.0 ** -digits
        value = point * CONTRACT
        if quote != "USD":  # USD is the base: the quote currency is converted back with the price
            price = self.prices.get(symbol)
            if not price or price <= 0:
                raise ValueError(f"{symbol}: no price yet to convert the tick value (call update_price first)")
            value = value / price
        return SimpleNamespace(point=point, digits=digits, trade_tick_size=point, trade_tick_value=value,
                               trade_contract_size=CONTRACT, volume_min=0.01, volume_step=0.01, volume_max=100.0,
                               trade_stops_level=0)
