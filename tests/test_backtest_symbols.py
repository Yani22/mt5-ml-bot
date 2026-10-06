import pytest

from src.backtest_symbols import BacktestSymbolClient


def test_a_usd_quoted_pair_has_five_digits_and_a_tick_value_of_one_dollar_per_point():
    info = BacktestSymbolClient().symbol_info("EURUSD#")
    assert (info.digits, info.point, info.trade_contract_size) == (5, 1e-5, 100_000)
    assert info.trade_tick_value == pytest.approx(1.0)
    assert (info.volume_min, info.volume_step, info.volume_max) == (0.01, 0.01, 100.0)


def test_a_usd_base_pair_converts_the_quote_currency_with_the_price():
    client = BacktestSymbolClient({"USDJPY#": 150.0})
    info = client.symbol_info("USDJPY#")
    assert info.digits == 3 and info.point == 1e-3
    assert info.trade_tick_value == pytest.approx(1e-3 * 100_000 / 150.0)
    client.update_price("USDJPY#", 100.0)
    assert client.symbol_info("USDJPY#").trade_tick_value == pytest.approx(1.0)


def test_a_usd_base_pair_without_a_price_is_an_error():
    with pytest.raises(ValueError, match="USDJPY"):
        BacktestSymbolClient().symbol_info("USDJPY#")


@pytest.mark.parametrize("symbol", ["GOLD#", "EURGBP#", "US500Cash"])
def test_a_symbol_it_cannot_derive_is_an_error_naming_it(symbol):
    with pytest.raises(ValueError, match=symbol.replace("#", "")):
        BacktestSymbolClient().symbol_info(symbol)
