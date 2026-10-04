"""B9: the order's filling type comes from what the symbol allows (SYMBOL_FILLING_* flags: FOK 1, IOC 2), not a hard-coded IOC.
Preference IOC, then FOK, then RETURN (RETURN is allowed except in market-execution mode). type_filling takes ORDER_FILLING_*
values: FOK 0, IOC 1, RETURN 2."""
from types import SimpleNamespace as NS

import pytest

from test_order_retry import FILLED, FakeClient, make, open_long, result, DONE  # noqa: F401  (fixtures: env, fakes)
from test_order_retry import _env  # noqa: F401


class FillingClient(FakeClient):
    def __init__(self, *a, filling_mode="absent", **k):
        super().__init__(*a, **k)
        self.filling_mode = filling_mode

    def symbol_info(self, symbol):
        info = super().symbol_info(symbol)
        if self.filling_mode != "absent":
            info.filling_mode = self.filling_mode
        return info


def sent_filling(filling_mode):
    client = FillingClient([result(DONE)], deals=FILLED, filling_mode=filling_mode)
    ex, _, _ = make(client)
    open_long(ex)
    (request,) = client.sent
    return request["type_filling"]


@pytest.mark.parametrize("filling_mode, expected", [
    (1, 0),         # FOK only -> ORDER_FILLING_FOK (testing the flag against ORDER_FILLING_IOC would pick IOC)
    (0, 2),         # neither flag -> ORDER_FILLING_RETURN
])
def test_the_filling_type_follows_what_the_symbol_allows(filling_mode, expected):
    assert sent_filling(filling_mode) == expected


@pytest.mark.parametrize("filling_mode, expected", [
    (2, 1),         # IOC only
    (3, 1),         # FOK and IOC: IOC stays the first choice
    ("absent", 1),  # no symbol info field: today's IOC
    (None, 1),
])
def test_ioc_is_used_when_allowed_or_when_the_symbol_does_not_say(filling_mode, expected):
    assert sent_filling(filling_mode) == expected


def test_an_unreadable_symbol_keeps_ioc():
    class NoInfo(FillingClient):
        def symbol_info(self, symbol):
            return None

    client = NoInfo([result(DONE)], deals=FILLED)
    ex, _, _ = make(client)
    open_long(ex)
    assert client.sent[0]["type_filling"] == 1
