from tradeapis.orderlang import OrderLang, OrderIntent, KIND
from decimal import Decimal


def test_stock():
    cmd = "AAPL 100 REL"
    result = OrderIntent(symbol="AAPL", kind=KIND.SHARES, qty=Decimal(100), algo="REL")

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_short():
    cmd = "AAPL -100 REL"
    result = OrderIntent(
        symbol="AAPL", kind=KIND.SHARES, qty=Decimal(100), algo="REL", isLong=False
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit():
    cmd = "AAPL 100 REL @ 33.33"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.SHARES,
        qty=Decimal(100),
        algo="REL",
        limit=Decimal("33.33"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_spaces():
    cmd = "AAPL 100 REL @33.33"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.SHARES,
        qty=Decimal(100),
        algo="REL",
        limit=Decimal("33.33"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_num():
    cmd = "AAPL 100 REL @ 33_33.33"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.SHARES,
        qty=Decimal(100),
        algo="REL",
        limit=Decimal("3333.33"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_down():
    cmd = "AAPL 100 REL @ 33.33 - 4.44"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.SHARES,
        qty=Decimal(100),
        algo="REL",
        limit=Decimal("33.33"),
        bracketLoss=Decimal("4.44"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_down_pct():
    cmd = "AAPL 100 REL @ 33.33 - 7%"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.SHARES,
        qty=Decimal(100),
        algo="REL",
        limit=Decimal("33.33"),
        bracketLoss=Decimal("7"),
        bracketLossIsPercent=True,
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_up():
    cmd = "AAPL 100 rel @ 33.33 + 4.44"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.SHARES,
        qty=Decimal(100),
        algo="REL",
        limit=Decimal("33.33"),
        bracketProfit=Decimal("4.44"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_up_down():
    cmd = "AAPL 100 REL @ 33.33 + 4.44 - 2.22"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.SHARES,
        qty=Decimal(100),
        algo="REL",
        limit=Decimal("33.33"),
        bracketProfit=Decimal("4.44"),
        bracketLoss=Decimal("2.22"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_down_up():
    cmd = "AAPL 100 REL @ 33.33 - 4.44 + 2.22"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.SHARES,
        qty=Decimal(100),
        algo="REL",
        limit=Decimal("33.33"),
        bracketProfit=Decimal("2.22"),
        bracketLoss=Decimal("4.44"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_down_up_algos():
    cmd = "AAPL 100 REL @ 33.33 - 4.44 ABC + 2.22 DEf"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.SHARES,
        qty=Decimal(100),
        algo="REL",
        limit=Decimal("33.33"),
        bracketProfit=Decimal("2.22"),
        bracketLoss=Decimal("4.44"),
        bracketProfitAlgo="DEF",
        bracketLossAlgo="ABC",
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_down_up_bracket_override():
    cmd = "AAPL 100 REL @ 33.33 - 4.44 + 2.22 ± 6"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.SHARES,
        qty=Decimal(100),
        algo="REL",
        limit=Decimal("33.33"),
        bracketProfit=Decimal("6"),
        bracketLoss=Decimal("6"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_down_up_bracket_override_algos():
    cmd = "AAPL 100 REL @ 33.33 - 4.44 + 2.22 ± 6 ABC DEF"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.SHARES,
        qty=Decimal(100),
        algo="REL",
        limit=Decimal("33.33"),
        bracketProfit=Decimal("6"),
        bracketLoss=Decimal("6"),
        bracketProfitAlgo="ABC",
        bracketLossAlgo="DEF",
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_down_up_bracket_override_preview():
    cmd = "AAPL 100 REL @ 33.33 - 4.44 + 2.22 ± 6 preview"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.SHARES,
        qty=Decimal(100),
        algo="REL",
        limit=Decimal("33.33"),
        bracketProfit=Decimal("6"),
        bracketLoss=Decimal("6"),
        preview=True,
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_preview():
    cmd = "AAPL 100 REL preview"
    result = OrderIntent(
        symbol="AAPL", kind=KIND.SHARES, qty=Decimal(100), algo="REL", preview=True
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_preview_limit():
    cmd = "AAPL 100 REL @ 33.33 preview"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.SHARES,
        qty=Decimal(100),
        algo="REL",
        limit=Decimal("33.33"),
        preview=True,
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_preview_limit_flat():
    cmd = "AAPL 1_00 REL @ 33 preview"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.SHARES,
        qty=Decimal(100),
        algo="REL",
        limit=Decimal("33"),
        preview=True,
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_preview_limit_cash():
    cmd = "AAPL $10_000 REL @ 33.33 preview"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.CASH,
        qty=Decimal(10_000),
        algo="REL",
        limit=Decimal("33.33"),
        preview=True,
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_preview_limit_cash_placeholder():
    cmd = ":10 $10_000 REL @ 33.33 preview"
    result = OrderIntent(
        symbol=":10",
        kind=KIND.CASH,
        qty=Decimal(10_000),
        algo="REL",
        limit=Decimal("33.33"),
        preview=True,
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_preview_limit_cash_testr():
    cmd = ":33 10 AS @ 2.50 + 2.50 preview"
    result = OrderIntent(
        symbol=":33",
        kind=KIND.SHARES,
        qty=Decimal(10),
        algo="AS",
        limit=Decimal("2.50"),
        bracketProfit=Decimal("2.50"),
        preview=True,
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_preview_limit_cash_testr222():
    cmd = "NVDA 666 AF @ 96.96"
    result = OrderIntent(
        symbol="NVDA",
        kind=KIND.SHARES,
        qty=Decimal(666),
        algo="AF",
        limit=Decimal("96.96"),
        preview=False,
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_preview_limit_cash_short():
    cmd = "AAPL -$10_000 REL @ 33.33 preview"
    result = OrderIntent(
        symbol="AAPL",
        kind=KIND.CASH,
        qty=Decimal(10_000),
        algo="REL",
        limit=Decimal("33.33"),
        preview=True,
        isLong=False,
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result
