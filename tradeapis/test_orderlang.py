from tradeapis.orderlang import (
    OrderLang,
    OrderIntent,
    DecimalShares,
    DecimalCash,
    DecimalLong,
    DecimalShort,
    DecimalPercent,
    DecimalLongShares,
    DecimalLongCash,
    DecimalShortShares,
    DecimalShortCash,
)

from decimal import (
    Decimal,
)


def test_stock():
    cmd = "AAPL 100 REL"
    result = OrderIntent(symbol="AAPL", qty=DecimalLongShares(100), algo="REL")

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_short():
    cmd = "AAPL -100 REL"
    result = OrderIntent(symbol="AAPL", qty=DecimalLongShares(100), algo="REL")

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit():
    cmd = "AAPL 100 REL @ 33.33"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Decimal("33.33"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_exchange():
    cmd = "AAPL 100 REL ON NASDAQ @ 33.33"
    result = OrderIntent(
        symbol="AAPL",
        exchange="NASDAQ",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Decimal("33.33"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_exchange():
    cmd = "AAPL 100 REL on OVERNIGHT @ 33.33"
    result = OrderIntent(
        symbol="AAPL",
        exchange="OVERNIGHT",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Decimal("33.33"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_spaces():
    cmd = "AAPL 100 REL @33.33"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Decimal("33.33"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_num():
    cmd = "AAPL 100 REL @ 33_33.33"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Decimal("3333.33"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_down():
    cmd = "AAPL 100 REL @ 33.33 - 4.44"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
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
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Decimal("33.33"),
        bracketLoss=DecimalPercent("7"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_up():
    cmd = "AAPL 100 rel @ 33.33 + 4.44"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Decimal("33.33"),
        bracketProfit=DecimalPercent("4.44"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_up_pcttest():
    cmd = "AAPL 100 rel @ 10 + 50% - 20%"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Decimal("10"),
        bracketProfit=DecimalPercent("50"),
        bracketLoss=DecimalPercent("20"),
    )

    ol = OrderLang()
    oi = ol.parse(cmd)
    assert oi == result

    # long profits are HIGHER
    assert oi.bracketProfitReal == Decimal("15")

    # long losses are LOWER
    assert oi.bracketLossReal == Decimal("8")

    assert isinstance(oi.qty, DecimalShares)
    assert isinstance(oi.qty, DecimalLong)
    assert isinstance(oi.qty, DecimalLongShares)


def test_stock_limit_up_pcttest_opposite():
    cmd = "AAPL -100 rel @ 10 + 50% - 20%"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Decimal("10"),
        bracketProfit=DecimalPercent("50"),
        bracketLoss=DecimalPercent("20"),
    )

    ol = OrderLang()
    oi = ol.parse(cmd)
    assert oi == result

    # short profits are LOWER
    assert oi.bracketProfitReal == Decimal("5")

    # short losses are HIGHER
    assert oi.bracketLossReal == Decimal("12")

    assert isinstance(oi.qty, DecimalShares)
    assert isinstance(oi.qty, DecimalShort)
    assert isinstance(oi.qty, DecimalShortShares)


def test_stock_limit_up_down():
    cmd = "AAPL 100 REL @ 33.33 + 4.44 - 2.22"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Decimal("33.33"),
        bracketProfit=Decimal("4.44"),
        bracketLoss=Decimal("2.22"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_down_up():
    cmd = "AAPL -100 REL @ 33.33 - 4.44 + 2.22"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalShortShares(100),
        algo="REL",
        limit=Decimal("33.33"),
        bracketProfit=Decimal("2.22"),
        bracketLoss=Decimal("4.44"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_down_up_no_limit():
    # Test creating order with no limit price but still full bracket details.
    # (Such a setup lets a client back-populate orderintent.limit *itself* with
    #  a dynamically determined future price then still use the bracket profit/loss
    #  encapsulation system to determine the final values for ordering).
    cmd = "AAPL -100 REL @ - 4.44 + 2.22"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalShortShares(100),
        algo="REL",
        limit=None,
        bracketProfit=Decimal("2.22"),
        bracketLoss=Decimal("4.44"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_down_up_algos():
    cmd = "AAPL -100 REL @ 33.33 - 4.44 ABC + 2.22 DEf"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalShortShares(100),
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
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Decimal("33.33"),
        bracketProfit=Decimal("6"),
        bracketLoss=Decimal("6"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_down_up_bracket_single():
    cmd = "AAPL 100 REL @ ± 6"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=None,
        bracketProfit=Decimal("6"),
        bracketLoss=Decimal("6"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_down_up_bracket_override_algos():
    cmd = "AAPL 100 REL @ 33.33 - 4.44 + 2.22 ± 6 ABC DEF"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
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
        qty=DecimalLongShares(100),
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
        symbol="AAPL", qty=DecimalLongShares(100), algo="REL", preview=True
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_preview_limit():
    cmd = "AAPL 100 REL @ 33.33 preview"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
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
        qty=DecimalLongShares(100),
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
        qty=DecimalLongCash(10_000),
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
        qty=DecimalLongCash(10_000),
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
        qty=DecimalLongShares(10),
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
        qty=DecimalLongShares(666),
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
        qty=DecimalShortCash(10_000),
        algo="REL",
        limit=Decimal("33.33"),
        preview=True,
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result
