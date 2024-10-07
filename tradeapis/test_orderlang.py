from decimal import Decimal

import pytest

from tradeapis.buylang import Order, OrderRequest, Side

from tradeapis.orderlang import (
    Calculation,
    DecimalCash,
    DecimalLong,
    DecimalLongCash,
    DecimalLongShares,
    DecimalPercent,
    DecimalPrice,
    DecimalShares,
    DecimalShort,
    DecimalShortCash,
    DecimalShortShares,
    OrderIntent,
    OrderLang,
)


def test_stock():
    cmd = "AAPL 100 REL"
    result = OrderIntent(symbol="AAPL", qty=DecimalLongShares(100), algo="REL")

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_by_position():
    cmd = ":13 100 REL"
    result = OrderIntent(symbol=":13", qty=DecimalLongShares(100), algo="REL")

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_by_negative_position():
    cmd = ":-1 100 REL"
    result = OrderIntent(symbol=":-1", qty=DecimalLongShares(100), algo="REL")

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted():
    cmd = "'AAPL' 100 REL"
    result = OrderIntent(symbol="AAPL", qty=DecimalLongShares(100), algo="REL")

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes():
    cmd = '"AAPL" 100 REL'
    result = OrderIntent(symbol="AAPL", qty=DecimalLongShares(100), algo="REL")

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_zero_price_is_zero():
    cmd = '"AAPL" 100 REL @ 0'
    result = OrderIntent(
        symbol="AAPL", qty=DecimalLongShares(100), algo="REL", limit=Decimal(0)
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_zero_price_is_zero_also_no_qty():
    cmd = '"AAPL" all REL @ 0'
    result = OrderIntent(symbol="AAPL", qty=None, algo="REL", limit=Decimal(0))

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_zero_price_is_zero_also_no_qty_preview():
    cmd = '"AAPL" all REL preview'
    result = OrderIntent(symbol="AAPL", qty=None, algo="REL", preview=True)

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_zero_price_is_zero_also_no_qty_preview_again():
    cmd = "MSFT241220C00500000 all REL preview"
    result = OrderIntent(
        symbol="MSFT241220C00500000", qty=None, algo="REL", preview=True
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_zero_price_is_zero_config():
    cmd = '"AAPL" 100 REL @ 0 conf red=blue sad happy start=vwap2'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Decimal(0),
        config=dict(red="blue", sad=True, happy=True, start="vwap2"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config():
    cmd = '"AAPL" 100 REL conf red=blue sad happy start=vwap2'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        config=dict(red="blue", sad=True, happy=True, start="vwap2"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_with_price_before():
    cmd = '"AAPL" 100 REL @ 3.33 conf red=blue sad happy start=vwap2'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        limit=Decimal("3.33"),
        algo="REL",
        config=dict(red="blue", sad=True, happy=True, start="vwap2"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_with_price_before_false():
    cmd = '"AAPL" 100 REL @ 3.33 conf red=blue sad=off happy=no start=vwap2'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        limit=Decimal("3.33"),
        algo="REL",
        config=dict(red="blue", sad=False, happy=False, start="vwap2"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_with_price_before_false_reset():
    cmd = '"AAPL" 100 REL @ 3.33 conf red=blue sad=off happy=no start=vwap2 @ 3.33 conf happy=yes sad'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        limit=Decimal("3.33"),
        algo="REL",
        config=dict(red="blue", sad=True, happy=True, start="vwap2"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_with_price_before_false_reset_inner():
    cmd = (
        '"AAPL" 100 REL @ 3.33 conf red=blue sad=off happy=no start=vwap2 happy=yes sad'
    )
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        limit=Decimal("3.33"),
        algo="REL",
        config=dict(red="blue", sad=True, happy=True, start="vwap2"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_with_price_before_false_reset_inner_single():
    cmd = '"AAPL" 100 REL @ 3.33 conf red=blue sad=off happy=no start=vwap2 happy=yes sad b'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        limit=Decimal("3.33"),
        algo="REL",
        config=dict(red="blue", sad=True, happy=True, start="vwap2", b=True),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_with_price_before_false_reset_inner_single_none():
    cmd = '"AAPL" 100 REL @ 3.33 conf red=blue sad=off happy=no start=vwap2 happy=yes sad b=none'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        limit=Decimal("3.33"),
        algo="REL",
        config=dict(red="blue", sad=True, happy=True, start="vwap2", b=None),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_with_price_before_dual_config():
    # Demo of overwriting price and config options by just adding more of them
    cmd = '"AAPL" 100 REL @ 3.33 conf red=blue sad happy start=vwap2 @ 2.22 conf red=green'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        limit=Decimal("2.22"),
        algo="REL",
        config=dict(red="green", sad=True, happy=True, start="vwap2"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_with_price_before_preview_after():
    cmd = '"AAPL" 100 REL @ 3.33 conf red=blue sad happy start=vwap2 preview'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        limit=Decimal("3.33"),
        algo="REL",
        preview=True,
        config=dict(red="blue", sad=True, happy=True, start="vwap2"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_with_price_after():
    cmd = '"AAPL" 100 REL conf red=blue sad happy start=vwap2 @ 3.33'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        limit=Decimal("3.33"),
        algo="REL",
        config=dict(red="blue", sad=True, happy=True, start="vwap2"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_with_price_after_preview_after():
    cmd = '"AAPL" 100 REL conf red=blue sad happy start=vwap2 @ 3.33 preview'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        limit=Decimal("3.33"),
        algo="REL",
        preview=True,
        config=dict(red="blue", sad=True, happy=True, start="vwap2"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_with_price_after_preview_after():
    cmd = '"AAPL" 100 REL conf red=blue sad happy start=vwap2 preview @ 3.33'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        limit=Decimal("3.33"),
        algo="REL",
        preview=True,
        config=dict(red="blue", sad=True, happy=True, start="vwap2"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_with_price_after_preview_before_before():
    cmd = '"AAPL" 100 REL preview conf red=blue sad happy start=vwap2 @ 3.33'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        limit=Decimal("3.33"),
        algo="REL",
        preview=True,
        config=dict(red="blue", sad=True, happy=True, start="vwap2"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_spaces_stuff():
    cmd = '"SELL 1 650581442" -13.0 LIM @ 5.823 preview'
    result = OrderIntent(
        symbol="SELL 1 650581442",
        qty=DecimalShortShares(13),
        algo="LIM",
        limit=Decimal("5.823"),
        preview=True,
        spread=OrderRequest(
            orders=[
                Order(side=Side.SELL, multiplier=1, symbol="650581442", limit=None),
            ]
        ),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_spaces_stuff_not_end():
    cmd = '"SELL 1 650581442" -13.0 LIM @ 5.823 preview config again=\'more str more\' extra="big string" evenmore=stuff'
    result = OrderIntent(
        symbol="SELL 1 650581442",
        qty=DecimalShortShares(13),
        algo="LIM",
        limit=Decimal("5.823"),
        preview=True,
        config=dict(extra="big string", again="more str more", evenmore="stuff"),
        spread=OrderRequest(
            orders=[
                Order(side=Side.SELL, multiplier=1, symbol="650581442", limit=None),
            ]
        ),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result

    # print(ol.parse(cmd))


def test_stock_quoted_quotes_spaces_stuff_not_end_becomes_order_spread():
    cmd = '"SELL 1 650581442 BUY 1 AAPL" -13.0 LIM @ 5.823 preview config again=\'more str more\' extra="big string" evenmore=stuff'
    result = OrderIntent(
        symbol="SELL 1 650581442 BUY 1 AAPL",
        qty=DecimalShortShares(13),
        algo="LIM",
        limit=Decimal("5.823"),
        preview=True,
        spread=OrderRequest(
            orders=[
                Order(side=Side.SELL, multiplier=1, symbol="650581442", limit=None),
                Order(side=Side.BUY, multiplier=1, symbol="AAPL", limit=None),
            ]
        ),
        config=dict(extra="big string", again="more str more", evenmore="stuff"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result

    # print(ol.parse(cmd))


def test_stock_quoted_quotes_spaces_stuff():
    cmd = '"SELL 1 650581442" -13.0 LIM @ 5.823 preview config extra="big string" again=\'more str more\''
    result = OrderIntent(
        symbol="SELL 1 650581442",
        qty=DecimalShortShares(13),
        algo="LIM",
        limit=Decimal("5.823"),
        preview=True,
        config=dict(extra="big string", again="more str more"),
        spread=OrderRequest(
            orders=[
                Order(side=Side.SELL, multiplier=1, symbol="650581442", limit=None),
            ]
        ),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_non_alpha():
    cmd = '"AAPL" 100 REL conf red=blue sad happy start=vwap2 aux=3.33 boom=%^&'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        config=dict(
            red="blue",
            sad=True,
            happy=True,
            start="vwap2",
            aux=Decimal("3.33"),
            boom="%^&",
        ),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_non_alpha_other_bad():
    cmd = '"AAPL" 100 REL conf red=blue sad happy start=2vwap2 aux=MORE boom=%^&'

    # verify key/value of `start=2vwap2` is a hard error.
    # (we have to exclude non-numeric values starting with numbers because the number parser
    #  would consume '2', generate an output of Decimal("2") then break the rest of the
    #  string into additional key/value parts)
    with pytest.raises(Exception):
        result = OrderIntent(
            symbol="AAPL",
            qty=DecimalLongShares(100),
            algo="REL",
            config=dict(
                red="blue", sad=True, happy=True, start="2vwap2", aux="MORE", boom="%^&"
            ),
        )

        ol = OrderLang()
        assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_non_alpha_other_good():
    cmd = '"AAPL" 100 REL conf red=blue sad happy start=vwap2 aux=MORE boom=%^&'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        config=dict(
            red="blue", sad=True, happy=True, start="vwap2", aux="MORE", boom="%^&"
        ),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_non_alpha_price():
    cmd = '"AAPL" 100 REL conf red=blue sad happy start=vwap2 aux = 3.33 boom=%^&'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        config=dict(
            red="blue",
            sad=True,
            happy=True,
            start="vwap2",
            aux=Decimal("3.33"),
            boom="%^&",
        ),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_non_alpha_price_negative():
    cmd = '"AAPL" 100 REL config red=blue sad happy start=vwap2 aux = -3.33 boom=%^&'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        config=dict(
            red="blue",
            sad=True,
            happy=True,
            start="vwap2",
            aux=Decimal("-3.33"),
            boom="%^&",
        ),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_non_alpha_price_calc():
    cmd = (
        '"AAPL" 100 REL c red=blue sad HAPPY start=vwap2 aux=(/ 1.2 3.4 4.5) boom=$%^&'
    )
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        config=dict(
            red="blue",
            sad=True,
            happy=True,
            start="vwap2",
            aux=Calculation("(/ 1.2 3.4 4.5)"),
            boom="$%^&",
        ),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_weirder():
    cmd = '"AAPL" 100 REL conf red=blue sad happy start=vwap2 aux=3.33 boom=$%^& extra34_33=99.3.312.|^&'

    # Same parser issue here: we have to verify values that aren't numbers don't start with numbers so
    # the number parser doesn't consume them incorrectly.
    with pytest.raises(Exception):
        result = OrderIntent(
            symbol="AAPL",
            qty=DecimalLongShares(100),
            algo="REL",
            config=dict(
                red="blue",
                sad=True,
                happy=True,
                start="vwap2",
                aux=Decimal("3.33"),
                boom="$%^&",
                extra34_33="99.3.312.|^&",
            ),
        )

        ol = OrderLang()
        assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_weirder_reordered_bad():
    cmd = '"AAPL" 100 REL conf red=blue sad boom=%^& extra34_33=99.3.312.|^& happy start=vwap2 aux=3.33'

    with pytest.raises(Exception):
        result = OrderIntent(
            symbol="AAPL",
            qty=DecimalLongShares(100),
            algo="REL",
            config=dict(
                red="blue",
                sad=True,
                happy=True,
                start="vwap2",
                aux=Decimal("3.33"),
                boom="%^&",
                extra34_33="99.3.312.|^&",
            ),
        )

        ol = OrderLang()
        assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_weirder_reordered_good():
    cmd = '"AAPL" 100 REL conf red=blue sad boom=%^& extra34_33=99.3 happy start=vwap2 aux=3.33'

    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        config=dict(
            red="blue",
            sad=True,
            happy=True,
            start="vwap2",
            aux=Decimal("3.33"),
            boom="%^&",
            extra34_33=Decimal("99.3"),
        ),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_no_price_config_weirder_dashes():
    cmd = '"AAPL" 100 REL conf red=blue s-ad bo-om=%^& extra34_33=99.3.312.|^& ha-ppy start=-vwap2 aux=-3.3-3'
    with pytest.raises(Exception):
        result = OrderIntent(
            symbol="AAPL",
            qty=DecimalLongShares(100),
            algo="REL",
            config={
                "red": "blue",
                "s-ad": True,
                "bo-om": "%^&",
                "extra34_33": "99.3.312.|^&",
                "ha-ppy": True,
                "start": "-vwap2",
                "aux": "-3.3-3",
            },
        )

        ol = OrderLang()
        assert ol.parse(cmd) == result


def test_stock_quoted_quotes_and_calculator():
    cmd = '"AAPL" 100 REL @ (/ 100 3)'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Calculation("(/ 100 3)"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_and_calculator_arbitrary():
    cmd = '"AAPL" 100 REL @ (/ (+ live 100) (* :BP3 3))'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Calculation("(/ (+ live 100) (* :BP3 3))"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_quotes_and_calculator_nested():
    cmd = '"AAPL" 100 REL @ (/ (* 100 7 (* (/ 5 4) 3) (/ 3_000 (* 4 2))))'
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Calculation("(/ (* 100 7 (* (/ 5 4) 3) (/ 3_000 (* 4 2))))"),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_big():
    cmd = "'buy 100 AAPL sell 100 TSLA' 1 REL"
    result = OrderIntent(
        symbol="BUY 100 AAPL SELL 100 TSLA",
        qty=DecimalLongShares(1),
        algo="REL",
        spread=OrderRequest(
            orders=[
                Order(side=Side.BUY, multiplier=100, symbol="AAPL", limit=None),
                Order(side=Side.SELL, multiplier=100, symbol="TSLA", limit=None),
            ]
        ),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_big_quotes():
    cmd = '"buy 100 AAPL sell 100 TSLA" 1 REL'
    result = OrderIntent(
        symbol="BUY 100 AAPL SELL 100 TSLA",
        qty=DecimalLongShares(1),
        algo="REL",
        spread=OrderRequest(
            orders=[
                Order(side=Side.BUY, multiplier=100, symbol="AAPL", limit=None),
                Order(side=Side.SELL, multiplier=100, symbol="TSLA", limit=None),
            ]
        ),
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_quoted_space():
    cmd = "'BRK B' 100 REL"
    result = OrderIntent(symbol="BRK B", qty=DecimalLongShares(100), algo="REL")

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
    cmd = "AAPL -100 REL @ live - 4.44 + 2.22"
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
    cmd = "AAPL 100 REL @ live ± 6"
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


def test_stock_limit_down_up_bracket_single():
    cmd = "AAPL 100 REL @ credit -6"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Decimal("-6"),
        bracketProfit=None,
        bracketLoss=None,
    )

    ol = OrderLang()
    assert ol.parse(cmd) == result


def test_stock_limit_down_up_bracket_single_bracket():
    cmd = "AAPL 100 REL @ credit -6 + 3 - 1"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalLongShares(100),
        algo="REL",
        limit=Decimal("-6"),
        bracketProfit=Decimal("3"),
        bracketLoss=Decimal("1"),
    )

    ol = OrderLang()

    oi = ol.parse(cmd)
    assert oi == result

    assert oi.bracketProfitReal == Decimal("3")
    assert oi.bracketLossReal == Decimal("7")


def test_stock_limit_down_up_bracket_single_bracket_double_short():
    cmd = "AAPL -100 REL @ credit -6 + 3 - 1"
    result = OrderIntent(
        symbol="AAPL",
        qty=DecimalShortShares(100),
        algo="REL",
        limit=Decimal("-6"),
        bracketProfit=Decimal("3"),
        bracketLoss=Decimal("1"),
    )

    ol = OrderLang()

    oi = ol.parse(cmd)
    assert oi == result

    assert oi.bracketProfitReal == Decimal("3")
    assert oi.bracketLossReal == Decimal("7")


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


def test_stock_limit_down_up_bracket_override_prev():
    cmd = "AAPL 100 REL @ 33.33 - 4.44 + 2.22 ± 6 prev"
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


def test_stock_limit_down_up_bracket_override_p():
    cmd = "AAPL 100 REL @ 33.33 - 4.44 + 2.22 ± 6 p"
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


def test_ladder_up_down_manual():
    cmd = "NVDA 100 AF @ 100"
    result = OrderIntent(
        symbol="NVDA",
        qty=DecimalLongShares(100),
        algo="AF",
        limit=DecimalPrice("100"),
        preview=False,
    )

    ol = OrderLang()
    order = ol.parse(cmd)

    assert order == result

    # generate 5 total orders each increasing by $10 from the previous price (including starting price)
    assert order.ladder(5, percent=0.10) == [
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("100"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("110"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("120"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("130"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("140"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
    ]

    # generate 5 total orders each decreasing by $10 from the previous price
    assert order.ladder(5, percent=-0.10) == [
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("100"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("90"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("80"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("70"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("60"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
    ]

    # test comparison also with non-Decimal() prices in the results
    assert order.ladder(5, percent=0.10) == [
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=100.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=110.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=120.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=130.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=140.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
    ]

    assert order.ladder(5, percent=-0.10) == [
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=100.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=90.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=80.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=70.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=60.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
    ]

    assert order.ladder(5, percent=0.13) == [
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=100.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=113.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=126.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=139.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=152.0,
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
    ]

    assert order.ladder(5, percent=-0.13) == [
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("100.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("87.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("74.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("61.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("48.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
    ]


def test_ladder_implicit_pts():
    cmd = "NVDA 100 AF @ 100 scale steps 5 points 10"
    result = OrderIntent(
        symbol="NVDA",
        qty=DecimalLongShares(100),
        algo="AF",
        limit=Decimal("100"),
        preview=False,
        scaleDesc={"steps": Decimal("5"), "points": Decimal("10")},
    )

    ol = OrderLang()
    order = ol.parse(cmd)

    assert order == result

    assert order.scale == [
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("100.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("110.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("120.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("130.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("140.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
        ),
    ]


def test_ladder_implicit_noscale():
    cmd = "NVDA 100 AF @ 100"
    result = OrderIntent(
        symbol="NVDA",
        qty=DecimalLongShares(100),
        algo="AF",
        limit=Decimal("100"),
        preview=False,
    )

    ol = OrderLang()
    order = ol.parse(cmd)

    assert order == result

    # verify no scale description generates no scale
    assert order.scale == []

    # verify no scale generates an order intent which *is not* the original one
    avgrecord = order.scaleAvgRecord

    assert isinstance(avgrecord, OrderIntent)
    assert avgrecord == result
    assert avgrecord is not order


def test_ladder_implicit_pct():
    cmd = "NVDA 100 AF @ 100 scale steps 5 33%"
    result = OrderIntent(
        symbol="NVDA",
        qty=DecimalLongShares(100),
        algo="AF",
        limit=Decimal("100"),
        preview=False,
        scaleDesc={"steps": Decimal("5"), "percent": Decimal("0.33")},
    )

    ol = OrderLang()
    order = ol.parse(cmd)

    assert order == result

    assert order.scale == [
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("100.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("133.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("166.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("199.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("232.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
    ]

    assert order.scaleAvgRecord == OrderIntent(
        symbol="NVDA",
        algo="AF",
        exchange=None,
        spread=None,
        qty=DecimalLongShares("500"),
        bracketProfitAlgo="LMT",
        bracketLossAlgo="STP",
        limit=Decimal("166.00"),
        bracketProfit=None,
        bracketLoss=None,
        preview=False,
        config={},
        scaleDesc=None,
    )


def test_ladder_implicit_pts_negative():
    cmd = "NVDA 100 AF @ 100 scale steps 3 points -7"
    result = OrderIntent(
        symbol="NVDA",
        qty=DecimalLongShares(100),
        algo="AF",
        limit=Decimal("100"),
        preview=False,
        scaleDesc={"steps": Decimal("3"), "points": Decimal("-7")},
    )

    ol = OrderLang()
    order = ol.parse(cmd)

    assert order == result

    assert order.scale == [
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("100.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("93.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("86.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
    ]

    assert order.scaleAvgRecord == OrderIntent(
        symbol="NVDA",
        algo="AF",
        exchange=None,
        spread=None,
        qty=DecimalLongShares("300"),
        bracketProfitAlgo="LMT",
        bracketLossAlgo="STP",
        limit=Decimal("93"),
        bracketProfit=None,
        bracketLoss=None,
        preview=False,
        config={},
        scaleDesc=None,
    )


def test_ladder_implicit_pct_negative():
    cmd = "NVDA 100 AF @ 100 scale steps 3 -33%"
    result = OrderIntent(
        symbol="NVDA",
        qty=DecimalLongShares(100),
        algo="AF",
        limit=Decimal("100"),
        preview=False,
        scaleDesc={"steps": Decimal("3"), "percent": Decimal("-0.33")},
    )

    ol = OrderLang()
    order = ol.parse(cmd)

    assert order == result

    assert order.scale == [
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("100.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("67.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=Decimal("34.0"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
    ]

    assert order.scaleAvgRecord == OrderIntent(
        symbol="NVDA",
        algo="AF",
        exchange=None,
        spread=None,
        qty=DecimalLongShares("300"),
        bracketProfitAlgo="LMT",
        bracketLossAlgo="STP",
        limit=Decimal("67.00"),
        bracketProfit=None,
        bracketLoss=None,
        preview=False,
    )


def test_ladder_implicit_pct_negative_growth():
    cmd = "NVDA 100 AF @ 100 scale steps 5 grow 100 -33%"
    result = OrderIntent(
        symbol="NVDA",
        qty=DecimalLongShares(100),
        algo="AF",
        limit=Decimal("100"),
        preview=False,
        scaleDesc={
            "steps": Decimal("5"),
            "percent": Decimal("-0.33"),
            "growth": Decimal("100"),
        },
    )

    ol = OrderLang()
    order = ol.parse(cmd)

    assert order == result

    # Also verifies we don't scale below a $0 price boundary
    assert order.scale == [
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("100.00"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("200"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("67.00"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("300"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("34.00"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("400"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("1.00"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
    ]

    assert order.scaleAvgRecord == OrderIntent(
        symbol="NVDA",
        algo="AF",
        exchange=None,
        spread=None,
        qty=DecimalLongShares("1000"),
        bracketProfitAlgo="LMT",
        bracketLossAlgo="STP",
        limit=DecimalPrice("34.00"),
        bracketProfit=None,
        bracketLoss=None,
        preview=False,
        config={},
        scaleDesc=None,
    )


def test_ladder_implicit_pct_positive_growth():
    cmd = "NVDA 100 AF @ 100 scale steps 5 grow 200 5%"
    result = OrderIntent(
        symbol="NVDA",
        qty=DecimalLongShares(100),
        algo="AF",
        limit=Decimal("100"),
        preview=False,
        scaleDesc={
            "steps": Decimal("5"),
            "percent": Decimal("0.05"),
            "growth": Decimal("200"),
        },
    )

    ol = OrderLang()
    order = ol.parse(cmd)

    assert order == result

    # Also verifies we don't scale below a $0 price boundary
    assert order.scale == [
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("100.00"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("300"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("105.00"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("500"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("110.00"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("700"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("115.00"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("900"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("120.00"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
    ]

    assert order.scaleAvgRecord == OrderIntent(
        symbol="NVDA",
        algo="AF",
        exchange=None,
        spread=None,
        qty=DecimalLongShares("2500"),
        bracketProfitAlgo="LMT",
        bracketLossAlgo="STP",
        limit=DecimalPrice("114.00"),
        bracketProfit=None,
        bracketLoss=None,
        preview=False,
        config={},
        scaleDesc=None,
    )


def test_ladder_implicit_pct_positive_negative_growth():
    cmd = "NVDA 100 AF @ 100 scale steps 10 grow -20 5%"
    result = OrderIntent(
        symbol="NVDA",
        qty=DecimalLongShares(100),
        algo="AF",
        limit=Decimal("100"),
        preview=False,
        scaleDesc={
            "steps": Decimal("10"),
            "percent": Decimal("0.05"),
            "growth": Decimal("-20"),
        },
    )

    ol = OrderLang()
    order = ol.parse(cmd)

    assert str(DecimalLongShares(100)) == "100"
    assert str(DecimalShortShares(100)) == "-100"
    assert str(DecimalLongCash(100)) == "$100"
    assert str(DecimalShortCash(100)) == "-$100"

    assert order == result

    # Also verifies we don't scale below a 0 qty price boundary
    assert order.scale == [
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("100"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("100.00"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("80"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("105.00"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("60"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("110.00"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("40"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("115.00"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
        OrderIntent(
            symbol="NVDA",
            algo="AF",
            exchange=None,
            spread=None,
            qty=DecimalLongShares("20"),
            bracketProfitAlgo="LMT",
            bracketLossAlgo="STP",
            limit=DecimalPrice("120.00"),
            bracketProfit=None,
            bracketLoss=None,
            preview=False,
            config={},
            scaleDesc=None,
        ),
    ]

    assert order.scaleAvgRecord == OrderIntent(
        symbol="NVDA",
        algo="AF",
        exchange=None,
        spread=None,
        qty=DecimalLongShares("300"),
        bracketProfitAlgo="LMT",
        bracketLossAlgo="STP",
        limit=Decimal("106.66666667"),
        bracketProfit=None,
        bracketLoss=None,
        preview=False,
        config={},
        scaleDesc=None,
    )
