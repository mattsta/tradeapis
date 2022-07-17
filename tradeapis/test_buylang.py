from tradeapis.buylang import OLang, OrderRequest, Order, Side


def test_stock():
    cmd = "aapl"
    result = OrderRequest(orders=[Order(Side.UNSET, 1, "AAPL")])

    ol = OLang()
    assert ol.parse(cmd) == result


def test_option():
    cmd = "AAPL222222C00200000"
    result = OrderRequest(orders=[Order(Side.UNSET, 1, "AAPL222222C00200000")])

    ol = OLang()
    assert ol.parse(cmd) == result


def test_stock_qty():
    cmd = "bto 3 aapl"
    result = OrderRequest(orders=[Order(Side.BTO, 3, "AAPL")])

    ol = OLang()
    assert ol.parse(cmd) == result

def test_stockr_qty():
    cmd = "bto 3 aapl4"
    result = OrderRequest(orders=[Order(Side.BTO, 3, "AAPL4")])

    ol = OLang()
    assert ol.parse(cmd) == result


def test_stock_noqty():
    cmd = "bto aapl"
    result = OrderRequest(orders=[Order(Side.BTO, 1, "AAPL")])

    ol = OLang()
    assert ol.parse(cmd) == result


def test_option_qty():
    cmd = "bto 3000 AAPL222222C00123000"
    result = OrderRequest(orders=[Order(Side.BTO, 3000, "AAPL222222C00123000")])

    ol = OLang()
    assert ol.parse(cmd) == result


def test_spread():
    cmd = "bto AAPL222222C00123000 sto AAPL222222C00200000"
    result = OrderRequest(
        orders=[
            Order(Side.BTO, 1, "AAPL222222C00123000"),
            Order(Side.STO, 1, "AAPL222222C00200000"),
        ]
    )

    ol = OLang()

    parsed = ol.parse(cmd)
    assert parsed == result
    assert parsed.orders[0].isBuy()
    assert parsed.orders[0].isOpen()
    assert parsed.orders[1].isSell()
    assert parsed.orders[1].isOpen()

    assert not parsed.isButterfly()


def test_butterfly():
    cmd = (
        "bto 1 AAPL222222C00123000 sto 2 AAPL222222C00200000 bto 1 AAPL222222C00250000"
    )
    result = OrderRequest(
        orders=[
            Order(Side.BTO, 1, "AAPL222222C00123000"),
            Order(Side.STO, 2, "AAPL222222C00200000"),
            Order(Side.BTO, 1, "AAPL222222C00250000"),
        ]
    )

    ol = OLang()

    parsed = ol.parse(cmd)
    assert parsed == result
    assert parsed.orders[0].isBuy()
    assert parsed.orders[0].isOpen()
    assert parsed.orders[1].isSell()
    assert parsed.orders[1].isOpen()
    assert parsed.orders[2].isBuy()
    assert parsed.orders[2].isOpen()

    assert parsed.isButterfly()
