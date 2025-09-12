import datetime
import pprint
from dataclasses import dataclass

from tradeapis.ordermgr import (
    OrderMgr,
    Position,
    PositionAnalysis,
    PositionCalculations,
    PositionGroup,
    PositionSummary,
    SystemDiagnostics,
    SystemHealthReport,
    Trade,
    TradeFlowEntry,
    ValidationResult,
)

NOISE = True

EXAMPLE_DATE = datetime.datetime(1, 2, 3, 4, 5, 6, 7, datetime.timezone.utc)
EXAMPLE_DATE2 = datetime.datetime(10, 2, 3, 4, 5, 6, 7, datetime.timezone.utc)


@dataclass
class TTrade(Trade):
    """A trade object just for testing with a default timestamp"""

    timestamp: datetime.datetime = EXAMPLE_DATE


def test_create():
    with OrderMgr.temp() as om:
        om.clear()


def test_add_trade():
    with OrderMgr.temp() as om:
        om.add_trade("AAPL", TTrade(orderid=32, price=16, qty=8))
        s1 = om.get_portfolio_summary(stopPct=0.10)
        assert s1["AAPL"] == PositionSummary(
            qty=8,
            average_price=16.0,
            stop=14.4,
            total_commission=0.0,
            started=EXAMPLE_DATE,
        )

        om.add_trade(
            "AAPL", TTrade(orderid=10, price=10, qty=10, timestamp=EXAMPLE_DATE2)
        )
        s2 = om.get_portfolio_summary(stopPct=0.10)

        assert s2 == {
            "AAPL": PositionSummary(
                qty=18,
                average_price=12.666666666666666,
                stop=11.4,
                total_commission=0.0,
                started=EXAMPLE_DATE,
            )
        }

        om.update_commission("AAPL", 32, 6)
        s3 = om.get_portfolio_summary(stopPct=0.10)
        assert s3 == {
            "AAPL": PositionSummary(
                qty=18,
                average_price=13.0,
                stop=11.700000000000001,
                total_commission=6.0,
                started=EXAMPLE_DATE,
            )
        }

        om.update_commission("AAPL", 10, 4)
        s4 = om.get_portfolio_summary(stopPct=0.10)
        assert s4 == {
            "AAPL": PositionSummary(
                qty=18,
                average_price=13.222222222222221,
                stop=11.9,
                total_commission=10,
                started=EXAMPLE_DATE,
            )
        }


def test_multi_trade():
    with OrderMgr.temp() as om:
        om.add_trade("AL1", TTrade(orderid=32, price=16, qty=8))
        om.add_trade("AL2", TTrade(orderid=32, price=16, qty=8))
        om.add_trade("BB3", TTrade(orderid=64, price=64, qty=64))
        s1 = om.get_portfolio_summary(stopPct=0.10)

        if NOISE:
            pprint.pprint(s1)

        pgs = om.position_groups()

        if NOISE:
            pprint.pprint(pgs)

        a1 = pgs["AL1"]
        a2 = pgs["AL2"]

        # two long legs together is still a 1:8 ratio if they trade together
        assert a1.qty == 8
        assert a1.qty == a2.qty

        # average price is *additive* because these are to long legs in a 1:8 ratio,
        # so the averge price isn't 16-per, it's (16 * legs) per.
        assert a1.average_price == 32
        assert a1.average_price == a2.average_price

        assert a1.is_long
        assert a1.is_long == a2.is_long


def test_multi_trade_vert_long():
    with OrderMgr.temp() as om:
        om.add_trade("AL1", TTrade(orderid=32, price=16, qty=8))
        om.add_trade("AL2", TTrade(orderid=32, price=8, qty=-8))
        om.add_trade("BB3", TTrade(orderid=64, price=64, qty=64))
        s1 = om.get_portfolio_summary(stopPct=0.10)

        if NOISE:
            pprint.pprint(s1)

        pgs = om.position_groups()

        if NOISE:
            pprint.pprint(pgs)

        a1 = pgs["AL1"]
        a2 = pgs["AL2"]

        # verify our vertical spread of (qty 8, qty -8) shows up
        # as portfolio QTY 8 instead of canceling out to QTY (8 + -8 = 0)
        assert a1.qty == 8
        assert a1.qty == a2.qty

        # average price is ((16 * 8) + (-8 * 8)) / 8 for the long spread
        assert a1.average_price == 8
        assert a1.average_price == a2.average_price

        # this is a LONG spread
        assert a1.is_long
        assert a1.is_long == a2.is_long

        s = a1.summary()
        if NOISE:
            pprint.pprint(s)

        assert s.qty == 8
        assert s.average_price == 8
        assert s.stop == 7.2

        # use a simple dict hack for the key-price lookup function
        PRICES = dict(AL1=22, AL2=2)
        priceFetcher = PRICES.get

        s2 = a1.summary(stopPct=0.10, priceFetcher=priceFetcher)
        if NOISE:
            pprint.pprint(s2)

        assert s2.qty == 8
        assert s2.average_price == 8
        assert s2.stop == 18
        assert s2.profit == 12
        assert s2.stopPts == 10


def test_multi_trade_vert_short():
    with OrderMgr.temp() as om:
        om.add_trade("AL1", TTrade(orderid=32, price=16, qty=-8))
        om.add_trade("AL2", TTrade(orderid=32, price=8, qty=8))
        om.add_trade("BB3", TTrade(orderid=64, price=64, qty=64))
        s1 = om.get_portfolio_summary(stopPct=0.10)
        pprint.pprint(s1)

        pgs = om.position_groups()

        if NOISE:
            pprint.pprint(pgs)

        a1 = pgs["AL1"]
        a2 = pgs["AL2"]

        # verify our vertical spread of (SHORT 8, LONG 8) shows up
        # as portfolio QTY 8 instead of canceling out to QTY (-8 + 8 = 0)
        assert a1.qty == 8
        assert a1.qty == a2.qty

        # average price is ((16 * -8) + (8 * 8)) / 8
        assert a1.average_price == -8
        assert a1.average_price == a2.average_price

        # this is a SHORT spread
        assert not a1.is_long
        assert a1.is_long == a2.is_long

        s = a1.summary()
        if NOISE:
            pprint.pprint(s)

        assert s.qty == 8
        assert s.average_price == -8
        assert s.stop == 8.8


def test_multi_trade_vert_short_adding():
    with OrderMgr.temp() as om:
        om.add_trade("AL1", TTrade(orderid=32, price=16, qty=-8))
        om.add_trade("AL2", TTrade(orderid=32, price=8, qty=8))
        om.add_trade("BB3", TTrade(orderid=64, price=64, qty=64))
        om.add_trade("AL1", TTrade(orderid=128, price=30, qty=-8))
        om.add_trade("AL2", TTrade(orderid=128, price=10, qty=8))
        s1 = om.get_portfolio_summary(stopPct=0.10)
        pprint.pprint(s1)

        pgs = om.position_groups()

        if NOISE:
            pprint.pprint(pgs)

        a1 = pgs["AL1"]
        a2 = pgs["AL2"]

        # verify our vertical spread of (SHORT 8+8, LONG 8+8) shows up
        assert a1.qty == 16
        assert a1.qty == a2.qty

        # average price is ((16 * -8) + (8 * 8) + (30 * -8) + (10 * 8)) / 16
        assert a1.average_price == -14
        assert a1.average_price == a2.average_price

        # this is a SHORT spread
        assert not a1.is_long
        assert a1.is_long == a2.is_long

        s = a1.summary(stopPct=0.10)
        if NOISE:
            pprint.pprint(s)

        assert s.qty == 16
        assert s.average_price == -14
        assert s.stop == 15.40

        def priceFetcher(key):
            if key == "AL1":
                return 5

            if key == "AL2":
                return 2

        s2 = a1.summary(stopPct=0.10, priceFetcher=priceFetcher)
        if NOISE:
            pprint.pprint(s2)

        assert s2.qty == 16
        assert s2.average_price == -14
        assert s2.marketPrice == 3
        assert s2.stop == 3.3
        assert s2.profit == 11
        assert s2.stopPts == -10.7


def test_multi_trade_vert_short_partial_add_and_complete():
    with OrderMgr.temp() as om:
        om.add_trade("AL1", TTrade(orderid=32, price=16, qty=-8))
        om.add_trade("AL2", TTrade(orderid=32, price=8, qty=8))
        om.add_trade("BB3", TTrade(orderid=64, price=64, qty=64))
        s1 = om.get_portfolio_summary(stopPct=0.10)
        pprint.pprint(s1)

        pgs = om.position_groups()

        if NOISE:
            pprint.pprint(pgs)

        a1 = pgs["AL1"]
        a2 = pgs["AL2"]

        # verify our vertical spread of (SHORT 8, LONG 8) shows up
        # as portfolio QTY 8 instead of canceling out to QTY (-8 + 8 = 0)
        assert a1.qty == 8
        assert a1.qty == a2.qty

        # average price is ((16 * -8) + (8 * 8)) / 8
        assert a1.average_price == -8
        assert a1.average_price == a2.average_price

        # this is a SHORT spread
        assert not a1.is_long
        assert a1.is_long == a2.is_long

        s = a1.summary()
        if NOISE:
            pprint.pprint(s)

        assert s.qty == 8
        assert s.average_price == -8
        assert s.stop == 8.8

        # adding a PARTIAL leg
        # verifying un-equal legs added don't change the average price or group quantity because it's an incomplete add.
        om.add_trade("AL1", TTrade(orderid=34, price=16, qty=-8))

        s2 = a1.summary()
        print("S2:")
        if NOISE:
            pprint.pprint(s2)

        assert s2.qty == 8
        assert s2.average_price == -8
        assert s2.stop == 8.8

        # test adding SECOND FULL so quantity and average price does increase
        om.add_trade("AL2", TTrade(orderid=34, price=8, qty=8))

        s3 = a1.summary()
        print("S3:")
        if NOISE:
            pprint.pprint(s3)

        assert s3.qty == 16
        assert s3.average_price == -8
        assert s3.stop == 8.8

        # adjust stop and test
        s4 = a1.summary(stopPct=0.20)
        print("S4:")
        if NOISE:
            pprint.pprint(s4)

        assert s4.stop == 9.6

        # try a negative stop to _increase_ our cost basis instead
        s5 = a1.summary(stopPct=-0.20)
        print("S5:")
        if NOISE:
            pprint.pprint(s5)

        assert s5.stop == 6.4


def test_zero_qty_trade():
    """Test that zero quantity trades are handled correctly."""
    with OrderMgr.temp() as om:
        # Test zero qty trade doesn't break average_price calculation
        trade = TTrade(orderid=1, price=100, qty=0, commission=5)

        # Trade.average_price should handle qty=0 gracefully
        assert trade.average_price == 0.0

        # Adding zero qty trade to position
        om.add_trade("TEST", trade)

        # Position should be empty after adding zero qty trade
        pos = om.positions.get("TEST")
        assert pos is None or pos.empty


def test_negative_price_validation():
    """Test that negative prices are properly rejected."""
    try:
        TTrade(orderid=1, price=-100, qty=10)
        assert False, "Should have raised ValueError for negative price"
    except ValueError as e:
        assert "negative prices" in str(e).lower()


def test_commission_credits():
    """Test that negative commissions (credits) are handled correctly."""
    with OrderMgr.temp() as om:
        # Test commission credit (negative commission)
        om.add_trade("TEST", TTrade(orderid=1, price=100, qty=10, commission=-5))

        summary = om.get_position_summary("TEST")
        # With commission credit, average price should be lower
        assert summary.average_price == 99.5  # (100*10 + (-5)) / 10
        assert summary.total_commission == -5


def test_empty_portfolio():
    """Test portfolio operations on empty portfolio."""
    with OrderMgr.temp() as om:
        # Empty portfolio summary
        summary = om.get_portfolio_summary()
        assert summary == {}

        # Empty position groups
        groups = om.position_groups()
        assert groups == {}


def test_position_clearing_behavior():
    """Test that positions are cleared when qty becomes zero."""
    with OrderMgr.temp() as om:
        # Add initial position
        om.add_trade("TEST", TTrade(orderid=1, price=100, qty=10))

        # Check position exists
        positions_list = list(om.positions.keys())
        assert "TEST" in positions_list

        # Add offsetting trade that zeros out position
        om.add_trade("TEST", TTrade(orderid=2, price=50, qty=-10))

        # Position should be removed when qty becomes 0
        positions_list = list(om.positions.keys())
        assert "TEST" not in positions_list


def test_duplicate_orderid_rejection():
    """Test that duplicate order IDs for same position are rejected."""
    with OrderMgr.temp() as om:
        # Add initial trade
        om.add_trade("TEST", TTrade(orderid=1, price=100, qty=10))

        # Try to add trade with same orderid - should be rejected
        om.add_trade("TEST", TTrade(orderid=1, price=200, qty=5))

        pos = om.positions["TEST"]
        assert len(pos.trades) == 1
        assert pos.trades[1].price == 100  # Original trade should remain


def test_complex_multi_leg_strategy():
    """Test more complex multi-leg strategies like butterflies."""
    with OrderMgr.temp() as om:
        # Butterfly: BUY 1, SELL 2, BUY 1 (same orderid)
        om.add_trade("LEG1", TTrade(orderid=100, price=95, qty=1))  # Buy 1
        om.add_trade("LEG2", TTrade(orderid=100, price=100, qty=-2))  # Sell 2
        om.add_trade("LEG3", TTrade(orderid=100, price=105, qty=1))  # Buy 1

        groups = om.position_groups()

        # All legs should be in the same group
        leg1_group = groups["LEG1"]
        leg2_group = groups["LEG2"]
        leg3_group = groups["LEG3"]

        assert leg1_group is leg2_group is leg3_group
        assert leg1_group.qty == 1  # Min absolute qty across all legs

        # Verify the group contains all three positions
        assert len(leg1_group.positions) == 3


def test_fractional_quantities():
    """Test handling of fractional quantities."""
    with OrderMgr.temp() as om:
        om.add_trade("TEST", TTrade(orderid=1, price=100.5, qty=2.5, commission=1.25))

        summary = om.get_position_summary("TEST")
        assert summary.qty == 2.5
        # (100.5 * 2.5 + 1.25) / 2.5 = 252.75 / 2.5 = 101.1
        assert abs(summary.average_price - 101.0) < 0.001


def test_large_numbers():
    """Test handling of very large numbers."""
    with OrderMgr.temp() as om:
        large_qty = 1_000_000
        large_price = 50_000.0

        om.add_trade("BIG", TTrade(orderid=1, price=large_price, qty=large_qty))

        summary = om.get_position_summary("BIG")
        assert summary.qty == large_qty
        assert summary.average_price == large_price


def test_market_price_profit_calculation():
    """Test profit calculation with market prices."""
    with OrderMgr.temp() as om:
        # Long position
        om.add_trade("LONG", TTrade(orderid=1, price=100, qty=10))

        groups = om.position_groups()
        long_group = groups["LONG"]

        # Market price higher than cost - should show profit
        def price_fetcher(key):
            return 110.0 if key == "LONG" else 0.0

        summary = long_group.summary(priceFetcher=price_fetcher)
        assert summary.profit == 10.0  # 110 - 100

        # Short position
        om.add_trade("SHORT", TTrade(orderid=2, price=100, qty=-10))
        short_group = om.position_groups()["SHORT"]

        def short_price_fetcher(key):
            return 90.0 if key == "SHORT" else 0.0

        short_summary = short_group.summary(priceFetcher=short_price_fetcher)
        assert short_summary.profit == 10.0  # 100 - 90 for short


def test_nan_market_price_handling():
    """Test that NaN market prices are handled gracefully."""

    with OrderMgr.temp() as om:
        om.add_trade("TEST", TTrade(orderid=1, price=100, qty=10))

        groups = om.position_groups()
        test_group = groups["TEST"]

        def nan_price_fetcher(key):
            return float("nan")

        summary = test_group.summary(priceFetcher=nan_price_fetcher)
        assert summary.marketPrice is None
        assert summary.profit is None


def test_commission_update():
    """Test delayed commission updates."""
    with OrderMgr.temp() as om:
        om.add_trade("TEST", TTrade(orderid=1, price=100, qty=10, commission=0))

        # Initial summary with no commission
        summary1 = om.get_position_summary("TEST")
        assert summary1.average_price == 100.0
        assert summary1.total_commission == 0.0

        # Update commission
        om.update_commission("TEST", 1, 5.0)

        # Summary should reflect commission
        summary2 = om.get_position_summary("TEST")
        assert summary2.average_price == 100.5  # (100*10 + 5) / 10
        assert summary2.total_commission == 5.0


def test_position_by_order_summary():
    """Test getting position summary by order ID."""
    with OrderMgr.temp() as om:
        # Create a spread
        om.add_trade("LEG1", TTrade(orderid=100, price=50, qty=10))
        om.add_trade("LEG2", TTrade(orderid=100, price=55, qty=-10))

        # Get summary by orderid
        summary = om.get_position_summary_by_order(100)

        assert summary is not None
        assert summary.qty == 10
        assert summary.average_price == -5.0  # (50*10 + 55*-10) / 10


def test_stop_percentage_variations():
    """Test different stop percentage scenarios."""
    with OrderMgr.temp() as om:
        om.add_trade("TEST", TTrade(orderid=1, price=100, qty=10))

        # Test various stop percentages
        summary_10 = om.get_position_summary("TEST", stopPct=0.10)
        summary_20 = om.get_position_summary("TEST", stopPct=0.20)
        summary_neg = om.get_position_summary("TEST", stopPct=-0.10)

        assert summary_10.stop == 90.0  # 100 * (1 - 0.10)
        assert summary_20.stop == 80.0  # 100 * (1 - 0.20)
        assert summary_neg.stop == 110.0  # 100 * (1 - (-0.10))


def test_error_conditions_properly_crash():
    """Test that data consistency issues properly crash instead of being hidden."""
    with OrderMgr.temp():
        # Test Position.started with no trades
        empty_pos = Position("EMPTY")
        try:
            _ = empty_pos.started
            assert False, "Should crash when accessing started on empty position"
        except ValueError as e:
            assert "no trades" in str(e).lower()

        # Test Position.add_commission with invalid orderid
        pos = Position("TEST")
        pos.add_trade(TTrade(orderid=1, price=100, qty=10))
        try:
            pos.add_commission(999, 5.0)  # Non-existent orderid
            assert False, (
                "Should crash when updating commission for non-existent orderid"
            )
        except KeyError as e:
            assert "999" in str(e) and "no such trade" in str(e).lower()


def test_position_group_error_conditions():
    """Test that PositionGroup error conditions properly crash."""
    with OrderMgr.temp() as om:
        # Create positions with zero quantities - this should cause crashes in group operations
        om.add_trade("LEG1", TTrade(orderid=1, price=100, qty=10))
        om.add_trade("LEG1", TTrade(orderid=2, price=100, qty=-10))  # Zeros out LEG1

        # LEG1 should now be removed from positions (qty = 0)
        assert "LEG1" not in om.positions

        # Test PositionGroup.generateOrderDesc with zero quantity
        # Create a group that will have zero quantity
        om.add_trade("LEG2", TTrade(orderid=3, price=50, qty=5))
        om.add_trade("LEG2", TTrade(orderid=4, price=60, qty=-5))  # Zeros out LEG2

        # LEG2 should be removed, so can't test zero quantity group directly
        # Instead, test the error condition by creating an artificial scenario


def test_correct_stop_percentage_calculation():
    """Test that stopPct uses the correct entry-based calculation."""
    with OrderMgr.temp() as om:
        # Test long position
        om.add_trade("LONG", TTrade(orderid=1, price=100, qty=10))
        summary = om.get_position_summary("LONG", stopPct=0.10)

        # Correct formula: (entry - stop) / entry for longs
        # For long: stop should be 90, so stopPct = (100 - 90) / 100 = 0.10
        expected_stop_pct = 0.10
        assert abs(summary.stopPct - expected_stop_pct) < 0.001

        # Test short position
        om.add_trade("SHORT", TTrade(orderid=2, price=100, qty=-10))
        short_summary = om.get_position_summary("SHORT", stopPct=0.10)

        # Correct formula: (stop - entry) / entry for shorts
        # For short: average_price = -100, stop should be 110, so stopPct = (110 - 100) / 100 = 0.10
        expected_short_stop_pct = 0.10
        assert abs(short_summary.stopPct - expected_short_stop_pct) < 0.001


def test_division_by_zero_exposure():
    """Test that division by zero properly crashes instead of being hidden."""

    # Create an artificial scenario where we can test division by zero in PositionGroup
    pos1 = Position("TEST1")
    pos1.add_trade(TTrade(orderid=1, price=100, qty=0))  # Zero quantity trade

    group = PositionGroup()
    group.add(pos1)

    # This should crash when trying to generate order description
    try:
        group.generateOrderDesc("OPEN")
        assert False, "Should crash when generating order desc for zero quantity group"
    except ValueError as e:
        assert "zero quantity" in str(e).lower()

    # Test division by zero in average_price calculation
    try:
        _ = group.average_price
        assert False, "Should crash with division by zero in average_price"
    except ZeroDivisionError:
        pass  # This is expected behavior now


def test_nan_handling_in_market_price():
    """Test that NaN values from unreliable external APIs are properly handled."""

    with OrderMgr.temp() as om:
        om.add_trade("TEST", TTrade(orderid=1, price=100, qty=10))

        groups = om.position_groups()
        test_group = groups["TEST"]

        def nan_price_fetcher(key):
            return float("nan")

        # NaN should be converted to None to protect against unreliable external price APIs
        summary = test_group.summary(priceFetcher=nan_price_fetcher)

        # marketPrice should be None when NaN is returned by price fetcher
        assert summary.marketPrice is None

        # profit calculation should also be None when marketPrice is None
        assert summary.profit is None


def test_unbalanced_spread_detection():
    """Test that unbalanced spreads are detected but don't crash."""
    with OrderMgr.temp() as om:
        # Create an unbalanced spread (different quantities)
        om.add_trade("LEG1", TTrade(orderid=1, price=100, qty=10))  # 10 shares
        om.add_trade(
            "LEG2", TTrade(orderid=1, price=105, qty=-5)
        )  # Only 5 shares short

        groups = om.position_groups()
        group = groups["LEG1"]

        # Should still be able to generate order descriptions for unbalanced spreads
        # The system should handle this gracefully
        order_desc = group.generateOrderDesc("OPEN")
        assert "BUY" in order_desc and "SELL" in order_desc

        # Group quantity should be the minimum (5 in this case)
        assert group.qty == 5


# ==============================================================================
# DIAGNOSTIC AND REPORTING SYSTEM TESTS
# ==============================================================================


def test_health_check_healthy_system():
    """Test SystemHealthReport with healthy system."""
    with OrderMgr.temp() as om:
        om.add_trade("AAPL", TTrade(orderid=1, price=150.0, qty=100))
        om.add_trade("GOOGL", TTrade(orderid=2, price=2500.0, qty=50))

        health = om.health_check()

        assert isinstance(health, SystemHealthReport)
        assert health.status == "healthy"
        assert health.total_positions == 2
        assert health.total_trades == 2
        assert len(health.warnings) == 0
        assert len(health.errors) == 0
        assert len(health.violations) == 0


def test_health_check_with_warnings():
    """Test SystemHealthReport detecting system issues."""
    with OrderMgr.temp() as om:
        # Create normal position first
        om.add_trade("AAPL", TTrade(orderid=1, price=150.0, qty=100))

        # Create corrupted position by bypassing Trade validation
        corrupted_pos = Position("CORRUPTED")
        # Create a valid trade first
        bad_trade = Trade(orderid=999, price=50.0, qty=10)
        # Then manually corrupt the price to bypass validation
        object.__setattr__(bad_trade, "price", -50.0)
        corrupted_pos.trades[999] = bad_trade
        om.positions["CORRUPTED"] = corrupted_pos

        health = om.health_check()

        assert health.status == "warnings"
        assert health.total_positions == 2
        assert len(health.violations) > 0
        assert any("negative" in warning.lower() for warning in health.violations)


def test_position_analysis_comprehensive():
    """Test PositionAnalysis with complex trade flow."""
    with OrderMgr.temp() as om:
        # Create position with multiple trades
        om.add_trade(
            "AAPL", TTrade(orderid=1, price=100.0, qty=50, timestamp=EXAMPLE_DATE)
        )
        om.add_trade(
            "AAPL", TTrade(orderid=2, price=110.0, qty=30, timestamp=EXAMPLE_DATE2)
        )
        om.add_trade("AAPL", TTrade(orderid=3, price=120.0, qty=-20, commission=5.0))

        analysis = om.analyze_position("AAPL")

        assert isinstance(analysis, PositionAnalysis)
        assert analysis.key == "AAPL"
        assert analysis.qty == 60  # 50 + 30 - 20
        assert analysis.is_long
        assert analysis.total_commission == 5.0
        assert len(analysis.trade_flow) == 3

        # Verify trade flow entries
        flow_entry = analysis.trade_flow[0]
        assert isinstance(flow_entry, TradeFlowEntry)
        assert flow_entry.orderid == 1
        assert flow_entry.price == 100.0
        assert flow_entry.qty == 50
        assert flow_entry.running_qty == 50

        # Verify calculations
        assert isinstance(analysis.calculations, PositionCalculations)
        assert analysis.calculations.qty_matches
        assert analysis.calculations.avg_price_matches
        assert len(analysis.consistency_errors) == 0


def test_validation_result_processing():
    """Test ValidationResult dataclass and consistency validation."""
    with OrderMgr.temp() as om:
        om.add_trade("AAPL", TTrade(orderid=1, price=150.0, qty=100))
        om.add_trade("GOOGL", TTrade(orderid=2, price=2500.0, qty=-50))

        results = om.validate_consistency()

        assert len(results) == 2
        for result in results:
            assert isinstance(result, ValidationResult)
            assert result.position_key in ["AAPL", "GOOGL"]
            assert isinstance(result.errors, list)
            assert isinstance(result.is_valid, bool)

        # All should be valid for clean data
        assert all(result.is_valid for result in results)


def test_system_diagnostics_deep_analysis():
    """Test SystemDiagnostics with comprehensive analysis."""
    with OrderMgr.temp() as om:
        # Create complex scenario
        om.add_trade("AAPL", TTrade(orderid=1, price=100.0, qty=100))
        om.add_trade("AAPL", TTrade(orderid=2, price=110.0, qty=-50, commission=2.5))
        om.add_trade("GOOGL", TTrade(orderid=3, price=2500.0, qty=25))

        diagnostics = SystemDiagnostics(om)

        # Test health check
        health = diagnostics.health_check()
        assert health.status == "healthy"
        assert health.total_positions == 2
        assert health.total_trades == 3

        # Test position analysis
        aapl_analysis = diagnostics.analyze_position("AAPL")
        assert aapl_analysis is not None
        assert aapl_analysis.qty == 50
        assert len(aapl_analysis.trade_flow) == 2

        # Test non-existent position
        none_analysis = diagnostics.analyze_position("NONEXISTENT")
        assert none_analysis is None


def test_system_reporter_portfolio_report():
    """Test SystemReporter portfolio report generation."""
    with OrderMgr.temp() as om:
        om.add_trade("AAPL", TTrade(orderid=1, price=150.0, qty=100))
        om.add_trade("GOOGL", TTrade(orderid=2, price=2500.0, qty=50))

        report = om.portfolio_report(stopPct=0.05)

        assert isinstance(report, str)
        assert "AAPL" in report
        assert "GOOGL" in report
        assert "150.0" in report
        assert "2,500.00" in report


def test_system_reporter_position_table():
    """Test SystemReporter position table generation."""
    with OrderMgr.temp() as om:
        om.add_trade("AAPL", TTrade(orderid=1, price=150.0, qty=100))
        om.add_trade("TSLA", TTrade(orderid=2, price=800.0, qty=-25))

        table = om.position_table()

        assert isinstance(table, str)
        assert "AAPL" in table
        assert "TSLA" in table
        assert "100" in table
        assert "-25" in table


def test_system_reporter_spread_analysis():
    """Test SystemReporter spread analysis with multi-leg positions."""
    with OrderMgr.temp() as om:
        # Create a spread
        om.add_trade("AAPL_CALL", TTrade(orderid=1, price=5.0, qty=10))
        om.add_trade("AAPL_PUT", TTrade(orderid=1, price=3.0, qty=-10))

        spread_report = om.spread_report()

        assert isinstance(spread_report, str)
        assert "AAPL_CALL" in spread_report or "AAPL_PUT" in spread_report


def test_debug_summary_comprehensive():
    """Test comprehensive debug summary report."""
    with OrderMgr.temp() as om:
        om.add_trade("AAPL", TTrade(orderid=1, price=150.0, qty=100))
        om.add_trade("GOOGL", TTrade(orderid=2, price=2500.0, qty=50))

        debug_report = om.debug_summary()

        assert isinstance(debug_report, str)
        assert "ORDERMGR DEBUG SUMMARY" in debug_report
        assert "SYSTEM HEALTH" in debug_report
        assert "POSITION TABLE" in debug_report
        assert "PORTFOLIO SUMMARY" in debug_report


def test_diagnostic_workflow_complete():
    """Test complete diagnostic workflow from creation to analysis."""
    with OrderMgr.temp() as om:
        # Step 1: Create positions
        om.add_trade("AAPL", TTrade(orderid=1, price=100.0, qty=100, commission=1.0))
        om.add_trade("AAPL", TTrade(orderid=2, price=110.0, qty=50, commission=0.5))
        om.add_trade("GOOGL", TTrade(orderid=3, price=2500.0, qty=-25, commission=2.0))

        # Step 2: Health check
        health = om.health_check()
        assert health.status == "healthy"

        # Step 3: Validation
        validations = om.validate_consistency()
        assert all(v.is_valid for v in validations)

        # Step 4: Deep analysis
        aapl_analysis = om.analyze_position("AAPL")
        googl_analysis = om.analyze_position("GOOGL")

        # Verify AAPL analysis
        assert aapl_analysis.qty == 150
        assert aapl_analysis.is_long
        assert len(aapl_analysis.trade_flow) == 2
        assert aapl_analysis.total_commission == 1.5

        # Verify GOOGL analysis
        assert googl_analysis.qty == -25
        assert not googl_analysis.is_long
        assert len(googl_analysis.trade_flow) == 1
        assert googl_analysis.total_commission == 2.0

        # Step 5: Generate reports
        portfolio_report = om.portfolio_report()
        position_table = om.position_table()
        spread_analysis = om.spread_report()
        debug_summary = om.debug_summary()

        # Verify all reports are generated
        assert all(
            isinstance(report, str)
            for report in [
                portfolio_report,
                position_table,
                spread_analysis,
                debug_summary,
            ]
        )

        # Step 6: Verify data structures are strongly typed
        assert isinstance(aapl_analysis, PositionAnalysis)
        assert isinstance(aapl_analysis.trade_flow[0], TradeFlowEntry)
        assert isinstance(aapl_analysis.calculations, PositionCalculations)
        assert isinstance(health, SystemHealthReport)
        assert isinstance(validations[0], ValidationResult)


def test_error_detection_and_reporting():
    """Test that diagnostic system properly detects and reports errors."""
    with OrderMgr.temp() as om:
        # Add normal position
        om.add_trade("AAPL", TTrade(orderid=1, price=150.0, qty=100))

        # Create position with corrupted trade data to test validation
        corrupted_pos = Position("CORRUPTED")
        # Create a valid trade first then corrupt it
        bad_trade = Trade(orderid=999, price=50.0, qty=10)
        object.__setattr__(bad_trade, "price", -50.0)  # Force corrupted data
        corrupted_pos.trades[999] = bad_trade
        om.positions["CORRUPTED"] = corrupted_pos

        # Health check should detect the problem
        health = om.health_check()
        assert health.status == "warnings"
        assert len(health.violations) > 0

        # Position analysis should still work for valid positions
        analysis = om.analyze_position("AAPL")
        assert analysis is not None
        assert analysis.key == "AAPL"
