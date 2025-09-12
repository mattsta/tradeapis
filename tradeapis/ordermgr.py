"""Track portfolio holdings even when a position contains multiple symbols.

One recurring problem with order management is once an order is placed, the instruments
appear in your portfolio as individual line items (unless your broker applies extra processing
magic to discover spreads or offsetting positions).

One way to detect/remember/calculate combined positions (which often must close together since they
were opened together) is to track symbols which traded via the same underlying orderid.

Here we present ordermgr in an attempt to track whole position values for risk tracking.

We need to distinguish between "orders" and "trades" and "positions" first.

- Order can have multiple Trades inside of it.
  - e.g. vertical spreads are two legs placed as a single atomic order.
- Trade is a complete market event with a timestamp, trade price, trade qty, and commission entry.
  - e.g. each leg of a vertical spread is an independent Trade.
- Position is what exists in your portfolio after orders complete. When you execute a vertical spread,
  your portoflio just has two legs now (one long, one short, both of equal qty), but now your portfolio
  doesn't remember those positions traded together to open in the first place.

So here, we track:

    - Individual Trades, each with (orderid, timestamp, price, qty, commission)
    - Individual Orders discovered via the Trade history of same-order-id matching.
    - Positions as reported orders matched against trades with a cost basis and stop-out limit.


Usage here is:

    - When a trade executes, populate a trade object with (orderid, timestamp, price, qty, commission)
    - Then you can query for groups of symbols with matching order ids placed over time for thier:
      - average order price (adjusted for commissions)
      - dynamic stop limits based on average order price
      - total qty holding

Example:
    - you order a spread of (BUY 1 C AA +10, SELL 1 C BB +20) executed as order id 1
    - two trades occur under order id 1 for two symbols:
      - orderid: 1, qty:  10, BUY 1 AA
      - orderid: 1, qty: -10, SELL 1 BB
    - you order more under order id 2
    - two trades occur under order id 2 for the same two symbols:
      - orderid: 2, qty:  5, BUY 1 AA
      - orderid: 2, qty: -5, SELL 1 BB

    - now we must report a total of:
      - qty: 15, with average price of all the trades with commissions combined.

    - so each symbol will have a set of orderIds under which it executed, then we can collect
      all matching orderIds for all symbols to retrieve which symbols "traded together" because
      they likely need to close together as well.

We need some dual indexing because we need to track:

    - combined positions (multiple instruments with same-order-id execution)
    - update individual trades with async commissions based on (symbol, orderid)

CRITICAL DATA MODEL CONSISTENCY PRINCIPLES:
==========================================

This system maintains strict consistency rules across all classes for qty, average_price,
and is_long determination. These rules are ESSENTIAL and must never be violated:

1. TRADE LEVEL (Trade class):
   - price: ALWAYS positive (never negative)
   - qty: SIGNED (positive for longs, negative for shorts)
   - Shorts are designated by NEGATIVE QUANTITY, not negative price

2. POSITION LEVEL (Position class):
   - qty: SIGNED (sum of trade quantities - can be negative for net short)
   - average_price: ALWAYS positive (mathematical result of cost/qty calculation)
   - is_long: Determined by qty > 0 (quantity sign indicates direction)
   - Note: Short positions have negative qty but positive average_price due to math

3. POSITION SUMMARY LEVEL (PositionSummary class):
   - qty: ALWAYS positive (display quantity, never negative)
   - average_price: SIGNED (positive for longs, negative for shorts for display)
   - is_long: Determined by average_price > 0 (sign indicates direction)
   - Conversion: Position -> PositionSummary transforms data for consistent display

4. POSITION GROUP LEVEL (PositionGroup class):
   - qty: Complex spread logic (minimum absolute quantity of legs)
   - average_price: SIGNED (can be negative for net short spreads)
   - is_long: Determined by average_price > 0 (spread direction)
   - Maintains consistency with PositionSummary format for display

CONSISTENCY FLOW EXAMPLE:
========================

LONG Trade: price=100, qty=10 →
  Position: qty=10, avg_price=100.0, is_long=True →
  PositionSummary: qty=10, avg_price=100.0, is_long=True

SHORT Trade: price=100, qty=-10 →
  Position: qty=-10, avg_price=100.0 (math: -1000/-10=100), is_long=False →
  PositionSummary: qty=10, avg_price=-100.0 (display format), is_long=False

DIAGNOSTIC SYSTEM USAGE EXAMPLES:
=================================

# Basic health monitoring workflow
order_mgr = OrderMgr("portfolio")
health = order_mgr.health_check()
if health.status != "healthy":
    print(f"System issues detected: {len(health.warnings)} warnings, {len(health.errors)} errors")
    for violation in health.violations:
        print(f"Data violation: {violation}")

# Position analysis and validation workflow
analysis = order_mgr.analyze_position("AAPL")
if analysis and analysis.consistency_errors:
    print(f"Position AAPL has consistency issues:")
    for error in analysis.consistency_errors:
        print(f"  Error: {error}")

# Trade flow analysis
if analysis:
    print(f"AAPL Trade Flow ({len(analysis.trade_flow)} trades):")
    for entry in analysis.trade_flow:
        print(f"  {entry.timestamp}: {entry.qty}@{entry.price} -> running avg {entry.running_avg:.2f}")

# System-wide validation workflow
results = order_mgr.validate_consistency()
invalid_positions = [r for r in results if not r.is_valid]
if invalid_positions:
    print(f"Found {len(invalid_positions)} positions with errors:")
    for result in invalid_positions:
        print(f"  {result.position_key}: {len(result.errors)} errors")
        for error in result.errors:
            print(f"    - {error}")

# Comprehensive reporting workflow
print("SYSTEM OVERVIEW:")
print(order_mgr.debug_summary())
print("\nPOSITION TABLE:")
print(order_mgr.position_table())
print("\nPORTFOLIO ANALYSIS:")
print(order_mgr.portfolio_report(stopPct=0.05))
print("\nSPREAD ANALYSIS:")
print(order_mgr.spread_report())

# Diagnostic data processing workflow
diagnostics = SystemDiagnostics(order_mgr)
for key in order_mgr.positions.keys():
    analysis = diagnostics.analyze_position(key)
    if analysis and not analysis.calculations.qty_matches:
        print(f"Calculation mismatch in {key}:")
        print(f"  Calculated qty: {analysis.calculations.final_qty}")
        print(f"  Stored qty: {analysis.calculations.position_qty}")

STOP PERCENTAGE CALCULATIONS:
============================

Uses industry-standard entry-based formula:
- For LONGS: stopPct = (entry_price - stop_price) / entry_price
- For SHORTS: stopPct = (stop_price - entry_price) / entry_price

This ensures consistent risk percentage regardless of position direction.

VALIDATION REQUIREMENTS:
=======================

1. Trade prices must never be negative (validation in Trade.__post_init__)
2. Position.is_long must always use qty > 0 for direction
3. PositionSummary.is_long must always use average_price > 0 for direction
4. Conversions between classes must maintain data integrity
5. Stop calculations must use absolute price values for execution
"""

from __future__ import annotations

import base64
import datetime
from collections.abc import Callable, Hashable, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from mutil.dualcache import DualCache

# We allow OrderId to techically be anything because _maybe_ you want to, instead of an order id, just place a full trade contract here instead?
# The OrderId is basically used as a join key for determining which positions were combined as atom trades to open, so OrderId is
# any vaule where all legs of a multi-instrument position shares the same value.
OrderId: TypeAlias = Hashable

# You can reference positions by anything really. A common format could be {Instrument}-{Symbol}, a contract id, or even a full contract object itself.
Key: TypeAlias = Hashable

# standard numerical formats
Price: TypeAlias = float
Qty: TypeAlias = float | int
Percent: TypeAlias = float


@dataclass(slots=True)
class PositionSummary:
    """Represents a normalized summary of a position for display and analysis.

    CRITICAL DESIGN PRINCIPLES:
    - qty: ALWAYS positive (absolute quantity regardless of long/short direction)
    - average_price: SIGNED (positive for longs, negative for shorts to indicate direction)
    - is_long: Determined by average_price > 0 (price sign indicates position direction)
    - stop: ALWAYS positive (executable stop price regardless of direction)

    This class transforms Position data (which has signed qty) into a consistent display format
    where direction is indicated by average_price sign rather than quantity sign.

    CONVERSION FROM POSITION:
    Long Position: qty=10, avg_price=100 → PositionSummary: qty=10, avg_price=100, is_long=True
    Short Position: qty=-10, avg_price=100 → PositionSummary: qty=10, avg_price=-100, is_long=False
    """

    # Quantity: ALWAYS positive (absolute position size)
    qty: Qty

    # Average price: SIGNED - positive for longs, negative for shorts (indicates direction)
    # This is the DISPLAY format - differs from Position.average_price which is always positive
    average_price: Price

    # note: we report stop prices as always positive prices, because to stop out
    # of a long you enter SELL $5, not SELL -$5 and to stop out of a short is BUY $5, not BUY -$5
    stop: Price
    total_commission: Price

    # stop as percentage of average cost
    stopPct: Percent = field(init=False)

    started: datetime.datetime

    # optionally provide a market price to re-normalize stop against
    # (e.g. if 10% stop level from average price, but marketPrice is provided, use 10% from marketPrice if marketPrice is above the stop out level)
    marketPrice: Price | None = None

    # these remaining instance variables are auto-generated from above content

    # if marketPrice is provided, we can calculate a live profit/loss value
    profit: Price | None = field(init=False)

    # stop points away from average cost
    stopPts: Price = field(init=False)

    # dynamic difference from `started` when this summary is created.
    # also ignore `age` when comparing summaries for equality
    age: datetime.timedelta = field(init=False, compare=False)

    def __post_init__(self):
        """Initialize derived fields including stop percentage, profit, and age.

        STOP PERCENTAGE CALCULATION (CRITICAL):
        Uses industry-standard entry-based formula where risk is expressed as percentage of entry price:

        For LONG positions: stopPct = (entry_price - stop_price) / entry_price
        For SHORT positions: stopPct = (stop_price - entry_price) / entry_price

        Example: $100 entry with $90 stop = (100-90)/100 = 10% risk regardless of direction

        PROFIT CALCULATION:
        - Long positions: profit = market_price - entry_price (positive when market > entry)
        - Short positions: profit = entry_price - market_price (positive when market < entry)

        Note: All calculations use absolute values of average_price since shorts have negative
        average_price in PositionSummary but calculations need positive price values.
        """
        self.age = datetime.datetime.now(datetime.timezone.utc) - self.started

        if self.marketPrice:
            if self.is_long:
                self.profit = self.marketPrice - self.average_price
            else:
                self.profit = abs(self.average_price) - self.marketPrice
        else:
            self.profit = None

        # create stop pct measure from params
        # Standard stop percentage: risk as % of entry price (industry standard)
        abs_avg_price = abs(self.average_price)
        if abs_avg_price == 0:
            self.stopPct = 0.0
        else:
            # For longs: (entry - stop) / entry, for shorts: (stop - entry) / entry
            if self.is_long:
                self.stopPct = round((abs_avg_price - self.stop) / abs_avg_price, 2)
            else:
                self.stopPct = round((self.stop - abs_avg_price) / abs_avg_price, 2)

        self.stopPts = round(self.stop - abs(self.average_price), 8)

        # just stop weird floating point math from leaking out
        self.stop = round(self.stop, 8)
        self.average_price = round(self.average_price, 8)

    @property
    def is_long(self) -> bool:
        """Determine if this is a long or short position based on average_price sign.

        CRITICAL: In PositionSummary, direction is determined by average_price > 0:
        - Positive average_price = Long position (is_long=True)
        - Negative average_price = Short position (is_long=False)

        This differs from Position.is_long which uses qty > 0, because PositionSummary
        transforms the data into a display format where:
        - qty is always positive (magnitude)
        - average_price sign indicates direction

        This design ensures consistent interpretation across the system while providing
        clear visual indication of position direction through price sign.
        """
        return self.average_price > 0


@dataclass(slots=True)
class Trade:
    """Represents a single executed trade - the foundational input to the position tracking system.

    CRITICAL INPUT FORMAT REQUIREMENTS:
    - price: MUST be positive (never negative) - validated in __post_init__
    - qty: SIGNED integer/float - positive for BUY/long, negative for SELL/short
    - commission: Positive value (cost of trade execution)

    EXAMPLES:
    BUY 10 shares at $100: Trade(orderid=1, price=100.0, qty=10)
    SELL 10 shares at $100: Trade(orderid=1, price=100.0, qty=-10)

    This is the ONLY place in the system where short positions are indicated by negative
    quantity. All higher-level classes transform this into their specific representations.

    The orderid groups related trades together for spread tracking - all legs of a multi-leg
    strategy should share the same orderid to be recognized as a single position group.
    """

    orderid: OrderId

    price: Price = 0.0
    qty: Qty = 0
    commission: Price = 0.0

    # timestamp defaults to now, but you can override on creation if you have better information yourself.
    # We mean this to be the time of execution. There are other times like "submitted time, updated time(s),"
    # but we care about how long a position has been our responsibility after execution at this point.
    timestamp: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )

    def __post_init__(self):
        # Verify we don't get negative prices.
        # We run our math as (price * -qty) for shorts so don't allow negative prices or the math breaks.
        if self.price < 0:
            raise ValueError(
                "Negative prices are not allowed. Use negative quantities to indicate a short sale."
            )

    @property
    def average_price(self) -> float:
        """Return average price accounting for commissions."""

        if self.qty == 0:
            return 0.0

        return (
            (self.price * self.qty)
            + (self.commission if self.qty > 0 else -self.commission)
        ) / self.qty


@dataclass(slots=True)
class Position:
    """Represents an aggregated position for a single instrument across multiple trades.

    CRITICAL DATA FORMAT:
    - qty: SIGNED (sum of all trade quantities - negative for net short positions)
    - average_price: ALWAYS positive (mathematical result of weighted average cost)
    - is_long: Determined by qty > 0 (quantity sign indicates fundamental position direction)

    MATHEMATICAL BEHAVIOR:
    Long example: 10 shares at $100 → qty=10, average_price=100.0, is_long=True
    Short example: -10 shares at $100 → qty=-10, average_price=100.0 (due to -1000/-10), is_long=False

    KEY INSIGHT: For short positions, average_price is positive due to the math (-cost/-qty = +price)
    but is_long correctly identifies direction via qty sign. This is transformed in PositionSummary
    for display purposes where shorts show negative average_price.

    POSITION LIFECYCLE:
    - Trades are aggregated by orderid (duplicate orderids are rejected)
    - If position qty becomes zero, all trade history is cleared (resets cost basis)
    - This zero-clearing behavior is critical for proper cost basis tracking
    """

    key: Key
    trades: dict[OrderId, Trade] = field(default_factory=dict)

    def __hash__(self) -> int:
        """Position objects are unique by their key only. Changes in trades doesn't change a position meaning itself."""
        return hash(self.key)

    def add_trade(self, trade: Trade) -> bool:
        """Add a new trade to the execution history for this position/symbol/instrument.

        Executed trades are assumed to be unique by their Trade.orderid field, so if the
        orderid already exists in our trade history, we deny adding it again.

        Important Note: If adding this trade results in the position's total quantity becoming zero
        (i.e., self.empty becomes True), all trades associated with this Position object are cleared.
        This effectively resets the position's history and cost basis. Future trades for this
        key will start from a clean slate. This behavior is crucial to understand, especially
        for multi-leg positions (spreads) where closing one leg might clear its history
        if not managed carefully in the context of the overall spread.
        """
        if trade.orderid in self.trades:
            return False

        self.trades[trade.orderid] = trade

        # If adding this trade REMOVES THE POSITION (qty becomes 0), we DROP THE POSITION LOG.
        # This resets the cost basis for this key.
        # TODO: Investigate impacts on spread tracking if legs are closed/reopened individually.
        #       Consider if alternative handling for zero-qty state is needed for spreads.
        if self.empty:
            self.trades.clear()

            # Signal that the trade was "added" but resulted in the position being cleared.
            # The caller might interpret this differently than a simple non-add.
            # For now, returning False indicates the trade isn't "active" in self.trades.
            return False

        # succesfully saved new trade update
        return True

    def add_commission(self, orderid, commission):
        """Add commission to existing execution event.

        Some systems report commissions on a delay versus actual trade executions, so we allow
        this field to be updated after Trade creation."""
        if orderid not in self.trades:
            raise KeyError(
                f"Cannot add commission for orderid {orderid} - no such trade exists in position {self.key}"
            )
        self.trades[orderid].commission = commission

    @property
    def started(self) -> datetime.datetime:
        if not self.trades:
            raise ValueError("Cannot get start time for position with no trades")
        return min([t.timestamp for t in self.trades.values()])

    @property
    def orderids(self) -> set[OrderId]:
        return set(self.trades.keys())

    @property
    def qty(self) -> float:
        return sum([trade.qty for trade in self.trades.values()])

    @property
    def empty(self) -> bool:
        return self.qty == 0

    @property
    def average_price(self) -> float:
        """Calculate weighted average cost per share including commissions.

        CRITICAL MATHEMATICAL BEHAVIOR:
        Result is ALWAYS positive due to mathematical calculation, even for short positions.

        Formula: (sum of price*qty + commissions) / total_qty

        Examples:
        - Long: (100*10 + 5) / 10 = 100.5 (positive)
        - Short: (100*-10 + -5) / -10 = (-1005) / (-10) = 100.5 (positive!)

        For short positions, both numerator and denominator are negative, yielding positive result.
        This is mathematically correct but can be conceptually confusing - the average_price
        represents the absolute cost basis per share, not the directional value.

        Direction is determined separately via is_long property based on qty sign.
        """

        pq = 0.0
        tq = 0.0
        for trade in self.trades.values():
            pq += trade.price * trade.qty + (
                trade.commission if trade.qty > 0 else -trade.commission
            )
            tq += trade.qty

        return pq / (tq or 1)

    @property
    def total_commission(self) -> float:
        return sum([trade.commission for trade in self.trades.values()])

    @property
    def is_long(self) -> bool:
        """Determine position direction based on net quantity sign.

        CRITICAL: This uses qty > 0, NOT average_price > 0.

        This is the fundamental position direction determination:
        - Positive qty = Long position (more bought than sold)
        - Negative qty = Short position (more sold than bought)

        This differs from PositionSummary.is_long which uses average_price > 0
        because Position represents raw aggregated trade data while PositionSummary
        represents transformed display data.

        Examples:
        - Buy 10, qty=10 → is_long=True
        - Sell 10, qty=-10 → is_long=False
        - Buy 15, Sell 10, qty=5 → is_long=True
        """
        return self.qty > 0


@dataclass(slots=True)
class PositionGroup:
    """A holder for positions all trading together as a bag/spread.

    The purpose of PositionGroup is to provide full instrument-like data but
    across multiple positions for things like offsetting vertical spreads where
    the actual average cost of one leg isn't very useful because the true value
     you are holding is offset via another leg we need to account for too."""

    positions: set[Position] = field(default_factory=set)

    # unique composite key for this combination of positions
    key: str | None = None

    def add(self, p: Position):
        self.positions.add(p)
        self.updatekey()

    def updatekey(self):
        """Create a simple way to identify this combination of orders using a short string."""
        symbols = tuple(sorted(self.positions, key=lambda x: str(x.key)))
        composite = abs(hash(symbols))
        shorter = composite >> 40

        # 3 byte keys encoded to 4 byte strings should be enough for anybody...
        self.key = base64.urlsafe_b64encode(
            shorter.to_bytes(byteorder="big", length=3)
        ).decode()

    @property
    def keys(self) -> set[Key]:
        """Return a set of position keys for all positions in this group."""
        return {p.key for p in self.positions}

    def generateOrderDesc(self, openclose: Literal["OPEN", "CLOSE"]) -> str:
        """Generate an open or close order description (assuming the position key is something you can use to order again)"""
        legs = []
        ratios = [abs(p.qty) for p in self.positions]

        # Verify all positions have equal ratios (for proper spread structure)
        if ratios and not all([ratios[0] == n for n in ratios]):
            # This indicates an unbalanced spread which may need special handling
            pass  # For now, continue with generation but could add logging/warning

        selfsize = self.qty

        # Prevent division by zero in ratio calculations
        if selfsize == 0:
            raise ValueError(
                "Cannot generate order description for position group with zero quantity"
            )

        for p in self.positions:
            if openclose == "OPEN":
                if p.is_long:
                    side = "BUY"
                else:
                    side = "SELL"
            else:
                if p.is_long:
                    side = "SELL"
                else:
                    side = "BUY"

            # ratio is LEG quantity divided by SPREAD QUANTITY
            ratio = p.qty / selfsize
            iratio = int(ratio)
            assert iratio == ratio, (
                f"How did you end up with a fractional position ratio? {ratio=} = {p.qty=} / {selfsize=}"
            )

            legs.append(f"{side} {abs(iratio)} {p.key}")

        return " ".join(legs)

    def open(self, *args) -> str:
        """Return an OrderLang string for buying more of this group."""
        return self.generateOrderDesc("OPEN")

    def close(self, *args) -> str:
        """Return an OrderLang string for closing this group.

        Note: this probably isn't useful because we generally accept closing as:
            SELL "BUY 1 X" for 3.33
            instead of closing as
            BUY "SELL 1 X" for credit -3.33
        """
        return self.generateOrderDesc("CLOSE")

    def start(self, stopPct: float = 0.10, algo: str = "LIM") -> str:
        """Generate a buylang opening order limited by increasing your cost basis by at most `stopPct`"""
        summary = self.summary(stopPct=-stopPct)

        # long close means SHORT EXIT (negative qty)
        # short close means LONG EXIT (regular positive qty)
        buysell = 1 if self.is_long else -1

        cmd = f"buy '{self.open()}' {buysell} {algo} @ {summary.stop}"
        return cmd

    def stop(self, stopPct: float = 0.10, algo: str = "LIM") -> str:
        """Generate a buylang closing order for the current stop limit."""
        summary = self.summary(stopPct=stopPct)

        qty = self.qty

        # long close means SHORT EXIT (negative qty)
        # short close means LONG EXIT (regular positive qty)
        buysell = -qty if self.is_long else qty

        # if value is same as an integer as the current qty value, then use the integer version
        if (ibuy := int(buysell)) == buysell:
            buysell = ibuy

        cmd = f"buy '{self.open()}' {buysell} {algo} @ {summary.stop}"
        return cmd

    @property
    def orderids(self) -> set[OrderId]:
        result = set()
        for p in self.positions:
            result |= p.orderids

        return result

    @property
    def qty(self) -> float:
        """Quantity for a position group is determined on the kind of positions being held in a group.

        The trick here is we don't want the long and short legs to cancel quantity out (LONG 8, SHORT -8),
        but if we have dual same side (LONG 8, LONG 8) those should add together.

        So a (LONG 8, SHORT -8) is still QTY 8 and also (LONG 8, LONG 8) is also QTY 8 (because it's a 1:8 ratio).
        But also a (LONG 1, SHORT -2, LONG 1) is also a QTY 1 butterfly.

        There is no general solution to the more-than-two leg problem without creating special cases
        for every symbol and ratio and quantity, but we aren't exposing exact instrument details here
        (we are only operating on quantity and price information).

        Note: this returns ABSOLUTE QUANTITY VALUE without regard to long or short interest. The purpose
              of reporting a group's side is for trading a group as a whole for, likely, closing the
              group as also an atomic unit.
        """

        # basically, the size is the smallest absolute quantity recorded in this group.
        # If you have broken spreads like (LONG 10, SHORT -8) this reports the smallest common number,
        # so you get a full quantity to close full matching positions (in this case, for (10, -8) it would report qty '8').

        # This probably doesn't hold for more weird things like split leg jade dragonfly condors, but
        # it should cover most common use cases.
        # (LONG 8, SHORT -8) == 8
        # (LONG 8, LONG 8) == 8
        # (LONG 1, SHORT -2, LONG 1) == 1
        # (LONG 2, SHORT -4, LONG 2) == 2
        return min([abs(p.qty) for p in self.positions])

    def summary(
        self,
        stopPct: Percent = 0.10,
        priceFetcher: Callable[[Key], float] | None = None,
    ) -> PositionSummary:
        """Generate collective metrics across all positions in the group as one quantity.

        You can optionally pass in a priceFetcher function to use for looking up a live price
        for each instrument in the group. The 'Key' of each position is passed to the priceFetcher
        as priceFetcher(position.key) and priceFecher should return a float for the current price.

        Note: if your price fetcher doesn't have a price, it can return 'nan'.
        """
        avgprice = self.average_price
        is_long = avgprice > 0

        marketPrice = None
        if priceFetcher:
            # market prices are always positive. only position quantities and average prices can be negative.
            marketPrice = abs(
                sum(
                    [
                        priceFetcher(p.key) * (1 if p.is_long else -1)
                        for p in self.positions
                    ]
                )
            )

            # if a NaN polluted our market price calculations, just make it nothing again
            if marketPrice != marketPrice:
                marketPrice = None

        if is_long:
            # for longs, we want to anchor the stop value on the highest of avg price or market price.
            useprice = max(avgprice, marketPrice or float("-inf"))

            # "long" stop-outs are LOWER than the benchmark price
            stop = useprice * (1 - stopPct)
        else:
            # for shorts, we want to anchor the stop on the lowest of (positive) avg price vs the market price.
            # shorts have negative prices, so convert back to flat for our "tradable price" math:
            useprice = min(abs(avgprice), marketPrice or float("inf"))

            # "short" stop-outs are HIGHER than the benchmark price
            stop = useprice * (1 + stopPct)

        return PositionSummary(
            qty=self.qty,
            average_price=avgprice,
            stop=stop,
            total_commission=self.total_commission,
            started=self.started,
            marketPrice=marketPrice,
        )

    @property
    def started(self) -> datetime.datetime:
        return min([p.started for p in self.positions])

    @property
    def average_price(self) -> float:
        """Generate average price across ALL members by re-running the math for each trade in each position."""

        # basically dot product / total weight, but with the commissions added in per trade.
        # Note: commissions always _increase_ cost basis (unless they are negative), so for long trades,
        #       commissions are ADDED while for short trades commissions are SUBTRACTED.
        totalQty = self.qty

        # If totalQty is 0, this indicates all positions have zero quantity, which suggests a data consistency issue
        # Rather than hiding this with a default return, let the division by zero expose the problem

        # We want to guard against a partial record case where we have added one leg of a spread, but the
        # second leg hasn't been added yet. We don't want the average price to be distorted by unequal partial
        # legs larger than the "total size" we are currently reporting.
        # The `totalQty` for the group is `min(abs(p.qty) for p in self.positions)`.
        # When summing values from individual trades within each position, we only consider
        # contributions from trades as long as the cumulative absolute quantity for that position
        # (processed in timestamp order) does not exceed the group's `totalQty`.
        # For example, in a 1x2 ratio spread (e.g., BUY 1 X, SELL 2 Y), `totalQty` would be 1.
        # The average price calculation will effectively use the cost of 1 X and 1 Y,
        # pricing the "1x1 unit" of the spread.
        total_value = 0.0
        for position in self.positions:
            pos_cumulative_qty_considered = 0.0
            # Process trades in chronological order to correctly apply the cumulative quantity cap
            sorted_trades = sorted(position.trades.values(), key=lambda t: t.timestamp)

            for trade in sorted_trades:
                if pos_cumulative_qty_considered >= totalQty:
                    break  # Already met the group's effective quantity for this leg

                trade_qty_abs = abs(trade.qty)

                # Quantity of this trade to consider for the group's average price
                qty_to_consider = min(
                    trade_qty_abs, totalQty - pos_cumulative_qty_considered
                )

                if qty_to_consider > 0:
                    # Calculate the value contribution of the portion of the trade being considered
                    # If trade.qty is 0 or trade_qty_abs is 0, this avoids division by zero
                    ratio_of_trade_considered = (
                        qty_to_consider / trade_qty_abs if trade_qty_abs != 0 else 0
                    )

                    value_contribution = (
                        trade.price * trade.qty * ratio_of_trade_considered
                    )
                    # Assuming commission scales with the portion of the trade considered
                    commission_contribution = (
                        trade.commission * ratio_of_trade_considered
                    )

                    total_value += value_contribution + (
                        commission_contribution
                        if trade.qty > 0
                        else -commission_contribution
                    )
                    pos_cumulative_qty_considered += qty_to_consider

        # Division by totalQty - if zero, this exposes a data consistency issue rather than hiding it
        return total_value / totalQty

    @property
    def total_commission(self) -> float:
        return sum([p.total_commission for p in self.positions])

    @property
    def is_long(self) -> bool:
        return self.average_price > 0


@dataclass(slots=True)
class OrderMgr:
    """Manage a portfolio full of instruments having multiple trades.

    The purpose of OrderMgr is primarily two things:
      - track cost basis details across all purchaes in real time
      - automatically assemble multi-instrument atomic purchases (spreads) into a single inventory unit

    Since we record trades happening _with_ their order id, we can track when a single trade has
    multiple legs executed inside of it, then we can use the per-leg, but single trade, knowledge
    to calculate summary stop, commission, cost basis details for those units.

    Currently we assume multi-instrument trades are occurring atomically, so if you "leg into" a spread,
    this system won't discover the spread by itself. We may want to add a way to edit the "order id" of
    a trade to designate a position as matching another position offset (we could of course be more clever
    about "detecting spreads" automatically, but by tracking multiple instruments executing trades against
    the same order id, it is then fairly trivial to see "which instruments offset each other."

    The OrderMgr is also designed to be persistent across restarts so we store everything locally as best
    as possible, but also note the OrderMgr must be _online_ when a trade executes to receive the trade
    notification of a new trade with its order ID being added. You will have to attach your own live event
    system, and potentially your own back-fill system, to maintain consistent state over time.
    """

    namespace: str

    # use a static, persistent positions mapping...
    positions: MutableMapping[Key, Position] = field(init=False)

    def __post_init__(self):
        self.namespace = self.namespace.replace(" ", "-").title()
        self.positions = DualCache(cacheName=self.namespace, cachePrefix="./positions-")  # type: ignore[assignment]

    def clear(self) -> None:
        """Remove all positions to start fresh"""
        self.positions.clear()

    @classmethod
    @contextmanager
    def temp(cls, name: str | None = None, keep=False):
        """Create a uniquely namespaced test instance then delete when complete"""
        import random

        if not name:
            name = f"Test Instance {random.randint(0, 200_000)}"

        created = cls(name)
        yield created

        if not keep:
            created.positions.destroy()  # type: ignore[attr-defined]

    def add_trade(self, key: Key, trade: Trade):
        """Note: ONLY add trades using .add_trade() here so the changes persist.

        If you attempt to modify self.positions directly, the changes won't be persisted.
        """
        if not (pos := self.positions.get(key)):
            pos = Position(key)

        # only record an update if the position is added as new. If this is a double-back-fill event,
        # don't add extra trades.
        was_existing_position = key in self.positions
        trade_added = pos.add_trade(trade)

        if trade_added:
            # yes, this looks backwards, but we are re-setting it each time so it saves to persistent storage
            self.positions[key] = pos
        elif pos.empty and was_existing_position:
            # Position became empty after adding trade AND was previously stored, remove it from storage
            del self.positions[key]

    def update_commission(self, key: Key, orderid: OrderId, commission: Price) -> None:
        """Note: ONLY update commissions .update_commission() here so the changes persist.

        If you attempt to modify Trade objects or self.positions directly, the changes won't be persisted.
        """
        self.positions[key].add_commission(orderid, commission)

        # yes, this looks backwards, but we are re-setting it each time so it saves to persistent storage
        self.positions[key] = self.positions[key]

    def get_position_summary(
        self, key: Key, stopPct: Percent = 0.10
    ) -> PositionSummary:
        position = self.positions[key]
        stop = abs(
            position.average_price
            * ((1 - stopPct) if position.is_long else (1 + stopPct))
        )

        return PositionSummary(
            qty=abs(position.qty),
            average_price=position.average_price
            if position.is_long
            else -position.average_price,
            stop=stop,
            total_commission=position.total_commission,
            started=position.started,
        )

    def get_position_summary_by_order(
        self, orderid: OrderId, stopPct: Percent = 0.10
    ) -> PositionSummary | None:
        """
        Get a PositionSummary for a group of positions that share a common orderid.
        Returns None if no such group is found or if the orderid links to multiple distinct groups (which shouldn't happen).
        """
        pgs_dict = self.position_groups(orderid)
        if not pgs_dict:
            return None

        # All PositionGroup objects in the values should be identical if they share the same orderid group.
        # We take the first one.
        unique_group_ids = set(id(g) for g in pgs_dict.values())
        if len(unique_group_ids) == 1:
            return list(pgs_dict.values())[0].summary(stopPct)
        elif not unique_group_ids:  # Should be caught by `if not pgs_dict`
            return None
        else:
            # This case (multiple unique groups for a single orderid) should ideally not occur
            # with the current grouping logic if an orderid is meant to define a single atomic trade unit.
            # For safety, return None or raise an error.
            # For now, let's assume the first group encountered is representative if this unlikely state occurs.
            # Or, more robustly:
            # logging.warning(f"Order ID {orderid} linked to multiple distinct position groups. This is unexpected.")
            return list(pgs_dict.values())[0].summary(stopPct)  # Or return None / raise

    def get_portfolio_summary(
        self, stopPct: Percent = 0.10
    ) -> dict[str, PositionSummary]:
        """Return all positions without matching any combined order details."""
        return {
            str(key): self.get_position_summary(key, stopPct)
            for key in self.positions.keys()
        }

    def position_groups(
        self, orderid: OrderId | None = None
    ) -> dict[Key, PositionGroup]:
        """Combine positions having shared orderIds into PositionGroups for multi-reporting."""
        pg: dict[Key, PositionGroup] = {}
        position_to_group: dict[Key, PositionGroup] = {}

        positionOrderIdsMapping = {key: p.orderids for key, p in self.positions.items()}

        # we want to combine all positions having orderids in common into the same PositionGroup group.
        # For each (key, orderId) we want to generate a (key, PositionGroup) in N^2 fashion when order ids match.

        # Is this efficient? No. Is it fast enough? Yes.
        # We could potentially be building these when new trades are added, but as long as we aren't rebuilding this
        # 100 times per second, it should be fine for a while. If you have enough positions where this starts
        # causing performance problems you can probably hire your own people to write bigger systems.
        # You could also think "why aren't we just indexing by orderid in the first place at insert time
        # to avoid all this discovery looping at lookup time?" and, well, uh, don't ask.

        # Basically: N^2 compare all positions against each other to find positions sharing
        #            the same orderids across any of their trades.
        for k, orderids in positionOrderIdsMapping.items():
            pos = self.positions[k]

            # Check if we should include this position based on orderid filter
            if orderid and orderid not in pos.orderids:
                continue

            # If this position is already in a group, use that group
            if k in position_to_group:
                group = position_to_group[k]
            else:
                group = PositionGroup()
                group.add(pos)
                position_to_group[k] = group

            # Look for other positions that share orderids with this one
            for nk, norderids in positionOrderIdsMapping.items():
                if k == nk:  # Don't compare with self
                    continue

                npos = self.positions[nk]

                # Check if we should include this position based on orderid filter
                if orderid and orderid not in npos.orderids:
                    continue

                # add to group if these positions share an order id in common
                if pos.orderids & npos.orderids:
                    group.add(npos)
                    position_to_group[nk] = group

        # Add any remaining positions that weren't matched
        for k in positionOrderIdsMapping:
            if k not in position_to_group:
                pos = self.positions[k]
                if not orderid or orderid in pos.orderids:
                    group = PositionGroup()
                    group.add(pos)
                    position_to_group[k] = group

        # Build the final result dict
        for k, group in position_to_group.items():
            pg[k] = group

        return pg

    # ==============================================================================
    # INTEGRATED DIAGNOSTIC AND REPORTING METHODS
    # ==============================================================================

    def health_check(self) -> SystemHealthReport:
        """Run comprehensive health check on this OrderMgr instance.

        Convenience method that creates a SystemDiagnostics instance and runs health_check.

        Returns:
            SystemHealthReport with structured health information
        """
        diagnostics = SystemDiagnostics(self)
        return diagnostics.health_check()

    def validate_consistency(self) -> list[ValidationResult]:
        """Validate data consistency across all positions.

        Convenience method for DataValidator.validate_system_consistency.

        Returns:
            List of ValidationResult objects for each position
        """
        return DataValidator.validate_system_consistency(self)

    def position_table(self) -> str:
        """Generate formatted table of all positions.

        Convenience method that creates a SystemReporter instance and generates position_table.

        Returns:
            Formatted table string showing all positions
        """
        reporter = SystemReporter(self)
        return reporter.position_table()

    def portfolio_report(self, stopPct: float = 0.10) -> str:
        """Generate comprehensive portfolio summary report.

        Convenience method that creates a SystemReporter instance and generates position_summary.

        Args:
            stopPct: Stop percentage for risk calculations

        Returns:
            Formatted summary string with risk analysis
        """
        reporter = SystemReporter(self)
        return reporter.position_summary(stopPct=stopPct)

    def spread_report(self) -> str:
        """Generate spread analysis report.

        Convenience method that creates a SystemReporter instance and generates spread_analysis.

        Returns:
            Formatted spread analysis string
        """
        reporter = SystemReporter(self)
        return reporter.spread_analysis()

    def analyze_position(self, key: Key) -> PositionAnalysis | None:
        """Deep analysis of a specific position.

        Convenience method that creates a SystemDiagnostics instance and analyzes position.

        Args:
            key: Position key to analyze

        Returns:
            PositionAnalysis with detailed trade flow and calculations, or None if not found
        """
        diagnostics = SystemDiagnostics(self)
        return diagnostics.analyze_position(key)

    def debug_summary(self) -> str:
        """Generate a complete debug summary of the system.

        Combines health check, position table, and portfolio report for comprehensive overview.

        Returns:
            Multi-section formatted report string
        """
        lines = [
            "ORDERMGR DEBUG SUMMARY",
            "=" * 80,
            "",
        ]

        # Health check section
        health = self.health_check()
        lines.extend(
            [
                f"SYSTEM HEALTH: {health.status.upper()}",
                f"Total Positions: {health.total_positions}",
                f"Total Trades: {health.total_trades}",
                "",
            ]
        )

        if health.warnings:
            lines.extend(
                ["WARNINGS:", *[f"  - {warning}" for warning in health.warnings], ""]
            )

        if health.errors:
            lines.extend(["ERRORS:", *[f"  - {error}" for error in health.errors], ""])

        # Position table
        lines.extend(
            [
                "POSITION TABLE:",
                self.position_table(),
                "",
            ]
        )

        # Portfolio summary
        lines.extend(
            [
                self.portfolio_report(),
                "",
            ]
        )

        # Spread analysis
        lines.extend(
            [
                self.spread_report(),
            ]
        )

        return "\n".join(lines)


# ==============================================================================
# DIAGNOSTIC DATA CLASSES
# ==============================================================================


@dataclass
class SystemHealthReport:
    """Structured health check report with proper typing.

    Provides comprehensive health status of the OrderMgr system including positions, trades,
    data consistency violations, mathematical inconsistencies, and overall system status.

    Attributes:
        status: Overall health status - "healthy", "warnings", or "errors"
        total_positions: Total number of positions being tracked
        total_trades: Total number of trades across all positions
        warnings: List of warning messages indicating potential issues
        errors: List of critical error messages requiring immediate attention
        violations: List of data consistency violations found during analysis

    Usage:
        health = order_mgr.health_check()
        if health.status != "healthy":
            print(f"System has {len(health.warnings)} warnings and {len(health.errors)} errors")
            for violation in health.violations:
                print(f"Violation: {violation}")
    """

    status: Literal["healthy", "warnings", "errors"]
    total_positions: int
    total_trades: int
    warnings: list[str]
    errors: list[str]
    violations: list[str]


@dataclass
class TradeFlowEntry:
    """Single entry in position trade flow analysis.

    Represents one trade in the chronological flow of trades for a position,
    including running calculations to show cumulative effects.

    Attributes:
        orderid: Order ID for this trade
        timestamp: When the trade occurred
        price: Trade execution price (always positive)
        qty: Trade quantity (positive for buys, negative for sells)
        commission: Commission paid on this trade
        running_qty: Cumulative quantity after this trade
        running_cost: Cumulative cost basis after this trade
        running_avg: Average price after this trade (running_cost / running_qty)

    Usage:
        analysis = order_mgr.analyze_position(key)
        for entry in analysis.trade_flow:
            print(f"Trade {entry.orderid}: {entry.qty}@{entry.price} -> running avg {entry.running_avg}")
    """

    orderid: OrderId
    timestamp: datetime.datetime
    price: float
    qty: float
    commission: float
    running_qty: float
    running_cost: float
    running_avg: float


@dataclass
class PositionCalculations:
    """Position calculation verification data.

    Compares manually calculated values against stored Position values to detect
    calculation inconsistencies or data corruption.

    Attributes:
        final_qty: Quantity calculated by summing all trade quantities
        final_cost: Cost basis calculated by summing all price*qty + commissions
        final_avg_price: Average price calculated as final_cost / final_qty
        position_qty: Actual qty value stored in the Position object
        position_avg_price: Actual average_price stored in the Position object
        qty_matches: True if final_qty matches position_qty within tolerance
        avg_price_matches: True if final_avg_price matches position_avg_price within tolerance

    Usage:
        analysis = order_mgr.analyze_position(key)
        if not analysis.calculations.qty_matches:
            print(f"Quantity mismatch: calculated {analysis.calculations.final_qty} vs stored {analysis.calculations.position_qty}")
    """

    final_qty: float
    final_cost: float
    final_avg_price: float
    position_qty: float
    position_avg_price: float
    qty_matches: bool
    avg_price_matches: bool


@dataclass
class PositionAnalysis:
    """Complete analysis of a single position.

    Provides comprehensive analysis of a position including current state, trade flow history,
    calculation verification, and consistency error detection.

    Attributes:
        key: Position key (instrument identifier)
        qty: Current position quantity (positive for long, negative for short)
        average_price: Current average price (always positive for Position class)
        is_long: True if this is a long position (qty > 0)
        total_commission: Total commissions paid across all trades
        empty: True if position quantity is zero
        started: Timestamp of the first trade in this position
        trade_flow: Chronological list of all trades with running calculations
        calculations: Verification data comparing calculated vs stored values
        consistency_errors: List of data consistency issues found

    Usage:
        analysis = order_mgr.analyze_position('AAPL')
        print(f"Position {analysis.key}: {analysis.qty}@{analysis.average_price}")
        print(f"Trade count: {len(analysis.trade_flow)}")
        if analysis.consistency_errors:
            print(f"Errors found: {analysis.consistency_errors}")
    """

    key: Key
    qty: float
    average_price: float
    is_long: bool
    total_commission: float
    empty: bool
    started: datetime.datetime
    trade_flow: list[TradeFlowEntry]
    calculations: PositionCalculations
    consistency_errors: list[str]


@dataclass
class ValidationResult:
    """Result of data validation checks.

    Represents the validation status of a single position, including specific errors
    found and overall validation result.

    Attributes:
        position_key: Key of the position that was validated
        errors: List of specific validation errors found
        is_valid: True if no errors were found (auto-calculated in __post_init__)

    Usage:
        results = order_mgr.validate_consistency()
        for result in results:
            if not result.is_valid:
                print(f"Position {result.position_key} has {len(result.errors)} errors")
                for error in result.errors:
                    print(f"  Error: {error}")
    """

    position_key: str
    errors: list[str]
    is_valid: bool

    def __post_init__(self):
        self.is_valid = len(self.errors) == 0


# ==============================================================================
# DIAGNOSTIC AND REPORTING CLASSES
# ==============================================================================


@dataclass
class SystemDiagnostics:
    """Comprehensive system health checks and consistency validation.

    Provides deep analysis of data integrity, consistency violations, and system health
    across all positions, trades, and groups. Essential for debugging and monitoring.

    This class performs extensive validation including:
    - Data consistency checks (is_long vs qty, positive prices, etc.)
    - Mathematical verification (recalculating averages, checking for anomalies)
    - Position-level deep analysis with trade flow reconstruction
    - System-wide health assessment

    Attributes:
        order_mgr: The OrderMgr instance to analyze

    Usage:
        diagnostics = SystemDiagnostics(order_mgr)
        health = diagnostics.health_check()
        analysis = diagnostics.analyze_position('AAPL')

    Or use convenience methods on OrderMgr:
        health = order_mgr.health_check()
        analysis = order_mgr.analyze_position('AAPL')
    """

    order_mgr: OrderMgr

    def health_check(self) -> SystemHealthReport:
        """Run comprehensive health checks on the entire system.

        Returns:
            SystemHealthReport with structured health information
        """
        positions = list(self.order_mgr.positions.values())
        total_positions = len(positions)
        total_trades = sum(len(p.trades) for p in positions)

        # Collect all violations and issues
        warnings: list[str] = []
        errors: list[str] = []

        # Validate data consistency
        violations = self._check_data_consistency()
        warnings.extend(violations)

        # Check for mathematical anomalies
        math_issues = self._check_mathematical_consistency()
        warnings.extend(math_issues)

        # Determine overall status
        if errors:
            status: Literal["healthy", "warnings", "errors"] = "errors"
        elif warnings:
            status = "warnings"
        else:
            status = "healthy"

        return SystemHealthReport(
            status=status,
            total_positions=total_positions,
            total_trades=total_trades,
            warnings=warnings,
            errors=errors,
            violations=violations,
        )

    def _check_data_consistency(self) -> list[str]:
        """Check for data consistency violations across the system."""
        violations: list[str] = []

        for key, position in self.order_mgr.positions.items():
            # Check if position.is_long matches qty > 0
            expected_is_long = position.qty > 0
            actual_is_long = position.is_long
            if expected_is_long != actual_is_long:
                violations.append(
                    f"Position {key}: is_long={actual_is_long} but qty={position.qty} (expected is_long={expected_is_long})"
                )

            # Check if average_price is positive (should always be for Position class)
            if position.average_price < 0:
                violations.append(
                    f"Position {key}: average_price={position.average_price} is negative (should always be positive)"
                )

            # Check trades for negative prices
            for orderid, trade in position.trades.items():
                if trade.price < 0:
                    violations.append(
                        f"Position {key}, Trade {orderid}: price={trade.price} is negative (not allowed)"
                    )

        return violations

    def _check_mathematical_consistency(self) -> list[str]:
        """Check for mathematical anomalies that might indicate calculation errors."""
        issues = []

        for key, position in self.order_mgr.positions.items():
            # Check for unusual average_price calculations
            if position.trades:
                # Manual calculation to verify
                pq = sum(
                    t.price * t.qty + (t.commission if t.qty > 0 else -t.commission)
                    for t in position.trades.values()
                )
                tq = sum(t.qty for t in position.trades.values())
                expected_avg = pq / (tq or 1)

                if abs(position.average_price - expected_avg) > 0.001:
                    issues.append(
                        f"Position {key}: average_price calculation mismatch "
                        f"(got {position.average_price}, expected {expected_avg})"
                    )

        return issues

    def analyze_position(self, key: Key) -> PositionAnalysis | None:
        """Deep analysis of a specific position.

        Args:
            key: Position key to analyze

        Returns:
            PositionAnalysis with detailed trade flow and calculations, or None if not found
        """
        if key not in self.order_mgr.positions:
            return None

        position = self.order_mgr.positions[key]

        # Analyze trade flow
        sorted_trades = sorted(position.trades.values(), key=lambda t: t.timestamp)
        running_qty: float = 0.0
        running_cost: float = 0.0
        trade_flow: list[TradeFlowEntry] = []

        for trade in sorted_trades:
            running_qty += trade.qty
            running_cost += trade.price * trade.qty + (
                trade.commission if trade.qty > 0 else -trade.commission
            )

            trade_flow.append(
                TradeFlowEntry(
                    orderid=trade.orderid,
                    timestamp=trade.timestamp,
                    price=trade.price,
                    qty=trade.qty,
                    commission=trade.commission,
                    running_qty=running_qty,
                    running_cost=running_cost,
                    running_avg=running_cost / (running_qty or 1),
                )
            )

        # Calculate verification data
        calculations = PositionCalculations(
            final_qty=running_qty,
            final_cost=running_cost,
            final_avg_price=running_cost / (running_qty or 1),
            position_qty=position.qty,
            position_avg_price=position.average_price,
            qty_matches=abs(running_qty - position.qty) < 0.001,
            avg_price_matches=abs(
                running_cost / (running_qty or 1) - position.average_price
            )
            < 0.001,
        )

        # Check for consistency issues
        consistency_errors = DataValidator.validate_position_consistency(position)

        return PositionAnalysis(
            key=key,
            qty=position.qty,
            average_price=position.average_price,
            is_long=position.is_long,
            total_commission=position.total_commission,
            empty=position.empty,
            started=position.started,
            trade_flow=trade_flow,
            calculations=calculations,
            consistency_errors=consistency_errors,
        )


@dataclass
class SystemReporter:
    """Generate formatted reports and tables for system visualization.

    Provides various reporting formats including tables, summaries, and detailed breakdowns
    for easy analysis and monitoring of the entire portfolio state.

    This class creates human-readable reports from OrderMgr data including:
    - Position tables with key metrics
    - Portfolio summaries with risk analysis
    - Spread analysis for multi-leg positions
    - Formatted output suitable for console display or logging

    Attributes:
        order_mgr: The OrderMgr instance to generate reports from

    Usage:
        reporter = SystemReporter(order_mgr)
        print(reporter.position_table())
        print(reporter.position_summary(stopPct=0.05))
        print(reporter.spread_analysis())

    Or use convenience methods on OrderMgr:
        print(order_mgr.position_table())
        print(order_mgr.portfolio_report(stopPct=0.05))
        print(order_mgr.spread_report())
    """

    order_mgr: OrderMgr

    def position_table(self) -> str:
        """Generate a formatted table of all positions."""
        if not self.order_mgr.positions:
            return "No positions found."

        # Table headers
        headers = [
            "Key",
            "Qty",
            "Avg Price",
            "Direction",
            "Commission",
            "Trades",
            "Started",
        ]

        # Calculate column widths
        col_widths = [len(h) for h in headers]

        rows = []
        for key, pos in self.order_mgr.positions.items():
            row = [
                str(key),
                f"{pos.qty:,.1f}",
                f"${pos.average_price:,.2f}",
                "LONG" if pos.is_long else "SHORT",
                f"${pos.total_commission:,.2f}",
                str(len(pos.trades)),
                pos.started.strftime("%Y-%m-%d %H:%M"),
            ]
            rows.append(row)

            # Update column widths
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))

        # Build table
        separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

        lines = [separator]

        # Headers
        header_line = (
            "|"
            + "|".join(f" {headers[i]:<{col_widths[i]}} " for i in range(len(headers)))
            + "|"
        )
        lines.append(header_line)
        lines.append(separator)

        # Data rows
        for row in rows:
            data_line = (
                "|"
                + "|".join(f" {row[i]:<{col_widths[i]}} " for i in range(len(row)))
                + "|"
            )
            lines.append(data_line)

        lines.append(separator)

        return "\n".join(lines)

    def position_summary(self, stopPct: float = 0.10) -> str:
        """Generate a summary report of all positions with stop information."""
        summaries = self.order_mgr.get_portfolio_summary(stopPct=stopPct)

        if not summaries:
            return "No positions found."

        lines = ["PORTFOLIO SUMMARY", "=" * 50, ""]

        total_value: float = 0.0
        total_risk: float = 0.0

        for key, summary in summaries.items():
            position_value = summary.qty * abs(summary.average_price)
            risk_amount = position_value * abs(summary.stopPct)

            total_value += position_value if summary.is_long else -position_value
            total_risk += risk_amount

            lines.extend(
                [
                    f"Position: {key}",
                    f"  Direction: {'LONG' if summary.is_long else 'SHORT'}",
                    f"  Quantity: {summary.qty:,.1f}",
                    f"  Avg Price: ${abs(summary.average_price):,.2f}",
                    f"  Stop Price: ${summary.stop:,.2f}",
                    f"  Stop %: {summary.stopPct:.1%}",
                    f"  Position Value: ${position_value:,.2f}",
                    f"  Risk Amount: ${risk_amount:,.2f}",
                    f"  Age: {summary.age}",
                    "",
                ]
            )

        lines.extend(
            [
                "TOTALS:",
                f"  Net Position Value: ${total_value:,.2f}",
                f"  Total Risk Amount: ${total_risk:,.2f}",
                f"  Risk as % of Total: {(total_risk / abs(total_value) if total_value != 0 else 0):.1%}",
            ]
        )

        return "\n".join(lines)

    def spread_analysis(self) -> str:
        """Analyze and report on all position groups (spreads)."""
        groups = self.order_mgr.position_groups()

        if not groups:
            return "No position groups found."

        lines = ["SPREAD ANALYSIS", "=" * 50, ""]

        # Group by actual PositionGroup object to find real spreads
        unique_groups: dict[int, dict[str, Any]] = {}
        for key, group in groups.items():
            group_id = id(group)
            if group_id not in unique_groups:
                unique_groups[group_id] = {"group": group, "keys": []}
            unique_groups[group_id]["keys"].append(key)

        spread_count = 0
        single_count = 0

        for group_data in unique_groups.values():
            group = group_data["group"]
            keys = group_data["keys"]

            if len(keys) > 1:
                spread_count += 1
                lines.append(f"SPREAD #{spread_count}: {', '.join(map(str, keys))}")
                lines.append(f"  Positions: {len(group.positions)}")
                lines.append(f"  Group Qty: {group.qty}")
                lines.append(f"  Group Avg Price: ${group.average_price:,.2f}")
                lines.append(f"  Direction: {'LONG' if group.is_long else 'SHORT'}")

                # Show individual legs
                for pos in group.positions:
                    lines.append(
                        f"    {pos.key}: qty={pos.qty}, price=${pos.average_price:,.2f}"
                    )

                lines.append("")
            else:
                single_count += 1

        lines.extend(
            [f"Summary: {spread_count} spreads, {single_count} single positions"]
        )

        return "\n".join(lines)


@dataclass
class DataValidator:
    """Input validation and data consistency guards.

    Provides validation functions to ensure data integrity and catch
    violations of the system's critical design principles.

    This class contains static methods for validating various data objects:
    - Trade validation (price, qty, commission, orderid checks)
    - Position validation (calculation consistency, is_long logic)
    - System-wide consistency validation across all positions

    These validators are used by diagnostic systems to detect data corruption,
    calculation errors, and violations of the documented data model.

    Usage:
        errors = DataValidator.validate_trade(trade)
        if errors:
            print(f"Trade validation errors: {errors}")

        errors = DataValidator.validate_position_consistency(position)
        if errors:
            print(f"Position consistency errors: {errors}")

        results = DataValidator.validate_system_consistency(order_mgr)
        invalid_positions = [r for r in results if not r.is_valid]
    """

    @staticmethod
    def validate_trade(trade: Trade) -> list[str]:
        """Validate a Trade object for consistency violations.

        Args:
            trade: Trade object to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Price must be positive
        if trade.price < 0:
            errors.append(f"Trade price {trade.price} is negative (not allowed)")

        # Commission should be non-negative
        if trade.commission < 0:
            errors.append(f"Trade commission {trade.commission} is negative (unusual)")

        # Qty should not be zero for meaningful trades
        if trade.qty == 0:
            errors.append("Trade quantity is zero (may indicate data error)")

        # OrderID should be meaningful
        if not trade.orderid:
            errors.append("Trade orderid is empty/None")

        return errors

    @staticmethod
    def validate_position_consistency(position: Position) -> list[str]:
        """Validate Position object for internal consistency.

        Args:
            position: Position object to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check qty calculation
        calculated_qty = sum(t.qty for t in position.trades.values())
        if abs(calculated_qty - position.qty) > 0.001:
            errors.append(
                f"Position qty mismatch: calculated {calculated_qty}, "
                f"actual {position.qty}"
            )

        # Check is_long consistency
        expected_is_long = position.qty > 0
        if position.is_long != expected_is_long:
            errors.append(
                f"Position is_long inconsistency: is_long={position.is_long}, "
                f"qty={position.qty} (expected is_long={expected_is_long})"
            )

        # Average price should be positive for Position class
        if position.average_price < 0:
            errors.append(
                f"Position average_price {position.average_price} is negative "
                f"(should always be positive in Position class)"
            )

        # Validate all trades
        for orderid, trade in position.trades.items():
            trade_errors = DataValidator.validate_trade(trade)
            for error in trade_errors:
                errors.append(f"Trade {orderid}: {error}")

        return errors

    @staticmethod
    def validate_position_summary_consistency(summary: PositionSummary) -> list[str]:
        """Validate PositionSummary object for consistency.

        Args:
            summary: PositionSummary object to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Qty should be positive
        if summary.qty <= 0:
            errors.append(f"PositionSummary qty {summary.qty} is not positive")

        # is_long should match average_price sign
        expected_is_long = summary.average_price > 0
        if summary.is_long != expected_is_long:
            errors.append(
                f"PositionSummary is_long inconsistency: is_long={summary.is_long}, "
                f"average_price={summary.average_price} (expected is_long={expected_is_long})"
            )

        # Stop should be positive
        if summary.stop <= 0:
            errors.append(f"PositionSummary stop {summary.stop} is not positive")

        return errors

    @staticmethod
    def validate_system_consistency(order_mgr: OrderMgr) -> list[ValidationResult]:
        """Run comprehensive validation on entire OrderMgr system.

        Args:
            order_mgr: OrderMgr instance to validate

        Returns:
            List of ValidationResult objects for each position
        """
        results: list[ValidationResult] = []

        for key, position in order_mgr.positions.items():
            errors = DataValidator.validate_position_consistency(position)
            results.append(
                ValidationResult(
                    position_key=str(key), errors=errors, is_valid=len(errors) == 0
                )
            )

        return results
