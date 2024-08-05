"""orderlang allows full purchase intent including pricing.
(versus buylang which is mainly for symbol and spread declarations)"""

import dataclasses
from dataclasses import dataclass, field
from typing import *

# The parser generator
import lark
from lark import Lark, Transformer, Token, v_args
from lark import UnexpectedToken, UnexpectedCharacters

import prettyprinter as pp  # type: ignore

from itertools import chain
from enum import Flag, Enum
from loguru import logger
from decimal import Decimal

# tell pretty printer to check input for dataclass printing
pp.install_extras(["dataclasses"], warn_on_error=False)

import datetime


# fmt: off
class DecimalPrice(Decimal): ...
class DecimalPercent(Decimal): ...
class DecimalShares(Decimal): ...
class DecimalCash(Decimal): ...
class DecimalLong(Decimal): ...
class DecimalShort(Decimal): ...
class DecimalLongCash(DecimalLong, DecimalCash): ...
class DecimalLongShares(DecimalLong, DecimalShares): ...
class DecimalShortShares(DecimalShort, DecimalShares): ...
class DecimalShortCash(DecimalShort, DecimalCash): ...
# fmt: on


@dataclass(slots=True)
class OrderIntent:
    """Description of a purchase request."""

    symbol: str | None = None
    algo: str | None = None

    # You can optionally provide an exchange for this order if you have specific requirements.
    exchange: str | None = None

    # Quantity is always represented as a POSITIVE VALUE which is then
    # designated long or short by its type. The type can easily be
    # meta-checked via the property handlers.
    # Reason: trading APIs usually say "BUY 100" or "SELL 100" and not "BUY -100" or "SELL -100",
    #         so we maintain quantity as POSITIVE while the BUY/SELL intent is a property of the
    #         order itself we can introspect when needed.
    qty: (
        DecimalLongShares
        | DecimalShortShares
        | DecimalLongCash
        | DecimalShortCash
        | None
    ) = None

    bracketProfitAlgo: str = "LMT"
    bracketLossAlgo: str = "STP"

    limit: DecimalPrice | None = None

    bracketProfit: DecimalPrice | DecimalPercent | None = None
    bracketLoss: DecimalPrice | DecimalPercent | None = None

    preview: bool = False

    @property
    def isLong(self) -> bool:
        return isinstance(self.qty, DecimalLong)

    @property
    def isShares(self) -> bool:
        return isinstance(self.qty, DecimalShares)

    @property
    def isMoney(self) -> bool:
        return isinstance(self.qty, DecimalCash)

    @property
    def isBracketProfitPercent(self) -> bool:
        return isinstance(self.bracketProfit, DecimalPercent)

    @property
    def isBracketLossPercent(self) -> bool:
        return isinstance(self.bracketLoss, DecimalPercent)

    @property
    def shortFix(self) -> int:
        """Indicates if we need to invert the direction of some math to compensate for short profit."""
        return 1 if self.isLong else -1

    @property
    def bracketProfitReal(self) -> Decimal:
        """Generate the _actual_ bracket profit value given the current limit price.

        Takes care of both direction (long vs. short) and value (percent vs. exact).

        NOTE: this interperts the bracket request as POINTS AWAY FROM THE LIMIT PRICE and NOT THE FULL BRACKET EXIT PRICE.

        This means if your limit is $10 and your bracketProfit is $3, this generates a bracketProfitReal == $13.

        You can of course avoid using bracketProfitReal and treat bracketProfit as any value you like as well."""

        # if percent, extract percent of whole to use for addition
        bracketProfit = self.bracketProfit
        if self.isBracketProfitPercent:
            bracketProfit = self.limit * self.bracketProfit / 100

        # else, the bracket price requested is the EXACT price to limit
        return self.limit + (bracketProfit * self.shortFix)

    @property
    def bracketLossReal(self) -> Decimal:
        """Generate the _actual_ bracket loss value given the current limit price.

        Takes care of both direction (long vs. short) and value (percent vs. exact).

        See notes in bracketProfitReal() about how the bracket prices are treated for exit here."""

        # if percent, do basically:  PRICE * 1.profitPercent
        # if percent, extract percent of whole to use for subtraction
        bracketLoss = self.bracketLoss
        if self.isBracketLossPercent:
            bracketLoss = self.limit * self.bracketLoss / 100

        # else, the bracket price requested is the EXACT price to limit
        return self.limit - (bracketLoss * self.shortFix)


lang = r"""

    // BUY something
    // SYMBOL SHARES|PRICE_QUANTITY ALGO [on EXCHANGE] [@ LIMIT_PRICE [+ PROFIT_POINTS ALGO | - LOSS_POINTS ALGO | ± EQUAL_PROFIT_LOSS_POINTS ALGO_PROFIT ALGO_LOSS]] [preview]
    cmd: symbol quantity orderalgo exchange? limit? preview?


    // TODO: we could actually use buylang to parse symbol allowing full spread descriptions here too, but
    //       then we would need to include the buylang Order() object instaed of just a symbol string in our OrderIntent() output.
    symbol: /\/?[:A-Za-z0-9\/\._]{1,15}/

    quantity: shares_short | shares_long | cash_amount_long | cash_amount_short

    shares_short: "-" price
    shares_long: price
    cash_amount_long: "$" price
    cash_amount_short: ("-$" | "$-") price

    orderalgo: /[A-Za-z_]+/
    algo: /[A-Za-z_]+/

    limit: "@" price (profit | loss | bracket)*

    exchange: "ON"i /[A-Za-z]+/

    // PRICE is a helper for "float parsing with optional , or _ allowed" also with unlimited decimal precision allowed
    price: /[0-9_,]+\.?[0-9]*/

    is_percentage: "%"

    // attach TAKE PROFIT order
    profit: "+" price is_percentage? algo?

    // attach STOP LOSS order as direct value OR as a percent difference request
    loss: "-" price is_percentage? algo?

    // attach TAKE PROFIT _and_ STOP LOSS order at equal width with optional PROFIT LOSS algo overrides
    bracket: "±" price (algo algo)?

    preview: "p"i | "pr"i | "pre"i | "prev"i | "preview"i

    WHITESPACE: (" " | "\t" | "\n")+
    COMMENT: /#[^\n]*/

    %import common.NUMBER

    %ignore WHITESPACE
    %ignore COMMENT
"""


@dataclass
class TreeToBuy(Transformer):
    @v_args(inline=False)
    def cmd(self, stuff):
        return self.b

    @v_args(inline=True)
    def symbol(self, got):
        # Symbol is parsed _first_ so we use this rule to always create a NEW order indent for each run
        # so if this parser is re-used we generate a new order on each parse.
        # Also means this parser is not re-entrant, but nobody should be doing such things anyway here.
        self.b = OrderIntent()
        self.b.symbol = got.replace("_", " ").upper()

    @v_args(inline=True)
    def shares_long(self, got):
        self.b.qty = DecimalLongShares(got)

    @v_args(inline=True)
    def shares_short(self, got):
        self.b.qty = DecimalShortShares(got)

    @v_args(inline=True)
    def cash_amount_long(self, got):
        self.b.qty = DecimalLongCash(got)

    @v_args(inline=True)
    def cash_amount_short(self, got):
        self.b.qty = DecimalShortCash(got)

    @v_args(inline=True)
    def limit(self, got, *extra):
        self.b.limit = DecimalPrice(got)

    @v_args(inline=True)
    def exchange(self, got):
        """Allow users to also tell us their desired exchange. Supah Fancy."""
        self.b.exchange = str(got).upper()

    @v_args(inline=True)
    def orderalgo(self, got):
        """A bit of a hack: we are special casing the limit algo to place it directly."""
        self.b.algo = str(got).upper()

    @v_args(inline=True)
    def algo(self, got):
        """This 'algo' is for algos attached to bracket requests."""
        return str(got).upper()

    def preview(self, _):
        self.b.preview = True

    @v_args(inline=True)
    def price(self, got):
        return DecimalPrice(got.replace("_", "").replace(",", ""))

    @v_args(inline=True)
    def profit(self, got, *algo):
        self.b.bracketProfit = got

        if algo:
            if algo[0] is True:
                self.b.bracketProfit = DecimalPercent(got)
                algo = algo[1:]

            if algo:
                self.b.bracketProfitAlgo = str(algo[0]).replace("_", " ")

    @v_args(inline=True)
    def is_percentage(self):
        # This is consumed under profit() and loss() directly as a 2-component 'algo' param.
        return True

    @v_args(inline=True)
    def loss(self, got, *algo):
        self.b.bracketLoss = got

        if algo:
            if algo[0] is True:
                self.b.bracketLoss = DecimalPercent(got)
                algo = algo[1:]

            if algo:
                self.b.bracketLossAlgo = str(algo[0]).replace("_", " ")

    @v_args(inline=True)
    def bracket(self, got, *algos):
        self.b.bracketProfit = got
        self.b.bracketLoss = got

        if algos:
            # if "algos" is present, then BOTH algos exist, and we can just pop them out here:
            # (algos is just a 2-tuple of: (PROFIT_ALGO, LOSS_ALGO)
            self.b.bracketProfitAlgo, self.b.bracketLossAlgo = algos


@dataclass
class OrderLang:
    def __post_init__(self):
        self.parser = Lark(lang, start="cmd", parser="lalr", transformer=TreeToBuy())

    def parse(self, text: str) -> OrderIntent:
        """Parse 'text' conforming to our triggerlang grammar into a
        list of symbol-trigger or watchlist-symbol-trigger results.

        On error, throws the raw lark.UnexpectedToken exception with
        fields '.token' and '.expected' describing where the problem
        occurred and what was expected instead."""

        # input text must be .strip()'d because the parser doesn't
        # like end of input newlines."""
        parsed = self.parser.parse(text.strip())
        return parsed

    def parseDebug(self, text: str) -> OrderIntent:
        """Parse a triggerlang grammar format like .parse() but also
        print the created datastructures to stdout as both their
        dictionary form and their object/dataclass form."""

        parsed = self.parse(text)

        print("\tResult", "as dict:")
        pp.cpprint(dataclasses.asdict(parsed))
        print("\n\tResult", "as class:")
        pp.cpprint(parsed)

        return parsed
