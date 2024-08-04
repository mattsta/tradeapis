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

KIND = Enum("KIND", "SHARES CASH")


@dataclass(slots=True)
class OrderIntent:
    """Description of a purchase request."""

    symbol: str | None = None
    algo: str | None = None

    kind: KIND | None = None

    isLong: bool = True

    bracketProfitAlgo: str = "LMT"
    bracketLossAlgo: str = "STP"

    limit: float | None = None

    # of type noted by KIND
    qty: float | Decimal | str | None = None

    bracketProfit: float | None = None
    bracketLoss: float | None = None

    bracketProfitIsPercent: bool = False
    bracketLossIsPercent: bool = False

    preview: bool = False


lang = r"""

    // BUY something
    // SYMBOL SHARES|PRICE_QUANTITY ALGO [@ LIMIT_PRICE [+ PROFIT ALGO | - LOSS ALGO | ± EQUAL_PROFIT_LOSS ALGO_PROFIT ALGO_LOSS]] [preview]
    cmd: symbol quantity orderalgo limit? preview?

    symbol: /\/?[:A-Za-z0-9\/\._]{1,15}/

    quantity: shares | shares_short | cash_amount | cash_amount_short

    shares: price
    shares_short: "-" shares
    cash_amount: "$" price
    cash_amount_short: "-" cash_amount

    orderalgo: /[A-Za-z_]+/
    algo: /[A-Za-z_]+/

    limit: "@" price (profit | loss | bracket)*

    // PRICE is a helper for "float parsing with optional , or _ allowed"
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
    def shares(self, got):
        self.b.qty = Decimal(got)
        self.b.kind = KIND.SHARES

    @v_args(inline=True)
    def cash_amount(self, got):
        self.b.qty = Decimal(got)
        self.b.kind = KIND.CASH

    @v_args(inline=True)
    def limit(self, got, *extra):
        self.b.limit = Decimal(got)

    @v_args(inline=True)
    def orderalgo(self, got):
        """A bit of a hack: we are special casing the limit algo to place it directly."""
        self.b.algo = str(got).upper()

    def shares_short(self, got):
        self.b.isLong = False

    def cash_amount_short(self, got):
        self.b.isLong = False

    @v_args(inline=True)
    def algo(self, got):
        """This 'algo' is for algos attached to bracket requests."""
        return str(got).upper()

    def preview(self, _):
        self.b.preview = True

    @v_args(inline=True)
    def price(self, got):
        return Decimal(got.replace("_", "").replace(",", ""))

    @v_args(inline=True)
    def profit(self, got, *algo):
        self.b.bracketProfit = got

        if algo:
            if algo[0] is True:
                self.b.bracketProfitIsPercent = True
                algo = algo[1:]

            if algo:
                self.b.bracketProfitAlgo = str(algo[0]).replace("_", " ")

    @v_args(inline=True)
    def is_percentage(self):
        return True

    @v_args(inline=True)
    def loss(self, got, *algo):
        self.b.bracketLoss = got

        if algo:
            if algo[0] is True:
                self.b.bracketLossIsPercent = True
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
