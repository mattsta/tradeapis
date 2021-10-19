import dataclasses
from dataclasses import dataclass, field
from typing import *

# The parser generator
import lark
from lark import Lark, Transformer, Token, v_args
from lark import UnexpectedToken, UnexpectedCharacters

# debug printing helper (colors! actually formats dataclasses properly,
# unlike pprint in Python <= 3.9 where pprint doesn't format them at all)
import prettyprinter as pp  # type: ignore

from itertools import chain
from enum import Flag
from loguru import logger

# tell pretty printer to check input for dataclass printing
pp.install_extras(["dataclasses"], warn_on_error=False)

import datetime

from tradeapis.fees import OptionFees

# Allow specific side-oriented buys as well as generic "BUY" / "SELL" designations
Side = Flag("Side", "BTO STO BTC STC BUY SELL UNSET")

# Custom types (convert to TypeVar after Python 3.10+)
Symbol = str


@dataclass
class Order:
    """Description of one order via buy/sell type, multiplier for quantity,
    and the symbol itself.

    Depending on usage, multiplier may be the ACTUAL share/contrat quantity OR
    it could represent a RATIO for multiplying by 'size' of the entire OrderRequest.
    (e.g. a butterfly will have 1 BTO, 2 STO, 1 BTO multipliers then the entire
          order quantity would be 'size' in the OrderRequest, then the complete
          transaction is (sum(multiplier) * size) etc)"""

    side: Side
    multiplier: int
    symbol: Symbol

    # For more detailed use cases, you can attach limit/stop prices directly
    # to the order symbol description itself too.
    limit: Optional[float] = None
    stop: Optional[float] = None

    def __post_init__(self):
        self.symbol = self.symbol.upper()

    def underlying(self) -> Symbol:
        return self.symbol[:-15]

    def isBuy(self) -> bool:
        """Return if any side is a buy operation.

        The designation of "to Open" or "to Close" is only used
        for position management either locally or at a broker,
        but has no impact on the actual placed order at an exchange.

        For example:
          - IBKR has no concept of "to open" or "to close,"
          you just place buy and sell orders for any position and you
          are expected to manage your total net positions yourself.
          management .
          - Other API brokers like Tradier or Alpaca expect you
          to trasmit your intent as "to Open" or "to Close" so they
          can validate your requests against your position counts
          before committing your request to an exchange."""

        return bool(self.side & (Side.BTO | Side.BTC | Side.BUY))

    def isSell(self) -> bool:
        """Return if any side is a sell operation.

        See .isBuy() for reasons why this aggregate check exists."""
        return bool(self.side & (Side.STO | Side.STC | Side.SELL))

    def isOpen(self) -> bool:
        return bool(self.side & (Side.BTO | Side.STO))

    def isClose(self) -> bool:
        return bool(self.side & (Side.BTC | Side.STC))

    def isCall(self) -> bool:
        if len(self.symbol) > 15:
            return self.symbol[-9] == "C"

        return False

    def isPut(self) -> bool:
        if len(self.symbol) > 15:
            return self.symbol[-9] == "P"

        return False

    def strike(self) -> Optional[float]:
        """ If symbol is an OCC symbol, return strike price as a float. """
        if len(self.symbol) > 15:
            return int(self.symbol[-8:]) / 1000

        return None


@dataclass
class OrderRequest:
    """Represents a single or multi-leg order potentially with custom order arguments.

    Note: only works when using order multiplier as actual multiplier argument and
          not number of contracts for the requested order. Full contract order
          should be calculated as request.size * sum(orders.multiplier)."""

    orders: list[Order]
    size: int = 1  # default to use 'multiplier' quantity of each order
    start: Optional[datetime.datetime] = None
    end: Optional[datetime.datetime] = None

    def isSpread(self) -> bool:
        """ True if more than 1 order is populated, False otherwise."""
        return len(self.orders) > 1

    def isSingle(self) -> bool:
        """ True if exactly 1 order is populated, False otherwise."""
        return len(self.orders) == 1

    def totalQuantity(self) -> float:
        """Returns sum of order multipliers times the order size request.

        Mainly useful if orders is just a single stock where the size was
        specified there and request size ended up as the default 1, but we
        still want to retrieve the total requested quantity."""
        return sum(o.multiplier for o in self.orders) * self.size

    def fees(self) -> float:
        """ Return all regulatory fees for spread (if prices are populated for orders)"""
        legsBuy = [
            (o.multiplier * self.size, o.limit or 0) for o in self.orders if o.isBuy()
        ]
        legsSell = [
            (o.multiplier * self.size, o.limit or 0) for o in self.orders if o.isSell()
        ]

        of = OptionFees(legs_buy=legsBuy, legs_sell=legsSell)
        return of.total

    def isButterfly(self) -> bool:
        """True if request is exactly 3 orders in 1:2:1 ratios, False otherwise.

        Orders are checked for:
            - exactly 3 orders with two having multiplier 1 and one having multiplier 2
            - all orders are calls or puts
            - the 1 ratios are both either all buys or all sells
            - the 2 ratio is the opposite buy/sell side of the 1 ratio

        Doesn't currently check:
            - the width of the strikes
        """

        if len(self.orders) == 3:
            max1 = []
            max2 = []

            for o in self.orders:
                if o.multiplier == 1:
                    max1.append(o)
                elif o.multiplier == 2:
                    max2.append(o)
                else:
                    # early exit because condition violated
                    return False

            # verify we have 2 (1) mults and 1 (2) mult
            if len(max1) == 2 and len(max2) == 1:
                # verify they are all calls or puts
                if all([x.isCall() for x in chain(max1, max2)]) or all(
                    [x.isPut() for x in chain(max1, max2)]
                ):
                    # verify the (2) multiplier is opposite side of (1) mults,
                    # either for 1 buy :: 2 sell :: 1 buy - OR - 1 sell :: 2 buy :: 1 sell
                    if all([x.isBuy() for x in max1]) and max2[0].isSell():
                        return True

                    if all([x.isSell() for x in max1]) and max2[0].isBuy():
                        return True

        return False


lang = r"""

    // Populate a single order request (with no buy/sell side indicated) or
    // create a full multi-symbol spread.
    cmd: single_order | spread

    // OCC symbol is technically 21 characters, but the leading
    // symbol is right-aligned against the date side with spaces removed.
    // also see: http://www.schwabcontent.com/symbology/int_eng/key_details.html
    // Also note: we allow leading slashes so we can pass futures-like
    // symbols through the entire pipline e.g. /MES /MES210604C04200000
    option: /\/?[A-Z]{1,6}\d{6}[PC]\d{8}/

    // Stock is anything else, but for our purpose can also be prefixed
    // with a contract namespace. Like STOCK:BTC vs. CRYPTO:BTC etc.
    stock: /\/?[:A-Za-z\/\.]{1,10}/

    single_order: stock | option

    spread_symbol: stock | option
    spread: side qty spread_symbol (side qty spread_symbol)*

    // optional quantity (defaults to 1 if not provided)
    qty: (/[0-9]+/)?

    side: bto | sto | btc | stc | buy | sell
    bto: "buy_to_open"i | "bto"i
    sto: "sell_to_open"i | "sto"i
    btc: "buy_to_close"i | "btc"i
    stc: "sell_to_close"i | "stc"i
    buy: "buy"i
    sell: "sell"i

    WHITESPACE: (" " | "\t" | "\n")+
    COMMENT: /#[^\n]*/

    %ignore WHITESPACE
    %ignore COMMENT
"""


class TreeToOrder(Transformer):
    @v_args(inline=True)
    def cmd(self, got):
        return OrderRequest(orders=got)

    @v_args(inline=True)
    def single_order(self, got):
        """ Single symbol representing a stock or option with no buy/sell side specified."""
        return [Order(Side.UNSET, 1, got.upper())]

    @v_args(inline=True)
    def stock(self, got):
        return got

    @v_args(inline=True)
    def option(self, got):
        return got

    @v_args(inline=True)
    def spread_symbol(self, got):
        return got.upper()

    def spread(self, got):
        gots = []

        # iterate spread results in groups of 3 to create orders for request
        for s in range(0, len(got), 3):
            gots.append(Order(got[s + 0], got[s + 1], got[s + 2]))

        # logger.info("spread got {}", gots)
        return gots

    def qty(self, got):
        """ Value is optional, so if not present, default to 1"""
        if got:
            return int(got[0])

        return 1

    @v_args(inline=True)
    def side(self, got):
        return got

    def bto(self, _):
        return Side.BTO

    def sto(self, _):
        return Side.STO

    def stc(self, _):
        return Side.STC

    def btc(self, _):
        return Side.BTC

    def buy(self, _):
        return Side.BUY

    def sell(self, _):
        return Side.SELL


@dataclass
class OLang:
    def __post_init__(self):
        self.parser = Lark(lang, start="cmd", parser="lalr", transformer=TreeToOrder())

    def parse(self, text: str) -> OrderRequest:
        """Parse 'text' conforming to our triggerlang grammar into a
        list of symbol-trigger or watchlist-symbol-trigger results.

        On error, throws the raw lark.UnexpectedToken exception with
        fields '.token' and '.expected' describing where the problem
        occurred and what was expected instead."""

        # input text must be .strip()'d because the parser doesn't
        # like end of input newlines."""
        parsed = self.parser.parse(text.strip())
        return parsed

    def parseDebug(self, text: str) -> OrderRequest:
        """Parse a triggerlang grammar format like .parse() but also
        print the created datastructures to stdout as both their
        dictionary form and their object/dataclass form."""

        parsed = self.parse(text)

        print("\tResult", "as dict:")
        pp.cpprint(dataclasses.asdict(parsed))
        print("\n\tResult", "as class:")
        pp.cpprint(parsed)

        return parsed
