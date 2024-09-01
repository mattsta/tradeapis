"""orderlang allows full purchase intent including pricing.
(versus buylang which is mainly for symbol and spread declarations)"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal

from lark import Lark, Token, Transformer, v_args

from .buylang import looksLikeOrderCommand, OLang as BuyLang, Order


# global for dividing things by 100 in decimal format
D100: Final = Decimal("100")


# fmt: off
class Calculation(str): ...
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


def becomeDecimal(x: int | float | None) -> Decimal | None:
    if isinstance(x, (int, float)):
        return Decimal(str(x))

    return x


@dataclass(slots=True)
class OrderIntent:
    """Description of a purchase request."""

    symbol: str | None = None
    algo: str | None = None

    # You can optionally provide an exchange for this order if you have specific requirements.
    exchange: str | None = None

    # we also parse 'symbol' directly to a buylang Order() instance if it's a spread request
    spread: Order | None = None

    # Quantity is always represented as a POSITIVE VALUE which is then
    # designated long or short by its type. The type can easily be
    # meta-checked via the property handlers.
    # Reason: trading APIs usually say "BUY 100" or "SELL 100" and not "BUY -100" or "SELL -100",
    #         so we maintain quantity as POSITIVE while the BUY/SELL intent is a property of the
    #         order itself we can introspect when needed.
    # Also, we retain using 'None' for quantity if user has requested "ALL" quantity (so the consuming
    # program can look up how much current quantity a position has for a symbol for exiting against, probably).
    qty: (
        DecimalLongShares
        | DecimalShortShares
        | DecimalLongCash
        | DecimalShortCash
        | None
    ) = None

    bracketProfitAlgo: str = "LMT"
    bracketLossAlgo: str = "STP"

    limit: DecimalPrice | Calculation | None = None

    bracketProfit: DecimalPrice | DecimalPercent | None = None
    bracketLoss: DecimalPrice | DecimalPercent | None = None

    preview: bool = False

    # custom key-value config for setting options if the consuming application wants to read them
    config: dict[str, str | bool] = field(default_factory=dict)

    # if a scale/ladder configuration is requested,
    # save the parameters here so the scale can be generated on-demand
    scaleDesc: dict[str, Decimal | float | int] | None = None

    @property
    def scale(self) -> list[OrderIntent]:
        """Return a scale in/out list of orders if scale/ladder params are embedded in this OrderIntent"""
        if self.scaleDesc:
            return self.ladder(**self.scaleDesc)

        return []

    @property
    def scaleAvgRecord(self) -> OrderIntent:
        """Generate a new OrderIntent representing the average cost of all scale orders as the limit price."""
        if scale := self.scale:
            pq = 0
            tq = 0
            for order in scale:
                pq += order.limit * order.qty
                tq += order.qty

            # average cost is sum(p * q) / sum(q)
            return replace(self, limit=pq / tq, qty=tq, scaleDesc=None)

        # else, if no scale/ladder defined, return a COPY of ourself so user doesn't edit the original object by mistake
        return replace(self)

    @property
    def isLong(self) -> bool:
        return isinstance(self.qty, DecimalLong)

    @property
    def isShort(self) -> bool:
        return not self.isLong

    @property
    def isLive(self) -> bool:
        """A "live" price request is a full OrderIntent just with no limit price.

        The user can interpert live prices to mean either MKT orders or determine live
        prices not requiring user input.

        Allowing "live" syntax also lets us still specify bracket endpoints against an unknown
        price at time of order entry."""
        return self.limit is not None

    @property
    def isCredit(self) -> bool:
        """Detect if this OrderIntent is an explicit credit event (negative price).

        Credit events are different from short invents because credits can also have
        negative quantities (shorts) and we need to adjust our bracket profit/loss math
        against having both negative prices and negative quantities so the directions
        for profit remains correct (profit for short == lower price and profit for credit
        also == lower price, so these must not conflict)."""
        return self.limit < 0

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

        # It is up to the user to verify if the short qty vs. price makes sense for their platform.
        # For example, on IBKR, you can have:
        #  - short quantity and negative price on a short spread (receive credit)
        #  - short quantity and positive price on a short spread (buy for debit)
        #  - long quantity and positive price on a short spread (receive credit)
        #  - short quantity and positive price on a short spread (buy for debit)
        #  - long quantity and positive price on a long spread (buy for debit)
        #  - short quantity and positive price on a long spread (receive credit)
        # Basically, IBKR will reject "long quantity with short price" even though we allow this configuration
        # in the grammar here, so the order processing _can_ reject some valid things parsed here if you get
        # your combinations wrong or backwards (but order previews/rejections fix those scenarios).
        return 1 if self.isLong else -1

    @property
    def bracketProfitReal(self) -> Decimal:
        """Generate the _actual_ bracket profit value given the current limit price.

        Takes care of both direction (long vs. short) and value (percent vs. exact).

        NOTE: this interperts the bracket request as POINTS AWAY FROM THE LIMIT PRICE and NOT THE FULL BRACKET EXIT PRICE.

        This means if your qty is long and limit is $10 and your bracketProfit is $3, this generates a bracketProfitReal == $13.

        You can of course avoid using bracketProfitReal and treat bracketProfit as any value you like as well.
        """

        # if percent, extract percent of whole to use for addition
        bracketProfit = self.bracketProfit
        if self.isBracketProfitPercent:
            bracketProfit = self.limit * self.bracketProfit / 100

        # credits have a NEGATIVE price which we need to invert for the loss math (regardless of short/long qty)
        # (e.g. profit for CREDIT=-3 PROFIT=2 LOSS=4 means final values are profit=(-(-3) - 2)=1 loss=(-(-3) + 4)=7)
        if self.isCredit:
            return -self.limit - bracketProfit

        # else, the bracket price requested is the EXACT price to limit
        return self.limit + (bracketProfit * self.shortFix)

    @property
    def bracketLossReal(self) -> Decimal:
        """Generate the _actual_ bracket loss value given the current limit price.

        Takes care of both direction (long vs. short) and value (percent vs. exact).

        See notes in bracketProfitReal() about how the bracket prices are treated for exit here.
        """

        # if percent, do basically:  PRICE * 1.profitPercent
        # if percent, extract percent of whole to use for subtraction
        bracketLoss = self.bracketLoss
        if self.isBracketLossPercent:
            bracketLoss = self.limit * self.bracketLoss / 100

        # credits have a NEGATIVE price which we need to invert for the loss math (regardless of short/long qty)
        if self.isCredit:
            return -self.limit + bracketLoss

        # else, the bracket price requested is the EXACT price to limit
        return self.limit - (bracketLoss * self.shortFix)

    def ladder(
        self,
        steps: int | Decimal = 0,
        points: int | float | Decimal | None = None,
        percent: float | Decimal | None = None,
    ) -> list[OrderIntent]:
        """Generate multiple OrderIntent requests where each grows by 'points' or 'percent' higher or lower than the starting order at each step.

        NOTE: down-scale-price ladders are defined by negative points/percent and NOT by negative steps. Steps are always positive.
        """

        # if nothing requested, nothing returned (and no other parameters processed)
        if not steps or steps < 0:
            return []

        assert bool(points) ^ bool(
            percent
        ), f"You must specfiy *one* of *either* 'points' or 'percent' but not both!"

        assert not isinstance(
            self.limit, Calculation
        ), "To generate a ladder you must have a concrete limit price instead of a Calculation!"

        # convert parameters to decimal objects so all the math works without errors
        # (these are noop if the values are already decimals)
        points = becomeDecimal(points)
        percent = becomeDecimal(percent)
        startPrice = becomeDecimal(self.limit)

        def nextVal(amt, step):
            if points:
                return startPrice + (points * step)

            # else, percent grow each time...
            return startPrice * (1 + (percent * step))

        currentOrder = self
        results = []

        for i in range(int(steps)):
            results.append(
                replace(
                    currentOrder,
                    limit=nextVal(currentOrder.limit, i),
                    # don't duplicate scaleDesc into these subsequent orders
                    scaleDesc=None,
                )
            )
            currentOrder = results[-1]

        return results


lang = r"""

    // BUY something
    // SYMBOL SHARES|PRICE_QUANTITY ALGO ["on" EXCHANGE] [@ [["credit"? LIMIT_PRICE] | "live"] [+ PROFIT_POINTS ALGO | - LOSS_POINTS ALGO | ± EQUAL_PROFIT_LOSS_POINTS ALGO_PROFIT ALGO_LOSS]] [preview] [config [key | key=value]+]?

    // We want to be flexible where the limit price, config options, and preview flag can be set, so allow anything in any order:
    cmd: symbol quantity orderalgo exchange? tail*
    tail: preview? (scale | limit | config)? preview?

    scale: "scale" scale_steps (scale_pts | scale_pct)
    scale_steps: "steps" price
    scale_pts: ("pt"i | "pts"i | "point"i | "points"i)? price
    scale_pct: price "%"

    // TODO: we could actually use buylang to parse symbol allowing full spread descriptions here too, but
    //       then we would need to include the buylang Order() object instaed of just a symbol string in our OrderIntent() output.
    symbol: /\/?[:A-Za-z0-9\/\._-]{1,22}/ | string

    quantity: shares_short | shares_long | cash_amount_long | cash_amount_short | qty_all

    shares_short: "-" price
    shares_long: price
    cash_amount_long: "$" price
    cash_amount_short: ("-$" | "$-") price
    qty_all: "all"i

    orderalgo: /[A-Za-z_]+/
    algo: /[A-Za-z_]+/

    limit: "@" (price | "live"i | credit) (profit | loss | bracket)*

    // credit spreads require a NEGATIVE price for us to receive money (or exiting a debit spread),
    // so we need an extra identifier to distinguish between stop loss '-' and credit limit price '-' requests.
    // Credit spreads can be either: BUY NEGATIVE PRICE, or SELL POSITIVE PRICE. For using credit spreads
    // by generating a regular long spread then just selling it, we use positive prices so this 'credit' feature
    // isn't actually _required_ for credit transactions.
    credit: "credit"i "-" price

    exchange: "ON"i /[A-Za-z]+/

    // PRICE is a helper for "float parsing with optional , or _ allowed" also with unlimited decimal precision allowed
    // Note on price format: for shares and cash, we EXTERNALIZE the negative sign and report a "short decimal" object, but
    // for regular numeric parsing (used by the key-value config system) this price rule *does* consume an optional leading
    // negative sign to generate a negative Decimal() object.
    price: /-?[0-9][0-9_,]*\.?[0-9_,]*/ | calculation

    // an in-line calculation container is anything between parens as long as it starts with an allowed operator
    // (We are allowing anything so (+ 1 2) or (+ live jfkladsjlku382) all work; it's the job of the consumer to
    //  determine _what_ to do with the calculation syntax here, we just enforce prefix-with-operator notation)
    calculation: "(" FUNC (/[^() ]+/ | calculation)+ ")"
    FUNC: "+" | "-" | "*" | "/"

    is_percentage: "%"

    // attach TAKE PROFIT order
    profit: "+" price is_percentage? algo?

    // attach STOP LOSS order as direct value OR as a percent difference request
    loss: "-" price is_percentage? algo?

    // attach TAKE PROFIT _and_ STOP LOSS order at equal width with optional PROFIT LOSS algo overrides
    bracket: "±" price (algo algo)?

    preview: "preview"i | "prev"i | "pre"i | "pr"i | "p"i

    config: ("config"i | "conf"i | "c"i) config_item+

    // config items can be single items (value for single item becomes True by default) or key=value items.
    // If you pass in price-like values or calculation values, those are also transformed into native data types.
    // Also note: ORDER matters here, so we must consider 'price' BEFORE the generic all-allowed-characters so the 'price'
    //            can properly capture price (or calculation) shaped inputs before returning to the "any string" matcher.
    // Also note: we convert special values of "false f no off" to False and "true t yes on" to True.
    config_item: config_truth | kvpair

    kvpair: ALLOWED_KEY_CHARS "=" (string | price | ALLOWED_KEY_CHARS)

    // Single config items are always true (key only, no value provided).
    config_truth: ALLOWED_KEY_CHARS

    // Keys and values must start with a letter or allowed symbol as to NOT CONFLICT WITH NUMBER PARSING OR STRING PARSING
    ALLOWED_KEY_CHARS: /[^\s=0-9-"'][^\s=]*/

    WHITESPACE: (" " | "\t" | "\n")+
    COMMENT: /#[^\n]*/

    string: ESCAPED_STRING_DOUBLE | ESCAPED_STRING_SINGLE

    _STRING_INNER: /.*?/
    _STRING_ESC_INNER: _STRING_INNER /(?<!\\)(\\\\)*?/

    // "string of things"
    ESCAPED_STRING_DOUBLE: "\"" _STRING_ESC_INNER "\""

    // 'string of things'
    ESCAPED_STRING_SINGLE: "'" _STRING_ESC_INNER "'"

    %import common.NUMBER

    %ignore WHITESPACE
    %ignore COMMENT
"""


class TreeToBuy(Transformer):
    def __init__(self, buylang) -> None:
        # save symbol to order request parser as a global we can reuse
        self.buylang = buylang

    @v_args(inline=True)
    def cmd(self, *stuff):
        # This is a little backwards since we're not collecting the results from the rules
        # then creating the output in the start rule, but it works here and is simpler
        # since we have some repeating rules and sections which we allow to overwrite
        # things in different places.
        return self.b

    @v_args(inline=True)
    def scale(self, steps, amt):
        # steps and amt are dicts we can use as arguments to the ladder() function of ourself.
        self.b.scaleDesc = steps | amt

    @v_args(inline=True)
    def scale_steps(self, steps):
        return dict(steps=steps)

    @v_args(inline=True)
    def scale_pts(self, pts):
        return dict(points=pts)

    @v_args(inline=True)
    def scale_pct(self, pct):
        # user percentages are 100-based, but our API percentages are 1-based
        return dict(percent=pct / D100)

    @v_args(inline=True)
    def symbol(self, got):
        # Symbol is parsed _first_ so we use this rule to always create a NEW order indent for each run
        # so if this parser is re-used we generate a new order on each parse.
        # Also means this parser is not re-entrant, but nobody should be doing such things anyway here.
        self.b = OrderIntent()
        self.b.symbol = got.replace("_", " ").replace("  ", " ").upper().strip()

        # re-add quotes to symbol for buylang parsing
        # only quote if it looks like a single symbol (has space and is short).
        # if is longer with space(s), don't quote because we need to parse it as a full side-size-symbol command.
        if self.b.symbol.count(" ") >= 2 and looksLikeOrderCommand(self.b.symbol):
            # if we have two spaces and it's long enough to be a buy command like "buy 100 AAPL buy 1 AAPL240920P00200000"
            # then we pass the symbol through as-provided.
            self.b.spread = self.buylang.parse(self.b.symbol)

    @v_args(inline=True)
    def string(self, got):
        # lark string rule INCLUDES the surrounding quotes, but we want to remove them, so we remove them
        return got[1:-1]

    @v_args(inline=True)
    def quantity(self, got):
        self.b.qty = got

    @v_args(inline=True)
    def shares_long(self, got):
        return DecimalLongShares(got)

    @v_args(inline=True)
    def shares_short(self, got):
        return DecimalShortShares(got)

    @v_args(inline=True)
    def cash_amount_long(self, got):
        return DecimalLongCash(got)

    @v_args(inline=True)
    def cash_amount_short(self, got):
        return DecimalShortCash(got)

    @v_args(inline=True)
    def qty_all(self):
        """If user says qty is all, then we use None to mean "no qty requested, look it up yourself."""
        return None

    @v_args(inline=True)
    def limit(self, *gotextra):
        # the limit price is now extra, but this rule still triggers.
        # Example: AAPL 100 AF @ - 3
        # (has no limit order, but still populates the limit with 'gotextra' of [None])
        # Note: we use 'is not None' here because 0 *is* a valid price we can attempt.
        if gotextra and gotextra[0] is not None:
            got = gotextra[0]
            self.b.limit = got

    @v_args(inline=True)
    def calculation(self, *got):
        # an individual calculation is just passed-through with parens re-added
        func, *args = got
        rebuilt = f"({func} {' '.join(args)})"
        # print("Calculation REBUILT:", rebuilt, "via", (func, args))
        return Calculation(rebuilt)

    @v_args(inline=True)
    def credit(self, got):
        # credit requests are just a negative limit price
        # ("got" is already a DecimalPrice()  here from the 'price' rule)
        self.b.limit = -got

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
    def config(self, *got):
        """Combine already-parsed config_items (now single dicts) into the final config object"""

        # Note: we allow MULTIPLE config blocks, so create dict on first config encounter,
        #       then for future config blocks, we just merge (or overwrite) values into the single
        #       config object. Why not?
        if not self.b.config:
            self.b.config = {}

        # every input here is a single-element dict we can combine into our final result
        for g in got:
            self.b.config |= g

    @v_args(inline=True)
    def config_item(self, got):
        """Parse a single config item as part of a trailing config key-value (or bare key) settings group"""
        return got

    @v_args(inline=True)
    def kvpair(self, k, v):
        # print("got kv of:", k, " = ", v)
        # convert token data to string value so the user doesn't get Lark token repr of Token(RULE, 'value')
        if isinstance(v, Token):
            v = str(v).strip()

        if isinstance(v, str):
            lv = v.lower()
            # if user value is a true or falsy string, convert it.
            if lv in {"true", "t", "yes", "on"}:
                v = True
            elif lv in {"false", "f", "no", "off"}:
                v = False
            elif lv == "none":
                v = None

        return {k.lower(): v}

    @v_args(inline=True)
    def config_truth(self, got):
        # A single value is just a truth value
        return {got.lower(): True}

    @v_args(inline=True)
    def price(self, got):
        # price can be either an input numeric string OR it can be an already-parsed "calculation"
        # Note: we don't execute or reduce "calculations" here because they can contain external variables
        #       only the consuming application can populate then resolve.
        if isinstance(got, Calculation):
            return got

        if isinstance(got, str):
            return DecimalPrice(got.replace("_", "").replace(",", ""))

        assert None, f"How did you get here with an unexpected data type? {got=}"
        return got

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
        self.parser = Lark(
            lang,
            start="cmd",
            # Use a more forgiving parser so it absorbs some of our logic burden:
            parser="earley",
            # Do what we mean, not what we say:
            ambiguity="resolve",
            # again, do what we mean, not what we say:
            lexer="dynamic",
            # big error if errors:
            strict=True,
            # ordered_sets improves reliability of test suites, otherwise
            # with un-ordered sets sometimes failures only happen randomly
            # instead of on every test run:
            ordered_sets=True,
            # debug=True
        )

        self.transformer = TreeToBuy(BuyLang())

    def parse(self, text: str) -> OrderIntent:
        """Parse 'text' conforming to our triggerlang grammar into a
        list of symbol-trigger or watchlist-symbol-trigger results.

        On error, throws the raw lark.UnexpectedToken exception with
        fields '.token' and '.expected' describing where the problem
        occurred and what was expected instead."""

        # input text must be .strip()'d because the parser doesn't
        # like end of input newlines."""
        parsed = self.parser.parse(text.strip())
        transformed = self.transformer.transform(parsed)
        return transformed
