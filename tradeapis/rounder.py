from dataclasses import dataclass, field

from mutil.numeric import ROUND, roundnear


@dataclass(slots=True)
class Tick:
    # actual tick value for price rounding
    tick: float

    # if symbol has multiple ticks for price ranges (like $0.01 under $3, $0.05 over, etc),
    # mark the LOWEST ACCEPTABLE price here for the tick to be valid, then we search/bisect
    # the ticks to find the appropriate match for a given future price.
    least: float = 0.0


@dataclass(slots=True)
class Product:
    symbol: str
    name: str = ""
    ticks: list[Tick] = field(default_factory=list)

    # stirng of futures month specificationsin F-Z format.
    months: str = ""

    valuePerTick: float = 1.0
    multiplier: float = 1

    # default 2 decimal places, but we can override for more detailed symbol requirements...
    decimals: int = 2

    def __post_init__(self) -> None:
        # enforce ticks from smallest to largest so our searching works properly
        self.ticks = sorted(self.ticks, key=lambda x: x.least, reverse=True)

    def price(self, near: float, up: bool = True) -> float:
        """Return a valid product price rounded near the requested input price."""

        assert near > 0

        # if no tick requirements, the input price is acceptable
        if not self.ticks:
            return near

        for tick in self.ticks:
            # if requested price is ABOVE the search mark (Search from HIGHEST to LOWEST), then
            # we found the correct trigger. else, try next lowest trigger.
            # The final trigger is always "X > 0" which is a baseline else-catch-all condition.
            if near > tick.least:
                return roundnear(
                    tick.tick, near, ROUND.UP if up else ROUND.DOWN, self.decimals
                )

        assert None, f"Failed to find matching tick? {self} :: {near}"


# Many products share "$0.05 below $3; $0.10 otherwise" so just reference it once here for multiple-use
Tick5_10 = [Tick(0.05, 0), Tick(0.10, 3)]

# Full list of products with special price requirements
# Futures made via:
# In [392]: df = pd.read_html(
#      ...:     "https://tickertape.tdameritrade.com/trading/index-futures-tick-sizes-17630"
#      ...: )
#
# In [393]: for row, name, sym, size, flux, val in df[0].itertuples():
#      ...:     print(
#      ...:         f"Product(symbol='{sym}', name='{name}', ticks=[Tick({float(flux.split()[0])})], valuePerTick={val[1:]}, multiplier={size.split()[0][1:]}),"
#      ...:     )


# I couldn't find an API for all futures and index option tick increment details so we could generate/update all of these automatically,
# so we're going to manage this collection manually from various public records documents. shrug.
# TODO: we should also consume the options Penny Pilot Program here to auto-detect $0.01-$0.05 vs $0.05-$0.10 increments per underlying.
syms = [
    ## Index Futs
    Product(
        symbol="/ES",
        name="E-mini S&P 500",
        ticks=[Tick(0.25)],
        valuePerTick=12.50,
        multiplier=50,
    ),
    Product(
        symbol="/NQ",
        name="E-mini Nasdaq-100",
        ticks=[Tick(0.25)],
        valuePerTick=5,
        multiplier=20,
    ),
    Product(
        symbol="/RTY",
        name="E-mini Russell 2000",
        ticks=[Tick(0.1)],
        valuePerTick=5,
        multiplier=50,
    ),
    Product(
        symbol="/YM",
        name="E-mini Dow ($5)",
        ticks=[Tick(1.0)],
        valuePerTick=5,
        multiplier=5,
    ),
    Product(
        symbol="/MES",
        name="Micro E-mini S&P 500",
        ticks=[Tick(0.25)],
        valuePerTick=1.25,
        multiplier=5,
    ),
    Product(
        symbol="/MNQ",
        name="Micro E-mini Nasdaq-100",
        ticks=[Tick(0.25)],
        valuePerTick=0.50,
        multiplier=2,
    ),
    Product(
        symbol="/M2K",
        name="Micro E-mini Russell 2000",
        ticks=[Tick(0.1)],
        valuePerTick=0.50,
        multiplier=5,
    ),
    Product(
        symbol="/MYM",
        name="Micro E-mini Dow",
        ticks=[Tick(1.0)],
        valuePerTick=0.50,
        multiplier=0.50,
    ),
    Product(
        symbol="/HE",
        name="LEAN HOGS",
        ticks=[Tick(0.025)],
        decimals=3,
        valuePerTick=10,
        multiplier=40_000,
    ),
    ## Index Options
    # https://www.cboe.com/tradable_products/sp_500/spx_options/specifications/
    Product(
        symbol="SPX",
        name="SPX Standard",
        ticks=Tick5_10,
        multiplier=100,
    ),
    Product(
        symbol="SPXW",
        name="SPX Weekly",
        ticks=Tick5_10,
        multiplier=100,
    ),
    # https://www.nasdaq.com/docs/2022/08/24/1926-Q22_NDX%20Fact%20Sheet_NAM_v3.pdf
    Product(
        symbol="NDX",
        name="NDX Standard",
        ticks=Tick5_10,
        multiplier=100,
    ),
    Product(
        symbol="NDXP",
        name="NDX Weekly",
        ticks=Tick5_10,
        multiplier=100,
    ),
    # https://www.nasdaq.com/VOLQindexoptions
    Product(
        symbol="VOLQ",
        name="NDX Volatility",
        ticks=Tick5_10,
        multiplier=100,
    ),
    # BTC futures are in 0.25 increments...
    # (but you also trade fractional quantities, so you have much
    #  more fine grained pricing power than just $0.25 ticks by slicing your
    #  8-decimal-precision quantity).
    Product(
        symbol="BTC",
        name="BTC",
        ticks=[Tick(0.25)],
        multiplier=100,
    ),
]


# Map of symbol to product description
SYMS = {prod.symbol: prod for prod in syms}


def round(symbol: str, price: float, up: bool = True) -> float:
    """Round price to comply with product requirements.

    Notes:
      - futures require a '/' prefix
      - option symbols must be the root or underlying symbol and NOT the full OCC symbol
      - you can bias the rounding for up or down with bool 'up' parameter
    """

    if symbol in SYMS:
        return SYMS[symbol].price(price, up)

    # else, if no special carve-out in our product listings, assume
    # regular $0.01 pricing interval so all prices are valid
    return price
