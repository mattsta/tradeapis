tradeapis: stock market api helpers
===================================

A collection of modules for US stock market data processing including some API clients for hosted data and trade services.

Includes:

| What | Description |
|------|-------------|
[`fees.py`](tradeapis/fees.py) | Calculate fees for buying and selling option contracts (and works with spreads). Includes current rates for SEC/TAF/OCC/ORF fees.
[`data.py`](tradeapis/data.py) | perfectly named module for fetching the option class of a symbol (standard pricing ($0.05 under $3, else $0.10 increment), penny pilot program (all $0.01 under $3, $0.05 above), or all ($0.01 increment for any price). Also can round prices to the appropriate increment for a given symbol (since the option price class map was fetched). Also can fetch+cache Fear & Greed from CNN.
[`cal.py`](tradeapis/cal.py) | friendly interface to market calendars for getting *only* market days between given dates (i.e. ignore weekends and holidays). Primary requests are market days "between two dates" or "from X days ago".
[`buylang.py`](tradeapis/buylang.py) | Convert text description of trades into Python objects for easy trade intention determination.
[`tradier.py`](tradeapis/tradier.py) | aiohttp interface to tradier data and trading APIs. Caches relevant data with reasonable TTLs to speed up subsequent requests (expiration dates, strikes per date, saves quotes and chains after hours since they won't keep changing, etc)
[`polygon.py`](tradeapis/polygon.py) | aiohttp interface to useful polygon (now "massive") stock API endpoints


## Account Login Config

### Polygon / Massive

Polygon/Massive auth is by environment variable (or dotfile) because we haven't refactored it into a cleaner interface yet.

Configure your polygon API key as an environment variable or in `.env.tradeapis`:

```haskell
TRADEAPIS_MASSIVE_KEY=yourpolygonmassivekeyhere
```

### Tradier

Tradier auth is provided in the client creation itself. We use mode `prod` to mean the live trade interface with access to executing live trades and receiving live non-delayed market data, then mode `dev` or `sandbox` for the paper trading non-live sandbox keys (also note: `sandbox` keys view the entire market from a 15 minute delay (including quotes). it's basically impossible to run near-spread limit orders in the sandbox for algo testing, everything in `dev`/`sandbox` should probably be market orders to trigger the fake trades.)

The tradier API needs both your account ID and your static authentication token. There's nothing else protecting your account once you specify your account ID and access key, so make sure never to commit your production live trading access key anywhere public (as opposed to IBKR which triggers a mobile in-app 2FA when you connect to their API gateway).

```python
import asyncio
from tradeapis.tradier import TradierClient

async processTradier():
    tc = TradierClient("prod", "ACCOUNT-ID-u9r32jifdslk", dict(prod="your-production-api-credential"))

    # must "setup" to attach the aiohttp session inside the event loop
    await tc.setup()

    profile_json = await tc.profile()
    print(profile_json)

asyncio.run(processTradier())
```

## fees.py

Even if you are using a "zero commission" option broker, you still have regulatory fees on your transactions, and `fees.py` can show you why your "zero commission" returned capital ended up being less than the executed prices.

Example of calculating fees for a 1,000 contract debit or credit spread:

```python
from tradeapis.fees import OptionFees

debit = OptionFees(legs_buy=[(1000, 30.33)], legs_sell=[(1000, 66.66)])
credit = OptionFees(legs_sell=[(1000, 30.33)], legs_buy=[(1000, 66.66)])
```

The `debit` values calculated are:

```haskell
BUY 1000; SELL 1000; TOTAL 2000
Selling: $6,666,000.00
Buying:  $3,033,000.00
Net:     -$3,633,000.00
SEC: $34.00
TAF: $2.00
ORF: $77.60
OCC: $90.00
Total estimate: $203.60
```

While the `credit` values calculated are:

```haskell
BUY 1000; SELL 1000; TOTAL 2000
Selling: $3,033,000.00
Buying:  $6,666,000.00
Net:     $3,633,000.00
SEC: $15.47
TAF: $2.00
ORF: $77.60
OCC: $90.00
Total estimate: $185.07
```

See [`fees.py`](tradeapis/fees.py) for methods to access each underlying field if you need numbers instead of the default `__repr__()` string report.

### Where do fees show up in your trading account?

Current standard broker practice for commission and fee reporting is to automatically add/subtract them from your per-share or per-contract cost basis when a transaction happens, so the fees are often transparent to your accounting.  You'll just see the amount you spend for a trade is higher than the executed price (due to adding fees to your cost basis) and the amount you receive from a sell is less than the executed price (due to fees being subtracted from your returned capital). Counterpoint: there's a special case when, mostly with IBKR, sometimes your commissions are *negative* and you actually get paid when a trade executes via market rebate mechanisms.

The "automatically integrate commissions and fees into your cost basis" is a relatively new approach instead of reporting them as independent line items, but it's a big help at tax reporting time because all deductible trade-related fees and commissions are already included in your broker's official accounting for you, so you don't need to sit in your tax app adding 77¢ deductions to all your trades.

## data.py

When dealing with automated option trading, you need to know what price increments are allowed for setting your bid/ask prices.

Options trade in three price classes:

- All pennies: contracts are priced in $0.01 increments for all contracts under a symbol
- Penny Pilot Program: a select group of symbols trade in $0.01 increments if a contract is priced under $3, else prices must meet $0.05 increments.
- Legacy / Traditional: $0.05 increments under $3, else $0.10 increments

Using `data.py`, you can request the live daily mapping of which symbols belong to which price classes then automatically round your target prices up/down to meet the next valid price increment.

```python
import asyncio
from tradeapis.data import MarketMetadata

async def doMM():
    mm = MarketMetadata()

    # setup the aiohttp client in the event loop
    await mm.setup()

    # fetch the symbol to option class CSV then cache results
    # for future symbol price request math.
    await mm.populateOptionTickLookupDict()

    # we use 'isBuy' to bias the rounding to be more likely to price
    # in the market direction. if 'isBuy' is True, we round UP to hopefully
    # hit an ask price. if 'isBuy' is False, we round DOWN to hopefully
    # hit a bid price.
    # You can provide 'symbol' as either a full OCC symbol or just the underlying
    # equity/etf symbol itself.
    valid_price = mm.adjustLimitTick(symbol="AAPL", price=4.22, isBuy=True)
    print(valid_price)

    # clean exit the aiohttp session or else it complains
    await mm.shutdown()

asyncio.run(doMM())
```

`data.py` also has a wrapper enabling cached retrieval of the CNN Fear/Greed index (automatically cached for 5 minutes):

```python
from tradeapis.data import MarketMetadata

async def getFG():
    mm = MarketMetadata()
    await mm.setup()
    fg = await mm.feargreed()
    await mm.shutdown()
    return fg

asyncio.run(getFG())
```

## cal.py

Dealing with dates of market time when requesting market data can be tricky because we have to account for weekends and market observed holidays.

Using [`pandas_market_calendars`](https://github.com/rsheftel/pandas_market_calendars) (which itself uses [`exchange_calendars`](https://github.com/gerrymanoim/exchange_calendars)) we have local access to query authoritative active market dates. This is much more stable and reliable than manually trying to exclude weekends (or even lazier, just requesting all dates from your data provider then letting them return errors or empty results for non-trading days).

Underlying calendar retrieval is cached for 6.5 hours before reloading (the calendars are somewhat slow to load on first access):

```python
from tradeapis.cal import marketDaysAgo, marketDaysBetweenDates

# Request start/end market date range for 40 days ago from right now.
start, end = marketDaysAgo(40)
# (Timestamp('2021-05-07 00:00:00-0400', tz='US/Eastern'),
#  Timestamp('2021-07-02 00:00:00-0400', tz='US/Eastern'))

days = marketDaysBetweenDates('2021-01-01', '2021-02-03')
# [datetime.date(2021, 1, 4),
#  datetime.date(2021, 1, 5),
#  datetime.date(2021, 1, 6),
#  datetime.date(2021, 1, 7),
#  datetime.date(2021, 1, 8),
#  datetime.date(2021, 1, 11),
#  datetime.date(2021, 1, 12),
#  datetime.date(2021, 1, 13),
#  datetime.date(2021, 1, 14),
#  datetime.date(2021, 1, 15),
#  datetime.date(2021, 1, 19),
#  datetime.date(2021, 1, 20),
#  datetime.date(2021, 1, 21),
#  datetime.date(2021, 1, 22),
#  datetime.date(2021, 1, 25),
#  datetime.date(2021, 1, 26),
#  datetime.date(2021, 1, 27),
#  datetime.date(2021, 1, 28),
#  datetime.date(2021, 1, 29),
#  datetime.date(2021, 2, 1),
#  datetime.date(2021, 2, 2),
#  datetime.date(2021, 2, 3)]
```

## buylang.py

buylang is a simple language to process short text descriptions of trades (including spreads) into Python objects we can more easily deconstruct.

```python
from tradeapis.buylang import OLang

ol = OLang()

# if no quantity is requested, qty '1' is assumed.
result = ol.parse("aapl")
# OrderRequest(orders=[Order(Side.UNSET, 1, "AAPL")])

result = ol.parse("bto 3 aapl")
# OrderRequest(orders=[Order(Side.BTO, 1, "AAPL")])

result = ol.parse("bto 3000 AAPL222222C00123000")
# OrderRequest(orders=[Order(Side.BTO, 3000, "AAPL222222C00123000")])

# again, qty '1' assumed for unspecified amounts
result = ol.parse("bto AAPL222222C00123000 sto AAPL222222C00200000")
# OrderRequest(
#        orders=[
#            Order(Side.BTO, 1, "AAPL222222C00123000"),
#            Order(Side.STO, 1, "AAPL222222C00200000"),
#        ]
#    )

# Depending on your trade API, it either wants all contract counts per leg *or*
# it may just want a _ratio_ of contracts per trade specification, then you manually set
# a multiplier on top of the spread ratio (e.g. butterfly 1:2:1 but you want 100 of them, so
# it would buy 100, sell 200, buy 100).
result = ol.parse("bto 1 AAPL222222C00123000 sto 2 AAPL222222C00200000 bto 1 AAPL222222C00250000")
# OrderRequest(
#        orders=[
#            Order(Side.BTO, 1, "AAPL222222C00123000"),
#            Order(Side.STO, 2, "AAPL222222C00200000"),
#            Order(Side.BTO, 1, "AAPL222222C00250000"),
#        ]
#    )

```

## tradier.py

`tradier.py` is an aiohttp interface supporting all [live account details](https://documentation.tradier.com/brokerage-api/accounts/get-account-balance), [live trade actions](https://documentation.tradier.com/brokerage-api/trading/getting-started), and [live](https://documentation.tradier.com/brokerage-api/markets/get-quotes)/[historical](https://documentation.tradier.com/brokerage-api/markets/get-history) data requests, including support for receiving the live quote and trade websocket stream while automatically managing the [weird expiring session ticket auth method](https://documentation.tradier.com/brokerage-api/streaming/create-market-session) they require for restricting streaming quote access.

Note: the tradier API itself has no notification mechanism for account changes (when trades execute, when balances update), so for live data you must manually query the API on a timer (i.e. you have to update your balance on a timer every 3 seconds instead of having updates pushed to you when it actually changes, but their [rate limits](https://documentation.tradier.com/brokerage-api/overview/rate-limiting) are generous enough to allow (encourage?) such rapid polling of basic data over and over and over again since they don't have a push or pub/sub account change notification mechanism).

Also includes an automation (`getQuoteAndAllOptionChains(underlying_symbol)`) for requesting all option details for a symbol across all [expirations](https://documentation.tradier.com/brokerage-api/markets/get-options-expirations) and [all strikes](https://documentation.tradier.com/brokerage-api/markets/get-options-strikes) for both [calls and puts](https://documentation.tradier.com/brokerage-api/markets/get-options-chains) just with a single call (with all requests running *async concurrently* for lowest latency of the complete aggregation, then returned as a nicely post-processed column-formatted dataframe instead of 30,000 JSON objects).

Also includes a helper (`expirationAway(underlying_symbol, distance)`) for getting the Nth furthest future expiration away (`distance=0` is the next immediate expiration which could be the current day). Useful for automatically formatting option symbols based on how aggressive of a timeframe you want to pursue (i.e. do you want the NEXT expiration or the 3rd NEXT expiration or the 20th next expiration, etc). Expiration distance varies wildly based on symbol type since SPY/QQQ will have 3 (sometimes 4!) expirations per week, weekly symbols will have one expiration per week, and the rest of the options universe will have 1 expiration per month.

Also includes a math utility (`strikeByWidthFromPrice(underlying_symbol, expiration, underlying_price, percentAway=0.07)`) for automatically calculating a valid option strike price based on the current underlying price and a given width (or dollar amount if using `dollarAway=`). Automatically requests the live market mapping between symbol, expiration date, and all strikes if not cached already. Useful for selecting a valid strike price without having to look through the chain manually. Returns the distance both above and below the given price as `(percent subtracted strike, ATM strike, percent added strike)`.

More practical usage examples and demonstrations of post-processing the returned data in cleaner dataframes will be provided in the `tcli` command line trading platform being released soon hopefully eventually.


## polygon.py

a *very* simple wrapper around polygon/massive API endpoints I've used for data aggregation. Not a complete client for all their endpoints, but it matches the aiohttp patterns we use everywhere else throughout the trade platforms.

Requires manually specifying your key via environment variable `TRADEAPIS_MASSIVE_KEY` (or place in `.env.tradeapis`).

Currently includes useful wrappers to retrieve:

- [`historicalTicks`](https://massive.com/docs/get_v2_ticks_stocks_trades__ticker___date__anchor)
    - retrieves all trades for a given symbol on a given date
        - (limited to 50k trades per query, so for all trades on a given symbol+date, you have to create your own iterator system to increment the recently received highest timestamp and provide it as the next smallest timestamp to request
            - (which *also* means you can't request a full symbol of daily trades concurrently because you need to always request the next 50k offset in a serial fashion, which we can't pre-calculate and spray into an async-gather up front)
                - —but the fancy auto-advance-then-download-then-aggregate feature is available in `historicalticks.py` in the `stats` package of `tplat/mattplat` which may or may not be released as you read this)
- [`historicalBars`](https://massive.com/docs/get_v2_aggs_ticker__stocksTicker__range__multiplier___timespan___from___to__anchor)
    - retrieves aggregate bars on any minute, hour, day, week, month, quarter, or year rollup you request
- [`groupedBars`](https://massive.com/docs/get_v2_aggs_grouped_locale_us_market_stocks__date__anchor)
    - retrieves entire stock market OHLC+vwap values for a given day
        - also note: polygon/massive performs back-dated trade corrections on these OHLC+vwap values even after the market closes, so if you request a full market snapshot at 2030 ET Monday then request the same snapshot again a day later, the values will have changed. For fast processing, use the latest data, but for accurate processing, grab the values again a day or two later to make sure they are fixed in place.
- [`splits`](https://massive.com/docs/get_v2_reference_splits__stocksTicker__anchor)
    - retrieve historical and recently announced stock splits for a single symbol
- [`snapshot`](https://massive.com/docs/get_v2_snapshot_locale_us_markets_stocks_tickers_anchor)
    - retrieve a live snapshot of the entire market including last trade price, current bid/ask spread, current minute bar, previous day OHLC, and current day OHLC up to request time for each symbol in the market

Also includes support for managing the [massive trade websocket subscribe format](https://massive.com/docs/websocket/quickstart#subscribing-to-data-feeds) (see: `polygonSubscribe`, `polygonUnsubscribe`, `polygonConnect`, `polygonReconnect`), but for production use (if distributing live trade processing among multiple processes), the preferred method of deconstructing the websocket feed is [trade balancer](https://github.com/mattsta/trade-balancer) which can split the entire websocket feed of trades into backend worker processes at a rate of ~500ns per trade (which is enough to handle the multi-million trade-per-second bursts at start and end of day without any significant backlog).
