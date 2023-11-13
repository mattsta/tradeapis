import asyncio
import os
from collections.abc import Iterable

from dataclasses import dataclass, field

import aiohttp
import orjson

import websockets
from dotenv import dotenv_values

from loguru import logger

ENDPOINT_STOCKS = "wss://socket.polygon.io/stocks"
ENDPOINT_STOCKS_DELAYED = ENDPOINT_STOCKS.replace("socket", "delayed")
ENDPOINT_OPTIONS = "wss://socket.polygon.io/options"
ENDPOINT_OPTIONS_DELAYED = ENDPOINT_OPTIONS.replace("socket", "delayed")

# TODO: refactor as better class encapsulation instead of being a module/env global.
CONFIG = {**dotenv_values(".env.tradeapis"), **os.environ}

try:
    KEY = CONFIG["TRADEAPIS_POLYGON_KEY"]
except:
    logger.error("Sorry, must specify TRADEAPIS_POLYGON_KEY in env or .env.tradeapis")

# Docs at:
# https://polygon.io/sockets
auth = {"action": "auth", "params": KEY}


# https://polygon.io/docs/stocks/get_v3_trades__stockticker
def historicalTrades(session, symbol: str, date: str):
    """Endpoint returns all trades for a date with a maximum of 50k results per query.

    Future results are fetched via the 'next_url' parameter to page through the result
    set using the original query parameters, but with future date offsets.

    End of results is signaled by no 'next_url' key present in the results."""
    url = f"https://api.polygon.io/v3/trades/{symbol}"
    args = {
        "timestamp": date,
        "limit": 50000,
        "sort": "timestamp",
        "order": "asc",
        "apiKey": auth["params"],
    }

    return session.get(url, params=args)


# https://polygon.io/docs/stocks/get_v3_quotes__stockticker
def historicalQuotes(session, symbol: str, date: str):
    """Fetch all NBBO quote updates for a symbol for any date in the past.

    Fetch all quotes for day via the readAll() generator."""

    url = f"https://api.polygon.io/v3/quotes/{symbol}"
    args = {
        "timestamp": date,
        "limit": 50000,
        "sort": "timestamp",
        "order": "asc",
        "apiKey": auth["params"],
    }

    return session.get(url, params=args)


async def readAll(what, session, *args, **kwargs):
    """Use Polygon V3 API pattern to read an entire dataset via generator.

    Usage:
        async for idx, page in readAll(historicalTrades, session, symbol, date):
            results = page.get("results")

    Also, the return is (idx, parsed) because we can't use enumerate() with
    async genenerators, but we want to count the results anyway.
    """

    # This layout is somewhat unappealing, but it works and since it's
    # all located in this generator, nobody else has to deal with it.

    idx = 0
    try:
        # First request intial result with a 'next' URL to fetch
        orig = what(session, *args, **kwargs)
        result = await (await orig).read()
        parsed = orjson.loads(result)
        yield idx, parsed

        # Note: this "next_url" check for continuation is always _after_ the
        # previous result is returned, so we are always returning a complete
        # result before checking if there's a next thing to retrieve (i.e. we
        # aren't skipping over the final result from being returned)
        while "next_url" in parsed:
            # Fetch next URLs until no more next URLs are returned
            orig = fetchNext(session, parsed)

            idx += 1
            result = await (await orig).read()
            parsed = orjson.loads(result)
            yield idx, parsed
    except asyncio.CancelledError:
        logger.error("Exit requested!")
        return


def fetchNext(session, result):
    """Given a Polygon V3 result, fetch the 'next_url' or return None."""

    # https://polygon.io/blog/api-pagination-patterns/
    url = result.get("next_url")
    assert url, "Only pass results having next_url into fetchNext!"

    # the continuation URL doesn't include the API key, so we need to add it back...
    return session.get(url + f"&apiKey={auth['params']}")


# https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksticker__range__multiplier___timespan___from___to
def historicalBars(
    session,
    symbol: str,
    combine: int,
    timespan: str,
    dateFrom: str, # | int,
    dateTo: str, #| int,
    adjusted: bool = True,
):
    """Endpoint returns bars aggregated.

    timespan is one of: minute, hour, day, week, month, quarter, year.
    combine is, their API says, "size of the timespan multiplier" whatever that means.
    """

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{combine}/{timespan}/{dateFrom}/{dateTo}"

    # yes, "false" here is correct because args aren't JSON, it's all just int/string conversions
    args = {
        "sort": "asc",
        "adjusted": str(adjusted).lower(),
        "limit": 50000,
        "apiKey": auth["params"],
    }

    return session.get(url, params=args)


# https://polygon.io/docs/options/get_v3_reference_options_contracts
def optionsContracts(
    session,
    underlying: str,
    contractType: str,
    expirationDate: dict[str, str],
    strikePrice: dict[str, str],
):
    """Endpoint returns options chains across date ranges and stripe price ranges.

    expirationDate and strikePrice are dicts with one or more keys: lt, gt, lte, gte.

    contractType is 'call' or 'put'
    """

    url = f"https://api.polygon.io/v3/reference/options/contracts"

    # yes, string "true" / "false" here is correct because args aren't JSON, it's all just int/string conversions
    args = {"sort": "expiration_date", "order": "asc", "apiKey": auth["params"]}
    args |= dict(
        underlying_ticker=underlying,
        contract_type=contractType,
        expired="true",
        limit=1000,
    )
    args |= {f"expiration_date.{k}": v for k, v in expirationDate.items()}
    args |= {f"strike_price.{k}": v for k, v in strikePrice.items()}

    return session.get(url, params=args)


# https://polygon.io/docs/get_v2_aggs_grouped_locale_us_market_stocks__date__anchor
def groupedBars(session, date: str):
    """Endpoint returns daily bars for all symbols for an entire day.

    Give a date, get result.
    """

    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date}"

    # yes, "false" here is correct because args aren't JSON, it's all just int/string conversions
    args = {"unadjusted": "false", "apiKey": auth["params"]}

    return session.get(url, params=args)


# https://polygon.io/docs/stocks/get_v3_reference_splits
def splits(session, symbol: str, reverse=False):
    """Endpoint returns all historical forward stock split dates for symbol.

    Note: this does NOT return reverse splits unless reverse_split=True, then it
          *only* returns reverse splits, so for _all_ splits we basically need
          to check all symbols twice every day. sigh."""

    # TODO: this API supports filtering tickers by range, so we could just
    #       request daily "tickers.gt=A" to get everything with one original
    #       fetch with full pagination until the end (instead of all directly)
    # See "Query Filter Extensions" at https://polygon.io/blog/api-pagination-patterns/
    url = f"https://api.polygon.io/v3/reference/splits"

    # we want a dual sort here so provide list of tuples so a dict
    # doesn't overwrite the same shared key...
    args = [
        ("ticker", symbol),
        ("reverse_split", str(reverse).lower()),
        ("limit", 1000),
        ("sort", "ticker"),
        ("sort", "execution_date"),
        ("order", "asc"),
        ("apiKey", auth["params"]),
    ]

    return session.get(url, params=args)


# https://polygon.io/docs/stocks/get_v3_reference_tickers__ticker
def symbolDetail(session, symbol: str):
    """Endpoint returns dict of symbol details described at doc link.

    Note: uses the new beta Ticker Details API.

    Most useful for retrieving the current share count so we can calculate
    a live market cap using (share count * last trade price)."""

    url = f"https://api.polygon.io/v3/reference/tickers/{symbol}"
    args = {"apiKey": auth["params"]}

    return session.get(url, params=args)


# https://polygon.io/docs/get_v2_reference_financials__stocksTicker__anchor
def symbolFinancial(session, symbol: str):
    """Endpoint returns dict of symbol details described at doc link.

    Note: these are only updated yearly? They all seem stale."""

    url = f"https://api.polygon.io/v2/reference/financials/{symbol}"
    args = {"limit": 1, "type": "Q", "apiKey": auth["params"]}

    return session.get(url, params=args)


def exchanges(session, asset: str = "stocks"):
    url = f"https://api.polygon.io/v3/reference/exchanges"

    # Other assets include: options, crypto, fx
    args = {"asset_class": asset, "apiKey": auth["params"]}

    return session.get(url, params=args)


def gl(session, what):
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/{what}"
    args = {"apiKey": auth["params"]}

    return session.get(url, params=args)


def gainers(session):
    return gl(session, "gainers")


def losers(session):
    return gl(session, "losers")


# https://polygon.io/docs/get_v2_snapshot_locale_us_markets_stocks_tickers_anchor
def snapshot(session):
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
    args = {"apiKey": auth["params"]}

    return session.get(url, params=args)


def snapshotOne(session, symbol):
    # symbol must be upper AND symbol must NOT have a slash because then it
    # looks like an invalid path request.
    symbol = symbol.upper().replace("/", ".")
    url = (
        f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
    )
    args = {"apiKey": auth["params"]}

    return session.get(url, params=args)


def askFor(method: str, channels: Iterable, symbols: Iterable) -> dict[str, str]:
    """Build the subscribe or unsubscribe PDU"""
    assert isinstance(channels, Iterable)
    assert isinstance(symbols, Iterable)
    # Channels are a list of one or more of:
    #   - Trades: T
    #   - Quotes: Q
    #   - Second bars: A
    #   - Minute bars: AM
    return {
        "action": method,
        "params": ",".join(
            [f"{channel}.{symbol}" for symbol in symbols for channel in channels]
        ),
    }


def polygonSubscribe(channels: list, symbols: list) -> dict[str, str]:
    return askFor("subscribe", channels, symbols)


def polygonUnsubscribe(channels: list, symbols: list) -> dict[str, str]:
    return askFor("unsubscribe", channels, symbols)


async def polygonConnect(cxn=None, endpoint=ENDPOINT_STOCKS):
    if cxn:
        try:
            await cxn.close()
        except:
            pass

    return await websockets.connect(
        endpoint,
        ping_interval=90,
        ping_timeout=90,
        close_timeout=1,
        max_queue=2**32,
        read_limit=2**20,
    )


async def polygonReconnect(cxn):
    try:
        await cxn.close()
    except:
        pass

    return await polygonConnection()


@dataclass
class PolygonClient:
    async def setup(self):
        self.session = aiohttp.ClientSession()

    async def shutdown(self):
        await self.session.close()

    def snapshotOne(self, symbol: str):
        return snapshotOne(self.session, symbol)

    def snapshot(self):
        return snapshot(self.session)

    def gainers(self):
        return gainers(self.session)

    def losers(self):
        return losers(self.session)
