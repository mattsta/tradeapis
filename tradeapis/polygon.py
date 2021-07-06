from collections.abc import Iterable

from dataclasses import dataclass, field
from dotenv import dotenv_values
import os

from loguru import logger

import websockets
import aiohttp

endpoint = "wss://socket.polygon.io/stocks"

# TODO: refactor as better class encapsulation instead of being a module/env global.
CONFIG = {**dotenv_values(".env.tradeapis"), **os.environ}

try:
    KEY = CONFIG["TRADEAPIS_POLYGON_KEY"]
except:
    logger.error("Sorry, must specify TRADEAPIS_POLYGON_KEY in env or .env.tradeapis")

# Docs at:
# https://polygon.io/sockets
auth = {"action": "auth", "params": KEY}

# https://polygon.io/docs/get_v2_ticks_stocks_trades__ticker___date__anchor (current, but ugly URL)
def historicalTicks(session, symbol: str, date: str, timestamp=None):
    """Endpoint returns all trades for a date with a maximum of 50k results per query.
    To get the next 50k results, use the timestamp of the last result as the
    timestamp parameter for the next query. Repeat until no more result returned.

    API returns a 'success' key on each result which will be false if we are out
    of results (or if we are querying a non-trading day)."""
    url = f"https://api.polygon.io/v2/ticks/stocks/trades/{symbol}/{date}"
    args = {"limit": 50000, "apiKey": auth["params"]}
    if timestamp:
        args["timestamp"] = timestamp

    return session.get(url, params=args)


# https://polygon.io/docs/get_v2_aggs_ticker__stocksTicker__range__multiplier___timespan___from___to__anchor
def historicalBars(
    session, symbol: str, combine: int, timespan: str, dateFrom: str, dateTo: str
):
    """Endpoint returns bars aggregated.

    timespan is one of: minute, hour, day, week, month, quarter, year.
    combine is, their API says, "size of the timespan multiplier" whatever that means.
    """

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{combine}/{timespan}/{dateFrom}/{dateTo}"

    # yes, "false" here is correct because args aren't JSON, it's all just int/string conversions
    args = {"sort": "asc", "unadjusted": "false", "apiKey": auth["params"]}

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


# https://polygon.io/docs/get_v2_reference_splits__stocksTicker__anchor
def splits(session, symbol: str):
    """Endpoint returns all historical stock split dates for symbol."""

    url = f"https://api.polygon.io/v2/reference/splits/{symbol}"
    args = {"apiKey": auth["params"]}

    return session.get(url, params=args)


# https://polygon.io/docs/get_vX_reference_tickers__ticker__anchor
def symbolDetail(session, symbol: str):
    """Endpoint returns dict of symbol details described at doc link.

    Note: uses the new beta Ticker Details API.

    Most useful for retrieving the current share count so we can calculate
    a live market cap using (share count * last trade price)."""

    url = f"https://api.polygon.io/vX/reference/tickers/{symbol}"
    args = {"apiKey": auth["params"]}

    return session.get(url, params=args)


# https://polygon.io/docs/get_v2_reference_financials__stocksTicker__anchor
def symbolFinancial(session, symbol: str):
    """Endpoint returns dict of symbol details described at doc link.

    Note: these are only updated yearly? They all seem stale."""

    url = f"https://api.polygon.io/v2/reference/financials/{symbol}"
    args = {"limit": 1, "type": "Q", "apiKey": auth["params"]}

    return session.get(url, params=args)


def exchanges(session):
    url = f"https://api.polygon.io/v1/meta/exchanges"
    args = {"apiKey": auth["params"]}

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


def askFor(method: str, channels: Iterable, symbols: Iterable):
    """Build the subscribe or unsubscribe PDU"""
    assert isinstance(channels, Iterable)
    assert isinstance(symbols, Iterable)
    # Channels are a list of one or more of:
    # Trades: T
    # Quotes: Q
    # Second bars: A
    # Minute bars: AM
    return {
        "action": method,
        "params": ",".join(
            [f"{channel}.{symbol}" for symbol in symbols for channel in channels]
        ),
    }


def polygonSubscribe(channels: list, symbols: list):
    return askFor("subscribe", channels, symbols)


def polygonUnsubscribe(channels: list, symbols: list):
    return askFor("unsubscribe", channels, symbols)


async def polygonConnect(cxn=None):
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
        max_queue=2 ** 32,
        read_limit=2 ** 20,
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

    def snapshotOne(self, symbol: str):
        return snapshotOne(self.session, symbol)

    def snapshot(self):
        return snapshot(self.session)

    def gainers(self):
        return gainers(self.session)

    def losers(self):
        return losers(self.session)
