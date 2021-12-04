import aiohttp
import asyncio
import orjson
import pandas as pd  # type: ignore
import websockets

import re
import os

from loguru import logger
from typing import Dict, List, Any, Union, Optional, Literal

from dataclasses import dataclass, field
import dataclasses

from bisect import bisect_left

import io
import arrow  # type: ignore
import time
import datetime
from collections import defaultdict

from mutil.dcache import FetchCache  # type: ignore

# from mutil.dates import thirdFridayForMonth, thirdFridays  # type: ignore
from mutil.numeric import roundnear5, roundnear10  # type: ignore

from tradeapis.data import MarketMetadata


def closestSortedIndex(elements, search):
    """Find the closest element inside sorted container 'elements'.

    Used for finding the ATM strike and wings against it using bisect_left
    then picking which result has the smallest difference between the search
    input and array values.

    Basically returns the index having the smallest difference between
    'search' in all of 'elements'. More efficient than a full scan
    which would be something like:
        (min([(abs(search - x), idx) for idx, x in enumerate(elements)]))[1]
    """
    pos = bisect_left(elements, search)

    # if input is smaller than any list element, return
    # the first list element index.
    if pos == 0:
        return 0

    # if input is larger than any list element, return the
    # index of just the last element index.
    if pos == len(elements):
        return len(elements) - 1

    before = elements[pos - 1]
    after = elements[pos]

    # if after is closer than before, use after.
    if after - search < search - before:
        return pos

    # else, before was the closest!
    return pos - 1


SPREAD_TYPES = {"market", "debit", "credit", "even"}
SINGLE_TYPES = {"market", "limit", "stop", "stop_limit"}
ORDER_TYPES = SPREAD_TYPES | SINGLE_TYPES
DURATIONS = {"day", "gtc", "pre", "post"}

EQUITY_SIDES = {"buy", "buy_to_cover", "sell", "sell_short"}
OPTION_SIDES = {"buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"}


def rootFromOCC(what):
    # OCC symbols always have a fixed-length 15 byte description
    # after the initial variable-length symbol name.
    # e.g. AAPL200828P00450000
    return what[:-15]


def occEncode(what):
    # TODO: add shorthand for date to allow
    #   - "end of this week"
    #   - "end of N weeks from now"
    #   - "next monthly expiration"
    #   - "next N monthly expiration"
    try:
        # re-creating: SPY190605C00282000
        # SYMBOL + DATE + CALL/PUT + 5 digit pre-digit + 3 digit cents
        symbol, when, type, price = what.split("/")

        # 8 digits wide, front-filled with zeros, and back filled by 1000
        # max contract price is $9999.999
        fmtPrice = f"{float(price) * 1000:08.0f}"
        what = f"{symbol}{when}{type}{fmtPrice}".upper()
    finally:
        # else, return complete symbol or return original symbol
        # (it's setup with try/finally because we want to handle both our
        #  custom OCC format with slashes and regular things like BRK/B which
        #  will exception-out from the try and just return here unaltered)
        return what


def prefixDictKeys(ds: List[Dict[str, Union[str, int, float]]]):
    # otoco: 'type' must be one of limit, stop, stop_limit
    # oto: first: limit, stop, stop_limit; second: market, + prev
    # oco: same as oto

    # f"duration[0]": "day",
    # return properly formatted param dict where values exist
    return [
        {f"{key}[{idx}]": val for key, val in d.items() if val is not None}
        for idx, d in enumerate(ds)
    ]


SIDE_CLOSE_MAP = {
    "buy": "sell",
    "buy_to_open": "sell_to_close",
    "sell_to_open": "buy_to_close",
    "sell_short": "buy_to_cover",
}

SIDE_OPPOSITE_MAP = {
    "buy": "sell_short",
    "buy_to_open": "sell_to_open",
    "sell_to_open": "buy_to_open",
    "sell_short": "buy",
}

OPT_SWITCH = {"P": "C", "C": "P"}


@dataclass
class Order:
    """One component of an order.
    For single orders, this represents the entire order.
    For OTO, OCO, and OTOCO orders, each leg is a different Order."""

    type: Literal[Literal["limit", "stop", "stop_limit"], "market"]
    duration: Literal["day", "gtc", "pre", "post"]
    symbol: str
    quantity: int
    side: Union[
        Literal["buy", "buy_to_cover", "sell", "sell_short"],
        Literal["buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"],
    ]
    price: Optional[float]
    option_symbol: Optional[str]
    stop: Optional[float] = None

    def createClosingOrder(self):
        closing = self.copy()
        closing.side = self.getClosingSide()

        # caller needs to reset the price, obviously
        closing.price = None
        closing.stop = None

        return closing

    def createFlipOrder(self):
        """Flip from long/short to short/long, but note this doesn't
        change calls to puts, to changes long calls to short calls, etc
        """
        flip = self.copy()
        flip.side = self.getFlipSide()

        # caller needs to reset the price, obviously
        flip.price = None
        flip.stop = None

        return flip

    def createOppositeOrder(self):
        """Flip exposure sides.
        If order is an option, calls and puts are switched instead of going
        short on calls or puts.

        Also note: if this is a "buy call, stop loss, switch to put" strategy,
        the new self.option_symbol probably isn't at the correct strike for
        switching sides.
        (e.g. bought call at $300, went to $315, stop out, you want to switch to
        probably (maybe?) puts at $315 instead of $300 depending on the price
        movement you want exposure to.)
        """
        opposite = self.copy()
        if self.isOption():
            # can't assign to string indexes, so split it, assign it, join it
            parts = list(self.option_symbol)
            parts[-9] = self.getOppositeExposure()
            self.option_symbol = "".join(parts)
        else:
            self.side = self.getOppositeExposure()

        # caller needs to reset the price, obviously
        opposite.price = None
        opposite.stop = None

        return opposite

    def getClosingSide(self):
        """How to close an open order for any given side"""
        # Note: this is only for the opening side.
        # Closing order types of 'sell', 'buy_to_close', 'sell_to_close' don't have opposite sides
        return SIDE_CLOSE_MAP[self.side]

    def getFlipSide(self):
        """Which side to take if we want to flip this position
        from long to short or short to long
        """
        return SIDE_OPPOSITE_MAP[self.side]

    def isOption(self):
        if self.option_symbol:
            return True

        return False

    def isCall(self):
        if self.option_symbol:
            pc = self.option_symbol[-9]
            if pc == "C":
                return True

            return False

        return None

    def getOppositeExposure(self):
        # If we are an option, switch put->call, call->put
        if self.option_symbol:
            pc = self.option_symbol[-9]
            assert pc in {"C", "P"}, f"OCC isn't right? {self}"
            return OPT_SWITCH[pc]

        # else, switch buy->short_sell, short_sell->buy
        return self.getFlipSide()


@dataclass
class OrderSingle:
    """single equity or option order"""

    what: str

    def __post_init__(self):
        assert self.what in {"equity", "option"}

    def create(self):
        out = dataclasses.asdict(self)
        out["class"] = out.pop("what")

        return {k: v for k, v in out.items() if v is not None}


@dataclass
class OrderEquity(OrderSingle, Order):
    what: str = "equity"


@dataclass
class OrderOption(OrderSingle, Order):
    what: str = "option"


@dataclass
class Leg:
    option_symbol: str
    side: Literal["buy_to_open", "buy_to_close", "sell_to_open", "sell_to_close"]
    quantity: int


@dataclass
class OrderMultileg:
    """multi-leg option order, up to 4 legs"""

    # https://documentation.tradier.com/brokerage-api/trading/place-multileg-order
    symbol: str
    type: Literal["market", "debit", "credit", "even"]
    duration: Literal["day", "gtc"]

    legs: List[Leg]
    price: Optional[float] = None  # not required for Market orders
    what: str = "multileg"

    def create(self):
        assert len(legs) <= 4, "too many legs? only four legs at most, please"
        out = dataclasses.asdict(self)
        out["class"] = out.pop("what")
        prefixedLegs = prefixDictKeys(out.pop("legs"))
        for pl in prefixedLegs:
            out.update(pl)

        return {k: v for k, v in out.items() if v is not None}


@dataclass
class OrderTrigger:
    legs: List[Order]

    def create(self):
        out = dataclasses.asdict(self)
        out["class"] = out.pop("what")
        prefixedLegs = prefixDictKeys(out.pop("legs"))
        for pl in prefixedLegs:
            out.update(pl)

        return {k: v for k, v in out.items() if v is not None}


@dataclass
class OTO(OrderTrigger):
    what: str = "oto"


@dataclass
class OCO(OrderTrigger):
    what: str = "oco"


@dataclass
class OTOCO(OrderTrigger):
    what: str = "otoco"


@dataclass
class Position:
    costBasis: float  # negative if short
    quantity: int  # negative if short
    symbol: str

    isOption: bool = False
    isLong: bool = False

    def __post_init__(self):
        if len(self.symbol) > 15:
            self.isOption = True

        if self.quantity > 0:
            self.isLong = True

    def asLeg(self, close=False):
        assert self.isOption, f"Not an option for {self}?"
        if close:
            if self.isLong:
                side = "sell_to_close"
            else:
                side = "buy_to_close"
        else:
            if self.isLong:
                side = "buy_to_open"
            else:
                side = "sell_to_open"

        return Leg(self.symbol, side, self.quantity)


@dataclass
class Spread:
    positions: List[Position]

    def close(self, price="walk"):
        root = rootFromOCC(positions[0].symbol)

        # if original cost basis is POSITIVE we have a CREDIT so
        # we need to close with a DEBIT (buy it back)
        # else, original cost basis is NEGATIVE so we need to
        # cloes with a CREDIT (sell it)
        # Note: this doesn't support rollout strategies where
        # you may want to swap at even. This is only for closing
        # simple spreads right now.
        if sum([p.costBasis for p in self.positions]) > 0:
            type = "credit"
        else:
            type = "debit"

        return OrderMultileg(
            symbol=root,
            type=type,
            duration="gtc",
            price=price,
            legs=[p.asLeg(close=True) for p in positions],
        )


def reverseTrade(position: Position):
    pass


@dataclass
class TradierCredentials:
    mode: str  # dev, prod, sandbox
    session: aiohttp.ClientSession

    # map of {mode: api key}, provided as a map in case you want to run
    # in 'dev' mode for testing orders, but still get live quotes which
    # are always accessed using the 'prod' key.
    credentialMap: dict[str, str]  # map of {mode: api key}

    # live market data websocket session id
    sessionId: Optional[str] = None

    # live account updates websocket session id
    sessionIdAccount: Optional[str] = None

    def __post_init__(self):
        mode = self.mode
        assert (
            mode == "dev" or mode == "prod" or mode == "sandbox"
        ), f"Mode must be 'dev' or 'prod' or 'sandbox' but you gave: {mode}"

        def genLoginHeader(key):
            return {
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
            }

        self.endpoints = {
            "dev": "https://sandbox.tradier.com",
            "prod": "https://api.tradier.com",
            "sandbox": "https://sandbox.tradier.com",
        }

        self.useEndpoint = mode

        self.key = self.credentialMap[mode]
        self.base = self.endpoints[mode]

        # cache credential header so we don't regenerate it every request
        self.omniCredentials = genLoginHeader(self.key)

        # Sometimes we need explicit prod credentials (streaming quotes)
        try:
            # if 'prod' in credential map, always use prod for quotes.
            self.omniCredentialsProd = genLoginHeader(self.credentialMap["prod"])
        except:
            # else, just use the current mode for quotes (which will be either
            # delayed by 15+ minutes or non-existing)
            self.omniCredentialsProd = self.omniCredentials

    def urlHeaders(self, endpoint):
        def genURL(endpoint):
            return f"{self.base}/{endpoint}"

        return genURL(endpoint), self.omniCredentials

    def urlHeadersProd(self, endpoint):
        """Require production key all the time (streaming quotes)"""

        def genURL(endpoint):
            return f"{self.endpoints['prod']}/{endpoint}"

        return genURL(endpoint), self.omniCredentialsProd

    # https://documentation.tradier.com/brokerage-api/markets/get-etb
    def getETB(self):
        url, headers = self.urlHeaders(f"v1/markets/etb")
        return FetchCache(
            self.session, url, "etb-list", refreshMinutes=300, headers=headers
        ).get()

    # https://documentation.tradier.com/brokerage-api/user/get-profile
    def getProfile(self):
        url, headers = self.urlHeaders(f"v1/user/profile")
        return self.session.get(url, headers=headers)

    # https://documentation.tradier.com/brokerage-api/accounts/get-account-balance
    def getBalances(self, accountId):
        url, headers = self.urlHeaders(f"v1/accounts/{accountId}/balances")
        return self.session.get(url, headers=headers)

    # https://documentation.tradier.com/brokerage-api/accounts/get-account-positions
    def getPositions(self, accountId):
        url, headers = self.urlHeaders(f"v1/accounts/{accountId}/positions")
        return self.session.get(url, headers=headers)

    # https://documentation.tradier.com/brokerage-api/accounts/get-account-history
    def getHistory(self, accountId):
        url, headers = self.urlHeaders(f"v1/accounts/{accountId}/history")
        args = {"limit": 5000}
        return self.session.get(url, headers=headers, params=args)

    # https://documentation.tradier.com/brokerage-api/accounts/get-account-gainloss
    def getGainLoss(self, accountId):
        url, headers = self.urlHeaders(f"v1/accounts/{accountId}/gainloss")
        args = {"limit": 5000}
        return self.session.get(url, headers=headers, params=args)

    # https://documentation.tradier.com/brokerage-api/accounts/get-account-order
    def getOrder(self, accountId, orderId):
        url, headers = self.urlHeaders(f"v1/accounts/{accountId}/orders/{orderId}")
        return self.session.get(url, headers=headers)

    # https://documentation.tradier.com/brokerage-api/accounts/get-account-orders
    def getOrders(self, accountId):
        url, headers = self.urlHeaders(f"v1/accounts/{accountId}/orders")
        return self.session.get(url, headers=headers)

    # https://documentation.tradier.com/brokerage-api/trading/place-oto-order
    # https://documentation.tradier.com/brokerage-api/trading/place-otoco-order
    # https://documentation.tradier.com/brokerage-api/trading/preview-order
    def placeOrder(self, accountId, order, preview=False):
        url, headers = self.urlHeaders(f"v1/accounts/{accountId}/orders")

        # order type must be one of: equity, option, multileg, combo, oto, oco, otoco
        data = order.create()

        if preview:
            data["preview"] = "true"

        logger.info("Sending order: {}", data)
        return self.session.post(url, headers=headers, data=data)

    # https://documentation.tradier.com/brokerage-api/trading/cancel-order
    def cancelOrder(self, accountId, orderId):
        url, headers = self.urlHeaders(f"v1/accounts/{accountId}/orders/{orderId}")
        return self.session.delete(url, headers=headers)

    # https://documentation.tradier.com/brokerage-api/trading/change-order
    def updateOrder(
        self,
        accountId,
        orderId,
        type=None,
        duration=None,
        price=None,
        stop=None,
    ):
        url, headers = self.urlHeaders(f"v1/accounts/{accountId}/orders/{orderId}")

        # yes, it's bad to use 'type' as a variable name, but it's contained to this
        # method where we don't need the underlying type() operator itself.
        if type and type not in ORDER_TYPES:
            raise Exception("Invalid order type!")

        if duration and duration not in DURATIONS:
            raise Exception("Invalid duration!")

        data = {"type": type, "duration": duration, "price": price, "stop": stop}
        sendData = {k: v for k, v in data.items() if v is not None}
        return self.session.put(url, headers=headers, data=sendData)

    # https://documentation.tradier.com/brokerage-api/markets/get-quotes
    def getQuotes(self, symbols):
        args = {"symbols": ",".join([occEncode(s) for s in symbols])}

        # Always use production header for quotes
        # (using sandbox header for quotes returns 15-minute delayed values!)
        url, headers = self.urlHeadersProd("v1/markets/quotes")
        return self.session.get(url, headers=headers, params=args)

    # https://documentation.tradier.com/brokerage-api/markets/get-history
    def getQuotesHistory(self, symbol, interval, start, end):
        args = {"symbol": symbol, "interval": interval, "start": start, "end": end}

        # Always use production header for quotes
        # (using sandbox header for quotes returns 15-minute delayed values!)
        url, headers = self.urlHeadersProd("v1/markets/history")
        return self.session.get(url, headers=headers, params=args)

    # https://documentation.tradier.com/brokerage-api/markets/get-timesales
    def getTimeAndSales(self, symbol, interval, start, end):
        args = {"symbol": symbol, "interval": interval, "start": start, "end": end}
        url, headers = self.urlHeadersProd("v1/markets/timesales")
        return self.session.get(url, headers=headers, params=args)

    # https://documentation.tradier.com/brokerage-api/markets/get-options-expirations
    def getExpirations(self, symbol):
        # Use SPXW expirations for SPX because we can't query SPXW directly.
        if symbol.upper() == "SPX":
            symbol = "SPXW"

        args = {"symbol": symbol}
        url, headers = self.urlHeaders("v1/markets/options/expirations")

        # Also modify symbol so things like BRK/A become BRK-A because
        # trying to write a path with an extra slash obviously breaks
        # our directory structure.
        return FetchCache(
            self.session,
            url,
            f"expirations-{symbol.replace('/', '-').upper()}",
            headers=headers,
            params=args,
        ).get()

    # https://documentation.tradier.com/brokerage-api/markets/get-options-strikes
    def getStrikes(self, symbol, expiration):
        args = {"symbol": symbol, "expiration": expiration}
        url, headers = self.urlHeaders("v1/markets/options/strikes")
        return FetchCache(
            self.session,
            url,
            f"strikes-{symbol.upper()}-{expiration}",
            headers=headers,
            params=args,
        ).get()

    # Endpoints under /markets have a 120 request per minute rate limit.
    # https://documentation.tradier.com/brokerage-api/markets/get-options-chains
    # Single result
    def getOptionsChains(self, symbol, expiration):
        # We intentionally don't return greeks because their greeks are only
        # updated once per hour by ORATS which doesn't make sense at all.
        args = {"symbol": symbol, "expiration": expiration, "greeks": "false"}
        url, headers = self.urlHeaders("v1/markets/options/chains")
        return self.session.get(url, headers=headers, params=args)

    # https://documentation.tradier.com/brokerage-api/markets/create-events-session
    def createStreamingSession(self):
        # Streaming always uses production endpoints
        url, headers = self.urlHeadersProd("v1/markets/events/session")
        return self.session.post(url, headers=headers)

    # https://documentation.tradier.com/brokerage-api/streaming/create-account-session
    def createStreamingSessionAccount(self):
        # Streaming always uses production endpoints
        url, headers = self.urlHeadersProd("v1/accounts/events/session")
        return self.session.post(url, headers=headers)

    # https://documentation.tradier.com/brokerage-api/streaming/wss-market-websocket
    def addWebsocketSymbols(
        self, sessionId, symbols, filter=["quote", "summary", "timesale"]
    ) -> bytes:
        return orjson.dumps(
            dict(symbols=list(symbols), sessionid=sessionId, filter=filter)
        )

    async def populateStreamingSession(self):
        # The tradier "session id for streaming quotes" system is a bit tricky
        # and their documentation is flat out wrong in describing how it works
        # in places.
        # It seems to work as:
        #  - for websocket streaming: one session ID is good for an entire
        #    websocket session, but if you disconnect, you need a new session.
        #  - for http streaming, you need a new session every time you update
        #    your symbols if the previous session is older than 5 minutes.
        logger.info("Requesting new streaming session...")
        ss = await self.createStreamingSession()
        sessionIdResult = await ss.json()
        logger.info(
            "Got streaming session: {}...", sessionIdResult["stream"]["sessionid"][:8]
        )
        self.sessionId = sessionIdResult["stream"]["sessionid"]

    async def populateStreamingSessionAccount(self):
        logger.info("Requesting new account streaming session...")
        ss = await self.createStreamingSessionAccount()
        sessionIdResult = await ss.json()
        logger.info(
            "Got account streaming session: {}...",
            sessionIdResult["stream"]["sessionid"][:8],
        )
        self.sessionIdAccount = sessionIdResult["stream"]["sessionid"]

    async def websocketSubscribe(self, ws, symbols):
        """Subscribe to symbols for streaming market data."""
        try:
            # If this is the first connection, create new ID
            if not self.sessionId:
                await self.populateStreamingSession()

            logger.info("Subscribing to {}", symbols)

            # use saved ID
            req = self.addWebsocketSymbols(self.sessionId, symbols)
            await ws.send(req)
            return True
        except:
            logger.exception(f"Failed to websocket subscribe?")
            return False

    # https://documentation.tradier.com/brokerage-api/streaming/wss-market-websocket
    async def websocketConnect(self, cxn=None):
        """Connect to WebSocket for streaming market data."""
        # if already connected, throw away the existing session and reconnect.
        try:
            if cxn:
                await cxn.close()
        except:
            pass
        finally:
            # session ID no longer valid if we close the connection
            self.sessionId = None

        # populate a new session because we'll probably subscribe() soon
        await self.populateStreamingSession()

        # now connect...
        return await websockets.connect(
            "wss://ws.tradier.com/v1/markets/events",
            compression=None,
            ping_interval=10,
            ping_timeout=30,
            close_timeout=1,
            max_queue=2 ** 32,
            read_limit=2 ** 20,
        )

    # https://documentation.tradier.com/brokerage-api/streaming/wss-account-websocket
    def addWebsocketSymbolsAccountEvents(self, sessionId, events=["order"]) -> bytes:
        """Subscribe to account updates with the current session ID.

        Currently the only streaming update type is "order" updates with
        status values of "open", "pending", "filled", etc."""
        return orjson.dumps(dict(sessionid=sessionId, events=events))

    async def websocketSubscribeAccount(self, ws):
        """Subscribe to account updates on a live websocket connection."""
        try:
            # Always create new ID because if we cache it, it may be
            # too old to be used again (and we don't want to TTL it)
            await self.populateStreamingSessionAccount()

            logger.info("Subscribing to account updates...")

            # use saved ID
            req = self.addWebsocketSymbolsAccountEvents(self.sessionIdAccount)
            await ws.send(req)
            return True
        except:
            logger.exception(f"Failed to websocket subscribe to account events?")
            return False

    # https://documentation.tradier.com/brokerage-api/streaming/wss-account-websocket
    async def websocketConnectAccount(self, cxn=None):
        """Connect to WebSocket for streaming order/account updates."""
        # if already connected, throw away the existing session and reconnect.
        try:
            if cxn:
                await cxn.close()
        except:
            pass
        finally:
            # session ID no longer valid if we close the connection
            # TODO: auto-expire cached session ids after 5-10 minutes
            self.sessionId = None

        # now connect...
        return await websockets.connect(
            "wss://ws.tradier.com/v1/accounts/events",
            compression=None,
            ping_interval=10,
            ping_timeout=30,
            close_timeout=1,
            max_queue=2 ** 32,
            read_limit=2 ** 20,
        )

    # Continuous http stream of JSON payloads:
    # https://documentation.tradier.com/brokerage-api/streaming/get-markets-events
    # Read using https://docs.aiohttp.org/en/stable/streams.html
    def getStream(self, symbols: list, sessionid):
        # Streaming doesn't use the API key.
        # Must request streaming  session key above, then use key for sessionid.
        args = {
            "symbols": ",".join(symbols),
            "sessionid": sessionid,
            "linebreak": "true",
        }
        url = "https://stream.tradier.com/v1/markets/events"

        # Remove default timeout settings since we are long polling
        # https://docs.aiohttp.org/en/latest/client_quickstart.html#timeouts
        timeout = aiohttp.ClientTimeout(total=0)
        return self.session.get(
            url, params=args, headers={"Accept": "application/json"}, timeout=timeout
        )


@dataclass
class TradierClient:
    # Rate limits:
    # - https://documentation.tradier.com/brokerage-api/overview/rate-limiting
    # Basically:
    # 120 requests per minute for /markets
    # 120 requests per minute for /accounts, /watchlists, /users, /orders
    # 60 requests per minute for trade requests
    # Tradier rate limits aren't sliding based on the time of the first
    # request, but rather they reset the hard limit every new minute.
    # (So: if doing 120 requests at 00:00.50, next 120 can start at 00:01.00
    #      instead of needing to wait for the full "1 minute limit" wraparound)
    accessLevel: str
    accountId: str
    credentialMap: dict[str, str]

    # cache some symbol details to avoid re-fetching and re-parsing
    strikeCache: dict = field(default_factory=dict)  # key: (symbol, expiration)
    expirationCache: dict = field(default_factory=dict)  # key: symbol

    def __post_init__(self):
        self.cr = None
        self.session = None

    async def setup(self):
        try:
            self.session = aiohttp.ClientSession()
            # Quotes are only available from "prod" endpoint
            self.cr = TradierCredentials(
                self.accessLevel, self.session, self.credentialMap
            )

            # We used to do this here, but this is purely a client
            # implementation detail if they are buying/selling at increments.
            # The client should populate the metadata and evaluate it on their own.
            if False:
                self.mmd = MarketMetadata()
                await self.mmd.setup()

                # Fetch and load penny tick options
                await self.mmd.populateOptionTickLookupDict()
        except:
            logger.exception("Failed to setup?")

        # load popular expirations into cache
        if True:
            await asyncio.gather(
                self.expirationAway("SPX"),
                self.expirationAway("QQQ"),
                self.expirationAway("IWM"),
                self.expirationAway("DIA"),
                self.expirationAway("AAPL"),
                self.expirationAway("FB"),
                self.expirationAway("NVDA"),
                self.expirationAway("SPY"),
                self.expirationAway("TSLA"),
                self.expirationAway("NFLX"),
                self.expirationAway("AMD"),
                self.expirationAway("SHOP"),
            )

    async def getQuoteAndAllOptionChains(self, symbol):
        """Return all strikes across every date for stock symbol 'symbol'
        Note: the API doesn't allow us to filter out calls vs. puts, so you must manually
        filter those on the result set yourself.

        Also, we are fetching ALL dates which is a lot of detail for 3-4 weekly options like SPY/SPX.
        Future improvement would expose "get for date range only."
        """
        # Steps:
        #  - get quote for underlying to base initial ITM/OTM guesses around
        #  - get expiration dates for symbol
        #  - get quotes for each expiration date
        #  - filter quotes by current highest ask price
        #  - return best candidates for a rapid buy-in
        # TODO: subscribe to live trades then track live acceleration of premiums
        #         - also useful for GTFO orders since Tradier doesn't have trailing stop

        getOptionDates = self.cr.getExpirations(symbol)
        getUnderlyingQuote = self.cr.getQuotes([symbol])

        # Run quote retrieval and options dates retrieval concurrently
        firstResponses = [
            asyncio.create_task(x) for x in [getOptionDates, getUnderlyingQuote]
        ]

        dates = []
        for startingResult in asyncio.as_completed(firstResponses):
            result = await startingResult
            if isinstance(result, str):
                # is cached (or fetched-then-cached) expirations
                got = orjson.loads(result)
            else:
                # else, is network result thing
                got = orjson.loads(await result.read())

            # Maybe we made a bad request?
            try:
                # Inspect output to see if this is the quote or dates result...
                if "quotes" in got:
                    # https://documentation.tradier.com/brokerage-api/markets/get-quotes
                    # Note: if providing multiple symbols for quotes, the ['quote'] would
                    # be an array and not a standalone objet.

                    # The Tradier API can randomly reutrn a list or a single object. sigh.
                    underlyingQuote = got["quotes"]["quote"]
                else:
                    assert "expirations" in got
                    # https://documentation.tradier.com/brokerage-api/markets/get-options-expirations
                    dates = got["expirations"]["date"]
            except:
                # any data errors we just skip over everything
                return None, None

        allChains = await self.allChainsForSymbolDates(symbol, dates)
        return underlyingQuote, allChains

    async def quickQuote(self, symbol):
        quote = await self.cr.getQuotes([symbol])

        # See note above where sometimes ["quotes"]["quote"] is an object and
        # other times its a list. shrug.
        quote = orjson.loads(await quote.read())
        quote = quote["quotes"]["quote"]
        return pd.json_normalize(quote)

    async def allChainsForSymbolDates(self, symbol, dates):
        # logger.info("Getting symbol for dates: {} {}", symbol, dates)
        allChainGetters = [
            asyncio.create_task(self.cr.getOptionsChains(symbol, date))
            for date in dates
        ]

        allChains = []
        # Run all option matrix/chain gets concurrently then collect
        for oneResult in asyncio.as_completed(allChainGetters):
            result = await oneResult
            chain = orjson.loads(await result.read())

            # https://documentation.tradier.com/brokerage-api/markets/get-options-chains
            # There's an edge case where this doesn't exist for things like de-listed
            # tickers (the tradier API maintains their expirations, but the listings are gone?)
            o = chain["options"]
            if not o:
                continue

            chain = chain["options"]["option"]

            # Load each date
            allChains.append(pd.json_normalize(chain))

        if allChains:
            return pd.concat(allChains)

    def etb(self):
        return self.cr.getETB()

    def profile(self):
        return self.cr.getProfile()

    def quotes(self, symbols):
        return self.cr.getQuotes(symbols)

    def quotesHistory(self, symbol, interval, start, end):
        return self.cr.getQuotesHistory(symbol, interval, start, end)

    def timesales(self, symbol, interval, start, end):
        return self.cr.getTimeAndSales(symbol, interval, start, end)

    def expirations(self, symbol):
        return self.cr.getExpirations(symbol)

    def strikes(self, symbol, expiration):
        return self.cr.getStrikes(symbol, expiration)

    async def expirationAway(self, symbol, away=0):
        """Return the expiration date N expirations away.

        Requesting 0 is usually a weekly unless it's SPXW/SPY/QQQ with
        their 3 (M/W/Last) or 4 (M/W/Last + quarterly) expirations per week
        (or unless the symbol doesn't have weeklies).

        If you request too far ahead you probably get LEAPS if the symbol has them.

        Quarterlies are the last calendar day of a quarter:
            - march 31
            - june 30
            - september 30
            - december 31
        """
        key = symbol.upper()
        try:
            expires = self.expirationCache[key]
        except:
            # key didn't exist, so create it
            expiresDict = orjson.loads(await self.expirations(symbol))
            expires = expiresDict["expirations"]

        try:
            ed = expires["date"]
            self.expirationCache[key] = expires
            return ed[away]
        except:
            logger.exception(f"No expirations for {symbol}!")
            return None

    async def strikeByWidthFromPrice(
        self, symbol, expiration, price, percentAway=None, dollarAway=None
    ):
        """Returns a pair of (low, ATM, high) strikes based on the given price.

        Can specify either a percent diff from current price or dollars off price.

        Percentages are decimal inputs so 10% is input as 0.10
        """
        symbol = symbol.upper()
        key = (symbol, expiration)
        try:
            strikes = self.strikeCache[key]
        except:
            # key didn't exist, so create it
            strikeDict = orjson.loads(await self.strikes(symbol, expiration))
            strikes = strikeDict["strikes"]["strike"]
            self.strikeCache[key] = strikes

        # "percentAway" is a decimal percentage.
        # e.g. 3 percent is 0.03
        # Note: it's valid to specify a zero percent/dollar to just
        # get (ATM - 1, ATM, ATM + 1) strike values
        if percentAway is not None:
            priceDiff = price * percentAway
        elif dollarAway is not None:
            # or you can specify an actual dollar value to target
            priceDiff = dollarAway
        else:
            raise Exception("Must specify either percent or dollar amount!")

        checkLow = price - priceDiff
        checkHigh = price + priceDiff

        # find the closest strikes to the price offsets
        atm = closestSortedIndex(strikes, price)
        low = closestSortedIndex(strikes, checkLow)
        high = closestSortedIndex(strikes, checkHigh)

        # if the low is ATM (and not the lowest), go one under ATM
        if low == atm and atm > 0:
            low -= 1

        # if high is ATM (and not the highest), go one above ATM
        if high == atm and atm < len(strikes):
            high += 1

        # low should never equal high (unless there's only one strike?)
        assert low != high, f"Why is low==high? ({low}, {atm}, {high})"

        # low should never equal atm if atm is higher than lowest strike
        assert atm > 0 and low != atm, f"Why is low atm?  ({low}, {atm}, {high})"

        return (strikes[low], strikes[atm], strikes[high])

    def balances(self):
        return self.cr.getBalances(self.accountId)

    def order(self, orderId):
        return self.cr.getOrder(self.accountId, orderId)

    def orders(self):
        return self.cr.getOrders(self.accountId)

    def positions(self):
        return self.cr.getPositions(self.accountId)

    def history(self):
        return self.cr.getHistory(self.accountId)

    def gainloss(self):
        return self.cr.getGainLoss(self.accountId)

    def placeOrder(self, order, preview):
        # order one of our order classes: equity, option, multileg, combo, oto, oco, otoco
        return self.cr.placeOrder(self.accountId, order, preview)

    def updateOrder(self, orderId, type=None, duration=None, price=None, stop=None):
        return self.cr.updateOrder(self.accountId, orderId, type, duration, price, stop)

    def cancelOrder(self, orderId):
        return self.cr.cancelOrder(self.accountId, orderId)

    def websocketConnect(self, cxn=None):
        return self.cr.websocketConnect(cxn)

    def websocketSubscribe(self, ws, symbols):
        return self.cr.websocketSubscribe(ws, symbols)

    def websocketConnectAccount(self, cxn=None):
        return self.cr.websocketConnectAccount(cxn)

    def websocketSubscribeAccount(self, ws):
        return self.cr.websocketSubscribeAccount(ws)
