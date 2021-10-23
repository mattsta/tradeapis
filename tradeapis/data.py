"""Helpers to fetch market metadata for option trading."""

import aiohttp
import arrow  # type: ignore
import csv

from enum import Enum
from typing import Optional
from collections import Counter
from dataclasses import dataclass, field
from collections import defaultdict
from bs4 import BeautifulSoup  # type: ignore
import io
import re

from loguru import logger
from mutil.dcache import FetchCache  # type: ignore
from mutil.numeric import roundnear5, roundnear10  # type: ignore

FG_URL = "https://money.cnn.com/data/fear-and-greed/"

# Collect all FG values from the FG page by text extraction.
FGS = dict(
    now=r"Greed Now:\s+(\d+)",
    prev=r"Greed Previous Close:\s+(\d+)",
    week=r"Greed 1 Week Ago:\s+(\d+)",
    month=r"Greed 1 Month Ago:\s+(\d+)",
    year=r"Greed 1 Year Ago:\s+(\d+)",
)

FAT_URL = "https://markets.cboe.com/us/options/notices/reasonability/"

# Enum for option price limits:
#   - all strikes allow 0.01 increments on all prices
#   - normal is $0.05 increments priced under $3, $0.10 above $3+
#   - PPP is $0.01 under $3 orders, $0.05 above $3+
Tick = Enum("Tick", "ALL_1 NORMAL PPP")


def pennyTickTypeURL():
    """Generate URL for current day's option tick intervals"""
    # http://markets.cboe.com/us/options/market_statistics/penny_tick_type/?mkt=exo
    now = arrow.now().to("US/Eastern")
    # if today is a weekend, back up to friday
    # (not checking for holidays currently since this would just
    #  be for weekend testing at the moment...)
    weekday = now.isoweekday()

    # If current time is before 9am ET, use the previous day.
    if now.hour < 9:
        if weekday == 0:
            # need to go back THREE days to friday
            weekday = 8
        else:
            # use previous day if not monday. if monday, use sunday.
            # (iso weekdays are 1-7)
            weekday = weekday - 1
            now -= datetime.timedelta(days=1)

    if weekday >= 6:
        # turn weekend back into friday by subtracting
        # number of days since the 5th day (sat=1, sun=2)
        now -= datetime.timedelta(days=weekday % 5)

    year = now.year
    month = f"{now.month:02}"
    day = f"{now.day:02}"

    # TODO: when is the day list posted? beginning of day? end of day?
    exchange = "bzx"  # "edgx" ?
    mkt = "opt"  # "exo" ?
    url = f"http://markets.cboe.com/us/options/market_statistics/penny_tick_type/{year}/{month}/{exchange}_options_rpt_penny_tick_type_{year}{month}{day}.csv-dl?mkt={mkt}"
    return url


@dataclass
class MarketMetadata:
    ppp: dict[str, Tick] = field(
        default_factory=lambda: defaultdict(lambda: Tick.NORMAL)
    )

    async def setup(self):
        self.session = aiohttp.ClientSession()

    async def shutdown(self):
        await self.session.close()

    async def feargreed(self) -> dict[str, int]:
        """Fetch the fear and greed index from money.cnn.com.

        Note: fetched result is cached on 5 minute boundaries."""
        content = await FetchCache(
            self.session, FG_URL, "greed-fear", refreshMinutes=5
        ).get()

        fgs = {}

        # run fear/greed extraction across all RG regexes
        # (the page has time periods of: now, yesterday, last {week,month,year})
        for name, regex in FGS.items():
            fgs[name] = int(re.findall(regex, content)[0])

        return fgs

    async def populateOptionTickLookupDict(self) -> None:
        """Fetch the current penny tick option list on a 4 hour cache.

        Use the fetched CSV to populate the local penny/5/10 cent interval cache.

        Note: Only penny options (below $3 or for all) are populated, so we use
        a default dict with a default value of Tick.NORMAL to allow all fetches
        (this also means just because a value is retrieved in self.ppp[] doesn't
        mean the symbol has options at all or is even valid)"""

        content = await FetchCache(
            self.session, pennyTickTypeURL(), "penny-tick", refreshMinutes=60 * 4
        ).get()

        for row in csv.reader(io.StringIO(content)):
            try:
                symbol = row[0]
                # name = row[1]
                unit = row[2]
            except:
                # content isn't formatted properly, we can't manage it
                logger.error(f"PPP URL has weird row of {row}")
                break

            if unit == "Pennies to 3.00":
                ticks = Tick.PPP
            elif unit == "Pennies all":
                ticks = Tick.ALL_1
            else:
                if unit == "Tick Type":
                    # just the header row
                    continue

                raise Exception(f"Unexpected unit type? {row}")

            self.ppp[symbol] = ticks

    def adjustLimitTick(self, symbol: str, price: float, isBuy: bool) -> float:
        """Method adjusts 'price' to the next proper 1/5/10 cent boundary
        based on what the price and option chain suppports.

        'isBuy' is used as a proxy where we want to round up to next increment
        if buying and round down if selling (for quicker fills).

        Quoth the CBOE:
        "Generally, minimum tick for options trading below $3 is $0.05
        and for all other series, $0.10. For classes participating in
        the Penny Pilot Program, the minimum tick for options trading below
        $3 is $0.01 and $0.05 for options trading at $3 or above."

        See:
         - http://www.cboe.com/products/options-on-single-stocks-and-exchange-traded-products/options-on-single-stocks/equity-options-specs
        """

        symbol = symbol.upper()

        # if this is a full option symbol, turn it into the equity symbol
        if len(symbol) > 15:
            symbol = symbol[:-15]

        unit = self.ppp[symbol]
        logger.info(f"[{symbol}] Adjusting for tick [{unit}]: {price}")

        if unit == Tick.ALL_1:
            # no restrictions on "all cents" symbols
            return price

        if unit == Tick.PPP:
            # PPP symbols allow $0.01 below $3, then $0.05 above
            if price <= 3.00:
                return price

            return roundnear5(price, isBuy)

        if unit == Tick.NORMAL:
            # Normal symbols are $0.05 under $3 and $0.10 above $3
            if price <= 3.00:
                return roundnear5(price, isBuy)

            return roundnear10(price, isBuy)

        assert None, "How did you get here?"

    async def fats(self):
        content = await FetchCache(self.session, FAT_URL, "fat-finger-openings").get()
        soup = BeautifulSoup(content, "html.parser")
        syms = Counter()

        # 0 will be today, 1 will be yesterday, etc
        daysAgo = []

        # get all rows holding symbols (matches document format as of 2020-08-20)
        spans = soup.find_all("span")
        for span in spans:
            if "style" in span.attrs:
                members = span.text.split(", ")
                daysAgo.append(members)
                syms.update(members)

        # returns all symbols (can use 'syms.most_common(X)' to get highest X)
        # returns the list of volatile symbols for this trading session
        # (if requested after about 8:45am on the trading day)
        return syms, spans[0]
