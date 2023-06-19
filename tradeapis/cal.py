from typing import Sequence
import datetime
import bisect

import pandas as pd

from mutil.timer import Timer

from cachetools import cached, TTLCache

# Default cache current-date calendars for for 6.5 hours
CALENDAR_CACHE_SECONDS = 60 * 60 * 6.5

# TODO: finish these abstractions and remove the copy/paste from mattplat.stats.*
# and also replace into icli and tcli


# max 6.5 hour timeout on calendar caching
@cached(cache=TTLCache(maxsize=128, ttl=CALENDAR_CACHE_SECONDS))
def getMarketCalendar(market: str, start=None, stop=None):
    """Return a market calendar describing market days with holidays/weekends excluded."""
    import pandas_market_calendars as mcal

    with Timer(f"Fetched Calendar {market}"):
        cal = mcal.get_calendar(market)
        # default two year max lookback
        if not start:
            start = pd.Timestamp("now").floor("D") - pd.Timedelta(365.25 * 2, "D")

        if not stop:
            stop = "today"

        sched = cal.schedule(start_date=start, end_date=stop, tz="US/Eastern")
        return sched


@cached(cache=TTLCache(maxsize=512, ttl=CALENDAR_CACHE_SECONDS // 2))
def marketDaysAgo(daysBack: int, market="NASDAQ") -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (start, end) pd.Timestamp for 'daysBack' market days ago from now."""
    # use dynamic start lookback date so we aren't pulling 2 years of calendar
    # days by default. The '* 3' provides a large enough buffer to counteract any
    # weekends/holidays.
    # Also, the dynamic start date will give us a new 'getMarketCalendar()' call
    # signature to always cause new lookups on new day rollovers so we don't get
    # a previous cached "most recent calendar up to the current day" result.
    # The DOWNSIDE of the dynamic start lookup date, is every new start lookup date
    # requests an entire new calendar.
    # TODO: add a "get oldest calendar" operation where if the current requested
    # date is with in the largest requested calendar start/end times, we return
    # the oldest calendar instead since it will include the current requested range.
    # startLookupDate = pd.Timestamp("now").floor("D") - pd.Timedelta((daysBack * 10) % 3, "D")

    # for now, revert to the "request 2 years up front, but always use the 2 year calendar for filtering"
    startLookupDate = None

    cal = getMarketCalendar(market, startLookupDate)

    lookbackStart = cal.iloc[-daysBack].market_open.floor("D")
    lookbackEnd = cal.iloc[-1].market_open.floor("D")

    return lookbackStart, lookbackEnd


def marketDaysBack(lookbackDays: Sequence[int], market="NASDAQ") -> list[pd.Timestamp]:
    """Return pd.Timestamp for days corresponding to N market days back for each N in 'lookbackDays'"""
    lookbackDays = sorted(lookbackDays)

    # Default multiple higher than largest lookback for (huge) buffer against holidays/weekends
    startLookupDate = pd.Timestamp("now").floor("D") - pd.Timedelta(
        lookbackDays[-1] * 3, "D"
    )

    marketDays = getMarketCalendar(market, startLookupDate)

    lookbackDaysAsTimestamps = [
        marketDays.iloc[-daysBack].market_open.floor("D") for daysBack in lookbackDays
    ]

    return lookbackDaysAsTimestamps


def marketDaysBetweenDates(start, stop, market="NASDAQ") -> list[datetime.date]:
    """Return date objects for each market day between 'start' and 'stop'"""
    dates = getMarketCalendar(market, start, stop)

    # still works fine if no dates are returned because the .market_open iterator
    # will just return nothing, so we return an empty list.
    return [x.date() for x in dates.market_open]


def marketLastDayOfWeekBetweenDates(
    start, stop, market="NASDAQ"
) -> list[datetime.date]:
    """Return date objects for each last-day-of-week market day between 'start' and 'stop'

    Note: this is smart enough to return Thursday if Friday is a holday, and normally Friday
          for all other normal weeks."""
    dates = getMarketCalendar(market, start, stop)

    # still works fine if no dates are returned because the .market_open iterator
    # will just return nothing, so we return an empty list.
    return [x.date() for x in dates.resample("W-FRI").last().market_open]


def indexExpirationDates(year: int) -> list[datetime.date]:
    """Calculate the index expiration dates for a year.

    Index futures expire on the third thursday of every end-of-quarter month.
    """

    expirations = []
    # only iterate end-of-quarter months
    for month in range(3, 13, 3):
        # start at the first day of the month
        third_thursday = datetime.date(year, month, 1)

        # this looks weird as a while loop, but it's fine.
        while third_thursday.weekday() != 4:  # Find the first Thursday then stop
            third_thursday += datetime.timedelta(days=1)

        # jump to the expiration thursday since we just found the first thursday in the loop
        third_thursday += datetime.timedelta(days=14)

        # collect
        expirations.append(third_thursday)

    return expirations


def indexRollDates(year: int) -> list[datetime.date]:
    """Calculate the logical "next roll forward dates" for futures in a year.

    Index futures aren't continuous and "expire" the 3rd Thursday of every end of quarter month,
    and the "roll forward date" is officially 4 days before the expiration: so, the Monday before
    the 4th Thursday of the month is when most volume jumps to the next expiration quarter.

    Also see: https://www.cmegroup.com/trading/equity-index/rolldates.html
    """

    # jump back to the monday before each expiration thursday
    roll_forward_dates = [
        x - datetime.timedelta(days=4) for x in indexExpirationDates(year)
    ]
    return roll_forward_dates


def nextDateFor(what, start: datetime.date) -> datetime.date:
    """Helper to abstract away the 'find next date accounting for end-of-year' condition.

    Used by nextFuturesExpirationDate / nextFuturesRollDate for underlying date fetching.
    """
    month = start.month

    # sleazy algorithm:
    #   if NOT december, get current year expirations, bisect, return
    #   IF december, get current year and next year, bisect, return

    bis = what(start.year)
    if month == 12:
        bis += what(start.year + 1)

    nextidx = bisect.bisect(bis, start)
    return bis[nextidx]


def nextFuturesExpirationDate(start: datetime.date) -> datetime.date:
    """Return the next index futures expirations date given a start date."""

    return nextDateFor(indexExpirationDates, start)


def nextFuturesRollDate(start: datetime.date) -> datetime.date:
    """Return the next index futures expirations date given a start date."""

    return nextDateFor(indexRollDates, start)
