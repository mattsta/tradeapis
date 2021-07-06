import datetime
from typing import Optional, Sequence

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
    """ Return a market calendar describing market days with holidays/weekends excluded. """
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
    """ Return (start, end) pd.Timestamp for 'daysBack' market days ago from now. """
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
    """ Return pd.Timestamp for days corresponding to N market days back for each N in 'lookbackDays' """
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
    """ Return date objects for each market day between 'start' and 'stop' """
    dates = getMarketCalendar(market, start, stop)

    # still works fine if no dates are returned because the .market_open iterator
    # will just return nothing, so we return an empty list.
    return [x.date() for x in dates.market_open]
