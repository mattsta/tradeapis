"""ifthen is a way to do things when you want them done.

if you follow one path, then where do the others lead?

let's codify an OODA loop.

Observe — read data source(s)
Orient — pre-process data to prepare for evaluation
Decide - evaluate condition(s)
Act — execute if condition matched

if [observation X]: [execute event Y]

For more details, see the test_ifthen.py file and read the grammar.

Once again, we ask for your patience while we implement half a programming langauge and a minimal virtual machine...
"""

from __future__ import annotations

import asyncio
import datetime
import graphlib
import math
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Final, Literal

import yaml
from lark import Lark, Transformer, v_args

Operator: Final = Enum("Operator", "GT GTE LT LTE EQ NE BY TRUE FALSE EXISTS NONE")
Trigger: Final = Enum("Trigger", "IF WHILE")

# Symbol is anything you uniquely use to identify an instrument inside your dataset/environment.
# You can use direct string names like AAPL, but more complex configurations like spreads would need more detailed details.
# type Symbol = Any
Symbol = Hashable
PredicateId = Hashable
PredicateSuccessCmd = str


###############################################################################
#
# Le Object Hierarchy
#
###############################################################################


def extract_dataextractor(start) -> Iterable[DataExtractor]:
    """We want to return every DataExtractor property.

    A DataExtractor property is either an lval or rval on a Condition or Operation or an argument to a DataFunction,
    but lvals and rvals and DataFunction arguments can also be *further nested* LogicBindings or Operations themselves.
    """
    if isinstance(start, DataExtractor):
        yield start
    elif isinstance(start, DataFunction):
        for arg in start.args:
            yield from extract_dataextractor(arg)
    elif isinstance(start, DataCondition | Operation):
        yield from extract_dataextractor(start.lval)
        yield from extract_dataextractor(start.rval)
    elif isinstance(start, LogicBinding):
        for row in start.conds:
            yield from extract_dataextractor(row)


@dataclass(slots=True)
class DateAware:
    """Simple checker for running conditions within or outside of time bounds.

    TODO: timezone as property for all the comparisons...
    """

    before: datetime.datetime | None = None
    after: datetime.datetime | None = None
    notbefore: datetime.datetime | None = None
    notafter: datetime.datetime | None = None

    def canRun(self) -> bool:
        now = datetime.datetime.now()

        if self.notbefore and now < self.notbefore:
            return False

        if self.notafter and now > self.notafter:
            return False

        if self.before and now > self.before:
            return False

        if self.after and now < self.after:
            return False

        if self.after and self.before:
            if self.after < self.before:
                return False

            if self.before and self.after:
                assert self.before <= now <= self.after

        return True


@dataclass(slots=True)
class Condition(DateAware):
    op: Operator = Operator.NONE

    checked: int = 0

    def execute(self, lval: Any, rval: Any = None) -> bool:
        if not self.canRun():
            return False

        result = False

        # fail fast up front if we are a numeric comparison but input values aren't populated
        if self.op in {Operator.GT, Operator.GTE, Operator.LT, Operator.LTE} and (
            lval is None or rval is None
        ):
            return False

        match self.op:
            case Operator.GT:
                result = lval > rval
            case Operator.GTE:
                result = lval >= rval
            case Operator.LT:
                result = lval < rval
            case Operator.LTE:
                result = lval <= rval
            case Operator.EQ:
                # fmt: off
                if isinstance(lval, int | float | Decimal) and isinstance(rval, int | float | Decimal):
                    result = math.isclose(lval, rval, rel_tol=1e-06)
                elif isinstance(rval, bool | type(None)):
                    result = lval is rval
                else:
                    result = lval == rval
                # fmt: on
            case Operator.NE:
                # fmt: off
                if isinstance(lval, int | float | Decimal) and isinstance(rval, int | float | Decimal):
                    result = not math.isclose(lval, rval, rel_tol=1e-06)
                elif isinstance(rval, bool | type(None)):
                    result = lval is not rval
                else:
                    result = lval != rval
                # fmt: on
            case Operator.EXISTS:
                result = lval is not None
            case Operator.TRUE:
                result = bool(lval) is True
            case Operator.FALSE:
                result = bool(lval) is False
            case Operator.BY:
                ...

        return result


class Operation(ABC):
    """add, sub, mul, div"""

    lval: Resolvable
    rval: Resolvable


class LogicBinding(ABC):
    conds: Iterable[Any]


class Resolvable(ABC):
    """Something with a zero-argument .resolve() method"""

    datafield: str | None
    current: Any | None

    symbol: Symbol | None
    alias: str | None

    @abstractmethod
    def resolve(self, *args, **kwargs):
        raise NotImplementedError


@dataclass(slots=True)
class DataString(Resolvable):
    """Just a string to compare against directly."""

    content: str

    def resolve(self) -> str:
        return self.content


@dataclass(slots=True)
class DataExtractor(Resolvable):
    """Data Extractor extracts data fields relevant to a single symbol (or instrument) you are using for decision making.

    For example, you could extract:
      - the current bid price (if any)
      - the current ask price (if any)
      - the current midpoint of bid/ask (if any)
      - VWAP (daily? anchored in the past?)
      - D timeframe N SMA
      - D timeframe N EMA
      - algo values (stop levels / entry triggers)
      - and probably other things? It all depends on what data you are _providing_ to be extracted in the first place.
    """

    # user input symbol (could just be a position alias)
    symbol: Symbol | None = None

    # actual underlying localSymbol
    actual: Symbol | None = None

    # if user requests to access this Extractor by a different name than the symbol for using elsewhere
    alias: str | None = None

    datafield: str | None = None
    timeframe: int | None = None

    # Optionally, just use a value directly without anything else populated...
    value: float | None = None

    # a custom-purpose, single-value provider to return a live view of what we represent.
    datafetcher: Callable | None = field(default=None, repr=False)

    # store dervived value for logging/reference/metrics
    current: Any = None

    def __hash__(self) -> int:
        return hash((self.symbol, self.datafield, self.timeframe, self.value))

    def resolve(self) -> float | int | bool | None:
        # if we are a directly provided value, just return without any other consideration
        # logger.info("Resolving here: {}", self)
        if self.value is not None:
            self.current = self.value
            return self.value

        # else, use live value extractor which could either be a generic data fetcher or a fully encapsulated data fetcher (just ignoring arguments)
        assert self.datafetcher
        # logger.info("resolving: {} {} {}", self.symbol, self.datafield, self.timeframe)
        self.current = self.datafetcher(self.symbol, self.datafield, self.timeframe)
        # logger.info("got: {}", self.current)

        return self.current


@dataclass(slots=True)
class DataFunction(Resolvable):
    """Run a parametertized function resolving to a single value."""

    datafield: str
    args: Iterable[Any]

    # if user requests to access this DataFunction by a name for using elsewhere
    alias: str | None = None

    datafetcher: Callable | None = field(default=None, repr=False)

    # if this is an async thing, wait for the result to appear here
    mailbox: dict[str, Any] = field(default_factory=dict)

    # if this is an async thing, run using this scheduler as a runnable fn directly
    # don't 'repr' scheduler because it shows the ENTIRE icli application state as the bound partial
    scheduler: Any = field(default=None, repr=False)
    scheduled: Any = None

    current: Any = None

    def resolve(self, *args, **kwargs):
        components = [a.resolve() for a in self.args]

        assert self.datafetcher
        if asyncio.iscoroutine(self.datafetcher) or asyncio.iscoroutinefunction(
            self.datafetcher
        ):
            assert self.scheduler

            # if we previously ran and we have data, use the data
            if self.mailbox:
                self.current = self.mailbox["result"]
            elif not self.scheduled:
                # if not previously run and also data isn't received yet, launch it.
                # API / Contract for a function datafetcher is (dictionary for return value, *arguments)
                self.scheduled = self.scheduler(
                    self.datafetcher(self.mailbox, *components)
                )

                # tell caller we have NO DATA yet, so fail the predicate until a result appears
                return None
        else:
            # else, if is just a regular function, we can run it directly without any of the other magic required
            self.current = self.datafetcher(*components)

        return self.current


@dataclass(slots=True)
class DataCondition(Resolvable):
    """Combination of data request and operator requests for unifying the data and operator execution.

    Example:
        - AAPL > VWAP
        - APPL 55s SMA 5 > SMA 10
        - AAPL 300s algo volstop side=long stopped=True

    So we need to contain: the operator check condition (above, below, exists/True, equals, ...) and the operator values to used for the compare.
    """

    # e.g. Condition(Operator.GT)
    condition: Condition

    # e.g. DataExtractor("AAPL", dataFields, "midpoint")
    #      DataExtractor("AAPL", dataFields, "VWAP", 15)
    #      DataExtractor("AAPL", dataFields, "SMA", 300, 5)
    #      DataExtractor("AAPL", dataFields, "SMA", 300, 10)
    lval: DataExtractor
    rval: DataExtractor | None = None

    # save the state of whether the last run was successful or not
    passing: bool | None = None

    def resolve(self, *args, **kwargs):
        lval = self.lval.resolve(*args, **kwargs)
        rval = self.rval.resolve(*args, **kwargs) if self.rval else None
        self.passing = self.condition.execute(lval, rval)
        return self.passing


@dataclass(slots=True)
class DurationTrigger(Resolvable, Operation):
    """A way to delay triggering predicate activation until an Operation has been successful for an entire duration of time.

    The purpose of this is to prevent bounce-out trigger problems where an ask jumps from $3 to $50 then back to $3 because
    the market is havig a volatility spike and the top-of-book goes offline revealing the underlying trap depth in the book.

    Our mechanism here is once an operation is successful, we begin the timer and if any future check fails before the
    specified duration, we reset the timer back to zero. The DurationTrigger is only successful if the underlying operation
    is successful for the entire time duration requested.

    e.g. if AAPL bid >= 200 for 5 minutes: say SAFE 200
    """

    lval: Resolvable
    rval: Resolvable | None = None  # type: ignore

    # duration of trigger in python time format (float seconds)
    duration: float = 0.0

    # timestamp in time.time() format when the operation was first successful
    activated: float = 0.0

    # current timestamp if the operation continues to be successful.
    # the DurationTrigger is successful when (continuing - activated >= duration)
    continuing: float = 0.0
    remaining: float | None = None

    @property
    def success(self) -> bool:
        self.current = self.continuing - self.activated
        return self.current >= self.duration

    def resolve(self, *args, **kwargs):
        if found := self.lval.resolve():
            # if value passed, we have an active result to track.
            # NOTE: Values here should be predicate return values (boolean results) and not numeric results, so we
            #       shouldn't have to worry about matching against a value of 0 as a valid result.
            now = time.time()
            if not self.activated:
                self.activated = now
            else:
                self.continuing = now
                self.remaining = self.duration - (now - self.activated)
                if self.success:
                    return found
        else:
            # else, if lval isn't passing, always reset our state back to zero.
            self.continuing = 0.0
            self.activated = 0.0
            self.remaining = None

        return None


@dataclass(slots=True)
class OperationAddPercent(Resolvable, Operation):
    lval: Resolvable
    rval: Resolvable

    current: Any | None = None

    def resolve(self, *args, **kwargs):
        # X + (X * percent)
        lval = self.lval.resolve()
        self.current = lval + (lval * self.rval.resolve())
        return self.current


@dataclass(slots=True)
class OperationSubtractPercent(Resolvable, Operation):
    lval: Resolvable
    rval: Resolvable

    current: Any | None = None

    def resolve(self, *args, **kwargs):
        # X - (X * percent)
        lval = self.lval.resolve()
        self.current = lval - (lval * self.rval.resolve())
        return self.current


@dataclass(slots=True)
class OperationAdd(Resolvable, Operation):
    lval: Resolvable
    rval: Resolvable

    current: Any | None = None

    def resolve(self, *args, **kwargs):
        self.current = self.lval.resolve() + self.rval.resolve()
        return self.current


@dataclass(slots=True)
class OperationSub(Resolvable, Operation):
    lval: Resolvable
    rval: Resolvable

    current: Any | None = None

    def resolve(self, *args, **kwargs):
        self.current = self.lval.resolve() - self.rval.resolve()
        return self.current


@dataclass(slots=True)
class OperationMul(Resolvable, Operation):
    lval: Resolvable
    rval: Resolvable

    current: Any | None = None

    def resolve(self, *args, **kwargs):
        self.current = self.lval.resolve() * self.rval.resolve()
        return self.current


@dataclass(slots=True)
class OperationDiv(Resolvable, Operation):
    lval: Resolvable
    rval: Resolvable

    current: Any | None = None

    def resolve(self, *args, **kwargs):
        self.current = self.lval.resolve() / self.rval.resolve()
        return self.current


@dataclass(slots=True)
class LogicBindingAND(Resolvable, LogicBinding):
    """Unlimited condition AND statement.

    We return True if EVERY condition evalutes to True."""

    conds: Iterable[Resolvable]

    def resolve(self, *args, **kwargs):
        for cond in self.conds:
            # AND: if ANY condition fails, return False (i.e. they must ALL pass for success)
            if not cond.resolve(*args, **kwargs):
                return False

        return True


@dataclass(slots=True)
class LogicBindingOR(Resolvable, LogicBinding):
    """Unlimited condition OR statement.

    We return True if ANY condition evalutes to True."""

    conds: Iterable[Resolvable]

    def resolve(self, *args, **kwargs):
        for cond in self.conds:
            # OR: if ANY condition passes, return True (i.e. only ONE is needed for success)
            if cond.resolve(*args, **kwargs):
                return True

        return False


@dataclass(slots=True)
class ConditionExecution(DateAware):
    checked: int = 0
    executed: int = 0

    # number of times to run this when the condition triggers. Runs once by default.
    run: int = 1

    lastRun: datetime.datetime | None = None

    # how long to wait (in seconds) between runs if running multiple executions
    delay: float = 0

    def execute(self, lval: Any, rval: Any) -> bool:
        # If we already ran up to (or beyond?!) our allowed run count, don't run again.
        if self.run >= self.executed:
            return False

        self.checked += 1

        if not self.canRun():
            return False

        self.lastRun = datetime.datetime.now()
        self.executed += 1

        assert None, "Not completed yet..."
        return True


class Checkable(ABC):
    def check(self) -> bool | PredicateSuccessCmd | None:
        raise NotImplementedError


@dataclass(slots=True)
class IfThenIntent(Checkable):
    """Representation of a boolean query involving one or more data sources to check against."""

    # One intent may potentially operate on multiple symbols, so we provide them
    # all up-front here. The intended use of 'symbols' is to index each symbol externally
    # so every symbol inbound update can then discover and trigger IfThen condition running.

    # original full input string
    request: str

    # .symbols can be actual string symbols or users can replace it to be application-specific lookup
    # keys for referencing discovery of these symbols bidirectionally in the application<->predicate interface.
    symbols: frozenset[Symbol] = field(default_factory=frozenset)

    # is this an If or a While?
    # 'If' conditions terminate on a single success.
    # 'While' conditions continue to repeat their command every time their match happens.
    trigger: Trigger | None = None

    # All the matching logic operations anchored to one initial condition tree root object.
    conditions: Resolvable | None = None

    # what command to provide back to the caller when this condition passes
    cmd: PredicateSuccessCmd = ""

    createdAt: float = field(default_factory=lambda: time.time())

    checked: int = 0
    lastCheckedAt: float | None = None

    # only run if not completed yet...
    complete: bool = False

    # some timestamps for nice reporting every time this completes (if running multiple times)
    completedAt: list[float] = field(default_factory=list)

    def check(self) -> bool | PredicateSuccessCmd | None:
        """Check all current conditions, and if the conditions are true, return list of executions to run."""

        # if complete, return no actions and do no checks. We are done forever.
        # TODO: how do we handle completion on repeated runtime intents? Do we just manually reset .complete in the runtime return since we are controlling it directly?
        if self.complete:
            # None means DELETE THIS AND NEVER CALL IT AGAIN
            return None

        self.checked += 1

        if self.conditions:
            self.lastCheckedAt = time.time()

            if self.conditions.resolve():
                # We can technically complete multiple times if callers reset 'complete=False' after returning.
                self.completedAt.append(time.time())

                # we are only complete if this is an IF condition
                # ("while" conditions never complete and run every time the conditions pass)
                if self.trigger == Trigger.IF:
                    self.complete = True

                # if command string has a format request, format then return
                if "." in self.cmd:
                    replacements = self.buildResultsDict()

                    sendcmd = self.cmd
                    for k, v in replacements.items():
                        if k:
                            sendcmd = sendcmd.replace(k, v)

                    return sendcmd

                # String means CONDITION PASSED SO EXECUTE THIS PLZZZZZZ
                return self.cmd

        # False means continue and try again later
        return False

    def buildResultsDict(self) -> dict[str, Any]:
        """Iterate all extractors and build a dict of their current values.

        Keys of the dict are represented as dotted: symbol, datafield, timeframe.

        Note: key lookup here uses the ORIGINAL extractor symbol as passed down from the original
              user predicate syntax input and _NOT_ any updated or replaced self.symbols changes.
        """

        results = {}
        for extractor in self.extractors():
            key = ".".join(
                map(
                    str,
                    filter(
                        None,
                        [
                            extractor.alias or extractor.symbol,
                            extractor.datafield,
                            extractor.timeframe,
                        ],
                    ),
                )
            )

            results[key] = str(extractor.current)

        # print("Results:", results)
        return results

    def functions(self) -> Iterable[DataFunction]:
        """Enumerate functions much like extractors."""

        def extract(start) -> Iterable[DataFunction]:
            """We want to return every DataFunction property in any lval, rval, or nested in other function arguments."""
            if isinstance(start, DataFunction):
                yield start
                for arg in start.args:
                    yield from extract(arg)
            elif isinstance(start, DataCondition | Operation):
                yield from extract(start.lval)
                yield from extract(start.rval)
            elif isinstance(start, LogicBinding):
                for row in start.conds:
                    yield from extract(row)

        if self.conditions:
            yield from extract(self.conditions)

    def extractors(self) -> Iterable[DataExtractor]:
        """Return each data extractor in this entire condition hierarchy.

        Useful for post-processing when we need to attach per-symbol data bindings
        based on live lookups after predicate creation.
        """

        if self.conditions:
            yield from extract_dataextractor(self.conditions)

    def operators(self) -> Iterable[DataExtractor]:
        # TODO: yield OPERATORS so we can add "operator alias" results to command replacement...
        return []

    @property
    def actuals(self) -> set[Symbol]:
        es = set()
        for e in self.extractors():
            # only add if exists (single numeric values don't have symbols)
            if actual := (e.actual or e.symbol):
                es.add(actual)

        return es


###############################################################################
#
# Le Parser
#
###############################################################################

lang: Final = r"""
    // result of 'cmd' is value returned to the caller
    cmd: iif | wwhile

    // if: run command once when condition matches then delete the command
    iif: "if" overspec runcmd

    // whie: run command every time condition matches
    wwhile: "while" overspec runcmd

    // command is remainder of input after a colon
    runcmd: (":" /[^\s+].+/)?

    // operations creation
    overspec: opspec

    // standard tree flow: input is operation, junction, opspec, or matching parens of any of the previous.
    ?opspec: logical

    ?logical: operation
        | logical  "or"i operation -> junction_or
        | logical "and"i operation -> junction_and

    ?operation: opspec_lvalrval | "(" opspec ")"

    // if 300s AAPL SMA 5 > SMA 10
    // we allow math WITHIN an opspec so you can do things like AAPL bid > AAPL low + 3 => (AAPL bid) > ((AAPL low) + 3)
    ?opspec_lvalrval: sum

    ?sum: product
        | sum "+" product -> add
        | sum "-" product -> sub

    ?product: valspec_final
        | product "*" valspec_final -> mul
        | product "/" valspec_final -> div

    // Note: we need to wrap entire spec captures using "for" and "exists" in parens so they associate
    //      to the ENTIRE statement and don't just capture the trailing condition (e.g. "AAPL bid > 20 exists" -> "20 exists" isn't what we want)
    ?valspec_final: opspec_lvalrval OP opspec_lvalrval -> opspec_lvalrval
                  | "(" opspec_lvalrval ")" "exists" -> opspec_exists
                  | "(" opspec_lvalrval ")" "for" time_duration -> opspec_duration
                  | "(" opspec_lvalrval ")"
                  | "(" opspec_lvalrval ")" "as"i CNAME -> resolvable_alias
                  | symbol "{" opspec_nosymbol "}" -> resolvable_distribute_symbol
                  | valspec
                  | function
                  | string -> resolvable_string
                  | (valspec | function) "as"i CNAME -> resolvable_alias

    # "nosymbol" operations are copies of "opspec" operations but where we don't include the symbol in each operation because we have factored it out.
    #  regular: if AAPL mid > AAPL ema 20: say GOING UP
    # nosymbol: if AAPL { mid > ema 20 }: say GOING UP
    ?opspec_nosymbol: logical_nosymbol

    ?logical_nosymbol: operation_nosymbol
        | logical_nosymbol  "or"i operation_nosymbol -> junction_or
        | logical_nosymbol "and"i operation_nosymbol -> junction_and

    ?operation_nosymbol: opspec_lvalrval_nosymbol | "(" opspec_nosymbol ")"

    ?opspec_lvalrval_nosymbol: sum_nosymbol

    ?sum_nosymbol: product_nosymbol
        | sum_nosymbol "+" product_nosymbol -> add
        | sum_nosymbol "-" product_nosymbol -> sub

    ?product_nosymbol: valspec_final_nosymbol
        | product_nosymbol "*" valspec_final_nosymbol -> mul
        | product_nosymbol "/" valspec_final_nosymbol -> div

    ?valspec_final_nosymbol: opspec_lvalrval_nosymbol OP opspec_lvalrval_nosymbol -> opspec_lvalrval
                  | "(" opspec_lvalrval_nosymbol ")" "exists" -> opspec_exists
                  | "(" opspec_lvalrval_nosymbol ")" "for" time_duration -> opspec_duration
                  | "(" opspec_lvalrval_nosymbol ")"
                  | "(" opspec_lvalrval_nosymbol ")" "as"i CNAME -> resolvable_alias
                  | valspec_nosymbol
                  | function
                  | string -> resolvable_string
                  | (valspec_nosymbol | function) "as"i CNAME -> resolvable_alias

    // Symbol can be any instrument-like thing or also all numeric if it's just contract ids or a position lookup request (:N)
    // The trailing "not dash or : or space" check is so things like "if AAPL mid > 0: say hello" doesn't trap '0:' as a symbol lookup because symbols can't end in a colon or ': '.
    symbol: /[:\/]?([_0-9A-Za-z-:]{1,32}[^-: ])/ | string

    string: ESCAPED_STRING_DOUBLE | ESCAPED_STRING_SINGLE

    _STRING_INNER: /.*?/
    _STRING_ESC_INNER: _STRING_INNER /(?<!\\)(\\\\)*?/

    // "string of things"
    ESCAPED_STRING_DOUBLE: "\"" _STRING_ESC_INNER "\""

    // 'string of things'
    ESCAPED_STRING_SINGLE: "'" _STRING_ESC_INNER "'"

    OP: ">" | ">=" | "<=" | "<" | "=" | "==" | "over"i | "under"i | "is"i | "is not"i | "not"i | "!="

    // specify a time duration we resolve in code later. e.g. "5 minutes", "5 mins", "5m", "3s", "3 seconds", "2 hours" etc
    time_duration: FLOAT|INT CNAME

    // custom function runner with comma separated arguments...
    // (arguments can also be other dynamic values)
    // Note: none of these functions are 'built in' so you must resolve .functions() and attach handlers after each predicate creation.
    function: CNAME "(" ((valspec | function) ("," (valspec | function))*)? ")"

    // Note: order matters here. We need to try 'valspec_direct' first so it captures standalone floats first
    //       (instead of turning '0.33' into Symbol=0, Field=.33)
    valspec: valspec_direct | valspec_single | valspec_full
    valspec_nosymbol: valspec_direct | valspec_single_nosymbol | valspec_full_nosymbol

    // a single lookup like: AAPL bid
    // also covers algo use cases like: AAPL AAPL.30.bar.vwap (yes, we need to double-state the symbol currently)
    // Note: we allow colons and dashes in the MIDDLE of namespaces, but NOT at the end because then it would conflict with the final command marker.
    valspec_single: symbol FIELD
    valspec_single_nosymbol: FIELD

    // string, number, sub-field
    // AAPL twema 60
    // AAPL ema:prev:log 180
    valspec_full: symbol FIELD INT
    valspec_full_nosymbol: FIELD INT

    // Allow fields to be things with colon or dot or dash separators (in the middle) or allow it to be a single character.
    FIELD:  /[A-Za-z0-9-:\.]+[A-Za-z0-9]+/ | /[A-Za-z]/

    // just a number for use cases against direct static comparisons
    valspec_direct: SIGNED_FLOAT | SIGNED_NUMBER | NUMBER | BOOLS | valspec_direct_percent

    valspec_direct_percent: (SIGNED_FLOAT | SIGNED_NUMBER | NUMBER) "%"

    BOOLS: "true"i | "false"i | "none"i

    %import common.NUMBER
    %import common.CNAME
    %import common.FLOAT
    %import common.INT
    %import common.SIGNED_FLOAT
    %import common.SIGNED_NUMBER
    %import common.NUMBER

    WHITESPACE: (" " | "\t" | "\n")+
    COMMENT: /#[^\n]*/

    %ignore WHITESPACE
    %ignore COMMENT
"""


def opToOp(op) -> Condition:
    r = Operator.NONE
    match str(op).lower():
        case ">" | "over":
            r = Operator.GT
        case ">=":
            r = Operator.GTE
        case "<=":
            r = Operator.LTE
        case "<" | "under":
            r = Operator.LT
        case "is" | "==" | "=":
            r = Operator.EQ
        case "not" | "is not" | "!=":
            r = Operator.NE
        case "exists":
            r = Operator.EXISTS

    assert r != Operator.NONE, f"Unexpected operation? Got: {op=}"
    return Condition(op=r)


@v_args(inline=True)
@dataclass(slots=True)
class TreeToIfThen(Transformer):
    src: str

    # We will populate the intent _as we parse along the way_ then return it at the end.
    dst: IfThenIntent = field(init=False)

    # for type correctness, we accumulate symbols in 'presymbols' then convert it
    # to a frozenset() for final placement in the intent
    presymbols: set[str] = field(default_factory=set)

    def __post_init__(self):
        self.dst = IfThenIntent(self.src)

    def cmd(self, *stuff):
        """The final result rule"""
        self.dst.symbols = frozenset(self.presymbols)
        return self.dst

    def iif(self, opspec, cmds):
        self.dst.trigger = Trigger.IF
        self.dst.cmd = cmds
        return opspec

    def runcmd(self, *val):
        """Command is technically optional, so only return something if it exists"""
        return val[0].strip() if val else None

    def wwhile(self, opspec, cmds):
        self.dst.trigger = Trigger.WHILE
        self.dst.cmd = cmds
        return opspec

    def string(self, anything):
        # remove the embedded quotes from the quoted string
        # because we just want the actual string value without quotes involved
        return str(anything)[1:-1]

    def symbol(self, symbol):
        # Add detected symbol to set of symbols to use for activation triggers
        symbol = symbol.upper()

        # track ALL symbols seen so we can add them to the collective .symbols frozenset upon completion.
        self.presymbols.add(symbol)

        return symbol

    def opspec(self, resolvedops):
        assert None, (
            "This rule should never trigger because 'opspec' is inlined with '?' into 'subspec'"
        )

    def operation(self, oper):
        # print("OPERATION:", oper)
        return oper

    def overspec(self, spec):
        # print(f"[{self.src}] SETTING CONDITIONS:\n", pprint.pformat(spec))
        self.dst.conditions = spec

    def subspec(self, spec):
        return spec

    def add(self, lval, rval):
        # Kinda hack: if we are adding a percentage, this is a different operation and we introspect it all directly here.
        if isinstance(rval.value, dict):
            rval.value = rval.value["percent"] / 100
            return OperationAddPercent(lval, rval)

        return OperationAdd(lval, rval)

    def sub(self, lval, rval):
        # Kinda hack: if we are subtracting a percentage, this is a different operation and we introspect it all directly here.
        if isinstance(rval.value, dict):
            rval.value = rval.value["percent"] / 100
            return OperationSubtractPercent(lval, rval)

        return OperationSub(lval, rval)

    def mul(self, lval, rval):
        assert not isinstance(rval.value, dict), (
            "We can't multiply percents, only add and subtract"
        )
        return OperationMul(lval, rval)

    def div(self, lval, rval):
        assert not isinstance(rval.value, dict), (
            "We can't divide percents, only add and subtract"
        )
        return OperationDiv(lval, rval)

    def junction_or(self, lcond, rcond):
        """This is for combining lvalrval evaluatons with OR"""
        # print(f"JUNCTION OR:", lcond, " :: ", rcond)

        # Even though we have 'lcond' and 'rcond' here, the OR/AND system
        # doesn't have a limit on how many conditions it checks to assert truth.
        conds = (lcond, rcond)
        return LogicBindingOR(conds)

    def junction_and(self, lcond, rcond):
        """This is for combining lvalrval evaluatons with AND"""
        # print(f"JUNCTION AND:", lcond, " :: ", rcond)
        conds = (lcond, rcond)
        return LogicBindingAND(conds)

    def opspec_lvalrval(self, lval, op, rval):
        """This is for ORDER comparison like GT, LT, EQ"""
        opresolved = opToOp(op)
        return DataCondition(opresolved, lval, rval)

    def opspec_exists(self, lval):
        """Just a single condition of if a thing actually exists"""
        opresolved = opToOp("exists")
        return DataCondition(opresolved, lval)

    def opspec_duration(self, lval, duration):
        """Create a wrapper to only trigger success if the underlying operation passes for entire 'duration' interval."""
        return DurationTrigger(lval, duration=duration)

    def valspec(self, vs):
        return DataExtractor(**vs)

    def valspec_nosymbol(self, vs):
        return DataExtractor(**vs)

    def valspec_full(self, symbol, field, timeframe):
        return dict(symbol=symbol, datafield=str(field), timeframe=int(timeframe))

    def valspec_full_nosymbol(self, field, timeframe):
        return dict(datafield=str(field), timeframe=int(timeframe))

    def BOOLS(self, value):
        # Convert a scalar value into something direct to use with no further lookups...
        match value.lower():
            case "true":
                value = True
            case "false":
                value = False
            case "none":
                value = None

        return value

    def valspec_direct(self, value):
        if isinstance(value, str):
            value = float(value)

        return dict(value=value)

    def valspec_direct_percent(self, value):
        value = float(value)
        return dict(percent=value)

    def valspec_single(self, symbol, field):
        return dict(symbol=symbol, datafield=str(field))

    def valspec_single_nosymbol(self, field):
        return dict(datafield=str(field))

    def function(self, fnname, *args):
        return DataFunction(str(fnname), args)

    def resolvable_alias(self, resolvable, alias):
        resolvable.alias = str(alias)
        # print("SETTING ALIAS", alias, "ON", resolvable)
        return resolvable

    def resolvable_string(self, content):
        return DataString(content)

    def resolvable_distribute_symbol(self, symbol, resolvable):
        """Replace all sub-extractors with the SAME symbol.

        Used in syntax where we factor out a symbol from operations like:
            AAPL { mid > low }
        instead of the "regular" way of:
            AAPL mid > AAPL low
        """
        for e in extract_dataextractor(resolvable):
            if e.value is None and e.symbol is None:
                e.symbol = symbol

        return resolvable

    def time_duration(self, num, dur) -> float:
        """Return duration in float second for requested interval."""
        dur = dur.lower()
        num = float(num)

        # Formats we'll accept:
        #  - s* for seconds
        #  - m* for minutes
        #  - h* for hours

        match dur[0]:
            case "s":
                return num
            case "m":
                return num * 60
            case "h":
                return num * 60 * 60
            case _:
                raise ValueError(f"Unsupported duration requested? Got: {dur}")


@dataclass
class IfThen:
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

    def parse(self, text: str) -> IfThenIntent:
        body = text.strip()
        parsed = self.parser.parse(body)

        transformer = TreeToIfThen(src=body)
        transformed = transformer.transform(parsed)

        return transformed


###############################################################################
#
# Le Runtime Add-Ons
#
###############################################################################


@dataclass(slots=True, frozen=True)
class IfThenRuntimeSuccess:
    """PUBLIC Result of the runtime generting a successful result for clients to consume."""

    pid: PredicateId
    cmd: PredicateSuccessCmd

    # include the successful predicate too
    # (because we may delete it ourselves, so this could be the only place the client can see the success state)
    predicate: CheckableRuntime


@dataclass(slots=True, frozen=True)
class IfThenRuntimeError:
    """PUBLIC Result of the runtime generting a failed check for clients to consume."""

    pid: PredicateId
    err: Exception


@dataclass(slots=True, frozen=True)
class IfThenRuntimeResultInternal:
    """INTERNAL Result of an internal Checkable.check() call for things to return/activate/deactivate"""

    # what to run since this predicate completed successfully
    cmd: PredicateSuccessCmd

    # which predicate ids to activate since this current predicate completed
    activate: frozenset[PredicateId] = field(default_factory=frozenset)

    # whether upon return, this predicate should be deactivated from running again
    # (because it completed for now)
    deactivate: bool = False


class CheckableRuntime(ABC):
    active: Any

    def check(self, symbol: Symbol) -> IfThenRuntimeResultInternal | Literal[False]:
        """Check underlying IfThenIntent objects (or other Checkable objects) with a single entry point and unified result type."""
        raise NotImplementedError

    @property
    def symbols(self) -> frozenset[Symbol]:
        """Either the single-predicate symbols result or a merger of _all_ combined predicate symbols if multiple predicates exist on this checker."""
        raise NotImplementedError

    @property
    def id(self) -> PredicateId:
        """Return index id for this runtime predicate wrapper.

        We could potentially create a better or more stable id system in the future,
        but currently IDs only need to remain stable per runtime session.
        """
        return id(self)

    def extractors(self):
        if isinstance(self.active, Iterable):
            for a in self.active:
                yield from a.extractors()
        else:
            yield from self.active.extractors()

    def functions(self):
        if isinstance(self.active, Iterable):
            for a in self.active:
                yield from a.functions()
        else:
            yield from self.active.functions()

    @property
    def actuals(self) -> set[Symbol]:
        """Collect all ACTUAL symbols (or regular if actual not populated) in all extractors anywhere in the active Intents.

        This is required because the active/creation/reference symbol may not be exactly
        the same as the indexing/updating symbol in predicate.symbols.
        """
        raise NotImplementedError

    @property
    def actives(self) -> Iterable[IfThenIntent]:
        """Return an iterable of all active IfThenIntent represented here.

        A CheckableRuntime could have one or more active predicates if this is a tree or has peers.
        """

        def getall(root):
            if isinstance(root, IfThenIntent):
                yield root
            elif isinstance(root, IfThenPeers):
                for p in root.active:
                    yield from getall(p)
            elif isinstance(root, tuple):
                # peer groups store their active members, which isn't matching under 'Peers' above because... reasons
                for r in root:
                    yield from getall(r)
            else:
                assert isinstance(root, CheckableRuntime), f"Got: {root=}?"
                yield from getall(root.active)

        yield from getall(self.active)


@dataclass(slots=True, frozen=True)
class IfThenSingle(CheckableRuntime):
    """Just a single thing. Nothing fancy."""

    active: IfThenIntent

    def check(self, _symbol: Symbol) -> IfThenRuntimeResultInternal | Literal[False]:
        """Run Active predicate then return when complete."""
        got = self.active.check()
        # logger.info("Got from check: {}", got)

        if got:
            assert isinstance(got, PredicateSuccessCmd)

            # We don't allow wrapped IfThenIntents to remain 'complete' because we may
            # want to run them over and over again.
            self.active.complete = False

            # Return: DEACTIVATE SELF, RESULT, ACTIVATE NOTHING ELSE
            return IfThenRuntimeResultInternal(got, deactivate=True)

        return False

    @property
    def symbols(self) -> frozenset[Symbol]:
        return self.active.symbols

    @property
    def actuals(self) -> set[Symbol]:
        return self.active.actuals


@dataclass(slots=True)
class IfThenTree(CheckableRuntime):
    """Represent predicates operating in a tree.

    One predicate is active then child predicates become active when the active predicate completes.
    """

    active: CheckableRuntime
    waiting: frozenset[PredicateId]

    def check(self, symbol: Symbol) -> IfThenRuntimeResultInternal | Literal[False]:
        """Run Active predicate then make 'waiting' become active if Active completes."""
        got = self.active.check(symbol)
        # logger.info("Got from check: {}", got)

        if got:
            assert isinstance(got, IfThenRuntimeResultInternal)

            # Return: ACTIVATE WAITING, DEACTIVATE SELF, RESULT
            # Note: merging activation of the result into our own 'waiting' activation, because
            #       the 'active' predicate check could even be another tree with its own waiting state?
            # One example is a meta-notification system where the 'active' adds its own next operations
            # after completion while _this_ tree augments an additional notification trigger the inner
            # 'active' doesn't need to know about? Maybe?
            return IfThenRuntimeResultInternal(
                got.cmd,
                activate=self.waiting | got.activate,
                deactivate=True,
            )

        return False

    @property
    def symbols(self) -> frozenset[Symbol]:
        return self.active.symbols

    @property
    def actuals(self) -> set[Symbol]:
        return self.active.actuals


@dataclass(slots=True)
class IfThenPeers(CheckableRuntime):
    """Represent predicates operating in a one-cancels-all group.

    ALL predicates are active, but when one predicate completes, all the predicates in the group are stopped.
    """

    # a tuple here because tuples are faster to iterate than lists or sets
    active: tuple[CheckableRuntime, ...]

    checkmap: dict[Symbol, tuple[CheckableRuntime, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Create local mapping of symbols in each intent to which intent to run.

        e.g. if active peers has a predicate mapping for AAPL -> cmd and also NVDA -> cmd,
             we only want to run the AAPL-related predicates on AAPL updates, so we create
             a custom mini-index here mapping symbols to which peers they need to check.
        """

        # build mapping out of lists
        checkmap = defaultdict(list)
        for a in self.active:
            for sym in a.symbols:
                # Note: this replace is SITLL NECESSARY because the client post-updating code DOESNT UPDATE CHECKMAP (yet, at least, maybe we should auto-adjust it somehow)
                checkmap[str(sym).replace("/", "").replace(" ", "")].append(a)

        # convert lists to tuples (slightly faster for iteration)
        # Cast keys to Symbol type for type compatibility
        toplevel: dict[Symbol, tuple[CheckableRuntime, ...]] = {
            k: tuple(v) for k, v in checkmap.items()
        }
        self.checkmap |= toplevel

    def check(self, symbol: Symbol) -> IfThenRuntimeResultInternal | Literal[False]:
        """Run ALL predicates and stop at the first one completed."""

        # For (hopefully) efficiency improvements, we check only predicates attached to the
        # input symbol requested.
        # The choice here was either: run _every_ predicate on _every_ check, but every peer group
        # predicate may not change on every update — OR — create an internal symbol->predicates map
        # then select only active predicates for the symbol update being requested for checking.
        for predicate in self.checkmap.get(symbol, []):
            if got := predicate.check(symbol):
                assert isinstance(got, IfThenRuntimeResultInternal)

                # Return: DEACTIVATE ALL PEERS (but we are the peer runner, so just STOP SELF), RESULT
                # Note: Current usage is callers will only be exposed to a peer group as
                #       this entire IfThenPeers class id key, so all the _individual_ predicates
                #       inside aren't exposed upwards to the user at all. Ergo, here to disable,
                #       we only need to tell the caller to remove _our peer id_ because, since one
                #       success was found here, we are now deactivating this group everywhere.
                # Also: adding any resulting activations into the result because maybe the active
                #       success check here is actually a tree needing its own 'waiting' events to
                #       become active since it completed successfully.
                return IfThenRuntimeResultInternal(
                    got.cmd, activate=got.activate, deactivate=True
                )

        return False

    @property
    def symbols(self) -> frozenset[Symbol]:
        """we are legion"""

        result: set[Symbol] = set()
        for predicate in self.active:
            result |= predicate.symbols

        return frozenset(result)

    @property
    def actuals(self) -> set[Symbol]:
        result: set[Symbol] = set()
        for predicate in self.active:
            result |= predicate.actuals

        return result


@dataclass(slots=True)
class IfThenRuntime:
    """A self-managing runtime allowing scheduling of multiple IfThen instances.

    IfThen instances can be run as:
      - singluar mode: fire-and-done runs (same as running a regular IfThen itself)
      - tree mode: one predicate can have child predicates which only begin running after the parent stops.
      - peer mode: multiple predicates can operate concurrently, but the first predicate to stop automatically removes all peer predicates.

    You can combine tree mode and peer mode to generate something like an attacahed bracket order where:
      - parent: if AAPL mid > 20: place order
      - children as peer group (only activated when 'parent' completes): if AAPL < 15: sell (loss) — OR — if AAPL > 20: sell (profit)
        - Note: children don't have to be all in one peer group; they could just be new individual predicates to launch too.

    The tree relationships are referenced by id so single IfThen predicate instances can even reference themselves like:
      - 1. if AAPL mid > 20: say AAPL UP AT AAPL.mid
      - 2. if AAPL mid < 15: say AAPL DOWN AT AAPL.mid

      set children of 1 to 2.
      set children of 2 to 1.

      Now when 1 completes, 2 begins, and when 2 completes, we re-schedule 1 again. Forever (or until one of them is deleted to stop the cycle).
    """

    # our parser god, praise be
    ifthen: IfThen = field(default_factory=IfThen)

    # mapping of all predicate IDs to individual predicates
    predicates: dict[PredicateId, CheckableRuntime] = field(default_factory=dict)

    # mapping of symbols to predicate ids
    symbolPredicates: dict[Symbol, set[PredicateId]] = field(default_factory=dict)

    # active predicate ids
    active: set[PredicateId] = field(default_factory=set)

    # also note which predicates to delete after they run instead of keeping around for re-use
    once: set[PredicateId] = field(default_factory=set)

    def parse(self, text: str) -> PredicateId:
        """Parse 'text' to an IfThenIntent, but return a local Checkable reference instead.

        We return a reference id to our local Checkable dictionary so we can manage the
        tree and peer hierarchies automatically.

        Note: we wrap all IfThenIntent objects in special Runtime wrappers enabling them
              all to report (result command, new activation ids, deactivation ids) for all results.

        Note: just running 'parse()' does NOT activate a predicate in this runtime. You must manually
              call runtime.activate(pid) for the symbols to appear in the scheduling and activation system.
        """

        created = IfThenSingle(self.ifthen.parse(text))

        pid = created.id
        self.predicates[pid] = created

        return pid

    def clearActive(self):
        self.active.clear()

    def clear(self):
        self.predicates.clear()
        self.symbolPredicates.clear()
        self.active.clear()

    def remove(self, pid: PredicateId) -> CheckableRuntime | None:
        """Remove predicate key (if exists) and return removed IfThenIntent (if it existed).

        Note: you can also just stop a predicate from being scheduled by using deactivate() instead
              which leaves the underlying predicate in-place for future usage without needing to re-create it.
        """

        if found := self.predicates.pop(pid, None):
            # also remove from active and any other top-level symbol details...
            # Note: doesn't remove from any peer groups or waiting tree members, but those will just
            #       fail to be scheduled again if they try to become active.
            # Note: Removing a predicate doesn't cause a peer group to fail (if it is a member of any peer group).
            #       A peer group only stops on a complete result (not on a "missing predicate" lookup).
            self.active.discard(pid)

            for sp in self.symbolPredicates.values():
                # TODO: if 'sp' is empty after the removal, we could remove Symbol from .symbolPredicates too?
                sp.discard(pid)

        return found

    def activate(self, pid: PredicateId, once: bool = False) -> bool:
        """Activate an existing predicate for all of its symbols.

        If 'once' is True, mark this predicate to self-delete after running.

        Note: we require this extra activation step because a user's input symbol data
              may not necessarily be the live update symbol key data (e.g. if user requests by quote position).
              So, after the client updates the predicate with proper symbol indexing overwrites, the client/caller
              can _then_ activate the predicate properly for symbol fetching updates.
        """
        iti = self.predicates.get(pid)
        assert iti, "Why didn't the predicate exist?"

        # We need to populate the symbolPredicates mapping on activation...

        for symbol in iti.symbols:
            if activeSymbolPredicates := self.symbolPredicates.get(symbol):
                activeSymbolPredicates.add(pid)
            else:
                # else, it doesn't exist, so we create it as new
                self.symbolPredicates[symbol] = set([pid])

        self.active.add(pid)

        if once:
            self.once.add(pid)

        return True

    def deactivate(self, pid: PredicateId) -> bool:
        """Remove a predicate id from being scheduled or run again, but don't delete the underlying predicate.

        Use remove() to de-schedule, de-activate, _and_ remove the predicate completely.
        """
        if iti := self.predicates.get(pid):
            # We need to de-populate the symbolPredicates mapping on activation...

            for symbol in iti.symbols:
                if activePredicates := self.symbolPredicates.get(symbol):
                    self.symbolPredicates[symbol].discard(pid)

            self.active.discard(pid)
            return True

        return False

    def report(self):
        return self.predicates, self.active

    def tree(
        self, parent: PredicateId, children: Iterable[PredicateId]
    ) -> PredicateId | None:
        """Create a new callable predicate tree using 'parent' as the active IfThenIntent.

        'children' are ids to be returned as new scheduled events when 'parent' is done.
        """
        if p := self.predicates.get(parent):
            itt = IfThenTree(p, frozenset(children))
            iditt = itt.id
            self.predicates[iditt] = itt
            return iditt

        return None

    def treeReplaceChildren(
        self, treeId: PredicateId, children: Iterable[PredicateId]
    ) -> bool:
        """Replace children on a previously-created tree.

        This is useful if you want to create a circular relationship where a -> b triggers b -> a forever.
        """

        children = frozenset(children)

        if got := self.predicates.get(treeId):
            assert isinstance(got, IfThenTree)

            # verify all the children ids exist
            if not all([c in self.predicates for c in children]):
                return False

            # REPLACE IN-PLACE
            # Note: this is a full replace operation.
            #       For an addition, you would want to manually generate: set(intent.children) | set([newId1, newId2, ...])
            got.waiting = children

            return True

        return False

    def peers(self, peers: Iterable[PredicateId]) -> PredicateId | None:
        """Create a new callable predicate peer group where all ids are active IfThenIntent instances.

        When any peer has a successful check, all peers become de-scheduled from checking anymore ("one cancels all").
        """

        try:
            ps = tuple([self.predicates[p] for p in peers])
        except KeyError:
            # if any id doesn't exist, abandon creating the entire peer group
            return None

        itp = IfThenPeers(ps)
        iditp = itp.id
        self.predicates[iditp] = itp

        return iditp

    def checkExternal(
        self, pid: PredicateId, symbol: Symbol
    ) -> IfThenRuntimeResultInternal | None | Literal[False]:
        """Run a predicate check for a single symbol requiring external predicate accounting.

        'runExternal' requires the client maintain its own mappings of which predicate ids are active for
        submission on each symbol update. This method returns a full (result, activate, deactivate) mapping
        so the client can maintain the mappings themselves.

        See regular 'check()' to let IfThenRuntime maintain all internal state for which predicates, trees,
        and peers are active at any given time.

        Note: symbol parameter is only used if this is a peer group with multiple active predicates,
              otherwise there is always only a single predicate to check against.
        """
        if not (iti := self.predicates.get(pid)):
            # if pid doesn't exist, return None so caller stops trying to call us (hopefully)
            return None

        # if check is successful, return the IfThenRuntimeResultInternal answer.
        # Note: we never get 'None' from these .check() calls because the check return _includes_ a self-deletion request when it completes.
        if got := iti.check(symbol):
            return got

        # else, check failed, so nothing to update
        return False

    def check(self, symbol: Symbol) -> list[IfThenRuntimeSuccess | IfThenRuntimeError]:
        """Run active predicates for 'symbol' with internal runtime accounting.

        See 'runExternal' documentation for differences and compatibility.

        Return value is a list of result descriptiors to execute (or errors or empty if no results found).
        """
        ps: Iterable[PredicateId] = self.symbolPredicates.get(symbol, [])

        # Note: symbolPredicates always has ALL predicates, while we then filter only for the 'active' once for actual checking below.

        # print("Checking:", symbol, ps, self.active)

        # if check is successful, run ALL active predicates for symbol
        results: list[IfThenRuntimeSuccess | IfThenRuntimeError] = []

        totalActivate: set[PredicateId] = set()
        for p in ps:
            if p in self.active and (pred := self.predicates.get(p)):
                try:
                    # print("Checking:", pprint.pformat(pred))
                    match pred.check(symbol):
                        case IfThenRuntimeResultInternal(
                            cmd=cmd, activate=activate, deactivate=deactivate
                        ):
                            # print("Got success to ACTIVATE", activate, "and DEACTIVATE", deactivate)

                            results.append(
                                IfThenRuntimeSuccess(pid=p, cmd=cmd, predicate=pred)
                            )

                            if deactivate:
                                self.active.discard(p)

                                if p in self.once:
                                    self.once.remove(p)
                                    del self.predicates[p]

                            totalActivate |= activate
                except Exception as e:
                    results.append(IfThenRuntimeError(pid=p, err=e))

        # Note: we generate totalActivate *after* all processing because we can't modify 'ps' during iteration
        for a in totalActivate:
            self.activate(a)

        # return if we found anything...
        return results

    def symbolsFor(self, pid: PredicateId) -> Iterable[Symbol]:
        """Return all _active_ symbols represented by the predicate(s) at predicate key.

        Note: does NOT return 'waiting' symbols in a tree (because those are not active).

        May contain result of multiple predicates if predicate id is a peer group with many active predicates.
        """

        if iti := self.predicates.get(pid):
            return iti.symbols

        # If it doesn't exist, just return an empty thing to fail iteration.
        return []

    def extractors(self) -> Iterable[DataExtractor]:
        for predicate in self.predicates.values():
            yield from predicate.extractors()

    def functions(self) -> Iterable[DataFunction]:
        for predicate in self.predicates.values():
            yield from predicate.functions()

    def __getitem__(self, pid: PredicateId) -> CheckableRuntime | None:
        return self.predicates.get(pid)


###############################################################################
#
# Le Hierarchical Data Model Input Parser Population System
#
###############################################################################


@dataclass(slots=True)
class IfThenConfigLoader:
    """Consume a yaml input format representing ifthenRuntime trees and/or peers.

    Consuming a descriptor yaml populates the attached 'ifthenRuntime' with the
    predicates and trees/peers hierarchy from the yaml description.
    """

    ifthenRuntime: IfThenRuntime = field(default_factory=IfThenRuntime)

    def load(
        self, yamltext: str | bytes, activate: bool = True
    ) -> tuple[int, set[Hashable], set[Hashable]]:
        """Load predicates created and described by config yaml into the runtime.

        'activate' is whether to enable all the predicates upon return or, if False, you will activate manually later."""

        body: Final = yaml.load(yamltext, yaml.SafeLoader)

        # Format is just:
        # predicates: [predicate bodies]
        # start: [predicate names]

        # predicate bodies are:
        #   1. - if: "if predicate"
        #   2. - active: [NAME REFERENCE]
        #        waiting: [NAME REFERENCES]
        #   3. - peers: [NAME REFERENCES]

        # Names of predicates are object keys in the yaml tree structure (i.e. there is no '- name: [NAME]' key)

        assert isinstance(body["predicates"], dict), (
            "Predicates must exist and must be a map of names to predicates!"
        )

        predicates: Final = body["predicates"]

        nameToIdMapper: dict[str, PredicateId] = {}

        # because of our data format using top-level named keys instead of lists-of-keys,
        # the yaml parser doesn't preserve the document order.
        # So we need to generate an accurate mapping of which names depend on which other names
        # to effectively process the _used_ names before the _using_ names so IDs already exist.
        keyToChildren: dict[str, set[str]] = defaultdict(set)

        for name, pbody in predicates.items():
            for thing, what in pbody.items():
                # ignore waiting items because waiting items can cause a cycle lookup error for intial load ordering.
                if thing != "waiting":
                    if isinstance(what, Iterable) and not isinstance(what, str):
                        keyToChildren[name] |= set(what)
                    else:
                        keyToChildren[name].add(what)

        processingOrder = tuple(
            graphlib.TopologicalSorter(keyToChildren).static_order()
        )

        # print("Got processing order:", processingOrder)

        def processMapToPredicate(ruleName, body) -> PredicateId:
            currentlyActive: PredicateId | None = None
            currentlyWaiting: list[PredicateId] = []
            currentlyPeers: list[PredicateId] = []

            for bodykey, bodybody in body.items():
                match bodykey.lower():
                    case "if":
                        # current data format requires 'if' keys as stanadlone entities.
                        # We don't support anonymous/self-defining sub-nested if statements (yet)
                        assert currentlyActive is None
                        assert not currentlyWaiting
                        assert not currentlyPeers
                        return self.ifthenRuntime.parse(bodybody)
                    case "active":
                        assert currentlyActive is None
                        currentlyActive = nameToIdMapper[bodybody]
                    case "waiting":
                        assert not currentlyWaiting
                        # if waiting is a string, it is a direct name de-reference
                        if isinstance(bodybody, str):
                            # allow self-referential or cycle looping using key lookups directly (we will post-process them to IDs on final creation)...
                            if isinstance(bodybody, str):
                                currentlyWaiting.append(bodybody)
                            else:
                                currentlyWaiting.append(nameToIdMapper[bodybody])
                        elif isinstance(bodybody, list):
                            for bb in bodybody:
                                if isinstance(bb, str):
                                    currentlyWaiting.append(nameToIdMapper[bb])
                        else:
                            assert None, f"Why did you give us {bodybody=} here?"
                    case "peers":
                        assert not currentlyActive
                        assert not currentlyWaiting
                        assert isinstance(bodybody, list)
                        for bb in bodybody:
                            currentlyPeers.append(nameToIdMapper[bb])

            if currentlyActive and currentlyWaiting:
                return self.ifthenRuntime.tree(currentlyActive, currentlyWaiting)

            if currentlyPeers:
                return self.ifthenRuntime.peers(currentlyPeers)

            raise ValueError("Failed to process successfully?")
            return None

        createdCounter = 0

        checkWaiting = []
        for name in processingOrder:
            # this is optional/conditional because our processingOrder includes some sub-elements which arne't top level keys
            if predicatebody := predicates.get(name):
                # Now, with the object description parsed, we need to determine:
                #  - is a single 'if' predicate?
                #  - is it a tree?
                #  - is it a peer group?
                created = processMapToPredicate(name, predicatebody)
                checkWaiting.append(created)
                nameToIdMapper[name] = created
                createdCounter += 1

        # TODO: after we have CREATED everything here, we need to ITERATE INSIDE _every_ 'tree' to replace any WAITING strings with ID strings of matches.
        # Especially: :self keys for the id of the tree itself, or names of OTHER components which were self-referential.
        # e.g. waiting should not look like this anymore:        waiting=frozenset(['wait-until-reset'])

        for cw in checkWaiting:
            if recheck := self.ifthenRuntime.predicates[cw]:
                if isinstance(recheck, IfThenTree):
                    waiting = recheck.waiting
                    # Lookup other predicates by NAME here and replace their string equivalents
                    # (if name not found, just use existing entry as it was provided)
                    # also, if we are waiting on ourself (recursion forever), then replace the 'self' waiting key with our own id
                    fixed = frozenset(
                        [
                            id(recheck)
                            if w == ":self"
                            else nameToIdMapper.get(str(w), w)
                            for w in waiting
                        ]
                    )
                    recheck.waiting = fixed

        # NOTE: activating here is BEFORE any task-specific predicate setup, so we may need to avoid setup here to manually run it after proper setup later.
        starts = body.get("start")
        if activate:
            if starts:
                assert isinstance(starts, list)
                for start in frozenset(starts):
                    self.ifthenRuntime.activate(nameToIdMapper[start])

        return (
            createdCounter,
            {nameToIdMapper[s] for s in starts},
            set(nameToIdMapper.values()),
        )

    def activate(self, pids):
        for start in frozenset(pids):
            self.ifthenRuntime.activate(start)
