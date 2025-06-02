"""Basic ifthen tests.

Note: these test are mostly checking for "does the grammar parsing crash" and not validating individual results currently.
"""

import pprint

import pytest

from tradeapis.ifthen import (
    Condition,
    ConditionExecution,
    DataCondition,
    DataExtractor,
    IfThen,
    IfThenIntent,
    IfThenConfigLoader,
)

from tradeapis.ifthen_dsl import IfThenDSLLoader

DATASET = dict()


def test_create():
    it = IfThen()
    assert it


def test_if_simple():
    cmd = "if AAPL SMA 5 > 20"

    it = IfThen()

    assert it.parse(cmd)


def test_if_simple_duration():
    cmd = "if (AAPL SMA 5 > 20) for 5 minutes"
    it = IfThen()
    assert it.parse(cmd)
    pprint.pprint(it.parse(cmd))


def test_if_simple_small():
    cmd = "if :21 t <= 35 and I:VIX last > 30: say hello"

    it = IfThen()

    assert it.parse(cmd)


def test_if_simple_algo():
    cmd = "if :21 AAPL.algo.runner.stopped is True: say hello"

    it = IfThen()

    assert it.parse(cmd)


def test_if_simple_algo_string():
    cmd = "if :21 AAPL.algo.runner.stopped is 'Hello': say hello"

    it = IfThen()

    assert it.parse(cmd)
    pprint.pprint(it.parse(cmd))


def test_if_simple_two():
    cmd = "if (AAPL SMA 5 > 20 and MSFT SMA 10 > 20) and (MSFT ema 30 - MSFT ema 10) as done > 3 and NVDA { mid > last } and QZLP { mid > last and high < low }"

    it = IfThen()

    assert it.parse(cmd)
    pprint.pprint(it.parse(cmd))


# Potential ideas:
# e.g. if vix < 19.20 and AAPL > VWAP and TLT < VWAP then sell straddle /ES v p -10 -30
# e.g. if vix at LOD by 1% and delta == 0 then buy straddle
#      when 35s /NQ EMA 5 > 19,500 then buy 5 "BUY 1 /NQ240913C19560000 SELL 1 /NQ240913C19660000" @ live - 20%
#      if 300s /NQ demark 9 setup, buy 10 SMA touch with N point/pct stop.
#      if 300s /NQ demark 9 setup, buy vwap of demark bar 8 on retrace with N pct stop.
#      when bag vwap > 0: buy bag
#      when bag vwap < 0: sell bag
#      if CONTRACT ask < 4.35, create buy-stop-in order for 5 on reversal upward again.
#      if /ES ask >= /ES high, go long
#      if IWM in 204 range and not falling, buy /RTY
#      if time > 1030 and SPY direction up, buy the daily 1030 dip
#      until 4pm: if 300s IWM SMA 5 > SMA 10: buy /RTY then if 15s IWM SMA 5 < SMA 10: sell /RTY
#      when /ES reclaims VWAP: buy with 4 pt stop


def test_if_simple_three():
    cmd = "if (AAPL bid > 20 and ((:21 ask:here:and:there < 7) or MSFT there:and-here:or:me > 20))"

    it = IfThen()

    pprint.pprint(it.parse(cmd))

    assert it.parse(cmd)


def test_if_simple_three_extra_side_unbound():
    cmd = "if (AAPL bid > 20 and ((JPM ask < 7 or 'MSFT' midpoint:and:more 64 > 20) and 'NVDA and friends' ask > 7))"

    it = IfThen()

    pprint.pprint(it.parse(cmd))

    assert it.parse(cmd)


def test_if_simple_three_extra_side():
    cmd = "if (AAPL bid > 20 and (((JPM ask < 7) or ((MSFT midpoint > 20) or (NVDA ask > 7)))))"

    it = IfThen()

    pprint.pprint(it.parse(cmd))

    assert it.parse(cmd)


def test_if_simple_three_extra_side_fewer_parens():
    cmd = "if (AAPL bid > AAPL cost + 7% and ((JPM ask < 7 or MSFT midpoint > 20 and NVDA ask > 7)))"

    it = IfThen()

    pprint.pprint(it.parse(cmd))

    assert it.parse(cmd)

    pprint.pprint(list(it.parse(cmd).extractors()))


def test_if_simple_three_extra_side_fewer_parens_more_parens():
    cmd = "if (vp(AAPL sym, AAPL vwap, -20) as GotResult) exists and (AAPL bid as lolbrunei > verticalPut(AAPL vwap, -20, extraThing(-70)) and ((JPM ask is True or (MSFT theta > 20 and :44 twema 60 > 7)))): HELLO WORLD"

    it = IfThen()

    pprint.pprint(it.parse(cmd))
    pprint.pprint(list(it.parse(cmd).extractors()))
    pprint.pprint(list(it.parse(cmd).functions()))

    assert it.parse(cmd)


def test_if_simple_three_extra_side_fewer_parens_more_parens_WITH_MATH():
    cmd = "if (AAPL bid + 3% > 20 and ((JPM ask - 8% < 7 or (MSFT midpoint > 20 and :44 twema 60 > (0.33 + AAPL bid))))): HELLO WORLD AND HELLO :44.twema.60"

    it = IfThen()

    pprint.pprint(it.parse(cmd))
    pprint.pprint(list(it.parse(cmd).extractors()))

    assert it.parse(cmd)


"""
# Potential format if we allow the syntax to in-line predicates into trees and peers directly instead of
# requiring all predicates to be pre-declared by name up-front.

    - name: OVER
      active: check low
      waiting:
        - check high

    - name: UNDER
      active: check high
      waiting:
        - check low

    check high:
      if: "if AAPL ask > AAPL high: say HIGH HIGH"
      waiting:
        - if: "if hello there > 3"
        - if: "if goodbye again < 2"
"""


def test_yaml_base():
    """Example of consuming arbitrary OTOCO constructions.

    Basically:
      - create individual predicates
      - attach predicates to either:
          - a tree (when single 'active' matches, start checking 'waiting' (if active matches -> schedule waiting))
            - but note, 'active' could also just be a peer group itself running a big OCO group...
          - a peer group (when _any_ 'peer' predicate matches, mark as complete and stop running others)
            - but note, the peer members can be single predicates, other peer groups or a tree with more predicates to schedule when it matches.
      - trees can have peers as 'waiting' members
      - peer groups can have trees as 'active' members which would then also contribute to the next 'waiting' group.

    The "tree" logic is basically OTO and "peers" logic is OCO, so a tree with waiting peers is OTOCO, but they can be
    arbitrarily nested and even reference themseves to create infinte loops of checks
    (e.g. "alert when high, then alert when low, then alert when high again, then alert when low again, repeat forever)

    Why is this a weird external YAML format? Including the extra if-then or if-else logic inside the already existing
    predicate if logic seemed too confising for now, so we are just creating multiple independent predicates and gluing
    them together with references here, then a meta 'IfThenRuntime' consumes the YAML to create the predicates then also
    create the proper tree and peer group structures along with running predicates when requested and generating next
    states when currently active predicates clear their tree or peer group (to then schedule the next 'waiting' batches).

    Also, by allowing this to be an external YAML file format, we can just store predicate YAML files and load them
    externally easily. We can even use template text in a YAML file if we want to create generic/abstract logic for more
    concrete logic or symbol replacement at ingest time.
    """

    predoc = """
predicates:
    first:
        if: "if position(/ES) == 0: buy /ES 1 LIM"

    flip:
        if: "if position(/ES) as Size != 0 and (Size * -2) as Flip: buy /ES Flip LIM

    triggerLong:
        if: "if /ES price:ema:score:ema 0 > 0: say Moving Long"

    triggerShort:
        if: "if /ES price:ema:score:ema 0 < 0: say Moving Short"

    order:
        peers:
            - first
            - flip

    triggers:
        peers:
            - triggerLong
            - triggerShort

    process:
        active: triggers
        waiting:
            - order
            - process
    """

    doc = """
predicates:
    check low:
        if: "if AAPL bid < AAPL low: say LOW LOW"

    check high:
        if: "if AAPL ask > AAPL high: say HIGH HIGH"
        # maybe time bounds too? before: after: notbefore: notafter:

    high or low:
        peers:
            - check low
            - check high

    rotate:
        active: high or low
        waiting: high or low

    OVER:
        active: check low
        waiting:
            - check high

    UNDER:
        active: check high
        waiting:
            - check low

    MINE and me:
        active: check high
        waiting: check low

    GROUPIES:
        peers:
            - check high
            - check low
            - MINE and me
start:
    - OVER
    - UNDER
    - rotate
    - GROUPIES
"""

    icl = IfThenConfigLoader()

    print("Created:", icl.load(doc))

    pprint.pprint(icl.ifthenRuntime)


def test_dsl():
    dsl_text = """
    # Predicate definitions
    check_short = "if /NQZ4 { MNQ.15.algos.temathma-8x-9x-vwap.stopped is True and (MNQ.15.algos.temathma-8x-9x-vwap.be is 'short') }: evict MNQ* -1 0 MKT; cancel MNQ*; say ALGO NQ stopped 15-FAST SHORT"

    check_long = "if /NQZ4 { MNQ.15.algos.temathma-8x-9x-vwap.stopped is True and (MNQ.15.algos.temathma-8x-9x-vwap.be is 'long') }: evict MNQ* -1 0 MKT; cancel MNQ*; say ALGO NQ stopped 15-FAST LONG"

    unstopped = "if /NQZ4 { MNQ.15.algos.temathma-8x-9x-vwap.stopped is False }: say algo reset"

    # Flow definitions
    flow run_trigger:
        check_short | check_long

    # in our DSL here, '@' just means "self-recurse" forever, so it restarts the flow again.
    flow rotate:
        run_trigger -> unstopped -> (check_short | check_long) -> unstopped -> @

    # Begin here!
    # start statements can have one or more flow or predicate labels. multiple can be space or comma or pipe separated. They are all activated concurrently at start time.
    start: rotate, run_trigger
    """

    # Create runtime and loader
    loader = IfThenDSLLoader()

    # Load DSL
    created_count, start_ids = loader.load(dsl_text)
    pprint.pprint(loader.ifthenRuntime)
    print(f"Created {created_count} predicates/flows")
    print(f"Started: {start_ids}")
