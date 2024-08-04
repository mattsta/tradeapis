import tradeapis.rounder as rounder
from decimal import Decimal


# SPX
def test_SPX_big_up():
    # "$0.10 increments >= $3"
    assert rounder.round("SPX", 33.33) == Decimal(str(33.40))


def test_SPX_small_up():
    # "$0.05 increments under $3"
    assert rounder.round("SPX", 1.22) == Decimal(str(1.25))


def test_SPX_big_down():
    assert rounder.round("SPX", 33.33, up=False) == Decimal(str(33.30))


def test_SPX_small_down():
    assert rounder.round("SPX", 1.22, up=False) == Decimal(str(1.20))


def test_SPX_edge():
    assert rounder.round("SPX", 3.00) == Decimal(str(3.00))


# SPXW
def test_SPXW_big_up():
    assert rounder.round("SPXW", 33.33) == Decimal(str(33.40))


def test_SPXW_small_up():
    assert rounder.round("SPXW", 1.22) == Decimal(str(1.25))


def test_SPXW_big_down():
    assert rounder.round("SPXW", 33.33, up=False) == Decimal(str(33.30))


def test_SPXW_small_down():
    assert rounder.round("SPXW", 1.22, up=False) == Decimal(str(1.20))


def test_SPXW_edge():
    assert rounder.round("SPXW", 3.00) == Decimal(str(3.00))


# ES
def test_ES_big_up():
    assert rounder.round("/ES", 33.33) == Decimal(str(33.50))


def test_ES_small_up():
    assert rounder.round("/ES", 1.22) == Decimal(str(1.25))


def test_ES_big_down():
    assert rounder.round("/ES", 33.33, up=False) == Decimal(str(33.25))


def test_ES_small_down():
    assert rounder.round("/ES", 1.22, up=False) == Decimal(str(1.00))


def test_ES_edge():
    assert rounder.round("/ES", 3.00) == Decimal(str(3.00))
