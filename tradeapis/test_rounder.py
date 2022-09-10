import tradeapis.rounder as rounder


# SPX
def test_SPX_big_up():
    assert rounder.round("SPX", 33.33) == 33.35


def test_SPX_small_up():
    assert rounder.round("SPX", 1.22) == 1.25


def test_SPX_big_down():
    assert rounder.round("SPX", 33.33, up=False) == 33.30


def test_SPX_small_down():
    assert rounder.round("SPX", 1.22, up=False) == 1.20


def test_SPX_edge():
    assert rounder.round("SPX", 3.00) == 3.00


# SPXW
def test_SPXW_big_up():
    assert rounder.round("SPXW", 33.33) == 33.35


def test_SPXW_small_up():
    assert rounder.round("SPXW", 1.22) == 1.25


def test_SPXW_big_down():
    assert rounder.round("SPXW", 33.33, up=False) == 33.30


def test_SPXW_small_down():
    assert rounder.round("SPXW", 1.22, up=False) == 1.20


def test_SPXW_edge():
    assert rounder.round("SPXW", 3.00) == 3.00


# ES
def test_ES_big_up():
    assert rounder.round("/ES", 33.33) == 33.50


def test_ES_small_up():
    assert rounder.round("/ES", 1.22) == 1.25


def test_ES_big_down():
    assert rounder.round("/ES", 33.33, up=False) == 33.25


def test_ES_small_down():
    assert rounder.round("/ES", 1.22, up=False) == 1.00


def test_ES_edge():
    assert rounder.round("/ES", 3.00) == 3.00
