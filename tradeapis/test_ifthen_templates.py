"""Tests for IfThen Template System."""

import pytest

from tradeapis.ifthen import IfThenRuntime
from tradeapis.ifthen_templates import (
    BUILTIN_TEMPLATES,
    IfThenMultiTemplateManager,
    IfThenRuntimeTemplateExecutor,
    _template_cache,
    create_template_args_for_algo_flipper,
)


def test_builtin_template_executor():
    """Test creating executor with built-in template."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Should load built-in template successfully
    assert executor._template_content == BUILTIN_TEMPLATES["algo_flipper.dsl"]

    # Should extract template variables
    expected_vars = {
        "algo_symbol",
        "watch_symbol",
        "trade_symbol",
        "evict_symbol",
        "timeframe",
        "algo",
        "qty",
        "offset",
        "profit_pts",
        "loss_pts",
        "flip_sides",
    }
    assert executor.get_template_variables() == expected_vars


def test_template_argument_validation():
    """Test template argument validation."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Test missing variables
    incomplete_args = {"algo_symbol": "MNQ", "timeframe": 35}
    is_valid, missing = executor.validate_template_args(incomplete_args)
    assert not is_valid
    assert "algo" in missing
    assert "watch_symbol" in missing
    assert "trade_symbol" in missing
    assert "evict_symbol" in missing

    # Test complete variables
    complete_args = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=1,
        offset=0.50,
    )
    is_valid, missing = executor.validate_template_args(complete_args)
    assert is_valid
    assert len(missing) == 0


def test_populate_template():
    """Test template population with valid arguments."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Create template arguments
    args = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=1,
        offset=0.50,
    )

    # Populate template
    dsl_text = executor.populate_template(args)

    # Verify substitutions occurred
    assert "MNQ" in dsl_text  # algo_symbol
    assert "/NQM5" in dsl_text  # watch_symbol
    assert "/MNQ" in dsl_text  # trade_symbol
    assert "cancel MNQ*" in dsl_text  # evict_symbol
    assert "35" in dsl_text
    assert "temathma-5x-12x-vwap" in dsl_text
    assert "check_short" in dsl_text
    assert "check_long" in dsl_text
    assert "flow primary:" in dsl_text
    assert "start: entrypoint" in dsl_text

    # Verify no Jinja2 template variables remain
    assert "{{" not in dsl_text
    assert "{%" not in dsl_text


def test_offset_handling():
    """Test that offset=None generates 'live' without offset."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Test with offset=None (should use 'live' directly)
    args_no_offset = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=1,
        offset=None,
    )

    dsl_no_offset = executor.populate_template(args_no_offset)
    assert "LIM @ live" in dsl_no_offset  # No offset parentheses, no stop terms
    assert "(+ live" not in dsl_no_offset
    assert "(- live" not in dsl_no_offset
    assert "+ 4" not in dsl_no_offset  # No profit_pts
    assert "- 7" not in dsl_no_offset  # No loss_pts

    # Test with offset and stops (should use all terms)
    args_with_all = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=1,
        offset=0.50,
        profit_pts=4,
        loss_pts=7,
    )

    dsl_with_all = executor.populate_template(args_with_all)
    assert "LIM @ (+ live 0.5) + 4 - 7" in dsl_with_all  # Short position
    assert "LIM @ (- live 0.5) + 4 - 7" in dsl_with_all  # Long position

    # Test with only offset (no stops)
    args_offset_only = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=1,
        offset=0.25,
    )

    dsl_offset_only = executor.populate_template(args_offset_only)
    assert "LIM @ (+ live 0.25)" in dsl_offset_only  # Short position, no stops
    assert "LIM @ (- live 0.25)" in dsl_offset_only  # Long position, no stops
    assert "+ 4" not in dsl_offset_only
    assert "- 7" not in dsl_offset_only


def test_populate_template_missing_args():
    """Test template population with missing arguments."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Missing required arguments
    incomplete_args = {"symbol": "MNQ"}

    with pytest.raises(ValueError, match="Missing required template variables"):
        executor.populate_template(incomplete_args)


def test_convenience_functions():
    """Test convenience functions."""
    runtime = IfThenRuntime()

    # Test explicit constructor
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")
    assert executor.template_source_id == "builtin:algo_flipper.dsl"
    assert executor.ifthen_runtime is runtime

    # Test create_template_args_for_algo_flipper - all params required
    args = create_template_args_for_algo_flipper(
        algo_symbol="ES",
        watch_symbol="/ESM5",
        trade_symbol="/ES",
        evict_symbol="ES",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=1,
        offset=0.50,
    )
    expected = {
        "algo_symbol": "ES",
        "watch_symbol": "/ESM5",
        "trade_symbol": "/ES",
        "evict_symbol": "ES",
        "timeframe": 35,
        "algo": "temathma-5x-12x-vwap",
        "qty": 1,
        "offset": 0.50,
        "profit_pts": None,
        "loss_pts": None,
        "flip_sides": "both",
    }
    assert args == expected

    # Test with custom values
    args = create_template_args_for_algo_flipper(
        algo_symbol="NQ",
        watch_symbol="/NQZ4",
        trade_symbol="/NQZ5",
        evict_symbol="NQ",
        timeframe=60,
        algo="custom-algo",
        qty=2,
        offset=0.75,
    )
    assert args["algo_symbol"] == "NQ"
    assert args["watch_symbol"] == "/NQZ4"
    assert args["trade_symbol"] == "/NQZ5"
    assert args["evict_symbol"] == "NQ"
    assert args["timeframe"] == 60
    assert args["flip_sides"] == "both"

    # Test flip_sides parameter
    args_long = create_template_args_for_algo_flipper(
        algo_symbol="ES",
        watch_symbol="/ESM5",
        trade_symbol="/ES",
        evict_symbol="ES",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=1,
        flip_sides="long",
    )
    assert args_long["flip_sides"] == "long"

    args_short = create_template_args_for_algo_flipper(
        algo_symbol="ES",
        watch_symbol="/ESM5",
        trade_symbol="/ES",
        evict_symbol="ES",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=1,
        flip_sides="short",
    )
    assert args_short["flip_sides"] == "short"


def test_algo_flipper_flip_sides_logic():
    """Test that flip_sides parameter correctly controls position reopening logic."""
    runtime = IfThenRuntime()

    # Test 'both' - both sides should reopen positions
    args_both = create_template_args_for_algo_flipper(
        algo_symbol="ES",
        watch_symbol="/ESM5",
        trade_symbol="/ES",
        evict_symbol="ES",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=1,
        flip_sides="both",
    )
    executor_both = IfThenRuntimeTemplateExecutor.from_builtin(
        runtime, "algo_flipper.dsl"
    )
    dsl_both = executor_both.populate_template(args_both)

    # Both checks should exist
    assert "check_short" in dsl_both
    assert "check_long" in dsl_both
    # Both should include buy orders after eviction
    assert "buy /ES -1 LIM" in dsl_both  # short action
    assert "buy /ES 1 LIM" in dsl_both  # long action

    # Test 'long' - only long should reopen position
    args_long = create_template_args_for_algo_flipper(
        algo_symbol="ES",
        watch_symbol="/ESM5",
        trade_symbol="/ES",
        evict_symbol="ES",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=1,
        flip_sides="long",
    )
    executor_long = IfThenRuntimeTemplateExecutor.from_builtin(
        runtime, "algo_flipper.dsl"
    )
    dsl_long = executor_long.populate_template(args_long)

    # Both checks should exist (for exits)
    assert "check_short" in dsl_long
    assert "check_long" in dsl_long
    # Only long should include buy order after eviction
    assert "buy /ES -1 LIM" not in dsl_long  # short action should only evict
    assert "buy /ES 1 LIM" in dsl_long  # long action should include buy

    # Test 'short' - only short should reopen position
    args_short = create_template_args_for_algo_flipper(
        algo_symbol="ES",
        watch_symbol="/ESM5",
        trade_symbol="/ES",
        evict_symbol="ES",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=1,
        flip_sides="short",
    )
    executor_short = IfThenRuntimeTemplateExecutor.from_builtin(
        runtime, "algo_flipper.dsl"
    )
    dsl_short = executor_short.populate_template(args_short)

    # Both checks should exist (for exits)
    assert "check_short" in dsl_short
    assert "check_long" in dsl_short
    # Only short should include buy order after eviction
    assert "buy /ES -1 LIM" in dsl_short  # short action should include buy
    assert "buy /ES 1 LIM" not in dsl_short  # long action should only evict


def test_other_builtin_templates():
    """Test other built-in templates work."""
    runtime = IfThenRuntime()

    # Test simple flipper
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "simple_flipper.dsl")
    args = {
        "symbol": "AAPL",
        "buy_condition": "AAPL price < 150",
        "sell_condition": "AAPL price > 160",
        "qty": 100,
        "action_buy": "buy AAPL 100 MKT",
        "action_sell": "sell AAPL 100 MKT",
    }
    dsl_text = executor.populate_template(args)
    assert "buy_signal" in dsl_text
    assert "sell_signal" in dsl_text

    # Test breakout monitor
    executor = IfThenRuntimeTemplateExecutor.from_builtin(
        runtime, "breakout_monitor.dsl"
    )
    args = {
        "symbol": "SPY",
        "high_level": "450",
        "low_level": "440",
        "timeframe": "60",
        "qty": "10",
    }
    dsl_text = executor.populate_template(args)
    assert "breakout_high" in dsl_text
    assert "breakout_low" in dsl_text


def test_template_caching():
    """Test that template compilation is cached properly."""
    runtime = IfThenRuntime()

    # Clear cache to start fresh
    _template_cache.clear()

    # Create first executor
    executor1 = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Create second executor with same template
    executor2 = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Both should use the same cached compiled template
    assert executor1._compiled_template is executor2._compiled_template


def test_jinja2_advanced_features():
    """Test Jinja2 advanced features like loops and conditionals."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(
        runtime, "multi_symbol_flipper.dsl"
    )

    # Test multi-symbol template with complex data
    from tradeapis.ifthen_templates import create_symbol_config_for_flipper

    args = {
        "symbols": [
            create_symbol_config_for_flipper(
                "MNQ", "/MNQM5", "/MNQ", "MNQ", timeframe=35, qty=1
            ),
            create_symbol_config_for_flipper(
                "ES", "/ESZ4", "/ESZ5", "ES", timeframe=60, qty=2
            ),
            create_symbol_config_for_flipper("YM", "/YMM5", "/YM", "YM", qty=1),
        ],
        "base_timeframe": 30,
        "base_algo": "temathma-5x-12x-vwap",
        "default_qty": 1,
    }

    dsl_text = executor.populate_template(args)

    # Verify loop generated content for all symbols
    assert "MNQ_check_short" in dsl_text
    assert "ES_check_long" in dsl_text
    assert "YM_unstopped" in dsl_text

    # Verify conditional logic worked (ES should use custom symbols)
    assert "/ESZ4" in dsl_text  # Custom watch symbol
    assert "/ESZ5" in dsl_text  # Custom trade symbol
    assert "/MNQ" in dsl_text  # Default trade symbol

    # Verify start combines all entrypoints
    assert "start: MNQ_entrypoint, ES_entrypoint, YM_entrypoint" in dsl_text


def test_activation_management():
    """Test template activation, deactivation, and tracking."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Test pure activate() method with proper DSL format
    dsl_text = """
    test_predicate = "if AAPL price > 100: say HELLO"

    start: test_predicate
    """

    # Should start with no active templates
    assert executor.list_active() == []
    assert executor.get_active_summary() == {}

    # Activate a template
    created_count, start_ids, all_ids = executor.activate("test_instance", dsl_text)

    # Should now have one active template
    assert executor.list_active() == ["test_instance"]
    assert created_count > 0

    # Get info about the active template
    info = executor.get_active_info("test_instance")
    assert info is not None
    assert info.name == "test_instance"
    assert info.dsl_text == dsl_text
    assert info.created_count == created_count
    assert info.start_ids == start_ids
    assert info.all_ids == all_ids

    # Test duplicate name rejection
    with pytest.raises(ValueError, match="already active"):
        executor.activate("test_instance", dsl_text)

    # Test summary
    summary = executor.get_active_summary()
    assert "test_instance" in summary
    assert summary["test_instance"].created_count == created_count
    assert summary["test_instance"].predicate_count == len(all_ids)

    # Deactivate the template
    assert executor.deactivate("test_instance") is True
    assert executor.list_active() == []

    # Deactivating non-existent template should return False
    assert executor.deactivate("nonexistent") is False


def test_create_and_activate_with_name():
    """Test create_and_activate with the new name parameter."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Create template args
    args = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=1,
        profit_pts=4,
        loss_pts=7,
    )

    # Test create_and_activate with name
    created_count, start_ids, all_ids = executor.create_and_activate(
        "mnq_flipper", args
    )

    # Should be tracked
    assert "mnq_flipper" in executor.list_active()

    info = executor.get_active_info("mnq_flipper")
    assert info is not None
    assert info.name == "mnq_flipper"
    assert "MNQ.35.algos.temathma-5x-12x-vwap" in info.dsl_text

    # Clean up
    executor.deactivate("mnq_flipper")


def test_deactivate_all():
    """Test deactivating all templates at once."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Activate multiple templates
    dsl1 = """
    pred1 = "if AAPL price > 100: say AAPL HIGH"
    start: pred1
    """
    dsl2 = """
    pred2 = "if MSFT price > 200: say MSFT HIGH"
    start: pred2
    """
    dsl3 = """
    pred3 = "if GOOGL price > 150: say GOOGL HIGH"
    start: pred3
    """

    executor.activate("test1", dsl1)
    executor.activate("test2", dsl2)
    executor.activate("test3", dsl3)

    assert len(executor.list_active()) == 3

    # Deactivate all
    deactivated_count = executor.deactivate_all()
    assert deactivated_count == 3
    assert len(executor.list_active()) == 0

    # Deactivating when none active should return 0
    assert executor.deactivate_all() == 0


def test_integration_example():
    """Test a complete integration example."""
    # This would be the typical usage pattern
    runtime = IfThenRuntime()

    # Create template executor
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Create MNQ flipper
    mnq_args = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=1,
        offset=0.50,
    )

    # Generate DSL
    dsl_text = executor.populate_template(mnq_args)

    # Verify DSL is valid by checking key components
    assert "MNQ.35.algos.temathma-5x-12x-vwap.stopped" in dsl_text
    assert "buy /MNQ -1 LIM" in dsl_text  # short position
    assert "buy /MNQ 1 LIM" in dsl_text  # long position
    assert "cancel MNQ*" in dsl_text  # evict symbol
    assert "evict MNQ*" in dsl_text  # evict symbol
    assert "+ live 0.5" in dsl_text and "- live 0.5" in dsl_text
    assert "start: entrypoint" in dsl_text

    # The new way with activation management:
    created_count, start_ids, all_ids = executor.create_and_activate(
        "mnq_live", mnq_args
    )

    # Verify it's tracked
    assert "mnq_live" in executor.list_active()

    # Clean up
    executor.deactivate("mnq_live")


def test_performance_tracking_basic():
    """Test basic performance tracking functionality."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Test state_update with valid data
    executor.state_update("algo1", {"profit": 100.0, "win": True})
    executor.state_update("algo1", {"profit": -50.0, "win": False})
    executor.state_update("algo1", {"profit": 75.0, "win": True})

    # Test score_get returns events in chronological order
    events = executor.score_get("algo1")
    assert len(events) == 3
    assert events[0]["profit"] == 100.0
    assert events[0]["win"] is True
    assert events[1]["profit"] == -50.0
    assert events[1]["win"] is False
    assert events[2]["profit"] == 75.0
    assert events[2]["win"] is True

    # All events should have ts and metadata fields
    for event in events:
        assert "ts" in event
        assert "profit" in event
        assert "win" in event


def test_performance_tracking_summary():
    """Test performance summary calculation."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Add events for testing
    executor.state_update("test_algo", {"profit": 100.0, "win": True})
    executor.state_update("test_algo", {"profit": 200.0, "win": True})
    executor.state_update("test_algo", {"profit": -150.0, "win": False})
    executor.state_update("test_algo", {"profit": 50.0, "win": True})
    executor.state_update("test_algo", {"profit": -75.0, "win": False})

    # Test score_summary
    summary = executor.score_summary("test_algo")

    assert summary.name == "test_algo"
    assert summary.total_events == 5
    assert summary.total_profit == 125.0  # 100 + 200 - 150 + 50 - 75
    assert summary.win_count == 3
    assert summary.loss_count == 2
    assert summary.win_rate == 0.6  # 3/5
    assert summary.avg_profit_per_trade == 25.0  # 125/5
    assert summary.current_streak == 1  # Last was a loss, so loss streak = 1
    assert summary.streak_type == "loss"
    assert summary.first_event is not None
    assert summary.last_event is not None
    assert summary.runtime_seconds >= 0


def test_performance_tracking_streak_calculation():
    """Test streak calculation logic."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Test win streak
    executor.state_update("streak_test", {"profit": 10.0, "win": True})
    executor.state_update("streak_test", {"profit": 20.0, "win": True})
    executor.state_update("streak_test", {"profit": 15.0, "win": True})

    summary = executor.score_summary("streak_test")
    assert summary.current_streak == 3
    assert summary.streak_type == "win"

    # Add a loss to break the streak
    executor.state_update("streak_test", {"profit": -25.0, "win": False})

    summary = executor.score_summary("streak_test")
    assert summary.current_streak == 1
    assert summary.streak_type == "loss"

    # Add more losses to extend loss streak
    executor.state_update("streak_test", {"profit": -10.0, "win": False})
    executor.state_update("streak_test", {"profit": -5.0, "win": False})

    summary = executor.score_summary("streak_test")
    assert summary.current_streak == 3
    assert summary.streak_type == "loss"
    assert summary.win_count == 3
    assert summary.loss_count == 3


def test_performance_tracking_with_metadata():
    """Test performance tracking with custom metadata."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Add event with custom metadata
    metadata = {
        "symbol": "MNQ",
        "timeframe": 35,
        "entry_price": 19500.0,
        "exit_price": 19525.0,
        "hold_time": 120,  # seconds
    }
    executor.state_update(
        "meta_test", {"profit": 25.0, "win": True, "metadata": metadata}
    )

    # Retrieve and verify metadata is preserved
    events = executor.score_get("meta_test")
    assert len(events) == 1
    assert events[0]["metadata"]["symbol"] == "MNQ"
    assert events[0]["metadata"]["timeframe"] == 35
    assert events[0]["metadata"]["entry_price"] == 19500.0
    assert events[0]["metadata"]["exit_price"] == 19525.0
    assert events[0]["metadata"]["hold_time"] == 120


def test_performance_tracking_nonexistent_algo():
    """Test performance tracking for non-existent algorithms."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Test score_get for nonexistent algo
    events = executor.score_get("nonexistent")
    assert events == []

    # Test score_summary for nonexistent algo
    summary = executor.score_summary("nonexistent")
    assert summary.name == "nonexistent"
    assert summary.total_events == 0
    assert summary.total_profit == 0.0
    assert summary.win_count == 0
    assert summary.loss_count == 0
    assert summary.win_rate == 0.0
    assert summary.current_streak == 0
    assert summary.streak_type == "none"
    assert summary.first_event is None
    assert summary.last_event is None
    assert summary.runtime_seconds == 0.0


def test_performance_tracking_edge_cases():
    """Test edge cases in performance tracking."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Test with zero profit
    executor.state_update("edge_test", {"profit": 0.0, "win": True})

    events = executor.score_get("edge_test")
    assert len(events) == 1
    assert events[0]["profit"] == 0.0
    assert events[0]["win"] is True

    summary = executor.score_summary("edge_test")
    assert summary.total_profit == 0.0
    assert summary.win_count == 1
    assert summary.avg_profit_per_trade == 0.0

    # Test with large profit/loss values
    executor.state_update("edge_test", {"profit": 999999.99, "win": True})
    executor.state_update("edge_test", {"profit": -999999.99, "win": False})

    summary = executor.score_summary("edge_test")
    assert summary.total_events == 3
    assert summary.total_profit == 0.0  # 0 + 999999.99 - 999999.99


def test_performance_tracking_state_update_validation():
    """Test state_update input validation."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Test missing required fields
    with pytest.raises(ValueError, match="profit"):
        executor.state_update("test", {"win": True})

    with pytest.raises(ValueError, match="win"):
        executor.state_update("test", {"profit": 100.0})

    # Test invalid data types
    with pytest.raises(ValueError, match="must be a number"):
        executor.state_update("test", {"profit": "not_a_number", "win": True})

    with pytest.raises(ValueError, match="must be a boolean"):
        executor.state_update("test", {"profit": 100.0, "win": "not_a_boolean"})


def test_performance_tracking_multiple_algos():
    """Test performance tracking for multiple algorithms independently."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Add events for different algorithms
    executor.state_update("algo_a", {"profit": 100.0, "win": True})
    executor.state_update("algo_b", {"profit": -50.0, "win": False})
    executor.state_update("algo_a", {"profit": 75.0, "win": True})
    executor.state_update("algo_c", {"profit": 200.0, "win": True})
    executor.state_update("algo_b", {"profit": 25.0, "win": True})

    # Verify each algorithm's events are tracked separately
    events_a = executor.score_get("algo_a")
    events_b = executor.score_get("algo_b")
    events_c = executor.score_get("algo_c")

    assert len(events_a) == 2
    assert len(events_b) == 2
    assert len(events_c) == 1

    # Verify summaries are independent
    summary_a = executor.score_summary("algo_a")
    summary_b = executor.score_summary("algo_b")
    summary_c = executor.score_summary("algo_c")

    assert summary_a.total_profit == 175.0
    assert summary_a.win_count == 2
    assert summary_a.current_streak == 2
    assert summary_a.streak_type == "win"

    assert summary_b.total_profit == -25.0
    assert summary_b.win_count == 1
    assert summary_b.loss_count == 1
    assert summary_b.current_streak == 1
    assert summary_b.streak_type == "win"  # Last event was a win

    assert summary_c.total_profit == 200.0
    assert summary_c.win_count == 1
    assert summary_c.loss_count == 0


def test_explicit_constructors():
    """Test all three explicit constructor methods."""
    runtime = IfThenRuntime()

    # Test from_builtin
    executor_builtin = IfThenRuntimeTemplateExecutor.from_builtin(
        runtime, "algo_flipper.dsl"
    )
    assert executor_builtin.template_source_id == "builtin:algo_flipper.dsl"
    assert "algo_symbol" in executor_builtin.get_template_variables()

    # Test from_string
    custom_template = """
    test_pred = "if {{symbol}} price > {{threshold}}: say {{message}}"
    start: test_pred
    """
    executor_string = IfThenRuntimeTemplateExecutor.from_string(
        runtime, custom_template
    )
    assert executor_string.template_source_id.startswith("string:")
    expected_vars = {"symbol", "threshold", "message"}
    assert executor_string.get_template_variables() == expected_vars

    # Test that string executor can populate and activate
    args = {"symbol": "AAPL", "threshold": 100, "message": "HIGH"}
    dsl_text = executor_string.populate_template(args)
    assert "AAPL" in dsl_text
    assert "100" in dsl_text
    assert "HIGH" in dsl_text

    # Test validation on string template
    is_valid, missing = executor_string.validate_template_args({"symbol": "AAPL"})
    assert not is_valid
    assert "threshold" in missing and "message" in missing

    # Test error handling for invalid built-in template name
    with pytest.raises(ValueError, match="Built-in template.*not found"):
        IfThenRuntimeTemplateExecutor.from_builtin(runtime, "nonexistent_template.dsl")

    # Test error handling for empty string template
    with pytest.raises(ValueError, match="Template content cannot be empty"):
        IfThenRuntimeTemplateExecutor.from_string(runtime, "")


def test_performance_integration_with_activation():
    """Test performance tracking integrated with template activation."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Create and activate a template
    args = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=1,
        offset=0.50,
    )

    created_count, start_ids, all_ids = executor.create_and_activate("perf_test", args)

    # Verify template is active
    assert "perf_test" in executor.list_active()

    # Add performance data for the active template
    executor.state_update("perf_test", {"profit": 150.0, "win": True})
    executor.state_update("perf_test", {"profit": -75.0, "win": False})

    # Verify performance data is tracked
    events = executor.score_get("perf_test")
    assert len(events) == 2

    summary = executor.score_summary("perf_test")
    assert summary.name == "perf_test"
    assert summary.total_profit == 75.0
    assert summary.win_count == 1
    assert summary.loss_count == 1

    # Deactivate template
    executor.deactivate("perf_test")

    # Performance data should still be available after deactivation
    events_after = executor.score_get("perf_test")
    assert len(events_after) == 2
    assert events_after == events


if __name__ == "__main__":
    # Example usage demonstration
    print("=== IfThen Template System with Jinja2 Demo ===")

    runtime = IfThenRuntime()

    # Method 1: Direct usage
    print("\n1. Direct Template Usage:")
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")
    print(f"Template variables: {executor.get_template_variables()}")

    args = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/MNQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="temathma-5x-12x-vwap",
        qty=2,
        offset=0.75,
    )
    dsl_text = executor.populate_template(args)
    print(f"Generated DSL length: {len(dsl_text)} characters")

    # Method 2: Multi-symbol advanced template
    print("\n2. Multi-Symbol Template with Jinja2 Features:")
    multi_executor = IfThenRuntimeTemplateExecutor.from_builtin(
        runtime, "multi_symbol_flipper.dsl"
    )

    from tradeapis.ifthen_templates import create_symbol_config_for_flipper

    multi_args = {
        "symbols": [
            create_symbol_config_for_flipper(
                "MNQ", "/MNQM5", "/MNQ", "MNQ", timeframe=35, qty=1
            ),
            create_symbol_config_for_flipper(
                "ES", "/ESZ4", "/ESZ5", "ES", timeframe=60, qty=2
            ),
            create_symbol_config_for_flipper("YM", "/YMM5", "/YM", "YM"),
        ],
        "base_timeframe": 30,
        "base_algo": "temathma-5x-12x-vwap",
        "default_qty": 1,
    }

    multi_dsl = multi_executor.populate_template(multi_args)
    print(f"Generated multi-symbol DSL length: {len(multi_dsl)} characters")
    print("Symbol configurations created:", [s["name"] for s in multi_args["symbols"]])

    # Method 3: Caching demonstration
    print("\n3. Template Caching:")
    executor3 = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")
    print(f"Cache hit: {executor3._compiled_template is executor._compiled_template}")

    # Method 4: Performance tracking demonstration
    print("\n4. Performance Tracking:")
    perf_executor = IfThenRuntimeTemplateExecutor.from_builtin(
        IfThenRuntime(), "algo_flipper.dsl"
    )

    # Simulate some trading events
    perf_executor.state_update("demo_algo", {"profit": 150.0, "win": True})
    perf_executor.state_update("demo_algo", {"profit": -75.0, "win": False})
    perf_executor.state_update("demo_algo", {"profit": 200.0, "win": True})

    events = perf_executor.score_get("demo_algo")
    summary = perf_executor.score_summary("demo_algo")

    print(f"Total events: {len(events)}")
    print(f"Total profit: ${summary.total_profit}")
    print(f"Win rate: {summary.win_rate:.1%}")
    print(f"Current {summary.streak_type} streak: {summary.current_streak}")

    print("\n=== Demo Complete ===")


# ==================== MULTI-TEMPLATE MANAGER TESTS ====================


def setup_manager_with_algo_flipper(runtime):
    """Helper to create manager with algo_flipper template for tests."""
    manager = IfThenMultiTemplateManager(runtime)
    manager.from_builtin("algo_flipper.dsl", "algo_flipper")
    return manager


def test_multi_template_manager_basic():
    """Test basic IfThenMultiTemplateManager functionality."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Test explicit creation of builtin templates
    executor = manager.from_builtin("algo_flipper.dsl", "algo_flipper")
    assert isinstance(executor, IfThenRuntimeTemplateExecutor)
    assert executor.template_source_id == "builtin:algo_flipper.dsl"

    # Test that calling from_builtin again returns the same instance
    executor_again = manager.from_builtin("algo_flipper.dsl", "algo_flipper")
    assert executor is executor_again

    # Test dictionary access after explicit creation
    executor2 = manager["algo_flipper"]
    assert executor is executor2

    # Test __contains__ after explicit creation
    assert "algo_flipper" in manager
    assert "nonexistent_template" not in manager

    # Test accessing non-existent template fails
    with pytest.raises(ValueError, match="not found. You must explicitly create"):
        manager["simple_flipper"]


def test_multi_template_manager_from_file_and_string():
    """Test creating templates from file and string sources."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Test from_string
    custom_template = """
    entry = "if {{symbol}} price > {{threshold}}: buy {{symbol}} {{qty}} MKT"
    exit = "if {{symbol}} price < {{exit_threshold}}: sell {{symbol}} {{qty}} MKT"

    flow trading:
        entry -> exit -> @

    start: trading
    """

    executor = manager.from_string(custom_template, "custom_breakout")
    assert "custom_breakout" in manager.get_template_names()
    assert executor.template_source_id.startswith("string:")

    expected_vars = {"symbol", "threshold", "qty", "exit_threshold"}
    assert executor.get_template_variables() == expected_vars

    # Test that calling from_string again with same content returns existing instance
    executor_again = manager.from_string(custom_template, "custom_breakout")
    assert executor is executor_again  # Should return existing, not create new

    # Test that calling from_string with different content but same name raises error
    with pytest.raises(
        ValueError, match="Template name 'custom_breakout' already exists"
    ):
        manager.from_string("different content", "custom_breakout")


def test_multi_template_manager_template_info():
    """Test template information and introspection methods."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Add a custom template
    manager.from_string(
        'test_template = "if {{symbol}} price > {{threshold}}: say {{message}}"',
        "test_template",
    )

    # Test get_template_names
    names = manager.get_template_names()
    assert "test_template" in names

    # Test get_available_builtin_names
    builtin_names = manager.get_available_builtin_names()
    assert "algo_flipper" in builtin_names
    assert "simple_flipper" in builtin_names
    assert "breakout_monitor" in builtin_names

    # Test get_template_info
    info = manager.get_template_info("test_template")
    assert info is not None
    assert info.template_name == "test_template"
    assert info.source_id.startswith("string:")
    expected_test_vars = {"symbol", "threshold", "message"}
    assert set(info.template_variables) == expected_test_vars
    assert info.active_count == 0

    # Test non-existent template
    assert manager.get_template_info("nonexistent") is None


def test_multi_template_manager_activation():
    """Test template activation with hierarchical naming."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # First explicitly create the template
    manager.from_builtin("algo_flipper.dsl", "algo_flipper")

    # Create template args
    args = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="test-algo",
        qty=1,
    )

    # Test activation
    created_count, start_ids, all_ids = manager.activate(
        "algo_flipper", "mnq_fast", args
    )
    assert created_count > 0
    assert len(start_ids) > 0
    assert len(all_ids) > 0

    # Test listing active instances
    active = manager.list_all_active()
    assert "algo_flipper.mnq_fast" in active

    # Test listing for specific template
    template_active = manager.list_active_for_template("algo_flipper")
    assert "algo_flipper.mnq_fast" in template_active

    # Test activation of second instance
    args2 = create_template_args_for_algo_flipper(
        algo_symbol="ES",
        watch_symbol="/ESZ4",
        trade_symbol="/ES",
        evict_symbol="ES",
        timeframe=60,
        algo="test-algo",
        qty=2,
    )

    manager.activate("algo_flipper", "es_slow", args2)

    # Should now have two active instances
    active = manager.list_all_active()
    assert len(active) == 2
    assert "algo_flipper.mnq_fast" in active
    assert "algo_flipper.es_slow" in active


def test_multi_template_manager_deactivation():
    """Test various deactivation methods."""
    runtime = IfThenRuntime()
    manager = setup_manager_with_algo_flipper(runtime)

    # Activate some test instances
    test_dsl = """
    test_pred = "if AAPL price > 100: say TEST"
    start: test_pred
    """

    manager["algo_flipper"].activate("algo_flipper.test1", test_dsl)
    manager["algo_flipper"].activate("algo_flipper.test2", test_dsl)
    manager.from_string(
        'simple = "if {{symbol}} price > {{threshold}}: say {{message}}"', "simple"
    )
    manager["simple"].activate(
        "simple.test3", 'simple = "if AAPL price > 100: say HELLO"\nstart: simple'
    )

    # Should have 3 active instances
    assert len(manager.list_all_active()) == 3

    # Test deactivating specific instance
    result = manager.deactivate("algo_flipper", "test1")
    assert result is True
    assert len(manager.list_all_active()) == 2
    assert "algo_flipper.test1" not in manager.list_all_active()

    # Test deactivating non-existent instance
    result = manager.deactivate("algo_flipper", "nonexistent")
    assert result is False

    # Test deactivating all instances of a template
    deactivated_count = manager.deactivate_template("algo_flipper")
    assert deactivated_count == 1  # test2 was deactivated
    assert len(manager.list_all_active()) == 1
    assert "simple.test3" in manager.list_all_active()

    # Test deactivating all instances across all templates
    total_deactivated = manager.deactivate_all()
    assert total_deactivated == 1  # test3 was deactivated
    assert len(manager.list_all_active()) == 0


def test_multi_template_manager_performance_tracking():
    """Test performance tracking through the multi-template manager."""
    runtime = IfThenRuntime()
    manager = setup_manager_with_algo_flipper(runtime)

    # Activate some instances
    args = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="test-algo",
        qty=1,
    )

    manager.activate("algo_flipper", "perf_test1", args)
    manager.activate("algo_flipper", "perf_test2", args)

    # Record performance events
    manager.state_update("algo_flipper", "perf_test1", {"profit": 100.0, "win": True})
    manager.state_update("algo_flipper", "perf_test1", {"profit": -50.0, "win": False})
    manager.state_update("algo_flipper", "perf_test2", {"profit": 200.0, "win": True})

    # Test individual performance summaries
    summary1 = manager.get_performance_summary("algo_flipper", "perf_test1")
    assert summary1.total_events == 2
    assert summary1.total_profit == 50.0
    assert summary1.win_count == 1
    assert summary1.loss_count == 1

    summary2 = manager.get_performance_summary("algo_flipper", "perf_test2")
    assert summary2.total_events == 1
    assert summary2.total_profit == 200.0
    assert summary2.win_count == 1
    assert summary2.loss_count == 0

    # Test get_all_performance_summaries
    all_summaries = manager.get_all_performance_summaries()
    assert len(all_summaries) == 2
    assert "algo_flipper.perf_test1" in all_summaries
    assert "algo_flipper.perf_test2" in all_summaries
    assert all_summaries["algo_flipper.perf_test1"].total_profit == 50.0
    assert all_summaries["algo_flipper.perf_test2"].total_profit == 200.0

    # Test performance events
    events1 = manager.get_performance_events("algo_flipper", "perf_test1")
    assert len(events1) == 2
    assert events1[0]["profit"] == 100.0
    assert events1[1]["profit"] == -50.0


def test_multi_template_manager_template_validation():
    """Test template validation methods."""
    runtime = IfThenRuntime()
    manager = setup_manager_with_algo_flipper(runtime)

    # Test get_template_variables
    vars = manager.get_template_variables("algo_flipper")
    expected_vars = {
        "algo_symbol",
        "watch_symbol",
        "trade_symbol",
        "evict_symbol",
        "timeframe",
        "algo",
        "qty",
        "offset",
        "profit_pts",
        "loss_pts",
        "flip_sides",
    }
    assert vars == expected_vars

    # Test validate_template_args
    incomplete_args = {"algo_symbol": "MNQ", "timeframe": 35}
    is_valid, missing = manager.validate_template_args("algo_flipper", incomplete_args)
    assert not is_valid
    assert len(missing) > 0

    complete_args = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="test",
        qty=1,
    )
    is_valid, missing = manager.validate_template_args("algo_flipper", complete_args)
    assert is_valid
    assert len(missing) == 0

    # Test populate_template (preview functionality)
    preview_dsl = manager["algo_flipper"].populate_template(complete_args)
    assert "MNQ.35.algos.test.stopped" in preview_dsl
    assert "buy /MNQ" in preview_dsl


def test_multi_template_manager_active_summary():
    """Test getting active summary across all templates."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Add custom template and activate instances
    manager.from_string(
        'test = "if {{symbol}} price > {{threshold}}: say {{message}}"', "custom"
    )

    test_dsl = 'test = "if AAPL price > 100: say HELLO"\nstart: test'
    manager["custom"].activate("custom.instance1", test_dsl)
    manager["custom"].activate("custom.instance2", test_dsl)

    # Get active summary
    summary = manager.get_active_summary()
    assert len(summary) == 2
    assert "custom.instance1" in summary
    assert "custom.instance2" in summary

    # Each summary should contain expected fields
    for instance_name, info in summary.items():
        assert hasattr(info, "activated_at")
        assert hasattr(info, "created_count")
        assert hasattr(info, "predicate_count")
        assert hasattr(info, "start_predicate_count")


def test_multi_template_manager_utility_methods():
    """Test utility methods of multi-template manager."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Add and activate a template
    manager.from_string(
        'test = "if {{symbol}} price > {{threshold}}: say {{message}}"', "temp_test"
    )
    test_dsl = 'test = "if AAPL price > 100: say HELLO"\nstart: test'
    manager["temp_test"].activate("temp_test.instance1", test_dsl)

    assert len(manager.list_all_active()) == 1

    # Test remove_template
    removed = manager.remove_template("temp_test")
    assert removed is True
    assert len(manager.list_all_active()) == 0  # Should be deactivated first
    assert "temp_test" not in manager.get_template_names()

    # Test removing non-existent template
    removed = manager.remove_template("nonexistent")
    assert removed is False

    # Test clear_template_cache
    manager.clear_template_cache()  # Should not crash


def test_multi_template_manager_multiple_templates():
    """Test managing multiple different templates simultaneously."""
    runtime = IfThenRuntime()
    manager = setup_manager_with_algo_flipper(runtime)

    # Create instances from different templates
    args = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="test",
        qty=1,
    )

    manager.activate("algo_flipper", "mnq_instance", args)

    # Add breakout monitor - explicitly create first
    manager.from_builtin("breakout_monitor.dsl", "breakout_monitor")
    breakout_args = {
        "symbol": "SPY",
        "high_level": "450",
        "low_level": "440",
        "timeframe": "60",
        "qty": "10",
    }
    manager.activate("breakout_monitor", "spy_breakout", breakout_args)

    # Add simple flipper - explicitly create first
    manager.from_builtin("simple_flipper.dsl", "simple_flipper")
    simple_args = {
        "symbol": "AAPL",
        "buy_condition": "AAPL price < 150",
        "sell_condition": "AAPL price > 160",
        "qty": 100,
        "action_buy": "buy AAPL 100 MKT",
        "action_sell": "sell AAPL 100 MKT",
    }
    manager.activate("simple_flipper", "aapl_flip", simple_args)

    # Should have 3 active instances from 3 different templates
    active = manager.list_all_active()
    assert len(active) == 3
    assert "algo_flipper.mnq_instance" in active
    assert "breakout_monitor.spy_breakout" in active
    assert "simple_flipper.aapl_flip" in active

    # Test template-specific listings
    assert len(manager.list_active_for_template("algo_flipper")) == 1
    assert len(manager.list_active_for_template("breakout_monitor")) == 1
    assert len(manager.list_active_for_template("simple_flipper")) == 1
    assert len(manager.list_active_for_template("nonexistent")) == 0

    # Test performance tracking across different templates
    manager.state_update("algo_flipper", "mnq_instance", {"profit": 100.0, "win": True})
    manager.state_update("simple_flipper", "aapl_flip", {"profit": -25.0, "win": False})

    summaries = manager.get_all_performance_summaries()
    # get_all_performance_summaries returns summaries for all active instances
    assert len(summaries) == 3
    assert "algo_flipper.mnq_instance" in summaries
    assert "simple_flipper.aapl_flip" in summaries
    assert "breakout_monitor.spy_breakout" in summaries

    # Only two have performance data
    performance_with_data = [s for s in summaries.values() if s.total_events > 0]
    assert len(performance_with_data) == 2

    # Test deactivating one template doesn't affect others
    manager.deactivate_template("algo_flipper")
    active_after = manager.list_all_active()
    assert len(active_after) == 2
    assert "breakout_monitor.spy_breakout" in active_after
    assert "simple_flipper.aapl_flip" in active_after


def test_multi_template_manager_error_handling():
    """Test error handling in multi-template manager."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Test accessing non-existent builtin template
    with pytest.raises(ValueError, match="Template 'nonexistent' not found"):
        manager["nonexistent"]

    # Test performance tracking on non-existent template
    with pytest.raises(ValueError, match="Template 'nonexistent' not found"):
        manager.state_update("nonexistent", "instance", {"profit": 100.0, "win": True})

    with pytest.raises(ValueError, match="Template 'nonexistent' not found"):
        manager.get_performance_summary("nonexistent", "instance")

    with pytest.raises(ValueError, match="Template 'nonexistent' not found"):
        manager.get_performance_events("nonexistent", "instance")


def test_multi_template_manager_integration_example():
    """Test a complete integration example with the multi-template manager."""
    runtime = IfThenRuntime()
    manager = setup_manager_with_algo_flipper(runtime)

    # Create multiple algorithm instances
    algorithms = [
        (
            "algo_flipper",
            "mnq_fast",
            create_template_args_for_algo_flipper(
                "MNQ", "/NQM5", "/MNQ", "MNQ", 15, "ema-cross-fast", 1
            ),
        ),
        (
            "algo_flipper",
            "mnq_slow",
            create_template_args_for_algo_flipper(
                "MNQ", "/NQM5", "/MNQ", "MNQ", 60, "ema-cross-slow", 1
            ),
        ),
        (
            "algo_flipper",
            "es_hedge",
            create_template_args_for_algo_flipper(
                "ES", "/ESZ4", "/ES", "ES", 300, "mean-revert", 2
            ),
        ),
    ]

    # Activate all algorithms
    for template_name, instance_name, args in algorithms:
        manager.activate(template_name, instance_name, args)

    # Verify all are active
    active_instances = manager.list_all_active()
    assert len(active_instances) == 3
    expected_instances = [f"{t}.{i}" for t, i, _ in algorithms]
    for expected in expected_instances:
        assert expected in active_instances

    # Simulate trading performance
    performance_data = [
        (
            "algo_flipper",
            "mnq_fast",
            {"profit": 150.0, "win": True, "metadata": {"timeframe": 15}},
        ),
        ("algo_flipper", "mnq_fast", {"profit": -25.0, "win": False}),
        (
            "algo_flipper",
            "mnq_slow",
            {"profit": 300.0, "win": True, "metadata": {"timeframe": 60}},
        ),
        ("algo_flipper", "es_hedge", {"profit": -75.0, "win": False}),
        ("algo_flipper", "mnq_fast", {"profit": 200.0, "win": True}),
    ]

    for template_name, instance_name, event_data in performance_data:
        manager.state_update(template_name, instance_name, event_data)

    # Analyze performance
    all_summaries = manager.get_all_performance_summaries()

    # mnq_fast should have 3 events (2 wins, 1 loss)
    mnq_fast_summary = all_summaries["algo_flipper.mnq_fast"]
    assert mnq_fast_summary.total_events == 3
    assert mnq_fast_summary.win_count == 2
    assert mnq_fast_summary.loss_count == 1
    assert mnq_fast_summary.total_profit == 325.0  # 150 - 25 + 200
    assert mnq_fast_summary.current_streak == 1  # Last event was a win
    assert mnq_fast_summary.streak_type == "win"

    # mnq_slow should have 1 win
    mnq_slow_summary = all_summaries["algo_flipper.mnq_slow"]
    assert mnq_slow_summary.total_events == 1
    assert mnq_slow_summary.win_count == 1
    assert mnq_slow_summary.total_profit == 300.0

    # es_hedge should have 1 loss
    es_hedge_summary = all_summaries["algo_flipper.es_hedge"]
    assert es_hedge_summary.total_events == 1
    assert es_hedge_summary.loss_count == 1
    assert es_hedge_summary.total_profit == -75.0

    # Performance-based decision making
    underperformers = []
    for instance_name, summary in all_summaries.items():
        if summary.total_events >= 1:  # Have enough data
            if summary.win_rate < 0.5 or summary.total_profit < -50:
                underperformers.append(instance_name)

    # Should identify es_hedge as underperformer
    assert "algo_flipper.es_hedge" in underperformers

    # Deactivate underperformers
    for instance_name in underperformers:
        template_name, instance = instance_name.split(".", 1)
        manager.deactivate(template_name, instance)

    # Should have 2 active instances remaining
    remaining_active = manager.list_all_active()
    assert len(remaining_active) == 2
    assert "algo_flipper.mnq_fast" in remaining_active
    assert "algo_flipper.mnq_slow" in remaining_active
    assert "algo_flipper.es_hedge" not in remaining_active

    # Performance data should still be available for deactivated algorithms
    events = manager.get_performance_events("algo_flipper", "es_hedge")
    assert len(events) == 1
    assert events[0]["profit"] == -75.0


def test_multi_template_manager_source_conflict_validation():
    """Test that template names check for exact source matches and raise conflicts appropriately."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Test from_builtin source conflict validation
    executor1 = manager.from_builtin("algo_flipper.dsl", "test_template")

    # Calling again with same name and builtin should return the same instance
    executor2 = manager.from_builtin("algo_flipper.dsl", "test_template")
    assert executor1 is executor2

    # Calling with same name but different builtin should raise ValueError
    with pytest.raises(
        ValueError, match="Template name 'test_template' already exists"
    ):
        manager.from_builtin("simple_flipper.dsl", "test_template")

    # Test from_string source conflict validation
    template_content1 = 'pred1 = "if AAPL price > 100: buy AAPL 1 MKT"'
    template_content2 = 'pred2 = "if MSFT price > 200: sell MSFT 1 MKT"'

    executor3 = manager.from_string(template_content1, "string_template")

    # Calling again with same name and content should return the same instance
    executor4 = manager.from_string(template_content1, "string_template")
    assert executor3 is executor4

    # Calling with same name but different content should raise ValueError
    with pytest.raises(
        ValueError, match="Template name 'string_template' already exists"
    ):
        manager.from_string(template_content2, "string_template")


def test_multi_template_manager_from_file_source_conflict():
    """Test from_file source conflict validation."""
    import os
    import tempfile

    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Create temporary template files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dsl", delete=False) as f1:
        f1.write(
            'pred1 = "if {{symbol}} price > {{threshold}}: buy {{symbol}} {{qty}} MKT"'
        )
        temp_file1 = f1.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dsl", delete=False) as f2:
        f2.write(
            'pred2 = "if {{symbol}} price < {{threshold}}: sell {{symbol}} {{qty}} MKT"'
        )
        temp_file2 = f2.name

    try:
        executor1 = manager.from_file(temp_file1, "file_template")

        # Calling again with same name and file should return the same instance
        executor2 = manager.from_file(temp_file1, "file_template")
        assert executor1 is executor2

        # Calling with same name but different file should raise ValueError
        with pytest.raises(
            ValueError, match="Template name 'file_template' already exists"
        ):
            manager.from_file(temp_file2, "file_template")

    finally:
        # Clean up temporary files
        os.unlink(temp_file1)
        os.unlink(temp_file2)


def test_multi_template_manager_cross_source_conflicts():
    """Test that conflicts are detected across different source types (builtin, file, string)."""
    import os
    import tempfile

    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Create a template from builtin
    manager.from_builtin("algo_flipper.dsl", "mixed_template")

    # Try to create with same name from string - should fail
    template_content = 'pred = "if AAPL price > 100: buy AAPL 1 MKT"'
    with pytest.raises(
        ValueError, match="Template name 'mixed_template' already exists"
    ):
        manager.from_string(template_content, "mixed_template")

    # Try to create with same name from file - should fail
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dsl", delete=False) as f:
        f.write(
            'pred = "if {{symbol}} price > {{threshold}}: buy {{symbol}} {{qty}} MKT"'
        )
        temp_file = f.name

    try:
        with pytest.raises(
            ValueError, match="Template name 'mixed_template' already exists"
        ):
            manager.from_file(temp_file, "mixed_template")
    finally:
        os.unlink(temp_file)


def test_multi_template_manager_exact_source_matching():
    """Test that exact source matching works correctly for determining when to return existing vs create new."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Test builtin exact matching
    executor1 = manager.from_builtin("algo_flipper.dsl", "template1")
    executor2 = manager.from_builtin(
        "algo_flipper.dsl", "template1"
    )  # Same builtin, same name
    assert executor1 is executor2

    # Test string exact matching
    content = 'pred = "if AAPL price > 100: buy AAPL 1 MKT"'
    executor3 = manager.from_string(content, "template2")
    executor4 = manager.from_string(content, "template2")  # Same content, same name
    assert executor3 is executor4

    # Test file exact matching
    import os
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dsl", delete=False) as f:
        f.write(
            'pred = "if {{symbol}} price > {{threshold}}: buy {{symbol}} {{qty}} MKT"'
        )
        temp_file = f.name

    try:
        executor5 = manager.from_file(temp_file, "template3")
        executor6 = manager.from_file(temp_file, "template3")  # Same file, same name
        assert executor5 is executor6
    finally:
        os.unlink(temp_file)


# ==================== ENHANCED OBSERVABILITY TESTS ====================


def test_template_preview_capability():
    """Test template preview without activation."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Test preview with valid arguments
    args = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="test-algo",
        qty=1,
    )

    preview_dsl = executor.populate_template(args)

    # Verify preview contains expected content
    assert "MNQ.35.algos.test-algo.stopped" in preview_dsl
    assert "buy /MNQ" in preview_dsl
    assert "cancel MNQ*" in preview_dsl
    assert "start: entrypoint" in preview_dsl

    # Verify no template variables remain
    assert "{{" not in preview_dsl
    assert "{%" not in preview_dsl

    # Verify no activation occurred
    assert len(executor.list_active()) == 0


def test_comprehensive_metadata_storage():
    """Test comprehensive metadata storage and retrieval."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Create and activate a template
    args = create_template_args_for_algo_flipper(
        algo_symbol="ES",
        watch_symbol="/ESZ4",
        trade_symbol="/ES",
        evict_symbol="ES",
        timeframe=60,
        algo="test-algo",
        qty=2,
    )

    created_count, start_ids, all_ids = executor.create_and_activate(
        "test_metadata", args
    )

    # Get comprehensive metadata
    metadata = executor.get_comprehensive_metadata("test_metadata")
    assert metadata is not None

    # Verify metadata content
    assert metadata.instance_name == "test_metadata"
    assert metadata.template_source_id == "builtin:algo_flipper.dsl"
    assert metadata.raw_template == BUILTIN_TEMPLATES["algo_flipper.dsl"]
    assert metadata.template_parameters == args
    assert "ES.60.algos.test-algo.stopped" in metadata.rendered_dsl
    assert metadata.created_count == created_count
    assert metadata.predicate_count == len(all_ids)
    assert metadata.start_predicate_count == len(start_ids)
    assert metadata.template_variables is not None
    assert "algo_symbol" in metadata.template_variables
    assert metadata.validation_status == "valid"
    assert metadata.performance_summary is None  # No performance data yet

    # Add some performance data
    executor.state_update("test_metadata", {"profit": 100.0, "win": True})
    executor.state_update("test_metadata", {"profit": -50.0, "win": False})

    # Get updated metadata
    updated_metadata = executor.get_comprehensive_metadata("test_metadata")
    assert updated_metadata.performance_summary is not None
    assert updated_metadata.performance_summary.total_events == 2
    assert updated_metadata.performance_summary.total_profit == 50.0
    assert updated_metadata.performance_summary.win_count == 1
    assert updated_metadata.performance_summary.loss_count == 1


def test_all_active_metadata():
    """Test getting metadata for all active templates."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Create multiple instances
    args1 = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="test1",
        qty=1,
    )
    args2 = create_template_args_for_algo_flipper(
        algo_symbol="ES",
        watch_symbol="/ESZ4",
        trade_symbol="/ES",
        evict_symbol="ES",
        timeframe=60,
        algo="test2",
        qty=2,
    )

    executor.create_and_activate("instance1", args1)
    executor.create_and_activate("instance2", args2)

    # Get all metadata
    all_metadata = executor.get_all_active_metadata()

    assert len(all_metadata) == 2
    assert "instance1" in all_metadata
    assert "instance2" in all_metadata

    # Verify each instance has correct metadata
    metadata1 = all_metadata["instance1"]
    assert metadata1.template_parameters["algo_symbol"] == "MNQ"
    assert metadata1.template_parameters["timeframe"] == 35

    metadata2 = all_metadata["instance2"]
    assert metadata2.template_parameters["algo_symbol"] == "ES"
    assert metadata2.template_parameters["timeframe"] == 60


def test_template_debug_info():
    """Test template debug information."""
    runtime = IfThenRuntime()
    executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

    # Create and activate a template
    args = create_template_args_for_algo_flipper(
        algo_symbol="NQ",
        watch_symbol="/NQZ4",
        trade_symbol="/NQ",
        evict_symbol="NQ",
        timeframe=45,
        algo="debug-test",
        qty=3,
    )

    executor.create_and_activate("debug_instance", args)

    # Get debug info
    debug_info = executor.get_template_debug_info("debug_instance")

    assert debug_info is not None
    assert debug_info.instance_name == "debug_instance"
    assert debug_info.template_source_id == "builtin:algo_flipper.dsl"
    assert "algo_symbol" in debug_info.template_variables
    assert debug_info.validation_status == "valid"
    assert debug_info.predicate_count > 0
    assert debug_info.performance_events_count == 0
    assert debug_info.has_performance_data is False
    assert debug_info.template_parameters["algo_symbol"] == "NQ"
    assert debug_info.rendered_dsl_length > 0
    assert debug_info.raw_template_length > 0

    # Test debug info for non-existent instance
    debug_info_nonexistent = executor.get_template_debug_info("nonexistent")
    assert debug_info_nonexistent is None


def test_multi_template_comprehensive_observability():
    """Test comprehensive observability features in multi-template manager."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Create templates
    manager.from_builtin("algo_flipper.dsl", "algo_flipper")
    manager.from_builtin("breakout_monitor.dsl", "breakout_monitor")

    # Create instances
    args1 = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="test1",
        qty=1,
    )
    args2 = create_template_args_for_algo_flipper(
        algo_symbol="ES",
        watch_symbol="/ESZ4",
        trade_symbol="/ES",
        evict_symbol="ES",
        timeframe=60,
        algo="test2",
        qty=2,
    )
    breakout_args = {
        "symbol": "SPY",
        "high_level": "450",
        "low_level": "430",
        "timeframe": "60",
        "qty": "10",
    }

    manager.activate("algo_flipper", "mnq_instance", args1)
    manager.activate("algo_flipper", "es_instance", args2)
    manager.activate("breakout_monitor", "spy_breakout", breakout_args)

    # Test comprehensive metadata retrieval
    all_metadata = manager.get_all_active_templates_comprehensive()
    assert len(all_metadata) == 3
    assert "algo_flipper.mnq_instance" in all_metadata
    assert "algo_flipper.es_instance" in all_metadata
    assert "breakout_monitor.spy_breakout" in all_metadata

    # Test individual instance metadata
    mnq_metadata = manager.get_template_instance_metadata(
        "algo_flipper", "mnq_instance"
    )
    assert mnq_metadata is not None
    assert mnq_metadata.template_parameters["algo_symbol"] == "MNQ"

    # Test debug info
    debug_info = manager.get_template_debug_info("algo_flipper", "mnq_instance")
    assert debug_info is not None
    assert debug_info.instance_name == "algo_flipper.mnq_instance"
    assert debug_info.template_parameters["algo_symbol"] == "MNQ"

    # Test system health report
    health_report = manager.get_system_health_report()
    assert health_report.total_templates == 2
    assert health_report.total_active_instances == 3
    assert health_report.templates_with_instances == 2
    assert "algo_flipper" in health_report.template_names
    assert "breakout_monitor" in health_report.template_names


def test_template_validation():
    """Test template validation capabilities."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Create templates
    manager.from_builtin("algo_flipper.dsl", "algo_flipper")
    manager.from_builtin("simple_flipper.dsl", "simple_flipper")

    # Create some instances
    args = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="test",
        qty=1,
    )
    manager.activate("algo_flipper", "test_instance", args)

    # Test validation
    validation_results = manager.validate_all_templates()

    assert "algo_flipper" in validation_results
    assert "simple_flipper" in validation_results

    algo_result = validation_results["algo_flipper"]
    assert algo_result.status == "valid"
    assert "algo_symbol" in algo_result.template_variables
    assert algo_result.active_instances == 1

    simple_result = validation_results["simple_flipper"]
    assert simple_result.status == "valid"
    assert simple_result.active_instances == 0
    assert "No active instances" in simple_result.warnings


def test_usage_statistics():
    """Test template usage statistics."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Create templates
    manager.from_builtin("algo_flipper.dsl", "algo_flipper")
    manager.from_builtin("breakout_monitor.dsl", "breakout_monitor")
    manager.from_builtin("simple_flipper.dsl", "simple_flipper")  # Unused

    # Create instances
    args1 = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="test1",
        qty=1,
    )
    args2 = create_template_args_for_algo_flipper(
        algo_symbol="ES",
        watch_symbol="/ESZ4",
        trade_symbol="/ES",
        evict_symbol="ES",
        timeframe=60,
        algo="test2",
        qty=2,
    )
    breakout_args = {
        "symbol": "SPY",
        "high_level": "450",
        "low_level": "430",
        "timeframe": "60",
        "qty": "10",
    }

    manager.activate("algo_flipper", "mnq_instance", args1)
    manager.activate("algo_flipper", "es_instance", args2)
    manager.activate("breakout_monitor", "spy_breakout", breakout_args)

    # Test usage statistics
    stats = manager.get_template_usage_statistics()

    assert stats.total_instances == 3
    assert stats.total_templates == 3
    assert stats.template_usage_counts["algo_flipper"] == 2
    assert stats.template_usage_counts["breakout_monitor"] == 1
    assert "simple_flipper" in stats.unused_templates
    assert stats.most_used_template == ("algo_flipper", 2)
    assert (
        stats.avg_instances_per_template == 1.5
    )  # 3 instances / 2 templates with instances


def test_export_configurations():
    """Test template configuration export."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Create template and instance
    manager.from_builtin("algo_flipper.dsl", "algo_flipper")
    args = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="test",
        qty=1,
    )
    manager.activate("algo_flipper", "test_instance", args)

    # Add performance data
    manager.state_update(
        "algo_flipper", "test_instance", {"profit": 100.0, "win": True}
    )

    # Test export
    export_data = manager.export_template_configurations()

    assert export_data.export_timestamp is not None
    assert export_data.system_info.total_templates == 1
    assert export_data.system_info.total_active_instances == 1

    # Check template export
    assert "algo_flipper" in export_data.templates
    template_data = export_data.templates["algo_flipper"]
    assert template_data.source_id == "builtin:algo_flipper.dsl"
    assert "algo_symbol" in template_data.template_variables
    assert len(template_data.raw_template) > 0

    # Check instance export
    assert "algo_flipper.test_instance" in export_data.instances
    instance_data = export_data.instances["algo_flipper.test_instance"]
    assert instance_data.template_name == "algo_flipper"
    assert instance_data.template_parameters["algo_symbol"] == "MNQ"
    assert len(instance_data.rendered_dsl) > 0
    assert instance_data.performance is not None
    assert instance_data.performance.total_profit == 100.0


def test_performance_analytics():
    """Test performance analytics."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Create templates and instances
    manager.from_builtin("algo_flipper.dsl", "algo_flipper")
    manager.from_builtin("breakout_monitor.dsl", "breakout_monitor")

    args1 = create_template_args_for_algo_flipper(
        algo_symbol="MNQ",
        watch_symbol="/NQM5",
        trade_symbol="/MNQ",
        evict_symbol="MNQ",
        timeframe=35,
        algo="test1",
        qty=1,
    )
    args2 = create_template_args_for_algo_flipper(
        algo_symbol="ES",
        watch_symbol="/ESZ4",
        trade_symbol="/ES",
        evict_symbol="ES",
        timeframe=60,
        algo="test2",
        qty=2,
    )

    manager.activate("algo_flipper", "mnq_instance", args1)
    manager.activate("algo_flipper", "es_instance", args2)

    # Add performance data
    manager.state_update("algo_flipper", "mnq_instance", {"profit": 150.0, "win": True})
    manager.state_update(
        "algo_flipper", "mnq_instance", {"profit": -50.0, "win": False}
    )
    manager.state_update("algo_flipper", "es_instance", {"profit": 200.0, "win": True})

    # Test analytics
    analytics = manager.get_performance_analytics()

    # Check system metrics
    system_metrics = analytics.system_metrics
    assert system_metrics.total_profit == 300.0  # 150 - 50 + 200
    assert system_metrics.total_events == 3
    assert system_metrics.total_wins == 2
    assert system_metrics.system_win_rate == 2 / 3
    assert system_metrics.active_instances_with_data == 2

    # Check best/worst instances
    assert analytics.best_instance is not None
    assert analytics.best_instance.instance_name == "algo_flipper.es_instance"
    assert analytics.best_instance.total_profit == 200.0
    assert analytics.worst_instance is not None
    assert analytics.worst_instance.instance_name == "algo_flipper.mnq_instance"
    assert analytics.worst_instance.total_profit == 100.0

    # Check template performance
    assert "algo_flipper" in analytics.template_performance
    algo_perf = analytics.template_performance["algo_flipper"]
    assert algo_perf.total_profit == 300.0
    assert algo_perf.total_events == 3
    assert algo_perf.instances == 2


def test_enhanced_observability_integration():
    """Test integration of all enhanced observability features."""
    runtime = IfThenRuntime()
    manager = IfThenMultiTemplateManager(runtime)

    # Create multiple templates
    manager.from_builtin("algo_flipper.dsl", "algo_flipper")
    manager.from_builtin("breakout_monitor.dsl", "breakout_monitor")

    # Create instances with different configurations
    algorithms = [
        (
            "algo_flipper",
            "mnq_fast",
            create_template_args_for_algo_flipper(
                "MNQ", "/NQM5", "/MNQ", "MNQ", 15, "fast-ema", 1
            ),
        ),
        (
            "algo_flipper",
            "mnq_slow",
            create_template_args_for_algo_flipper(
                "MNQ", "/NQM5", "/MNQ", "MNQ", 60, "slow-ema", 1
            ),
        ),
        (
            "breakout_monitor",
            "spy_breakout",
            {
                "symbol": "SPY",
                "high_level": "450",
                "low_level": "430",
                "timeframe": "60",
                "qty": "10",
            },
        ),
    ]

    for template_name, instance_name, args in algorithms:
        manager.activate(template_name, instance_name, args)

    # Add performance data
    performance_events = [
        ("algo_flipper", "mnq_fast", {"profit": 100.0, "win": True}),
        ("algo_flipper", "mnq_fast", {"profit": -25.0, "win": False}),
        ("algo_flipper", "mnq_slow", {"profit": 200.0, "win": True}),
        ("breakout_monitor", "spy_breakout", {"profit": 75.0, "win": True}),
    ]

    for template_name, instance_name, event_data in performance_events:
        manager.state_update(template_name, instance_name, event_data)

    # Test comprehensive observability
    all_metadata = manager.get_all_active_templates_comprehensive()
    assert len(all_metadata) == 3

    # Test system health
    health = manager.get_system_health_report()
    assert health.total_templates == 2
    assert health.total_active_instances == 3
    assert health.instances_with_performance == 3

    # Test validation
    validation = manager.validate_all_templates()
    assert all(result.status == "valid" for result in validation.values())

    # Test usage statistics
    usage_stats = manager.get_template_usage_statistics()
    assert usage_stats.total_instances == 3
    assert usage_stats.template_usage_counts["algo_flipper"] == 2

    # Test performance analytics
    analytics = manager.get_performance_analytics()
    assert analytics.system_metrics.total_profit == 350.0
    assert analytics.system_metrics.system_win_rate == 0.75

    # Test export
    export = manager.export_template_configurations()
    assert len(export.templates) == 2
    assert len(export.instances) == 3

    # Test individual metadata retrieval
    mnq_fast_metadata = manager.get_template_instance_metadata(
        "algo_flipper", "mnq_fast"
    )
    assert mnq_fast_metadata is not None
    assert mnq_fast_metadata.template_parameters["timeframe"] == 15
    assert mnq_fast_metadata.performance_summary is not None
    assert mnq_fast_metadata.performance_summary.total_profit == 75.0

    # Test debug info
    debug_info = manager.get_template_debug_info("algo_flipper", "mnq_fast")
    assert debug_info is not None
    assert debug_info.template_parameters["timeframe"] == 15
    assert debug_info.performance_events_count == 2
    assert debug_info.has_performance_data is True
