"""Tests for IfThen Template System."""

import pytest

from tradeapis.ifthen import IfThenRuntime
from tradeapis.ifthen_templates import (
    BUILTIN_TEMPLATES,
    IfThenRuntimeTemplateExecutor,
    _template_cache,
    create_algo_flipper_executor,
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
    }
    assert executor.get_template_variables() == expected_vars


def test_template_validation():
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
    assert args["qty"] == 2


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
    executor = create_algo_flipper_executor(runtime)

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
    assert summary["test_instance"]["created_count"] == created_count
    assert summary["test_instance"]["predicate_count"] == len(all_ids)

    # Deactivate the template
    assert executor.deactivate("test_instance") is True
    assert executor.list_active() == []

    # Deactivating non-existent template should return False
    assert executor.deactivate("nonexistent") is False


def test_create_and_activate_with_name():
    """Test create_and_activate with the new name parameter."""
    runtime = IfThenRuntime()
    executor = create_algo_flipper_executor(runtime)

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
    executor = create_algo_flipper_executor(runtime)

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
    executor = create_algo_flipper_executor(runtime)

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
    executor = create_algo_flipper_executor(runtime)

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
    executor = create_algo_flipper_executor(runtime)

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
    executor = create_algo_flipper_executor(runtime)

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
    executor = create_algo_flipper_executor(runtime)

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
    executor = create_algo_flipper_executor(runtime)

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
    executor = create_algo_flipper_executor(runtime)

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
    executor = create_algo_flipper_executor(runtime)

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
    executor = create_algo_flipper_executor(runtime)

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
    executor = create_algo_flipper_executor(runtime)

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
    executor = IfThenRuntimeTemplateExecutor("algo_flipper.dsl", runtime)
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
    multi_executor = IfThenRuntimeTemplateExecutor("multi_symbol_flipper.dsl", runtime)

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
    executor3 = create_algo_flipper_executor(runtime)
    print(f"Cache hit: {executor3._compiled_template is executor._compiled_template}")

    # Method 4: Performance tracking demonstration
    print("\n4. Performance Tracking:")
    perf_executor = create_algo_flipper_executor(IfThenRuntime())

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
