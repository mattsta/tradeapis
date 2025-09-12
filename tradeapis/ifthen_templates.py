"""IfThen Template System for parameterized DSL generation with performance tracking.

This module provides a comprehensive system for creating, activating, and monitoring
reusable IfThen DSL templates for trading algorithms. It combines Jinja2 templating
with activation lifecycle management and real-time performance tracking.

## Core Features

### 1. Template Management
- **Jinja2 Templating**: Powerful template system with variables, loops, conditionals
- **Built-in Templates**: Pre-defined templates for common trading patterns
- **Template Caching**: Compiled templates are cached for performance
- **Validation**: Template variable validation before rendering

### 2. Activation Lifecycle
- **Named Activation**: Each template instance gets a unique name for tracking
- **Lifecycle Management**: Clean activation, deactivation, and bulk operations
- **Status Tracking**: Monitor which templates are currently active
- **Resource Cleanup**: Automatic predicate cleanup when deactivating

### 3. Performance Monitoring
- **Real-time Updates**: Record profit/loss events as they happen
- **Historical Tracking**: Complete event history with timestamps
- **Summary Statistics**: Win rate, streaks, runtime, profitability
- **Audit Trail**: Full performance data for algorithm evaluation

## Quick Start

### Explicit Template Source Usage
```python
from tradeapis.ifthen import IfThenRuntime
from tradeapis.ifthen_templates import IfThenRuntimeTemplateExecutor, create_template_args_for_algo_flipper

# Create runtime
runtime = IfThenRuntime()

# Built-in template (recommended for common patterns)
executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

# File template (for custom templates)
# executor = IfThenRuntimeTemplateExecutor.from_file(runtime, "/path/to/custom.dsl")

# String template (for dynamic generation)
# template = "check = 'if {{symbol}} price > {{threshold}}: buy {{symbol}} {{qty}} MKT'"
# executor = IfThenRuntimeTemplateExecutor.from_string(runtime, template)

# Create algorithm parameters using helper function
args = create_template_args_for_algo_flipper(
    algo_symbol="MNQ",
    watch_symbol="/NQM5",
    trade_symbol="/MNQ",
    evict_symbol="MNQ",
    timeframe=35,
    algo="temathma-5x-12x-vwap",
    qty=1,
    offset=0.50,
    profit_pts=4,
    loss_pts=7
)

# Activate the algorithm with a name
created_count, start_ids, all_ids = executor.create_and_activate("mnq_scalper", args)

# Monitor performance
executor.state_update("mnq_scalper", {"profit": 150.0, "win": True})
executor.state_update("mnq_scalper", {"profit": -25.0, "win": False})

# Get performance summary
summary = executor.score_summary("mnq_scalper")
logger.info(f"Win rate: {summary.win_rate:.1%}, Total profit: {summary.total_profit}")
```

### Advanced Template Usage

#### Custom String Templates
```python
# Create a custom template inline
custom_template = '''
# Custom scalping template
entry_signal = "if {{symbol}} { {{entry_condition}} }: buy {{symbol}} {{qty}} MKT"
exit_signal = "if {{symbol}} { {{exit_condition}} }: sell {{symbol}} {{qty}} MKT"

flow main:
    entry_signal -> exit_signal -> @

start: main
'''

executor = IfThenRuntimeTemplateExecutor.from_string(runtime, custom_template)
args = {
    "symbol": "AAPL",
    "entry_condition": "price > vwap and volume > 1000000",
    "exit_condition": "price < vwap or profit > 50",
    "qty": 100
}
executor.create_and_activate("aapl_scalp", args)
```

#### File-based Templates
```python
# Load from external file
executor = IfThenRuntimeTemplateExecutor.from_file(runtime, "/strategies/momentum.dsl")

# Template files use same Jinja2 syntax
# momentum.dsl content:
# entry = "if {{symbol}} rsi > {{rsi_threshold}}: buy {{symbol}} {{qty}} MKT"
# exit = "if {{symbol}} rsi < {{exit_rsi}}: sell {{symbol}} {{qty}} MKT"
# start: entry -> exit

args = {"symbol": "SPY", "rsi_threshold": 70, "exit_rsi": 30, "qty": 50}
executor.create_and_activate("spy_momentum", args)
```

### Complete Lifecycle Management
```python
# Create and activate multiple algorithms
algorithms = [
    ("mnq_fast", {"timeframe": 15, "algo": "ema-cross-fast"}),
    ("mnq_slow", {"timeframe": 60, "algo": "ema-cross-slow"}),
    ("es_hedge", {"symbol": "ES", "timeframe": 300, "algo": "mean-revert"})
]

executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")
for name, custom_args in algorithms:
    base_args = create_template_args_for_algo_flipper(
        algo_symbol="MNQ", watch_symbol="/NQM5", trade_symbol="/MNQ",
        evict_symbol="MNQ", qty=1, **custom_args
    )
    executor.create_and_activate(name, base_args)

# Monitor all active algorithms
active_algos = executor.list_active()
logger.info(f"Running {len(active_algos)} algorithms: {active_algos}")

# Performance tracking for each
for algo_name in active_algos:
    summary = executor.score_summary(algo_name)
    logger.info(f"{algo_name}: {summary.win_rate:.1%} win rate, {summary.total_profit} profit")

# Selective deactivation
if executor.score_summary("mnq_fast").win_rate < 0.3:
    executor.deactivate("mnq_fast")
    logger.warning("Deactivated underperforming algorithm")

# Bulk cleanup
executor.deactivate_all()
```

### Template Development and Debugging
```python
# Validate template variables before rendering
executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")
required_vars = executor.get_template_variables()
logger.info(f"Required variables: {required_vars}")

# Check if args are complete before activation
args = {"algo_symbol": "MNQ", "timeframe": 35}  # Incomplete
is_valid, missing = executor.validate_template_args(args)
if not is_valid:
    logger.warning(f"Missing variables: {missing}")

# Preview generated DSL without activation
complete_args = create_template_args_for_algo_flipper(
    algo_symbol="MNQ", watch_symbol="/NQM5", trade_symbol="/MNQ",
    evict_symbol="MNQ", timeframe=35, algo="test", qty=1
)
preview_dsl = executor.populate_template(complete_args)
logger.debug("Generated DSL preview:")
logger.debug(preview_dsl[:200] + "...")
```

### Performance Tracking and Analytics
```python
from loguru import logger

# Record trading events in real-time
executor.state_update("algo1", {"profit": 300.0, "win": True, "metadata": {"symbol": "MNQ", "hold_time": 120}})
executor.state_update("algo1", {"profit": -75.0, "win": False, "metadata": {"symbol": "MNQ", "hold_time": 45}})

# Get detailed event history
events = executor.score_get("algo1")
logger.info(f"Recorded {len(events)} trading events")

# Comprehensive performance summary
summary = executor.score_summary("algo1")
logger.info(f"Algorithm {summary.name}: {summary.win_rate:.1%} win rate, profit {summary.total_profit}")

# Performance-based algorithm management
for algo_name in executor.list_active():
    perf = executor.score_summary(algo_name)
    if perf.total_events >= 10:  # Enough data to evaluate
        if perf.win_rate < 0.4 or perf.total_profit < -500:
            executor.deactivate(algo_name)
            logger.warning(f"Deactivated {algo_name} due to poor performance")
        elif perf.win_rate > 0.8 and perf.total_profit > 1000:
            logger.success(f"Strong performer {algo_name} - consider increasing position size")
```

### Lifecycle Management
```python
# Check what's running
active_algos = executor.list_active()  # ["mnq_scalper", "es_momentum"]

# Get detailed info
info = executor.get_active_info("mnq_scalper")
print(f"Activated at: {info.activated_at}")
print(f"Predicates: {len(info.all_ids)}")

# Get summary of all active templates
summary = executor.get_active_summary()
for name, details in summary.items():
    print(f"{name}: {details['predicate_count']} predicates, active for {details['activated_at']}")

# Deactivate specific algorithm
executor.deactivate("mnq_scalper")

# Emergency shutdown - deactivate everything
deactivated_count = executor.deactivate_all()
```

## Advanced Usage

### Custom Templates
```python
# Create your own template file
custom_template = '''
entry = "if {{ symbol }} {{ condition }}: {{ action }}"
exit = "if {{ symbol }} {{ exit_condition }}: {{ exit_action }}"

flow strategy:
    entry -> exit

start: strategy
'''

# Load and use custom template
executor = IfThenRuntimeTemplateExecutor("custom.dsl", runtime)
args = {"symbol": "AAPL", "condition": "price > 150", "action": "buy AAPL 100"}
executor.create_and_activate("apple_breakout", args)
```

### Multi-Symbol Templates
```python
# Use the multi-symbol template
executor = IfThenRuntimeTemplateExecutor("multi_symbol_flipper.dsl", runtime)

symbols_config = [
    create_symbol_config_for_flipper("MNQ", "/NQM5", "/MNQ", "MNQ", timeframe=35),
    create_symbol_config_for_flipper("ES", "/ESZ4", "/ESZ5", "ES", timeframe=60),
]

args = {
    "symbols": symbols_config,
    "base_timeframe": 30,
    "base_algo": "momentum-reversal",
    "default_qty": 1
}

executor.create_and_activate("multi_futures", args)
```

### Pure Activation (Pre-rendered DSL)
```python
# Generate DSL in one process
dsl_text = executor.populate_template(args)

# Activate in another process (useful for distributed systems)
executor.activate("distributed_algo", dsl_text)
```

## Performance Monitoring Details

### Event Recording
- **Required Fields**: `profit` (float), `win` (bool)
- **Optional Fields**: `ts` (datetime, defaults to now), plus any custom metadata
- **Metadata**: Additional fields are stored as metadata (e.g., trade_type, market_condition)

### Summary Statistics
- **Total Events**: Number of trades recorded
- **Win Rate**: Percentage of winning trades
- **Total P&L**: Sum of all profit/loss amounts
- **Current Streak**: Consecutive wins (positive) or losses (negative) from most recent trades
- **Runtime**: Time elapsed from first to last event
- **Average P&L**: Average profit per trade

### Streak Calculation
The streak tracks consecutive wins or losses from the most recent event backwards:
- Positive streak: consecutive wins (e.g., current_streak=5, streak_type="win")
- Negative streak: consecutive losses (e.g., current_streak=3, streak_type="loss")

## Built-in Templates

### algo_flipper.dsl
Monitors algorithm status and flips positions when stopped.
**Variables**: algo_symbol, watch_symbol, trade_symbol, evict_symbol, timeframe, algo, qty, offset, profit_pts, loss_pts

### multi_symbol_flipper.dsl
Creates multiple algorithm flippers for different symbols.
**Variables**: symbols (list), base_timeframe, base_algo, default_qty

### simple_flipper.dsl
Basic buy/sell alternating pattern.
**Variables**: symbol, buy_condition, sell_condition, qty, action_buy, action_sell

### breakout_monitor.dsl
Monitors price breakouts above/below key levels.
**Variables**: symbol, high_level, low_level, timeframe, qty

### mean_reversion.dsl
RSI-based mean reversion strategy.
**Variables**: symbol, timeframe, rsi_high, rsi_low, qty

## Error Handling
- Template validation errors for missing variables
- Jinja2 template rendering errors
- Duplicate activation name errors
- Performance tracking validation errors

Uses Jinja2 for powerful templating with caching for performance.
"""

import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import jinja2
from jinja2 import BaseLoader, Environment, Template, meta

from .ifthen import IfThenRuntime, PredicateId
from .ifthen_dsl import IfThenDSLLoader


@dataclass(slots=True)
class CachedTemplate:
    """Represents a cached compiled template with metadata."""

    template: Template
    mtime: float
    content_hash: str


@dataclass(slots=True)
class ActiveTemplate:
    """Represents an active template instance running in the IfThen runtime."""

    name: str
    dsl_text: str
    created_count: int
    start_ids: set[PredicateId]
    all_ids: set[PredicateId]
    activated_at: datetime


@dataclass(slots=True)
class PerformanceEvent:
    """Represents a single performance event for an algorithm."""

    timestamp: datetime
    profit: float
    win: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PerformanceSummary:
    """Summary statistics for an algorithm's performance."""

    name: str
    total_events: int
    total_profit: float
    win_count: int
    loss_count: int
    win_rate: float
    current_streak: int  # Positive for wins, negative for losses
    streak_type: str  # "win" or "loss"
    runtime_seconds: float
    first_event: datetime | None
    last_event: datetime | None
    avg_profit_per_trade: float


@dataclass(slots=True)
class TemplateCache:
    """Caches compiled Jinja2 templates with file modification time tracking."""

    _cache: dict[str, CachedTemplate] = field(default_factory=dict)

    def get_template(self, template_path: str, content: str) -> Template:
        """Get cached template or compile and cache if not available or outdated."""
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Check if we have a cached template
        if template_path in self._cache:
            cached = self._cache[template_path]

            # For file-based templates, check modification time
            if Path(template_path).exists():
                current_mtime = os.path.getmtime(template_path)
                if (
                    cached.mtime >= current_mtime
                    and cached.content_hash == content_hash
                ):
                    return cached.template
            else:
                # For built-in templates, just check content hash
                if cached.content_hash == content_hash:
                    return cached.template

        # Compile new template
        template = Template(content)

        # Cache it
        mtime = (
            os.path.getmtime(template_path)
            if Path(template_path).exists()
            else time.time()
        )
        self._cache[template_path] = CachedTemplate(
            template=template, mtime=mtime, content_hash=content_hash
        )

        return template

    def clear(self):
        """Clear the template cache."""
        self._cache.clear()


# Global template cache instance
_template_cache = TemplateCache()


@dataclass(slots=True)
class IfThenRuntimeTemplateExecutor:
    """Template executor for generating parameterized IfThen DSL configurations.

    This class provides a unified interface for loading templates from multiple sources
    (built-in, file, or string content) and executing them through the IfThen runtime system.

    Features:
    - Explicit template source constructors (from_builtin, from_file, from_string)
    - Jinja2 templating with variables, conditionals, loops, filters
    - Template caching for performance optimization
    - Activation lifecycle management with named instances
    - Real-time performance tracking and analytics

    Usage Examples:
        # Built-in template
        executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")

        # File template
        executor = IfThenRuntimeTemplateExecutor.from_file(runtime, "/path/to/custom.dsl")

        # String template
        template = "check = 'if {{symbol}} price > {{threshold}}: buy {{symbol}} {{qty}} MKT'"
        executor = IfThenRuntimeTemplateExecutor.from_string(runtime, template)

        # Populate and activate
        args = {"symbol": "MNQ", "threshold": 19500, "qty": 1}
        executor.create_and_activate("trade1", args)
    """

    # Core dependencies - these are the only public fields
    ifthen_runtime: IfThenRuntime
    template_source_id: str  # Identifier for caching and debugging

    # Jinja2 environment for advanced templating features
    jinja_env: Environment = field(
        default_factory=lambda: Environment(
            loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True
        )
    )

    # DSL loader for loading templates into runtime
    _dsl_loader: IfThenDSLLoader = field(init=False)

    # Internal template storage
    _template_content: str = field(init=False)
    _template_vars: set[str] = field(init=False, default_factory=set)
    _compiled_template: Template | None = field(init=False, default=None)

    # Active template tracking
    _active_templates: dict[str, ActiveTemplate] = field(default_factory=dict)

    # Performance tracking
    _performance_events: dict[str, list[PerformanceEvent]] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize internal state after construction."""
        # Note: template content should already be set via private constructor
        self._dsl_loader = IfThenDSLLoader()
        self._dsl_loader.ifthenRuntime = self.ifthen_runtime
        self._compile_template()
        self._extract_template_variables()

    @classmethod
    def from_builtin(
        cls, runtime: IfThenRuntime, template_name: str
    ) -> "IfThenRuntimeTemplateExecutor":
        """Create executor from a built-in template.

        Args:
            runtime: IfThenRuntime instance to use
            template_name: Name of built-in template (e.g. "algo_flipper.dsl")

        Returns:
            Configured executor instance

        Raises:
            ValueError: If template_name is not found in built-in templates

        Example:
            executor = IfThenRuntimeTemplateExecutor.from_builtin(runtime, "algo_flipper.dsl")
        """
        if template_name not in BUILTIN_TEMPLATES:
            available = ", ".join(BUILTIN_TEMPLATES.keys())
            raise ValueError(
                f"Built-in template '{template_name}' not found. Available: {available}"
            )

        template_content = BUILTIN_TEMPLATES[template_name]
        return cls._create_from_content(
            runtime, f"builtin:{template_name}", template_content
        )

    @classmethod
    def from_file(
        cls, runtime: IfThenRuntime, template_path: str | Path
    ) -> "IfThenRuntimeTemplateExecutor":
        """Create executor from a file template.

        Args:
            runtime: IfThenRuntime instance to use
            template_path: Path to template file

        Returns:
            Configured executor instance

        Raises:
            FileNotFoundError: If template file does not exist
            IOError: If template file cannot be read

        Example:
            executor = IfThenRuntimeTemplateExecutor.from_file(runtime, "/path/to/custom.dsl")
        """
        template_path = Path(template_path)
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        try:
            with open(template_path, encoding="utf-8") as f:
                template_content = f.read()
        except OSError as e:
            raise OSError(f"Failed to read template file {template_path}: {e}")

        return cls._create_from_content(
            runtime, f"file:{template_path}", template_content
        )

    @classmethod
    def from_string(
        cls, runtime: IfThenRuntime, template_content: str, source_id: str | None = None
    ) -> "IfThenRuntimeTemplateExecutor":
        """Create executor from a string template.

        Args:
            runtime: IfThenRuntime instance to use
            template_content: Template content as string
            source_id: Optional identifier for debugging/caching (defaults to content hash)

        Returns:
            Configured executor instance

        Raises:
            ValueError: If template_content is empty

        Example:
            template = "check = 'if {{symbol}} price > {{threshold}}: buy {{symbol}} {{qty}} MKT'"
            executor = IfThenRuntimeTemplateExecutor.from_string(runtime, template)
        """
        if not template_content or not template_content.strip():
            raise ValueError("Template content cannot be empty")

        if source_id is None:
            # Generate a source ID from content hash
            content_hash = hashlib.md5(template_content.encode("utf-8")).hexdigest()[
                :12
            ]
            source_id = f"string:{content_hash}"

        return cls._create_from_content(runtime, source_id, template_content)

    @classmethod
    def _create_from_content(
        cls, runtime: IfThenRuntime, source_id: str, template_content: str
    ) -> "IfThenRuntimeTemplateExecutor":
        """Private constructor that creates instance with template content.

        This bypasses the dataclass constructor to provide a unified creation path.
        """
        # Create instance without calling __post_init__
        instance = cls.__new__(cls)

        # Set required fields
        instance.ifthen_runtime = runtime
        instance.template_source_id = source_id
        instance.jinja_env = Environment(
            loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True
        )
        instance._active_templates = {}
        instance._performance_events = {}

        # Set template content directly
        instance._template_content = template_content

        # Initialize the rest
        instance.__post_init__()

        return instance

    def _compile_template(self):
        """Compile the template using cached compilation."""
        self._compiled_template = _template_cache.get_template(
            self.template_source_id, self._template_content
        )

    def _extract_template_variables(self):
        """Extract all template variable names from the Jinja2 template."""
        # Use Jinja2's meta module to parse template variables
        ast = self.jinja_env.parse(self._template_content)
        self._template_vars = meta.find_undeclared_variables(ast)

    def get_template_variables(self) -> set[str]:
        """Return the set of all template variables found in the template."""
        return self._template_vars.copy()

    def validate_template_args(
        self, template_args: dict[str, Any]
    ) -> tuple[bool, set[str]]:
        """Validate that all required template variables are provided.

        Returns:
            (is_valid, missing_variables)
        """
        provided_vars = set(template_args.keys())
        missing_vars = self._template_vars - provided_vars
        return len(missing_vars) == 0, missing_vars

    def populate_template(self, template_args: dict[str, Any]) -> str:
        """Populate the template with provided arguments and return DSL text.

        Args:
            template_args: Dictionary of template variable values

        Returns:
            Populated DSL text ready for runtime.load()

        Raises:
            ValueError: If required template variables are missing
            jinja2.TemplateError: If template rendering fails
        """
        # Validate template arguments
        is_valid, missing_vars = self.validate_template_args(template_args)
        if not is_valid:
            raise ValueError(f"Missing required template variables: {missing_vars}")

        try:
            # Render template with arguments using cached compiled template
            if self._compiled_template is None:
                raise ValueError("Template not compiled - this should not happen")
            dsl_text = self._compiled_template.render(**template_args)
            return dsl_text
        except jinja2.TemplateError as e:
            raise jinja2.TemplateError(f"Template rendering failed: {e}")

    def activate(
        self, name: str, dsl_text: str, enable: bool = True
    ) -> tuple[int, set[PredicateId], set[PredicateId]]:
        """Activate a pre-rendered DSL template in the runtime with tracking.

        Args:
            name: Unique name for this template activation (for tracking/management)
            dsl_text: Pre-rendered DSL text ready for DSL loader
            enable: Whether to activate the loaded predicates immediately

        Returns:
            Tuple of (created_count, start_ids, all_ids) from DSL loader

        Raises:
            ValueError: If template name is already active
        """
        if name in self._active_templates:
            raise ValueError(f"Template '{name}' is already active")

        # Load DSL into runtime via DSL loader
        created_count, start_ids, all_ids = self._dsl_loader.load(
            dsl_text, activate=enable
        )

        # Track the activation
        self._active_templates[name] = ActiveTemplate(
            name=name,
            dsl_text=dsl_text,
            created_count=created_count,
            start_ids=start_ids,
            all_ids=all_ids,
            activated_at=datetime.now(),
        )

        return created_count, start_ids, all_ids

    def deactivate(self, name: str) -> bool:
        """Deactivate and remove a template instance by name.

        Args:
            name: Name of the template activation to remove

        Returns:
            True if template was found and deactivated, False if not found
        """
        if name not in self._active_templates:
            return False

        active_template = self._active_templates[name]

        # Remove predicates from runtime
        for predicate_id in active_template.all_ids:
            self.ifthen_runtime.remove(predicate_id)

        # Remove from tracking
        del self._active_templates[name]
        return True

    def list_active(self) -> list[str]:
        """Return list of currently active template names."""
        return list(self._active_templates.keys())

    def get_active_info(self, name: str) -> ActiveTemplate | None:
        """Get detailed information about an active template.

        Args:
            name: Name of the active template

        Returns:
            ActiveTemplate instance or None if not found
        """
        return self._active_templates.get(name)

    def deactivate_all(self) -> int:
        """Deactivate all currently active templates.

        Returns:
            Number of templates that were deactivated
        """
        active_names = list(self._active_templates.keys())
        deactivated_count = 0

        for name in active_names:
            if self.deactivate(name):
                deactivated_count += 1

        return deactivated_count

    def get_active_summary(self) -> dict[str, dict[str, Any]]:
        """Get summary information about all active templates.

        Returns:
            Dictionary mapping template names to summary info
        """
        summary = {}
        for name, active in self._active_templates.items():
            summary[name] = {
                "activated_at": active.activated_at,
                "created_count": active.created_count,
                "predicate_count": len(active.all_ids),
                "start_predicate_count": len(active.start_ids),
            }
        return summary

    def create_and_activate(
        self, name: str, template_args: dict[str, Any], enable: bool = True
    ) -> tuple[int, set[PredicateId], set[PredicateId]]:
        """Convenience method to populate template and activate in runtime.

        Args:
            name: Unique name for this template activation (for tracking/management)
            template_args: Dictionary of template variable values
            enable: Whether to activate the loaded predicates immediately

        Returns:
            Tuple of (created_count, start_ids, all_ids) from runtime.load()
        """
        dsl_text = self.populate_template(template_args)
        return self.activate(name, dsl_text, enable=enable)

    # Performance Tracking Methods

    def state_update(self, name: str, event_data: dict[str, Any]) -> None:
        """Record a performance event for the named algorithm.

        Args:
            name: Name of the algorithm to update
            event_data: Event data containing 'profit' (float), 'win' (bool),
                       and optionally 'ts' (datetime) and other metadata

        Example:
            executor.state_update("algo1", {"profit": 300, "win": True})
            executor.state_update("algo1", {"ts": datetime.now(), "profit": -20, "win": False})
        """
        # Extract required fields
        profit = event_data.get("profit")
        win = event_data.get("win")

        if profit is None:
            raise ValueError("Event data must contain 'profit' field")
        if win is None:
            raise ValueError("Event data must contain 'win' field")

        # Validate types
        try:
            profit_value = float(profit)
        except (TypeError, ValueError):
            raise ValueError(
                f"Event data 'profit' must be a number, got {type(profit).__name__}"
            )

        if not isinstance(win, bool):
            raise ValueError(
                f"Event data 'win' must be a boolean, got {type(win).__name__}"
            )

        # Extract optional timestamp, default to now
        timestamp = event_data.get("ts", datetime.now())

        # Extract metadata (everything except the core fields)
        metadata = {
            k: v for k, v in event_data.items() if k not in ("profit", "win", "ts")
        }

        # Create performance event
        event = PerformanceEvent(
            timestamp=timestamp, profit=profit_value, win=win, metadata=metadata
        )

        # Initialize list if this is the first event for this algorithm
        if name not in self._performance_events:
            self._performance_events[name] = []

        # Add event to the list
        self._performance_events[name].append(event)

    def score_get(self, name: str) -> list[dict[str, Any]]:
        """Get all performance events for the named algorithm.

        Args:
            name: Name of the algorithm to get performance data for

        Returns:
            List of dictionaries containing event data, sorted by timestamp.
            Each dict contains: timestamp, profit, win, and any metadata fields.
            Returns empty list if no events recorded for this algorithm.

        Example:
            events = executor.score_get("algo1")
            # Returns: [{"ts": datetime, "profit": 300, "win": True}, ...]
        """
        if name not in self._performance_events:
            return []

        events = []
        for event in sorted(self._performance_events[name], key=lambda e: e.timestamp):
            event_dict = {
                "ts": event.timestamp,
                "profit": event.profit,
                "win": event.win,
            }
            # Add any metadata fields
            event_dict.update(event.metadata)
            events.append(event_dict)

        return events

    def score_summary(self, name: str) -> PerformanceSummary:
        """Get performance summary statistics for the named algorithm.

        Args:
            name: Name of the algorithm to get summary for

        Returns:
            PerformanceSummary object with calculated statistics including:
            - Total events, profit, win/loss counts and rates
            - Current streak (positive for wins, negative for losses)
            - Runtime duration and average profit per trade
            - First and last event timestamps

            Returns a summary with all zeros if no events recorded for this algorithm.

        Example:
            summary = executor.score_summary("algo1")
            print(f"Win rate: {summary.win_rate:.1%}")
            print(f"Current streak: {summary.current_streak} {summary.streak_type}s")
        """
        if name not in self._performance_events or not self._performance_events[name]:
            return PerformanceSummary(
                name=name,
                total_events=0,
                total_profit=0.0,
                win_count=0,
                loss_count=0,
                win_rate=0.0,
                current_streak=0,
                streak_type="none",
                runtime_seconds=0.0,
                first_event=None,
                last_event=None,
                avg_profit_per_trade=0.0,
            )

        events = sorted(self._performance_events[name], key=lambda e: e.timestamp)

        # Basic counts
        total_events = len(events)
        win_count = sum(1 for e in events if e.win)
        loss_count = total_events - win_count
        total_profit = sum(e.profit for e in events)

        # Calculate win rate
        win_rate = win_count / total_events if total_events > 0 else 0.0

        # Calculate current streak (consecutive wins or losses from the end)
        current_streak = 0
        streak_type = "win"

        if events:
            # Start from the last event and count backwards
            last_result = events[-1].win
            streak_type = "win" if last_result else "loss"

            for event in reversed(events):
                if event.win == last_result:
                    current_streak += 1
                else:
                    break

            # Make streak negative for losses
            if not last_result:
                current_streak = -current_streak

        # Calculate runtime
        first_event = events[0].timestamp if events else None
        last_event = events[-1].timestamp if events else None
        runtime_seconds = 0.0

        if first_event and last_event:
            runtime_seconds = (last_event - first_event).total_seconds()

        # Calculate average profit per trade
        avg_profit_per_trade = total_profit / total_events if total_events > 0 else 0.0

        return PerformanceSummary(
            name=name,
            total_events=total_events,
            total_profit=total_profit,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            current_streak=abs(current_streak),
            streak_type=streak_type,
            runtime_seconds=runtime_seconds,
            first_event=first_event,
            last_event=last_event,
            avg_profit_per_trade=avg_profit_per_trade,
        )


# Built-in template definitions using Jinja2 syntax
BUILTIN_TEMPLATES = {
    "algo_flipper.dsl": """# Algo Flipper Template
# Monitors an algorithm and flips positions when it stops
# Template variables: watch_symbol, algo_symbol, trade_symbol, evict_symbol, timeframe, algo, qty, offset, profit_pts, loss_pts

{% set short_action -%}
cancel {{ evict_symbol }}*; evict {{ evict_symbol }}* -1 0 MKT; buy {{ trade_symbol }} -{{ qty }} LIM @ {% if offset %}(+ live {{ offset }}){% else %}live{% endif %}{% if profit_pts %} + {{ profit_pts }}{% endif %}{% if loss_pts %} - {{ loss_pts }}{% endif %}
{%- endset %}

{% set long_action -%}
cancel {{ evict_symbol }}*; evict {{ evict_symbol }}* -1 0 MKT; buy {{ trade_symbol }} {{ qty }} LIM @ {% if offset %}(- live {{ offset }}){% else %}live{% endif %}{% if profit_pts %} + {{ profit_pts }}{% endif %}{% if loss_pts %} - {{ loss_pts }}{% endif %}
{%- endset %}

check_short = "if {{ watch_symbol }} { {{ algo_symbol }}.{{ timeframe }}.algos.{{ algo }}.stopped is True and ({{ algo_symbol }}.{{ timeframe }}.algos.{{ algo }}.be is 'short') }: {{ short_action }}"

check_long = "if {{ watch_symbol }} { {{ algo_symbol }}.{{ timeframe }}.algos.{{ algo }}.stopped is True and ({{ algo_symbol }}.{{ timeframe }}.algos.{{ algo }}.be is 'long') }: {{ long_action }}"

unstopped = "if {{ watch_symbol }} { {{ algo_symbol }}.{{ timeframe }}.algos.{{ algo }}.stopped is False }: say algo reset"

flow primary:
    check_short | check_long

flow entrypoint:
    primary -> unstopped -> @

start: entrypoint""",
    "simple_flipper.dsl": """# Simple Buy/Sell Flipper Template
# Basic pattern for alternating between buy and sell signals
# Template variables: symbol, buy_condition, sell_condition, qty, action_buy, action_sell

buy_signal = "if {{ buy_condition }}: {{ action_buy }}"
sell_signal = "if {{ sell_condition }}: {{ action_sell }}"

flow trade_signals:
    buy_signal | sell_signal

flow rotate:
    trade_signals -> @

start: rotate""",
    "breakout_monitor.dsl": """# Breakout Monitor Template
# Monitors for price breakouts above/below key levels
# Template variables: symbol, high_level, low_level, timeframe, qty

breakout_high = "if {{ symbol }} price > {{ high_level }}: buy {{ symbol }} {{ qty }} MKT; say BREAKOUT HIGH at {{ symbol }}.price"
breakout_low = "if {{ symbol }} price < {{ low_level }}: sell {{ symbol }} {{ qty }} MKT; say BREAKOUT LOW at {{ symbol }}.price"

flow breakouts:
    breakout_high | breakout_low

start: breakouts""",
    "mean_reversion.dsl": """# Mean Reversion Template
# Monitors for oversold/overbought conditions and trades mean reversion
# Template variables: symbol, timeframe, rsi_high, rsi_low, qty

oversold = "if {{ symbol }} rsi {{ timeframe }} < {{ rsi_low }}: buy {{ symbol }} {{ qty }} LIM; say OVERSOLD ENTRY"
overbought = "if {{ symbol }} rsi {{ timeframe }} > {{ rsi_high }}: sell {{ symbol }} {{ qty }} LIM; say OVERBOUGHT ENTRY"

flow mean_revert:
    oversold | overbought

start: mean_revert""",
    "multi_symbol_flipper.dsl": """# Multi-Symbol Algo Flipper Template
# Demonstrates Jinja2's power with loops and conditionals
# Template variables: symbols (list), base_timeframe, base_algo, default_qty

{% for symbol_config in symbols %}
# {{ symbol_config.name }} Configuration
{% set algo_symbol = symbol_config.name %}
{% set timeframe = symbol_config.get('timeframe', base_timeframe) %}
{% set algo = symbol_config.get('algo', base_algo) %}
{% set qty = symbol_config.get('qty', default_qty) %}
{% set watch_symbol = symbol_config.watch_symbol %}
{% set trade_symbol = symbol_config.trade_symbol %}
{% set evict_symbol = symbol_config.evict_symbol %}

{{ algo_symbol }}_check_short = "if {{ watch_symbol }} { {{ algo_symbol }}.{{ timeframe }}.algos.{{ algo }}.stopped is True and ({{ algo_symbol }}.{{ timeframe }}.algos.{{ algo }}.be is 'short') }: cancel {{ evict_symbol }}*; evict {{ evict_symbol }}* -1 0 MKT; buy {{ trade_symbol }} -{{ qty }} LIM @ live + 4 - 7"

{{ algo_symbol }}_check_long = "if {{ watch_symbol }} { {{ algo_symbol }}.{{ timeframe }}.algos.{{ algo }}.stopped is True and ({{ algo_symbol }}.{{ timeframe }}.algos.{{ algo }}.be is 'long') }: cancel {{ evict_symbol }}*; evict {{ evict_symbol }}* -1 0 MKT; buy {{ trade_symbol }} {{ qty }} LIM @ live + 4 - 7"

{{ algo_symbol }}_unstopped = "if {{ watch_symbol }} { {{ algo_symbol }}.{{ timeframe }}.algos.{{ algo }}.stopped is False }: say {{ algo_symbol }} algo reset"

flow {{ algo_symbol }}_primary:
    {{ algo_symbol }}_check_short | {{ algo_symbol }}_check_long

flow {{ algo_symbol }}_entrypoint:
    {{ algo_symbol }}_primary -> {{ algo_symbol }}_unstopped -> @

{% endfor %}

start: {% for symbol_config in symbols %}{{ symbol_config.name }}_entrypoint{% if not loop.last %}, {% endif %}{% endfor %}""",
}


def create_template_args_for_algo_flipper(
    algo_symbol: str,
    watch_symbol: str,
    trade_symbol: str,
    evict_symbol: str,
    timeframe: int,
    algo: str,
    qty: int,
    offset: float | None = None,
    profit_pts: int | None = None,
    loss_pts: int | None = None,
) -> dict[str, Any]:
    """Helper function to create properly formatted template arguments for algo flipper.

    Args:
        algo_symbol: Symbol used in algo feed namespace (e.g., 'MNQ')
        watch_symbol: Symbol to watch for triggers (e.g., '/NQM5')
        trade_symbol: Symbol used for trading orders (e.g., '/MNQ')
        evict_symbol: Symbol used for cancellations (e.g., 'MNQ')
        timeframe: Algorithm timeframe in seconds
        algo: Algorithm name
        qty: Quantity to trade
        offset: Price offset from live price (None = omit from template)
        profit_pts: Profit target distance in points (None = omit from template)
        loss_pts: Loss limit distance in points (None = omit from template)

    Returns:
        Dictionary of template arguments ready for populate_template()
    """
    return {
        "algo_symbol": algo_symbol,
        "watch_symbol": watch_symbol,
        "trade_symbol": trade_symbol,
        "evict_symbol": evict_symbol,
        "timeframe": timeframe,
        "algo": algo,
        "qty": qty,
        "offset": offset,
        "profit_pts": profit_pts,
        "loss_pts": loss_pts,
    }


def create_symbol_config_for_flipper(
    algo_symbol: str,
    watch_symbol: str,
    trade_symbol: str,
    evict_symbol: str,
    timeframe: int | None = None,
    algo: str | None = None,
    qty: int | None = None,
) -> dict[str, str | int | None]:
    """Helper function to create symbol config for multi-symbol flipper template.

    Args:
        algo_symbol: Symbol used in algo feed namespace
        watch_symbol: Symbol to watch for triggers
        trade_symbol: Symbol used for trading orders
        evict_symbol: Symbol used for cancellations
        timeframe: Algorithm timeframe (uses template default if None)
        algo: Algorithm name (uses template default if None)
        qty: Quantity to trade (uses template default if None)

    Returns:
        Dictionary suitable for symbols list in multi_symbol_flipper template
    """
    config: dict[str, str | int | None] = {
        "name": algo_symbol,
        "watch_symbol": watch_symbol,
        "trade_symbol": trade_symbol,
        "evict_symbol": evict_symbol,
    }

    if timeframe is not None:
        config["timeframe"] = timeframe
    if algo is not None:
        config["algo"] = algo
    if qty is not None:
        config["qty"] = qty

    return config
