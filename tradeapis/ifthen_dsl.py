import graphlib
from collections import defaultdict
from dataclasses import dataclass, field

from lark import Lark, Transformer

from .ifthen import IfThenRuntime, PredicateId

IFTHEN_GRAMMAR = r"""
    ?start: program

    program: (statement)* start_statement

    ?statement: assignment
              | flow_def

    assignment: IDENTIFIER "=" STRING

    flow_def: "flow" IDENTIFIER ":" flow_expr

    // Precedence: parentheses > peer > tree
    ?flow_expr: peer_expr
              | tree_expr

    // Tree has lower precedence - requires parentheses to mix with peers
    tree_expr: peer_expr "->" flow_expr

    // Peer has higher precedence
    ?peer_expr: atom_expr ("|" atom_expr)+
              | atom_expr

    ?atom_expr: IDENTIFIER  -> identifier_ref
              | "@"         -> self_ref
              | "(" flow_expr ")"

    # 'start' statement have one or more ids (predicates or flow names) separated by spaces, commas, or pipes.
    start_statement: "start" ":" IDENTIFIER (("," | "|")? IDENTIFIER)*

    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
    STRING: /"([^"\\]|\\.)*"/

    %import common.WS
    %ignore WS
    %ignore /\#[^\n]*/
"""


@dataclass
class FlowAST:
    """Base class for flow AST nodes"""

    pass


@dataclass
class IdentifierRef(FlowAST):
    name: str


@dataclass
class SelfRef(FlowAST):
    pass


@dataclass
class TreeExpr(FlowAST):
    active: FlowAST
    waiting: FlowAST


@dataclass
class PeerExpr(FlowAST):
    peers: list[FlowAST]


@dataclass
class PredicateAssignment:
    name: str
    predicate: str


@dataclass
class FlowDefinition:
    name: str
    expr: FlowAST


class IfThenDSLTransformer(Transformer):
    """Transform Lark parse tree into AST"""

    def STRING(self, s):
        # Remove quotes and handle escape sequences
        return s[1:-1].encode().decode("unicode_escape")

    def IDENTIFIER(self, s):
        return str(s)

    def assignment(self, items):
        name, predicate = items
        return PredicateAssignment(name, predicate)

    def flow_def(self, items):
        name, expr = items
        return FlowDefinition(name, expr)

    def identifier_ref(self, items):
        return IdentifierRef(items[0])

    def self_ref(self, items):
        return SelfRef()

    def tree_expr(self, items):
        active, waiting = items
        return TreeExpr(active, waiting)

    def peer_expr(self, items):
        return PeerExpr(list(items))

    def start_statement(self, items):
        return ("start", items)

    def program(self, items):
        statements = []
        start_names = []

        for item in items:
            if isinstance(item, tuple) and item[0] == "start":
                start_names = item[1]
            else:
                statements.append(item)

        return statements, start_names


class DSLValidationError(Exception):
    """Custom exception for DSL validation errors"""

    pass


@dataclass(slots=True)
class IfThenDSLLoader:
    """Load DSL format into IfThenRuntime

    This is an advancement versus the original yaml-based control flow format with just more specific
    text layouts here.

    Some examples of usage:
    -------------
    # Example 1: Multi-stage peer groups
    entry_a = "if conditionA: action A"
    entry_b = "if conditionB: action B"
    entry_c = "if conditionC: action C"

    exit_a = "if exitCondA: exit A"
    exit_b = "if exitCondB: exit B"

    confirm = "if confirmed: say CONFIRMED"

    flow first_stage:
        entry_a | entry_b | entry_c

    flow second_stage:
        exit_a | exit_b

    flow full_flow:
        first_stage -> confirm -> second_stage -> @

    # Example 2: Deeply nested
    flow inner_peer:
        pred_a | pred_b

    flow middle_tree:
        pred_c -> inner_peer -> pred_d

    flow outer_peer:
        middle_tree | pred_e | pred_f

    flow top_tree:
        pred_g -> outer_peer -> @
    ------------------
    # Two flows that reference each other
    bull_signal = "if SPY.rsi > 70: say OVERBOUGHT"
    bear_signal = "if SPY.rsi < 30: say OVERSOLD"

    flow watch_bull:
        bull_signal -> watch_bear

    flow watch_bear:
        bear_signal -> watch_bull

    start: watch_bull
    ------------------
    # One cancels all
    stop_loss = "if SPY < 445: sell all; say STOPPED"
    take_profit = "if SPY > 455: sell all; say PROFIT"
    time_exit = "if time > 15:00: sell all; say TIMED OUT"

    flow exit_conditions:
        stop_loss | take_profit | time_exit

    start: exit_conditions
    -------------
    """

    ifthenRuntime: IfThenRuntime = field(default_factory=IfThenRuntime)
    transformer: IfThenDSLTransformer = field(default_factory=IfThenDSLTransformer)
    parser: Lark = field(init=False)

    def __post_init__(self):
        self.parser = Lark(
            IFTHEN_GRAMMAR,
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
        )

    def load(
        self, dsl_text: str, activate: bool = True
    ) -> tuple[int, set[PredicateId], set[PredicateId]]:
        """Parse DSL and load into runtime, returning (created_count, start_ids, all_ids)"""

        # Parse and transform
        try:
            tree = self.parser.parse(dsl_text)
        except Exception as e:
            # Add line numbers to error messages
            lines = dsl_text.split("\n")

            # Extract line/column from Lark error
            error_line = getattr(e, "line", 0)
            error_column = getattr(e, "column", 0)

            if error_line > 0:
                context = self._get_error_context(lines, error_line, error_column)
                raise DSLValidationError(
                    f"Syntax error at line {error_line}:\n{context}\n{e}"
                )
            raise DSLValidationError(f"Syntax error: {e}")

        statements, start_names = self.transformer.transform(tree)

        # Validate before processing
        self._validate_dsl(statements, start_names)

        # Separate predicates and flows
        predicates: dict[str, str] = {}
        flows: dict[str, FlowAST] = {}

        for stmt in statements:
            if isinstance(stmt, PredicateAssignment):
                predicates[stmt.name] = stmt.predicate
            elif isinstance(stmt, FlowDefinition):
                flows[stmt.name] = stmt.expr

        # Build dependency graph for topological sort
        dependencies = self._build_dependencies(predicates, flows)
        processing_order = list(graphlib.TopologicalSorter(dependencies).static_order())

        # Process in dependency order
        name_to_id: dict[str, PredicateId] = {}
        created_count = 0

        for name in processing_order:
            if name in predicates:
                # Create predicate
                pid = self.ifthenRuntime.parse(predicates[name])
                name_to_id[name] = pid
                created_count += 1
            elif name in flows:
                # Create flow (tree/peer structure)
                fid = self._create_flow(name, flows[name], name_to_id)
                if fid:
                    name_to_id[name] = fid
                    created_count += 1

        # Handle self-references and circular dependencies
        self._fix_self_references(flows, name_to_id)

        # Activate start flows
        start_ids = set()
        if activate and start_names:
            for name in start_names:
                if name in name_to_id:
                    self.ifthenRuntime.activate(name_to_id[name])
                    start_ids.add(name_to_id[name])

        return created_count, start_ids, set(name_to_id.values())

    def _get_error_context(self, lines: list[str], line_no: int, col_no: int):
        """Show error context with pointer"""
        if 0 < line_no <= len(lines):
            line = lines[line_no - 1]
            pointer = " " * (col_no - 1) + "^"
            return f"  {line}\n  {pointer}"
        return ""

    def _validate_dsl(self, statements: list, start_names: list[str]):
        """Comprehensive validation of DSL structure"""

        predicates = {}
        flows = {}
        all_names = set()

        # 1. Check for duplicate names
        for stmt in statements:
            name = None
            if isinstance(stmt, PredicateAssignment):
                name = stmt.name
                predicates[name] = stmt.predicate
            elif isinstance(stmt, FlowDefinition):
                name = stmt.name
                flows[name] = stmt.expr

            if name:
                if name in all_names:
                    raise DSLValidationError(f"Duplicate definition: '{name}'")
                all_names.add(name)

        # 2. Check all references exist
        def check_refs(expr: FlowAST, context: str):
            if isinstance(expr, IdentifierRef):
                if expr.name not in all_names:
                    raise DSLValidationError(
                        f"Undefined reference '{expr.name}' in {context}"
                    )
            elif isinstance(expr, TreeExpr):
                check_refs(expr.active, f"{context} (active)")
                check_refs(expr.waiting, f"{context} (waiting)")
            elif isinstance(expr, PeerExpr):
                if len(expr.peers) < 2:
                    raise DSLValidationError(
                        f"Peer group in {context} must have at least 2 members"
                    )
                for peer in expr.peers:
                    check_refs(peer, f"{context} (peer)")

        for name, expr in flows.items():
            check_refs(expr, f"flow '{name}'")

        # 3. Check start references
        for start_name in start_names:
            if start_name not in all_names:
                raise DSLValidationError(
                    f"Start reference '{start_name}' is not defined"
                )

        # 4. Check for circular dependencies (excluding self-references)
        self._check_circular_deps(flows)

        # 5. Validate predicate syntax
        for name, pred_text in predicates.items():
            try:
                # Use the runtime's parser to validate
                self.ifthenRuntime.ifthen.parse(pred_text)
            except Exception as e:
                raise DSLValidationError(f"Invalid predicate syntax in '{name}': {e}")

    def _check_circular_deps(self, flows: dict[str, FlowAST]):
        """Check for unintentional circular dependencies"""
        # Build adjacency list excluding self-references
        graph = defaultdict(set)

        def add_edges(name: str, expr: FlowAST):
            if isinstance(expr, IdentifierRef) and expr.name != name:
                graph[name].add(expr.name)
            elif isinstance(expr, TreeExpr):
                add_edges(name, expr.active)
                add_edges(name, expr.waiting)
            elif isinstance(expr, PeerExpr):
                for peer in expr.peers:
                    add_edges(name, peer)

        for name, expr in flows.items():
            add_edges(name, expr)

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in flows:
            if node not in visited:
                if has_cycle(node):
                    raise DSLValidationError(
                        f"Circular dependency detected involving '{node}'"
                    )

    def _build_dependencies(
        self, predicates: dict[str, str], flows: dict[str, FlowAST]
    ) -> dict[str, set[str]]:
        """Build dependency graph for topological sorting"""
        deps: dict[str, set[str]] = {name: set() for name in predicates}
        deps.update({name: set() for name in flows})

        def extract_deps(expr: FlowAST, exclude_self: bool = False) -> set[str]:
            """Extract all identifier dependencies from an expression"""
            if isinstance(expr, IdentifierRef):
                return {expr.name}

            if isinstance(expr, SelfRef):
                return set() if exclude_self else {":self"}

            if isinstance(expr, TreeExpr):
                # For trees, we need both active and waiting deps for initial creation
                # but exclude self-refs in waiting to avoid cycles
                active_deps = extract_deps(expr.active, exclude_self)
                waiting_deps = extract_deps(expr.waiting, exclude_self=True)
                return active_deps | waiting_deps

            if isinstance(expr, PeerExpr):
                deps = set()
                for peer in expr.peers:
                    deps |= extract_deps(peer, exclude_self)
                return deps

            return set()

        # Build dependency edges
        for name, expr in flows.items():
            flow_deps = extract_deps(expr, exclude_self=True)
            deps[name] = {d for d in flow_deps if d != ":self"}

        return deps

    def _create_flow(
        self, name: str, expr: FlowAST, name_to_id: dict[str, PredicateId]
    ) -> PredicateId | None:
        """Create a flow from an AST expression"""

        def resolve_expr(expr: FlowAST) -> PredicateId | None:
            """Resolve an expression to a predicate ID"""
            if isinstance(expr, IdentifierRef):
                return name_to_id.get(expr.name)
            elif isinstance(expr, SelfRef):
                # Return a placeholder that will be fixed later
                return ":self"
            elif isinstance(expr, TreeExpr):
                active_id = resolve_expr(expr.active)
                waiting_id = resolve_expr(expr.waiting)

                if active_id and waiting_id:
                    # Handle self-reference placeholder
                    if waiting_id == ":self":
                        # Create tree with temporary waiting
                        return self.ifthenRuntime.tree(active_id, [active_id])
                    else:
                        return self.ifthenRuntime.tree(active_id, [waiting_id])
                return None
            elif isinstance(expr, PeerExpr):
                peer_ids = []
                for peer in expr.peers:
                    pid = resolve_expr(peer)
                    if pid and pid != ":self":
                        peer_ids.append(pid)

                if len(peer_ids) >= 2:
                    return self.ifthenRuntime.peers(peer_ids)
                elif len(peer_ids) == 1:
                    # Single peer is just the predicate itself
                    return peer_ids[0]
                return None
            return None

        return resolve_expr(expr)

    def _fix_self_references(
        self, flows: dict[str, FlowAST], name_to_id: dict[str, PredicateId]
    ):
        """Fix self-references in trees after all flows are created"""

        def has_self_ref(expr: FlowAST) -> bool:
            """Check if expression contains self-reference"""
            if isinstance(expr, SelfRef):
                return True

            if isinstance(expr, TreeExpr):
                return has_self_ref(expr.active) or has_self_ref(expr.waiting)

            if isinstance(expr, PeerExpr):
                return any(has_self_ref(p) for p in expr.peers)

            return False

        def collect_self_refs(
            name: str, expr: FlowAST, current_id: PredicateId
        ) -> list[tuple[PredicateId, list[PredicateId]]]:
            """Collect all trees that need self-reference updates"""
            updates = []

            if isinstance(expr, TreeExpr) and isinstance(expr.waiting, SelfRef):
                # This tree needs its waiting list updated to include self
                updates.append((current_id, [current_id]))
            elif isinstance(expr, TreeExpr):
                # Check if we need to update nested trees
                if isinstance(expr.active, IdentifierRef):
                    active_id = name_to_id.get(expr.active.name)
                    if active_id and expr.active.name in flows:
                        updates.extend(
                            collect_self_refs(
                                expr.active.name, flows[expr.active.name], active_id
                            )
                        )

                if isinstance(expr.waiting, IdentifierRef):
                    waiting_id = name_to_id.get(expr.waiting.name)
                    if waiting_id and expr.waiting.name in flows:
                        updates.extend(
                            collect_self_refs(
                                expr.waiting.name, flows[expr.waiting.name], waiting_id
                            )
                        )

            return updates

        # Fix all self-references
        for name, expr in flows.items():
            if name in name_to_id and has_self_ref(expr):
                current_id = name_to_id[name]
                updates = collect_self_refs(name, expr, current_id)

                for tree_id, new_waiting in updates:
                    self.ifthenRuntime.treeReplaceChildren(tree_id, new_waiting)
