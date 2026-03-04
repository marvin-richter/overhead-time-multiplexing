from collections import defaultdict
from typing import Literal, Dict, Tuple
import time


from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import Layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import Qubit

from .switch_gate import Switch2Gate, SwitchGate
import logging

log = logging.getLogger(__name__)


class SerializeGatesSwitchPass(TransformationPass):
    """Transformation pass that models controller time-multiplexing overhead.

    In a time-multiplexed control architecture, a single controller drives multiple
    qubits but can only address one at a time. When consecutive single-qubit gates
    target different qubits within the same controller group, a switching delay is
    incurred. This pass makes that overhead explicit by inserting:

    - A :class:`Switch2Gate` between the two qubits involved in a controller switch,
      capturing the dependency and the switching event.
    - A :class:`SwitchGate` (identity with switching duration) on the newly addressed
      qubit, representing the delay experienced while the controller switches over.
      When ``delay_check=True``, the delay gate is only inserted if timing analysis
      shows it actually extends the circuit duration.

    After the pass runs, the property set contains the following metrics:
    ``serial_switch2_gates``, ``serial_switch_del_gates``, ``serial_wall_time``,
    ``serial_start_time``, and ``serial_max_duration``.
    """

    def __init__(
        self,
        qubit_to_group: Dict[int, int],
        virtual_rz: bool = False,
        topord_method: Literal[
            "default", "prio_two", "prio_two_controller_aware", "prio_two_fw"
        ] = "prio_two",
        delay_check: bool = True,
        debug_labels: bool = False,
        operation_durations=None,
        switch_duration: float = 2e-9,
    ):
        """Initialize the serialization pass.

        Args:
            qubit_to_group: Mapping from physical qubit index to controller group ID,
                as returned by :attr:`ControllerLayout.qubit_to_group`.
            virtual_rz: If ``True``, RZ gates are treated as virtual (zero duration)
                and do not trigger a controller switch. Corresponds to hardware that
                implements RZ via frame changes. Default: ``False``.
            topord_method: Strategy for ordering single-qubit gates in the topological
                traversal to minimise the number of controller switches:

                - ``"default"``: Qiskit's default topological order.
                - ``"prio_two"``: Single-qubit gates close to a two-qubit gate are
                  scheduled first (recommended).
                - ``"prio_two_controller_aware"``: Like ``prio_two`` but also
                  accounts for qubit index to break ties deterministically.
                - ``"prio_two_fw"``: Forward-pass variant of ``prio_two``.

            delay_check: If ``True``, a timing analysis is performed before inserting
                each :class:`SwitchGate` to determine whether the switching delay
                actually prolongs the circuit (it may already be hidden behind a
                two-qubit gate duration). If ``False``, delay gates are always
                inserted. Default: ``True``.
            debug_labels: If ``True``, each inserted switch gate is labelled with a
                sequential integer for circuit inspection and debugging.
                Default: ``False``.
            operation_durations: Gate duration table used for timing-aware delay
                insertion, obtained via ``target.durations()``. If ``None``, duration
                tracking is disabled and delay gates are always inserted.
            switch_duration: Duration of a single controller switch in seconds.
                Default: ``2e-9`` (2 nanoseconds).
        """
        self.qubit_to_group = qubit_to_group
        self.ignore_rz = virtual_rz
        self.topord_method = topord_method
        self.delay_check = delay_check
        self.debug_labels = debug_labels
        self.switch_duration = switch_duration
        self.durations = operation_durations

        super().__init__()

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the serialization pass on the DAG.

        Args:
            dag: Input DAG representing the transpiled and routed circuit.

        Returns:
            A new DAG with :class:`Switch2Gate` and :class:`SwitchGate` nodes
            inserted at every controller-switch point.
        """
        # Start timing
        start_time = time.time()

        # Create a new DAG for the output circuit
        new_dag = DAGCircuit()

        # Add all qubits and clbits from the original DAG
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        # Track the last qubit used by each controller
        last_controller_qubit = {
            controller: None for controller in set(self.qubit_to_group.values())
        }

        # Get layout from property set
        layout = self.property_set.get("layout", None)
        if layout is None:
            # If no layout is provided, create a trivial layout
            from qiskit.transpiler.layout import Layout

            layout = Layout.generate_trivial_layout(*dag.qregs.values())

        # if there is a qreg in the layout that is not in the dag (e.g. ancilla), create
        # a new layout to use here with a combined qreg
        layout = self._translate_layout_if_needed(layout, dag)

        def sort_key_priority(dag):
            topological_sort_key = {}
            qubit_to_last_two_qubit_gate_layer_index = defaultdict(lambda: 0)
            for layer_idx, layer in enumerate(reversed(list(dag.multigraph_layers()))):
                for node in layer:
                    if isinstance(node, DAGOpNode):
                        if node.op.num_qubits == 1:
                            # give high priority (low sort key) by subtracting the dist.
                            # to next two-qubit gate
                            topological_sort_key[node._node_id] = str(
                                99999997 - qubit_to_last_two_qubit_gate_layer_index[node.qargs[0]]
                            )
                        elif node.op.num_qubits == 2:
                            topological_sort_key[node._node_id] = (
                                99999998  # Two-qubit gates come after single-qubit gates
                            )
                            qubit_to_last_two_qubit_gate_layer_index[node.qargs[0]] = layer_idx
                            qubit_to_last_two_qubit_gate_layer_index[node.qargs[1]] = layer_idx
                    else:
                        topological_sort_key[node._node_id] = 0

            # everything else (no single or two-qubit gate, comes last)
            return lambda node: str(topological_sort_key.get(node._node_id, 99999999))

        def sort_key_priority_controller_aware(dag):
            topological_sort_key = {}
            qubit_to_last_two_qubit_gate_layer_index = defaultdict(lambda: 0)

            for layer_idx, layer in enumerate(reversed(list(dag.multigraph_layers()))):
                for node in layer:
                    if isinstance(node, DAGOpNode):
                        if node.op.num_qubits == 1:
                            # give high priority (low sort key) by subtracting the dist.
                            # to next two-qubit gate
                            topological_sort_key[node._node_id] = str(
                                999_999_97
                                - qubit_to_last_two_qubit_gate_layer_index[node.qargs[0]]
                                * dag.num_qubits()
                                * (node.qargs[0]._index + 1)
                                - dag.num_qubits() * (node.qargs[0]._index + 1)
                            )
                        elif node.op.num_qubits == 2:
                            topological_sort_key[node._node_id] = 999_999_96
                            qubit_to_last_two_qubit_gate_layer_index[node.qargs[0]] = layer_idx
                            qubit_to_last_two_qubit_gate_layer_index[node.qargs[1]] = layer_idx
                    else:
                        topological_sort_key[node._node_id] = 0

            # everything else (no single or two-qubit gate, comes last)
            return lambda node: str(topological_sort_key.get(node._node_id, 999_999_99))

        def sort_key_priority_fw(dag):
            topological_sort_key = {}
            qubit_node_collection: dict[Qubit, list[int]] = {qubit: [] for qubit in dag.qubits}

            for layer_idx, layer in enumerate(dag.multigraph_layers()):
                for node in layer:
                    if isinstance(node, DAGOpNode):
                        if node.op.num_qubits == 2:
                            # get the nodes from the collection and assign the current (2q) layer index
                            # to them as sort key, with qubit index as tiebreaker
                            nodes = (
                                qubit_node_collection[node.qargs[0]]
                                + qubit_node_collection[node.qargs[1]]
                            )
                            for idx in nodes:
                                # Use layer_idx as primary sort key and qubit index as tiebreaker
                                qubit_index = dag.node(idx).qargs[0]._index
                                topological_sort_key[idx] = (layer_idx, qubit_index)

                            # clear the collection for the next layer
                            qubit_node_collection[node.qargs[0]] = []
                            qubit_node_collection[node.qargs[1]] = []
                        elif node.op.num_qubits == 1:
                            # add node to collection
                            qubit_node_collection[node.qargs[0]].append(node._node_id)

            def get_sort_key(node) -> str:
                if node._node_id in topological_sort_key:
                    layer_idx, qubit_idx = topological_sort_key[node._node_id]
                    # Workaround by prefixing with "z"
                    # Format: layer (5 digits) + qubit index (5 digits) for proper lexicographic ordering
                    return f"z{layer_idx:05d}{qubit_idx:05d}"
                return "z9999999999"  # comes last

            return get_sort_key

        if self.topord_method == "default":
            sort_key_fn = None
        elif self.topord_method == "prio_two":
            sort_key_fn = sort_key_priority(dag)
        elif self.topord_method == "prio_two_controller_aware":
            sort_key_fn = sort_key_priority_controller_aware(dag)
        elif self.topord_method == "prio_two_fw":
            sort_key_fn = sort_key_priority_fw(dag)
        else:
            raise ValueError(f"Unknown topological ordering method: {self.topord_method}")

        skip_ops = {"measure", "barrier"}

        # only used for debugging purposes
        debug_label_count = 0

        node_to_circuit_duration = defaultdict(float)

        # Main processing loop
        for gate in dag.topological_op_nodes(
            # to make sure every node is returning a proper sort key, problems with DAGInNode else
            key=sort_key_fn,
        ):
            # Skip measurement and barrier operations - they don't require controller switching
            if gate.op.name in skip_ops:
                # working_dag.remove_op_node(gate)
                added_node = new_dag.apply_operation_back(gate.op, gate.qargs, gate.cargs)
                # update node durations
                if self.durations:
                    node_to_circuit_duration[added_node._node_id] = max(
                        node_to_circuit_duration[pred._node_id]
                        for pred in new_dag.predecessors(added_node)
                    )
                continue

            if gate.op.num_qubits == 1:
                if self.ignore_rz and gate.op.name == "rz":
                    # If virtual RZ gate, check if there is a neighboring gate to merge with
                    mergeable = False
                    neighbors = list(dag.op_successors(gate)) + list(dag.op_predecessors(gate))
                    for neighboring_gate in neighbors:
                        if neighboring_gate.op.name != "rz":
                            mergeable = True
                            break
                    if mergeable:
                        # If there is a neighboring gate, apply the RZ gate
                        added_node = new_dag.apply_operation_back(gate.op, gate.qargs, gate.cargs)
                        # update node durations
                        if self.durations:
                            node_to_circuit_duration[added_node._node_id] = max(
                                node_to_circuit_duration[pred._node_id]
                                for pred in new_dag.predecessors(added_node)
                            )
                        continue
                    else:
                        # treat as normal single-qubit gate
                        pass

                gate_qubit = gate.qargs[0]
                physical_qubit = layout[gate_qubit]
                gate_controller = self.qubit_to_group[physical_qubit]

                # Check if we the last ph qubit of the switch is different
                switch_change = (
                    last_controller_qubit[gate_controller] is not None
                    and last_controller_qubit[gate_controller] != gate_qubit
                )

                if switch_change:
                    # Insert a Switch2Gate to model the dependency and delay
                    debug_label = f"{debug_label_count}" if self.debug_labels else None
                    debug_label_count += 1
                    added_switch2_node = new_dag.apply_operation_back(
                        Switch2Gate(label=debug_label),
                        [last_controller_qubit[gate_controller], gate_qubit],
                    )

                    if self.durations:
                        # If we have durations, check if the delay gate is needed
                        op_predecessors = list(new_dag.op_predecessors(added_switch2_node))

                        # Update the circuit duration for the switch gate
                        node_to_circuit_duration[added_switch2_node._node_id] = max(
                            node_to_circuit_duration[pred._node_id] for pred in op_predecessors
                        )

                        if self.delay_check:
                            delay_needed, max_gate_duration = self.check_delay_gate_needed(
                                new_dag, op_predecessors, node_to_circuit_duration
                            )
                        else:
                            delay_needed = True
                            max_gate_duration = (
                                node_to_circuit_duration[added_switch2_node._node_id]
                                + self.switch_duration
                            )

                        if delay_needed:
                            added_delay_gate = new_dag.apply_operation_back(
                                SwitchGate(label=debug_label),
                                [gate_qubit],
                            )
                            # update the circuit duration for the switch gate
                            node_to_circuit_duration[added_delay_gate._node_id] = (
                                max_gate_duration + self.switch_duration
                            )

                        else:
                            pass
                    else:
                        added_delay_gate = new_dag.apply_operation_back(
                            SwitchGate(label=debug_label),
                            [gate_qubit],
                        )

                # Update last controller qubit
                last_controller_qubit[gate_controller] = gate_qubit
            else:
                # For two-qubit gates, we don't need to insert switch gates
                pass

            # Apply the gate
            added_node = new_dag.apply_operation_back(gate.op, gate.qargs, gate.cargs)

            # update node durations
            if self.durations:
                qubit_ids = [qubit._index for qubit in gate.qargs]
                gate_duration = self.durations.get(inst=gate.op.name, qubits=qubit_ids, unit="s")
                max_duration_upto = max(
                    node_to_circuit_duration[pred_node._node_id]
                    for pred_node in new_dag.predecessors(added_node)
                )
                node_to_circuit_duration[added_node._node_id] = max_duration_upto + gate_duration

        # Store metrics in property set
        self._store_metrics(
            new_dag, start_time, max_duration=max(node_to_circuit_duration.values(), default=0.0)
        )

        return new_dag

    def check_delay_gate_needed(
        self, dag, pred_op_nodes, node_to_circuit_duration
    ) -> Tuple[bool, float]:
        """Determine whether a :class:`SwitchGate` delay is required at a switch point.

        A delay is only needed when the switching time cannot be hidden behind an
        already-scheduled two-qubit gate. When all predecessors of the switch are
        single-qubit gates the delay is always required. When a two-qubit gate is
        among the predecessors, the delay is required only if the single-qubit path
        plus the switch duration exceeds the two-qubit gate duration.

        Args:
            dag: The DAG being constructed, used to traverse predecessors.
            pred_op_nodes: Predecessor operation nodes of the :class:`Switch2Gate`
                that was just inserted.
            node_to_circuit_duration: Running map from node ID to its cumulative
                circuit time (in seconds) up to and including that node.

        Returns:
            A ``(needed, max_duration)`` tuple where ``needed`` is ``True`` if a
            :class:`SwitchGate` should be inserted, and ``max_duration`` is the
            cumulative circuit time at the switch point (used to set the duration
            of the inserted gate). Returns ``(False, None)`` when no delay is needed.
        """

        def get_non_rz_predecessor(rz_node):
            """Walk backwards through a chain of RZ gates and return the first non-RZ predecessor.

            RZ gates are single-qubit and form linear chains, so each RZ has at most one
            predecessor. Raises ``ValueError`` if multiple predecessors are found, as that
            would indicate an unexpected circuit structure.
            """
            rz_pred = list(dag.op_predecessors(rz_node))
            if len(rz_pred) == 0:
                return rz_node
            elif len(rz_pred) == 1:
                pred = rz_pred[0]
                if pred.op.name == "rz":
                    return get_non_rz_predecessor(pred)
                else:
                    return pred
            else:
                raise ValueError(
                    f"Multiple predecessors found for RZ gate: {rz_pred}. Cannot determine non-RZ predecessor."
                )

        # 1. if virtual RZ, replace all the RZ predecessors with the first non-RZ predecessor
        if self.ignore_rz:
            pred_op_nodes = [
                get_non_rz_predecessor(pred) if pred.op.name == "rz" else pred
                for pred in pred_op_nodes
            ]

        if all(pred.op.num_qubits == 1 for pred in pred_op_nodes):
            # if all predecessors are single-qubit gates, we need a delay
            return True, max(node_to_circuit_duration[pred._node_id] for pred in pred_op_nodes)
        elif any(pred.op.num_qubits == 2 for pred in pred_op_nodes):
            # If we have a two-qubit gate, we need to check the circuit duration
            # of the single-qubit gate and compare it to the two-qubit gate
            # to see if the switch gate can fit in
            single_qubit_gate_circuit_duration = max(
                [
                    node_to_circuit_duration.get(pred._node_id, 0)
                    for pred in pred_op_nodes
                    if pred.op.num_qubits == 1
                ],
                default=0.0,
            )
            two_qubit_gate_circuit_duration = max(
                [
                    node_to_circuit_duration.get(pred._node_id, 0)
                    for pred in pred_op_nodes
                    if pred.op.num_qubits == 2
                ],
                default=0.0,
            )
            if (
                single_qubit_gate_circuit_duration + self.switch_duration
                >= two_qubit_gate_circuit_duration
            ):
                # If the single-qubit gate duration + switch duration is longer than
                # the two-qubit gate, we need to insert a switch gate
                return True, single_qubit_gate_circuit_duration
            else:
                # If the two-qubit gate is longer, we don't need a switch gate
                return False, None
        else:
            # If there are only two-qubit gates, we don't need a switch gate
            return False, None

    def _store_metrics(self, new_dag: DAGCircuit, start_time: float, max_duration: float):
        """Store serialization metrics in property set."""
        # Calculate wall time
        wall_time = time.time() - start_time

        # Count operations in the original and new DAGs
        new_ops_count = new_dag.count_ops()

        # Count switch gates
        switch2_count = new_ops_count.get("sw2", 0)
        switch_del_count = new_ops_count.get("SDel", 0)

        # Store in property set for later access
        self.property_set["serial_switch2_gates"] = switch2_count
        self.property_set["serial_switch_del_gates"] = switch_del_count
        self.property_set["serial_wall_time"] = wall_time
        self.property_set["serial_start_time"] = start_time
        self.property_set["serial_max_duration"] = max_duration

    @staticmethod
    def _translate_layout_if_needed(layout: Layout, dag):
        """Reconcile the pass-manager layout with the DAG's quantum registers.

        When the preset pass manager adds ancilla qubits (e.g. for routing), the layout
        may reference registers that are not present in the final DAG. This method
        constructs a corrected :class:`~qiskit.transpiler.Layout` whose virtual qubits
        are drawn exclusively from the registers that exist in ``dag``.

        Args:
            layout: Layout from the property set, potentially containing ancilla registers.
            dag: The DAG being transformed.

        Returns:
            The original ``layout`` if it already matches the DAG registers, otherwise
            a new :class:`~qiskit.transpiler.Layout` remapped to the DAG's registers.
        """
        initial_layout = layout

        layout_qregs = {}
        for qubit_index in range(len(initial_layout)):
            qubit = initial_layout[qubit_index]
            if qubit._register.name not in layout_qregs:
                layout_qregs[qubit._register.name] = qubit._register

        if dag.qregs != layout_qregs:
            log.warning(
                f"The layout qregs does not match the DAG qregs. {layout_qregs} != {dag.qregs}"
            )

            # create a new translation from the layout qubits to the dag qregs qubits
            translation = {}
            unassigned_idx = 0
            num_qubits_with_correspondence = sum(
                [qreg.size for qreg in layout_qregs.values() if qreg.name in dag.qregs]
            )

            assert len(dag.qregs) == 1, "DAG should have only one qreg for this pass"

            for qreg_name, qreg in layout_qregs.items():
                for qubit in qreg:
                    if qreg_name in dag.qregs:
                        dag_qreg = dag.qregs[qreg_name]
                        if qubit._index < dag_qreg.size:
                            translation[qubit] = dag_qreg[qubit._index]
                        else:
                            log.error(f"Warning: {qreg_name}[{qubit._index}] not in DAG qregs")
                    else:
                        # use the dag qreg
                        translation[qubit] = dag.qubits[
                            unassigned_idx + num_qubits_with_correspondence
                        ]
                        unassigned_idx += 1

            # create a new layout with the translation
            new_layout_dict = {}
            for physical_index in range(len(initial_layout)):
                qubit = initial_layout[physical_index]
                new_layout_dict[physical_index] = translation[qubit]

            new_layout = Layout(input_dict=new_layout_dict)

            return new_layout
        else:
            # If the layout matches the DAG qregs, return the original layout
            return layout
