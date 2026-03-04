from overhead_time_multiplexing.targets import construct_chalmers_target, construct_ibm_target
from qiskit.transpiler import CouplingMap, Target
from overhead_time_multiplexing.targets import (
    chalmers_single_qubit_gates as CHALMERS_SINGLE_QUBIT_GATES,
    chalmers_two_qubit_gates as CHALMERS_TWO_QUBIT_GATES,
)


class TestChalmersTargetConstruction:
    def test_construct_chalmers_target(self):
        target = construct_chalmers_target(
            num_rows=5, num_cols=5, full_coupling=False, with_switch=True
        )
        assert target.num_qubits == 25

        # Check that switch gates are present
        assert "SDel" in target.operation_names
        assert "sw2" in target.operation_names

        assert isinstance(target, Target)

    def test_construct_chalmers_target_coupling(self):
        target = construct_chalmers_target(
            num_rows=6, num_cols=6, full_coupling=False, with_switch=False
        )
        assert target.num_qubits == 36

        # Check that the coupling map is a grid
        expected_edges = set(CouplingMap.from_grid(6, 6).get_edges())
        actual_edges = set(target.build_coupling_map().get_edges())
        assert expected_edges == actual_edges

        # Check that switch gates are not present
        assert "SDel" not in target.operation_names
        assert "sw2" not in target.operation_names

    def test_construct_chalmers_target_full_coupling(self):
        target = construct_chalmers_target(
            num_rows=5, num_cols=5, full_coupling=True, with_switch=False
        )
        assert target.num_qubits == 25

        # Check that switch gates are not present
        assert "SDel" not in target.operation_names
        assert "sw2" not in target.operation_names

        # Check that the coupling map is fully connected
        cm = target.build_coupling_map()
        expected_edges = set()
        for i in range(25):
            for j in range(25):
                if i != j:
                    expected_edges.add((i, j))
        actual_edges = set(cm.get_edges())
        assert expected_edges == actual_edges

    def test_construct_chalmers_target_not_virtual_rz(self):
        target = construct_chalmers_target(
            num_rows=5,
            num_cols=5,
            full_coupling=False,
            with_switch=False,
            virtual_rz=False,
        )
        assert target.num_qubits == 25

        # Check that the RZ duration is not zero (i.e. virtual RZ is disabled)
        rz_durations = [
            duration[0]
            for specifier, duration in target.durations().duration_by_name_qubits.items()
            if specifier[0] == "rz"
        ]
        assert all(d > 0 for d in rz_durations)

    def test_construct_chalmers_target_virtual_rz(self):
        target = construct_chalmers_target(
            num_rows=5,
            num_cols=5,
            full_coupling=False,
            with_switch=False,
            virtual_rz=True,
        )
        assert target.num_qubits == 25

        # Check that the RZ duration is not zero (i.e. virtual RZ is disabled)
        rz_durations = [
            duration[0]
            for specifier, duration in target.durations().duration_by_name_qubits.items()
            if specifier[0] == "rz"
        ]
        assert all(d == 0.0 for d in rz_durations)

    def test_construct_chalmers_target_switch_duration(self):
        switch_duration = 3.5
        target = construct_chalmers_target(
            num_rows=5,
            num_cols=5,
            full_coupling=False,
            with_switch=True,
            switch_duration=switch_duration,
        )

        # Check that the switch gate durations are set correctly
        for specifier, duration in target.durations().duration_by_name_qubits.items():
            if specifier[0] in ["SDel"]:
                assert duration[0] == switch_duration
            if specifier[0] in ["sw2"]:
                assert duration[0] == 0.0

    def test_construct_chalmers_target_durations(self):
        single_qubit_duration = 15.0
        two_qubit_duration = 150.0
        target = construct_chalmers_target(
            num_rows=5,
            num_cols=5,
            full_coupling=False,
            with_switch=False,
            single_qubit_duration=single_qubit_duration,
            two_qubit_duration=two_qubit_duration,
            virtual_rz=False,
        )

        # Check that single-qubit gate durations are set correctly
        for specifier, duration in target.durations().duration_by_name_qubits.items():
            if specifier[0] in [name for _, name in CHALMERS_SINGLE_QUBIT_GATES]:
                assert duration[0] == single_qubit_duration
            if specifier[0] in [name for _, name in CHALMERS_TWO_QUBIT_GATES]:
                assert duration[0] == two_qubit_duration


class TestIBMTargetConstruction:
    def test_construct_ibm_target(self):
        target = construct_ibm_target(
            fake_label="fake_brisbane", full_coupling=False, with_switch=True
        )
        assert target.num_qubits == 127

        # Check that switch gates are present
        assert "SDel" in target.operation_names
        assert "sw2" in target.operation_names

        assert isinstance(target, Target)

    def test_construct_ibm_target_full_coupling(self):
        target = construct_ibm_target(
            fake_label="fake_brisbane", full_coupling=True, with_switch=False
        )
        assert target.num_qubits == 127

        # Check that switch gates are not present
        assert "SDel" not in target.operation_names
        assert "sw2" not in target.operation_names

        # Check that the coupling map is fully connected
        cm = target.build_coupling_map()
        expected_edges = set()
        for i in range(127):
            for j in range(127):
                if i != j:
                    expected_edges.add((i, j))
        actual_edges = set(cm.get_edges())
        assert expected_edges == actual_edges

    def test_construct_ibm_target_switch_duration(self):
        switch_duration = 4.0
        target = construct_ibm_target(
            fake_label="fake_brisbane",
            full_coupling=False,
            with_switch=True,
            switch_duration=switch_duration,
        )

        # Check that the switch gate durations are set correctly
        for specifier, duration in target.durations().duration_by_name_qubits.items():
            if specifier[0] in ["SDel"]:
                assert duration[0] == switch_duration
            if specifier[0] in ["sw2"]:
                assert duration[0] == 0.0

    def test_construct_ibm_target_no_switch(self):
        target = construct_ibm_target(
            fake_label="fake_brisbane", full_coupling=False, with_switch=False
        )

        # Check that switch gates are not present
        assert "SDel" not in target.operation_names
        assert "sw2" not in target.operation_names
