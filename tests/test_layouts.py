import pytest

from overhead_time_multiplexing import generate_layout
from overhead_time_multiplexing.targets import construct_chalmers_target, construct_ibm_target


LAYOUT_TYPES = ("trivial", "dispersed", "clustered", "random")

TARGET_CASES = (
    pytest.param(
        {
            "name": "chalmers",
            "constructor_args": (4, 3),
            "expected_num_qubits": 12,
            "hardware_label": "chalmers_12",
            "qpg": 4,
        },
        id="chalmers-grid-4x3",
    ),
    pytest.param(
        {
            "name": "brisbane",
            "constructor_args": (),
            "expected_num_qubits": 127,
            "hardware_label": "brisbane",
            "qpg": 16,
        },
        id="ibm-brisbane",
    ),
)


@pytest.fixture(params=TARGET_CASES)
def target_setup(request):
    return request.param


@pytest.fixture
def target(target_setup):
    constructors = {
        "chalmers": construct_chalmers_target,
        "brisbane": construct_ibm_target,
    }

    constructor = constructors[target_setup["name"]]
    backend_target = constructor(*target_setup["constructor_args"])
    assert backend_target.num_qubits == target_setup["expected_num_qubits"]

    return {
        "backend_target": backend_target,
        "hardware_label": target_setup["hardware_label"],
        "qpg": target_setup["qpg"],
    }


class TestLayoutGeneration:
    @pytest.mark.parametrize("layout_type", LAYOUT_TYPES)
    def test_layout_generation(self, target, layout_type):
        layout = generate_layout(
            target=target["backend_target"],
            layout_type=layout_type,
            qpg=target["qpg"],
            hardware_label=target["hardware_label"],
        )

        assert layout is not None
        assert layout.layout_type == layout_type

    def test_trivial_generation(self, target):
        layout = generate_layout(
            target=target["backend_target"],
            layout_type="trivial",
            qpg=target["qpg"],
            hardware_label=target["hardware_label"],
        )

        assert layout is not None
        assert layout.layout_type == "trivial"

        # check that sum over all groups equals total qubits
        total_qubits = target["backend_target"].num_qubits
        group_qubits_sum = sum(layout.group_sizes.values())
        assert group_qubits_sum == total_qubits

        # check that qubit to group, when sorted by qubit index, is a contiguous sequence of group indices
        qubit_to_group = layout.qubit_to_group
        sorted_qubits = sorted(qubit_to_group.keys())
        sorted_groups = [qubit_to_group[q] for q in sorted_qubits]
        # assert that only group indix stays the same or increases by 1
        for i in range(1, len(sorted_groups)):
            assert (
                sorted_groups[i] == sorted_groups[i - 1]
                or sorted_groups[i] == sorted_groups[i - 1] + 1
            )

        # assert that each group of qubits is contiguous in the qubit index space
        groups = layout.group_qubits.values()
        for group in groups:
            sorted_group = sorted(group)
            for i in range(1, len(sorted_group)):
                assert sorted_group[i] == sorted_group[i - 1] + 1

    def test_random_generation(self, target):
        layout = generate_layout(
            target=target["backend_target"],
            layout_type="random",
            qpg=target["qpg"],
            hardware_label=target["hardware_label"],
        )

        assert layout is not None
        assert layout.layout_type == "random"

        # check that sum over all groups equals total qubits
        total_qubits = target["backend_target"].num_qubits
        group_qubits_sum = sum(layout.group_sizes.values())
        assert group_qubits_sum == total_qubits

        # check that qubit to group mapping is valid
        qubit_to_group = layout.qubit_to_group
        for qubit, group in qubit_to_group.items():
            assert 0 <= qubit < total_qubits
            assert 0 <= group < len(layout.group_sizes)

        # check that each group has at least one qubit
        group_qubits = layout.group_qubits
        for group, qubits in group_qubits.items():
            assert len(qubits) > 0

        # check that qubits in each group are unique
        all_qubits = set()
        for qubits in group_qubits.values():
            for qubit in qubits:
                assert qubit not in all_qubits
                all_qubits.add(qubit)

    def test_clustered_generation(self, target):
        layout = generate_layout(
            target=target["backend_target"],
            layout_type="clustered",
            qpg=target["qpg"],
            hardware_label=target["hardware_label"],
        )

        assert layout is not None
        assert layout.layout_type == "clustered"

        # check that sum over all groups equals total qubits
        total_qubits = target["backend_target"].num_qubits
        group_qubits_sum = sum(layout.group_sizes.values())
        assert group_qubits_sum == total_qubits

        # check that qubit to group mapping is valid
        qubit_to_group = layout.qubit_to_group
        for qubit, group in qubit_to_group.items():
            assert 0 <= qubit < total_qubits
            assert 0 <= group < len(layout.group_sizes)

        # check that each group has at least one qubit
        group_qubits = layout.group_qubits
        for group, qubits in group_qubits.items():
            assert len(qubits) > 0

        # check that qubits in each group are unique
        all_qubits = set()
        for qubits in group_qubits.values():
            for qubit in qubits:
                assert qubit not in all_qubits
                all_qubits.add(qubit)

    def test_dispersed_generation(self, target):
        layout = generate_layout(
            target=target["backend_target"],
            layout_type="dispersed",
            qpg=target["qpg"],
            hardware_label=target["hardware_label"],
        )

        assert layout is not None
        assert layout.layout_type == "dispersed"

        # check that sum over all groups equals total qubits
        total_qubits = target["backend_target"].num_qubits
        group_qubits_sum = sum(layout.group_sizes.values())
        assert group_qubits_sum == total_qubits

        # check that qubit to group mapping is valid
        qubit_to_group = layout.qubit_to_group
        for qubit, group in qubit_to_group.items():
            assert 0 <= qubit < total_qubits
            assert 0 <= group < len(layout.group_sizes)

        # check that each group has at least one qubit
        group_qubits = layout.group_qubits
        for group, qubits in group_qubits.items():
            assert len(qubits) > 0

        # check that qubits in each group are unique
        all_qubits = set()
        for qubits in group_qubits.values():
            for qubit in qubits:
                assert qubit not in all_qubits
                all_qubits.add(qubit)
