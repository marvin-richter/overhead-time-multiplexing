from overhead_time_multiplexing.experiments import load_experiments, flatten_config, worker
from pathlib import Path
import polars as pl
from overhead_time_multiplexing.layouts import get_balanced_group_sizes


class TestLoadExperiments:
    def test_random_grids(self):
        # Load and expand
        experiments = load_experiments(Path("configs/paper/fig2_5x5.yaml"))
        assert len(experiments) > 0, "No experiments generated from config"

        # Each experiment is a validated ExperimentConfig ready for workers
        for exp in experiments[:3]:
            print(f"  - {exp.exp_id}: {exp.hardware.id} k={exp.hardware.layout.k}")

        rows = []
        for exp in experiments:
            rows.append(flatten_config(exp))

        df = pl.DataFrame(rows)
        assert len(df) == len(experiments), (
            "Flattened DataFrame should have same number of rows as experiments"
        )

    def test_random_ibm(self):
        # Load and expand
        experiments = load_experiments(Path("configs/paper/fig17_random_brisbane.yaml"))
        assert len(experiments) > 0, "No experiments generated from config"

        # Each experiment is a validated ExperimentConfig ready for workers
        for exp in experiments[:3]:
            print(f"  - {exp.exp_id}: {exp.hardware.id} k={exp.hardware.layout.k}")

        rows = []
        for exp in experiments:
            rows.append(flatten_config(exp))

        df = pl.DataFrame(rows)
        assert len(df) == len(experiments), (
            "Flattened DataFrame should have same number of rows as experiments"
        )

        hw_kinds = df["hardware.kind"].unique()
        assert len(hw_kinds) == 1, f"Expected only one hardware kind, got {hw_kinds}"
        assert hw_kinds[0] == "ibm", f"Expected hardware kind 'ibm', got {hw_kinds[0]}"

        hw_id = df["hardware.id"].unique()[0]
        assert hw_id == "brisbane", f"Unexpected hardware id, expected 'brisbane', got {hw_id}"

    def test_mqt(self):
        # Load and expand
        experiments = load_experiments(Path("configs/paper/fig14_mqt_5x5.yaml"))
        assert len(experiments) > 0, "No experiments generated from config"

        # Each experiment is a validated ExperimentConfig ready for workers
        for exp in experiments[:3]:
            print(f"  - {exp.exp_id}: {exp.hardware.id} k={exp.hardware.layout.k}")

        rows = []
        for exp in experiments:
            rows.append(flatten_config(exp))

        df = pl.DataFrame(rows)
        assert len(df) == len(experiments), (
            "Flattened DataFrame should have same number of rows as experiments"
        )
        # assert that there are multiple circuit names in the MQT config
        assert df["circuit.id"].n_unique() > 10

    def test_mqt_11x11(self):
        # Load and expand
        experiments = load_experiments(Path("configs/paper/fig4_mqt_11x11.yaml"))
        assert len(experiments) > 0, "No experiments generated from config"

        # Each experiment is a validated ExperimentConfig ready for workers
        for exp in experiments[:3]:
            print(f"  - {exp.exp_id}: {exp.hardware.id} k={exp.hardware.layout.k}")

        rows = []
        for exp in experiments:
            rows.append(flatten_config(exp))

        df = pl.DataFrame(rows)
        assert len(df) == len(experiments), (
            "Flattened DataFrame should have same number of rows as experiments"
        )
        # assert that there are multiple circuit names in the MQT config
        assert df["circuit.id"].n_unique() > 10
        assert len(df) == 27 * 7 * 20, (
            f"Expected 27 circuits * 7 k values * 20 seeds = 3780 experiments, got {len(df)}"
        )

    def test_mqt_qft(self):
        path_config = Path("configs/paper/fig6_qft_grids.yaml")
        experiments = load_experiments(path_config)
        assert len(experiments) > 0, "No experiments generated from config"

        # load yaml directly to check that we have the expected number of experiments
        with open(path_config, "r") as f:
            import yaml

            config_yaml = yaml.safe_load(f)

        # Each experiment is a validated ExperimentConfig ready for workers
        for exp in experiments[:3]:
            print(f"  - {exp.exp_id}: {exp.hardware.id} k={exp.hardware.layout.k}")

        rows = []
        for exp in experiments:
            rows.append(flatten_config(exp))

        df = pl.DataFrame(rows)
        assert len(df) == len(experiments), (
            "Flattened DataFrame should have same number of rows as experiments"
        )
        # assert that there are multiple circuit names in the MQT config
        assert df["circuit.id"].n_unique() == 1

        t1 = float(config_yaml["generic_gate_durations"]["single_qubit_gate"][0])
        t2 = float(config_yaml["generic_gate_durations"]["two_qubit_gate"][0])
        tsw = float(config_yaml["generic_gate_durations"]["switch_gate"][0])

        # check that only these durations appear in the experiments
        assert df["hardware.t1"].unique()[0] == t1
        assert df["hardware.t2"].unique()[0] == t2
        assert df["hardware.tsw"].unique()[0] == tsw

        assert df["serialization.topord_method"].n_unique() == 1
        assert df["serialization.topord_method"].unique()[0] == "prio_two"

        # specific checks for this config
        # check that there for each hardware, there are qft circuits with hw num_qubits,

        expected_num_qubits = {4, 9, 16, 36, 64, 100, 144, 225, 400, 625, 841, 1024, 1225}
        num_transpiler_seeds = config_yaml["circuits"]["num_transpiler_seeds"]

        for hw_id, group in df.group_by("hardware.id"):
            num_qubits = group["hardware.num_qubits"].unique()[0]
            # pop the expected num qubits for this hardware size
            assert num_qubits in expected_num_qubits, (
                f"Unexpected num_qubits {num_qubits} for hardware {hw_id}"
            )
            expected_num_qubits.remove(num_qubits)

            circuit_ids = group["circuit.id"].unique()
            assert len(circuit_ids) == 1, (
                f"Expected 1 circuit id per hardware, got {len(circuit_ids)} for {hw_id}"
            )
            circuit_id = circuit_ids[0]
            assert circuit_id == "qft"

            # for each hw, there should be num_transpiler_seeds qft circuts per hardware.layout.k
            expected_k_values = {k for k in {2, 4, 8} if k <= num_qubits}
            for k, group_k in group.group_by("hardware.layout.k"):
                k = k[0]
                assert k in expected_k_values, (
                    f"Unexpected k={k} for hardware {hw_id}, expected one of {expected_k_values}"
                )
                expected_k_values.remove(k)

                seeds = group_k["circuit.seed"].unique()
                assert len(seeds) == num_transpiler_seeds, (
                    f"Expected {num_transpiler_seeds} transpiler seeds for hardware {hw_id} k={k}, got {len(seeds)}"
                )
            assert len(expected_k_values) == 0, (
                f"Expected k values not covered for hardware {hw_id}: {expected_k_values}"
            )

        # assert that we covered all expected num qubits
        assert len(expected_num_qubits) == 0, (
            f"Expected num_qubits not covered: {expected_num_qubits}"
        )

    def test_mqt_allk(self):
        path_config = Path("configs/paper/fig8_mqt_11x11_scaling.yaml")
        experiments = load_experiments(path_config)
        assert len(experiments) > 0, "No experiments generated from config"

        # load yaml directly to check that we have the expected number of experiments
        with open(path_config, "r") as f:
            import yaml

            config_yaml = yaml.safe_load(f)

        # Each experiment is a validated ExperimentConfig ready for workers
        for exp in experiments[:3]:
            print(f"  - {exp.exp_id}: {exp.hardware.id} k={exp.hardware.layout.k}")

        rows = []
        for exp in experiments:
            rows.append(flatten_config(exp))

        df = pl.DataFrame(rows)
        assert len(df) == len(experiments), (
            "Flattened DataFrame should have same number of rows as experiments"
        )
        # assert that there are multiple circuit names in the MQT config
        assert df["circuit.id"].n_unique() >= 5

        t1 = float(config_yaml["generic_gate_durations"]["single_qubit_gate"][0])
        t2 = float(config_yaml["generic_gate_durations"]["two_qubit_gate"][0])
        tsw = float(config_yaml["generic_gate_durations"]["switch_gate"][0])

        # check that only these durations appear in the experiments
        assert df["hardware.t1"].unique()[0] == t1
        assert df["hardware.t2"].unique()[0] == t2
        assert df["hardware.tsw"].unique()[0] == tsw

        assert df["serialization.topord_method"].n_unique() == 1
        assert df["serialization.topord_method"].unique()[0] == "prio_two"

        # specific checks for this config
        # check that there for each hardware, there are qft circuits with hw num_qubits,
        expected_num_qubits = {121}
        num_transpiler_seeds = config_yaml["circuits"]["num_transpiler_seeds"]

        for hw_id, group in df.group_by("hardware.id"):
            num_qubits = group["hardware.num_qubits"].unique()[0]
            # pop the expected num qubits for this hardware size
            assert num_qubits in expected_num_qubits, (
                f"Unexpected num_qubits {num_qubits} for hardware {hw_id}"
            )
            expected_num_qubits.remove(num_qubits)

            circuit_ids = group["circuit.id"].unique()
            assert len(circuit_ids) > 10

            # for each circuit, there should be many k
            for circuit_id, group_circuit in group.group_by("circuit.id"):
                ks = group_circuit["hardware.layout.k"].unique()
                assert len(ks) > 5, (
                    f"Expected many k values for circuit {circuit_id} on hardware {hw_id}, got {len(ks)}"
                )
                # check that num_transpiler_seeds circuits per k
                for k, group_k in group_circuit.group_by("hardware.layout.k"):
                    num_seeds = group_k["circuit.seed"].n_unique()
                    assert num_seeds == num_transpiler_seeds

        # assert that we covered all expected num qubits
        assert len(expected_num_qubits) == 0, (
            f"Expected num_qubits not covered: {expected_num_qubits}"
        )

    def test_mqt_different_t1t2(self):
        path_config = Path("configs/paper/fig9_mqt_t1t2.yaml")
        experiments = load_experiments(path_config)
        assert len(experiments) > 0, "No experiments generated from config"

        # load yaml directly to check that we have the expected number of experiments
        with open(path_config, "r") as f:
            import yaml

            config_yaml = yaml.safe_load(f)

        # Each experiment is a validated ExperimentConfig ready for workers
        for exp in experiments[:3]:
            print(f"  - {exp.exp_id}: {exp.hardware.id} k={exp.hardware.layout.k}")

        rows = []
        for exp in experiments:
            rows.append(flatten_config(exp))

        df = pl.DataFrame(rows)
        assert len(df) == len(experiments), (
            "Flattened DataFrame should have same number of rows as experiments"
        )
        # assert that there are multiple circuit names in the MQT config
        assert df["circuit.id"].n_unique() == 3

        t1_list = [float(x) for x in config_yaml["generic_gate_durations"]["single_qubit_gate"]]
        t2_list = [float(x) for x in config_yaml["generic_gate_durations"]["two_qubit_gate"]]
        tsw_list = [float(x) for x in config_yaml["generic_gate_durations"]["switch_gate"]]

        # check that only these durations appear in the experiments
        assert list(df["hardware.t1"].unique()) == t1_list
        assert list(df["hardware.t2"].unique()) == t2_list
        assert list(df["hardware.tsw"].unique()) == tsw_list

        assert df["serialization.topord_method"].n_unique() == 1
        assert df["serialization.topord_method"].unique()[0] == "prio_two"

        # specific checks for this config
        # check that there for each hardware, there are qft circuits with hw num_qubits,
        expected_num_qubits = {121}
        num_transpiler_seeds = config_yaml["circuits"]["num_transpiler_seeds"]

        num_ks = len(
            set(tuple(get_balanced_group_sizes(num_qubits=121, qpg=x)) for x in range(1, 121 + 1))
        )

        for hw_id, group in df.group_by("hardware.id"):
            num_qubits = group["hardware.num_qubits"].unique()[0]
            # pop the expected num qubits for this hardware size
            assert num_qubits in expected_num_qubits, (
                f"Unexpected num_qubits {num_qubits} for hardware {hw_id}"
            )
            expected_num_qubits.remove(num_qubits)

            circuit_ids = group["circuit.id"].unique()
            assert len(circuit_ids) == 3

            # for each circuit, there should be many k
            for circuit_id, group_circuit in group.group_by("circuit.id"):
                ks = group_circuit["hardware.layout.k"].n_unique()
                assert ks == num_ks, (
                    f"Unexpected number of k values for circuit {circuit_id} on hardware {hw_id}, got {ks}, expected one of {num_ks}"
                )

                # check that num_transpiler_seeds circuits per k
                for k, group_k in group_circuit.group_by("hardware.layout.k"):
                    num_seeds = group_k["circuit.seed"].n_unique()
                    assert num_seeds == num_transpiler_seeds

        # assert that we covered all expected num qubits
        assert len(expected_num_qubits) == 0, (
            f"Expected num_qubits not covered: {expected_num_qubits}"
        )


class TestWorker:
    def test_random_grid(self):
        experiments = load_experiments(Path("configs/paper/fig2_5x5.yaml"))

        results = []
        for exp_config in experiments[:3]:  # Test on first 3 experiments for speed
            result = worker(exp_config)
            results.append(result)

        assert len(results) == 3, "Should have results for 3 experiments"

    def test_random_densities_grid(self):
        experiments = load_experiments(Path("configs/paper/fig3_random_grids.yaml"))

        results = []
        for exp_config in experiments[-10:]:  # Test on last 10 experiments
            result = worker(exp_config)
            results.append(result)

        assert len(results) == 10, "Should have results for 10 experiments"

    def test_mqt_grid(self):
        experiments = load_experiments(Path("configs/paper/fig14_mqt_5x5.yaml"))

        results = []
        for exp_config in experiments[-10:]:  # Test on last 10 experiments
            result = worker(exp_config)
            results.append(result)

        assert len(results) == 10, "Should have results for 10 experiments"

    def test_brisbane(self):
        experiments = load_experiments(Path("configs/demo_wstate_brisbane.yaml"))

        results = []
        for exp_config in experiments[-3:]:  # Test on last 3 experiments
            result = worker(exp_config)
            results.append(result)

        assert len(results) == 3, "Should have results for 10 experiments"
