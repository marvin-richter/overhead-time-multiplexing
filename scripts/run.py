import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
from pathlib import Path
from random import shuffle

from polars import DataFrame
from tqdm import tqdm

from overhead_time_multiplexing.experiments import load_experiments
from overhead_time_multiplexing.experiments.worker import worker

import logging


def setup_logging(name=None) -> logging.Logger:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Suppress verbose logging from external libraries
    loggers_to_suppress = [
        "qiskit.passmanager.base_tasks",
        "qiskit.transpiler",
        "qiskit.compiler",
        "stevedore.extension",
    ]

    for logger_name in loggers_to_suppress:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if name:
        return logging.getLogger(name)
    else:
        return logging.getLogger(__name__)


logger = setup_logging()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run experiment configurations from a YAML config file path."
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run experiments in parallel using multiple processes.",
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the experiment config YAML file (e.g. configs/local/demo.yaml).",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Save intermediate results every N experiments (default: 100).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel workers. Overrides automatic detection (default: auto).",
    )

    args = parser.parse_args()

    config_path = args.config_path

    experiments = load_experiments(config_path)
    print(f"Generated {len(experiments)} experiment configurations")

    shuffle(experiments)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    parquet_filename = f"{timestamp}_{experiments[0].exp_name}_N{len(experiments)}.parquet"
    parquet_filepath = experiments[0].path_output / parquet_filename

    def save_intermediate(results_so_far):
        if results_so_far:
            df_tmp = DataFrame(results_so_far)
            df_tmp.write_parquet(parquet_filepath.with_suffix(".partial.parquet"))
            print(f"Saved {len(results_so_far)} intermediate results")

    if args.parallel:
        results = []

        # Use SLURM allocation if available, otherwise fall back to cpu_count
        if args.n_workers is not None:
            num_workers = args.n_workers
        else:
            num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)) - 1
        print(f"running on {num_workers} cores")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(worker, exp_config): exp_config for exp_config in experiments
            }

            for future in tqdm(as_completed(futures), total=len(futures), smoothing=0.0):
                try:
                    result = future.result()
                    results.append(result)
                    if len(results) % args.save_every == 0:
                        save_intermediate(results)
                except Exception as e:
                    exp_config = futures[future]
                    print(f"Failed for {exp_config.exp_id}: {e}")

    else:
        results = []
        for i, exp_config in enumerate(tqdm(experiments)):
            result = worker(exp_config)
            results.append(result)
            if (i + 1) % args.save_every == 0:
                save_intermediate(results)

    df = DataFrame(results)

    df.write_parquet(parquet_filepath)
    df.write_csv(parquet_filepath.with_suffix(".csv"))
    print(f"Done. Wrote to: {parquet_filepath}")

    # Clean up partial file if it exists
    partial_file = parquet_filepath.with_suffix(".partial.parquet")
    if partial_file.exists():
        partial_file.unlink()
        print(f"Deleted partial file: {partial_file}")
