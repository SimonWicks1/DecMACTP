#!/usr/bin/env python3

import os
import sys
import subprocess
import re
import yaml
from glob import glob

train_script = "scripts/train.py"
eval_script = "scripts/eval.py"
configs_dir = "configs"


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run GNARL experiments for all configs and multiple seeds."
    )
    parser.add_argument(
        "--n_seeds", type=int, default=1, help="Number of seeds/runs per config"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiment_outputs",
        help="Directory to store outputs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config_files = sorted(glob(os.path.join(configs_dir, "*.yaml")))
    run_ids = {}

    for config_path in config_files:
        config_name = os.path.basename(config_path)
        run_ids[config_name] = []
        for seed in range(args.n_seeds):
            train_outfile = os.path.join(
                args.output_dir, f"{config_name}_seed{seed}_train.out"
            )
            print(f"Running training for {config_name}, seed {seed}...")
            with open(train_outfile, "w") as fout:
                proc = subprocess.run(
                    [
                        sys.executable,
                        train_script,
                        "-c",
                        config_path,
                        "--seed",
                        str(seed),
                    ],
                    stdout=fout,
                    stderr=subprocess.STDOUT,
                )
            # Extract run id from output file
            run_id = None
            with open(train_outfile, "r") as fin:
                for line in fin:
                    # Look for line like: 'Beginning run: <run.id>' or 'run: <run.id>'
                    m = re.search(r"Beginning run: ([\w\d]+)", line)
                    if m:
                        run_id = m.group(1)
                        break
            if run_id is None:
                print(f"Warning: Could not find run id in {train_outfile}")
                continue
            run_ids[config_name].append(run_id)
            # Run evaluation for this run
            eval_outfile = os.path.join(
                args.output_dir, f"{config_name}_seed{seed}_eval.out"
            )
            print(
                f"Running evaluation for {config_name}, seed {seed}, run_id {run_id}..."
            )
            with open(eval_outfile, "w") as fout:
                subprocess.run(
                    [sys.executable, eval_script, "-c", config_path, "-r", run_id],
                    stdout=fout,
                    stderr=subprocess.STDOUT,
                )

    # Save run_ids dict to yaml
    runids_path = os.path.join(args.output_dir, "run_ids.yaml")
    with open(runids_path, "w") as f:
        yaml.dump(run_ids, f)
    print(f"Run IDs saved to {runids_path}")


if __name__ == "__main__":
    main()
