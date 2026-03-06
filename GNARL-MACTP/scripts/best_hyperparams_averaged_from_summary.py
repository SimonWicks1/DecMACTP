import wandb
import argparse
from collections import defaultdict
import statistics


def main():
    parser = argparse.ArgumentParser(
        description="Find the best hyperparameters from a wandb sweep by averaging over different seeds (using run summary)"
    )
    parser.add_argument("sweep_id", help="The wandb sweep ID (e.g., 'si67rvp3')")
    parser.add_argument(
        "metric", help="The metric name to optimize (e.g., 'mean_reward')"
    )
    parser.add_argument(
        "--seed-param",
        default="PPO.seed",
        help="The name of the seed parameter to exclude from hyperparameter grouping (default: PPO.seed)",
    )

    args = parser.parse_args()

    api = wandb.Api()

    sweep = api.sweep("GNARL-CTP/" + args.sweep_id)
    metric = args.metric
    seed_param = args.seed_param

    # Collect all runs with their metrics and hyperparameters
    all_runs_data = []
    seen_run_ids = set()  # Track seen run IDs to avoid duplicates
    failed_runs = []  # Track runs that failed to get metrics

    for run in sweep.runs:
        if run.id in seen_run_ids:
            print(f"Warning: Skipping duplicate run ID {run.id}")
            continue

        metric_value = run.summary.get(metric, None)
        step = run.summary.get("_step", None)

        if metric_value is not None:
            hyperparams = {}
            seed_value = None
            for param_name, param_value in run.config.items():
                if not param_name.startswith("_"):
                    if param_name == seed_param:
                        seed_value = param_value
                    elif (
                        param_name.startswith("PPO.")
                        or param_name.startswith("BC.")
                        or param_name.startswith("policy_kwargs.")
                    ):
                        hyperparams[param_name] = param_value
            all_runs_data.append(
                {
                    "run": run,
                    "metric_value": float(metric_value),
                    "step": step,
                    "hyperparams": hyperparams,
                    "seed": seed_value,
                }
            )
            seen_run_ids.add(run.id)
        else:
            failed_runs.append(
                {
                    "run_id": run.id,
                    "run_name": run.name,
                    "seed": run.config.get(seed_param, "unknown"),
                    "metric_value": metric_value,
                    "step": step,
                }
            )

    if not all_runs_data:
        print("No valid runs found in the sweep.")
        return

    print(f"Found {len(all_runs_data)} valid runs with {metric} values")

    if failed_runs:
        print(f"Found {len(failed_runs)} runs that failed to get valid metrics:")
        failed_by_config = defaultdict(list)
        for failed_run in failed_runs:
            try:
                run = next(r for r in sweep.runs if r.id == failed_run["run_id"])
                hyperparams = {}
                for param_name, param_value in run.config.items():
                    if not param_name.startswith("_") and param_name != seed_param:
                        if (
                            param_name.startswith("PPO.")
                            or param_name.startswith("BC.")
                            or param_name.startswith("policy_kwargs.")
                        ):
                            hyperparams[param_name] = param_value
                config_key = tuple(sorted(hyperparams.items()))
                failed_by_config[config_key].append(failed_run)
            except:
                print(f"  Could not analyze failed run {failed_run['run_id']}")
        for config_key, failed_group in failed_by_config.items():
            if len(failed_group) > 0:
                print(f"  Config with {len(failed_group)} failed runs:")
                hyperparams_dict = dict(config_key)
                for param, value in sorted(hyperparams_dict.items()):
                    print(f"    {param}: {value}")
                print(f"    Failed seeds: {[f['seed'] for f in failed_group]}")
                print(f"    Failed run IDs: {[f['run_id'] for f in failed_group]}")
                print()
    print()

    hyperparam_groups = defaultdict(list)
    for run_data in all_runs_data:
        hyperparam_key = tuple(sorted(run_data["hyperparams"].items()))
        hyperparam_groups[hyperparam_key].append(run_data)

    averaged_results = []
    for hyperparam_key, runs_group in hyperparam_groups.items():
        unique_runs = {}
        for run_data in runs_group:
            run_id = run_data["run"].id
            if run_id not in unique_runs:
                unique_runs[run_id] = run_data
            else:
                print(
                    f"Warning: Found duplicate run ID {run_id} in same hyperparameter group"
                )
        runs_group = list(unique_runs.values())
        metrics = [run_data["metric_value"] for run_data in runs_group]
        seeds = [run_data["seed"] for run_data in runs_group]
        expected_seeds = {1, 2, 3, 4, 5}
        actual_seeds = set(seeds)
        missing_seeds = expected_seeds - actual_seeds
        if missing_seeds:
            hyperparams_dict = dict(hyperparam_key)
            print(
                f"Warning: Missing seeds {sorted(missing_seeds)} for hyperparameters:"
            )
            for param, value in sorted(hyperparams_dict.items()):
                print(f"  {param}: {value}")
            print()
        avg_metric = statistics.mean(metrics)
        std_metric = statistics.stdev(metrics) if len(metrics) > 1 else 0
        averaged_results.append(
            {
                "hyperparams": dict(hyperparam_key),
                "avg_metric": avg_metric,
                "std_metric": std_metric,
                "num_seeds": len(metrics),
                "seeds": sorted(seeds),
                "individual_metrics": metrics,
                "runs": runs_group,
                "missing_seeds": sorted(missing_seeds) if missing_seeds else None,
            }
        )
    averaged_results.sort(key=lambda x: x["avg_metric"], reverse=True)
    print(f"Found {len(averaged_results)} unique hyperparameter combinations")
    print()
    print("=" * 80)
    print(f"Top hyperparameter combinations ranked by average {metric}:")
    print("=" * 80)
    for i, result in enumerate(averaged_results[:10], 1):
        print(f"Rank #{i}:")
        print(
            f"  Average {metric}: {result['avg_metric']:.4f} ± {result['std_metric']:.4f}"
        )
        print(f"  Number of seeds: {result['num_seeds']}")
        print(f"  Seeds tested: {result['seeds']}")
        if result["missing_seeds"]:
            print(f"  Missing seeds: {result['missing_seeds']}")
        print(
            f"  Individual values: {[f'{v:.4f}' for v in result['individual_metrics']]}"
        )
        print(f"  Hyperparameters:")
        for param_name, param_value in sorted(result["hyperparams"].items()):
            print(f"    {param_name}: {param_value}")
        print(f"  Run IDs:")
        for run_data in result["runs"]:
            print(
                f"    {run_data['run'].id} (seed={run_data['seed']}, {metric}={run_data['metric_value']:.4f})"
            )
        print()
    print("=" * 80)
    print("Parameter value analysis across all combinations:")
    print("=" * 80)
    param_performance = defaultdict(lambda: defaultdict(list))
    for result in averaged_results:
        for param_name, param_value in result["hyperparams"].items():
            param_performance[param_name][param_value].append(result["avg_metric"])
    for param_name, value_metrics in param_performance.items():
        print(f"{param_name}:")
        value_stats = []
        for value, metrics in value_metrics.items():
            avg = statistics.mean(metrics)
            std = statistics.stdev(metrics) if len(metrics) > 1 else 0
            value_stats.append((value, avg, std, len(metrics)))
        value_stats.sort(key=lambda x: x[1], reverse=True)
        for value, avg, std, count in value_stats:
            print(f"  {value}: {avg:.4f} ± {std:.4f} (n={count})")
        print()
    if averaged_results:
        best_result = averaged_results[0]
        print("=" * 80)
        print("BEST HYPERPARAMETER COMBINATION:")
        print("=" * 80)
        print(
            f"Average {metric}: {best_result['avg_metric']:.4f} ± {best_result['std_metric']:.4f}"
        )
        print(f"Tested with {best_result['num_seeds']} seeds: {best_result['seeds']}")
        print()
        print("Hyperparameters:")
        for param_name, param_value in sorted(best_result["hyperparams"].items()):
            print(f"  {param_name}: {param_value}")
        print()
        print("Individual run details:")
        for run_data in best_result["runs"]:
            print(
                f"  Run {run_data['run'].id}: seed={run_data['seed']}, {metric}={run_data['metric_value']:.4f}, step={run_data['step']}"
            )
            print(f"    URL: {run_data['run'].url}")


if __name__ == "__main__":
    main()
