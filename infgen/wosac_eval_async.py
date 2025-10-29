import argparse
import json
import os
import pathlib
import pickle
import numpy as np
import tensorflow as tf
import tqdm
import multiprocessing
from google.protobuf import text_format
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_metrics_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

FOLDER = pathlib.Path(__file__).resolve().parent / "eval"

scenario_metrics_keys = [
    'metametric', 'average_displacement_error', 'min_average_displacement_error', 'linear_speed_likelihood',
    'linear_acceleration_likelihood', 'angular_speed_likelihood', 'angular_acceleration_likelihood',
    'distance_to_nearest_object_likelihood', 'collision_indication_likelihood', 'time_to_collision_likelihood',
    'distance_to_road_edge_likelihood', 'offroad_indication_likelihood'
]

aggregate_metrics_keys = [
    'realism_meta_metric', 'kinematic_metrics', 'interactive_metrics', 'map_based_metrics', 'min_ade'
]


def scenario_metrics_to_dict(scenario_metrics):
    return {k: getattr(scenario_metrics, k) for k in scenario_metrics_keys}


def aggregate_metrics_to_dict(aggregate_metrics):
    return {k: getattr(aggregate_metrics, k) for k in aggregate_metrics_keys}


def load_metrics_config(use_2024) -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
    if use_2024:
        config_path = FOLDER / 'challenge_2024_config.textproto'
    else:
        config_path = FOLDER / 'challenge_2023_config.textproto'
    with open(config_path, 'r') as f:
        config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
        text_format.Parse(f.read(), config)
    return config


def process_files(files, config, output_dir, raw_data_dir, gpu_id, allow_scenarios, all_gpus):
    print("Process {} uses GPU {}".format(gpu_id, all_gpus[gpu_id]))
    gpu_id = all_gpus[gpu_id]
    # Set the specific GPU visible to this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Initialize TensorFlow settings after setting GPU visibility
    all_gpus = tf.config.experimental.list_physical_devices('GPU')
    if all_gpus:
        for cur_gpu in all_gpus:
            tf.config.experimental.set_memory_growth(cur_gpu, True)

    output_metrics = []
    for f in tqdm.tqdm(files, desc="[{}] Files".format(gpu_id)):
        fp = output_dir / f
        with open(fp, 'rb') as f:
            shard_submission = f.read()
        submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission.FromString(shard_submission)

        scount = 0
        for scenario_rollouts in submission.scenario_rollouts:
            sid = scenario_rollouts.scenario_id
            scount += 1
            if allow_scenarios is not None and sid not in allow_scenarios:
                print("[{}] Skip scenario {}".format(gpu_id, sid))
                continue

            print("[{}] Processing scenario {}/{} {}".format(gpu_id, scount, len(submission.scenario_rollouts), sid))

            raw_data_fp = raw_data_dir / f"{sid}.binproto"
            with open(raw_data_fp, 'rb') as f:
                raw_data = f.read()

            scenario = scenario_pb2.Scenario.FromString(raw_data)
            scenario_metrics = metrics.compute_scenario_metrics_for_bundle(config, scenario, scenario_rollouts)
            aggregate_metrics = metrics.aggregate_metrics_to_buckets(config, scenario_metrics)
            scenario_metrics_result = scenario_metrics_to_dict(scenario_metrics)
            aggregate_metrics_result = aggregate_metrics_to_dict(aggregate_metrics)
            stat = {}
            for k in scenario_metrics_keys:
                stat[f"scenario_metrics/{k}"] = scenario_metrics_result[k]
            for k in aggregate_metrics_keys:
                stat[f"aggregate_metrics/{k}"] = aggregate_metrics_result[k]
            output_metrics.append(stat)

            with open(output_dir / "raw_outputs_tmp_{}.pkl".format(gpu_id), "wb") as f:
                pickle.dump(output_metrics, f)

            print(
                "======== [{}] SCENARIO {}/{} {} ==========".format(
                    gpu_id, scount, len(submission.scenario_rollouts), sid
                )
            )
            print("Temporary results average over {} scenarios:".format(scount))
            for k, v in stat.items():
                print(f"{k}: {np.mean([m[k] for m in output_metrics]):.5f}")
            print("===========================================================")

    return output_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    # Allow user input a list of GPUs:
    parser.add_argument("--gpus", "--gpu", nargs="+", type=int, required=True)

    parser.add_argument("--use_2023", action="store_true")
    parser.add_argument("--limit_batches", type=int, default=-1)
    args = parser.parse_args()
    use_2024 = not args.use_2023

    output_dir = pathlib.Path(args.output_dir)
    raw_data_dir = output_dir / "scenario_pb"

    all_gpus = args.gpus
    print("All GPUs:", all_gpus)

    limit_batches = args.limit_batches
    allow_scenarios = None
    if limit_batches >= 0:
        with open("/bigdata/datasets/scenarionet/waymo/validation_first_800/dataset_summary.pkl", "rb") as f:
            dataset_summary = pickle.load(f)
        allow_scenarios = [s.split("_")[-1].split(".pkl")[0] for s in dataset_summary.keys()]
        allow_scenarios = allow_scenarios[:limit_batches]

        print("Allow scenarios:", len(allow_scenarios))

    # Load test configuration
    config = load_metrics_config(use_2024=use_2024)

    # List all files
    prefix = "submission.binproto-tmp"
    assert output_dir.is_dir()
    assert raw_data_dir.is_dir()
    files = [f for f in os.listdir(output_dir) if f.startswith(prefix)]

    # Read num_gpus:
    num_gpus = len(all_gpus)

    # Split files among available GPUs
    file_splits = np.array_split(files, num_gpus)

    # Use multiprocessing to handle file splits on each GPU
    with multiprocessing.Pool(num_gpus) as pool:
        results = pool.starmap(
            process_files, [
                (file_split, config, output_dir, raw_data_dir, gpu_id, allow_scenarios, all_gpus)
                for gpu_id, file_split in enumerate(file_splits)
            ]
        )

    # Aggregate results
    output_metrics = [metric for gpu_result in results for metric in gpu_result]

    print("totally", len(output_metrics), "scenarios")

    with open(output_dir / "raw_outputs.pkl", "wb") as f:
        pickle.dump(output_metrics, f)

    # Calculate and save aggregate metrics
    final_results = {k: np.mean([m[k] for m in output_metrics]) for k in output_metrics[0].keys()}

    with open(output_dir / "aggregate_metrics.pkl", "wb") as f:
        pickle.dump(final_results, f)
    with open(output_dir / "results.json", "w") as f:
        json.dump(final_results, f)

    print("======== FINAL RESULT WITH {} SCENARIOS ==========".format(len(output_metrics)))
    for k, v in final_results.items():
        print(f"{k}: {v:.5f}")
    print("Results saved at:", output_dir / "results.json")


if __name__ == "__main__":
    # Ensure each process initializes independently with a specific start method
    multiprocessing.set_start_method("spawn")
    main()
