import argparse
import json
import os
import pathlib
import pickle

import numpy as np
import tensorflow as tf
import tqdm
from google.protobuf import text_format
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import sim_agents_metrics_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

# Set memory growth on all gpus.
all_gpus = tf.config.experimental.list_physical_devices('GPU')
if all_gpus:
    # try:
    for cur_gpu in all_gpus:
        tf.config.experimental.set_memory_growth(cur_gpu, True)
    # except RuntimeError as e:
    #     print(e)

# tf.config.set_visible_devices([], 'GPU')
# visible_devices = tf.config.get_visible_devices()
# for device in visible_devices:
#     assert device.device_type != 'GPU', f"Expected device type to be CPU, got {device.device_type}."

FOLDER = pathlib.Path(__file__).resolve().parent / "eval"

scenario_metrics_keys = [
    # 'scenario_id',
    'metametric',
    'average_displacement_error',
    'min_average_displacement_error',
    'linear_speed_likelihood',
    'linear_acceleration_likelihood',
    'angular_speed_likelihood',
    'angular_acceleration_likelihood',
    'distance_to_nearest_object_likelihood',
    'collision_indication_likelihood',
    'time_to_collision_likelihood',
    'distance_to_road_edge_likelihood',
    'offroad_indication_likelihood'
]

aggregate_metrics_keys = [
    'realism_meta_metric', 'kinematic_metrics', 'interactive_metrics', 'map_based_metrics', 'min_ade'
]


def scenario_metrics_to_dict(scenario_metrics):
    return {k: getattr(scenario_metrics, k) for k in scenario_metrics_keys}


def aggregate_metrics_to_dict(aggregate_metrics):
    return {k: getattr(aggregate_metrics, k) for k in aggregate_metrics_keys}


def load_metrics_config(use_2024) -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
    """Loads the `SimAgentMetricsConfig` used for the challenge."""
    # pylint: disable=line-too-long
    # pyformat: disable

    # As noted in: https://github.com/waymo-research/waymo-open-dataset/issues/817
    # The config have changed. So we need to switch between them.
    if use_2024:
        config_path = FOLDER / 'challenge_2024_config.textproto'
    else:
        config_path = FOLDER / 'challenge_2023_config.textproto'

    with open(config_path, 'r') as f:
        config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
        text_format.Parse(f.read(), config)
    return config


parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--use_2023", action="store_true")
args = parser.parse_args()

use_2024 = not args.use_2023

output_dir = pathlib.Path(args.output_dir)

raw_data_dir = output_dir / "scenario_pb"

# Load the test configuration.
config = load_metrics_config(use_2024=use_2024)

prefix = "submission.binproto-tmp"

scenario_metrics_result = {}
aggregate_metrics_result = {}
output_metrics = []

assert output_dir.is_dir()
assert raw_data_dir.is_dir()
files = os.listdir(output_dir)
files = [f for f in files if f.startswith(prefix)]
for f in tqdm.tqdm(files, desc="Files"):
    fp = output_dir / f
    with open(fp, 'rb') as f:
        shard_submission = f.read()
    submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission.FromString(shard_submission)

    for scenario_rollouts in tqdm.tqdm(submission.scenario_rollouts, desc="Scenarios"):
        sid = scenario_rollouts.scenario_id

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

        print("======== TEMPORARY RESULT WITH {} SCENARIOS ==========".format(len(output_metrics)))
        for k in output_metrics[0].keys():
            print(f"{k}: {np.mean([m[k] for m in output_metrics]):.5f}")
        print("===========================================================")

# Save the results
with open(output_dir / "aggregate_metrics.pkl", "wb") as f:
    pickle.dump(aggregate_metrics_result, f)
with open(output_dir / "scenario_metrics.pkl", "wb") as f:
    pickle.dump(scenario_metrics_result, f)

print("======== FINAL RESULT WITH {} SCENARIOS ==========".format(len(output_metrics)))
for k in output_metrics[0].keys():
    print(f"{k}: {np.mean([m[k] for m in output_metrics]):.5f}")
print("===========================================================")
results = {k: np.mean([m[k] for m in output_metrics]) for k in output_metrics[0].keys()}
with open(output_dir / "results.json", "w") as f:
    json.dump(results, f)
print("Results saved at: ", output_dir / "results.json")
