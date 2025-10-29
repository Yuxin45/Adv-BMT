"""
This file provides some utility functions for locally validating on the Waymo Open Sim Agents Challenge metrics.

Installation:

conda install python=3.10
pip install waymo-open-dataset-tf-2-12-0==1.6.4

https://github.com/waymo-research/waymo-open-dataset.git
"""

# os.chdir("waymo-open-dataset/src")
# Load Scenario Description for passthrough
import pathlib
from collections import defaultdict
from infgen.utils import utils
import numpy as np
import tensorflow as tf
from google.protobuf import text_format

try:
    from waymo_open_dataset.protos import scenario_pb2
    from waymo_open_dataset.protos import sim_agents_metrics_pb2
    from waymo_open_dataset.protos import sim_agents_submission_pb2
    from waymo_open_dataset.utils.sim_agents import submission_specs
    from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics
except ModuleNotFoundError:
    scenario_pb2 = None
    sim_agents_metrics_pb2 = None
    sim_agents_submission_pb2 = None
    submission_specs = None
    metrics = None

from infgen.utils import wrap_to_pi

# Set memory growth on all gpus.
# all_gpus = tf.config.experimental.list_physical_devices('GPU')
# if all_gpus:
#     try:
#         for cur_gpu in all_gpus:
#             tf.config.experimental.set_memory_growth(cur_gpu, True)
#     except RuntimeError as e:
#         print(e)

FOLDER = pathlib.Path(__file__).resolve().parent

# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU', f"Expected device type to be CPU, got {device.device_type}."


def joint_scene_from_states(states: np.ndarray, object_ids: tf.Tensor) -> sim_agents_submission_pb2.JointScene:
    # States shape: (num_objects, num_steps, 4).
    # Objects IDs shape: (num_objects,).
    # states = states.numpy()
    simulated_trajectories = []
    for i_object in range(object_ids.shape[0]):
        simulated_trajectories.append(
            sim_agents_submission_pb2.SimulatedTrajectory(
                center_x=states[i_object, :, 0],
                center_y=states[i_object, :, 1],
                center_z=states[i_object, :, 2],
                heading=states[i_object, :, 3],
                object_id=object_ids[i_object]
            )
        )
    return sim_agents_submission_pb2.JointScene(simulated_trajectories=simulated_trajectories)


def scenario_rollouts_from_states(
    scenario_id, states: tf.Tensor, object_ids: tf.Tensor
) -> sim_agents_submission_pb2.ScenarioRollouts:
    """
  Aggregate agent states into a ScenarioRollouts proto message.
  """
    # States shape: (num_rollouts, num_objects, num_steps, 4).
    # Objects IDs shape: (num_objects,).
    joint_scenes = []
    for i_rollout in range(states.shape[0]):
        joint_scenes.append(joint_scene_from_states(states[i_rollout], object_ids))
    return sim_agents_submission_pb2.ScenarioRollouts(
        # Note: remember to include the Scenario ID in the proto message.
        joint_scenes=joint_scenes,
        scenario_id=scenario_id
    )


from google.protobuf import json_format


def load_protobuf_from_dict(scenario_dict, scenario_type=scenario_pb2.Scenario):
    """
    Load a Scenario protobuf message from a dictionary.
    
    :param scenario_dict: A dictionary representing the Scenario.
    :return: A Scenario protobuf message.
    """
    scenario = scenario_type()
    json_format.ParseDict(scenario_dict, scenario)
    return scenario


def scenario_description_to_scenario_pb2(sd: dict) -> scenario_pb2.Scenario:
    """
    Converts a scenario description dict to a scenario_pb2.Scenario proto.
    """

    # 1. Parse tracks into the format expected by the scenario proto.
    tracks_to_predict = []
    scenario_tracks = []
    for track_count, (track_name, track) in enumerate(sd["tracks"].items()):
        type_mapping = {
            "VEHICLE": scenario_pb2.Track.TYPE_VEHICLE,
            "PEDESTRIAN": scenario_pb2.Track.TYPE_PEDESTRIAN,
            "CYCLIST": scenario_pb2.Track.TYPE_CYCLIST,
            "UNSET": scenario_pb2.Track.TYPE_UNSET,
            "OTHER": scenario_pb2.Track.TYPE_OTHER
        }

        if str(track_name) in sd["metadata"]["tracks_to_predict"]:
            tracks_to_predict.append(
                {
                    "track_index": track_count,
                    "difficulty": sd["metadata"]["tracks_to_predict"][str(track_name)]["difficulty"],
                }
            )

        # track["state"] is formatted as a dict of arrays of shape (timesteps, dim). We want to convert it into a list of dictionaries sharing the same keys.
        timesteps = track["state"]["position"].shape[0]
        center_x, center_y, center_z = track["state"]["position"].T

        one_d_keys = ["length", "width", "height", "heading", "valid"]
        for key in one_d_keys:
            assert track["state"][key].shape == (
                timesteps,
            ), f"Expected shape (timesteps,), got {track['state'][key].shape}."
        length, width, height, heading, valid = [track["state"][key] for key in one_d_keys]
        velocity_x, velocity_y = track["state"]["velocity"].T
        scenario_tracks.append(
            {
                "id": int(track_name),
                "object_type": type_mapping[track["type"]],
                "states": [
                    {
                        "center_x": center_x[i].tolist(),
                        "center_y": center_y[i].tolist(),
                        "center_z": center_z[i].tolist(),
                        "length": length[i].tolist(),
                        "width": width[i].tolist(),
                        "height": height[i].tolist(),
                        "heading": heading[i].tolist(),
                        "velocity_x": velocity_x[i].tolist(),
                        "velocity_y": velocity_y[i].tolist(),
                        "valid": valid[i].tolist()
                    } for i in range(timesteps)
                ]
            }
        )

    # 2. Build up scenario map features. The protobuf expects a list of map_pb2.MapFeature, each with a type and built up with polylines.
    scenario_map_features = []
    for map_feature_id, map_feature in sd["map_features"].items():
        map_feature_mapping = {
            "LANE": "lane",
            "ROAD_LINE": "road_line",
            "ROAD_EDGE": "road_edge",
            "STOP_SIGN": "stop_sign",
            "CROSSWALK": "crosswalk",
            "SPEED_BUMP": "speed_bump",
            "DRIVEWAY": "driveway",
            "UNKNOWN": "unknown"
        }
        for key in map_feature_mapping:  # E.g if "LANE" in map_feature["type"], then map_feature_type = map_pb2.MapFeature.lane

            if key == "UNKNOWN":
                continue  # TODO: Deal with this in future

            if key in map_feature["type"]:
                map_feature_type = map_feature_mapping[key]
                break
            else:
                map_feature_type = -1

        if map_feature_type == -1:
            continue  # TODO: Deal with this in future
        assert map_feature_type != -1, f"Map feature type {map_feature['type']} not recognized."

        # MAIN DICT FOR MAP FEATURE
        map_feature_dict = {"id": int(map_feature_id), map_feature_type: {}}

        if map_feature_type in ["road_line", "road_edge"]:
            # Polyline features only exist for road_line and road_edge map features.
            # The polylines in the dict are of shape (N, 3). We want to convert them into a list of MapFeature.Polyline, each with an x, y, and z field.
            if map_feature_type == "road_line":
                map_feature_dict[map_feature_type]["type"] = "TYPE_" + '_'.join(map_feature["type"].split("_")[2:])
            else:
                map_feature_dict[map_feature_type]["type"] = "TYPE_" + map_feature["type"]
            polylines = map_feature["polyline"]
            formatted_polylines = [{"x": polyline[0], "y": polyline[1], "z": polyline[2]} for polyline in polylines]
            map_feature_dict[map_feature_type]["polyline"] = formatted_polylines

        elif map_feature_type == "lane":
            # Add lane-specific fields. These can be pass-through fields from the ScenarioDescription.
            passthrough_keys = [
                "speed_limit_mph", "interpolating", "entry_lanes", "exit_lanes", "left_boundaries", "right_boundaries"
            ]
            for key in passthrough_keys:
                try:
                    if type(map_feature_dict[map_feature_type][key]) == list:
                        if len(map_feature_dict[map_feature_type][key]) == 0:
                            continue
                    map_feature_dict[map_feature_type][key] = map_feature[key]
                except:
                    pass

        elif map_feature_type == "stop_sign":
            map_feature_dict[map_feature_type]["lane"] = map_feature["lane"]
            map_feature_dict[map_feature_type]["position"] = {
                "x": map_feature["position"][0],
                "y": map_feature["position"][1],
                "z": map_feature["position"][2]
            }

        elif map_feature_type in ["driveway", "crosswalk", "speed_bump"]:
            # Driveways, crosswalks, and speedbumps are represented by a singular polygon instead of multiple polylines. Each polygon is an array with shape (4, 3)
            polygon = map_feature["polygon"]
            map_feature_dict[map_feature_type]["polygon"] = [
                {
                    "x": vertex[0],
                    "y": vertex[1],
                    "z": vertex[2]
                } for vertex in polygon
            ]

        elif map_feature_type in ["unknown"]:
            # polylines = map_feature["polyline"]
            # map_feature_dict[map_feature_type]["polyline"] = [{"x": polyline[0], "y": polyline[1], "z": polyline[2]} for polyline in polylines]
            pass
            # TODO: Deal with this in future

        else:
            raise ValueError(f"Map feature type {map_feature_type} not recognized.")

        scenario_map_features.append(map_feature_dict)

    assert len(tracks_to_predict) > 0

    # print(f"In scenario {sd['metadata']['scenario_id']}, number of tracks to predict: {len(tracks_to_predict)}")

    scenario_parsedict = {
        "compressed_frame_laser_data": [],  # Not filled
        "current_time_index": sd["metadata"]["current_time_index"],
        "dynamic_map_states": [],  # TODO: Traffic light is not filled
        "map_features": scenario_map_features,
        "objects_of_interest": [],  # Not filled
        "scenario_id": sd['metadata']["scenario_id"],
        "sdc_track_index": sd["metadata"]["sdc_track_index"],
        "timestamps_seconds": sd["metadata"]["ts"].tolist(),
        "tracks": scenario_tracks,
        "tracks_to_predict": tracks_to_predict,
    }

    scenario = load_protobuf_from_dict(scenario_parsedict)
    return scenario


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


def wosac_evaluation(pred_dicts: list, disable_eval, use_2024):
    """
    pred_dicts: A list of dictionaries with the data for evaluation. For more, see data_dict_to_motion_prediction in test_waymo_eval.py.

    Returns:
    scenario_metrics: sim_agents_submission_pb2.SimAgentMetrics -> The metrics for the scenario.
    aggregate_metrics: sim_agents_submission_pb2.SimAgentsBucketedMetrics -> The aggregated metrics for the scenario.
    """
    """    
    scenario: scenario_pb2.Scenario -> The scenario to evaluate. WOSAC uses it for the map data.
    simulated_states: tf.Tensor -> The simulated states of the agents. Shape: (num_rollouts (by default, 32), num_agents, num_steps (80), 4).
    logged_trajectories: waymo_open_dataset.utils.trajectory_utils.ObjectTrajectories
    """
    # Split all data based on scenario
    split_data = defaultdict(list)
    scenario_id_list = []
    # Split the prediction for each scenario, also flatten the data
    for d in pred_dicts:
        for sid in np.unique(d["pred_to_scenario_id"]):
            # For every unique scenario id:
            for k, v in d.items():
                if k in ["pred_trajs", "pred_headings"]:
                    # Filter out the data that corresponds to the particular scenario id.
                    entry_in_same_scenario = [v[idx] for idx in range(len(v)) if d["pred_to_scenario_id"][idx] == sid]
                    assert entry_in_same_scenario  # is not None
                    entry_in_same_scenario = np.stack(entry_in_same_scenario, axis=0)
                    split_data[k].append(entry_in_same_scenario)
                elif k in ["decoder/agent_position", "decoder/agent_velocity", "decoder/agent_heading",
                           "decoder/agent_valid_mask", "decoder/agent_shape", "decoder/agent_type",
                           "raw_scenario_description", "decoder/track_name"]:
                    entry_in_same_scenario = [v[idx] for idx in range(len(v)) if d["scenario_id"][idx] == sid]
                    assert entry_in_same_scenario
                    assert len(
                        entry_in_same_scenario
                    ) == 1  # Assert there's only one scenario data instance for each object in the list.
                    entry_in_same_scenario = entry_in_same_scenario[0]

                    # little workaround
                    if k == "raw_scenario_description" and isinstance(entry_in_same_scenario, list):
                        assert len(entry_in_same_scenario) == 1
                        entry_in_same_scenario = entry_in_same_scenario[0]

                    split_data[k].append(entry_in_same_scenario)
            scenario_id_list.append(sid)

    assert len(split_data["raw_scenario_description"]) > 0, "No scenario description found in the data."

    # PZH: This part is only used to reorder the GT data because the preprocessor filters some agents.
    # for scenario_index, scenario_dict in enumerate(split_data["raw_scenario_description"]):
    #     sdc_name = int(scenario_dict['metadata']['sdc_id'])
    #     tracks = split_data["raw_scenario_description"][scenario_index]["tracks"]
    #     track_name = split_data["decoder/track_name"][scenario_index]  # Mapping from index to old_track_name
    #     modeled_indices = split_data["decoder/modeled_agent_id"][scenario_index]
    #     track_name = track_name[modeled_indices >= 0]
    #     new_track_index_to_old_track_name_mapping = {
    #         int(ind): int(track_name)
    #         for ind, track_name in enumerate(track_name)
    #     }
    #     new_tracks = {
    #         int(track_name): track
    #         for track_name, track in tracks.items()
    #         if int(track_name) in new_track_index_to_old_track_name_mapping.values()
    #     }
    #
    #     if sdc_name not in new_track_index_to_old_track_name_mapping.values():
    #         print(
    #             f"The scenario {scenario_index}, {scenario_dict['metadata']['scenario_id']} does not contain the SDC track.",
    #             sdc_name, new_track_index_to_old_track_name_mapping
    #         )
    #         raise ValueError(
    #             f"The scenario {scenario_index}, {scenario_dict['metadata']['scenario_id']} does not contain the SDC track.",
    #             sdc_name, new_track_index_to_old_track_name_mapping
    #         )
    #
    #     sdc_index = list(new_track_index_to_old_track_name_mapping.values()).index(sdc_name)
    #     split_data["raw_scenario_description"][scenario_index]["metadata"]["sdc_track_index"] = sdc_index
    #     split_data["raw_scenario_description"][scenario_index]["tracks"] = new_tracks

    # PZH: This part is used to reorder the prediction data according to the GT's ordering.
    # for scenario_index, scenario_dict in enumerate(split_data["raw_scenario_description"]):
    #     gt_index_to_preprocessed_index = {
    #         int(track_name): int(ind)
    #         for ind, track_name in enumerate(split_data["decoder/track_name"][scenario_index])
    #     }
    #     new_index = [
    #         gt_index_to_preprocessed_index[k] for k in split_data["raw_scenario_description"][scenario_index]["tracks"]
    #     ]
    #     for k, v in split_data.items():
    #         if k in ["pred_trajs", "pred_headings"]:
    #             split_data[k][scenario_index] = split_data[k][scenario_index][:, :, new_index]
    #         elif k in ['pred_scores']:
    #             split_data[k][scenario_index] = split_data[k][scenario_index][:, new_index]
    #         elif k in [
    #                 "decoder/agent_position",
    #                 "decoder/agent_velocity",
    #                 "decoder/agent_heading",
    #                 "decoder/agent_valid_mask",
    #                 "decoder/agent_shape",
    #         ]:
    #             split_data[k][scenario_index] = split_data[k][scenario_index][:, new_index]
    #         elif k in [
    #                 "decoder/modeled_agent_id",
    #                 "decoder/track_name",
    #                 "decoder/agent_type",
    #         ]:
    #             split_data[k][scenario_index] = split_data[k][scenario_index][new_index]
    #         elif k in ["raw_scenario_description"]:
    #             continue
    #         else:
    #             raise ValueError(f"Key {k} not recognized.")

    scenario_metrics_result = {}
    aggregate_metrics_result = {}

    scenario_rollouts_list = []
    scenario_pb_list = []

    # print("Creating scenario rollouts...")
    for scenario_index, scenario_dict in enumerate(split_data["raw_scenario_description"]):
        # scenario: scenario_pb2.Scenario
        scenario_id = scenario_dict["metadata"]["scenario_id"]
        # simulated states: tf.Tensor with shape (num_modes, num_agents, num_steps, 4)
        # The 4 dimensions are: center_x, center_y, center_z, heading

        states = split_data["pred_trajs"][
            scenario_index]  # torch.Tensor with shape: (num_modes, num_steps, num_agents, 2 -> (center_x, center_y))
        num_modes = states.shape[0]
        headings = wrap_to_pi(split_data["pred_headings"][scenario_index])  # shape: (num_modes, num_steps, num_agents)

        # Change headings to (-pi, pi)
        headings = utils.wrap_to_pi(headings)

        # PZH: Fill in Z here at current step.
        # TODO: Pass in 'current_step=10' as an argument.
        z_values = split_data["decoder/agent_position"][scenario_index][None, 10:11, :, 2]
        z_values = np.repeat(z_values, num_modes, axis=0)
        Modes, T, N, _ = states.shape
        z_values = np.repeat(z_values, T, axis=1)

        # NOTE: Write these if you want to see GT's WOSAC scores.
        # headings[:, :91] = split_data["decoder/agent_heading"][scenario_index][None].repeat(32, axis=0)
        # z_values[:, :91] = split_data["decoder/agent_position"][scenario_index][..., 2][None].repeat(32, axis=0)
        # states[:, :91] = split_data["decoder/agent_position"][scenario_index][..., :2][None].repeat(32, axis=0)

        states = np.concatenate([states, z_values[..., None]], axis=-1)
        states = np.concatenate([states, headings[..., None]], axis=-1)
        states = states.transpose(0, 2, 1, 3)  # shape: (num_modes, num_agents, num_steps, 4 -> (center_x, center_y))

        # TODO: Hardcoded
        if states.shape[2] == 91:
            states = states[:, :, 11:, :]
        elif states.shape[2] == 96:
            states = states[:, :, 11:-5, :]

        # states = tf.convert_to_tensor(states, dtype=tf.float32)

        # Get the trajectory ids for each prediction.
        trajectory_ids = split_data["decoder/track_name"][scenario_index]

        trajectory_ids = np.array([[int(vv) for vv in v] for v in trajectory_ids])
        assert trajectory_ids.shape[0] == 1
        trajectory_ids = trajectory_ids[0]
        trajectory_ids = tf.convert_to_tensor(trajectory_ids, dtype=tf.int32)

        scenario_rollouts = scenario_rollouts_from_states(scenario_id, states, trajectory_ids)

        scenario = scenario_description_to_scenario_pb2(scenario_dict)

        # pred_head = states[0, :, :, -1]
        # gt_head = np.array(
        #     [scenario_dict["tracks"][str(int(i))]["state"]["heading"] for i in trajectory_ids]
        # )[:, 11:]

        # Validate the scenario rollouts. Should raise an exception if it's invalid.
        submission_specs.validate_scenario_rollouts(scenario_rollouts, scenario)

        scenario_rollouts_list.append(scenario_rollouts)

        scenario_pb_list.append(scenario)

        if disable_eval:
            continue

        # Load the test configuration.
        config = load_metrics_config(use_2024=use_2024)

        # PZH: ========== Debug code ==========
        # gt = []
        # for i in range(len(scenario.tracks)):
        #     gta = []
        #     for t in range(91):
        #         gta.append(
        #             [
        #                 scenario.tracks[i].states[t].center_x, scenario.tracks[i].states[t].center_y,
        #                 scenario.tracks[i].states[t].center_z, scenario.tracks[i].states[t].heading
        #             ]
        #         )
        #     gt.append(gta)
        # gt = np.array(gt)[:, 11:]
        # gt_mask = []
        # for i in range(len(scenario.tracks)):
        #     gta = []
        #     for t in range(91):
        #         gta.append(scenario.tracks[i].states[t].valid, )
        #     gt_mask.append(gta)
        # gt_mask = np.array(gt_mask)[:, 11:]
        #
        # preds = []
        # for jointi in range(32):
        #     pred = []
        #     for i in range(len(scenario_rollouts.joint_scenes[jointi].simulated_trajectories)):
        #         pred.append(
        #             np.stack(
        #                 [
        #                     np.array(list(scenario_rollouts.joint_scenes[jointi].simulated_trajectories[i].center_x)),
        #                     np.array(list(scenario_rollouts.joint_scenes[jointi].simulated_trajectories[i].center_y)),
        #                     np.array(list(scenario_rollouts.joint_scenes[jointi].simulated_trajectories[i].center_z)),
        #                     np.array(list(scenario_rollouts.joint_scenes[jointi].simulated_trajectories[i].heading))
        #                 ],
        #                 axis=-1
        #             )
        #         )
        #     pred = np.stack(pred, axis=0)
        #     preds.append(pred)
        #     ade = (np.linalg.norm((gt - preds)[..., :2], axis=-1) * gt_mask).sum() / gt_mask.sum()
        # preds = np.stack(preds, axis=0)
        # ades = (np.linalg.norm(gt[None] - preds, axis=-1) * gt_mask[None, :, :])
        # PZH: ========== Debug code ==========

        scenario_metrics = metrics.compute_scenario_metrics_for_bundle(config, scenario, scenario_rollouts)

        aggregate_metrics = metrics.aggregate_metrics_to_buckets(config, scenario_metrics)

        scenario_metrics_result[scenario_dict["metadata"]["scenario_id"]] = scenario_metrics
        aggregate_metrics_result[scenario_dict["metadata"]["scenario_id"]] = aggregate_metrics

    return scenario_metrics_result, aggregate_metrics_result, scenario_rollouts_list, scenario_pb_list
