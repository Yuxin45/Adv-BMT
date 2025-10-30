"""
Script to generate submission files for Waymo SimAgent Challenge.
Please check out the end of this file where we provide a script to merge submission files.
"""
import copy
import os
import pathlib
import uuid

import numpy as np
import torch

from bmt.dataset.preprocessor import centralize_to_map_center
from bmt.eval.waymo_motion_prediction_evaluator import _repeat_for_modes
from bmt.eval.wosac_eval import wosac_evaluation
from bmt.tokenization import get_tokenizer
from bmt.utils import wrap_to_pi, rotate


def transform_to_global_coordinate(data_dict):
    map_center = data_dict["metadata/map_center"].reshape(-1, 1, 1, 3)
    map_heading = data_dict["metadata/map_heading"].reshape(-1, 1, 1)
    B, T, N, _ = data_dict["decoder/agent_position"].shape
    map_heading = map_heading.repeat(T, axis=1).repeat(N, axis=2)
    assert map_heading.shape == (B, T, N)
    data_dict["decoder/agent_position"] = rotate(
        x=data_dict["decoder/agent_position"][..., 0],
        y=data_dict["decoder/agent_position"][..., 1],
        angle=map_heading,
        z=data_dict["decoder/agent_position"][..., 2]
    )
    assert data_dict["decoder/agent_position"].ndim == 4
    data_dict["decoder/agent_position"] += map_center

    data_dict["decoder/agent_heading"] = wrap_to_pi(data_dict["decoder/agent_heading"] + map_heading)

    data_dict["decoder/agent_velocity"] = rotate(
        x=data_dict["decoder/agent_velocity"][..., 0],
        y=data_dict["decoder/agent_velocity"][..., 1],
        angle=map_heading,
    )

    data_dict["decoder/agent_position"][~data_dict["decoder/agent_valid_mask"]] = 0
    data_dict["decoder/agent_heading"][~data_dict["decoder/agent_valid_mask"]] = 0
    data_dict["decoder/agent_velocity"][~data_dict["decoder/agent_valid_mask"]] = 0

    data_dict["pred_trajs"] = [
        centralize_to_map_center(
            traj, map_center=-data_dict["expanded_map_center"][b], map_heading=-data_dict["expanded_map_heading"][b]
        ) for b, traj in enumerate(data_dict["pred_trajs"])
    ]

    return data_dict


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


class WaymoSimAgentEvaluator:
    def __init__(self, config):
        self.config = config

        self.metrics = []
        self.scenario_rollouts_list = []
        self.scenario_pb_list = []
        self.shard_count = 0
        self.scenario_count = 0
        # self.output_filenames = []

        self.num_scenarios_per_shard = 10

        if self.config.EVALUATION.NAME == "wosac2024":
            self.use_2024 = True
        elif self.config.EVALUATION.NAME == "wosac2023":
            self.use_2024 = False
        else:
            raise ValueError()

        print("[Sim Agent] SAMPLING CONFIG IS: ", self.config.SAMPLING)

    def _call_model(self, data_dict, model):
        """We might want to create mini batches to call model in case the of OOM..."""
        num_decode_steps = 16

        temperature = self.config.SAMPLING.TEMPERATURE
        sampling_method = self.config.SAMPLING.SAMPLING_METHOD
        topp = self.config.SAMPLING.TOPP
        use_cache = self.config.EVALUATION.USE_CACHE

        # ===== Autoregressive Decoding =====
        with torch.no_grad():
            expanded_data_dict = model.autoregressive_rollout(
                data_dict=data_dict,
                # num_decode_steps=num_decode_steps,
                temperature=temperature,
                sampling_method=sampling_method,
                topp=topp,
                use_cache=use_cache,
            )
            tokenizer = model.tokenizer
            expanded_data_dict = tokenizer.detokenize(
                expanded_data_dict, flip_wrong_heading=self.config.TOKENIZATION.FLIP_WRONG_HEADING
            )

        # ===== Postprocessing to extract predictions for the modeled agents =====
        scores = expanded_data_dict["decoder/output_score"]
        pred_trajs = expanded_data_dict["decoder/reconstructed_position"]
        pred_heading = expanded_data_dict["decoder/reconstructed_heading"]

        # If training to predict all agents, but asking for eval on modeled agents,
        # need to pick the prediction for the modeled agents only.
        assert self.config.TRAINING.PREDICT_ALL_AGENTS
        assert self.config.EVALUATION.PREDICT_ALL_AGENTS

        return pred_trajs, pred_heading, scores, expanded_data_dict

    def validation_step(self, data_dict, batch_idx, model, log_dict_func, global_rank, logger, **kwargs):

        num_modes_for_eval = self.config.EVALUATION.NUM_MODES
        maximum_batch_size = self.config.EVALUATION.MAXIMUM_BATCH_SIZE

        if num_modes_for_eval <= maximum_batch_size:
            num_repeat_calls = 1
        else:
            assert num_modes_for_eval % maximum_batch_size == 0
            num_repeat_calls = num_modes_for_eval // maximum_batch_size

        NUM_MODES_WAYMO_SIM_AGENTS = 32

        B = data_dict["encoder/agent_feature"].shape[0]
        data_dict["batch_idx"] = torch.arange(B)

        # DEBUG:
        # print("RAW SCENARIO DESCPTION: (BEFORE) ", data_dict["raw_scenario_description"][0]['id'])

        if num_repeat_calls == 1:
            expanded_data_dict = {
                k: _repeat_for_modes(data_dict[k], num_modes=num_modes_for_eval)
                for k in data_dict.keys() if (
                    k.startswith("encoder/") or k.startswith("decoder/") or k.startswith("metadata/")
                    or k.startswith("decoder/") or k in ["batch_idx", "in_evaluation", "scenario_id"]
                )
            }
            pred_trajs_of_interested_agents, pred_heading_of_interested_agents, scores_of_interested_agents, output_data_dict = self._call_model(
                expanded_data_dict, model
            )

        else:
            assert B == 1
            num_modes_per_call = num_modes_for_eval // num_repeat_calls
            assert num_modes_per_call * num_repeat_calls == num_modes_for_eval
            expanded_data_dict = {
                k: _repeat_for_modes(data_dict[k], num_modes=num_modes_per_call)
                for k in data_dict.keys() if (
                    k.startswith("encoder/") or k.startswith("decoder/") or k.startswith("metadata/")
                    or k.startswith("decoder/")
                    or k in ["batch_idx", "in_evaluation", "scenario_id", "in_backward_prediction"]
                )
            }

            pred_trajs_of_interested_agents = []
            scores_of_interested_agents = []
            pred_heading_of_interested_agents = []
            for call in range(num_repeat_calls):
                traj, head, score, output_data_dict = self._call_model(copy.deepcopy(expanded_data_dict), model)
                pred_trajs_of_interested_agents.append(traj)
                scores_of_interested_agents.append(score)
                pred_heading_of_interested_agents.append(head)
            pred_trajs_of_interested_agents = [vv for v in pred_trajs_of_interested_agents for vv in v]
            scores_of_interested_agents = [vv for v in scores_of_interested_agents for vv in v]
            pred_heading_of_interested_agents = [vv for v in pred_heading_of_interested_agents for vv in v]

        pred_to_scenario_id = _repeat_for_modes(data_dict["scenario_id"], num_modes=NUM_MODES_WAYMO_SIM_AGENTS)
        expanded_map_center = _repeat_for_modes(data_dict["metadata/map_center"], num_modes=NUM_MODES_WAYMO_SIM_AGENTS)
        expanded_map_heading = _repeat_for_modes(
            data_dict["metadata/map_heading"], num_modes=NUM_MODES_WAYMO_SIM_AGENTS
        )

        # Conduct non-maximum suppression (NMS) to reduce the number of modes
        if num_modes_for_eval > NUM_MODES_WAYMO_SIM_AGENTS:
            raise ValueError("No NMS for SimAgent!")

        # ===== Cache the prediction results =====
        prediction_dict = {
            "pred_trajs": pred_trajs_of_interested_agents,
            "pred_headings": pred_heading_of_interested_agents,
            "pred_scores": scores_of_interested_agents,
            "pred_to_scenario_id": pred_to_scenario_id,
            "expanded_map_center": expanded_map_center,
            "expanded_map_heading": expanded_map_heading,
        }
        for k, v in data_dict.items():
            if k.startswith("decoder/") or k.startswith("decoder/") or k.startswith("metadata/") or k in [
                    "raw_scenario_description", "scenario_id"
            ]:
                prediction_dict[k] = v

        new_prediction_dict = {}
        for k, v in prediction_dict.items():
            if isinstance(v, torch.Tensor):
                new_prediction_dict[k] = v.detach().cpu().numpy()
            elif isinstance(v, list):
                new_prediction_dict[k] = [vv.detach().cpu().numpy() if isinstance(vv, torch.Tensor) else v for vv in v]
            else:
                new_prediction_dict[k] = v
        # prediction_dict = copy.deepcopy(new_prediction_dict)  # Avoid memory issue

        # Transform back to global coordinate
        new_prediction_dict = transform_to_global_coordinate(new_prediction_dict)
        # self.validation_outputs.append(new_prediction_dict)
        scenario_metrics, aggregate_metrics, scenario_rollouts_list, scenario_pb_list = wosac_evaluation(
            [new_prediction_dict], disable_eval=True, use_2024=self.use_2024
        )

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.gca().set_aspect('equal', adjustable='box')
        # from infgen.gradio_ui.plot import _plot_map
        # from infgen.utils import utils
        # np_d = utils.torch_to_numpy(data_dict)
        # np_d = {k: v[0] for k, v in np_d.items()}
        # _plot_map(np_d, ax=plt.gca())
        # AID = 2
        # for mode in pred_trajs_of_interested_agents:
        #     mode = utils.torch_to_numpy(mode)
        #     plt.plot(mode[:, AID, 0], mode[:, AID, 1])
        # plt.show()

        # TODO: Some assertions here to avoid WOSAC error ......
        # https://github.com/waymo-research/waymo-open-dataset/issues/807
        # Scenario 891805f154b4f0dd: Sim agents {1178} are missing from the simulation.
        # Scenario de8c427e65487b93: Sim agents {432, 569, 554} are missing from the simulation.
        # Scenario 386f0b2faebe74af: Sim agents {3361, 3332, 3399, 3371, 3375, 3311, 3345, 3346, 3378, 3319, 3352, 3388} are missing from the simulation.
        # Scenario cd861218ceb2dc1e: Sim agents {2145, 4805, 2150, 2123, 2162, 2131, 4730, 2139} are missing from the simulation.
        # Scenario 46a12cf2da1fdda8: Sim agents {1540, 1544, 4507, 4514, 4545, 1608, 1616, 1623, 1626, 1627, 1500, 1630, 1631, 1632, 1634, 4450, 1636, 4599, 1638, 1639, 4455, 4583, 4585, 4586, 1515, 4461, 1649, 4469, 4597, 4598, 4477} are missing from the simulation.
        watching = {
            "891805f154b4f0dd": [1178],
            "de8c427e65487b93": [432, 569, 554],
            "386f0b2faebe74af": [3361, 3332, 3399, 3371, 3375, 3311, 3345, 3346, 3378, 3319, 3352, 3388],
            "cd861218ceb2dc1e": [2145, 4805, 2150, 2123, 2162, 2131, 4730, 2139],
            "46a12cf2da1fdda8": [
                1540, 1544, 4507, 4514, 4545, 1608, 1616, 1623, 1626, 1627, 1500, 1630, 1631, 1632, 1634, 4450, 1636,
                4599, 1638, 1639, 4455, 4583, 4585, 4586, 1515, 4461, 1649, 4469, 4597, 4598, 4477
            ],
        }
        for r in scenario_rollouts_list:
            sid = r.scenario_id
            if sid in watching:
                obj_ids = {j.object_id for j in r.joint_scenes[10].simulated_trajectories}
                for oid in watching[sid]:
                    assert oid in obj_ids
        # # TODO: Some assertions here to avoid WOSAC error ......

        assert data_dict["raw_scenario_description"][0]['id'] == scenario_rollouts_list[0].scenario_id

        # if not disable_eval:
        #     scenario_id = list(scenario_metrics.keys())
        #
        #     scenario_metrics = {k: scenario_metrics_to_dict(scenario_metrics[k]) for k in scenario_metrics}
        #     aggregate_metrics = {k: aggregate_metrics_to_dict(aggregate_metrics[k]) for k in aggregate_metrics}
        #
        #     stat = {}
        #     for k in scenario_metrics_keys:
        #         stat[f"scenario_metrics/{k}"] = np.mean([d[k] for d in scenario_metrics.values()])
        #     for k in aggregate_metrics_keys:
        #         stat[f"aggregate_metrics/{k}"] = np.mean([d[k] for d in aggregate_metrics.values()])
        #
        #     log_dict_func(
        #         stat,
        #         batch_size=data_dict["encoder/agent_feature"].shape[0],
        #         on_epoch=True,
        #         prog_bar=True,
        #     )
        #
        #     self.metrics.append(stat)

        # print(
        #     "\n=============== RANK {} FINISHED {} SCENARIOS =============".format(global_rank, len(self.metrics))
        # )
        # print("Latest scenario ID: ", scenario_id)
        # for k in self.metrics[0].keys():
        #     print(f"{k}: {np.mean([m[k] for m in self.metrics]):.4f}")
        # print("===========================================================".format(len(self.metrics)))

        self.scenario_rollouts_list.extend(scenario_rollouts_list)
        self.scenario_pb_list.extend(scenario_pb_list)
        if len(self.scenario_rollouts_list) >= self.num_scenarios_per_shard:
            output_dir = logger.log_dir
            self.generate_submission_shard(output_dir, global_rank)

    def on_validation_epoch_end(self, *args, global_rank, logger, trainer, **kwargs):
        if self.metrics:
            print("======== FINAL RESULT RANK {} WITH {} SCENARIOS ==========".format(global_rank, len(self.metrics)))
            for k in self.metrics[0].keys():
                print(f"{k}: {np.mean([m[k] for m in self.metrics]):.4f}")
            print("===========================================================".format(len(self.metrics)))

        output_dir = logger.log_dir
        print(
            f"RANK {global_rank} Storing the final submission files with {len(self.scenario_rollouts_list)} rollouts..."
        )
        import time
        sleep = np.random.randint(1, 5)
        print(f"RANK {global_rank} sleep {sleep} seconds.")
        time.sleep(sleep)
        self.generate_submission_shard(output_dir, global_rank)
        print(f"RANK {global_rank} finished. Entering barrier...")
        # trainer.strategy.barrier()
        print(f"RANK {global_rank} left barrier...")
        # if global_rank == 0:
        print("RANK {} Generated {} shards total.".format(global_rank, self.shard_count))
        print("RANK {} Generated {} scenarios total.".format(global_rank, self.scenario_count))
        print("RANK {} ========== Please manually merge the submission files!!! ==========".format(global_rank))
        output_dir = pathlib.Path(output_dir).resolve()
        print("===============================================================================================\n")
        print("RANK {} Shard submission is saved at: {}".format(global_rank, output_dir))
        print("\n===============================================================================================")
        print("RANK {} Exit.".format(global_rank))

    def generate_submission_shard(self, output_dir, this_rank):
        from waymo_open_dataset.protos import sim_agents_submission_pb2
        account_name = self.config.SUBMISSION.ACCOUNT
        unique_method_name = self.config.SUBMISSION.METHOD_NAME
        num_model_parameters = self.config.SUBMISSION.num_model_parameters
        shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
            scenario_rollouts=self.scenario_rollouts_list,
            submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
            account_name=account_name,
            unique_method_name=unique_method_name,

            # New fields, need changed.
            uses_lidar_data=False,
            uses_camera_data=False,
            uses_public_model_pretraining=False,
            num_model_parameters=num_model_parameters,
            acknowledge_complies_with_closed_loop_requirement=True
        )

        # output_filename = f'submission.binproto-{global_rank:05d}-of-{total_ranks:05d}'
        output_filename = f'submission.binproto-tmp{uuid.uuid4()}'

        scenario_id_list = [s.scenario_id for s in self.scenario_rollouts_list]
        print("Scenario ID to be saved in shard: ", scenario_id_list, output_filename)

        output_dir = pathlib.Path(output_dir).absolute()

        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = pathlib.Path(output_dir) / output_filename
        file_path = file_path.resolve()
        with open(file_path, 'wb') as f:
            f.write(shard_submission.SerializeToString())

        if self.config.SUBMISSION.SAVE_EVAL_DATA and (not self.config.SUBMISSION.GENERATE_SUBMISSION):
            for s in self.scenario_pb_list:
                print("Scenario ID to be saved together apart from shard: ", s.scenario_id)
                file_path = pathlib.Path(output_dir) / "scenario_pb"
                file_path.mkdir(parents=True, exist_ok=True)
                file_path = file_path / f"{s.scenario_id}.binproto"
                file_path = file_path.resolve()
                with open(file_path, 'wb') as f:
                    f.write(s.SerializeToString())

        print("=====================================================================================================\n")
        print("RANK {} Shard submission is saved at: {}".format(this_rank, file_path))
        print("To generate final submission, please manually run:")
        print("\npython -m infgen.merge_shards --output_dir={}".format(output_dir))
        print("\n\nTo see evaluation results, please manually run: (please make sure SUBMISSION.SAVE_EVAL_DATA=True)")
        print("\npython -m infgen.wosac_eval_async --output_dir={}".format(output_dir))
        print("\npython -m infgen.wosac_eval --output_dir={}".format(output_dir))
        print("\n=====================================================================================================")
        # self.output_filenames.append(output_filename)
        self.scenario_rollouts_list = []
        self.scenario_pb_list = []
        self.shard_count += 1
        self.scenario_count += len(scenario_id_list)

    # def generate_submission(self, output_dir, global_rank):
    #     import tarfile
    #
    #     # Once we have created all the shards, we can package them directly into a
    #     # tar.gz archive, ready for submission.
    #     file_name = os.path.join(output_dir, f'rank{global_rank}_submission.tar.gz')
    #     with tarfile.open(file_name, 'w:gz') as tar:
    #         for output_filename in self.output_filenames:
    #             tar.add(os.path.join(output_dir, output_filename), arcname=output_filename)
    #     print("Submission is saved at: {}".format(file_name))
    #
    #     if self.config.EVALUATION.DELETE_EVAL_RESULT:
    #         for f in self.output_filenames:
    #             shutil.rmtree(f)
