"""
This file generates a final submission file by merging all the shards.
It also removes duplicate scenarios from the shards.

Usage:

    python -m infgen.merge_shards --output_dir /path/to/output_dir
"""
import glob
import os
import pathlib
import tarfile

import tqdm
from waymo_open_dataset.protos import sim_agents_submission_pb2
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge shards and remove duplicates.')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for the final submission.')
    args = parser.parse_args()
    output_dir = pathlib.Path(args.output_dir)

    existing_scenario_ids = set()

    total_number_of_scenarios = 0

    old_output_filenames = sorted(list(glob.glob(f"{output_dir}/submission.binproto-tmp*")))
    print("Before processing shards, total files: ", len(old_output_filenames))
    count = 0
    new_output_filenames = []
    for old_output_filename in tqdm.tqdm(old_output_filenames, desc="Step 1/3 Reading Shards"):
        new_output_filename = f'submission.binproto-{count + 1:05d}-of-XXXXX'
        to_remove = set()
        with open(old_output_filename, 'rb') as f:
            sub = sim_agents_submission_pb2.SimAgentsChallengeSubmission.FromString(f.read())
        # print(f"File {old_output_filename} has {len(sub.scenario_rollouts)} scenarios.")
        for scenario in sub.scenario_rollouts:
            sid = scenario.scenario_id
            if sid in existing_scenario_ids:
                to_remove.add(sid)
            else:
                existing_scenario_ids.add(sid)
        if to_remove:
            print(f"Removing {len(to_remove)} scenarios from {new_output_filename}!")
            new_rollouts = [s for s in sub.scenario_rollouts if s.scenario_id not in to_remove]
            if len(new_rollouts) > 0:
                # sub.scenario_rollouts = new_rollouts
                sub = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
                    scenario_rollouts=new_rollouts,
                    submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
                    account_name=sub.account_name,
                    unique_method_name=sub.unique_method_name,

                    # New fields, need changed.
                    uses_lidar_data=sub.uses_lidar_data,
                    uses_camera_data=sub.uses_camera_data,
                    uses_public_model_pretraining=sub.uses_public_model_pretraining,
                    num_model_parameters=sub.num_model_parameters,
                    acknowledge_complies_with_closed_loop_requirement=sub.
                    acknowledge_complies_with_closed_loop_requirement,
                )
                total_number_of_scenarios += len(new_rollouts)
                with open(old_output_filename, 'wb') as f:
                    f.write(sub.SerializeToString())
            else:
                print("All scenarios removed from ", old_output_filename)
                print("That is, this new file is useless: ", new_output_filename)
                continue
        else:
            total_number_of_scenarios += len(sub.scenario_rollouts)

        new_path = os.path.join(output_dir, new_output_filename)
        with open(new_path, 'wb') as f:
            f.write(sub.SerializeToString())
        count += 1
        new_output_filenames.append(new_output_filename)

    # Rename all files from submission.binproto-00000-of-XXXXX to submission.binproto-00000-of-00000
    for i, new_output_filename in tqdm.tqdm(enumerate(new_output_filenames), desc="Step 2/3 Rename files"):
        new_path = os.path.join(output_dir, new_output_filename)
        new_output_filename = f'submission.binproto-{i + 1:05d}-of-{len(new_output_filenames):05d}'
        new_path_new = os.path.join(output_dir, new_output_filename)
        os.rename(new_path, new_path_new)
        new_output_filenames[i] = new_output_filename

    print("Total number of unique scenarios: ", len(existing_scenario_ids))
    print("Total number of scenarios: ", total_number_of_scenarios)
    print("After processing shards, total files: ", len(new_output_filenames))
    print("First file: ", new_output_filenames[0])
    print("Last file: ", new_output_filenames[-1])
    # Once we have created all the shards, we can package them directly into a
    # tar.gz archive, ready for submission.
    output_filenames = []
    final_file = os.path.join(output_dir, f'final_submission.tar.gz')
    with tarfile.open(final_file, 'w:gz') as tar:
        for output_filename in tqdm.tqdm(new_output_filenames, desc="Step 3/3 Merging shards"):
            file_name = pathlib.Path(output_filename).name
            tar.add(os.path.join(output_dir, output_filename), arcname=file_name)
            output_filenames.append(output_filename)
    print("Final submission is saved at: {}".format(final_file))
