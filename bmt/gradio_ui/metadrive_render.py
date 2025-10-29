import random
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs import ScenarioEnv
import argparse
import mediapy
import numpy as np
from infgen.utils import REPO_ROOT
from itertools import count

extra_args = dict(film_size=(900, 600), screen_size=(900, 600))


def render(input_dir, output_path):
    goose = False
    try:
        env = ScenarioEnv(
            {
                "manual_control": False,
                "reactive_traffic": False,
                "use_render": False,
                "agent_policy": ReplayEgoCarPolicy,
                # "data_directory": REPO_ROOT / input_dir,
                # "data_directory": "/bigdata/xuanhao/SEAL",
                "data_directory": "/bigdata/xuanhao/GOOSE" if goose else "/bigdata/xuanhao/SEAL",
                "sequential_seed": True,
                "num_scenarios": 371 if not goose else 336,
                # "start_scenario_index": 7,
            }
        )
        for i in count(0, 1):
            o, _ = env.reset(seed=i)
            if env.engine.data_manager.current_scenario['id'] != "10af3d70d93ef629":
                continue
            frames = []
            for i in range(1, 100000):
                o, r, tm, tc, info = env.step([1.0, 0.])
                frame = env.render(mode="top_down", **extra_args)
                frames.append(frame)
                if tm or tc:
                    break
            break

    except Exception as e:
        raise e
    finally:
        env.close()

    imgs = np.stack([frame for frame in frames], axis=0)
    mediapy.write_video(REPO_ROOT / output_path, imgs, fps=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    render(args.input_dir, args.output_path)
