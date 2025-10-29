import argparse
import copy
import functools
import os
import pathlib
import pickle

import gradio as gr
import numpy as np
import torch
import uuid
from omegaconf import OmegaConf

from infgen.dataset.preprocess_action_label import TurnAction
from infgen.dataset.preprocessor import preprocess_scenario_description_for_motionlm
from infgen.gradio_ui.plot import plot_gt, plot_pred, create_animation_from_pred, create_animation_from_gt, plot_map
from infgen.utils import REPO_ROOT
from infgen.utils import utils


os.environ['GRADIO_TEMP_DIR'] = str(REPO_ROOT / "gradio_tmp")

dpi = 100

parser = argparse.ArgumentParser()
parser.add_argument("--share", action="store_true", help="Enable sharing")
parser.add_argument("--default_ckpt", "--default_model", "--ckpt", type=str, default="/bigdata/zhenghao/infgen/lightning_logs/infgen/0205_MidGPT_V18_WBackward_2025-02-05/checkpoints")
parser.add_argument("--default_data", "--data", type=str, default="data/20scenarios")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--title", type=str, default="Motion Generation")
parser.add_argument("--default_config_path", type=str, default="0202_midgpt.yaml")
parser.add_argument(
    "--display_video", "--video", action="store_true", help="by default display static images, can display mp4 video"
)
args = parser.parse_args()

default_config_path = args.default_config_path
default_config = OmegaConf.load(REPO_ROOT / "cfgs" / default_config_path)

OmegaConf.set_struct(default_config, False)
default_config.PREPROCESSING.keep_all_data = True
default_config.ROOT_DIR = str(REPO_ROOT.resolve())
OmegaConf.set_struct(default_config, True)

DEFAULT_DATA_PATH = args.default_data or "/bigdata/datasets/scenarionet/waymo/validation_interactive/validation_interactive_0"
DEFAULT_MODEL = args.default_ckpt or None

NUM_OF_MODES = 6
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LENGTH = 1000


class State:
    model = None
    model_path = None
    config: dict = default_config
    dataset_path: pathlib.Path = REPO_ROOT / DEFAULT_DATA_PATH

    scenario = None

    raw_data_files = None
    data_files = None

    raw_data_dict = None
    data_dict = None

    default_config: dict = default_config

    # click support
    sel_point = None
    xlim, ylim = None, None
    fig_width, fig_height = None, None  # in inches
    fig_dpi = None
    bbox_x0, bbox_y0, bbox_w, bbox_h = None, None, None, None
    original_img = None
    modified_img = None

    predict_count = 0


state = State()


def ckpt_callback(ckpt_path):
    from infgen.models.motionlm_lightning import MotionLMLightning

    msg = "Failed!"
    temperature = 1.0
    safe_agents = ""
    turn_agents = ""
    main_vis = None
    sampling_method = "topp"

    if ckpt_path.lower() == "debug":
        try:
            config = copy.deepcopy(state.default_config)
            OmegaConf.set_struct(config, False)
            config.MODEL.D_MODEL = 32
            config.MODEL.NUM_DECODER_LAYERS = 1
            config.MODEL.NUM_ATTN_LAYERS = 1
            # config.ACTION_LABEL.USE_SAFETY_LABEL = True
            # config.ACTION_LABEL.USE_ACTION_LABEL = True
            OmegaConf.set_struct(config, True)
            model = MotionLMLightning(config)
            model = model.to(device)
            msg = "DEBUG MODEL LOADED!"
            config = model.config
            temperature = config.SAMPLING.TEMPERATURE
            state.model = model
            state.config = config
            sampling_method = config.SAMPLING.SAMPLING_METHOD
        except Exception as e:
            print("Error: ", e)
            msg = "Failed to load DEBUG model!"

        return [msg, sampling_method, temperature, main_vis] + [""] * 7 + [0.0] * 5

    ckpt_path = ckpt_path.replace("\\", "")
    path = pathlib.Path(ckpt_path)
    path = REPO_ROOT / path

    if path.is_dir():
        path = path / "last.ckpt"

    print("Loading model from: ", path.absolute())
    if not path.exists():
        msg = "{} does not exist!".format(path)
        return [msg, sampling_method, temperature, main_vis] + [""] * 7 + [0.0] * 5

    try:
        # model = utils.get_model(
        #     config=None, checkpoint_path=path, device=device, default_config=default_config_path
        # ).eval()
        model = utils.get_model(checkpoint_path=path).eval()
        msg = "Model loaded successfully!"
        config = model.config
        temperature = config.SAMPLING.TEMPERATURE
        state.model = model
        state.config = config
        sampling_method = config.SAMPLING.SAMPLING_METHOD

        infer_heading_params = (
            config.TOKENIZATION.MIN_SPEED,  #or 0.0,
            config.TOKENIZATION.MAX_HEADING_DIFF,
            config.TOKENIZATION.MIN_DISPLACEMENT_INIT,  # or 0.0,
            config.TOKENIZATION.MIN_DISPLACEMENT,  #or 0.0,
            config.TOKENIZATION.SMOOTH_FACTOR,  #or 1.0,
        )

    except Exception as e:
        print("Error: ", e)
        raise e
        msg = "Failed to load model!"
        infer_heading_params = [0.0] * 5

    return [msg, sampling_method, temperature, main_vis] + [""] * 7 + list(infer_heading_params)


def on_dataset_path_submit(path):

    print(state, type(state))

    FAILED_MSG = "Failed!"

    path = pathlib.Path(path)
    path = REPO_ROOT / path

    if not path.exists():
        return FAILED_MSG

    if not path.is_dir():
        return FAILED_MSG

    state.dataset_path = path
    print(state.dataset_path)

    files = os.listdir(path)
    files = [f for f in files if f.endswith(".pkl")]
    print("Files: ", files)

    if not hasattr(state, "count"):
        state.count = 0
    state.count += 1

    state.predict_count = 0

    return [
        "Dataset with {} Scenarios Listed!".format(len(files)),
        gr.FileExplorer(
            file_count="single",
            root_dir=state.dataset_path,
            # root_dir=path,
            glob="**/*.pkl",
            scale=1
            # label="UPDATED={}".format(state.count),
            # interactive=True
        )
    ]


def on_data_file_name_search(search):
    return gr.FileExplorer(file_count="single", root_dir=state.dataset_path, glob=f"**/*{search}*.pkl", scale=1)


def on_data_file_select(file_path):
    if not file_path:
        return (None, ) + ("", ) * 2

    file_path = pathlib.Path(file_path)
    assert state.dataset_path is not None
    file_path = state.dataset_path / file_path

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    state.scenario = data
    scenario_data_dict = preprocess_scenario_description_for_motionlm(
        scenario=data,
        config=state.config,
        in_evaluation=True,
        keep_all_data=True,
        # cache=None,
        tokenizer=state.model.model.tokenizer,
        backward_prediction=False
    )

    state.raw_data_dict = scenario_data_dict

    if args.display_video:
        gt_gif_path = create_animation_from_gt(
            scenario_data_dict,
            save_path=str(REPO_ROOT / "gradio_tmp" / "gt_animation_{}.mp4".format(uuid.uuid4())),
            dpi=dpi,
            # draw_non_ooi=False
        )
        return (
            gr.Video(value=gt_gif_path, label="Ground Truth Trajectories",
                     autoplay=True)  # Use gr.Video instead of gr.Image
        )
    else:
        img, info_dict = plot_gt(scenario_data_dict, get_info=True, draw_map_only=True)
        state.original_img = img
        state.xlim = info_dict["xlim"]
        state.ylim = info_dict["ylim"]
        state.fig_width, state.fig_height = info_dict["fig_size"]
        state.fig_dpi = info_dict["fig_dpi"]
        state.bbox_x0 = info_dict["bbox_x0"]
        state.bbox_y0 = info_dict["bbox_y0"]
        state.bbox_w = info_dict["bbox_w"]
        state.bbox_h = info_dict["bbox_h"]
        return (gr.Image(value=img, label=data["id"]))
    

def on_generate_button_click(
    # MIN_SPEED, MAX_HEADING_DIFF, MIN_DISPLACEMENT_INIT, MIN_DISPLACEMENT, smooth_factor,
    sampling_method,
    temperature,
    seed,
    only_draw_gt,
    only_draw_detokenized,
    draw_backward_prediction=False,
    draw_SCGen=False,
    draw_reactive_SCGen=True,
):

    # TODO: Seed is not respect!
    # TODO: Seed is not respect!
    # TODO: Seed is not respect!

    assert sampling_method in ["softmax", "topp"], "Invalid sampling method! {}".format(sampling_method)

    if state.scenario is None:
        return (
            None,
            "Data is not loaded!",
        ) + (None, ) * 2

    if (not state.config.get("BACKWARD_PREDICTION", False)) and draw_backward_prediction:
        print("BACKWARD_PREDICTION is not enabled in the config!")
        return (
            None,
            "BACKWARD_PREDICTION is not enabled in the config!",
        ) + (None, ) * 2

    model = state.model
    if model is None and not only_draw_gt:
        return (
            None,
            "Model is not loaded!",
        ) + (None, ) * 2

    # if smooth_factor:
    config = copy.deepcopy(state.config)
    OmegaConf.set_struct(config, False)
    # state.config.TOKENIZATION.SMOOTH_FACTOR = None
    # state.config.TOKENIZATION.MIN_DISPLACEMENT = MIN_DISPLACEMENT
    # state.config.TOKENIZATION.MIN_DISPLACEMENT_INIT = MIN_DISPLACEMENT_INIT
    # state.config.TOKENIZATION.MAX_HEADING_DIFF = MAX_HEADING_DIFF
    # state.config.TOKENIZATION.MIN_SPEED = MIN_SPEED
    OmegaConf.set_struct(config, True)

    vis_ooi = None
    print("draw_backward_prediction", draw_backward_prediction)

    data_dict = preprocess_scenario_description_for_motionlm(
        scenario=copy.deepcopy(state.scenario),
        config=config,
        in_evaluation=True,
        keep_all_data=True,
        # cache=None,
        backward_prediction=draw_backward_prediction,
        tokenizer=model.model.tokenizer
    )

    if only_draw_gt:
        output_dict = data_dict

        if args.display_video:
            video_path = str(REPO_ROOT / "gradio_tmp" / "gt_animation_{}.mp4".format(uuid.uuid4()))
            video_path = create_animation_from_gt(output_dict, save_path=video_path, dpi=dpi)
            print("gt_mp4_path", video_path)
        else:
            img = plot_gt(output_dict)

    else:
        input_data_dict = copy.deepcopy(data_dict)
        input_data_dict = utils.numpy_to_torch(input_data_dict, device)

        # Extend the batch dim:
        input_data_dict = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in input_data_dict.items()}
        input_data_dict["in_evaluation"] = torch.tensor([1], dtype=bool).to(device)
        input_data_dict["in_backward_prediction"] = torch.tensor([draw_backward_prediction], dtype=bool).to(device)

        if not only_draw_detokenized:

            if draw_SCGen or draw_reactive_SCGen:
                NUM_MODE = 1
                # SCGen rollout
                from infgen.utils.safety_critical_generation_utils import create_new_adv, _get_mode, _overwrite_datadict_all_agents


                mode_count = 0
                mode_success_count = 0

                while True:
                    mode_count += 1

                    cur_data_dict = copy.deepcopy(data_dict)
                    cur_data_dict, adv_id = create_new_adv(cur_data_dict)
                    coll_step = 90

                    input_data_dict = utils.numpy_to_torch(cur_data_dict, device)
                    input_data_dict["in_evaluation"] = torch.tensor([1], dtype=bool).to(device)


                    # Extend the batch dim:
                    input_data_dict = {
                        k: utils.expand_for_modes(v.unsqueeze(0), num_modes=NUM_MODE) if isinstance(v, torch.Tensor) else v
                        for k, v in input_data_dict.items()
                    }

                    # Force to run backward prediction first to make sure the data is tokenized correctly.
                    tok_data_dict, _ = model.model.tokenizer.tokenize(input_data_dict, backward_prediction=True)
                    input_data_dict.update(tok_data_dict)

                    input_data_dict["in_backward_prediction"] = torch.tensor([True] * NUM_MODE, dtype=bool).to(model.device)
                    not_teacher_forcing_ids = [adv_id] # all_agents_except_sdc # [adv_id]

                    with torch.no_grad():
                        ar_func = model.model.autoregressive_rollout_backward_prediction_with_replay
                        backward_output_dict = ar_func(
                            input_data_dict,
                            # num_decode_steps=None,
                            sampling_method=config.SAMPLING.SAMPLING_METHOD,
                            temperature=config.SAMPLING.TEMPERATURE,
                            topp=config.SAMPLING.TOPP,
                            not_teacher_forcing_ids=not_teacher_forcing_ids,
                        
                        )

                    backward_output_dict = model.model.tokenizer.detokenize(
                        backward_output_dict,
                        detokenizing_gt=False,
                        backward_prediction=True,
                        flip_wrong_heading=config.TOKENIZATION.FLIP_WRONG_HEADING,
                        teacher_forcing=True
                    )

                    curvature_threshold = 0.8
                    # step 1: filter short trajectories; skip, if adv goes less than 9m in 9 seconds.
                    adv_traj = backward_output_dict["decoder/reconstructed_position"][:,:,-1][backward_output_dict["decoder/reconstructed_valid_mask"][:,:,-1]] # (T_valid, 2)
                    displacements = torch.norm(torch.diff(adv_traj, dim=0), dim=1) + 1e-6

                    heading = backward_output_dict["decoder/reconstructed_heading"][:,:,-1][backward_output_dict["decoder/reconstructed_valid_mask"][:,:,-1]] 
                    heading_diffs = torch.abs(torch.diff(heading))
                    heading_diffs = torch.minimum(heading_diffs, 2*torch.pi - heading_diffs)
                    curvatures = heading_diffs / displacements

                    adv_dist = torch.linalg.norm(adv_traj[-1, :] - adv_traj[0, :], dim=-1)
                    adv_dist = adv_dist.mean()  # Shape (1,)

                    if adv_dist < coll_step/10 or torch.max(curvatures).item() > curvature_threshold:
                        print("failed cases:")
                        continue

                    print("Success after at mode", mode_count)
                    break

                output_dict = backward_output_dict


            if draw_reactive_SCGen:

                # step 2: overwrite the agent's trajectory to GT and then start forward
                forward_input_dict = _overwrite_datadict_all_agents(source_data_dict=backward_output_dict, dest_data_dict=input_data_dict, ooi=[adv_id])

                # Force to run forward prediction first to make sure the data is tokenized correctly.
                tok_data_dict, _ = model.model.tokenizer.tokenize(forward_input_dict, backward_prediction=False)
                forward_input_dict.update(tok_data_dict)

                # run forward with teacher forcing
                forward_input_dict["in_backward_prediction"] = torch.tensor([False] * NUM_MODE, dtype=bool).to(model.device)
                
                # not_teacher_forcing_ids = [] # we are teacher forcing all agents
                all_agents = cur_data_dict["decoder/agent_id"]
                all_agents_except_sdc = all_agents[all_agents != 0]
                all_agents_except_sdc_adv = all_agents_except_sdc[all_agents_except_sdc != adv_id]
                not_teacher_forcing_ids = all_agents_except_sdc_adv
                print("not_teacher_forcing_ids for forward", not_teacher_forcing_ids)

                alpha=5
                with torch.no_grad():
                    ar_func = model.model.autoregressive_rollout_with_replay
                    forward_output_dict = ar_func(
                        forward_input_dict,
                        # num_decode_steps=None,
                        sampling_method=config.SAMPLING.SAMPLING_METHOD,
                        temperature=config.SAMPLING.TEMPERATURE,
                        topp=config.SAMPLING.TOPP,
                        not_teacher_forcing_ids=not_teacher_forcing_ids,
                        alpha=alpha
                    )

                forward_output_dict = model.model.tokenizer.detokenize( # forward detokenize
                    forward_output_dict,
                    detokenizing_gt=False,
                    backward_prediction=False,
                    flip_wrong_heading=config.TOKENIZATION.FLIP_WRONG_HEADING,
                    teacher_forcing=True # we are teacher forcing all agents
                )

                output_dict = forward_output_dict



            else:

                with torch.no_grad():
                    output_dict = model.model.autoregressive_rollout(
                        input_data_dict,
                        # num_decode_steps=None,
                        sampling_method=sampling_method,
                        temperature=temperature,
                        backward_prediction=draw_backward_prediction,
                    )

                output_dict = model.model.tokenizer.detokenize(
                    output_dict, detokenizing_gt=False, backward_prediction=draw_backward_prediction, flip_wrong_heading=config.TOKENIZATION.FLIP_WRONG_HEADING, teacher_forcing=False
                )


        else:
            # ===== DEBUG =====
            output_dict = input_data_dict
            output_dict["decoder/output_action"] = output_dict["decoder/target_action"]
            fill_zero = ~output_dict["decoder/target_action_valid_mask"]
            output_dict["decoder/input_action_valid_mask"][fill_zero] = False

            output_dict = model.model.tokenizer.detokenize(output_dict, detokenizing_gt=True)

        output_dict = {
            k: (v.squeeze(0).cpu().numpy() if isinstance(v, torch.Tensor) else v)
            for k, v in output_dict.items()
        }

        # # TODO: now, post-process the whole ADV trajectory for better collision position
        # from infgen.utils.safety_critical_generation_utils import post_process_adv_traj
        # output_dict = post_process_adv_traj(output_dict, adv_id=adv_id) # TODO: test the post-processing

        if args.display_video:
            video_path = str(REPO_ROOT / "gradio_tmp" / "pred_animation_{}.mp4".format(uuid.uuid4()))
            video_path = create_animation_from_pred(output_dict, save_path=video_path, dpi=dpi)
            print("predict gif path:", video_path)

        else:
            img = plot_pred(output_dict, ooi=vis_ooi)

    if args.display_video:
        return (
            # gr.Image(value=img, label=state.scenario["id"]),
            gr.Video(
                value=video_path,
                label="Generated Trajectory Prediction",
                show_download_button=True,
                width=LENGTH,
                height=LENGTH,
                interactive=False,
                format="mp4",
                autoplay=True,
            ),  # Use gr.Video instead of gr.Image for GIF
            "Scenario Generated!"
        )
    else:
        return (
            gr.Image(value=img, label=state.scenario["id"]),
            "Scenario Generated!"
        )


import cv2
from PIL import Image


def on_scenario_vis_click(event: gr.SelectData):
    # Get axes relative coordinates (0 to 1 within the axes)
    width, height = state.fig_width * state.fig_dpi, state.fig_height * state.fig_dpi
    if state.fig_dpi is not None:
        # Convert to relative figure coordinates (0 to 1 within figure)
        x_rel_fig = (event.index[0] - state.bbox_x0 * width) / (state.bbox_w * width)
        y_rel_fig = (event.index[1] - state.bbox_y0 * height) / (state.bbox_h * height)  # note that y-axis is inverted
        print(
            f"Data: ({event.index[0]:.2f}, {event.index[1]:.2f}), Relative to Figure: ({x_rel_fig:.2f}, {y_rel_fig:.2f})"
        )
        print(f"xlim: {state.xlim}, ylim: {state.ylim}")
        state.sel_point = (
            x_rel_fig * (state.xlim[1] - state.xlim[0]) + state.xlim[0],
            y_rel_fig * (state.ylim[0] - state.ylim[1]) + state.ylim[1]
        )
        print(state.sel_point)
        cv2_original = cv2.cvtColor(np.array(state.original_img), cv2.COLOR_BGR2RGB)
        cv2.circle(cv2_original, event.index, 10, (0, 0, 255), -1)
        state.modified_img = Image.fromarray(cv2.cvtColor(cv2_original, cv2.COLOR_BGR2RGB))
        return state.modified_img, state.sel_point
    else:
        print("Select a scenario first")
        return gr.update()


def on_clear_click():
    raise ValueError()
    state.sel_point = None
    state.modified_img = None
    return state.original_img, state.sel_point


# ============================================================
# ======================== GRADIO UI =========================
# ============================================================
with gr.Blocks(theme=gr.themes.Soft(text_size="lg"), title=args.title) as demo:
    with gr.Group():
        gr.Markdown("  ## Data")
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(label="Path to Dataset Folder", value=DEFAULT_DATA_PATH)

            with gr.Column(scale=1):
                out = gr.Textbox(label="Status", placeholder="Enter to submit...")

        # gr.Markdown("## Visualization")
        with gr.Row(equal_height=True):  # Future release fix: https://github.com/gradio-app/gradio/pull/9577
            with gr.Column(scale=1):
                with gr.Group():
                    file_name_input = gr.Textbox(label="Search Scenario ID", max_lines=1)
                    file_explorer = gr.FileExplorer(
                        root_dir=state.dataset_path,
                        glob="**/*.pkl",
                        file_count="single",
                        interactive=True,
                        container=True,
                        max_height=900
                    )
            with gr.Column(scale=2):
                with gr.Row():
                    sel_point = gr.Textbox(
                        label="Selected Point",
                        interactive=False,
                        placeholder="Click on scenario map to select a point"
                    )
                    clear = gr.Button(value="Clear Selection", interactive=True, visible=True)

                if args.display_video:
                    gt_vis = gr.Video(
                        label="Ground Truth Trajectories",
                        show_download_button=True,
                        width=LENGTH,
                        height=LENGTH,
                        interactive=False,
                        format="mp4"
                    )
                else:
                    gt_vis = gr.Image(
                        label="Original Scenario",
                        show_download_button=True,
                        width=LENGTH,
                        height=LENGTH,
                        interactive=False
                    )

    with gr.Group():
        gr.Markdown("## Model")
        with gr.Row():
            with gr.Column(scale=3):
                if DEFAULT_MODEL:
                    ckpt_input = gr.Textbox(
                        label="Path to model checkpoint", value=DEFAULT_MODEL, placeholder="/home/.../last.ckpt"
                    )
                else:
                    ckpt_input = gr.Textbox(
                        label="Path to model checkpoint",
                        placeholder="/home/.../last.ckpt (Type 'debug' for debug model!)"
                    )
            with gr.Column(scale=1):
                ckpt_output = gr.Textbox(label="Status", placeholder="Enter to load...")

        gr.Markdown("## Visualization")
        with gr.Row():
            with gr.Column(scale=1):
                sampling_method = gr.Radio(label="Sampling Method", choices=["softmax", "topp"], value="topp")
                temperature = gr.Slider(
                    label="Sampling Temperature", minimum=0.0, maximum=2.0, step=0.1, value=1.0, interactive=True
                )
                seed = gr.Number(label="Seed (TODO: not used)", value=42, precision=0)
                degree_input = gr.Number(label="Inject Heading Degree Value", value=0, precision=0)

                gr.Markdown(
                    "### Agents ID to assign labels:\n1. Split by comma ','\n2. If empty, original labels are used and printed as `[RAW]`"
                )
                generate_button = gr.Button(value="Generate")
                generate_backward_prediction_button = gr.Button(value="Generate Backward Prediction")

                generate_safety_critical_button = gr.Button(value="Generate SC Prediction")
                generate_reactive_safety_critical_button = gr.Button(value="Simulate Reactive SC")
                draw_detok_button = gr.Button(value="Draw Raw Detokenized Scenario")
                draw_map_button = gr.Button(value="Draw Raw Map")

            with gr.Column(scale=2):
                main_vis_text = gr.Textbox(label="Status", placeholder="", interactive=False)
                if args.display_video:
                    main_vis = gr.Video(
                        label="Generated Scenario",
                        show_download_button=True,
                        width=LENGTH,
                        height=LENGTH,
                        format="mp4"
                    )
                else:
                    main_vis = gr.Image(
                        label="Generated Scenario", show_download_button=True, width=LENGTH, height=LENGTH
                    )

    inp.submit(on_dataset_path_submit, inputs=inp, outputs=[out, file_explorer])
    file_name_input.change(on_data_file_name_search, inputs=file_name_input, outputs=file_explorer)
    file_explorer.change(
        on_data_file_select,
        inputs=file_explorer,
        outputs=[
            gt_vis
        ],
    )

    ckpt_input.submit(
        ckpt_callback,
        inputs=ckpt_input,
        outputs=[
            ckpt_output,
            sampling_method,
            temperature,
            main_vis,
        ],
    )

    # TODO: Interesting feature to allow user click the scenario map to select a point
    # gt_vis.select(on_scenario_vis_click, outputs=[gt_vis, sel_point])
    # clear.click(on_clear_click, outputs=[gt_vis, sel_point])

    generate_button.click(
        functools.partial(on_generate_button_click, only_draw_gt=False, only_draw_detokenized=False),
        inputs=[
            sampling_method,
            temperature,
            seed,
            
        ],
        outputs=[
            main_vis,
            main_vis_text,
        ],
    )

    generate_backward_prediction_button.click(
        functools.partial(
            on_generate_button_click, only_draw_gt=False, only_draw_detokenized=False, draw_backward_prediction=True
        ),
        inputs=[
            sampling_method,
            temperature,
            seed,
        ],
        outputs=[
            main_vis,
            main_vis_text,
        ],
    )
    

    generate_safety_critical_button.click(
        functools.partial(
            on_generate_button_click, only_draw_gt=False, only_draw_detokenized=False, draw_backward_prediction=True, draw_SCGen=True
        ),
        inputs=[
            sampling_method,
            temperature,
            seed,
        ],
        outputs=[
            main_vis,
            main_vis_text,
        ],
    )


    generate_reactive_safety_critical_button.click(
        functools.partial(
            on_generate_button_click, only_draw_gt=False, only_draw_detokenized=False, draw_backward_prediction=True, draw_SCGen=True, draw_reactive_SCGen=True
        ),
        inputs=[
            sampling_method,
            temperature,
            seed,
        ],
        outputs=[
            main_vis,
            main_vis_text,
        ],
    )


    draw_detok_button.click(
        functools.partial(on_generate_button_click, only_draw_gt=False, only_draw_detokenized=True),
        inputs=[
            sampling_method,
            temperature,
            seed,
        ],
        outputs=[
            main_vis,
            main_vis_text,
        ],
    )

    draw_map_button.click(
        functools.partial(on_generate_button_click, only_draw_gt=False, only_draw_detokenized=False, only_draw_map=True),
        inputs=[
            sampling_method,
            temperature,
            seed,
        ],
        outputs=[
            main_vis,
            main_vis_text,
        ],
    )

if DEFAULT_MODEL:
    print("Loading default model from: ", DEFAULT_MODEL)
    ckpt_callback(DEFAULT_MODEL)

demo.queue().launch(server_port=args.port, share=args.share)

