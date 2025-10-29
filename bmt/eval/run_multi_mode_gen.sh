#!/bin/bash

# Check if a paper name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <PAPER>"
    echo "PAPER must be one of: SEAL, GOOSE, CAT, STRIVE"
    exit 1
fi

SKIP=false
SKIP_FIRST=false
SKIP_TWO=false

for var in "$@"; do
    if [ "$var" == "--skip" ]; then
        SKIP=true
    fi
    if [ "$var" == "--skip_first" ]; then
        SKIP_FIRST=true
    fi
    if [ "$var" == "--skip_two" ]; then
        SKIP_TWO=true
    fi
done

source /home/xuanhao/anaconda3/etc/profile.d/conda.sh

PAPER=$1

if [ "$SKIP" = false ]; then
    # Define the command for each paper
    case "$PAPER" in
        SEAL)
            export CUDA_VISIBLE_DEVICES="0"
            cd /home/xuanhao/infgen/baselines/SEAL
            CONDA_COMMAND="conda activate SEAL"
            COMMAND="python /home/xuanhao/infgen/baselines/SEAL/cat_advgen.py"
            SECOND_COMMAND="python /home/xuanhao/infgen/baselines/SEAL/write_back.py"
            ;;
        GOOSE)
            export CUDA_VISIBLE_DEVICES="1"
            cd /home/xuanhao/infgen/baselines/SEAL
            CONDA_COMMAND="conda activate SEAL"
            COMMAND="python /home/xuanhao/infgen/baselines/SEAL/cat_advgen.py --goose_adv --AV_traj_num 5"
            SECOND_COMMAND="python /home/xuanhao/infgen/baselines/SEAL/write_back.py"
            ;;
        CAT)
            # export CUDA_VISIBLE_DEVICES="2"
            cd /home/xuanhao/infgen/baselines/cat_scenario_generation
            CONDA_COMMAND="conda activate CAT"
            COMMAND="python /home/xuanhao/infgen/baselines/cat_scenario_generation/adv_trajectory_generation.py  --dir /bigdata/yuxin/scenarionet_waymo_training_500 \
              --num_scenario 500"
            ;;
        STRIVE)
            export CUDA_VISIBLE_DEVICES="3"
            cd /home/xuanhao/infgen/adv_with_ped
            CONDA_COMMAND="conda activate tmp"
            COMMAND="python /home/xuanhao/infgen/adv_with_ped/generate.py"
            ;;
        *)
            echo "Invalid paper name: $PAPER"
            exit 1
            ;;
    esac

    OVERALL_START=$(date +%s)
    # Run the command 6 times with different mode_index
    for MODE_INDEX in {0..5}; do
        START_TIME=$(date +%s)
        OUTPUT_DIR="/bigdata/xuanhao/${PAPER}/${MODE_INDEX}"
        mkdir -p "$OUTPUT_DIR"
        $CONDA_COMMAND
        cd /home/xuanhao/infgen/infgen/eval
        if ! $SKIP_FIRST && ! $SKIP_TWO; then
            if [[ "$PAPER" == "STRIVE" ]]; then
                $COMMAND --save_dir "$OUTPUT_DIR"
            elif [[ "$PAPER" == "CAT" ]]; then
                SEED=$((100 + $MODE_INDEX))
                $COMMAND --save_dir "$OUTPUT_DIR" --seed $SEED
            else
                SEED=$((100 + $MODE_INDEX))
                # weird segfault on scenario 483 when running 0 to 500 in one shot
                $COMMAND --seed $SEED --end_idx 400
                $COMMAND --seed $SEED --start_idx 400
            fi
        fi
        conda activate infgen
        if ! $SKIP_TWO; then
            if [[ "$PAPER" == "SEAL" || "$PAPER" == "GOOSE" ]]; then
                $SECOND_COMMAND --save_dir "$OUTPUT_DIR" --type "$PAPER" --mode_idx "$MODE_INDEX"
            fi
        fi
        # if [[ "$PAPER" == "CAT" ]]; then
        #     conda activate infgen
        #     python /home/xuanhao/infgen/baselines/cat_scenario_generation/rename_output.py --dir "$OUTPUT_DIR"
        # fi
        if [[ "$PAPER" == "CAT" || "$PAPER" == "STRIVE" ]]; then
            rm -rf "$OUTPUT_DIR/tmp"
            rm "/bigdata/xuanhao/${PAPER}/dataset_mapping.pkl"
            rm "/bigdata/xuanhao/${PAPER}/dataset_summary.pkl"
            cd /home/xuanhao/
            conda activate scenarionet
            python -m scenarionet.merge --from "$OUTPUT_DIR" --to "$OUTPUT_DIR/tmp"
            mv $OUTPUT_DIR/tmp/* "/bigdata/xuanhao/${PAPER}"
            rm -rf $OUTPUT_DIR/tmp
        fi
        END_TIME=$(date +%s)
        ELAPSED_TIME=$(($END_TIME - $START_TIME))
        echo "Run completed for $PAPER with mode_index=$MODE_INDEX, output at $OUTPUT_DIR, time taken: $ELAPSED_TIME seconds"
    done
    OVERALL_END=$(date +%s)
    OVERALL_TIME=$(($OVERALL_END - $OVERALL_START))
    echo "Overall time taken: $OVERALL_TIME seconds"
fi

conda activate infgen
# # Define the command for each paper
# case "$PAPER" in
#     SEAL)
#         export CUDA_VISIBLE_DEVICES="5"
#         ;;
#     GOOSE)
#         export CUDA_VISIBLE_DEVICES="6"
#         ;;
#     CAT)
#         export CUDA_VISIBLE_DEVICES="7"
#         ;;
#     STRIVE)
#         export CUDA_VISIBLE_DEVICES="4"
#         ;;
#     *)
#         echo "Invalid paper name: $PAPER"
#         exit 1
#         ;;
# esac
python /home/xuanhao/infgen/infgen/eval/evaluate_scenario_metrics.py eval_mode=$PAPER
