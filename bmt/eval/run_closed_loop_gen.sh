#!/bin/bash

# Check if a paper name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <PAPER>"
    echo "PAPER must be one of: SEAL, GOOSE, CAT, STRIVE"
    exit 1
fi

PAPER=$1
shift

SKIP=false
SKIP_FIRST=false
SKIP_TWO=false

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --skip)
            SKIP=true
            shift
            ;;
        --skip_first)
            SKIP_FIRST=true
            shift
            ;;
        --skip_two)
            SKIP_TWO=true
            shift
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

source /home/xuanhao/anaconda3/etc/profile.d/conda.sh

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
    for MODE_INDEX in {0..0}; do
        START_TIME=$(date +%s)
        INPUT_DIR="/bigdata/xuanhao/${PAPER}_closed_loop_input"
        OUTPUT_DIR="/bigdata/xuanhao/${PAPER}_closed_loop_output"
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
                $COMMAND --seed $SEED --closed_loop --start_idx 0 --end_idx 1
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