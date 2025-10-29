# This script should be called from the repo root:
# cd ~/infgen
# bash scripts/xxx.sh


export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

filename=$(basename "$0")
extension="${filename##*.}"

EXP_NAME="${filename%.*}"
#EXP_NAME="1030_gpt_Delta256_NoAgentHistory_WTLSingle"

nohup python infgen/train_motion.py \
--config-name='0202_midgpt' \
exp_name=${EXP_NAME} \
wandb=True \
\
limit_val_batches=500 \
\
DATA.TRAINING_DATA_DIR="/data_zhenghao/datasets/scenarionet/waymo/training/" \
DATA.TEST_DATA_DIR="/data_zhenghao/datasets/scenarionet/waymo/validation" \
\
\
batch_size=2 \
val_batch_size=4 \
\
REMOVE_AGENT_FROM_SCENE_ENCODER=True \
\
TOKENIZATION.TOKENIZATION_METHOD="BicycleModelTokenizerFixed0124" \
\
USE_MOTION=True \
EVAL_MOTION=True \
\
pretrain="/bigdata/zhenghao/infgen/lightning_logs/infgen/0205_MidGPT_V18_WBackward_2025-02-05/checkpoints"  \
\
BACKWARD_PREDICTION=True \
\
\
> ${EXP_NAME}.log 2>&1 &
