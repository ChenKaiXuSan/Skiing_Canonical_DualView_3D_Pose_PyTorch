#!/bin/bash
#PBS -A SSR
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N fusion_SSM_train
#PBS -o logs/pegasus/train_${PBS_JOBID}.log
#PBS -e logs/pegasus/train_${PBS_JOBID}_err.log

# === 1. 環境準備 ===
PROJECT_ROOT="/work/SSR/share/code/Skiing_Canonical_DualView_3D_Pose_PyTorch"
cd "${PROJECT_ROOT}" || exit 1

mkdir -p logs/pegasus/

source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
conda activate /home/SSR/luoxi/miniconda3/envs/sam_3d_body

conda env list

# === 2. 训练参数（按需修改） ===
DATA_ROOT="/work/SSR/share/data/skiing/skiing_unity_dataset"
INDEX_MAPPING_DIR="${DATA_ROOT}/index_mapping"
INDEX_MAPPING_FILE="camera_pairs_by_action.json"

GPU_ID=0
MAX_EPOCHS=50
NUM_WORKERS=16

# 可选覆盖，留空则使用 configs/train.yaml 的默认值
MODEL_BACKBONE="3dcnn"
FUSE_METHOD="mamba_ssm"
TRAIN_VIEW="multi"

echo "🏁 Train job started at: $(date)"
echo "Project Root: ${PROJECT_ROOT}"
echo "Data Root: ${DATA_ROOT}"
echo "Index Mapping: ${INDEX_MAPPING_DIR}/${INDEX_MAPPING_FILE}"
echo "GPU: ${GPU_ID}, Epochs: ${MAX_EPOCHS}, Workers: ${NUM_WORKERS}"
echo "View: ${TRAIN_VIEW}, Backbone: ${MODEL_BACKBONE}, Fuse: ${FUSE_METHOD}"

# === 3. 执行训练（默认会跑所有 fold） ===
python -m project.main \
    data.root_path=${DATA_ROOT} \
    data.index_mapping=${INDEX_MAPPING_DIR} \
    data.index_mapping_file=${INDEX_MAPPING_FILE} \
    train.gpu=${GPU_ID} \
    train.max_epochs=${MAX_EPOCHS} \
    data.num_workers=${NUM_WORKERS} \
    train.view=${TRAIN_VIEW} \
    model.backbone=${MODEL_BACKBONE} \
    model.fuse_method=${FUSE_METHOD}

echo "🏁 Train job finished at: $(date)"