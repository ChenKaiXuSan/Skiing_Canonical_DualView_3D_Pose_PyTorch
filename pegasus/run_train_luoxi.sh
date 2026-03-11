#!/bin/bash
#PBS -A SSR
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N fusion_SSM_train
#PBS -J 0-4
#PBS -o logs/pegasus/train_${PBS_JOBID}_${PBS_ARRAY_INDEX}.log
#PBS -e logs/pegasus/train_${PBS_JOBID}_${PBS_ARRAY_INDEX}_err.log

# === 1. 環境準備 ===
PROJECT_ROOT="/work/SSR/share/code/Skiing_Canonical_DualView_3D_Pose_PyTorch"
cd "${PROJECT_ROOT}" || exit 1

mkdir -p logs/pegasus/

source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
conda activate /home/SSR/luoxi/miniconda3/envs/multiview-video-cls

conda env list

# === 2. 训练参数（按需修改） ===
DATA_ROOT="/work/SSR/share/data/skiing/skiing_unity_dataset"
INDEX_MAPPING_DIR="${DATA_ROOT}/index_mapping"
INDEX_MAPPING_FILE="camera_pairs_by_action.json"

MAX_EPOCHS=50

MODEL_BACKBONE="3dcnn"
FUSE_METHOD="mamba_ssm"

NUM_WORKERS=16
BATCH_SIZE=16

# fold assignment:
# - PBS array mode: use PBS_ARRAY_INDEX
# - non-array/manual mode: allow env FOLD_ID override, default 0
FOLD_ID=${PBS_ARRAY_INDEX:-${FOLD_ID:-0}}

echo "🏁 Train job started at: $(date)"
echo "Project Root: ${PROJECT_ROOT}"
echo "Data Root: ${DATA_ROOT}"
echo "Index Mapping: ${INDEX_MAPPING_DIR}/${INDEX_MAPPING_FILE}"
echo "GPU: 0, Epochs: ${MAX_EPOCHS}, Workers: ${NUM_WORKERS}"
echo "Backbone: ${MODEL_BACKBONE}, Fuse: ${FUSE_METHOD}"
echo "Fold: ${FOLD_ID}"

# === 3. 执行训练（每个作业只跑一个 fold） ===
python -m project.main \
    data.root_path=${DATA_ROOT} \
    data.index_mapping=${INDEX_MAPPING_DIR} \
    data.index_mapping_file=${INDEX_MAPPING_FILE} \
    train.gpu=0 \
    train.max_epochs=${MAX_EPOCHS} \
    data.num_workers=${NUM_WORKERS} \
    data.batch_size=${BATCH_SIZE} \
    model.backbone=${MODEL_BACKBONE} \
    model.fuse_method=${FUSE_METHOD} \
    train.fold=${FOLD_ID}


echo "🏁 Train job finished at: $(date)"