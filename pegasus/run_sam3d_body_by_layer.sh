#!/bin/bash
#PBS -A SSR
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N sam3d_layer_run
#PBS -t 0-4                           # 5 layers: L0, L1, L2, L3, L4
#PBS -o logs/pegasus/sam3d_layer_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/sam3d_layer_${PBS_SUBREQNO}_err.log

# === 相機層級推導脚本 ===
# 每個 node 處理一個相機層級（L0-L4）
# 可以同時啟動 5 個 nodes，分別處理 5 個層級

# === 1. 環境準備 ===
PROJECT_ROOT="/work/SSR/share/code/Skiing_Canonical_DualView_3D_Pose_PyTorch"
cd "${PROJECT_ROOT}"

mkdir -p logs/pegasus/

module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
conda activate /home/SSR/luoxi/miniconda3/envs/sam_3d_body

conda env list

# === 2. 設定相機層級 ===
# 每個 node 處理一個層級
LAYER_INDEX=${PBS_SUBREQNO:-0}

if (( LAYER_INDEX < 0 || LAYER_INDEX > 4 )); then
    echo "[ERROR] LAYER_INDEX=${LAYER_INDEX} out of range (0-4)"
    exit 1
fi

# 每個 node 內的並發線程數
THREADS_PER_NODE=6

echo "==================================="
echo "Layer Index: L${LAYER_INDEX}"
echo "Threads Per Node: ${THREADS_PER_NODE}"
echo "==================================="

# === 3. パス設定と実行 ===
UNITY_DATASET_PATH="/work/SSR/share/data/skiing/skiing_unity_dataset/data"
RESULT_ROOT_PATH="/work/SSR/share/data/skiing/skiing_unity_dataset/sam3d_body_results"
CKPT_ROOT="/work/SSR/share/ckpt/sam-3d-body-dinov3"

echo "🏁 Layer L${LAYER_INDEX} started at: $(date)"
echo "Unity Dataset Path: $UNITY_DATASET_PATH"
echo "Result Root Path: $RESULT_ROOT_PATH"
echo "Checkpoint Root: $CKPT_ROOT"

# 使用 camera_layers 參數指定要處理的層級
python -m SAM3Dbody.main \
    paths.unity.unity_dataset_data_path=${UNITY_DATASET_PATH} \
    paths.unity.unity_sam3d_result_root=${RESULT_ROOT_PATH} \
    model.root_path=${CKPT_ROOT} \
    infer.gpu="[0]" \
    infer.workers_per_gpu=${THREADS_PER_NODE} \
    infer.camera_layers="[${LAYER_INDEX}]"

echo "🏁 Layer L${LAYER_INDEX} finished at: $(date)"
