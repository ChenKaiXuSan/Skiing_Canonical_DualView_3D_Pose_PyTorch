#!/bin/bash
#PBS -A SSR
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N sam3d_fine_grained
#PBS -t 0-119                         # 2 persons × 12 actions × 5 layers = 120 nodes
#PBS -o logs/pegasus/sam3d_fine_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/sam3d_fine_${PBS_SUBREQNO}_err.log

# === 細粒度推導脚本 ===
# 每個 node 處理：1個人物 + 1個動作 + 1個相機層級
# 總共 120 個 nodes = 2 persons × 12 actions × 5 layers

# === 1. 環境準備 ===
PROJECT_ROOT="/work/SSR/share/code/Skiing_Canonical_DualView_3D_Pose_PyTorch"
cd "${PROJECT_ROOT}"

mkdir -p logs/pegasus/

module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
conda activate /home/SSR/luoxi/miniconda3/envs/sam_3d_body

# === 2. 定義人物和動作列表 ===
PERSONS=("male" "female")
ACTIONS=(
    "Anim_Male_Skier_Braking"
    "Anim_Male_Skier_Braking_Loop"
    "Anim_Male_Skier_Fall"
    "Anim_Male_Skier_From_Idle_To_Skiing"
    "Anim_Male_Skier_From_Losing_To_Idle"
    "Anim_Male_Skier_From_Skiing_To_Straight"
    "Anim_Male_Skier_From_Winning_To_Idle"
    "Anim_Male_Skier_Idle"
    "Anim_Male_Skier_Losing"
    "Anim_Male_Skier_Skiing"
    "Anim_Male_Skier_Skiing_Straight"
    "Anim_Male_Skier_Winning"
)

# === 3. 從 NODE_INDEX 計算人物、動作、層級 ===
NODE_INDEX=${PBS_SUBREQNO:-0}

if (( NODE_INDEX < 0 || NODE_INDEX >= 120 )); then
    echo "[ERROR] NODE_INDEX=${NODE_INDEX} out of range (0-119)"
    exit 1
fi

# 計算公式：
# - PERSON_INDEX = NODE_INDEX / 60  (0-1)
# - ACTION_INDEX = (NODE_INDEX % 60) / 5  (0-11)
# - LAYER_INDEX = NODE_INDEX % 5  (0-4)

PERSON_INDEX=$((NODE_INDEX / 60))
ACTION_INDEX=$(((NODE_INDEX % 60) / 5))
LAYER_INDEX=$((NODE_INDEX % 5))

PERSON=${PERSONS[$PERSON_INDEX]}
ACTION=${ACTIONS[$ACTION_INDEX]}

# 將男性動作名稱轉換為女性（如果需要）
if [ "$PERSON" == "female" ]; then
    ACTION=${ACTION/Male/Female}
fi

# 每個 node 內的並發線程數
THREADS_PER_NODE=6

echo "========================================="
echo "Node Index: ${NODE_INDEX}"
echo "Person: ${PERSON} (index=${PERSON_INDEX})"
echo "Action: ${ACTION} (index=${ACTION_INDEX})"
echo "Layer: L${LAYER_INDEX}"
echo "Threads Per Node: ${THREADS_PER_NODE}"
echo "========================================="

# === 4. パス設定と実行 ===
UNITY_DATASET_PATH="/work/SSR/share/data/skiing/skiing_unity_dataset/data"
RESULT_ROOT_PATH="/work/SSR/share/data/skiing/skiing_unity_dataset/sam3d_body_results"
CKPT_ROOT="/work/SSR/share/ckpt/sam-3d-body-dinov3"

echo "🏁 Started at: $(date)"
echo "Unity Dataset Path: $UNITY_DATASET_PATH"
echo "Result Root Path: $RESULT_ROOT_PATH"
echo "Checkpoint Root: $CKPT_ROOT"

# 使用 person_filter, action_filter, camera_layers 精確指定要處理的範圍
python -m SAM3Dbody.main \
    paths.unity.unity_dataset_data_path=${UNITY_DATASET_PATH} \
    paths.unity.unity_sam3d_result_root=${RESULT_ROOT_PATH} \
    model.root_path=${CKPT_ROOT} \
    infer.gpu="[0]" \
    infer.workers_per_gpu=${THREADS_PER_NODE} \
    infer.camera_layers="[${LAYER_INDEX}]" \
    infer.person_filter="${PERSON}" \
    infer.action_filter="${ACTION}"

echo "🏁 Finished at: $(date)"
