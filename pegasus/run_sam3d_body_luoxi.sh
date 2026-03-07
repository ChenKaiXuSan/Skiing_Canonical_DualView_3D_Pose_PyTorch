#!/bin/bash
#PBS -A SSR
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N sam3d_8nodes_run
#PBS -t 0-7                           # 8 nodes (modify with TOTAL_NODES below)
#PBS -o logs/pegasus/sam3d_group_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/sam3d_group_${PBS_SUBREQNO}_err.log

# === 1. 環境準備 ===
PROJECT_ROOT="/work/SSR/share/code/Skiing_Canonical_DualView_3D_Pose_PyTorch"
cd "${PROJECT_ROOT}"

mkdir -p logs/pegasus/

module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
conda activate /home/SSR/luoxi/miniconda3/envs/sam_3d_body

conda env list

# === 2. 多node任务划分 ===
# Keep this value consistent with #PBS -t range: 0-(TOTAL_NODES-1)
TOTAL_NODES=8
THREADS_PER_NODE=6

NODE_INDEX=${PBS_SUBREQNO:-0}

if (( NODE_INDEX < 0 || NODE_INDEX >= TOTAL_NODES )); then
    echo "[ERROR] NODE_INDEX=${NODE_INDEX} out of range for TOTAL_NODES=${TOTAL_NODES}"
    exit 1
fi

echo "Node Index: $PBS_SUBREQNO"
echo "Total Nodes: $TOTAL_NODES"
echo "Threads Per Node: $THREADS_PER_NODE"
echo "Action Shard: ${NODE_INDEX}/${TOTAL_NODES}"

# === 3. パス設定と実行 ===
UNITY_DATASET_PATH="/work/SSR/share/data/skiing_unity_dataset/data"
RESULT_ROOT_PATH="/work/SSR/share/data/skiing_unity_dataset/sam3d_body_results"
CKPT_ROOT="${PROJECT_ROOT}/ckpt/sam-3d-body-dinov3"

echo "🏁 Node ${PBS_SUBREQNO} started at: $(date)"
echo "Unity Dataset Path: $UNITY_DATASET_PATH"
echo "Result Root Path: $RESULT_ROOT_PATH"
echo "Checkpoint Root: $CKPT_ROOT"

python -m SAM3Dbody.main \
    paths.unity.unity_dataset_data_path=${UNITY_DATASET_PATH} \
    paths.unity.unity_sam3d_result_root=${RESULT_ROOT_PATH} \
    model.root_path=${CKPT_ROOT} \
    infer.gpu="[0]" \
    infer.workers_per_gpu=${THREADS_PER_NODE} \
    infer.shard_count=${TOTAL_NODES} \
    infer.shard_index=${NODE_INDEX}

echo "🏁 Node ${PBS_SUBREQNO} finished at: $(date)"
# 一个 node 内部并发 worker 数由 THREADS_PER_NODE 控制（当前为 6）