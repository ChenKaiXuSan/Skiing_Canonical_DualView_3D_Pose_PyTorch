#!/bin/bash

# === 1. 参数设定 ===
PROJECT_NAME="SKIING"                 # 你的项目名（必须改）
TIME="01:00:00"                        # 申请时长（最大 01:00:00）

# === 1. 環境準備 ===
cd /work/SKIING/chenkaixu/code/Drive_Face_Mesh_PyTorch
module load intelpython/2022.3.1
conda deactivate
conda activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body

# === 2. 启动 debug 节点会话 ===
echo "🟡 请求 debug 节点会话: ${TIME}"
qlogin -A "$PROJECT_NAME" -q debug -l elapstim_req=$TIME

# qlogin -A $PROJECT_NAME -q debug -b 2 -l elapstim_req=00:10:00 -T openmpi -v NQSV_MPI_VER=4.1.6/gcc11.4.0-cuda11.8.0 -V

