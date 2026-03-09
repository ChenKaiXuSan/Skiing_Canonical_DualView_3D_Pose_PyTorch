# SAM-3D-Body 細粒度推導使用指南

## 📋 概述

本指南說明如何使用 **120 個 PBS nodes** 並行推導 SAM-3D-Body，做到最細粒度的任務分配。

### 數據結構

你的數據集包含：
- **2 個人物**: male, female
- **12 個動作**: Braking, Braking_Loop, Fall, From_Idle_To_Skiing, From_Losing_To_Idle, From_Skiing_To_Straight, From_Winning_To_Idle, Idle, Losing, Skiing, Skiing_Straight, Winning
- **5 層相機**: L0, L1, L2, L3, L4
- **每層 36 角度**: A000-A350（每10°一個）

**總計**: 2 × 12 × 5 × 36 = **4,320 個 captures**

目錄結構：
```
data/
├── male/
│   ├── Anim_Male_Skier_Braking/
│   │   └── frames/
│   │       ├── capture_L0_A000/
│   │       ├── capture_L0_A010/
│   │       ├── ...
│   │       ├── capture_L4_A350/
│   └── ...
└── female/
    └── ...
```

## 🚀 推薦使用方式：120 Nodes 細粒度推導

### 腳本：`run_sam3d_body_fine_grained.sh`

**特點**：
- ✅ **最大化並行度**：每個 node 只處理 1人 × 1動作 × 1層（36個captures）
- ✅ **故障隔離**：某個 node 失敗只影響小範圍
- ✅ **易於追蹤**：清晰知道每個 node 的任務
- ✅ **快速完成**：充分利用集群資源

**提交命令**：
```bash
qsub pegasus/run_sam3d_body_fine_grained.sh
```

**任務分配邏輯**：
```
Node 索引計算公式：
NODE_INDEX = PERSON_INDEX × 60 + ACTION_INDEX × 5 + LAYER_INDEX

其中：
- PERSON_INDEX: 0 (male), 1 (female)
- ACTION_INDEX: 0-11 (12個動作)
- LAYER_INDEX: 0-4 (5層相機)
```

**示例**：
- Node 0: male, Braking, L0
- Node 1: male, Braking, L1
- Node 4: male, Braking, L4
- Node 5: male, Braking_Loop, L0
- Node 60: female, Braking, L0
- Node 119: female, Winning, L4

## 📊 驗證完整性

### 自動檢查（推薦）

```bash
bash pegasus/check_fine_grained_completeness.sh
```

輸出示例：
```
預期總 captures: 4320 (2人 × 12動作 × 5層 × 36角度)
實際找到 captures: 4320
完成率: 100.00 %
✅ 所有任務已完整推導！
```

如果有缺失，會顯示：
```
需要重新執行的任務 (3 個):
  - Node 15: male/Anim_Male_Skier_From_Skiing_To_Straight/L0
  - Node 72: female/Anim_Female_Skier_Idle/L2
  - Node 103: female/Anim_Female_Skier_Skiing_Straight/L3
```

### 手動檢查

```bash
# 檢查特定層的結果數量
find /work/SSR/share/data/skiing/skiing_unity_dataset/sam3d_body_results/inference \
    -type d -name "capture_L0_*" | wc -l
# 應該輸出: 864 (2人 × 12動作 × 36角度)

# 檢查所有層
for layer in 0 1 2 3 4; do
    count=$(find /work/SSR/share/data/skiing/skiing_unity_dataset/sam3d_body_results/inference \
        -type d -name "capture_L${layer}_*" | wc -l)
    echo "L${layer}: ${count}/864 captures"
done

# 檢查特定人物和動作的完整性
find /work/SSR/share/data/skiing/skiing_unity_dataset/sam3d_body_results/inference/male/Anim_Male_Skier_Idle/frames \
    -type d -name "capture_L*" | wc -l
# 應該輸出: 180 (5層 × 36角度)
```

## 🔧 其他使用方式

### 方式 2: 單層推導（5 nodes）

使用 `run_sam3d_body_by_layer.sh`：
```bash
qsub pegasus/run_sam3d_body_by_layer.sh
```
- 每個 node 處理一個完整層級（所有人物和動作）
- 適合快速測試

### 方式 3: 層級 + 動作分片（10 nodes）

使用 `run_sam3d_body_multi_layer.sh`：
```bash
qsub pegasus/run_sam3d_body_multi_layer.sh
```
- 每層分成 2 個 nodes
- 適合中等規模並行

### 方式 4: 手動測試單個任務

```bash
# 只處理 male, Idle 動作, L0 層
python -m SAM3Dbody.main \
    infer.camera_layers="[0]" \
    infer.person_filter="male" \
    infer.action_filter="Anim_Male_Skier_Idle"

# 只處理 female, Skiing 動作, L2 層
python -m SAM3Dbody.main \
    infer.camera_layers="[2]" \
    infer.person_filter="female" \
    infer.action_filter="Anim_Female_Skier_Skiing"
```

## ⚙️ 配置參數說明

在 `configs/sam3d_body.yaml` 中新增的參數：

```yaml
infer:
  # 相機層級過濾 (L0-L4)
  camera_layers: null  # null=所有層，或 [0], [0,1], [0,1,2,3,4]
  
  # 人物過濾
  person_filter: null  # null=所有人，或 "male", "female"
  
  # 動作過濾
  action_filter: null  # null=所有動作，或具體動作名如 "Anim_Male_Skier_Idle"
```

## 📝 日誌和追蹤

### PBS 日誌位置
```
logs/pegasus/sam3d_fine_{0..119}.log       # 標準輸出
logs/pegasus/sam3d_fine_{0..119}_err.log   # 錯誤輸出
```

### 分片任務記錄
```
logs/unity_sam3d-body/*/shard_actions/shard_000_of_001.txt
```
包含：
- camera_layers
- person_filter
- action_filter
- 被分配的動作列表

### 動作推導日誌
```
logs/unity_sam3d-body/*/action_logs/male__Anim_Male_Skier_Idle.log
```

## 🔍 故障排除

### 1. 某個 node 失敗了怎麼辦？

找到對應的 node 索引，手動重跑：
```bash
# 假設 Node 25 失敗
# 計算：25 = 0×60 + 5×5 + 0 → male, 動作5, L0
# 動作5 = Anim_Male_Skier_From_Skiing_To_Straight

python -m SAM3Dbody.main \
    infer.camera_layers="[0]" \
    infer.person_filter="male" \
    infer.action_filter="Anim_Male_Skier_From_Skiing_To_Straight"
```

### 2. 如何快速計算 node 對應的任務？

使用公式：
```python
node_index = 25
person_index = node_index // 60       # 0 = male, 1 = female
action_index = (node_index % 60) // 5 # 0-11
layer_index = node_index % 5          # 0-4
```

### 3. 部分 captures 缺失

檢查對應的動作日誌：
```bash
grep -i "skip\|error\|warning" logs/unity_sam3d-body/*/action_logs/male__Anim_Male_Skier_Idle.log
```

## 📈 性能建議

### 調整並發線程數

在 PBS 腳本中修改：
```bash
THREADS_PER_NODE=6  # 根據 GPU 記憶體調整（建議 4-8）
```

### GPU 資源配置

每個 node 使用：
- 1 個 GPU
- 6 個並發 workers（可調整）
- 每個 worker 處理 ~6 個 captures

總推導時間估算：
- 假設每個 capture 需要 10 秒
- 每個 node: 36 captures ÷ 6 workers ÷ 10s ≈ **1 分鐘**
- 120 個 nodes 並行完成全部推導

## 📚 相關文件

- [run_sam3d_body_fine_grained.sh](run_sam3d_body_fine_grained.sh) - 120 nodes 細粒度腳本
- [check_fine_grained_completeness.sh](check_fine_grained_completeness.sh) - 完整性檢查腳本
- [run_sam3d_body_by_layer.sh](run_sam3d_body_by_layer.sh) - 5 nodes 單層腳本
- [run_sam3d_body_multi_layer.sh](run_sam3d_body_multi_layer.sh) - 10 nodes 分層腳本
- [configs/sam3d_body.yaml](../configs/sam3d_body.yaml) - 配置文件

## ✅ 快速開始

```bash
# 1. 提交 120 個並行任務
cd /work/SSR/share/code/Skiing_Canonical_DualView_3D_Pose_PyTorch
qsub pegasus/run_sam3d_body_fine_grained.sh

# 2. 監控任務狀態
qstat -u $USER

# 3. 檢查完成情況
bash pegasus/check_fine_grained_completeness.sh

# 4. 如果有缺失，根據提示重跑特定任務
```
