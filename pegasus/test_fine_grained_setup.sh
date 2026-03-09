#!/bin/bash
# 測試細粒度推導配置是否正確

PROJECT_ROOT="/work/SSR/share/code/Skiing_Canonical_DualView_3D_Pose_PyTorch"
cd "${PROJECT_ROOT}"

echo "========================================"
echo "SAM-3D-Body 細粒度推導配置測試"
echo "========================================"
echo ""

# 定義測試用的數據
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

DATA_ROOT="/work/SSR/share/data/skiing/skiing_unity_dataset/data"

echo "1. 檢查數據目錄結構..."
echo ""

# 檢查人物目錄
for person in "${PERSONS[@]}"; do
    if [ -d "${DATA_ROOT}/${person}" ]; then
        echo "  ✅ ${person}/ 存在"
        
        # 計算動作數量
        action_count=$(ls -1d "${DATA_ROOT}/${person}"/Anim_* 2>/dev/null | wc -l)
        echo "     → 找到 ${action_count} 個動作目錄"
        
        # 檢查第一個動作的 capture 數量
        first_action=$(ls -1d "${DATA_ROOT}/${person}"/Anim_* 2>/dev/null | head -1)
        if [ -n "$first_action" ]; then
            capture_count=$(find "${first_action}/frames" -maxdepth 1 -type d -name "capture_L*" 2>/dev/null | wc -l)
            layer_counts=""
            for layer in 0 1 2 3 4; do
                layer_count=$(find "${first_action}/frames" -maxdepth 1 -type d -name "capture_L${layer}_*" 2>/dev/null | wc -l)
                layer_counts="${layer_counts} L${layer}:${layer_count}"
            done
            echo "     → 第一個動作 ($(basename $first_action)) 的 captures: ${capture_count} 個"
            echo "       詳細層級分布:${layer_counts}"
        fi
    else
        echo "  ❌ ${person}/ 不存在"
    fi
done

echo ""
echo "2. 驗證動作名稱對應..."
echo ""

# 檢查前3個動作的對應關係
for action_idx in 0 1 2; do
    male_action=${ACTIONS[$action_idx]}
    female_action=${male_action/Male/Female}
    
    male_exists="❌"
    female_exists="❌"
    
    [ -d "${DATA_ROOT}/male/${male_action}" ] && male_exists="✅"
    [ -d "${DATA_ROOT}/female/${female_action}" ] && female_exists="✅"
    
    echo "  動作 ${action_idx}: ${male_exists} male/${male_action}"
    echo "            ${female_exists} female/${female_action}"
done

echo ""
echo "3. 測試 Node 索引計算..."
echo ""

# 測試幾個關鍵的 node 索引
test_cases=(
    "0:male:0:0"      # 第一個 node
    "4:male:0:4"      # male 第一個動作最後一層
    "5:male:1:0"      # male 第二個動作第一層
    "59:male:11:4"    # male 最後一個 node
    "60:female:0:0"   # female 第一個 node
    "119:female:11:4" # 最後一個 node
)

echo "  Node格式: node_index → person, action_index, layer"
echo ""

for test_case in "${test_cases[@]}"; do
    IFS=':' read -r node expected_person expected_action expected_layer <<< "$test_case"
    
    person_idx=$((node / 60))
    action_idx=$(((node % 60) / 5))
    layer_idx=$((node % 5))
    
    person=${PERSONS[$person_idx]}
    
    result="✅"
    if [ "$person" != "$expected_person" ] || [ "$action_idx" != "$expected_action" ] || [ "$layer_idx" != "$expected_layer" ]; then
        result="❌"
    fi
    
    echo "  ${result} Node ${node} → ${person}, action=${action_idx}, L${layer_idx}"
done

echo ""
echo "4. 檢查環境和依賴..."
echo ""

# 檢查 conda 環境
if command -v conda &> /dev/null; then
    echo "  ✅ conda 可用"
else
    echo "  ❌ conda 不可用"
fi

# 檢查腳本文件
scripts=(
    "pegasus/run_sam3d_body_fine_grained.sh"
    "pegasus/check_fine_grained_completeness.sh"
    "SAM3Dbody/main.py"
    "configs/sam3d_body.yaml"
)

for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "  ✅ ${script}"
    else
        echo "  ❌ ${script} 不存在"
    fi
done

echo ""
echo "5. 預計算任務量..."
echo ""

total_persons=2
total_actions=12
total_layers=5
total_angles=36

total_nodes=$((total_persons * total_actions * total_layers))
total_captures=$((total_nodes * total_angles))

echo "  總人物數: ${total_persons}"
echo "  總動作數: ${total_actions}"
echo "  總層級數: ${total_layers}"
echo "  每層角度: ${total_angles}"
echo ""
echo "  → 總 nodes: ${total_nodes}"
echo "  → 總 captures: ${total_captures}"
echo "  → 每個 node 處理: ${total_angles} captures"

echo ""
echo "========================================"
echo "測試完成"
echo "========================================"
echo ""
echo "下一步："
echo "  1. 確認以上檢查結果都是 ✅"
echo "  2. 提交任務: qsub pegasus/run_sam3d_body_fine_grained.sh"
echo "  3. 監控狀態: qstat -u \$USER"
echo "  4. 檢查完成: bash pegasus/check_fine_grained_completeness.sh"
