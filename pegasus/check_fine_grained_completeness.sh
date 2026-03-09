#!/bin/bash
# 驗證細粒度推導的完整性腳本

PROJECT_ROOT="/work/SSR/share/code/Skiing_Canonical_DualView_3D_Pose_PyTorch"
cd "${PROJECT_ROOT}"

RESULT_ROOT="/work/SSR/share/data/skiing/skiing_unity_dataset/sam3d_body_results"

echo "========================================"
echo "SAM-3D-Body 細粒度推導完整性檢查"
echo "========================================"
echo ""

# 定義人物和動作
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

TOTAL_EXPECTED=0
TOTAL_FOUND=0
MISSING_JOBS=()

echo "檢查每個任務單元的推導結果..."
echo ""

for PERSON_IDX in 0 1; do
    PERSON=${PERSONS[$PERSON_IDX]}
    
    for ACTION_IDX in {0..11}; do
        ACTION=${ACTIONS[$ACTION_IDX]}
        
        # 轉換動作名稱（female）
        if [ "$PERSON" == "female" ]; then
            ACTION=${ACTION/Male/Female}
        fi
        
        for LAYER_IDX in {0..4}; do
            NODE_INDEX=$((PERSON_IDX * 60 + ACTION_IDX * 5 + LAYER_IDX))
            
            # 檢查這個任務單元的結果數量（應該有36個captures）
            RESULT_PATH="${RESULT_ROOT}/inference/${PERSON}/${ACTION}/frames"
            
            if [ -d "$RESULT_PATH" ]; then
                CAPTURE_COUNT=$(find "$RESULT_PATH" -type d -name "capture_L${LAYER_IDX}_*" 2>/dev/null | wc -l)
                TOTAL_FOUND=$((TOTAL_FOUND + CAPTURE_COUNT))
                
                if [ $CAPTURE_COUNT -ne 36 ]; then
                    echo "[WARN] Node ${NODE_INDEX}: ${PERSON}/${ACTION}/L${LAYER_IDX} - 找到 ${CAPTURE_COUNT}/36 captures"
                    MISSING_JOBS+=("Node ${NODE_INDEX}: ${PERSON}/${ACTION}/L${LAYER_IDX}")
                fi
            else
                echo "[MISS] Node ${NODE_INDEX}: ${PERSON}/${ACTION}/L${LAYER_IDX} - 結果目錄不存在"
                MISSING_JOBS+=("Node ${NODE_INDEX}: ${PERSON}/${ACTION}/L${LAYER_IDX}")
            fi
            
            TOTAL_EXPECTED=$((TOTAL_EXPECTED + 36))
        done
    done
done

echo ""
echo "========================================"
echo "統計摘要"
echo "========================================"
echo "預期總 captures: ${TOTAL_EXPECTED} (2人 × 12動作 × 5層 × 36角度)"
echo "實際找到 captures: ${TOTAL_FOUND}"
echo "完成率: $(awk "BEGIN {printf \"%.2f\", ${TOTAL_FOUND}*100/${TOTAL_EXPECTED}}") %"
echo ""

if [ ${#MISSING_JOBS[@]} -gt 0 ]; then
    echo "需要重新執行的任務 (${#MISSING_JOBS[@]} 個):"
    for job in "${MISSING_JOBS[@]}"; do
        echo "  - $job"
    done
else
    echo "✅ 所有任務已完整推導！"
fi

echo ""
echo "詳細結果目錄: ${RESULT_ROOT}"
