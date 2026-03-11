# 双视角 3D 姿态估计方法说明

> **项目**: Skiing Canonical DualView 3D Pose (PyTorch)  
> **核心任务**: 利用双视角 SAM-3D 单目预测结果，通过不确定性感知融合 + 时序 SSM 精炼，恢复高质量的 3D 人体姿态

---

## 目录

1. [问题定义](#1-问题定义)
2. [数据流水线](#2-数据流水线)
3. [模型架构：FusionSSM](#3-模型架构fusionssm)
   - 3.1 ViewGating（视角门控）
   - 3.2 SSMRefiner（时序精炼）
4. [损失函数](#4-损失函数)
5. [训练策略](#5-训练策略)
6. [评价指标](#6-评价指标)
7. [超参数配置](#7-超参数配置)
8. [Skeleton 骨骼定义](#8-skeleton-骨骼定义)

---

## 1. 问题定义

### 输入

| 符号 | 含义 | 形状 |
|------|------|------|
| $\mathbf{p}_L$ | 左视角 SAM-3D 单目预测 3D 关节点 | $(B, T, J, 3)$ |
| $\mathbf{p}_R$ | 右视角 SAM-3D 单目预测 3D 关节点 | $(B, T, J, 3)$ |

- $B$：批大小（合并时间维度后）
- $T$：时间帧数（每个样本可变，通过 B×T 展开为伪批次）
- $J = 15$：目标关节点数（Unity MHR70 骨骼子集）
- $3$：XYZ 坐标

### 输出

| 符号 | 含义 |
|------|------|
| $\hat{\mathbf{p}}$ | 融合精炼后的 3D 姿态（最终输出） |
| $\mathbf{p}_0$ | 视角融合初始估计（精炼前） |
| $\boldsymbol{\alpha}$ | per-joint 视角置信权重（可解释性） |

### 目标

$$\hat{\mathbf{p}} = f(\mathbf{p}_L, \mathbf{p}_R) \approx \mathbf{p}_{GT}$$

最小化预测姿态与 GT 之间的误差，同时保持骨长一致性与时序平滑性。

---

## 2. 数据流水线

### 2.1 多模态数据构成

每个训练样本包含以下模态（以一个 `(人物, 动作, 相机对)` 为单位）：

```
sample/
  frames/cam1         (1, C, T, H, W)   # 左视角视频帧（RGB, 归一化后 224×224）
  frames/cam2         (1, C, T, H, W)   # 右视角视频帧
  kpt2d_gt/cam1       (T, J, 2)         # 左视角 2D GT 关键点
  kpt2d_gt/cam2       (T, J, 2)         # 右视角 2D GT 关键点
  kpt2d_sam/cam1      (T, J, 2)         # 左视角 SAM-2D 预测
  kpt2d_sam/cam2      (T, J, 2)         # 右视角 SAM-2D 预测
  kpt3d_gt            (T, J, 3)         # GT 3D 关节点（监督信号）
  kpt3d_sam/cam1      (T, J, 3)         # 左视角 SAM-3D 预测（模型输入）
  kpt3d_sam/cam2      (T, J, 3)         # 右视角 SAM-3D 预测（模型输入）
  frame_indices       (T,)              # 原始帧索引
  meta                dict              # 人物/动作/相机元信息
```

### 2.2 图像预处理

```python
transform = Compose([
    Div255(),              # [0, 255] -> [0.0, 1.0]
    Resize((224, 224)),    # 等比缩放到 224×224
])
```

### 2.3 变长时间序列 (Variable-T) 的处理

不同相机对采样的视频帧数 $T_i$ 不同（例如 20 帧 vs 25 帧），标准 `collate_fn` 无法直接 stack。

**解决方案：B×T 维度合并**

将 $(1, C, T_i, H, W)$ 展开为 $(T_i, C, H, W)$，再 concat 成伪批次：

```
样本 0: T=20  →  20个独立帧
样本 1: T=25  →  25个独立帧
        ↓ cat
伪批次:  B' = 45 帧，每帧 T_eff = 1
```

对应 pose 张量的处理：

```
(1, T_i, J, 3) → (T_i, 1, J, C) → cat → (B', 1, J, 3)
```

这样模型在时序维度 T=1 上运行，但有效处理了所有帧，无需统一时序采样。

### 2.4 交叉验证划分

使用 5-fold 交叉验证，支持两种划分策略：

| 策略 | 含义 |
|------|------|
| `by_person` | 按人物 ID 划分（评估跨人泛化能力） |
| `by_action` | 按动作类别划分（评估跨动作泛化能力） |

索引由 `cross_validation/generate_cv_index.py` 预先生成，存储于 `data/index_mapping/`。

---

## 3. 模型架构：FusionSSM

```
p_left  →─────────────────────────────────────────────────┐
               ↓ build_velocity_confidence_proxy           │
          c_left (速度置信度)                              │
                                                           ↓
                        ViewGating  ───→  p0 (融合初值)  ──→  SSMRefiner  ──→  p_hat（最终输出）
                                    ───→  α (视角权重)
p_right →─────────────────────────────────────────────────┘
          c_right
```

### 3.1 ViewGating（不确定性感知视角门控）

**输入特征构建：**

$$\mathbf{x} = [\mathbf{p}_L \;|\; \mathbf{p}_R \;|\; (\mathbf{p}_L - \mathbf{p}_R) \;|\; \mathbf{c}_L \;|\; \mathbf{c}_R]$$

- 维度：$9 + 2 = 11$（使用置信度时）或 $9$（不使用时）
- 每帧每关节独立计算（per-joint, per-frame）

**速度置信度代理 (Velocity Confidence Proxy)：**

基于时域速度估计各视角预测的稳定性：

$$\mathbf{v}_t = \mathbf{p}_t - \mathbf{p}_{t-1}, \quad \|\mathbf{v}\| \in \mathbb{R}^{B,T,J,1}$$

$$c = \exp\!\left(-\frac{\|\mathbf{v}\|}{\text{median}(\|\mathbf{v}\|) + \epsilon}\right) \in [0, 1]$$

运动越大 → 置信度越低；静止越稳定 → 置信度越高。

**MLP 门控网络：**

```python
MLP: Linear(in_dim, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 1/2)
```

输出 $\alpha \in [0, 1]^{B,T,J,1}$ 为左视角权重（sigmoid 激活）：

$$\mathbf{p}_0 = \alpha \cdot \mathbf{p}_L + (1 - \alpha) \cdot \mathbf{p}_R$$

**可选输出 logvar（异方差不确定性）：**

$$\text{out} \in \mathbb{R}^2 \Rightarrow \alpha = \sigma(\text{out}_{:1}), \;\; \log\sigma^2 = \text{out}_{1:2}$$

### 3.2 SSMRefiner（时序 SSM 精炼器）

以融合初值 $\mathbf{p}_0$ 为输入，学习残差修正：

$$\hat{\mathbf{p}} = \mathbf{p}_0 + \Delta\mathbf{p}$$

**结构：**

```
p0 (B,T,J,3)
    → reshape: (B, T, J×3)
    → Linear(J×3, d_model)                    # 特征投影
    → [TemporalSSMBlock × n_layers]            # 时序建模
    → Linear(d_model, J×3)                     # 残差预测
    → reshape: (B, T, J, 3)
    → p0 + Δp = p_hat                          # 残差连接
```

**TemporalSSMBlock（Mamba 风格时序混合器）：**

```python
# 输入 x: (B, T, D)
h = LayerNorm(x)
h = Linear(D → D×expansion)                  # 扩展维度
h = DepthwiseConv1d(kernel=5, groups=D×exp)  # 因果感受野捕获
h = GELU(h)
h = Dropout(0.1)
h = Linear(D×expansion → D)
output = x + h                                # 残差连接
```

**设计意义：**

- `LayerNorm` → 防止激活爆炸，加速收敛
- `DepthwiseConv1d(kernel=5)` → 每个通道独立建模时序依赖，近似 Mamba/S4 的状态空间模型的局部感受野
- `残差连接` → 贯穿所有 SSMBlock，防止深层网络退化

**超参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 256 | 隐藏层维度 |
| `n_layers` | 4 | SSM Block 数量 |
| `expansion` | 2 | 内层维度扩展倍数 |
| `kernel_size` | 5 | DepthwiseConv 卷积核大小 |

---

## 4. 损失函数

损失函数根据是否有 GT 标签自动切换模式：

```python
if p_gt is not None:
    loss = supervised(p_hat, p_gt)
else:
    loss = self_supervised(p_hat, p_left, p_right, alpha)
```

### 4.1 监督损失（有 GT 标签）

$$\mathcal{L} = \lambda_{\text{mpjpe}} \cdot \mathcal{L}_{\text{mpjpe}} + \lambda_{\text{bone}} \cdot \mathcal{L}_{\text{bone}} + \lambda_{\text{vel}} \cdot \mathcal{L}_{\text{vel}} + \lambda_{\text{acc}} \cdot \mathcal{L}_{\text{acc}}$$

| 项 | 公式 | 含义 |
|----|------|------|
| $\mathcal{L}_{\text{mpjpe}}$ | $\frac{1}{BTJ}\sum \|\hat{\mathbf{p}} - \mathbf{p}_{GT}\|_1$ | 位置误差 |
| $\mathcal{L}_{\text{bone}}$ | $\frac{1}{E}\sum_e \|\hat{b}_e - b_e^{GT}\|_1$ | 骨长一致性 |
| $\mathcal{L}_{\text{vel}}$ | $\frac{1}{BJ}\sum\|\hat{\mathbf{p}}_t - \hat{\mathbf{p}}_{t-1}\|_1$ | 时序速度平滑 |
| $\mathcal{L}_{\text{acc}}$ | $\frac{1}{BJ}\sum\|\hat{\mathbf{p}}_{t+1} - 2\hat{\mathbf{p}}_t + \hat{\mathbf{p}}_{t-1}\|_1$ | 时序加速度平滑 |

**异方差不确定性回归**（`predict_logvar=True` 时）：

$$\mathcal{L}_{\text{mpjpe}} = \mathbb{E}\!\left[e^{-\log\sigma^2} \cdot \|\hat{\mathbf{p}} - \mathbf{p}_{GT}\|^2 + \log\sigma^2\right]$$

### 4.2 自监督损失（无 GT 标签）

$$\mathcal{L} = \lambda_{\text{agree}} \cdot \mathcal{L}_{\text{agree}} + \lambda_{\text{vel}} \cdot \mathcal{L}_{\text{vel}} + \lambda_{\text{acc}} \cdot \mathcal{L}_{\text{acc}} + \lambda_{\text{bone\_stab}} \cdot \mathcal{L}_{\text{bone\_stab}}$$

| 项 | 公式 | 含义 |
|----|------|------|
| $\mathcal{L}_{\text{agree}}$ | $\mathbb{E}[\alpha\|\hat{\mathbf{p}} - \mathbf{p}_L\| + (1-\alpha)\|\hat{\mathbf{p}} - \mathbf{p}_R\|]$ | 一致性约束（权重贡献越大的视角，融合结果应更接近该视角） |
| $\mathcal{L}_{\text{bone\_stab}}$ | $\frac{1}{T-1}\sum_t\|b_{t+1} - b_t\|_1$ | 骨长时序稳定性 |

### 4.3 默认权重

| 权重 | 默认值 |
|------|--------|
| $\lambda_{\text{mpjpe}}$ | 1.0 |
| $\lambda_{\text{bone}}$ | 0.2 |
| $\lambda_{\text{vel}}$ | 0.05 |
| $\lambda_{\text{acc}}$ | 0.02 |
| $\lambda_{\text{agree}}$ | 0.1 |
| $\lambda_{\text{bone\_stab}}$ | 0.05 |

---

## 5. 训练策略

### 5.1 优化器

```python
AdamW(lr=1e-4, weight_decay=1e-4)
```

### 5.2 学习率调度

```python
CosineAnnealingLR(T_max=estimated_stepping_batches)
```

从初始 lr 余弦衰减到 0，一般 50 epochs 完成。

### 5.3 监控指标

训练过程中记录以下指标：

| 指标 | 说明 |
|------|------|
| `train/loss` | 总训练损失 |
| `train/mpjpe` | 训练 MPJPE（毫米，与 GT 比较） |
| `val/loss` | 验证总损失 |
| `val/mpjpe` | 验证 MPJPE |
| `train/alpha_mean` | 平均视角权重（监控视角偏好） |
| `train/alpha_std` | 视角权重方差（反映不确定性分布） |
| `train/loss/bone` | 骨长损失分量 |
| `train/loss/vel` | 速度平滑损失分量 |
| `train/loss/acc` | 加速度平滑损失分量 |

### 5.4 交叉验证流程

```
for fold in range(n_splits):
    构建 train_idx / val_idx
    初始化 FusionSSMTrainer
    trainer.fit(...)
    trainer.test(...)  → 保存 fold_{k}_pose_outputs.pt
```

---

## 6. 评价指标

使用 `analysis/evaluate_pose_metrics.py` 离线计算。

### 6.1 MPJPE（Mean Per-Joint Position Error）

$$\text{MPJPE} = \frac{1}{BTJ}\sum_{b,t,j}\|\hat{\mathbf{p}}_{b,t,j} - \mathbf{p}^{GT}_{b,t,j}\|_2$$

基础位置误差，以毫米（mm）为单位（取决于坐标系scale）。

### 6.2 N-MPJPE（Scale-Normalized MPJPE）

对每帧独立进行全局尺度对齐：

$$s^* = \frac{\langle \hat{\mathbf{p}}, \mathbf{p}^{GT} \rangle}{\|\hat{\mathbf{p}}\|^2}, \quad \text{N-MPJPE} = \text{MPJPE}(s^* \cdot \hat{\mathbf{p}},\; \mathbf{p}^{GT})$$

消除视角不同导致的尺度差异，更好反映姿态结构准确性。

### 6.3 P-MPJPE（Procrustes-MPJPE）

对每帧进行刚体 + 尺度对齐（奇异值分解）：

$$R^*, t^*, s^* = \arg\min_{R,t,s}\|s\cdot R\hat{\mathbf{p}} + t - \mathbf{p}^{GT}\|_F$$

$$\text{P-MPJPE} = \text{MPJPE}(s^* R^* \hat{\mathbf{p}} + t^*,\; \mathbf{p}^{GT})$$

消除全局旋转、平移、缩放误差，评估纯姿态结构质量（上限指标）。

### 6.4 分层分析

按以下维度聚合结果：

| 分组键 | 说明 |
|--------|------|
| `fold` | 每折 CV 结果 |
| `person_id` | 每个人物的泛化性 |
| `action_id` | 每种动作的预测难度 |
| `cam_pair` | 每对相机视角的效果 |

### 6.5 运行评估脚本

```bash
# 计算单个训练结果的 metrics
python analysis/evaluate_pose_metrics.py \
    --pose-dir logs/train/3dcnn_fuse_method_mamba_ssm_fold_1/2026-03-11/pose_analysis

# 横向对比多次实验
python analysis/compare_pose_metric_runs.py \
    logs/train/exp_A/pose_analysis \
    logs/train/exp_B/pose_analysis \
    --sort-by p_mpjpe
```

---

## 7. 超参数配置

主配置文件：`configs/train.yaml`

```yaml
loss:
  lr: 0.0001
  lambda_mpjpe: 1.0
  lambda_bone: 0.2
  lambda_vel: 0.05
  lambda_acc: 0.02
  lambda_agree: 0.1
  lambda_bone_stab: 0.05

model:
  d_model: 256          # SSMRefiner 隐藏维度
  n_layers: 4           # SSM Block 数量
  use_conf: true        # 是否使用速度置信度
  predict_logvar: false # 是否预测异方差不确定性

data:
  batch_size: 16
  num_workers: 16
  img_size: 224

train:
  max_epochs: 50
  fold: 0               # 当前使用第几折
```

---

## 8. Skeleton 骨骼定义

使用 Unity MHR70 骨骼的 15 个关节子集：

| 关节 ID | 名称 | 目标索引 |
|---------|------|---------|
| 1 | Bone_Eye_L（左眼） | 0 |
| 2 | Bone_Eye_R（右眼） | 1 |
| 5 | Upperarm_L（左上臂） | 2 |
| 6 | Upperarm_R（右上臂） | 3 |
| 7 | lowerarm_l（左前臂） | 4 |
| 8 | lowerarm_r（右前臂） | 5 |
| 9 | Thigh_L（左大腿） | 6 |
| 10 | Thigh_R（右大腿） | 7 |
| 11 | calf_l（左小腿） | 8 |
| 12 | calf_r（右小腿） | 9 |
| 13 | Foot_L（左脚） | 10 |
| 14 | Foot_R（右脚） | 11 |
| 41 | Hand_R（右手） | 12 |
| 62 | Hand_L（左手） | 13 |
| 69 | neck_01（颈部） | 14 |

骨骼连接（12 条骨段）：

```
neck → shoulder_L/R → elbow_L/R → hand_L/R   (双臂)
neck → hip_L/R → knee_L/R → foot_L/R         (双腿)
```

关节角度定义（用于下游动作分析）：

| 角度 | 关节三元组 |
|------|----------|
| knee_l | (Thigh_L, calf_l, Foot_L) |
| knee_r | (Thigh_R, calf_r, Foot_R) |
| elbow_l | (Upperarm_L, lowerarm_l, Hand_L) |
| elbow_r | (Upperarm_R, lowerarm_r, Hand_R) |
| shoulder_l | (neck, Upperarm_L, lowerarm_l) |
| hip_l | (neck, Thigh_L, calf_l) |

---

## 附录：模块依赖

```
project/
├── models/
│   └── fusion_ssm_pose_refiner.py   # FusionSSM, ViewGating, SSMRefiner, PoseRefineLoss
├── trainer/
│   └── train_fusion_SSM.py          # FusionSSMTrainer (LightningModule)
├── dataloader/
│   ├── whole_video_dataset.py       # LabeledUnityDataset
│   ├── data_loader.py               # UnityDataModule (含 _collate_fn)
│   └── utils.py                     # Div255 transform
├── map_config.py                    # ID_TO_INDEX, SKELETON_CONNECTIONS, ANGLE_DEFS
└── main.py                          # Hydra 入口，K-fold 训练调度

analysis/
├── evaluate_pose_metrics.py         # 离线 MPJPE/N-MPJPE/P-MPJPE 计算
└── compare_pose_metric_runs.py      # 多实验横向对比
```
