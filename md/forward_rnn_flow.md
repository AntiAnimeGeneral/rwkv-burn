# TimeMix `forward_rnn` 前向传播流程图

本文档详细说明 RWKV v7 模型中 `TimeMix::forward_rnn` 函数的前向传播计算流程。

> **参考**: RWKV-7 "Goose" 架构，结合了线性注意力与 Delta Rule 的状态更新机制，实现了 $O(Td^2)$ 的线性时间复杂度。

---

## 符号约定

| 类型 | 约定 | 示例 |
|------|------|------|
| **标量** | 普通斜体小写 | $d$, $H$, $t$, $d_h$ |
| **向量** | 粗体小写 | $\mathbf{x}$, $\mathbf{r}$, $\mathbf{k}$, $\mathbf{v}$ |
| **矩阵** | 粗体大写 | $\mathbf{W}$, $\mathbf{S}$, $\mathbf{A}$ |
| **集合/空间** | 花体大写 | $\mathbb{R}^d$ |

---

## 符号说明

### 标量参数

| 符号 | 含义 | 说明 |
|------|------|------|
| $d$ | 模型维度 | `d_model` |
| $H$ | 注意力头数 | `n_heads` |
| $d_h$ | 每头维度 | `head_size = d/H` |
| $t$ | 当前时间步 | 时序索引 |

### 向量变量

| 符号 | 含义 | 维度 |
|------|------|------|
| $\mathbf{x}_t$ | 当前时刻输入 | $\mathbb{R}^d$ |
| $\mathbf{x}_{t-1}$ | 前一时刻输入 | $\mathbb{R}^d$ |
| $\mathbf{r}_t$ | Receptance 向量 | $\mathbb{R}^d$ |
| $\mathbf{k}_t$ | Key 向量 | $\mathbb{R}^d$ |
| $\mathbf{v}_t$ | Value 向量 | $\mathbb{R}^d$ |
| $\mathbf{w}_t$ | 衰减权重向量 | $\mathbb{R}^d$ |
| $\boldsymbol{\alpha}_t$ | 注意力门向量 | $\mathbb{R}^d$ |
| $\mathbf{g}_t$ | 输出门向量 | $\mathbb{R}^d$ |
| $\boldsymbol{\mu}$ | 混合系数向量 | $\mathbb{R}^d$ |

### 矩阵变量

| 符号 | 含义 | 维度 |
|------|------|------|
| $\mathbf{S}_{t}$ | 状态矩阵 (`vk_state`) | $\mathbb{R}^{H \times d_h \times d_h}$ |
| $\mathbf{W}_r, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_o$ | 投影矩阵 | $\mathbb{R}^{d \times d}$ |
| $\mathbf{VK}_t$ | Value-Key 外积矩阵 | $\mathbb{R}^{H \times d_h \times d_h}$ |
| $\mathbf{AB}_t$ | Delta Rule 校正矩阵 | $\mathbb{R}^{H \times d_h \times d_h}$ |

---

## 函数签名

```rust
pub fn forward_rnn(
    &self,
    x: Tensor<B, 1>,          // x_t: 当前输入 [d]
    x_prev: Tensor<B, 1>,     // x_{t-1}: 前一时刻输入 [d]
    v_first: Option<Tensor<B, 1>>,  // v_0: 可选的初始 v 值 [d]
    vk_state: Tensor<B, 3>,   // S_{t-1}: 状态张量 [H, d_h, d_h]
) -> (
    Tensor<B, 1>,             // o_t: 输出 [d]
    Tensor<B, 1>,             // x_t: 当前输入(用于下一步的 x_{t-1})
    Tensor<B, 3>,             // S_t: 更新后的状态 [H, d_h, d_h]
    Option<Tensor<B, 1>>,     // v_first: 更新后的 v_0
)
```

---

## 数学公式详解

### 1️⃣ 时间移位 (Time Shift / Token Shift)

**目的**: 计算当前时刻与前一时刻输入的差分，用于后续的时间混合。

$$
\Delta \mathbf{x}_t = \mathbf{x}_{t-1} - \mathbf{x}_t
$$

其中：
- $\mathbf{x}_t \in \mathbb{R}^d$：当前时刻的隐藏状态输入向量
- $\mathbf{x}_{t-1} \in \mathbb{R}^d$：前一时刻的隐藏状态输入向量
- $\Delta \mathbf{x}_t \in \mathbb{R}^d$：时间差分向量

---

### 2️⃣ Token Mixing (时间混合插值)

**目的**: 通过学习到的混合系数，在当前时刻和前一时刻特征之间进行加权插值，为不同的下游计算提供定制化的输入。

$$
\begin{aligned}
\tilde{\mathbf{x}}_t^{(r)} &= \mathbf{x}_t + \Delta \mathbf{x}_t \odot \boldsymbol{\mu}_r \\
\tilde{\mathbf{x}}_t^{(w)} &= \mathbf{x}_t + \Delta \mathbf{x}_t \odot \boldsymbol{\mu}_w \\
\tilde{\mathbf{x}}_t^{(k)} &= \mathbf{x}_t + \Delta \mathbf{x}_t \odot \boldsymbol{\mu}_k \\
\tilde{\mathbf{x}}_t^{(v)} &= \mathbf{x}_t + \Delta \mathbf{x}_t \odot \boldsymbol{\mu}_v \\
\tilde{\mathbf{x}}_t^{(a)} &= \mathbf{x}_t + \Delta \mathbf{x}_t \odot \boldsymbol{\mu}_a \\
\tilde{\mathbf{x}}_t^{(g)} &= \mathbf{x}_t + \Delta \mathbf{x}_t \odot \boldsymbol{\mu}_g
\end{aligned}
$$

其中：
- $\boldsymbol{\mu}_r, \boldsymbol{\mu}_w, \boldsymbol{\mu}_k, \boldsymbol{\mu}_v, \boldsymbol{\mu}_a, \boldsymbol{\mu}_g \in \mathbb{R}^d$ 是可学习的混合系数向量（对应代码中的 `x_r`, `x_w`, `x_k`, `x_v`, `x_a`, `x_g`）
- $\odot$ 表示逐元素乘法 (Hadamard product)
- 当 $\boldsymbol{\mu} = \mathbf{0}$ 时，$\tilde{\mathbf{x}}_t = \mathbf{x}_t$（使用当前输入）
- 当 $\boldsymbol{\mu} = \mathbf{1}$ 时，$\tilde{\mathbf{x}}_t = \mathbf{x}_{t-1}$（使用前一时刻输入）

---

### 3️⃣ 核心投影计算 (Linear Projections)

#### 3.1 Receptance（接收门）

$$
\mathbf{r}_t = \mathbf{W}_r \tilde{\mathbf{x}}_t^{(r)}
$$

其中 $\mathbf{W}_r \in \mathbb{R}^{d \times d}$ 是 receptance 投影矩阵。

#### 3.2 Key（键向量）

$$
\mathbf{k}_t = \mathbf{W}_k \tilde{\mathbf{x}}_t^{(k)}
$$

其中 $\mathbf{W}_k \in \mathbb{R}^{d \times d}$ 是 key 投影矩阵。

#### 3.3 Value（值向量）

$$
\mathbf{v}_t = \mathbf{W}_v \tilde{\mathbf{x}}_t^{(v)}
$$

其中 $\mathbf{W}_v \in \mathbb{R}^{d \times d}$ 是 value 投影矩阵。

#### 3.4 Decay Weight（衰减权重）

RWKV-7 中的衰减权重通过 LoRA 结构动态计算：

$$
\mathbf{w}_t = \exp\left( \sigma\left( \mathbf{W}_2^{(w)} \tanh\left( \mathbf{W}_1^{(w)} \tilde{\mathbf{x}}_t^{(w)} \right) + \mathbf{w}_0 \right) \cdot (-0.606531) \right)
$$

其中：
- $\mathbf{W}_1^{(w)} \in \mathbb{R}^{d \times d_{\text{lora}}}$, $\mathbf{W}_2^{(w)} \in \mathbb{R}^{d_{\text{lora}} \times d}$ 是 LoRA 分解矩阵
- $\mathbf{w}_0 \in \mathbb{R}^d$ 是基础衰减偏置向量
- $\sigma(\cdot)$ 是 sigmoid 函数
- 常数 $-0.606531 \approx -\ln(e-1)$，确保衰减值在合理范围内
- $\mathbf{w}_t \in (0, 1)^d$：逐元素衰减系数向量

#### 3.5 Attention Gate（注意力门 $\boldsymbol{\alpha}$）

$$
\boldsymbol{\alpha}_t = \sigma\left( \mathbf{W}_2^{(a)} \mathbf{W}_1^{(a)} \tilde{\mathbf{x}}_t^{(a)} + \mathbf{a}_0 \right)
$$

其中：
- $\mathbf{W}_1^{(a)}, \mathbf{W}_2^{(a)}$ 是 LoRA 分解矩阵
- $\mathbf{a}_0 \in \mathbb{R}^d$ 是偏置向量
- $\boldsymbol{\alpha}_t \in (0, 1)^d$ 控制 Delta Rule 的更新强度

#### 3.6 Output Gate（输出门）

$$
\mathbf{g}_t = \mathbf{W}_2^{(g)} \sigma\left( \mathbf{W}_1^{(g)} \tilde{\mathbf{x}}_t^{(g)} \right)
$$

---

### 4️⃣ Key 归一化与调制

#### 4.1 归一化 Key（用于 Delta Rule）

$$
\bar{\mathbf{k}}_t = \frac{\mathbf{k}_t \odot \boldsymbol{\kappa}_k}{\| \mathbf{k}_t \odot \boldsymbol{\kappa}_k \|_2}
$$

其中：
- $\boldsymbol{\kappa}_k \in \mathbb{R}^d$ 是可学习的缩放参数向量 (`k_k`)
- 归一化按每个注意力头独立进行（reshape 为 $[H, d_h]$ 后按 dim=1 归一化）
- $\bar{\mathbf{k}}_t$ 是单位向量，用于 Delta Rule 的正交投影

#### 4.2 调制 Key（用于状态更新）

$$
\mathbf{k}_t' = \mathbf{k}_t \odot \left( \mathbf{1} + (\boldsymbol{\alpha}_t - \mathbf{1}) \odot \boldsymbol{\kappa}_a \right)
$$

其中：
- $\boldsymbol{\kappa}_a \in \mathbb{R}^d$ 是可学习参数向量 (`k_a`)
- 当 $\boldsymbol{\alpha}_t = \mathbf{1}$ 时，$\mathbf{k}_t' = \mathbf{k}_t$
- 当 $\boldsymbol{\alpha}_t = \mathbf{0}$ 时，$\mathbf{k}_t' = \mathbf{k}_t \odot (\mathbf{1} - \boldsymbol{\kappa}_a)$

---

### 5️⃣ Value 混合 (First-Token Value Mixing)

**目的**: 将当前 value 与序列起始位置的 value 混合，增强长距离依赖建模。

$$
\mathbf{v}_t \leftarrow \begin{cases}
\mathbf{v}_t + (\mathbf{v}_0 - \mathbf{v}_t) \odot \sigma\left( \mathbf{v}^{(0)} + \mathbf{W}_2^{(v)} \mathbf{W}_1^{(v)} \tilde{\mathbf{x}}_t^{(v)} \right) & \text{if } \mathbf{v}_0 \text{ exists} \\
\mathbf{v}_t & \text{otherwise (and set } \mathbf{v}_0 := \mathbf{v}_t \text{)}
\end{cases}
$$

其中：
- $\mathbf{v}_0$ 是序列第一个 token 的 value 向量（`v_first`）
- $\mathbf{v}^{(0)}, \mathbf{W}_1^{(v)}, \mathbf{W}_2^{(v)}$ 是可学习参数
- 这种设计让模型能够在整个序列中保持对起始位置信息的访问

---

### 6️⃣ 状态更新 (WKV State Update) ⭐ 核心

这是 RWKV-7 的核心创新，结合了**线性注意力**与 **Delta Rule**：

#### 6.1 构建外积矩阵

**Value-Key 外积矩阵**（新信息写入）：
$$
\mathbf{VK}_t = \mathbf{v}_t \otimes \mathbf{k}_t' = \mathbf{v}_t (\mathbf{k}_t')^\top \in \mathbb{R}^{H \times d_h \times d_h}
$$

**Delta Rule 校正矩阵**（旧信息擦除）：
$$
\mathbf{AB}_t = -\bar{\mathbf{k}}_t \otimes (\bar{\mathbf{k}}_t \odot \boldsymbol{\alpha}_t) = -\bar{\mathbf{k}}_t (\bar{\mathbf{k}}_t \odot \boldsymbol{\alpha}_t)^\top \in \mathbb{R}^{H \times d_h \times d_h}
$$

#### 6.2 状态递推公式

$$
\boxed{
\mathbf{S}_t = \mathbf{S}_{t-1} \odot \mathbf{w}_t + \mathbf{S}_{t-1} \cdot \mathbf{AB}_t + \mathbf{VK}_t
}
$$

展开形式：

$$
\mathbf{S}_t = \underbrace{\mathbf{S}_{t-1} \odot \mathbf{w}_t}_{\text{(1) 指数衰减}} + \underbrace{\mathbf{S}_{t-1} \cdot \left( -\bar{\mathbf{k}}_t (\bar{\mathbf{k}}_t \odot \boldsymbol{\alpha}_t)^\top \right)}_{\text{(2) Delta Rule 校正}} + \underbrace{\mathbf{v}_t (\mathbf{k}_t')^\top}_{\text{(3) 新信息写入}}
$$

**三个组成部分的作用**：

| 项 | 公式 | 作用 | 来源 |
|---|---|---|---|
| (1) 衰减 | $\mathbf{S}_{t-1} \odot \mathbf{w}_t$ | 指数遗忘旧信息 | RWKV 传统设计 |
| (2) 校正 | $\mathbf{S}_{t-1} \cdot \mathbf{AB}_t$ | 沿 $\bar{\mathbf{k}}_t$ 方向擦除旧记忆 | **Delta Rule** |
| (3) 写入 | $\mathbf{v}_t (\mathbf{k}_t')^\top$ | 写入新的 key-value 关联 | 线性注意力 |

> **Delta Rule 的直觉**：在写入新信息之前，先将状态矩阵在 $\bar{\mathbf{k}}_t$ 方向上的分量减去。这实现了**正交化**，保证新旧信息不会混淆，类似于 Hopfield 网络的联想记忆更新规则。

---

### 7️⃣ 输出计算

#### 7.1 状态查询（主输出）

$$
\mathbf{o}_t^{(1)} = \text{GroupNorm}\left( \mathbf{S}_t \cdot \mathbf{r}_t \right)
$$

其中：
- $\mathbf{r}_t$ 被 reshape 为 $[H, d_h, 1]$ 后与 $\mathbf{S}_t \in \mathbb{R}^{H \times d_h \times d_h}$ 进行矩阵乘法
- 结果 reshape 为 $[d]$ 后通过 GroupNorm

#### 7.2 直接残差（Bonus Term）

$$
\mathbf{o}_t^{(2)} = \sum_{h=1}^{H} \left( \mathbf{r}_t^{(h)} \odot \mathbf{k}_t^{(h)} \odot \boldsymbol{\rho}_k^{(h)} \right) \cdot \mathbf{v}_t^{(h)}
$$

其中：
- $\boldsymbol{\rho}_k \in \mathbb{R}^{H \times d_h}$ 是可学习参数向量 (`r_k`)
- 上标 $(h)$ 表示第 $h$ 个注意力头的对应分量
- 这是一个绕过状态矩阵的直接 key-value 交互项

#### 7.3 合并

$$
\mathbf{o}_t = \mathbf{o}_t^{(1)} + \mathbf{o}_t^{(2)}
$$

---

### 8️⃣ 最终输出

$$
\boxed{
\mathbf{y}_t = \mathbf{W}_o \left( \mathbf{o}_t \odot \mathbf{g}_t \right)
}
$$

其中：
- $\mathbf{g}_t$ 是输出门向量（output gate）
- $\mathbf{W}_o \in \mathbb{R}^{d \times d}$ 是输出投影矩阵
- $\mathbf{y}_t \in \mathbb{R}^d$ 是最终输出向量

---

## 完整计算流程图

```
                              输入
                                │
              ┌─────────────────┴─────────────────┐
              ▼                                   ▼
         ┌─────────┐                        ┌───────────┐
         │  𝐱_t    │                        │  𝐱_{t-1}  │
         │   [d]   │                        │    [d]    │
         └────┬────┘                        └─────┬─────┘
              │                                   │
              └─────────────┬─────────────────────┘
                            ▼
                   ┌─────────────────┐
                   │ Δ𝐱_t = 𝐱_{t-1} │
                   │      - 𝐱_t      │
                   └────────┬────────┘
                            │
    ┌─────────┬─────────────┼─────────────┬─────────┬─────────┐
    ▼         ▼             ▼             ▼         ▼         ▼
┌───────┐ ┌───────┐     ┌───────┐     ┌───────┐ ┌───────┐ ┌───────┐
│𝐱̃_t^(r)│ │𝐱̃_t^(w)│     │𝐱̃_t^(k)│     │𝐱̃_t^(v)│ │𝐱̃_t^(a)│ │𝐱̃_t^(g)│
└───┬───┘ └───┬───┘     └───┬───┘     └───┬───┘ └───┬───┘ └───┬───┘
    │         │             │             │         │         │
    ▼         ▼             ▼             ▼         ▼         ▼
┌───────┐ ┌───────┐     ┌───────┐     ┌───────┐ ┌───────┐ ┌───────┐
│  𝐖_r  │ │LoRA_w │     │  𝐖_k  │     │  𝐖_v  │ │LoRA_a │ │LoRA_g │
└───┬───┘ └───┬───┘     └───┬───┘     └───┬───┘ └───┬───┘ └───┬───┘
    │         │             │             │         │         │
    ▼         ▼             ▼             ▼         ▼         ▼
   𝐫_t       𝐰_t           𝐤_t           𝐯_t       𝛂_t       𝐠_t
    │         │             │             │         │         │
    │         │        ┌────┴────┐        │         │         │
    │         │        │ 𝐤̄_t=    │        │         │         │
    │         │        │normalize│◄───────┼─────────┤         │
    │         │        └────┬────┘        │         │         │
    │         │             │             │         │         │
    │         │        ┌────┴─────────────┴─────────┤         │
    │         │        │  𝐤'_t = 𝐤_t ⊙ (𝟏+(𝛂_t-𝟏)⊙𝛋_a)      │
    │         │        └────┬────────────────────────┘        │
    │         │             │             │                   │
    │         │             │        ┌────┴─────┐             │
    │         │             │        │ 𝐯_first  │             │
    │         │             │        │  混合?   │             │
    │         │             │        └────┬─────┘             │
    │         │             │             │                   │
    │         │      ┌──────┴─────────────┴──────┐            │
    │         │      │  𝐕𝐊_t = 𝐯_t ⊗ 𝐤'_t        │            │
    │         │      │  𝐀𝐁_t = -𝐤̄_t ⊗ (𝐤̄_t⊙𝛂_t)  │            │
    │         │      └──────────┬────────────────┘            │
    │         │                 │                             │
    │         │      ┌──────────┴─────────────────────────┐   │
    │         │      │  𝐒_t = 𝐒_{t-1} ⊙ 𝐰_t               │   │
    │         │      │       + 𝐒_{t-1} · 𝐀𝐁_t             │   │
    │         │      │       + 𝐕𝐊_t                       │   │
    │         │      └──────────┬─────────────────────────┘   │
    │         │                 │                             │
    │    ┌────┴─────────────────┼──────────────────┐          │
    │    │                      │                  │          │
    │    │           ┌──────────┴──────────┐       │          │
    │    │           │ 𝐨_t^(1) = GroupNorm │       │          │
    │    │           │         (𝐒_t · 𝐫_t) │       │          │
    │    │           └──────────┬──────────┘       │          │
    │    │                      │                  │          │
    │    │           ┌──────────┴──────────┐       │          │
    │    │           │ 𝐨_t^(2) = Σ(𝐫⊙𝐤⊙𝛒)·𝐯│       │          │
    │    │           └──────────┬──────────┘       │          │
    │    │                      │                  │          │
    │    │           ┌──────────┴──────────┐       │          │
    │    │           │   𝐨_t = 𝐨_t^(1)     │       │          │
    │    │           │       + 𝐨_t^(2)     │       │          │
    │    │           └──────────┬──────────┘       │          │
    │    │                      │                  │          │
    │    │                      │                  └──────────┤
    │    │                      │                             │
    │    │           ┌──────────┴────────────────────────────┐│
    │    │           │       𝐲_t = 𝐖_o(𝐨_t ⊙ 𝐠_t)            ││
    │    │           └──────────┬────────────────────────────┘│
    │    │                      │                             │
    │    │                      ▼                             │
    │    │                ┌──────────┐                        │
    │    │                │   𝐲_t    │                        │
    │    │                │   [d]    │                        │
    │    │                └──────────┘                        │
    │    │                                                    │
    │    └────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                          输出                                │
│  (𝐲_t, 𝐱_t, 𝐒_t, 𝐯_first)                                   │
│   [d]   [d]  [H,d_h,d_h]  Option<[d]>                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 张量形状汇总

| 代码变量 | 数学符号 | 类型 | 形状 | 说明 |
|----------|----------|------|------|------|
| `x`, `x_prev` | $\mathbf{x}_t$, $\mathbf{x}_{t-1}$ | 向量 | $[d]$ | 输入 |
| `xx` | $\Delta \mathbf{x}_t$ | 向量 | $[d]$ | 时间差分 |
| `xr`, `xw`, ... | $\tilde{\mathbf{x}}_t^{(\cdot)}$ | 向量 | $[d]$ | 混合后的特征 |
| `r` | $\mathbf{r}_t$ | 向量 | $[d]$ | Receptance |
| `w` | $\mathbf{w}_t$ | 向量 | $[d] \to [H, 1, d_h]$ | 衰减权重 |
| `k` | $\mathbf{k}_t \to \mathbf{k}_t'$ | 向量 | $[d]$ | Key（原始 → 调制后）|
| `v` | $\mathbf{v}_t$ | 向量 | $[d]$ | Value |
| `a` | $\boldsymbol{\alpha}_t$ | 向量 | $[d]$ | Attention gate |
| `g` | $\mathbf{g}_t$ | 向量 | $[d]$ | Output gate |
| `kk` | $\bar{\mathbf{k}}_t$ | 向量 | $[d]$ | 归一化的 key |
| `vk` | $\mathbf{VK}_t$ | 矩阵 | $[H, d_h, d_h]$ | $\mathbf{v}_t \otimes \mathbf{k}_t'$ 外积 |
| `ab` | $\mathbf{AB}_t$ | 矩阵 | $[H, d_h, d_h]$ | Delta Rule 矩阵 |
| `vk_state` | $\mathbf{S}_t$ | 矩阵 | $[H, d_h, d_h]$ | 状态张量 |
| `out` | $\mathbf{y}_t$ | 向量 | $[d]$ | 最终输出 |

---

## 关键公式总结卡片

### Token Mixing
$$
\tilde{\mathbf{x}}_t = \mathbf{x}_t + (\mathbf{x}_{t-1} - \mathbf{x}_t) \odot \boldsymbol{\mu}
$$

### Decay Weight
$$
\mathbf{w}_t = \exp\left( \sigma(\mathbf{w}_{\text{combined}}) \cdot (-0.606531) \right) \in (0, 1)^d
$$

### Delta Rule 状态更新
$$
\mathbf{S}_t = \mathbf{S}_{t-1} \odot \mathbf{w}_t - \mathbf{S}_{t-1} \bar{\mathbf{k}}_t (\bar{\mathbf{k}}_t \odot \boldsymbol{\alpha}_t)^\top + \mathbf{v}_t (\mathbf{k}_t')^\top
$$

### 输出计算
$$
\mathbf{y}_t = \mathbf{W}_o \left( \left[ \text{GroupNorm}(\mathbf{S}_t \cdot \mathbf{r}_t) + \mathbf{Bonus}_t \right] \odot \mathbf{g}_t \right)
$$

---

## 与标准 Transformer 的对比

| 特性 | Transformer | RWKV-7 |
|------|-------------|--------|
| 注意力复杂度 | $O(T^2 d)$ | $O(T d^2)$ |
| 状态大小 | $O(T)$ KV Cache | $O(1)$ 固定大小 $[H, d_h, d_h]$ |
| 信息更新 | Softmax 加权 | Delta Rule + 指数衰减 |
| 推理模式 | 需要完整历史 | 可逐 token RNN 推理 |

---

*此文档基于 RWKV-7 "Goose" TimeMix 实现生成，参考 Delta Rule 线性注意力机制设计*
