# TimeMix `forward_rnn` 前向传播流程图

本文档详细说明 RWKV v7 模型中 `TimeMix::forward_rnn` 函数的前向传播计算流程。

> **参考**: RWKV-7 "Goose" 架构，结合了线性注意力与 Delta Rule 的状态更新机制，实现了 $O(Td^2)$ 的线性时间复杂度。

---

## 符号说明

| 符号 | 含义 | 维度 |
|------|------|------|
| $d$ | 模型维度 (`d_model`) | 标量 |
| $H$ | 注意力头数 (`n_heads`) | 标量 |
| $d_h$ | 每头维度 (`head_size = d/H`) | 标量 |
| $t$ | 当前时间步 | 标量 |
| $x_t$ | 当前时刻输入 | $\mathbb{R}^d$ |
| $x_{t-1}$ | 前一时刻输入 | $\mathbb{R}^d$ |
| $S_{t-1}$ | 前一时刻状态矩阵 (`vk_state`) | $\mathbb{R}^{H \times d_h \times d_h}$ |

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
\Delta x_t = x_{t-1} - x_t
$$

其中：
- $x_t \in \mathbb{R}^d$：当前时刻的隐藏状态输入
- $x_{t-1} \in \mathbb{R}^d$：前一时刻的隐藏状态输入
- $\Delta x_t \in \mathbb{R}^d$：时间差分向量

---

### 2️⃣ Token Mixing (时间混合插值)

**目的**: 通过学习到的混合系数，在当前时刻和前一时刻特征之间进行加权插值，为不同的下游计算提供定制化的输入。

$$
\begin{aligned}
\tilde{x}_t^{(r)} &= x_t + \Delta x_t \odot \mu_r \\
\tilde{x}_t^{(w)} &= x_t + \Delta x_t \odot \mu_w \\
\tilde{x}_t^{(k)} &= x_t + \Delta x_t \odot \mu_k \\
\tilde{x}_t^{(v)} &= x_t + \Delta x_t \odot \mu_v \\
\tilde{x}_t^{(a)} &= x_t + \Delta x_t \odot \mu_a \\
\tilde{x}_t^{(g)} &= x_t + \Delta x_t \odot \mu_g
\end{aligned}
$$

其中：
- $\mu_r, \mu_w, \mu_k, \mu_v, \mu_a, \mu_g \in \mathbb{R}^d$ 是可学习的混合系数（对应代码中的 `x_r`, `x_w`, `x_k`, `x_v`, `x_a`, `x_g`）
- $\odot$ 表示逐元素乘法 (Hadamard product)
- 当 $\mu = 0$ 时，$\tilde{x}_t = x_t$（使用当前输入）
- 当 $\mu = 1$ 时，$\tilde{x}_t = x_{t-1}$（使用前一时刻输入）

---

### 3️⃣ 核心投影计算 (Linear Projections)

#### 3.1 Receptance（接收门）

$$
r_t = W_r \tilde{x}_t^{(r)}
$$

其中 $W_r \in \mathbb{R}^{d \times d}$ 是 receptance 投影矩阵。

#### 3.2 Key（键向量）

$$
k_t = W_k \tilde{x}_t^{(k)}
$$

其中 $W_k \in \mathbb{R}^{d \times d}$ 是 key 投影矩阵。

#### 3.3 Value（值向量）

$$
v_t = W_v \tilde{x}_t^{(v)}
$$

其中 $W_v \in \mathbb{R}^{d \times d}$ 是 value 投影矩阵。

#### 3.4 Decay Weight（衰减权重）

RWKV-7 中的衰减权重通过 LoRA 结构动态计算：

$$
w_t = \exp\left( \sigma\left( W_2^{(w)} \tanh\left( W_1^{(w)} \tilde{x}_t^{(w)} \right) + w_0 \right) \cdot (-0.606531) \right)
$$

其中：
- $W_1^{(w)} \in \mathbb{R}^{d \times d_{\text{lora}}}$, $W_2^{(w)} \in \mathbb{R}^{d_{\text{lora}} \times d}$ 是 LoRA 分解矩阵
- $w_0 \in \mathbb{R}^d$ 是基础衰减偏置
- $\sigma(\cdot)$ 是 sigmoid 函数
- 常数 $-0.606531 \approx -\ln(e-1)$，确保衰减值在合理范围内
- $w_t \in (0, 1)^d$：逐元素衰减系数

#### 3.5 Attention Gate（注意力门 $\alpha$）

$$
\alpha_t = \sigma\left( W_2^{(a)} W_1^{(a)} \tilde{x}_t^{(a)} + a_0 \right)
$$

其中：
- $W_1^{(a)}, W_2^{(a)}$ 是 LoRA 分解矩阵
- $a_0 \in \mathbb{R}^d$ 是偏置
- $\alpha_t \in (0, 1)^d$ 控制 Delta Rule 的更新强度

#### 3.6 Output Gate（输出门）

$$
g_t = W_2^{(g)} \sigma\left( W_1^{(g)} \tilde{x}_t^{(g)} \right)
$$

---

### 4️⃣ Key 归一化与调制

#### 4.1 归一化 Key（用于 Delta Rule）

$$
\bar{k}_t = \frac{k_t \odot \kappa_k}{\| k_t \odot \kappa_k \|_2}
$$

其中：
- $\kappa_k \in \mathbb{R}^d$ 是可学习的缩放参数 (`k_k`)
- 归一化按每个注意力头独立进行（reshape 为 $[H, d_h]$ 后按 dim=1 归一化）
- $\bar{k}_t$ 是单位向量，用于 Delta Rule 的正交投影

#### 4.2 调制 Key（用于状态更新）

$$
k_t' = k_t \odot \left( 1 + (\alpha_t - 1) \odot \kappa_a \right)
$$

其中：
- $\kappa_a \in \mathbb{R}^d$ 是可学习参数 (`k_a`)
- 当 $\alpha_t = 1$ 时，$k_t' = k_t$
- 当 $\alpha_t = 0$ 时，$k_t' = k_t \odot (1 - \kappa_a)$

---

### 5️⃣ Value 混合 (First-Token Value Mixing)

**目的**: 将当前 value 与序列起始位置的 value 混合，增强长距离依赖建模。

$$
v_t = \begin{cases}
v_t + (v_0 - v_t) \odot \sigma\left( v^{(0)} + W_2^{(v)} W_1^{(v)} \tilde{x}_t^{(v)} \right) & \text{if } v_0 \text{ exists} \\
v_t & \text{otherwise (and set } v_0 := v_t \text{)}
\end{cases}
$$

其中：
- $v_0$ 是序列第一个 token 的 value（`v_first`）
- $v^{(0)}, W_1^{(v)}, W_2^{(v)}$ 是可学习参数
- 这种设计让模型能够在整个序列中保持对起始位置信息的访问

---

### 6️⃣ 状态更新 (WKV State Update) ⭐ 核心

这是 RWKV-7 的核心创新，结合了**线性注意力**与 **Delta Rule**：

#### 6.1 构建外积矩阵

**Value-Key 外积**（新信息写入）：
$$
\text{VK}_t = v_t \otimes k_t' = v_t (k_t')^\top \in \mathbb{R}^{H \times d_h \times d_h}
$$

**Delta Rule 校正矩阵**（旧信息擦除）：
$$
\text{AB}_t = -\bar{k}_t \otimes (\bar{k}_t \odot \alpha_t) = -\bar{k}_t (\bar{k}_t \odot \alpha_t)^\top \in \mathbb{R}^{H \times d_h \times d_h}
$$

#### 6.2 状态递推公式

$$
\boxed{
S_t = S_{t-1} \odot w_t + S_{t-1} \cdot \text{AB}_t + \text{VK}_t
}
$$

展开形式：

$$
S_t = \underbrace{S_{t-1} \odot w_t}_{\text{(1) 指数衰减}} + \underbrace{S_{t-1} \cdot \left( -\bar{k}_t (\bar{k}_t \odot \alpha_t)^\top \right)}_{\text{(2) Delta Rule 校正}} + \underbrace{v_t (k_t')^\top}_{\text{(3) 新信息写入}}
$$

**三个组成部分的作用**：

| 项 | 公式 | 作用 | 来源 |
|---|---|---|---|
| (1) 衰减 | $S_{t-1} \odot w_t$ | 指数遗忘旧信息 | RWKV 传统设计 |
| (2) 校正 | $S_{t-1} \cdot \text{AB}_t$ | 沿 $\bar{k}_t$ 方向擦除旧记忆 | **Delta Rule** |
| (3) 写入 | $v_t (k_t')^\top$ | 写入新的 key-value 关联 | 线性注意力 |

> **Delta Rule 的直觉**：在写入新信息之前，先将状态矩阵在 $\bar{k}_t$ 方向上的分量减去。这实现了**正交化**，保证新旧信息不会混淆，类似于 Hopfield 网络的联想记忆更新规则。

---

### 7️⃣ 输出计算

#### 7.1 状态查询（主输出）

$$
o_t^{(1)} = \text{GroupNorm}\left( S_t \cdot r_t \right)
$$

其中：
- $r_t$ 被 reshape 为 $[H, d_h, 1]$ 后与 $S_t \in \mathbb{R}^{H \times d_h \times d_h}$ 进行矩阵乘法
- 结果 reshape 为 $[d]$ 后通过 GroupNorm

#### 7.2 直接残差（Bonus Term）

$$
o_t^{(2)} = \sum_{h=1}^{H} \left( r_t^{(h)} \odot k_t^{(h)} \odot \rho_k^{(h)} \right) \cdot v_t^{(h)}
$$

其中：
- $\rho_k \in \mathbb{R}^{H \times d_h}$ 是可学习参数 (`r_k`)
- 上标 $(h)$ 表示第 $h$ 个注意力头的对应分量
- 这是一个绕过状态矩阵的直接 key-value 交互项

#### 7.3 合并

$$
o_t = o_t^{(1)} + o_t^{(2)}
$$

---

### 8️⃣ 最终输出

$$
\boxed{
\text{output}_t = W_o \left( o_t \odot g_t \right)
}
$$

其中：
- $g_t$ 是输出门（output gate）
- $W_o \in \mathbb{R}^{d \times d}$ 是输出投影矩阵

---

## 完整计算流程图

```
                            输入
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
       ┌─────────┐                        ┌───────────┐
       │   x_t   │                        │  x_{t-1}  │
       │   [d]   │                        │    [d]    │
       └────┬────┘                        └─────┬─────┘
            │                                   │
            └─────────────┬─────────────────────┘
                          ▼
                 ┌─────────────────┐
                 │ Δx_t = x_{t-1}  │
                 │      - x_t      │
                 └────────┬────────┘
                          │
    ┌─────────┬───────────┼───────────┬─────────┬─────────┐
    ▼         ▼           ▼           ▼         ▼         ▼
┌───────┐ ┌───────┐   ┌───────┐   ┌───────┐ ┌───────┐ ┌───────┐
│x̃_t^(r)│ │x̃_t^(w)│   │x̃_t^(k)│   │x̃_t^(v)│ │x̃_t^(a)│ │x̃_t^(g)│
└───┬───┘ └───┬───┘   └───┬───┘   └───┬───┘ └───┬───┘ └───┬───┘
    │         │           │           │         │         │
    ▼         ▼           ▼           ▼         ▼         ▼
┌───────┐ ┌───────┐   ┌───────┐   ┌───────┐ ┌───────┐ ┌───────┐
│  W_r  │ │LoRA_w │   │  W_k  │   │  W_v  │ │LoRA_a │ │LoRA_g │
└───┬───┘ └───┬───┘   └───┬───┘   └───┬───┘ └───┬───┘ └───┬───┘
    │         │           │           │         │         │
    ▼         ▼           ▼           ▼         ▼         ▼
   r_t       w_t         k_t         v_t       α_t       g_t
    │         │           │           │         │         │
    │         │      ┌────┴────┐      │         │         │
    │         │      │ k̄_t=    │      │         │         │
    │         │      │normalize│◄─────┼─────────┤         │
    │         │      └────┬────┘      │         │         │
    │         │           │           │         │         │
    │         │      ┌────┴───────────┴─────────┤         │
    │         │      │  k'_t = k_t ⊙ (1+(α_t-1)⊙κ_a)     │
    │         │      └────┬──────────────────────┘        │
    │         │           │           │                   │
    │         │           │      ┌────┴─────┐             │
    │         │           │      │ v_first  │             │
    │         │           │      │  混合?   │             │
    │         │           │      └────┬─────┘             │
    │         │           │           │                   │
    │         │    ┌──────┴───────────┴──────┐            │
    │         │    │  VK_t = v_t ⊗ k'_t      │            │
    │         │    │  AB_t = -k̄_t ⊗ (k̄_t⊙α_t) │            │
    │         │    └──────────┬──────────────┘            │
    │         │               │                           │
    │         │    ┌──────────┴───────────────────────┐   │
    │         │    │  S_t = S_{t-1} ⊙ w_t             │   │
    │         │    │      + S_{t-1} · AB_t            │   │
    │         │    │      + VK_t                      │   │
    │         │    └──────────┬───────────────────────┘   │
    │         │               │                           │
    │    ┌────┴───────────────┼────────────────┐          │
    │    │                    │                │          │
    │    │         ┌──────────┴──────────┐     │          │
    │    │         │ o_t^(1) = GroupNorm │     │          │
    │    │         │        (S_t · r_t)  │     │          │
    │    │         └──────────┬──────────┘     │          │
    │    │                    │                │          │
    │    │         ┌──────────┴──────────┐     │          │
    │    │         │ o_t^(2) = Σ(r⊙k⊙ρ)·v│     │          │
    │    │         └──────────┬──────────┘     │          │
    │    │                    │                │          │
    │    │         ┌──────────┴──────────┐     │          │
    │    │         │  o_t = o_t^(1)      │     │          │
    │    │         │      + o_t^(2)      │     │          │
    │    │         └──────────┬──────────┘     │          │
    │    │                    │                │          │
    │    │                    │                └──────────┤
    │    │                    │                           │
    │    │         ┌──────────┴──────────────────────────┐│
    │    │         │      output_t = W_o(o_t ⊙ g_t)      ││
    │    │         └──────────┬──────────────────────────┘│
    │    │                    │                           │
    │    │                    ▼                           │
    │    │              ┌──────────┐                      │
    │    │              │output_t  │                      │
    │    │              │   [d]    │                      │
    │    │              └──────────┘                      │
    │    │                                                │
    │    └────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                         输出                                 │
│  (output_t, x_t, S_t, v_first)                              │
│    [d]      [d]  [H,d_h,d_h]  Option<[d]>                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 张量形状汇总

| 变量 | 数学符号 | 形状 | 说明 |
|------|----------|------|------|
| `x`, `x_prev` | $x_t$, $x_{t-1}$ | $[d]$ | 输入向量 |
| `xx` | $\Delta x_t$ | $[d]$ | 时间差分 |
| `xr`, `xw`, ... | $\tilde{x}_t^{(\cdot)}$ | $[d]$ | 混合后的特征 |
| `r` | $r_t$ | $[d]$ | Receptance |
| `w` | $w_t$ | $[d] \to [H, 1, d_h]$ | 衰减权重 |
| `k` | $k_t$ → $k_t'$ | $[d]$ | Key（原始→调制后）|
| `v` | $v_t$ | $[d]$ | Value |
| `a` | $\alpha_t$ | $[d]$ | Attention gate |
| `g` | $g_t$ | $[d]$ | Output gate |
| `kk` | $\bar{k}_t$ | $[d]$ | 归一化的 key |
| `vk` | $\text{VK}_t$ | $[H, d_h, d_h]$ | $v_t \otimes k_t'$ 外积 |
| `ab` | $\text{AB}_t$ | $[H, d_h, d_h]$ | Delta Rule 矩阵 |
| `vk_state` | $S_t$ | $[H, d_h, d_h]$ | 状态张量 |
| `out` | $\text{output}_t$ | $[d]$ | 最终输出 |

---

## 关键公式总结卡片

### Token Mixing
$$
\tilde{x}_t = x_t + (x_{t-1} - x_t) \odot \mu
$$

### Decay Weight
$$
w_t = \exp\left( \sigma(w_{\text{combined}}) \cdot (-0.606531) \right) \in (0, 1)^d
$$

### Delta Rule 状态更新
$$
S_t = S_{t-1} \odot w_t - S_{t-1} \bar{k}_t (\bar{k}_t \odot \alpha_t)^\top + v_t (k_t')^\top
$$

### 输出计算
$$
\text{output}_t = W_o \left( \left[ \text{GroupNorm}(S_t \cdot r_t) + \text{Bonus}_t \right] \odot g_t \right)
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
