[← 返回主目录](index.md)

# 第一章：早期基石与 Transformer 体系

## 1. 引言

近年来，大语言模型（LLM）彻底改变了人工智能领域。从早期的 RNN 到如今的 Transformer，再到新兴的线性注意力架构，模型架构的演进始终围绕着两个核心矛盾：**训练效率（并行性）**与**推理效率（成本与长序列能力）**。本文将梳理这一演化历史，重点阐述 Transformer 为何成功，以及线性注意力（Linear Attention）如何通过三代演进，试图在保持 Transformer 性能的同时找回 RNN 的推理优势。

## 2. RNN ：串行计算

在 Transformer 出现之前，RNN（循环神经网络）及其变体 LSTM/GRU 是处理序列数据的主流。然而，它们在处理大规模数据和长序列时遇到了难以克服的物理障碍。

### 2.1 无法并行训练 (Sequential Computation)

RNN 的核心公式定义了当前时刻的状态 $h_t$ 严格依赖于上一时刻的状态 $h_{t-1}$：

$$ h*t = \sigma(W_h h*{t-1} + W_x x_t + b) $$

其中 $\sigma$ 是非线性激活函数（如 tanh 或 ReLU）。

- **时间依赖链**：为了计算第 100 个时间步的 $h_{100}$，必须先计算 $h_{99}$，而 $h_{99}$ 又依赖 $h_{98}$……直到 $h_0$。
- **GPU 利用率低**：现代 GPU 擅长进行大规模矩阵并行运算（如一次性计算所有 Token 的 $Q \cdot K^T$），但 RNN 这种“接力跑”式的计算模式迫使 GPU 必须等待上一步完成才能进行下一步。这导致在长序列训练时，GPU 大部分计算单元处于闲置状态，训练速度极慢。

## 3. Transformer ：并行与注意力

2017 年，Google 提出的《Attention Is All You Need》改变了一切。它不仅是一个新架构，更是一种计算范式的转移。Transformer 的强大不仅仅在于并行训练，更在于其组件设计的精妙之处。

### 3.1 输入层：词向量空间 (Token Embeddings)

在进入复杂的注意力机制之前，我们首先要解决的问题是：**如何让计算机理解单词？**

- **离散符号到连续向量**：计算机无法直接处理 "Apple" 或 "Banana" 这样的字符串。我们需要建立一个查找表（Embedding Table），将每个词映射为一个固定长度的实数向量（例如 4096 维）。
- **语义空间**：这个向量空间具有良好的几何性质。语义相近的词，在空间中的距离更近。
  - **经典例子**： $\vec{King} - \vec{Man} + \vec{Woman} \approx \vec{Queen}$ 。
  - 这意味着模型在输入层就已经具备了初步的推理能力。

### 3.2 训练范式：自回归生成 (Autoregression)

现代 LLM（如 GPT 系列）通常采用**自回归（Autoregressive）**的方式进行训练。

- **目标**：预测下一个 Token。即计算概率 $P(w_t | w_1, w_2, \dots, w_{t-1})$。
- **过程**：模型读入当前的上下文，预测下一个最可能出现的词。然后将这个词加入上下文，继续预测下下个词。这就像人类写文章一样，一个字一个字地往后写。

### 3.3 Self-Attention：信息的“路由”与“聚合”

Transformer 的核心在于自注意力机制（Self-Attention）。对于初学者来说，直接看矩阵公式 $Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V$ 可能比较抽象。让我们把它拆解开来，看看对于序列中的**某一个词**，到底发生了什么。

#### 3.3.1 核心直觉：回顾历史 (Looking Back)

在自回归推理中，模型只能看到**当前和过去**的信息。
想象模型正在生成句子：“**Alice** **likes** ...”。
当前输入是 "likes"，模型需要预测下一个词（比如 "Bob"）。为了做出准确预测，模型必须回顾历史，搞清楚“谁”在喜欢。

Self-Attention 的本质就是：**用当前词（Query）去查询历史记忆中所有的词（Key），根据匹配程度（Attention Score）将它们的内容（Value）加权融合过来。**

**Q 与 K 的相似度理论**：
为什么我们要计算 $Q \cdot K^T$？在几何上，两个向量的点积（Dot Product）衡量了它们的**相似度**或**对齐程度**。

- 如果 $Q$ 和 $K$ 指向相同的方向，点积最大（关注度最高）。
- 如果 $Q$ 和 $K$ 垂直（无关），点积为 0（不关注）。
- 这就像数据库查询：$Q$ 是你的搜索关键词，$K$ 是数据库中每条记录的索引标签。点积越高，说明这条记录越符合你的搜索意图。

#### 3.3.2 序列形式公式 (The Sequence Form)

对于序列中的第 $t$ 个 Token，其输出向量 $y_t$ 的计算公式如下（注意求和上限是 $t$，不能看未来）：

$$ y*t = \sum*{i=1}^{t} \underbrace{\text{softmax}\left( \frac{q*t \cdot k_i^T}{\sqrt{d}} \right)}*{\alpha\_{t,i} \text{ (注意力权重)}} \cdot v_i $$

- $q_t$：当前词 $t$ 的查询向量（Query）。
- $k_i$：历史中第 $i$ 个词的键向量（Key），$i \le t$。
- $v_i$：历史中第 $i$ 个词的值向量（Value）。
- $\alpha_{t,i}$：第 $t$ 个词对第 $i$ 个词的关注度。

#### 3.3.3 图解与例子：一次真实的推理步骤

假设模型已经处理了 "Alice"，现在轮到 **"likes"**。
**当前时刻**：$t=2$
**输入**：`likes`
**历史上下文**：`[Alice, likes]` (注意：`Bob` 还没出现，是我们要预测的目标)

我们来看看如何计算 "likes" 的输出向量，以便预测下一个词。

**第一步：打分 (Matching)**
"likes" 发出查询 $q_{\text{likes}}$，回顾历史（包括自己）：

```text
q_likes · k_Alice  = 0.9  (很高，因为需要找到主语是谁)
q_likes · k_likes  = 0.5  (关注自己，提取动词本身的含义)
```

_(注：此时模型完全不知道 Bob 的存在)_

**第二步：归一化 (Softmax)**
将分数转化为概率：

```text
Softmax([0.9, 0.5]) ≈ [0.6, 0.4]
          ↑    ↑
       Alice likes
```

**第三步：聚合 (Aggregation)**
根据权重，融合历史信息：

```text
y_likes = 0.6 * v_Alice + 0.4 * v_likes
```

**结果**：
得到的 $y_{\text{likes}}$ 向量融合了 "Alice" (主语) 和 "likes" (谓语) 的信息。
这个向量经过后续层处理，最终会预测出高概率的词：**"Bob"** 或 **"Apples"**。

**可视化流程**：

```text
       [v_Alice]      [v_likes]      [v_???]
           |              |             ^
           | 0.6          | 0.4         | 预测
           |              |             |
           +--------------+             |
                  ↓                     |
               y_likes -----------------+
```

#### 3.3.4 多头注意力 (Multi-Head Attention)

- 如果只有一个头，"likes" 可能只能关注到语法成分。但它还需要关注情感色彩等。
- 多头机制允许模型在不同的**子空间**里并行关注不同的特征。Head 1 关注指代，Head 2 关注句法依存，Head 3 关注上下文情感。

### 3.4 MLP (Feed-Forward Networks)：知识的“存储”与“加工”

在 Attention 层之后，总是紧跟着一个 MLP（多层感知机）层。为什么？

- **结构**：通常是 `Linear -> Activation (GeLU/SwiGLU) -> Linear`。它独立地作用于每个 Token，不涉及 Token 间的交互。
- **分工**：
  - **Attention** 负责 **Token 之间** 的信息流动（Mixing information between tokens）。它把上下文信息搬运到当前 Token。
  - **MLP** 负责 **Token 内部** 的信息加工（Mixing information within a token）。
- **深度理解：键值记忆网络 (Key-Value Memories)**：
  - 有研究（Geva et al.）认为，MLP 层充当了模型的**知识库**。
  - 第一层 Linear 类似于检测某种模式（Key），激活函数筛选模式，第二层 Linear 输出该模式对应的属性或结果（Value）。
  - 例如，Attention 把 "法国" 和 "首都" 搬运到了一起，MLP 层检测到这个组合，然后输出 "巴黎"。

### 3.5 位置编码：赋予序列秩序

Self-Attention 本质上是集合运算（置换不变性），无法区分 "A hit B" 和 "B hit A"。必须显式注入位置信息。

- **绝对位置编码 (Sinusoidal / Learnable)**：
  - 直接将位置向量 $P_i$ 加到输入 $X_i$ 上。
  - **缺点**：外推性差。训练时只见过长度 1024，推理时遇到 2048 就不知道 $P_{2048}$ 是什么，或者正弦波从未见过的相位会导致混乱。
- **相对位置编码 (ALiBi)**：
  - 不加在输入上，而是直接加在 Attention Score 上。距离越远，扣分越多（$QK^T - m \cdot |i-j|$）。
  - **优点**：外推性极强，训练短序列，推理长序列效果好。
- **旋转位置编码 (RoPE - Rotary Positional Embedding)**：
  - **当前主流**（LLaMA, Qwen 等使用）。
  - 通过旋转向量的角度来编码相对位置。$q_i$ 旋转 $i\theta$，$k_j$ 旋转 $j\theta$，它们的点积只与相对距离 $(i-j)\theta$ 有关。
  - **优点**：完美结合了绝对位置的实现便利性和相对位置的数学性质，外推性较好。

### 3.6 残差连接与层归一化：深度的基石 (Residuals & LayerNorm)

为什么 Transformer 可以堆叠到上百层而不会梯度消失？这归功于 **残差连接 (Residual Connection)**。

#### 3.6.1 残差连接：缓解梯度消失

在深层网络中，如果每一层都是 $x_{l+1} = F(x_l)$，那么反向传播时的梯度是连乘的：$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_L} \cdot \prod \frac{\partial F}{\partial x}$。一旦某层的导数小于 1，连乘几十次后梯度就会趋近于 0，导致**梯度消失**。

Transformer 采用了残差结构：

$$ x\_{l+1} = x_l + F(x_l) $$

其中 $F(x_l)$ 是 Attention 或 MLP 层的计算结果。

**数学原理**：
根据链式法则，反向传播时的梯度计算如下：

$$ \frac{\partial x\_{l+1}}{\partial x_l} = 1 + \frac{\partial F(x_l)}{\partial x_l} $$

最终的梯度流向是：

$$ \frac{\partial L}{\partial x*l} = \frac{\partial L}{\partial x*{l+1}} \cdot \left( 1 + \frac{\partial F}{\partial x_l} \right) $$

- **恒等映射 (Identity Mapping)**：公式中的常数项 **$1$** 保证了梯度可以直接传回前一层。
- **梯度保持**：即使某层的非线性变换部分 $\frac{\partial F}{\partial x_l}$ 梯度很小，梯度信号依然可以通过 $1$ 这一项继续向前传播。这有效缓解了梯度消失问题，使得训练深层网络成为可能。

#### 3.6.2 层归一化 (LayerNorm / RMSNorm)

$$ x\_{l+1} = \text{LayerNorm}(x_l + F(x_l)) $$

- **作用**：将每一层的输出归一化到均值为 0，方差为 1。
- **意义**：这防止了数值在深层网络中剧烈波动（梯度爆炸），进一步稳定了训练过程。

### 3.7 表达能力与深度：电路的升级

为什么 Transformer 需要堆叠几十层？

- **残差流 (Residual Stream)**：
  - Transformer 的主干是一个贯穿始终的向量流 $x + \text{Sublayer}(x)$。
  - 每一层（Attention 或 MLP）都是从这个流中“读”出信息，处理后，再“写”回一个增量（Residual Update）。
- **归纳头 (Induction Heads) 与多步推理**：
  - **第 1 层**：可能只是简单的词法关注。
  - **第 2 层**：可以利用第 1 层搬运来的信息。
  - **Induction Head** 是 In-context Learning 的核心电路。它由两层 Attention 组成：
    - Layer 1 关注当前 Token 的上一个 Token（复制历史）。
    - Layer 2 搜索历史中出现过类似“上一个 Token”的地方，并把那之后的 Token 搬运过来。
    - **功能**：实现了“如果 A 后面通常跟 B，那么这次看到 A，我也预测 B”的模式复制能力。
  - 随着层数加深，模型能组合出极其复杂的逻辑电路，实现推理能力。

### 3.8 自回归推理：从向量回归文本

当模型经过几十层的计算，输出了最后一个 Token 的向量 $h_{last}$ 后，我们如何得到文本？

1.  **Unembedding (反嵌入)**：
    - 将 $h_{last}$ 乘以 Embedding 矩阵的转置（或单独的输出头），得到一个长度为词表大小（如 50,000）的向量 **Logits**。
    - Logits 中的每个数值代表对应词的“得分”。

2.  **Softmax**：
    - 将 Logits 转化为概率分布 $P$。
    - $P(\text{"apple"}) = 0.1, P(\text{"banana"}) = 0.05, \dots$

3.  **采样 (Sampling)**：
    - **Greedy**：直接选概率最大的词。
    - **Top-K / Top-P (Nucleus)**：从概率最高的 K 个词或累积概率为 P 的集合中随机抽取。这增加了生成的多样性。

4.  **循环 (Loop)**：
    - 选出的词被追加到输入序列末尾，再次输入模型，预测下一个词。这就是**自回归**过程。

## 待办事项 (TODO)

- [ ] **并行训练与矩阵形式**：解释 Transformer 如何在训练阶段利用矩阵运算一次性处理整个序列（Teacher Forcing）。

---

[← 返回主目录](index.md)
