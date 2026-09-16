# 第18章 扩散模型

第 15 章介绍了生成对抗模型，它可以产生看起来合理的样本但不定义数据上的概率分布。第 16 章讨论了归一化流，它确实定义了概率分布但必须对网络施加架构约束；每一层必须是可逆的，且其雅可比行列式必须易于计算。第 17 章介绍了变分自编码器，它也有坚实的概率基础但似然的计算是不可处理的，必须用下界来近似。

本章介绍**扩散模型**（diffusion models）。与归一化流类似，它们是定义从潜变量到观测数据的非线性映射的概率模型，其中两者具有相同的维度。与变分自编码器类似，它们使用基于编码器映射到潜变量的下界来近似数据似然。然而，在扩散模型中，这个编码器是预设的；目标是学习一个解码器，它是编码过程的逆过程，可以用来产生样本。扩散模型易于训练，能够产生非常高质量的样本，其真实感超过 GAN 产生的样本。读者在阅读本章前应熟悉变分自编码器（第 17 章）。

## 18.1 概述

扩散模型由一个**编码器**（encoder）和一个**解码器**（decoder）组成。编码器接收一个数据样本 $\mathbf{x}$，将其映射到一系列中间潜变量 $\mathbf{z}_1 \ldots \mathbf{z}_T$。解码器反转这个过程；它从 $\mathbf{z}_T$ 开始，映射回 $\mathbf{z}_{T-1},\ldots,\mathbf{z}_1$，直到最终（重新）创建一个数据点 $\mathbf{x}$。在编码器和解码器中，映射都是随机的而非确定性的。

编码器是预先设定的；它逐渐将输入与白噪声样本混合（图 18.1）。经过足够多的步骤，条件分布 $q(\mathbf{z}_T|\mathbf{x})$ 和边缘分布 $q(\mathbf{z}_T)$ 都变成标准正态分布。由于这个过程是预设的，所有学习到的参数都在解码器中。

在解码器中，一系列网络被训练来学习每对相邻潜变量 $\mathbf{z}_t$ 和 $\mathbf{z}_{t-1}$ 之间的反向映射。损失函数鼓励每个网络反转相应的编码器步骤。结果是噪声从表示中被逐渐去除，直到剩下一个看起来真实的数据样本。为了生成新的数据样本 $\mathbf{x}$，我们从 $q(\mathbf{z}_T)$ 中抽取一个样本并通过解码器传递。

<div align="center">

![图 18.1](/figures/ch18/DiffusionOverview.png)

</div>

> **图 18.1** 扩散模型。编码器（前向或扩散过程）将输入 $\mathbf{x}$ 映射到一系列潜变量 $\mathbf{z}_1 \ldots \mathbf{z}_T$。这个过程是预设的，逐渐将数据与噪声混合直到只剩噪声。解码器（逆过程）是学习的，将数据通过潜变量传回，在每个阶段去除噪声。训练后，通过采样噪声向量 $\mathbf{z}_T$ 并通过解码器传递来生成新样本。

在第 18.2 节中，我们详细考虑编码器。它的性质虽不显而易见但对学习算法至关重要。在第 18.3 节中，我们讨论解码器。第 18.4 节推导训练算法，第 18.5 节将其重新表述为更实用的形式。第 18.6 节讨论实现细节，包括如何使生成以文本提示为条件。

## 18.2 编码器（前向过程）

**扩散**（diffusion）或**前向过程**（forward process）（图 18.2）将数据样本 $\mathbf{x}$ 映射到一系列与 $\mathbf{x}$ 相同大小的中间变量 $\mathbf{z}_1,\mathbf{z}_2,\ldots,\mathbf{z}_T$，按照以下规则：

$$
\begin{aligned}
\mathbf{z}_1 &= \sqrt{1-\beta_1} \cdot \mathbf{x} + \sqrt{\beta_1} \cdot \boldsymbol{\epsilon}_1 \\
\mathbf{z}_t &= \sqrt{1-\beta_t} \cdot \mathbf{z}_{t-1} + \sqrt{\beta_t} \cdot \boldsymbol{\epsilon}_t \qquad \forall\; t \in 2,\ldots,T,
\end{aligned}
\tag{18.1}
$$

其中 $\boldsymbol{\epsilon}_t$ 是从标准正态分布中抽取的噪声。第一项衰减到目前为止的数据加上任何已添加的噪声，第二项添加更多噪声。超参数 $\beta_t \in [0,1]$ 决定噪声混合的速度，统称为**噪声调度**（noise schedule）。前向过程也可以等价地写为：

$$
\begin{aligned}
q(\mathbf{z}_1|\mathbf{x}) &= \text{Norm}_{\mathbf{z}_1}\left[\sqrt{1-\beta_1}\mathbf{x}, \beta_1\mathbf{I}\right] \\
q(\mathbf{z}_t|\mathbf{z}_{t-1}) &= \text{Norm}_{\mathbf{z}_t}\left[\sqrt{1-\beta_t}\mathbf{z}_{t-1}, \beta_t\mathbf{I}\right] \qquad \forall\; t \in \{2,\ldots,T\}.
\end{aligned}
\tag{18.2}
$$

这是一个**马尔可夫链**（Markov chain），因为 $\mathbf{z}_t$ 的概率完全由紧邻的前一个变量 $\mathbf{z}_{t-1}$ 的值决定。经过足够多的步骤 $T$，原始数据的所有痕迹都被去除，$q(\mathbf{z}_T|\mathbf{x}) = q(\mathbf{z}_T)$ 变为标准正态分布。

给定输入 $\mathbf{x}$ 时，所有潜变量 $\mathbf{z}_1,\mathbf{z}_2,\ldots,\mathbf{z}_T$ 的联合分布是：

$$
q(\mathbf{z}_{1 \ldots T}|\mathbf{x}) = q(\mathbf{z}_1|\mathbf{x})\prod_{t=2}^{T}q(\mathbf{z}_t|\mathbf{z}_{t-1}).
\tag{18.3}
$$

<div align="center">

![图 18.2](/figures/ch18/DiffusionForward2.png)

</div>

> **图 18.2** 前向过程。a) 考虑一维数据 $x$，有 $T=100$ 个潜变量 $z_1,\ldots,z_{100}$ 且 $\beta=0.03$。三个 $x$ 值（灰色、青色和橙色）被初始化（顶行），传播经过 $z_1,\ldots,z_{100}$。在每一步，变量通过将其值衰减 $\sqrt{1-\beta}$ 并添加均值为零、方差为 $\beta$ 的噪声来更新（等式 18.1）。因此，三个样本带噪声地向零移动。b) 条件概率 $Pr(z_1|x)$ 和 $Pr(z_t|z_{t-1})$ 是正态分布，均值略微比当前点更接近零，方差固定为 $\beta_t$（等式 18.2）。

### 18.2.1 扩散核 $q(\mathbf{z}_t|\mathbf{x})$

为了训练解码器反转这个过程，我们需要在同一数据样本 $\mathbf{x}$ 的时间 $t$ 处使用多个样本 $\mathbf{z}_t$。然而，当 $t$ 很大时，使用等式 18.1 按顺序生成这些样本很耗时。幸运的是，存在 $q(\mathbf{z}_t|\mathbf{x})$ 的封闭形式表达式，允许我们在给定初始数据点 $\mathbf{x}$ 时直接抽取样本 $\mathbf{z}_t$，而无需计算中间变量 $\mathbf{z}_1 \ldots \mathbf{z}_{t-1}$。这被称为**扩散核**（diffusion kernel）（图 18.3）。

为了推导 $q(\mathbf{z}_t|\mathbf{x})$ 的表达式，考虑前向过程的前两步：

$$
\begin{aligned}
\mathbf{z}_1 &= \sqrt{1-\beta_1} \cdot \mathbf{x} + \sqrt{\beta_1} \cdot \boldsymbol{\epsilon}_1 \\
\mathbf{z}_2 &= \sqrt{1-\beta_2} \cdot \mathbf{z}_1 + \sqrt{\beta_2} \cdot \boldsymbol{\epsilon}_2.
\end{aligned}
\tag{18.4}
$$

将第一个等式代入第二个，得到：

$$
\begin{aligned}
\mathbf{z}_2 &= \sqrt{1-\beta_2}\left(\sqrt{1-\beta_1} \cdot \mathbf{x} + \sqrt{\beta_1} \cdot \boldsymbol{\epsilon}_1\right) + \sqrt{\beta_2} \cdot \boldsymbol{\epsilon}_2 \\
&= \sqrt{(1-\beta_2)(1-\beta_1)} \cdot \mathbf{x} + \sqrt{1-\beta_2-(1-\beta_2)(1-\beta_1)} \cdot \boldsymbol{\epsilon}_1 + \sqrt{\beta_2} \cdot \boldsymbol{\epsilon}_2.
\end{aligned}
\tag{18.5}
$$

最后两项是均值为零的独立正态分布的样本，方差分别为 $1-\beta_2-(1-\beta_2)(1-\beta_1)$ 和 $\beta_2$。这个和的均值为零，方差是各分量方差之和，因此：

$$
\mathbf{z}_2 = \sqrt{(1-\beta_2)(1-\beta_1)} \cdot \mathbf{x} + \sqrt{1-(1-\beta_2)(1-\beta_1)} \cdot \boldsymbol{\epsilon},
\tag{18.6}
$$

其中 $\boldsymbol{\epsilon}$ 也是标准正态分布的样本。

如果我们继续将这个等式代入 $\mathbf{z}_3$ 的表达式，依此类推，可以证明：

$$
\mathbf{z}_t = \sqrt{\alpha_t} \cdot \mathbf{x} + \sqrt{1-\alpha_t} \cdot \boldsymbol{\epsilon},
\tag{18.7}
$$

其中 $\alpha_t = \prod_{s=1}^{t}(1-\beta_s)$。我们可以等价地将其写成概率形式：

$$
q(\mathbf{z}_t|\mathbf{x}) = \text{Norm}_{\mathbf{z}_t}\left[\sqrt{\alpha_t} \cdot \mathbf{x}, (1-\alpha_t)\mathbf{I}\right].
\tag{18.8}
$$

对于任何起始数据点 $\mathbf{x}$，变量 $\mathbf{z}_t$ 是具有已知均值和方差的正态分布。因此，如果我们不关心通过中间变量 $\mathbf{z}_1 \ldots \mathbf{z}_{t-1}$ 的演化历史，就可以很容易地从 $q(\mathbf{z}_t|\mathbf{x})$ 生成样本。

<div align="center">

![图 18.3](/figures/ch18/DiffusionKernel.png)

</div>

> **图 18.3** 扩散核。a) 点 $x^*=2.0$ 通过等式 18.1 传播经过潜变量（五条路径以灰色显示）。扩散核 $q(z_t|x^*)$ 是在给定从 $x^*$ 出发时变量 $z_t$ 上的概率分布。它可以以封闭形式计算，是一个均值趋向零且方差随 $t$ 增大而增大的正态分布。热力图显示了每个变量的 $q(z_t|x^*)$。青色线显示均值偏离 $\pm 2$ 个标准差。b) 明确显示了 $t=20,40,80$ 时的扩散核 $q(z_t|x^*)$。实际上，扩散核允许我们在给定 $x^*$ 时采样对应于任意时间 $t$ 的潜变量 $z_t$，而无需计算中间变量 $z_1,\ldots,z_{t-1}$。当 $t$ 变得很大时，扩散核趋近于标准正态分布。

### 18.2.2 边缘分布 $q(\mathbf{z}_t)$

边缘分布 $q(\mathbf{z}_t)$ 是在给定可能的起始点 $\mathbf{x}$ 的分布和每个起始点的可能扩散路径下，观察到 $\mathbf{z}_t$ 值的概率（图 18.4）。它可以通过考虑联合分布 $q(\mathbf{x},\mathbf{z}_{1\ldots t})$ 并对除 $\mathbf{z}_t$ 以外的所有变量进行边缘化来计算：

$$
\begin{aligned}
q(\mathbf{z}_t) &= \iint q(\mathbf{z}_{1\ldots t},\mathbf{x})d\mathbf{z}_{1\ldots t-1}d\mathbf{x} \\
&= \iint q(\mathbf{z}_{1\ldots t}|\mathbf{x})Pr(\mathbf{x})d\mathbf{z}_{1\ldots t-1}d\mathbf{x},
\end{aligned}
\tag{18.9}
$$

然而，由于我们有一个"跳过"中间变量的扩散核 $q(\mathbf{z}_t|\mathbf{x})$ 的表达式，可以等价地写为：

$$
q(\mathbf{z}_t) = \int q(\mathbf{z}_t|\mathbf{x})Pr(\mathbf{x})d\mathbf{x}.
\tag{18.10}
$$

因此，如果我们反复从数据分布 $Pr(\mathbf{x})$ 采样并在每个样本上叠加扩散核 $q(\mathbf{z}_t|\mathbf{x})$，得到的就是边缘分布 $q(\mathbf{z}_t)$（图 18.4）。然而，边缘分布无法写成封闭形式，因为我们不知道原始数据分布 $Pr(\mathbf{x})$。

<div align="center">

![图 18.4](/figures/ch18/DiffusionDensity.png)

</div>

> **图 18.4** 边缘分布。a) 给定初始密度 $Pr(x)$（顶行），扩散过程在通过潜变量 $z_t$ 时逐渐模糊分布并将其移向标准正态分布。每一行热力图代表一个边缘分布 $q(z_t)$。b) 顶部图显示初始分布 $Pr(x)$。另外两个图分别显示边缘分布 $q(z_{20})$ 和 $q(z_{60})$。

### 18.2.3 条件分布 $q(\mathbf{z}_{t-1}|\mathbf{z}_t)$

我们将条件概率 $q(\mathbf{z}_t|\mathbf{z}_{t-1})$ 定义为混合过程（等式 18.2）。为了反转这个过程，我们应用贝叶斯法则：

$$
q(\mathbf{z}_{t-1}|\mathbf{z}_t) = \frac{q(\mathbf{z}_t|\mathbf{z}_{t-1})q(\mathbf{z}_{t-1})}{q(\mathbf{z}_t)}.
\tag{18.11}
$$

这是不可处理的，因为我们无法计算边缘分布 $q(\mathbf{z}_{t-1})$。

对于简单的一维示例，可以数值地评估 $q(\mathbf{z}_{t-1}|\mathbf{z}_t)$（图 18.5）。一般来说，它的形式是复杂的，但在许多情况下，它可以被正态分布很好地近似。这很重要，因为当我们构建解码器时，将使用正态分布来近似逆过程。

<div align="center">

![图 18.5](/figures/ch18/DiffusionReverse.png)

</div>

> **图 18.5** 条件分布 $q(z_{t-1}|z_t)$。a) 边缘密度 $q(z_t)$ 中高亮三个点 $z_t^*$。b) 概率 $q(z_{t-1}|z_t^*)$（青色曲线）通过贝叶斯法则计算，与 $q(z_t^*|z_{t-1})q(z_{t-1})$ 成正比。一般来说，它不是正态分布（顶部图），但通常正态分布是一个好的近似（底部两个图）。

### 18.2.4 条件扩散分布 $q(\mathbf{z}_{t-1}|\mathbf{z}_t, \mathbf{x})$

还有与编码器相关的最后一个分布需要考虑。我们注意到无法求得条件分布 $q(\mathbf{z}_{t-1}|\mathbf{z}_t)$，因为我们不知道边缘分布 $q(\mathbf{z}_{t-1})$。然而，如果我们知道起始变量 $\mathbf{x}$，那么我们就知道前一时刻的分布 $q(\mathbf{z}_{t-1}|\mathbf{x})$。这就是扩散核（图 18.3），它是正态分布的。

因此，可以以封闭形式计算条件扩散分布 $q(\mathbf{z}_{t-1}|\mathbf{z}_t,\mathbf{x})$（图 18.6）。这是当我们知道当前潜变量 $\mathbf{z}_t$ 和训练数据样本 $\mathbf{x}$ 时，$\mathbf{z}_{t-1}$ 上的分布。为了计算 $q(\mathbf{z}_{t-1}|\mathbf{z}_t,\mathbf{x})$ 的表达式，我们从贝叶斯法则出发：

$$
\begin{aligned}
q(\mathbf{z}_{t-1}|\mathbf{z}_t,\mathbf{x}) &= \frac{q(\mathbf{z}_t|\mathbf{z}_{t-1},\mathbf{x})q(\mathbf{z}_{t-1}|\mathbf{x})}{q(\mathbf{z}_t|\mathbf{x})} \\
&\propto \text{Norm}_{\mathbf{z}_t}\left[\sqrt{1-\beta_t} \cdot \mathbf{z}_{t-1}, \beta_t\mathbf{I}\right]\text{Norm}_{\mathbf{z}_{t-1}}\left[\sqrt{\alpha_{t-1}} \cdot \mathbf{x}, (1-\alpha_{t-1})\mathbf{I}\right]
\end{aligned}
\tag{18.12}
$$

其中第一行和第二行之间，我们利用了 $q(\mathbf{z}_t|\mathbf{z}_{t-1},\mathbf{x}) = q(\mathbf{z}_t|\mathbf{z}_{t-1})$ 这一事实，因为扩散过程是马尔可夫的，关于 $\mathbf{z}_t$ 的所有信息都包含在 $\mathbf{z}_{t-1}$ 中。利用高斯变量替换恒等式：

$$
\text{Norm}_{\mathbf{v}}\left[\mathbf{A}\mathbf{w},\mathbf{B}\right] \propto \text{Norm}_{\mathbf{w}}\left[(\mathbf{A}^T\mathbf{B}^{-1}\mathbf{A})^{-1}\mathbf{A}^T\mathbf{B}^{-1}\mathbf{v}, (\mathbf{A}^T\mathbf{B}^{-1}\mathbf{A})^{-1}\right],
\tag{18.13}
$$

以及两个高斯乘积的恒等式：

$$
\text{Norm}_{\mathbf{w}}[\mathbf{a},\mathbf{A}] \cdot \text{Norm}_{\mathbf{w}}[\mathbf{b},\mathbf{B}] \propto \text{Norm}_{\mathbf{w}}\left[(\mathbf{A}^{-1}+\mathbf{B}^{-1})^{-1}(\mathbf{A}^{-1}\mathbf{a}+\mathbf{B}^{-1}\mathbf{b}), (\mathbf{A}^{-1}+\mathbf{B}^{-1})^{-1}\right],
\tag{18.14}
$$

得到：

$$
q(\mathbf{z}_{t-1}|\mathbf{z}_t,\mathbf{x}) = \text{Norm}_{\mathbf{z}_{t-1}}\left[\frac{(1-\alpha_{t-1})}{1-\alpha_t}\sqrt{1-\beta_t}\mathbf{z}_t + \frac{\sqrt{\alpha_{t-1}}\beta_t}{1-\alpha_t}\mathbf{x}, \frac{\beta_t(1-\alpha_{t-1})}{1-\alpha_t}\mathbf{I}\right].
\tag{18.15}
$$

<div align="center">

![图 18.6](/figures/ch18/DiffusionReverseConditional.png)

</div>

> **图 18.6** 条件扩散分布 $q(z_{t-1}|z_t,x)$。a) 对于 $x^*=-2.1$ 的扩散核 $q(z_t|x^*)$，高亮三个点 $z_t^*$。b) 概率 $q(z_{t-1}|z_t^*,x^*)$ 通过贝叶斯法则计算，与 $q(z_t^*|z_{t-1})q(z_{t-1}|x^*)$ 成正比。这*是*正态分布的，可以以封闭形式计算。

## 18.3 解码器模型（逆过程）

当我们学习扩散模型时，我们学习一系列从潜变量 $\mathbf{z}_T$ 到 $\mathbf{z}_{T-1}$、从 $\mathbf{z}_{T-1}$ 到 $\mathbf{z}_{T-2}$ 的概率映射，依此类推，直到到达数据 $\mathbf{x}$。扩散过程的真实逆分布 $q(\mathbf{z}_{t-1}|\mathbf{z}_t)$ 是复杂的多模态分布（图 18.5），取决于数据分布 $Pr(\mathbf{x})$。我们将它们近似为正态分布：

$$
\begin{aligned}
Pr(\mathbf{z}_T) &= \text{Norm}_{\mathbf{z}_T}[\mathbf{0},\mathbf{I}] \\
Pr(\mathbf{z}_{t-1}|\mathbf{z}_t,\boldsymbol{\phi}_t) &= \text{Norm}_{\mathbf{z}_{t-1}}\left[\mathbf{f}_t[\mathbf{z}_t,\boldsymbol{\phi}_t], \sigma_t^2\mathbf{I}\right] \\
Pr(\mathbf{x}|\mathbf{z}_1,\boldsymbol{\phi}_1) &= \text{Norm}_{\mathbf{x}}\left[\mathbf{f}_1[\mathbf{z}_1,\boldsymbol{\phi}_1], \sigma_1^2\mathbf{I}\right],
\end{aligned}
\tag{18.16}
$$

其中 $\mathbf{f}_t[\mathbf{z}_t,\boldsymbol{\phi}_t]$ 是一个神经网络，计算从 $\mathbf{z}_t$ 到前一个潜变量 $\mathbf{z}_{t-1}$ 估计映射中正态分布的均值。项 $\{\sigma_t^2\}$ 是预设的。如果扩散过程中超参数 $\beta_t$ 接近零（且时间步 $T$ 很大），那么这个正态近似将是合理的。

我们使用祖先采样从 $Pr(\mathbf{x})$ 生成新样本。首先从 $Pr(\mathbf{z}_T)$ 中抽取 $\mathbf{z}_T$。然后从 $Pr(\mathbf{z}_{T-1}|\mathbf{z}_T,\boldsymbol{\phi}_T)$ 中采样 $\mathbf{z}_{T-1}$，从 $Pr(\mathbf{z}_{T-2}|\mathbf{z}_{T-1},\boldsymbol{\phi}_{T-1})$ 中采样 $\mathbf{z}_{T-2}$，依此类推直到最终从 $Pr(\mathbf{x}|\mathbf{z}_1,\boldsymbol{\phi}_1)$ 生成 $\mathbf{x}$。

## 18.4 训练

观测变量 $\mathbf{x}$ 和潜变量 $\{\mathbf{z}_t\}$ 的联合分布是：

$$
Pr(\mathbf{x},\mathbf{z}_{1\ldots T}|\boldsymbol{\phi}_{1\ldots T}) = Pr(\mathbf{x}|\mathbf{z}_1,\boldsymbol{\phi}_1)\prod_{t=2}^{T}Pr(\mathbf{z}_{t-1}|\mathbf{z}_t,\boldsymbol{\phi}_t) \cdot Pr(\mathbf{z}_T).
\tag{18.17}
$$

观测数据 $Pr(\mathbf{x}|\boldsymbol{\phi}_{1\ldots T})$ 的似然通过对潜变量边缘化获得：

$$
Pr(\mathbf{x}|\boldsymbol{\phi}_{1\ldots T}) = \int Pr(\mathbf{x},\mathbf{z}_{1\ldots T}|\boldsymbol{\phi}_{1\ldots T})d\mathbf{z}_{1\ldots T}.
\tag{18.18}
$$

为了训练模型，我们最大化训练数据 $\{\mathbf{x}_i\}$ 关于参数 $\boldsymbol{\phi}$ 的对数似然：

$$
\hat{\boldsymbol{\phi}}_{1\ldots T} = \underset{\boldsymbol{\phi}_{1\ldots T}}{\text{argmax}}\left[\sum_{i=1}^{I}\log\left[Pr(\mathbf{x}_i|\boldsymbol{\phi}_{1\ldots T})\right]\right].
\tag{18.19}
$$

我们无法直接最大化，因为等式 18.18 中的边缘化是不可处理的。因此，我们使用 Jensen 不等式来定义似然的下界，并像对 VAE 所做的那样（见第 17.3.1 节），关于参数 $\boldsymbol{\phi}_{1\ldots T}$ 优化该下界。

### 18.4.1 证据下界（ELBO）

为了推导下界，我们将对数似然乘以并除以编码器分布 $q(\mathbf{z}_{1\ldots T}|\mathbf{x})$，并应用 Jensen 不等式（见第 17.3.2 节）：

$$
\begin{aligned}
\log\left[Pr(\mathbf{x}|\boldsymbol{\phi}_{1\ldots T})\right] &= \log\left[\int Pr(\mathbf{x},\mathbf{z}_{1\ldots T}|\boldsymbol{\phi}_{1\ldots T})d\mathbf{z}_{1\ldots T}\right] \\
&= \log\left[\int q(\mathbf{z}_{1\ldots T}|\mathbf{x})\frac{Pr(\mathbf{x},\mathbf{z}_{1\ldots T}|\boldsymbol{\phi}_{1\ldots T})}{q(\mathbf{z}_{1\ldots T}|\mathbf{x})}d\mathbf{z}_{1\ldots T}\right] \\
&\geq \int q(\mathbf{z}_{1\ldots T}|\mathbf{x})\log\left[\frac{Pr(\mathbf{x},\mathbf{z}_{1\ldots T}|\boldsymbol{\phi}_{1\ldots T})}{q(\mathbf{z}_{1\ldots T}|\mathbf{x})}\right]d\mathbf{z}_{1\ldots T}.
\end{aligned}
\tag{18.20}
$$

这给出了证据下界（ELBO）：

$$
\text{ELBO}\left[\boldsymbol{\phi}_{1\ldots T}\right] = \int q(\mathbf{z}_{1\ldots T}|\mathbf{x})\log\left[\frac{Pr(\mathbf{x},\mathbf{z}_{1\ldots T}|\boldsymbol{\phi}_{1\ldots T})}{q(\mathbf{z}_{1\ldots T}|\mathbf{x})}\right]d\mathbf{z}_{1\ldots T}.
\tag{18.21}
$$

在 VAE 中，编码器 $q(\mathbf{z}|\mathbf{x})$ 近似潜变量上的后验分布以使下界紧致，解码器最大化这个下界（图 17.10）。在扩散模型中，解码器必须做所有工作，因为编码器没有参数。它通过 (i) 改变自身参数使静态编码器确实近似后验 $Pr(\mathbf{z}_{1\ldots T}|\mathbf{x},\boldsymbol{\phi}_{1\ldots T})$ 以及 (ii) 关于自身参数优化该下界来使下界更紧致。

### 18.4.2 化简 ELBO

我们现在将 ELBO 中的对数项操纵为最终优化的形式。首先分别代入等式 18.17 和 18.3 中分子和分母的定义：

$$
\log\left[\frac{Pr(\mathbf{x},\mathbf{z}_{1\ldots T}|\boldsymbol{\phi}_{1\ldots T})}{q(\mathbf{z}_{1\ldots T}|\mathbf{x})}\right] = \log\left[\frac{Pr(\mathbf{x}|\mathbf{z}_1,\boldsymbol{\phi}_1)\prod_{t=2}^{T}Pr(\mathbf{z}_{t-1}|\mathbf{z}_t,\boldsymbol{\phi}_t) \cdot Pr(\mathbf{z}_T)}{q(\mathbf{z}_1|\mathbf{x})\prod_{t=2}^{T}q(\mathbf{z}_t|\mathbf{z}_{t-1})}\right]
\tag{18.22}
$$

经过展开分母并利用比值 $q(\mathbf{z}_{t-1}|\mathbf{x})/q(\mathbf{z}_t|\mathbf{x})$ 的消去，化简后的 ELBO 为：

$$
\begin{aligned}
\text{ELBO}\left[\boldsymbol{\phi}_{1\ldots T}\right] &= \int q(\mathbf{z}_{1\ldots T}|\mathbf{x})\log\left[\frac{Pr(\mathbf{x},\mathbf{z}_{1\ldots T}|\boldsymbol{\phi}_{1\ldots T})}{q(\mathbf{z}_{1\ldots T}|\mathbf{x})}\right]d\mathbf{z}_{1\ldots T} \\
&\approx \int q(\mathbf{z}_{1\ldots T}|\mathbf{x})\left(\log\left[Pr(\mathbf{x}|\mathbf{z}_1,\boldsymbol{\phi}_1)\right] + \sum_{t=2}^{T}\log\left[\frac{Pr(\mathbf{z}_{t-1}|\mathbf{z}_t,\boldsymbol{\phi}_t)}{q(\mathbf{z}_{t-1}|\mathbf{z}_t,\mathbf{x})}\right]\right)d\mathbf{z}_{1\ldots T} \\
&= \mathbb{E}_{q(\mathbf{z}_1|\mathbf{x})}\left[\log\left[Pr(\mathbf{x}|\mathbf{z}_1,\boldsymbol{\phi}_1)\right]\right] - \sum_{t=2}^{T}\mathbb{E}_{q(\mathbf{z}_{t,t+1}|\mathbf{x})}\left[\text{D}_{KL}\left[q(\mathbf{z}_{t-1}|\mathbf{z}_t,\mathbf{x})\,\|\,Pr(\mathbf{z}_{t-1}|\mathbf{z}_t,\boldsymbol{\phi}_t)\right]\right],
\end{aligned}
\tag{18.25}
$$

其中我们对 $q(\mathbf{z}_{1\ldots T}|\mathbf{x})$ 中的无关变量进行了边缘化，并使用了 KL 散度的定义。

### 18.4.3 分析 ELBO

ELBO 中的第一个概率项在等式 18.16 中定义：

$$
Pr(\mathbf{x}|\mathbf{z}_1,\boldsymbol{\phi}_1) = \text{Norm}_{\mathbf{x}}\left[\mathbf{f}_1[\mathbf{z}_1,\boldsymbol{\phi}_1], \sigma_1^2\mathbf{I}\right],
\tag{18.26}
$$

等价于 VAE 中的重构项。如果模型预测与观测数据匹配，ELBO 会更大。与 VAE 一样，我们使用蒙特卡罗估计来近似该量的对数的期望（见等式 17.22-17.23），用 $q(\mathbf{z}_1|\mathbf{x})$ 的样本来估计。

ELBO 中的 KL 散度项衡量 $Pr(\mathbf{z}_{t-1}|\mathbf{z}_t,\boldsymbol{\phi}_t)$ 和 $q(\mathbf{z}_{t-1}|\mathbf{z}_t,\mathbf{x})$ 之间的距离，它们分别在等式 18.16 和 18.15 中定义：

$$
\begin{aligned}
Pr(\mathbf{z}_{t-1}|\mathbf{z}_t,\boldsymbol{\phi}_t) &= \text{Norm}_{\mathbf{z}_{t-1}}\left[\mathbf{f}_t[\mathbf{z}_t,\boldsymbol{\phi}_t], \sigma_t^2\mathbf{I}\right] \\
q(\mathbf{z}_{t-1}|\mathbf{z}_t,\mathbf{x}) &= \text{Norm}_{\mathbf{z}_{t-1}}\left[\frac{(1-\alpha_{t-1})}{1-\alpha_t}\sqrt{1-\beta_t}\mathbf{z}_t + \frac{\sqrt{\alpha_{t-1}}\beta_t}{1-\alpha_t}\mathbf{x}, \frac{\beta_t(1-\alpha_{t-1})}{1-\alpha_t}\mathbf{I}\right].
\end{aligned}
\tag{18.27}
$$

两个正态分布之间的 KL 散度有封闭形式表达式。而且，该表达式中的许多项不依赖于 $\boldsymbol{\phi}$，因此简化为均值之间的差的平方加上一个常数 $C$：

$$
\text{D}_{KL}\left[q(\mathbf{z}_{t-1}|\mathbf{z}_t,\mathbf{x})\,\|\,Pr(\mathbf{z}_{t-1}|\mathbf{z}_t,\boldsymbol{\phi}_t)\right] = \frac{1}{2\sigma_t^2}\left\|\frac{(1-\alpha_{t-1})}{1-\alpha_t}\sqrt{1-\beta_t}\mathbf{z}_t + \frac{\sqrt{\alpha_{t-1}}\beta_t}{1-\alpha_t}\mathbf{x} - \mathbf{f}_t[\mathbf{z}_t,\boldsymbol{\phi}_t]\right\|^2 + C.
\tag{18.28}
$$

<div align="center">

![图 18.7](/figures/ch18/DiffusionPredict.png)

</div>

> **图 18.7** 拟合的模型。a) 可以通过从标准正态分布 $Pr(z_T)$（底行）采样生成单个样本，然后从 $Pr(z_{T-1}|z_T) = \text{Norm}_{z_{T-1}}[\mathbf{f}_T[z_T,\boldsymbol{\phi}_T],\sigma_T^2\mathbf{I}]$ 采样 $z_{T-1}$，依此类推直到到达 $x$（五条路径）。估计的边缘密度（热力图）与图 18.4 中的真实边缘密度相似。b) 估计的分布 $Pr(z_{t-1}|z_t)$（棕色曲线）是图 18.5 中扩散模型真实后验 $q(z_{t-1}|z_t)$（青色曲线）的合理近似。

### 18.4.4 扩散损失函数

为了拟合模型，我们关于参数 $\boldsymbol{\phi}_{1\ldots T}$ 最大化 ELBO。我们乘以负一并用样本近似期望，将其重新表述为最小化，得到损失函数：

$$
\begin{aligned}
L[\boldsymbol{\phi}_{1\ldots T}] &= \sum_{i=1}^{I}\left(-\log\left[\text{Norm}_{\mathbf{x}_i}\left[\mathbf{f}_1[\mathbf{z}_{i1},\boldsymbol{\phi}_1], \sigma_1^2\mathbf{I}\right]\right]\right. \\
&\left.+ \sum_{t=2}^{T}\frac{1}{2\sigma_t^2}\left\|\underbrace{\frac{1-\alpha_{t-1}}{1-\alpha_t}\sqrt{1-\beta_t}\mathbf{z}_{it} + \frac{\sqrt{\alpha_{t-1}}\beta_t}{1-\alpha_t}\mathbf{x}_i}_{\text{目标：}q(\mathbf{z}_{t-1}|\mathbf{z}_t,\mathbf{x})\text{ 的均值}} - \underbrace{\mathbf{f}_t[\mathbf{z}_{it},\boldsymbol{\phi}_t]}_{\text{预测 }\mathbf{z}_{t-1}}\right\|^2\right),
\end{aligned}
\tag{18.29}
$$

其中 $\mathbf{x}_i$ 是第 $i$ 个数据点，$\mathbf{z}_{it}$ 是扩散步骤 $t$ 处的关联潜变量。

### 18.4.5 训练过程

这个损失函数可以用来为每个扩散时间步训练一个网络。它最小化隐变量在前一时间步的估计 $\mathbf{f}_t[\mathbf{z}_t,\boldsymbol{\phi}_t]$ 与给定去噪后真实数据 $\mathbf{x}$ 时最可能值之间的差异。

<div align="center">

![图 18.8](/figures/ch18/DiffusionResultsFinal.png)

</div>

> **图 18.8** 拟合模型的结果。青色和棕色曲线分别是原始和估计密度，分别对应图 18.4 和 18.7 的顶行。垂直条是从模型中分箱的样本，通过从 $Pr(\mathbf{z}_T)$ 采样并传播回 $\mathbf{z}_{T-1},\mathbf{z}_{T-2},\ldots$ 生成，如图 18.7 中的五条路径所示。

## 18.5 损失函数的重参数化

虽然等式 18.29 中的损失函数可以使用，但已发现扩散模型在不同的参数化下表现更好；损失函数被修改为让模型预测与原始数据样本混合以产生当前变量的噪声。第 18.5.1 节讨论目标（等式 18.29 第二行的前两项）的重参数化，第 18.5.2 节讨论网络（等式 18.29 第二行的最后一项）的重参数化。

### 18.5.1 目标的重参数化

原始扩散更新由下式给出：

$$
\mathbf{z}_t = \sqrt{\alpha_t} \cdot \mathbf{x} + \sqrt{1-\alpha_t} \cdot \boldsymbol{\epsilon}.
\tag{18.30}
$$

由此得出等式 18.28 中的数据项 $\mathbf{x}$ 可以表示为扩散后的图像减去添加的噪声：

$$
\mathbf{x} = \frac{1}{\sqrt{\alpha_t}} \cdot \mathbf{z}_t - \frac{\sqrt{1-\alpha_t}}{\sqrt{\alpha_t}} \cdot \boldsymbol{\epsilon}.
\tag{18.31}
$$

将其代入等式 18.29 的目标项并化简，得到：

$$
\frac{(1-\alpha_{t-1})}{1-\alpha_t}\sqrt{1-\beta_t}\mathbf{z}_t + \frac{\sqrt{\alpha_{t-1}}\beta_t}{1-\alpha_t}\mathbf{x} = \frac{1}{\sqrt{1-\beta_t}}\mathbf{z}_t - \frac{\beta_t}{\sqrt{1-\alpha_t}\sqrt{1-\beta_t}}\boldsymbol{\epsilon},
\tag{18.33}
$$

将其代回损失函数（等式 18.29），得到：

$$
\begin{aligned}
L[\boldsymbol{\phi}_{1\ldots T}] &= \sum_{i=1}^{I}\left(-\log\left[\text{Norm}_{\mathbf{x}_i}\left[\mathbf{f}_1[\mathbf{z}_{i1},\boldsymbol{\phi}_1], \sigma_1^2\mathbf{I}\right]\right]\right. \\
&\left.+ \sum_{t=2}^{T}\frac{1}{2\sigma_t^2}\left\|\left(\frac{1}{\sqrt{1-\beta_t}}\mathbf{z}_{it} - \frac{\beta_t}{\sqrt{1-\alpha_t}\sqrt{1-\beta_t}}\boldsymbol{\epsilon}_{it}\right) - \mathbf{f}_t[\mathbf{z}_{it},\boldsymbol{\phi}_t]\right\|^2\right).
\end{aligned}
\tag{18.34}
$$

### 18.5.2 网络的重参数化

现在我们用一个预测与 $\mathbf{x}$ 混合以创建 $\mathbf{z}_t$ 的噪声 $\boldsymbol{\epsilon}$ 的新模型 $\hat{\boldsymbol{\epsilon}} = \mathbf{g}_t[\mathbf{z}_t,\boldsymbol{\phi}_t]$ 替换模型 $\hat{\mathbf{z}}_{t-1} = \mathbf{f}_t[\mathbf{z}_t,\boldsymbol{\phi}_t]$：

$$
\mathbf{f}_t[\mathbf{z}_t,\boldsymbol{\phi}_t] = \frac{1}{\sqrt{1-\beta_t}}\mathbf{z}_t - \frac{\beta_t}{\sqrt{1-\alpha_t}\sqrt{1-\beta_t}}\mathbf{g}_t[\mathbf{z}_t,\boldsymbol{\phi}_t].
\tag{18.35}
$$

将新模型代入等式 18.34，经过化简，忽略各项的缩放因子（它们在每个时间步可能不同），得到更简洁的表述：

$$
\begin{aligned}
L[\boldsymbol{\phi}_{1\ldots T}] &= \sum_{i=1}^{I}\sum_{t=1}^{T}\left\|\mathbf{g}_t[\mathbf{z}_{it},\boldsymbol{\phi}_t] - \boldsymbol{\epsilon}_{it}\right\|^2 \\
&= \sum_{i=1}^{I}\sum_{t=1}^{T}\left\|\mathbf{g}_t\left[\sqrt{\alpha_t} \cdot \mathbf{x}_i + \sqrt{1-\alpha_t} \cdot \boldsymbol{\epsilon}_{it}, \boldsymbol{\phi}_t\right] - \boldsymbol{\epsilon}_{it}\right\|^2,
\end{aligned}
\tag{18.40}
$$

其中第二行使用扩散核（等式 18.30）重写了 $\mathbf{z}_t$。

## 18.6 实现

这导出了直观的训练模型（算法 18.1）和采样（算法 18.2）算法。训练算法的优点是 (i) 实现简单，(ii) 自然地增强数据集；我们可以在每个时间步以不同的噪声实例化 $\boldsymbol{\epsilon}$ 重复使用每个原始数据点 $\mathbf{x}_i$ 任意多次。采样算法的缺点是它需要串行处理多个神经网络 $\mathbf{g}_t[\mathbf{z}_t,\boldsymbol{\phi}_t]$，因此耗时较长。

**算法 18.1：扩散模型训练**
- 输入：训练数据 $\mathbf{x}$
- 输出：模型参数 $\boldsymbol{\phi}_t$
- 重复：
  - 对批次中的每个训练样本 $i$：
    - 采样随机时间步 $t \sim \text{Uniform}[1,\ldots T]$
    - 采样噪声 $\boldsymbol{\epsilon} \sim \text{Norm}[\mathbf{0},\mathbf{I}]$
    - 计算个体损失 $\ell_i = \|\mathbf{g}_t[\sqrt{\alpha_t}\mathbf{x}_i + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}, \boldsymbol{\phi}_t] - \boldsymbol{\epsilon}\|^2$
  - 累积批次损失并进行梯度步
- 直到收敛

**算法 18.2：采样**
- 输入：模型 $\mathbf{g}_t[\bullet,\boldsymbol{\phi}_t]$
- 输出：样本 $\mathbf{x}$
- $\mathbf{z}_T \sim \text{Norm}_{\mathbf{z}}[\mathbf{0},\mathbf{I}]$（采样最后的潜变量）
- 对 $t = T \ldots 2$：
  - $\hat{\mathbf{z}}_{t-1} = \frac{1}{\sqrt{1-\beta_t}}\mathbf{z}_t - \frac{\beta_t}{\sqrt{1-\alpha_t}\sqrt{1-\beta_t}}\mathbf{g}_t[\mathbf{z}_t,\boldsymbol{\phi}_t]$（预测前一个潜变量）
  - $\boldsymbol{\epsilon} \sim \text{Norm}_{\boldsymbol{\epsilon}}[\mathbf{0},\mathbf{I}]$（抽取新的噪声向量）
  - $\mathbf{z}_{t-1} = \hat{\mathbf{z}}_{t-1} + \sigma_t\boldsymbol{\epsilon}$（向前一个潜变量添加噪声）
- $\mathbf{x} = \frac{1}{\sqrt{1-\beta_1}}\mathbf{z}_1 - \frac{\beta_1}{\sqrt{1-\alpha_1}\sqrt{1-\beta_1}}\mathbf{g}_1[\mathbf{z}_1,\boldsymbol{\phi}_1]$（从 $\mathbf{z}_1$ 生成样本，不加噪声）

### 18.6.1 应用于图像

扩散模型在建模图像数据方面非常成功。这里，我们需要构建能够接受一个有噪图像并预测在每一步添加了什么噪声的模型。对于这种图像到图像映射，U-Net（图 11.10）是显而易见的架构选择。然而，可能有非常多的扩散步骤，训练和存储多个 U-Net 效率低下。解决方案是训练一个单一的 U-Net，它还接收一个代表时间步的预设向量作为输入（图 18.9）。实际上，这个向量被调整大小以匹配 U-Net 每个阶段的通道数，用于偏移和/或缩放每个空间位置的表示。

需要大量时间步，因为当超参数 $\beta_t$ 接近零时，条件概率 $q(\mathbf{z}_{t-1}|\mathbf{z}_t)$ 更接近正态分布，与解码器分布 $Pr(\mathbf{z}_{t-1}|\mathbf{z}_t,\boldsymbol{\phi}_t)$ 的形式匹配。然而，这使得采样变慢。我们可能需要运行 $T=1000$ 步的 U-Net 模型才能生成好的图像。

<div align="center">

![图 18.9](/figures/ch18/DiffusionUNet.png)

</div>

> **图 18.9** 用于图像的扩散模型中的 U-Net。网络旨在预测添加到图像中的噪声。它由一个降低尺度并增加通道数的编码器和一个增加尺度并减少通道数的解码器组成。编码器表示与解码器中的对应部分连接。相邻表示之间的连接由残差块和周期性全局自注意力组成，其中每个空间位置与其他每个空间位置交互。所有时间步使用单一网络，通过将正弦时间嵌入（图 12.5）通过一个浅层神经网络传递并将结果添加到 U-Net 每个阶段每个空间位置的通道上。

### 18.6.2 提高生成速度

损失函数（等式 18.40）要求扩散核具有形式 $q(\mathbf{z}_t|\mathbf{x}) = \text{Norm}[\sqrt{\alpha_t}\mathbf{x}, \sqrt{1-\alpha_t} \cdot \mathbf{I}]$。对于*任何*与此关系兼容的前向过程，相同的损失函数都有效，且存在一族这样的兼容过程。这些过程都通过相同的损失函数来优化，但对前向过程有不同的规则以及在逆过程中如何使用估计的噪声 $\mathbf{g}[\mathbf{z}_t,\boldsymbol{\phi}_t]$ 从 $\mathbf{z}_t$ 预测 $\mathbf{z}_{t-1}$ 的不同规则（图 18.10）。

其中包括**去噪扩散隐式模型**（denoising diffusion implicit models），它在从 $\mathbf{x}$ 到 $\mathbf{z}_1$ 的第一步之后不再是随机的，以及**加速采样**（accelerated sampling）模型，其前向过程仅在时间步的子序列上定义。这允许一个跳过时间步的逆过程，从而使采样更加高效；用 50 个时间步就可以创建好的样本，当前向过程不再是随机时。这比以前快得多，但仍比大多数其他生成模型慢。

<div align="center">

![图 18.10](/figures/ch18/DiffusionImplicit.png)

</div>

> **图 18.10** 与同一模型兼容的不同扩散过程。a) 重参数化模型的五个采样轨迹。b) 重参数化模型生成的样本直方图。c-d) 去噪扩散隐式模型（DDIM），它是确定性的，在每一步不添加噪声。e-f) 加速扩散模型，跳过推理步骤以提高采样速度。

### 18.6.3 条件生成

如果数据具有关联标签 $c$，可以利用它们来控制生成。有时这可以改善 GAN 中的生成结果，我们可能期望扩散模型也是如此；如果你知道图像包含什么，去噪更容易。扩散模型中条件合成的一种方法是**分类器引导**（classifier guidance）。这修改了从 $\mathbf{z}_t$ 到 $\mathbf{z}_{t-1}$ 的去噪更新以考虑类别信息 $c$。实际上，这意味着在算法 18.2 的最终更新步骤中添加一个额外项：

$$
\mathbf{z}_{t-1} = \hat{\mathbf{z}}_{t-1} + \sigma_t^2\frac{\partial \log[Pr(c|\mathbf{z}_t)]}{\partial \mathbf{z}_t} + \sigma_t\boldsymbol{\epsilon}.
\tag{18.41}
$$

新项取决于基于潜变量 $\mathbf{z}_t$ 的分类器 $Pr(c|\mathbf{z}_t)$ 的梯度。像 U-Net 一样，它通常在所有时间步之间共享并以时间步作为输入。从 $\mathbf{z}_t$ 到 $\mathbf{z}_{t-1}$ 的更新现在使类别 $c$ 更可能。

**无分类器引导**（classifier-free guidance）避免了学习单独的分类器 $Pr(c|\mathbf{z}_t)$，而是将类别信息直接整合到主模型 $\mathbf{g}_t[\mathbf{z}_t,\boldsymbol{\phi}_t,c]$ 中。实际上，这通常采用将基于 $c$ 的嵌入以与添加时间步类似的方式添加到 U-Net 各层的形式（见图 18.9）。该模型在训练期间通过随机丢弃类别信息来联合训练条件和无条件目标。因此，它可以在测试时生成无条件或条件数据样本，或者两者的任意加权组合。这带来了一个令人惊讶的优势；如果条件信息被过度加权，模型倾向于产生非常高质量但略为刻板的样本。这在某种程度上类似于 GAN 中使用截断的做法（图 15.10）。

<div align="center">

![图 18.11](/figures/ch18/DiffusionCascade_C.png)

</div>

> **图 18.11** 基于文本提示的级联条件生成。a) 由一系列 U-Net 组成的扩散模型用于生成 $64\times64$ 图像。b) 该生成以语言模型计算的句子嵌入为条件。c) 更高分辨率的 $256\times256$ 图像在较小图像*和*文本编码上生成和调节。d) 重复此过程以创建 $1024\times1024$ 图像。e) 最终图像序列。改编自 Saharia et al. (2022b)。

### 18.6.4 提高生成质量

与其他生成模型一样，最高质量的结果来自对基本模型应用一系列技巧和扩展。首先，发现同时估计逆过程的方差 $\sigma_t^2$ 和均值（即图 18.7 中棕色正态分布的宽度）也有帮助。这在用更少步骤采样时尤其改善结果。其次，可以修改前向过程中的噪声调度使得 $\beta_t$ 在每一步变化，这也可以改善结果。

第三，为了生成高分辨率图像，使用扩散模型的级联。首先创建一个低分辨率图像（可能由类别信息引导）。后续的扩散模型生成逐步更高分辨率的图像。它们通过将低分辨率图像调整大小并附加到组成 U-Net 的各层以及任何其他类别信息来以低分辨率图像为条件（图 18.11）。

结合所有这些技术，可以生成非常高质量的图像。图 18.12 展示了从以 ImageNet 类别为条件的模型生成的图像样本。该模型能够学习生成如此多样化的类别，尤其令人印象深刻。图 18.13 展示了从训练为以语言模型编码的文本标题为条件的模型生成的图像，这些编码以与时间步相同的方式插入模型（图 18.9 和 18.11）。这产生了与标题一致的非常逼真的图像。由于扩散模型本质上是随机的，可以生成以同一标题为条件的多个不同图像。

<div align="center">

![图 18.12](/figures/ch18/DiffusionConditional_C.png)

</div>

> **图 18.12** 使用分类器引导的条件生成。以不同 ImageNet 类别为条件的图像样本。同一模型可以产生高度多样化的图像类别的高质量样本。改编自 Dhariwal & Nichol (2021)。

<div align="center">

![图 18.13](/figures/ch18/DiffusionImagen_C.png)

</div>

> **图 18.13** 使用文本提示的条件生成。从级联生成框架合成的图像，以大型语言模型编码的文本提示为条件。随机模型可以产生许多与提示兼容的不同图像。模型可以计数物体并将文本整合到图像中。改编自 Saharia et al. (2022b)。

## 18.7 总结

扩散模型通过反复将当前表示与随机噪声混合，将数据样本映射到一系列潜变量。经过足够多的步骤，表示变得与白噪声不可区分。由于这些步骤很小，每一步的逆去噪过程可以用正态分布近似并由深度学习模型预测。损失函数基于证据下界（ELBO），最终导出简单的最小二乘形式。

对于图像生成，每个去噪步骤使用 U-Net 实现，因此与其他生成模型相比采样较慢。为了提高生成速度，可以将扩散模型改为确定性形式，此时用更少步骤采样也能获得良好效果。已经提出了多种方法来在类别信息、图像和文本信息上进行条件生成。结合这些方法产生了令人印象深刻的文本到图像合成结果。
