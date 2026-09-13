# 第16章 归一化流

第 15 章介绍了生成对抗网络（GAN）。这些是将潜变量通过深度网络传递以创建新样本的生成模型。GAN 使用样本应与真实数据不可区分的原则进行训练。然而，它们不在数据样本上定义分布。因此，评估新样本属于同一数据集的概率并不直接。

在本章中，我们描述**归一化流**（normalizing flows）。它们通过使用深度网络将简单分布变换为更复杂的分布来学习概率模型。归一化流既可以从该分布中采样，也可以评估新样本的概率。然而，它们需要专门的架构：每一层都必须是**可逆的**（invertible）。换言之，它必须能够在两个方向上变换数据。

## 16.1 一维示例

归一化流是概率生成模型：它们将概率分布拟合到训练数据（图 14.2b）。考虑对一维分布 $Pr(x)$ 建模。归一化流从潜变量 $z$ 上的一个简单可处理的**基础分布**（base distribution）$Pr(z)$ 开始，并应用函数 $x = \text{f}[z, \boldsymbol{\phi}]$，其中参数 $\boldsymbol{\phi}$ 被选择使得 $Pr(x)$ 具有期望的分布（图 16.1）。生成新样本 $x^*$ 很容易；我们从基础密度中抽取 $z^*$ 并将其通过函数，得到 $x^* = \text{f}[z^*, \boldsymbol{\phi}]$。

> **图 16.1** 变换概率分布。a) 基础密度是定义在潜变量 $z$ 上的标准正态分布。b) 该变量通过函数 $x = \text{f}[z, \boldsymbol{\phi}]$ 变换为新变量 $x$，它 c) 具有新的分布。要从该模型中采样，我们从基础密度中抽取 $z$ 的值（(a) 中的绿色和棕色箭头展示了两个示例）。我们将其通过 (b) 中的函数（如虚线箭头所示）来生成 $x$ 的值，在 (c) 中以箭头表示。

> **图 16.2** 变换分布。基础密度（青色，底部）通过函数（蓝色曲线，右上）变换以创建模型密度（橙色，左侧）。考虑将基础密度分成等间隔的区间（灰色竖线）。相邻线之间的概率质量在变换后必须保持不变。青色阴影区域通过函数中梯度大于一的部分，因此该区域被拉伸。因此，橙色阴影区域的高度必须更低，以保持与青色阴影区域相同的面积。在其他地方（例如 $z = -2$），梯度小于一，模型密度相对于基础密度增加。

### 16.1.1 度量概率

度量数据点 $x$ 的概率更具挑战性。考虑对具有已知密度 $Pr(z)$ 的随机变量 $z$ 应用函数 $\text{f}[z, \boldsymbol{\phi}]$。概率密度在函数拉伸的区域会降低，在函数压缩的区域会增加，使得新分布下的面积保持为一。函数 $\text{f}[z, \boldsymbol{\phi}]$ 拉伸或压缩其输入的程度取决于其梯度的大小。如果输入的微小变化导致输出的更大变化，它就拉伸了函数。如果输入的微小变化导致输出的更小变化，它就压缩了函数（图 16.2）。

更精确地说，变换后分布下数据 $x$ 的概率为：

$$
Pr(x|\boldsymbol{\phi}) = \left|\frac{\partial \text{f}[z, \boldsymbol{\phi}]}{\partial z}\right|^{-1} \cdot Pr(z),
\tag{16.1}
$$

其中 $z = \text{f}^{-1}[x, \boldsymbol{\phi}]$ 是产生 $x$ 的潜变量。项 $Pr(z)$ 是该潜变量在基础密度下的原始概率。这根据函数导数的大小进行调节。如果导数大于一，概率降低。如果导数小于一，概率增加。

> **图 16.3** 逆映射（归一化方向）。如果函数是可逆的，则可以将模型密度变换回原始基础密度。模型密度下点 $x$ 的概率部分取决于基础密度下等价点 $z$ 的概率（见公式 16.1）。

### 16.1.2 正向和逆向映射

要从分布中采样，我们需要正向映射 $x = \text{f}[z, \boldsymbol{\phi}]$，但要度量似然，我们需要计算逆映射 $z = \text{f}^{-1}[x, \boldsymbol{\phi}]$。因此，我们需要巧妙地选择 $\text{f}[z, \boldsymbol{\phi}]$ 使其是**可逆的**。

正向映射有时被称为**生成方向**（generative direction）。基础密度通常选择为标准正态分布。因此，逆映射被称为**归一化方向**（normalizing direction），因为它将 $x$ 上的复杂分布转换为 $z$ 上的正态分布（图 16.3）。

### 16.1.3 学习

为了学习分布，我们找到使训练数据 $\{x_i\}_{i=1}^{I}$ 的似然最大化的参数 $\boldsymbol{\phi}$，或等价地最小化负对数似然：

$$
\begin{aligned}
\hat{\boldsymbol{\phi}} &= \underset{\boldsymbol{\phi}}{\text{argmax}}\left[\prod_{i=1}^{I}Pr(x_i|\boldsymbol{\phi})\right] \\
&= \underset{\boldsymbol{\phi}}{\text{argmin}}\left[\sum_{i=1}^{I}-\log\Big[Pr(x_i|\boldsymbol{\phi})\Big]\right] \\
&= \underset{\boldsymbol{\phi}}{\text{argmin}}\left[\sum_{i=1}^{I}\log\left[\left|\frac{\partial \text{f}[z_i, \boldsymbol{\phi}]}{\partial z_i}\right|\right] - \log\Big[Pr(z_i)\Big]\right],
\end{aligned}
\tag{16.2}
$$

其中我们假设数据独立同分布，并在第三行使用了公式 16.1 的似然定义。

## 16.2 一般情形

上一节发展了一个简单的一维示例，通过变换较简单的基础密度 $Pr(z)$ 来建模概率分布 $Pr(x)$。我们现在将其扩展到多变量分布 $Pr(\mathbf{x})$ 和 $Pr(\mathbf{z})$，并添加由深度神经网络定义变换的复杂性。

考虑对具有基础密度 $Pr(\mathbf{z})$ 的随机变量 $\mathbf{z} \in \mathbb{R}^D$ 应用函数 $\mathbf{x} = \text{f}[\mathbf{z}, \boldsymbol{\phi}]$，其中 $\text{f}[\mathbf{z}, \boldsymbol{\phi}]$ 是一个深度网络。结果变量 $\mathbf{x} \in \mathbb{R}^D$ 具有新的分布。可以通过 (i) 从基础密度中抽取样本 $\mathbf{z}^*$ 并 (ii) 将其通过神经网络使得 $\mathbf{x}^* = \text{f}[\mathbf{z}^*, \boldsymbol{\phi}]$ 来从该分布中采样。

类比公式 16.1，该分布下样本的似然为：

$$
Pr(\mathbf{x}|\boldsymbol{\phi}) = \left|\frac{\partial \text{f}[\mathbf{z}, \boldsymbol{\phi}]}{\partial \mathbf{z}}\right|^{-1} \cdot Pr(\mathbf{z}),
\tag{16.3}
$$

其中 $\mathbf{z} = \text{f}^{-1}[\mathbf{x}, \boldsymbol{\phi}]$ 是产生 $\mathbf{x}$ 的潜变量。第一项是 $D \times D$ 的 **Jacobian 矩阵** $\partial \text{f}[\mathbf{z}, \boldsymbol{\phi}]/\partial \mathbf{z}$（其中位置 $(i, j)$ 的元素为 $\partial \text{f}_j[\mathbf{z}, \boldsymbol{\phi}]/\partial z_i$）的**行列式**（determinant）的逆。正如一维情形中绝对导数度量函数在某点处面积的变化一样，绝对行列式度量多变量函数在某点处体积的变化。第二项是潜变量在基础密度下的概率。

### 16.2.1 使用深度神经网络的正向映射

在实践中，正向映射 $\text{f}[\mathbf{z}, \boldsymbol{\phi}]$ 通常由一系列层 $\text{f}_k[\bullet, \boldsymbol{\phi}_k]$（参数为 $\boldsymbol{\phi}_k$）组成的神经网络定义：

$$
\mathbf{x} = \text{f}[\mathbf{z}, \boldsymbol{\phi}] = \text{f}_K\Big[\text{f}_{K-1}\Big[\ldots \text{f}_2\big[\text{f}_1[\mathbf{z}, \boldsymbol{\phi}_1], \boldsymbol{\phi}_2\big], \ldots \boldsymbol{\phi}_{K-1}\Big], \boldsymbol{\phi}_K\Big].
\tag{16.4}
$$

逆映射（归一化方向）由每一层的逆 $\text{f}_k^{-1}[\bullet, \boldsymbol{\phi}_k]$ 以相反顺序组合定义：

$$
\mathbf{z} = \text{f}^{-1}[\mathbf{x}, \boldsymbol{\phi}] = \text{f}_1^{-1}\Big[\text{f}_2^{-1}\Big[\ldots \text{f}_{K-1}^{-1}\big[\text{f}_K^{-1}[\mathbf{x}, \boldsymbol{\phi}_K], \boldsymbol{\phi}_{K-1}\big], \ldots \boldsymbol{\phi}_2\Big], \boldsymbol{\phi}_1\Big].
\tag{16.5}
$$

基础密度 $Pr(\mathbf{z})$ 通常定义为均值为零、协方差为单位矩阵的多元标准正态分布。因此，每个后续逆层的效果是逐步将数据密度移动或"流动"到该正态分布（图 16.4）。这就是"归一化流"名称的由来。

> **图 16.4** 深度神经网络的正向和逆向映射。基础密度（左）通过网络层 $\text{f}_1[\bullet, \boldsymbol{\phi}_1], \text{f}_2[\bullet, \boldsymbol{\phi}_2], \ldots$ 逐步变换以创建模型密度。每一层都是可逆的，我们可以等价地将各层的逆视为逐步变换（或"流动"）模型密度回到基础密度。

正向映射的 Jacobian 可以表示为：

$$
\frac{\partial \text{f}[\mathbf{z}, \boldsymbol{\phi}]}{\partial \mathbf{z}} = \frac{\partial \text{f}_1[\mathbf{z}, \boldsymbol{\phi}_1]}{\partial \mathbf{z}} \cdot \frac{\partial \text{f}_2[\text{f}_1, \boldsymbol{\phi}_2]}{\partial \text{f}_1} \cdots \frac{\partial \text{f}_{K-1}[\text{f}_{K-2}, \boldsymbol{\phi}_{K-1}]}{\partial \text{f}_{K-2}} \cdots \frac{\partial \text{f}_K[\text{f}_{K-1}, \boldsymbol{\phi}_K]}{\partial \text{f}_{K-1}},
\tag{16.6}
$$

该 Jacobian 的绝对行列式可以通过取各层绝对行列式的乘积来计算：

$$
\left|\frac{\partial \text{f}[\mathbf{z}, \boldsymbol{\phi}]}{\partial \mathbf{z}}\right| = \left|\frac{\partial \text{f}_1[\mathbf{z}, \boldsymbol{\phi}_1]}{\partial \mathbf{z}}\right| \cdot \left|\frac{\partial \text{f}_2[\text{f}_1, \boldsymbol{\phi}_2]}{\partial \text{f}_1}\right| \cdots \left|\frac{\partial \text{f}_{K-1}[\text{f}_{K-2}, \boldsymbol{\phi}_{K-1}]}{\partial \text{f}_{K-2}}\right| \cdots \left|\frac{\partial \text{f}_K[\text{f}_{K-1}, \boldsymbol{\phi}_K]}{\partial \text{f}_{K-1}}\right|.
\tag{16.7}
$$

逆映射的 Jacobian 的绝对行列式通过对公式 16.5 应用相同的规则求得。它是正向映射中绝对行列式的倒数。

我们使用负对数似然准则对 $I$ 个训练样本 $\{\mathbf{x}_i\}$ 的数据集训练归一化流：

$$
\begin{aligned}
\hat{\boldsymbol{\phi}} &= \underset{\boldsymbol{\phi}}{\text{argmax}}\left[\prod_{i=1}^{I}Pr(\mathbf{z}_i) \cdot \left|\frac{\partial \text{f}[\mathbf{z}_i, \boldsymbol{\phi}]}{\partial \mathbf{z}_i}\right|^{-1}\right] \\
&= \underset{\boldsymbol{\phi}}{\text{argmin}}\left[\sum_{i=1}^{I}\log\left[\left|\frac{\partial \text{f}[\mathbf{z}_i, \boldsymbol{\phi}]}{\partial \mathbf{z}_i}\right|\right] - \log\Big[Pr(\mathbf{z}_i)\Big]\right],
\end{aligned}
\tag{16.8}
$$

其中 $\mathbf{z}_i = \text{f}^{-1}[\mathbf{x}_i, \boldsymbol{\phi}]$，$Pr(\mathbf{z}_i)$ 在基础分布下度量，绝对行列式 $|\partial \text{f}[\mathbf{z}_i, \boldsymbol{\phi}]/\partial \mathbf{z}_i|$ 由公式 16.7 给出。

### 16.2.2 网络层的理想特性

归一化流的理论很直接。然而，要使其具有实用性，我们需要具有四个性质的网络层 $\text{f}_k$：

1. 总体而言，网络层的集合必须具有足够的**表达能力**，能将多元标准正态分布映射到任意密度。
2. 网络层必须是**可逆的**；每层必须定义从任意输入点到唯一输出点的一一映射（**双射**（bijection））。如果多个输入映射到同一输出，逆就会有歧义。
3. 必须能够**高效**地计算每一层的**逆**。我们需要在每次评估似然时都这样做。这在训练期间反复发生，因此必须有逆的闭式解或快速算法。
4. 必须能够**高效**地评估正向或逆向映射的 Jacobian 的**行列式**。

## 16.3 可逆网络层

我们现在描述用于这些模型的不同的可逆网络层或**流**（flows）。我们从线性流和逐元素流开始。它们容易求逆，也容易计算其 Jacobian 的行列式，但都不具备足够的表达能力来描述基础密度的任意变换。然而，它们构成了耦合流、自回归流和残差流的构建模块，这些更具表达能力。

### 16.3.1 线性流

线性流的形式为 $\text{f}[\mathbf{h}] = \boldsymbol{\beta} + \boldsymbol{\Omega}\mathbf{h}$。如果矩阵 $\boldsymbol{\Omega}$ 可逆，则线性变换可逆。对于 $\boldsymbol{\Omega} \in \mathbb{R}^{D\times D}$，逆的计算量为 $\mathcal{O}[D^3]$。Jacobian 的行列式就是 $\boldsymbol{\Omega}$ 的行列式，计算量同样为 $\mathcal{O}[D^3]$。这意味着随着维度 $D$ 的增加，线性流变得昂贵。

如果矩阵 $\boldsymbol{\Omega}$ 采用特殊形式，则求逆和行列式的计算可以更高效，但变换的通用性会降低。例如，对角矩阵只需要 $\mathcal{O}[D]$ 的计算量来求逆和计算行列式，但 $\mathbf{h}$ 的各元素互不交互。正交矩阵也更高效，但不允许单独缩放各维度，且其行列式固定不变。三角矩阵更实用；它们可以通过称为**回代**（back-substitution）的过程以 $\mathcal{O}[D^2]$ 求逆，行列式就是对角元素的乘积。

一种构造通用、高效可逆且 Jacobian 可高效计算的线性流的方法是直接用 LU 分解来参数化。换言之，我们使用：

$$
\boldsymbol{\Omega} = \mathbf{PL}(\mathbf{U} + \mathbf{D}),
\tag{16.9}
$$

其中 $\mathbf{P}$ 是预定的置换矩阵，$\mathbf{L}$ 是下三角矩阵，$\mathbf{U}$ 是对角线为零的上三角矩阵，$\mathbf{D}$ 是提供缺失对角元素的对角矩阵。这可以在 $\mathcal{O}[D^2]$ 内求逆，对数行列式就是 $\mathbf{L}$ 和 $\mathbf{D}$ 的对角线上绝对值的对数之和。

不幸的是，线性流的表达能力不足。当线性函数 $\text{f}[\mathbf{h}] = \boldsymbol{\beta} + \boldsymbol{\Omega}\mathbf{h}$ 应用于正态分布输入 $\text{Norm}_{\mathbf{h}}[\boldsymbol{\mu}, \boldsymbol{\Sigma}]$ 时，结果仍然是正态分布，均值和方差分别为 $\boldsymbol{\beta} + \boldsymbol{\Omega}\boldsymbol{\mu}$ 和 $\boldsymbol{\Omega}\boldsymbol{\Sigma}\boldsymbol{\Omega}^T$。因此，仅使用线性流不可能将正态分布映射到任意分布。

### 16.3.2 逐元素流

由于线性流的表达能力不足，我们必须转向非线性流。最简单的是逐元素流，它对输入的每个元素应用逐点非线性函数 $\text{f}[\bullet, \boldsymbol{\phi}]$（参数为 $\boldsymbol{\phi}$）：

$$
\text{f}[\mathbf{h}] = \Big[\text{f}[h_1, \boldsymbol{\phi}], \text{f}[h_2, \boldsymbol{\phi}], \ldots \text{f}[h_D, \boldsymbol{\phi}]\Big]^T.
\tag{16.10}
$$

Jacobian $\partial \text{f}[\mathbf{h}]/\partial \mathbf{h}$ 是对角的，因为第 $d$ 个输入只影响第 $d$ 个输出。其行列式是对角元素的乘积：

$$
\left|\frac{\partial \text{f}[\mathbf{h}]}{\partial \mathbf{h}}\right| = \prod_{d=1}^{D}\left|\frac{\partial \text{f}[h_d]}{\partial h_d}\right|.
\tag{16.11}
$$

函数 $\text{f}[\bullet, \boldsymbol{\phi}]$ 可以是一个固定的可逆非线性函数，如 leaky ReLU（图 3.13），此时没有参数，也可以是任何参数化的可逆一一映射。一个简单的例子是分段线性函数（图 16.5），它将 $[0, 1]$ 映射到 $[0, 1]$，共有 $K$ 个区域：

$$
\text{f}[h, \boldsymbol{\phi}] = \left(\sum_{k=1}^{b-1}\phi_k\right) + (hK - b + 1)\phi_b,
\tag{16.12}
$$

其中参数 $\phi_1, \phi_2, \ldots, \phi_K$ 为正且和为一，$b = \lfloor Kh \rfloor + 1$ 是包含 $h$ 的区间的索引。第一项是所有前面区间的和，第二项表示当前区间中 $h$ 所在位置的比例。该函数容易求逆，其梯度几乎处处可计算。

> **图 16.5** 分段线性映射。可逆的分段线性映射 $h' = \text{f}[h, \boldsymbol{\phi}]$ 可以通过将输入域 $h \in [0, 1]$ 分成 $K$ 个等大小的区域（此处 $K = 5$）来创建。每个区域有一个斜率参数 $\phi_k$。a) 如果这些参数�正且和为一，则 b) 函数将是可逆的并映射到输出域 $h' \in [0, 1]$。

逐元素流是非线性的，但不混合输入维度，因此不能创建变量之间的相关性。当与线性流（能混合维度）交替使用时，可以建模更复杂的变换。然而，在实践中，逐元素流被用作更复杂层（如**耦合流**）的组件。

### 16.3.3 耦合流

**耦合流**（coupling flows）将输入 $\mathbf{h}$ 分为两部分 $\mathbf{h} = [\mathbf{h}_1^T, \mathbf{h}_2^T]^T$，并定义流 $\text{f}[\mathbf{h}, \boldsymbol{\phi}]$ 为：

$$
\begin{aligned}
\mathbf{h}_1' &= \mathbf{h}_1 \\
\mathbf{h}_2' &= \mathbf{g}\Big[\mathbf{h}_2, \boldsymbol{\phi}[\mathbf{h}_1]\Big].
\end{aligned}
\tag{16.13}
$$

这里 $\mathbf{g}[\bullet, \boldsymbol{\phi}]$ 是逐元素流（或其他可逆层），其参数 $\boldsymbol{\phi}[\mathbf{h}_1]$ 本身是输入 $\mathbf{h}_1$ 的非线性函数（图 16.6）。函数 $\boldsymbol{\phi}[\bullet]$ 通常是某种形式的神经网络，它本身不需要可逆。原始变量可以恢复为：

$$
\begin{aligned}
\mathbf{h}_1 &= \mathbf{h}_1' \\
\mathbf{h}_2 &= \mathbf{g}^{-1}\Big[\mathbf{h}_2', \boldsymbol{\phi}[\mathbf{h}_1]\Big].
\end{aligned}
\tag{16.14}
$$

> **图 16.6** 耦合流。a) 输入（橙色向量）被分为 $\mathbf{h}_1$ 和 $\mathbf{h}_2$。输出（青色向量）的第一部分 $\mathbf{h}_1'$ 是 $\mathbf{h}_1$ 的副本。输出 $\mathbf{h}_2'$ 通过对 $\mathbf{h}_2$ 应用可逆变换 $\mathbf{g}[\bullet, \boldsymbol{\phi}]$ 来创建，其中参数 $\boldsymbol{\phi}$ 本身是 $\mathbf{h}_1$ 的一个（不一定可逆的）函数。b) 在逆映射中，$\mathbf{h}_1 = \mathbf{h}_1'$。这允许我们计算参数 $\boldsymbol{\phi}[\mathbf{h}_1]$，然后应用逆 $\mathbf{g}^{-1}[\mathbf{h}_2', \boldsymbol{\phi}]$ 来恢复 $\mathbf{h}_2$。

如果函数 $\mathbf{g}[\bullet, \boldsymbol{\phi}]$ 是逐元素流，则 Jacobian 是下三角的：左上象限是单位矩阵，右下角是逐元素变换的导数。其行列式是这些对角值的乘积。

逆和 Jacobian 都可以高效计算，但这种方法只变换参数取决于前一半的后一半。为了使变换更通用，$\mathbf{h}$ 的元素在层之间使用**置换矩阵**（permutation matrices）随机打乱，使得每个变量最终都被其他每个变量变换。在实践中，这些置换矩阵难以学习。因此，它们被随机初始化后冻结。对于图像等结构化数据，通道被分为两半 $\mathbf{h}_1$ 和 $\mathbf{h}_2$，并使用 $1\times1$ 卷积在层之间进行置换。

### 16.3.4 自回归流

**自回归流**（autoregressive flows）是耦合流的推广，将每个输入维度视为单独的"块"（图 16.7）。它们基于输入 $\mathbf{h}$ 的前 $d-1$ 个维度来计算输出 $\mathbf{h}'$ 的第 $d$ 个维度：

$$
h_d' = \text{g}\Big[h_d, \boldsymbol{\phi}[\mathbf{h}_{1:d-1}]\Big].
\tag{16.15}
$$

函数 $\mathbf{g}[\bullet, \bullet]$ 被称为**变换器**（transformer），参数 $\boldsymbol{\phi}, \boldsymbol{\phi}[h_1], \boldsymbol{\phi}[h_1, h_2], \ldots$ 被称为**条件器**（conditioners）。与耦合流一样，变换器 $\mathbf{g}[\bullet, \boldsymbol{\phi}]$ 必须是可逆的，但条件器 $\boldsymbol{\phi}[\bullet]$ 可以采用任何形式，通常是神经网络。如果变换器和条件器具有足够的灵活性，自回归流是**通用近似器**（universal approximators），能表示任意概率分布。

可以使用带有适当掩码的网络并行计算输出 $\mathbf{h}'$ 的所有条目，使得位置 $d$ 处的参数 $\boldsymbol{\phi}$ 只依赖于前面的位置。这被称为**掩码自回归流**（masked autoregressive flow）。其原理与掩码自注意力（第 12.7.2 节）非常相似；将输入关联到前面输出的连接被剪枝。

反转变换效率较低。考虑正向映射（公式 16.16）和逆（公式 16.17）——由于 $h_d$ 的计算依赖于 $h_{1:d-1}$（即前面的部分结果），因此不能并行计算。因此，当输入很大时，求逆很耗时。

> **图 16.7** 自回归流。输入 $\mathbf{h}$（橙色列）和输出 $\mathbf{h}'$（青色列）被分成各个维度（此处四个维度）。a) 输出 $h_1'$ 是输入 $h_1$ 的可逆变换。输出 $h_2'$ 是输入 $h_2$ 的可逆函数，其参数依赖于 $h_1$。输出 $h_3'$ 是输入 $h_3$ 的可逆函数，其参数依赖于前面的输入 $h_1$ 和 $h_2$，依此类推。没有任何输出相互依赖，因此可以并行计算。b) 自回归流的逆使用与耦合流类似的方法计算。然而，注意计算 $h_2$ 需要已知 $h_1$，计算 $h_3$ 需要已知 $h_1$ 和 $h_2$，依此类推。因此，逆不能并行计算。

### 16.3.5 逆自回归流

掩码自回归流在归一化（逆）方向上定义。这需要高效地评估似然，从而学习模型。然而，采样需要正向方向，其中每个变量必须在每一层顺序计算，这很慢。如果我们对正向（生成）变换使用自回归流，则采样高效，但计算似然（和训练）慢。这被称为**逆自回归流**（inverse autoregressive flow）。

一种允许快速学习和快速（但近似）采样的技巧是使用掩码自回归流来学习分布（教师），然后使用它来训练一个逆自回归流（学生），我们可以从中高效采样。这需要一种不同的归一化流公式，即从另一个函数而不是样本集合中学习（见第 16.5.3 节）。

### 16.3.6 残差流：iRevNet

**残差流**（residual flows）从残差网络中获取灵感。它们将输入分为两部分 $\mathbf{h} = [\mathbf{h}_1^T, \mathbf{h}_2^T]^T$（与耦合流相同），并定义输出为：

$$
\begin{aligned}
\mathbf{h}_1' &= \mathbf{h}_1 + \text{f}_1[\mathbf{h}_2, \boldsymbol{\phi}_1] \\
\mathbf{h}_2' &= \mathbf{h}_2 + \text{f}_2[\mathbf{h}_1', \boldsymbol{\phi}_2],
\end{aligned}
\tag{16.18}
$$

其中 $\text{f}_1[\bullet, \boldsymbol{\phi}_1]$ 和 $\text{f}_2[\bullet, \boldsymbol{\phi}_2]$ 是两个不需要可逆的函数（图 16.8）。逆可以通过反转计算顺序来计算：

$$
\begin{aligned}
\mathbf{h}_2 &= \mathbf{h}_2' - \text{f}_2[\mathbf{h}_1', \boldsymbol{\phi}_2] \\
\mathbf{h}_1 &= \mathbf{h}_1' - \text{f}_1[\mathbf{h}_2, \boldsymbol{\phi}_1].
\end{aligned}
\tag{16.19}
$$

> **图 16.8** 残差流。a) 通过将输入分为 $\mathbf{h}_1$ 和 $\mathbf{h}_2$ 并创建两个残差层来计算可逆函数。在第一层中，$\mathbf{h}_2$ 被处理并加到 $\mathbf{h}_1$ 上。在第二层中，结果被处理并加到 $\mathbf{h}_2$ 上。b) 在反向机制中，函数以相反顺序计算，加法操作变为减法。

与耦合流一样，分块限制了可表示的变换族。因此，输入在层之间被置换，使得变量可以以任意方式混合。

这种公式容易求逆，但对于一般函数 $\text{f}_1[\bullet, \boldsymbol{\phi}_1]$ 和 $\text{f}_2[\bullet, \boldsymbol{\phi}_2]$，没有高效计算 Jacobian 的方法。这种公式有时用于训练残差网络以节省内存；因为网络是可逆的，不需要存储前向传播中每一层的激活值。

### 16.3.7 残差流与收缩映射：iResNet

利用残差网络的另一种方法是利用 **Banach 不动点定理**（Banach fixed point theorem）或**收缩映射定理**（contraction mapping theorem），它指出每个收缩映射都有一个不动点。收缩映射 $\text{f}[\bullet]$ 具有以下性质：

$$
\text{dist}\Big[\text{f}[z'], \text{f}[z]\Big] < \beta \cdot \text{dist}\Big[z', z\Big] \qquad \forall\; z, z',
\tag{16.20}
$$

其中 $\text{dist}[\bullet, \bullet]$ 是距离函数，$0 < \beta < 1$。当具有此性质的函数被迭代（即输出反复作为输入传回）时，结果收敛到不动点 $\text{f}[z] = z$（图 16.9）。

> **图 16.9** 收缩映射。如果函数处处的绝对斜率小于一，则迭代该函数会收敛到不动点 $\text{f}[z] = z$。a) 从 $z_0$ 开始，我们评估 $z_1 = \text{f}[z_0]$。然后将 $z_1$ 传回函数并迭代。最终，过程收敛到 $\text{f}[z] = z$ 的点（即函数与虚线对角恒等线交叉处）。b) 这可以用来对给定值 $y^*$ 求解形如 $y = z + \text{f}[z]$ 的方程的逆，方法是注意 $y^* - \text{f}[z]$ 的不动点（橙色线与虚线对角恒等线交叉处）与 $y^* = z + \text{f}[z]$ 的位置相同。

该定理可以用来求解形如 $y = z + \text{f}[z]$ 的方程的逆——如果 $\text{f}[z]$ 是收缩映射。换言之，它可以用来找到映射到给定值 $y^*$ 的 $z^*$。这可以通过从任意点 $z_0$ 开始并迭代 $z_{k+1} = y^* - \text{f}[z_k]$ 来完成。这在 $z + \text{f}[z] = y^*$ 处有不动点（图 16.9b）。

同样的原理可以用来求解形如 $\mathbf{h}' = \mathbf{h} + \text{f}[\mathbf{h}, \boldsymbol{\phi}]$ 的残差网络层的逆，前提是我们确保 $\text{f}[\mathbf{h}, \boldsymbol{\phi}]$ 是收缩映射。在实践中，这意味着 Lipschitz 常数必须小于一。假设激活函数的斜率不大于一，这等价于确保每个权重矩阵 $\boldsymbol{\Omega}$ 的最大**奇异值**（singular value）小于一。一种粗略的方法是通过裁剪来确保权重 $\boldsymbol{\Omega}$ 的绝对值较小。

Jacobian 行列式不能直接计算，但其对数可以用一系列技巧来近似：

$$
\log\left[\left|\mathbf{I} + \frac{\partial \text{f}[\mathbf{h}, \boldsymbol{\phi}]}{\partial \mathbf{h}}\right|\right] = \text{trace}\left[\log\left[\mathbf{I} + \frac{\partial \text{f}[\mathbf{h}, \boldsymbol{\phi}]}{\partial \mathbf{h}}\right]\right] = \sum_{k=1}^{\infty}\frac{(-1)^{k-1}}{k}\text{trace}\left[\frac{\partial \text{f}[\mathbf{h}, \boldsymbol{\phi}]}{\partial \mathbf{h}}\right]^k,
\tag{16.22}
$$

其中我们在第一行使用了恒等式 $\log[|\mathbf{A}|] = \text{trace}[\log[\mathbf{A}]]$，并在第二行将其展开为幂级数。

即使截断这个级数，计算各组成部分的**迹**（trace）仍然计算量很大。因此，我们使用 **Hutchinson 迹估计**（Hutchinson's trace estimator）来近似。考虑均值为 $\mathbf{0}$、方差为 $\mathbf{I}$ 的正态随机变量 $\boldsymbol{\epsilon}$。矩阵 $\mathbf{A}$ 的迹可以估计为：

$$
\begin{aligned}
\text{trace}[\mathbf{A}] &= \text{trace}\Big[\mathbf{A}\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^T]\Big] \\
&= \text{trace}\Big[\mathbb{E}[\mathbf{A}\boldsymbol{\epsilon}\boldsymbol{\epsilon}^T]\Big] \\
&= \mathbb{E}\Big[\text{trace}[\mathbf{A}\boldsymbol{\epsilon}\boldsymbol{\epsilon}^T]\Big] \\
&= \mathbb{E}\Big[\text{trace}[\boldsymbol{\epsilon}^T\mathbf{A}\boldsymbol{\epsilon}]\Big] \\
&= \mathbb{E}\Big[\boldsymbol{\epsilon}^T\mathbf{A}\boldsymbol{\epsilon}\Big],
\end{aligned}
\tag{16.23}
$$

其中第一行成立是因为 $\mathbb{E}[\boldsymbol{\epsilon}\boldsymbol{\epsilon}^T] = \mathbf{I}$。我们通过从 $Pr(\boldsymbol{\epsilon})$ 中采样 $\boldsymbol{\epsilon}_i$ 来估计迹：

$$
\text{trace}[\mathbf{A}] = \mathbb{E}\Big[\boldsymbol{\epsilon}^T\mathbf{A}\boldsymbol{\epsilon}\Big] \approx \frac{1}{I}\sum_{i=1}^{I}\boldsymbol{\epsilon}_i^T\mathbf{A}\boldsymbol{\epsilon}_i.
\tag{16.24}
$$

通过这种方式，我们可以近似 Taylor 展开（公式 16.22）中幂次的迹，并评估对数概率。

## 16.4 多尺度流

在归一化流中，潜空间 $\mathbf{z}$ 必须与数据空间 $\mathbf{x}$ 具有相同的大小，但我们知道自然数据集通常可以用更少的底层变量来描述。在某个阶段，我们必须引入所有这些变量，但让它们全部通过整个网络是低效的。这就引出了**多尺度流**（multi-scale flows）的概念（图 16.10）。

在生成方向上，多尺度流将潜向量分区为 $\mathbf{z} = [\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_N]$。第一个分区 $\mathbf{z}_1$ 由一系列与 $\mathbf{z}_1$ 维度相同的可逆层处理，直到在某个时刻 $\mathbf{z}_2$ 被附加并与第一个分区合并。这一过程持续到网络与数据 $\mathbf{x}$ 相同大小。在归一化方向上，网络从 $\mathbf{x}$ 的完整维度开始，但当到达添加 $\mathbf{z}_n$ 的点时，该部分与基础分布进行比较评估。

> **图 16.10** 多尺度流。归一化流中潜空间 $\mathbf{z}$ 必须与模型密度大小相同。然而，它可以被分成几个组件，在不同层逐步引入。这使得密度估计和采样都更快。对于逆过程，黑色箭头反转，每个块的最后部分跳过剩余处理。例如，$\text{f}_5^{-1}[\bullet, \boldsymbol{\phi}_5]$ 只在前三个块上操作，第四个块成为 $\mathbf{z}_4$ 并与基础密度进行评估。

## 16.5 应用

我们现在描述归一化流的三个应用。首先，我们考虑建模概率密度。其次，我们考虑用于合成图像的 GLOW 模型。最后，我们讨论使用归一化流来近似其他分布。

### 16.5.1 密度建模

在本书讨论的四种生成模型中，归一化流是唯一能精确计算新样本对数似然的模型。生成对抗网络不是概率性的，而变分自编码器和扩散模型都只能返回似然的下界。图 16.11 展示了使用 i-ResNet 在两个玩具问题上估计的概率分布。密度估计的一个应用是异常检测；使用归一化流模型描述干净数据集的数据分布。低概率的新样本被标记为异常值。然而，需要注意的是，可能存在具有高概率但不落在典型集中的异常值（见图 8.13）。

> **图 16.11** 密度建模。a) 玩具二维数据样本。b) 使用 iResNet 建模的密度。c-d) 第二个示例。改编自 Behrmann et al. (2019)。

### 16.5.2 合成

**生成流**（generative flows）或 **GLOW** 是一种归一化流模型，可以创建高保真度的图像（图 16.12），使用了本章中的许多思想。它最容易从归一化方向理解。GLOW 从包含 RGB 图像的 $256\times256\times3$ 张量开始。它使用耦合层，将通道分成两半。后一半在每个空间位置进行不同的仿射变换，其中仿射变换的参数由在另一半通道上运行的二维卷积神经网络计算。耦合层与参数化为 LU 分解的 $1\times1$ 卷积交替使用，用于混合通道。

分辨率周期性地通过将每个 $2\times2$ 补丁合并为一个位置（具有四倍通道数）来减半。GLOW 是多尺度流，部分通道周期性地被移除成为潜向量 $\mathbf{z}$ 的一部分。图像是离散的（由于 RGB 值的量化），因此在输入中添加噪声以防止训练似然无限增长。这被称为**去量化**（dequantization）。

为了采样更逼真的图像，GLOW 模型从提升到正幂次的基础密度中采样。这选择更靠近密度中心而非尾部的样本。这类似于 GAN 中的截断技巧（图 15.10）。值得注意的是，样本不如 GAN 或扩散模型的好。目前尚不清楚这是由于与可逆层相关的根本限制，还是仅仅因为在此方向上投入的研究较少。

图 16.13 展示了使用 GLOW 进行插值的示例。两个潜向量通过在归一化方向上变换两幅真实图像来计算。这些潜向量之间的中间点通过线性插值计算，然后使用网络在生成方向上投射回图像空间。结果是一组在两张真实图像之间逼真插值的图像。

> **图 16.12** 在 CelebA HQ 数据集（Karras et al., 2018）上训练的 GLOW 的样本。样本质量尚可，尽管 GAN 和扩散模型能产生更好的结果。改编自 Kingma & Dhariwal (2018)。

> **图 16.13** 使用 GLOW 模型的插值。左右图像是真实人物。中间图像通过将真实图像投射到潜空间、进行插值、然后将插值点投射回图像空间来计算。改编自 Kingma & Dhariwal (2018)。

### 16.5.3 近似其他密度模型

归一化流还可以学习生成近似于易于评估但难以采样的现有密度的样本。在这个场景中，我们将归一化流 $Pr(\mathbf{x}|\boldsymbol{\phi})$ 称为**学生**（student），目标密度 $q(\mathbf{x})$ 称为**教师**（teacher）。

为此，我们从学生中生成样本 $\mathbf{x}_i = \text{f}[\mathbf{z}_i, \boldsymbol{\phi}]$。由于样本是我们自己生成的，我们知道对应的潜变量 $\mathbf{z}_i$，可以无需求逆就计算学生模型中的似然。因此，我们可以使用像掩码自回归流这样求逆很慢的模型。我们定义一个基于反向 KL 散度的损失函数，鼓励学生和教师的似然相同，并用它来训练学生模型（图 16.14）：

$$
\hat{\boldsymbol{\phi}} = \underset{\boldsymbol{\phi}}{\text{argmin}}\left[\text{KL}\left[\frac{1}{I}\sum_{i=1}^{I}\delta\big[\mathbf{x} - \text{f}[\mathbf{z}_i, \boldsymbol{\phi}]\big]\bigg\|q(\mathbf{x})\right]\right].
\tag{16.25}
$$

这与归一化流的典型用法形成对比，典型用法是基于来自未知分布的样本 $\mathbf{x}_i$ 使用最大似然来构建概率模型 $Pr(\mathbf{x}_i, \boldsymbol{\phi})$，这依赖于正向 KL 散度的交叉熵项（第 5.7 节）：

$$
\hat{\boldsymbol{\phi}} = \underset{\boldsymbol{\phi}}{\text{argmin}}\left[\text{KL}\left[\frac{1}{I}\sum_{i=1}^{I}\delta[\mathbf{x} - \mathbf{x}_i]\bigg\|Pr(\mathbf{x}_i, \boldsymbol{\phi})\right]\right].
\tag{16.26}
$$

归一化流可以使用这个技巧来建模 VAE 中的后验分布（见第 17 章）。

> **图 16.14** 近似密度模型。a) 训练数据。b) 通常，我们修改流模型参数以最小化从训练数据到流模型的 KL 散度。这等价于最大似然拟合（第 5.7 节）。c) 或者，我们可以修改流参数 $\boldsymbol{\phi}$ 以最小化从流样本 $x_i = \text{f}[z_i, \boldsymbol{\phi}]$ 到 d) 目标密度的 KL 散度。

## 16.6 总结

归一化流将基础分布（通常是正态分布）变换为新的密度。它们的优势在于既能精确评估样本的似然，又能生成新样本。然而，它们有架构约束：每一层必须是可逆的；我们需要正向变换来生成样本，反向变换来评估似然。

高效评估 Jacobian 以计算似然也很重要；这在学习密度时必须反复进行。然而，可逆层即使在无法高效估计 Jacobian 时也有其自身的用处；它们将训练 $K$ 层网络的内存需求从 $\mathcal{O}[K]$ 降低到 $\mathcal{O}[1]$。

本章回顾了可逆网络层或流。我们考虑了线性流和逐元素流，它们简单但表达能力不足。然后我们描述了更复杂的流，如耦合流、自回归流和残差流。最后，我们展示了归一化流如何用于估计似然、生成和在图像之间插值，以及近似其他分布。
