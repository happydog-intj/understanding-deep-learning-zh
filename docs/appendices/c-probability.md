# 附录C 概率论基础

概率对深度学习至关重要。在监督学习中，深度网络隐含地依赖于损失函数的概率表述。在无监督学习中，生成模型旨在生成与训练数据来自相同概率分布的样本。强化学习在马尔可夫决策过程中进行，这些过程以概率分布来定义。本附录提供机器学习中所用概率的入门知识。

## C.1 随机变量与概率分布

*随机变量*（random variable）$x$ 表示一个不确定的量。它可以是*离散的*（discrete）（仅取某些值，如整数）或*连续的*（continuous）（在连续域上取任意值，如实数）。如果我们观察随机变量 $x$ 的多个实例，它将取不同的值，取不同值的相对倾向由*概率分布*（probability distribution）$Pr(x)$ 描述。

对于离散变量，该分布为每个可能的结果 $k$ 关联一个*概率*（probability）$Pr(x=k) \in [0, 1]$，所有概率之和为1。对于连续变量，与每个值 $a$ 关联的是一个非负的*概率密度*（probability density）$Pr(x=a) \geq 0$，且该概率密度函数（PDF）在定义域上的积分必须为1。对于任意点 $a$，该密度可以大于1。从这里开始，我们假设随机变量是连续的。对于离散分布，思想完全相同，只是将积分替换为求和。

### C.1.1 联合概率

考虑有两个随机变量 $x$ 和 $y$ 的情况。*联合分布*（joint distribution）$Pr(x, y)$ 描述了 $x$ 和 $y$ 取特定值组合的倾向（图 C.1a）。现在与每对值 $x = a$ 和 $y = b$ 关联的是一个非负概率密度 $Pr(x=a, y=b)$，且必须满足：

$$\iint Pr(x, y) \cdot dx dy = 1. \tag{C.1}$$

这一思想可以扩展到两个以上的变量，因此 $x$、$y$ 和 $z$ 的联合密度写为 $Pr(x, y, z)$。有时，我们将多个随机变量存储在向量 $\mathbf{x}$ 中，并将其联合密度写为 $Pr(\mathbf{x})$。进一步扩展，我们可以将两个向量 $\mathbf{x}$ 和 $\mathbf{y}$ 中所有变量的联合密度写为 $Pr(\mathbf{x}, \mathbf{y})$。

> **图 C.1** 联合分布和边缘分布。a) 联合分布 $Pr(x, y)$ 描述了变量 $x$ 和 $y$ 取不同值组合的倾向。这里，概率密度用颜色图表示，较亮的位置概率更大。b) 变量 $x$ 的边缘分布 $Pr(x)$ 可以通过对 $y$ 积分来恢复。c) 变量 $y$ 的边缘分布 $Pr(y)$ 可以通过对 $x$ 积分来恢复。

### C.1.2 边缘化

如果我们知道两个变量的联合分布 $Pr(x, y)$，可以通过对另一个变量积分来恢复*边缘*（marginal）分布 $Pr(x)$ 和 $Pr(y)$（图 C.1b-c）：

$$\int Pr(x, y) \cdot dx = Pr(y)$$
$$\int Pr(x, y) \cdot dy = Pr(x). \tag{C.2}$$

这一过程称为*边缘化*（marginalization），其含义是我们在计算一个变量的分布时*不考虑*另一个变量取何值。边缘化的思想可以扩展到更高维度，因此如果我们有联合分布 $Pr(x, y, z)$，我们可以通过对 $y$ 积分来恢复联合分布 $Pr(x, z)$。

### C.1.3 条件概率与似然

*条件概率*（conditional probability）$Pr(x|y)$ 是在已知 $y$ 值的条件下变量 $x$ 取某个值的概率。竖线读作英语单词"given"（给定），因此 $Pr(x|y)$ 是"在给定 $y$ 的条件下 $x$ 的概率"。条件概率 $Pr(x|y)$ 可以通过取联合分布 $Pr(x, y)$ 对于固定 $y$ 值的一个切片来得到。然后将这个切片除以该 $y$ 值出现的概率（使切片下的总面积为 $Pr(y=3.0)$），从而使条件分布的总和为1（图 C.2）：

$$Pr(x|y) = \frac{Pr(x, y)}{Pr(y)}. \tag{C.3}$$

类似地，

$$Pr(y|x) = \frac{Pr(x, y)}{Pr(x)}. \tag{C.4}$$

> **图 C.2** 条件分布。a) 变量 $x$ 和 $y$ 的联合分布 $Pr(x, y)$。b) 在给定变量 $y$ 取值 3.0 条件下变量 $x$ 的条件概率 $Pr(x|y=3.0)$，通过取联合概率 $Pr(x, y=3.0)$ 的水平"切片"（面板 a 中的上方青色线），然后除以该切片的总面积 $Pr(y=3.0)$，使其形成积分为1的有效概率分布。c) 联合概率 $Pr(x, y=-1.0)$ 类似地使用 $y=-1.0$ 处的切片得到。

当我们将条件概率 $Pr(x|y)$ 看作 $x$ 的函数时，它必须求和（积分）为1。当我们将同一个量 $Pr(x|y)$ 看作 $y$ 的函数时，它被称为 $x$ 在给定 $y$ 条件下的*似然*（likelihood），且不需要求和为1。

### C.1.4 贝叶斯规则

从公式 C.3 和 C.4，我们得到联合概率 $Pr(x, y)$ 的两种表达式：

$$Pr(x, y) = Pr(x|y)Pr(y) = Pr(y|x)Pr(x), \tag{C.5}$$

对其重新排列得到：

$$Pr(x|y) = \frac{Pr(y|x)Pr(x)}{Pr(y)}. \tag{C.6}$$

这个表达式将 $x$ 在给定 $y$ 条件下的条件概率 $Pr(x|y)$ 与 $y$ 在给定 $x$ 条件下的条件概率 $Pr(y|x)$ 联系起来，被称为*贝叶斯规则*（Bayes' rule）。

贝叶斯规则中的每一项都有名称。项 $Pr(y|x)$ 是 $y$ 在给定 $x$ 条件下的*似然*（likelihood），项 $Pr(x)$ 是 $x$ 的*先验概率*（prior probability）。分母 $Pr(y)$ 被称为*证据*（evidence），左侧的 $Pr(x|y)$ 被称为*后验概率*（posterior probability）。该方程将先验 $Pr(x)$（观察 $y$ 之前我们对 $x$ 的了解）映射到后验 $Pr(x|y)$（观察 $y$ 之后我们对 $x$ 的了解）。

### C.1.5 独立性

如果随机变量 $y$ 的值对我们了解 $x$ 没有任何帮助，反之亦然，我们称 $x$ 和 $y$ 是*独立的*（independent），可以写为 $Pr(x|y) = Pr(x)$ 且 $Pr(y|x) = Pr(y)$。由此可得所有条件分布 $Pr(y|x=\bullet)$ 都是相同的，条件分布 $Pr(x|y=\bullet)$ 也是相同的。

从公式 C.5 中联合概率的第一个表达式出发，我们看到当变量独立时，联合分布等于边缘分布的乘积：

$$Pr(x, y) = Pr(x|y)Pr(y) = Pr(x)Pr(y) \tag{C.7}$$

（图 C.3）。

> **图 C.3** 独立性。a) 当两个变量 $x$ 和 $y$ 独立时，联合分布分解为边缘分布的乘积，即 $Pr(x, y) = Pr(x)Pr(y)$。独立性意味着知道一个变量的值不能告诉我们关于另一个变量的任何信息。b-c) 因此，所有条件分布 $Pr(x|y=\bullet)$ 都相同，且等于边缘分布 $Pr(x)$。

## C.2 期望

考虑函数 $\text{f}[x]$ 和定义在 $x$ 上的概率分布 $Pr(x)$。函数 $\text{f}[\bullet]$ 相对于随机变量 $x$ 的概率分布 $Pr(x)$ 的*期望值*（expected value）定义为：

$$\mathbb{E}_x\left[\text{f}[x]\right] = \int \text{f}[x] Pr(x) dx. \tag{C.8}$$

顾名思义，这是在考虑看到 $x$ 不同值的概率后，$\text{f}[x]$ 的期望或平均值。这一思想可以推广到多个随机变量的函数 $\text{f}[\bullet, \bullet]$：

$$\mathbb{E}_{x,y}\left[\text{f}[x, y]\right] = \iint \text{f}[x, y] Pr(x, y) dx dy. \tag{C.9}$$

期望总是相对于一个或多个变量的分布来取的。然而，当分布的选择显而易见时，我们通常不显式说明，写为 $\mathbb{E}[\text{f}[x]]$ 而非 $\mathbb{E}_x[\text{f}[x]]$。

如果我们从 $Pr(x)$ 中抽取大量样本 $\{x_i\}_{i=1}^{I}$，对每个样本计算 $\text{f}[x_i]$ 并取这些值的平均，结果将近似函数的期望 $\mathbb{E}[\text{f}[x]]$：

$$\mathbb{E}_x\left[\text{f}[x]\right] \approx \frac{1}{I}\sum_{i=1}^{I} \text{f}[x_i]. \tag{C.10}$$

### C.2.1 期望的运算规则

操纵期望有四条规则：

$$\mathbb{E}[k] = k$$
$$\mathbb{E}[k \cdot \text{f}[x]] = k \cdot \mathbb{E}[\text{f}[x]]$$
$$\mathbb{E}[\text{f}[x] + \text{g}[x]] = \mathbb{E}[\text{f}[x]] + \mathbb{E}[\text{g}[x]]$$
$$\mathbb{E}_{x,y}[\text{f}[x] \cdot \text{g}[y]] = \mathbb{E}_x[\text{f}[x]] \cdot \mathbb{E}_y[\text{g}[y]] \quad \text{若} \; x, y \; \text{独立}, \tag{C.11}$$

其中 $k$ 是任意常数。以下对连续情况进行证明。

**规则1：** 常数 $k$ 的期望 $\mathbb{E}[k]$ 就是 $k$。

$$\mathbb{E}[k] = \int k \cdot Pr(x) dx = k \cdot \int Pr(x) dx = k. \tag{C.12}$$

**规则2：** 常数 $k$ 乘以变量 $x$ 的函数的期望 $\mathbb{E}[k \cdot \text{f}[x]]$ 是 $k$ 乘以该函数的期望 $\mathbb{E}[\text{f}[x]]$：

$$\mathbb{E}[k \cdot \text{f}[x]] = \int k \cdot \text{f}[x] Pr(x) dx = k \cdot \int \text{f}[x] Pr(x) dx = k \cdot \mathbb{E}[\text{f}[x]]. \tag{C.13}$$

**规则3：** 和 $\mathbb{E}[\text{f}[x] + \text{g}[x]]$ 的期望是各项期望 $\mathbb{E}[\text{f}[x]] + \mathbb{E}[\text{g}[x]]$ 的和：

$$\mathbb{E}[\text{f}[x] + \text{g}[x]] = \int (\text{f}[x] + \text{g}[x]) \cdot Pr(x) dx$$
$$= \int (\text{f}[x] \cdot Pr(x) + \text{g}[x] \cdot Pr(x)) dx$$
$$= \int \text{f}[x] \cdot Pr(x) dx + \int \text{g}[x] \cdot Pr(x) dx$$
$$= \mathbb{E}[\text{f}[x]] + \mathbb{E}[\text{g}[x]]. \tag{C.14}$$

**规则4：** 当 $x$ 和 $y$ 独立时，乘积 $\mathbb{E}[\text{f}[x] \cdot \text{g}[y]]$ 的期望是各项期望的乘积 $\mathbb{E}[\text{f}[x]] \cdot \mathbb{E}[\text{g}[y]]$：

$$\mathbb{E}[\text{f}[x] \cdot \text{g}[y]] = \iint \text{f}[x] \cdot \text{g}[y] Pr(x, y) dx dy$$
$$= \iint \text{f}[x] \cdot \text{g}[y] Pr(x) Pr(y) dx dy$$
$$= \int \text{f}[x] \cdot Pr(x) dx \int \text{g}[y] \cdot Pr(y) dy$$
$$= \mathbb{E}[\text{f}[x]] \mathbb{E}[\text{g}[y]], \tag{C.15}$$

其中我们在前两行之间使用了独立性的定义（公式 C.7）。

这四条规则可以推广到多变量情况：

$$\mathbb{E}[\mathbf{A}] = \mathbf{A}$$
$$\mathbb{E}[\mathbf{A} \cdot \text{f}[\mathbf{x}]] = \mathbf{A}\mathbb{E}[\text{f}[\mathbf{x}]]$$
$$\mathbb{E}[\text{f}[\mathbf{x}] + \text{g}[\mathbf{x}]] = \mathbb{E}[\text{f}[\mathbf{x}]] + \mathbb{E}[\text{g}[\mathbf{x}]]$$
$$\mathbb{E}_{\mathbf{x},\mathbf{y}}[\text{f}[\mathbf{x}]^T\text{g}[\mathbf{y}]] = \mathbb{E}_{\mathbf{x}}[\text{f}[\mathbf{x}]]^T\mathbb{E}_{\mathbf{y}}[\text{g}[\mathbf{y}]] \quad \text{若} \; \mathbf{x}, \mathbf{y} \; \text{独立}, \tag{C.16}$$

其中 $\mathbf{A}$ 是一个常数矩阵，$\text{f}[\mathbf{x}]$ 是一个以向量 $\mathbf{x}$ 为输入返回向量的函数，$\text{g}[\mathbf{y}]$ 是一个以向量 $\mathbf{y}$ 为输入也返回向量的函数。

### C.2.2 均值、方差和协方差

对于函数 $\text{f}[\bullet]$ 的某些选择，期望有一个特殊的名称。这些量经常用于概括复杂分布的性质。例如，当 $\text{f}[x] = x$ 时，得到的期望 $\mathbb{E}[x]$ 被称为*均值*（mean）$\mu$。它是分布中心的度量。类似地，与均值的期望平方偏差 $\mathbb{E}[(x - \mu)^2]$ 被称为*方差*（variance）$\sigma^2$。它是分布离散程度的度量。*标准差*（standard deviation）$\sigma$ 是方差的正平方根。它也度量分布的离散程度，但具有与变量 $x$ 相同单位的优点。

顾名思义，两个变量 $x$ 和 $y$ 的*协方差*（covariance）$\mathbb{E}[(x - \mu_x)(y - \mu_y)]$ 度量它们共同变化的程度。这里 $\mu_x$ 和 $\mu_y$ 分别表示变量 $x$ 和 $y$ 的均值。当两个变量的方差都很大且 $x$ 的值增加时 $y$ 的值也倾向于增加时，协方差将很大。

如果两个变量独立，则它们的协方差为零。然而，协方差为零并不意味着独立。例如，考虑一个分布 $Pr(x, y)$，其中概率均匀分布在以原点为圆心的单位半径圆上。$x$ 的增加平均而言不伴随 $y$ 的增加，反之亦然。然而，知道 $x = 0$ 的值告诉我们 $y$ 有相等的概率取 $\pm 1$，所以这两个变量不能是独立的。

存储在列向量 $\mathbf{x} \in \mathbb{R}^D$ 中的多个随机变量的协方差可以由 $D \times D$ *协方差矩阵*（covariance matrix）$\mathbb{E}[(\mathbf{x} - \boldsymbol{\mu}_x)(\mathbf{x} - \boldsymbol{\mu}_x)^T]$ 表示，其中向量 $\boldsymbol{\mu}_x$ 包含均值 $\mathbb{E}[\mathbf{x}]$。该矩阵在位置 $(i, j)$ 的元素表示变量 $x_i$ 和 $x_j$ 之间的协方差。

### C.2.3 方差恒等式

期望的运算规则（附录 C.2.1）可以用来证明以下恒等式，它允许我们以不同的形式写出方差：

$$\mathbb{E}[(x - \mu)^2] = \mathbb{E}[x^2] - \mathbb{E}[x]^2. \tag{C.17}$$

**证明：**

$$\mathbb{E}[(x - \mu)^2] = \mathbb{E}[x^2 - 2\mu x + \mu^2]$$
$$= \mathbb{E}[x^2] - \mathbb{E}[2\mu x] + \mathbb{E}[\mu^2]$$
$$= \mathbb{E}[x^2] - 2\mu \cdot \mathbb{E}[x] + \mu^2$$
$$= \mathbb{E}[x^2] - 2\mu^2 + \mu^2$$
$$= \mathbb{E}[x^2] - \mu^2$$
$$= \mathbb{E}[x^2] - \mathbb{E}[x]^2, \tag{C.18}$$

其中我们在第一行和第二行之间使用了规则3，在第二行和第三行之间使用了规则1和2，并在第四行和第六行使用了定义 $\mu = \mathbb{E}[x]$。

### C.2.4 标准化

将随机变量的均值设为零、方差设为1的过程称为*标准化*（standardization）。这通过以下变换实现：

$$z = \frac{x - \mu}{\sigma}, \tag{C.19}$$

其中 $\mu$ 是 $x$ 的均值，$\sigma$ 是标准差。

**证明：** 新分布关于 $z$ 的均值为：

$$\mathbb{E}[z] = \mathbb{E}\left[\frac{x - \mu}{\sigma}\right] = \frac{1}{\sigma}\mathbb{E}[x - \mu] = \frac{1}{\sigma}(\mathbb{E}[x] - \mathbb{E}[\mu]) = \frac{1}{\sigma}(\mu - \mu) = 0, \tag{C.20}$$

新分布的方差为：

$$\mathbb{E}[(z - \mu_z)^2] = \mathbb{E}[(z - \mathbb{E}[z])^2] = \mathbb{E}[z^2] = \mathbb{E}\left[\left(\frac{x - \mu}{\sigma}\right)^2\right] = \frac{1}{\sigma^2} \cdot \mathbb{E}[(x - \mu)^2] = \frac{1}{\sigma^2} \cdot \sigma^2 = 1. \tag{C.21}$$

通过类似的论证，我们可以将均值为零方差为1的标准化变量 $z$ 转换为均值为 $\mu$ 方差为 $\sigma^2$ 的变量 $x$：

$$x = \mu + \sigma z. \tag{C.22}$$

在多变量情况下，我们可以使用以下公式标准化均值为 $\boldsymbol{\mu}$、协方差矩阵为 $\boldsymbol{\Sigma}$ 的变量 $\mathbf{x}$：

$$\mathbf{z} = \boldsymbol{\Sigma}^{-1/2}(\mathbf{x} - \boldsymbol{\mu}). \tag{C.23}$$

结果的均值为 $\mathbb{E}[\mathbf{z}] = \mathbf{0}$，协方差矩阵为单位矩阵 $\mathbb{E}[(\mathbf{z} - \mathbb{E}[\mathbf{z}])(\mathbf{z} - \mathbb{E}[\mathbf{z}])^T] = \mathbf{I}$。要逆转这一过程，我们使用：

$$\mathbf{x} = \boldsymbol{\mu} + \boldsymbol{\Sigma}^{1/2}\mathbf{z}. \tag{C.24}$$

## C.3 正态概率分布

本书中使用的概率分布包括伯努利分布（图5.6）、分类分布（图5.9）、泊松分布（图5.15）、冯·米塞斯分布（图5.13）和高斯混合模型（图5.14和17.1）。然而，机器学习中最常见的分布是正态分布（normal distribution）或高斯分布（Gaussian distribution）。

### C.3.1 单变量正态分布

单变量正态分布（图5.3）定义在标量变量 $x$ 上，有两个参数：均值 $\mu$ 和方差 $\sigma^2$，定义为：

$$Pr(x) = \text{Norm}_x[\mu, \sigma^2] = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left[-\frac{(x - \mu)^2}{2\sigma^2}\right]. \tag{C.25}$$

不出所料，正态分布变量的均值 $\mathbb{E}[x]$ 由均值参数 $\mu$ 给出，方差 $\mathbb{E}[(x - \mathbb{E}[x])^2]$ 由方差参数 $\sigma^2$ 给出。当均值为零且方差为1时，我们称之为*标准正态分布*（standard normal distribution）。

正态分布的形状可以从以下论证推断。项 $-(x-\mu)^2/2\sigma^2$ 是一个二次函数，在 $x = \mu$ 处从零开始下降，当 $\sigma$ 变小时下降速率增加。当我们通过指数函数（图 B.1）传递这个值时，得到一个在 $x = \mu$ 处值为1并向两侧下降的钟形曲线。除以常数 $\sqrt{2\pi\sigma^2}$ 确保函数积分为1，是一个有效的分布。由此可知，均值 $\mu$ 控制钟形曲线中心的位置，方差的平方根 $\sigma$（即标准差）控制钟形曲线的宽度。

### C.3.2 多变量正态分布

多变量正态分布将正态分布推广为描述长度为 $D$ 的向量量 $\mathbf{x}$ 上的概率。它由一个 $D \times 1$ 的*均值向量* $\boldsymbol{\mu}$ 和一个对称正定的 $D \times D$ *协方差矩阵* $\boldsymbol{\Sigma}$ 定义：

$$\text{Norm}_{\mathbf{x}}[\boldsymbol{\mu}, \boldsymbol{\Sigma}] = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left[-\frac{(\mathbf{x} - \boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})}{2}\right]. \tag{C.26}$$

其解释与单变量情况类似。二次项 $-(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})/2$ 返回一个标量，随着 $\mathbf{x}$ 远离均值 $\boldsymbol{\mu}$ 而减小，减小的速率取决于矩阵 $\boldsymbol{\Sigma}$。指数函数将其转化为钟形曲线，除以 $(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}$ 确保分布积分为1。

协方差矩阵可以取球形、对角和完整形式：

$$\boldsymbol{\Sigma}_{spher} = \begin{bmatrix} \sigma^2 & 0 \\ 0 & \sigma^2 \end{bmatrix} \quad \boldsymbol{\Sigma}_{diag} = \begin{bmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{bmatrix} \quad \boldsymbol{\Sigma}_{full} = \begin{bmatrix} \sigma_{11}^2 & \sigma_{12}^2 \\ \sigma_{21}^2 & \sigma_{22}^2 \end{bmatrix}. \tag{C.27}$$

> **图 C.4** 二元正态分布。a-b) 当协方差矩阵是单位矩阵的倍数时，等密度轮廓是圆，我们称之为球形协方差。c-d) 当协方差是任意对角矩阵时，等密度轮廓是与坐标轴对齐的椭圆，我们称之为对角协方差。e-f) 当协方差是任意对称正定矩阵时，等密度轮廓是一般椭圆，我们称之为完整协方差。

在二维情况下（图 C.4），球形协方差产生圆形等密度轮廓，对角协方差产生与坐标轴对齐的椭圆等密度轮廓。完整协方差产生一般的椭圆等密度轮廓。当协方差是球形或对角的时，各个变量是独立的。

### C.3.3 两个正态分布的乘积

两个正态分布的乘积正比于第三个正态分布，关系如下：

$$\text{Norm}_{\mathbf{x}}[\mathbf{a}, \mathbf{A}] \text{Norm}_{\mathbf{x}}[\mathbf{b}, \mathbf{B}] \propto \text{Norm}_{\mathbf{x}}\left[(\mathbf{A}^{-1} + \mathbf{B}^{-1})^{-1}(\mathbf{A}^{-1}\mathbf{a} + \mathbf{B}^{-1}\mathbf{b}), (\mathbf{A}^{-1} + \mathbf{B}^{-1})^{-1}\right]. \tag{C.29}$$

这可以通过展开指数项并配方来轻松证明（参见问题18.5）。

### C.3.4 变量替换

当多变量正态分布 $\mathbf{x}$ 的均值是第二个变量 $\mathbf{y}$ 的线性函数 $\mathbf{Ay} + \mathbf{b}$ 时，这正比于 $\mathbf{y}$ 中的另一个正态分布，其中均值是 $\mathbf{x}$ 的线性函数：

$$\text{Norm}_{\mathbf{x}}[\mathbf{Ay} + \mathbf{b}, \boldsymbol{\Sigma}] \propto \text{Norm}_{\mathbf{y}}[(\mathbf{A}^T\boldsymbol{\Sigma}^{-1}\mathbf{A})^{-1}\mathbf{A}^T\boldsymbol{\Sigma}^{-1}(\mathbf{x} - \mathbf{b}), (\mathbf{A}^T\boldsymbol{\Sigma}^{-1}\mathbf{A})^{-1}]. \tag{C.30}$$

> **图 C.5** 变量替换。a) 条件分布 $Pr(x|y)$ 是一个方差恒定、均值线性依赖于 $y$ 的正态分布。青色分布展示了 $y = -0.2$ 时的一个例子。b) 这正比于条件概率 $Pr(y|x)$，它是一个方差恒定、均值线性依赖于 $x$ 的正态分布。青色分布展示了 $x = -3$ 时的一个例子。

乍看之下，这个关系相当晦涩，但图 C.5 展示了标量 $x$ 和 $y$ 的情况，这很容易理解。与前一个关系一样，这可以通过展开指数项中的二次乘积并配方使之成为 $\mathbf{y}$ 的分布来证明（参见问题18.4）。

## C.4 采样

要从单变量分布 $Pr(x)$ 中采样，我们首先计算累积分布函数 $\text{F}[x]$（即 $Pr(x)$ 的积分）。然后从 $[0, 1]$ 范围内的均匀分布中抽取一个样本 $z^*$，并对其求累积分布函数的逆值，这样样本 $x^*$ 就创建为：

$$x^* = \text{F}^{-1}[z^*]. \tag{C.31}$$

### C.4.1 从正态分布采样

上述方法可用于从单变量标准正态分布生成样本 $x^*$。然后可以使用公式 C.22 创建均值为 $\mu$、方差为 $\sigma^2$ 的正态分布的样本。类似地，可以通过独立采样 $D$ 个单变量标准正态随机变量来创建 $D$ 维多变量标准分布的样本 $\mathbf{x}^*$。然后可以使用公式 C.24 创建均值为 $\boldsymbol{\mu}$、协方差为 $\boldsymbol{\Sigma}$ 的多变量正态分布的样本。

### C.4.2 祖先采样

当联合分布可以分解为一系列条件概率时，我们可以使用*祖先采样*（ancestral sampling）来生成样本。基本思想是从根变量生成一个样本，然后基于这个实例从后续的条件分布中采样。这一过程称为*祖先采样*，通过一个例子最容易理解。考虑三个变量 $x$、$y$ 和 $z$ 上的联合分布 $Pr(x, y, z)$，它（在这个特定情况下）分解为：

$$Pr(x, y, z) = Pr(x)Pr(y|x)Pr(z|y). \tag{C.32}$$

要从这个联合分布中采样，我们首先从 $Pr(x)$ 中抽取一个样本 $x^*$。然后从 $Pr(y|x^*)$ 中抽取一个样本 $y^*$。最后，从 $Pr(z|y^*)$ 中抽取一个样本 $z^*$。

## C.5 概率分布之间的距离

监督学习可以表述为最小化模型所暗示的概率分布与样本所暗示的离散概率分布之间的距离（第5.7节）。无监督学习通常也可以表述为最小化真实样本的概率分布与模型数据分布之间的距离。在两种情况下，我们都需要两个概率分布之间的距离度量。本节考虑几种不同距离度量的性质（另请参见图15.8中关于 Wasserstein 或推土机距离的讨论）。

### C.5.1 KL散度

两个概率分布 $p(x)$ 和 $q(x)$ 之间最常见的距离度量是 *Kullback-Leibler 散度*（KL divergence），定义为：

$$D_{KL}[p(x)||q(x)] = \int p(x) \log\left[\frac{p(x)}{q(x)}\right] dx. \tag{C.33}$$

这个距离始终大于或等于零，这可以通过注意到 $-\log[y] \geq 1 - y$（图 C.6）来轻松证明。

> **图 C.6** 负对数的下界。函数 $1 - y$ 总是小于函数 $-\log[y]$。这个关系用于证明 Kullback-Leibler 散度始终大于或等于零。

当存在 $q(x)$ 为零但 $p(x)$ 非零的地方时，KL 散度为无穷大。这在基于该距离最小化函数时可能导致问题。

### C.5.2 Jensen-Shannon 散度

KL 散度不是对称的（即 $D_{KL}[p(x)||q(x)] \neq D_{KL}[q(x)||p(x)]$）。*Jensen-Shannon 散度*（Jensen-Shannon divergence）是一个构造上对称的距离度量：

$$D_{JS}[p(x)||q(x)] = \frac{1}{2}D_{KL}\left[p(x)\left|\left|\frac{p(x)+q(x)}{2}\right.\right.\right] + \frac{1}{2}D_{KL}\left[q(x)\left|\left|\frac{p(x)+q(x)}{2}\right.\right.\right]. \tag{C.35}$$

它是 $p(x)$ 和 $q(x)$ 分别到两个分布平均值的平均散度。

### C.5.3 Fréchet 距离

两个分布 $p(x)$ 和 $q(x)$ 之间的 *Fréchet 距离*（Fréchet distance）$D_{Fr}$ 定义为：

$$D_{Fr}\left[p(x)||q(y)\right] = \sqrt{\min_{\pi(x,y)} \left[\iint \pi(x,y)|x - y|^2 dx dy\right]}, \tag{C.36}$$

其中 $\pi(x, y)$ 表示与边缘分布 $p(x)$ 和 $q(y)$ 兼容的联合分布的集合。Fréchet 距离也可以表述为累积概率曲线之间的最大距离。

### C.5.4 正态分布之间的距离

我们经常需要计算两个均值分别为 $\boldsymbol{\mu}_1$ 和 $\boldsymbol{\mu}_2$、协方差分别为 $\boldsymbol{\Sigma}_1$ 和 $\boldsymbol{\Sigma}_2$ 的多变量正态分布之间的距离。在这种情况下，各种距离度量可以写成闭式形式。

KL 散度可以计算为：

$$D_{KL}\left[\text{Norm}[\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1] || \text{Norm}[\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2]\right] = \tag{C.37}$$
$$\frac{1}{2}\left(\log\left[\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|}\right] - D + \text{tr}\left[\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1\right] + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^T\boldsymbol{\Sigma}_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)\right),$$

其中 $tr[\bullet]$ 是矩阵参数的迹。Fréchet/2-Wasserstein 距离为：

$$D_{Fr/W_2}^2\left[\text{Norm}[\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1] || \text{Norm}[\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2]\right] = |\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2|^2 + \text{tr}\left[\boldsymbol{\Sigma}_1 + \boldsymbol{\Sigma}_2 - 2(\boldsymbol{\Sigma}_1\boldsymbol{\Sigma}_2)^{1/2}\right]. \tag{C.38}$$



