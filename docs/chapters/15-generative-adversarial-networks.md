# 第15章 生成对抗网络

**生成对抗网络**（generative adversarial network）或 **GAN** 是一种无监督模型，旨在生成与训练样本不可区分的新样本。GAN 只是创建新样本的机制；它们不构建数据上的概率分布，因此无法评估新数据点属于同一分布的概率。

在 GAN 中，主要的**生成器**（generator）网络通过将随机噪声映射到输出数据空间来创建样本。如果第二个**判别器**（discriminator）网络无法区分生成样本和真实样本，则样本必然是合理的。如果该网络*能够*区分差异，则提供了一个训练信号，可以反馈回来改善样本质量。这个想法很简单，但训练 GAN 是困难的：学习算法可能不稳定，而且虽然 GAN 可能学会生成逼真的样本，但这并不意味着它们学会了生成*所有*可能的样本。

GAN 已被应用于多种类型的数据，包括音频、3D 模型、文本、视频和图。然而，它们在图像领域取得了最大的成功，能够生成与真实照片几乎不可区分的样本。因此，本章的示例集中于合成图像。

## 15.1 判别作为信号

我们的目标是生成新样本 $\{\mathbf{x}_j^*\}$，使其与真实训练数据 $\{\mathbf{x}_i\}$ 来自相同的分布。单个新样本 $\mathbf{x}_j^*$ 通过 (i) 从简单基础分布（例如标准正态分布）中选择一个**潜变量**（latent variable）$\mathbf{z}_j$，然后 (ii) 将该数据通过网络 $\mathbf{x}_j^* = \mathbf{g}[\mathbf{z}_j, \boldsymbol{\theta}]$（参数为 $\boldsymbol{\theta}$）来生成。该网络被称为**生成器**。在学习过程中，目标是找到参数 $\boldsymbol{\theta}$ 使得样本 $\{\mathbf{x}_j^*\}$ 看起来与真实数据 $\{\mathbf{x}_i\}$ "相似"（见图 14.2a）。

相似性可以用多种方式定义，但 GAN 使用的原则是样本应该在统计上与真实数据不可区分。为此，引入第二个带参数 $\boldsymbol{\phi}$ 的网络 $\text{f}[\bullet, \boldsymbol{\phi}]$，称为**判别器**。该网络旨在将其输入分类为真实样本或生成样本。如果这被证明是不可能的，则生成样本与真实样本不可区分，我们就成功了。如果可以区分，判别器提供了一个可用于改善生成过程的信号。

图 15.1 说明了这一方案。我们从一组真实的一维样本 $\{x_i\}$ 开始。每个面板中都展示了从中选取的十个样本 $\{x_i\}_{i=1}^{10}$（青色箭头）的不同批次。为了创建一批样本 $\{x_j^*\}$，我们使用简单的生成器：

$$
x_j^* = \text{g}[z_j, \theta] = z_j + \theta,
\tag{15.1}
$$

其中潜变量 $\{z_j\}$ 从标准正态分布中抽取，参数 $\theta$ 沿 x 轴平移生成样本（图 15.1）。

在初始化时，$\theta = 3.0$，生成样本（橙色箭头）位于真实样本（青色箭头）的左侧。判别器被训练来区分生成样本和真实样本（sigmoid 曲线表示数据点为真实的估计概率）。在训练过程中，生成器参数 $\theta$ 被调整以增加其样本被分类为真实的概率。这里意味着增大 $\theta$，使样本向右移动到 sigmoid 曲线较高的位置。

我们交替更新判别器和生成器。图 15.1b-c 展示了这个过程的两次迭代。分类数据逐渐变得更困难，因此改变 $\theta$ 的驱动力减弱（即 sigmoid 变得更平坦）。在过程结束时，无法区分两组数据；判别器此时只有随机的表现，被丢弃，而我们得到了一个能生成合理样本的生成器。

<div align="center">

![图 15.1](/figures/ch15/GanDiscrimGen.png)

</div>

> **图 15.1** GAN 机制。a) 给定一个参数化函数（生成器），合成样本（橙色箭头）和一批真实样本（青色箭头），我们训练判别器来区分真实样本和生成样本（sigmoid 曲线表示数据点为真实的估计概率）。b) 通过修改生成器参数使判别器对样本是否合成变得不确定（在此例中，将橙色样本向右移动）来训练生成器。然后更新判别器。c) 对生成器和判别器的交替更新使生成样本变得与真实样本不可区分，改变生成器的驱动力（即 sigmoid 函数的斜率）逐渐减小。

### 15.1.1 GAN 损失函数

我们现在更精确地定义训练 GAN 的损失函数。判别器 $\text{f}[\mathbf{x}, \boldsymbol{\phi}]$ 接受输入 $\mathbf{x}$，具有参数 $\boldsymbol{\phi}$，当它认为输入是真实样本时返回一个较高的标量。这是一个二分类任务，因此我们采用二元交叉熵损失函数（第 5.4 节），其原始形式为：

$$
\hat{\boldsymbol{\phi}} = \underset{\boldsymbol{\phi}}{\text{argmin}}\left[\sum_i -(1-y_i)\log\Big[1 - \text{sig}[\text{f}[\mathbf{x}_i, \boldsymbol{\phi}]]\Big] - y_i\log\Big[\text{sig}[\text{f}[\mathbf{x}_i, \boldsymbol{\phi}]]\Big]\right],
\tag{15.2}
$$

其中 $y_i \in \{0, 1\}$ 是标签，$\text{sig}[\bullet]$ 是逻辑 sigmoid 函数（图 5.7）。

在本例中，我们假设真实样本 $\mathbf{x}$ 的标签为 $y = 1$，生成样本 $\mathbf{x}^*$ 的标签为 $y = 0$，因此：

$$
\hat{\boldsymbol{\phi}} = \underset{\boldsymbol{\phi}}{\text{argmin}}\left[\sum_j -\log\Big[1 - \text{sig}[\text{f}[\mathbf{x}_j^*, \boldsymbol{\phi}]]\Big] - \sum_i \log\Big[\text{sig}[\text{f}[\mathbf{x}_i, \boldsymbol{\phi}]]\Big]\right],
\tag{15.3}
$$

其中 $i$ 和 $j$ 分别索引真实样本和生成样本。

现在我们代入生成器的定义 $\mathbf{x}_j^* = \mathbf{g}[\mathbf{z}_j, \boldsymbol{\theta}]$，并注意我们必须关于 $\boldsymbol{\theta}$ 最大化，因为我们希望生成样本被误分类（即具有低的合成可能性或高的负对数似然）：

$$
\hat{\boldsymbol{\theta}} = \underset{\boldsymbol{\theta}}{\text{argmax}}\left[\min_{\boldsymbol{\phi}}\left[\sum_j -\log\Big[1 - \text{sig}[\text{f}[\mathbf{g}[\mathbf{z}_j, \boldsymbol{\theta}], \boldsymbol{\phi}]]\Big] - \sum_i \log\Big[\text{sig}[\text{f}[\mathbf{x}_i, \boldsymbol{\phi}]]\Big]\right]\right].
\tag{15.4}
$$

### 15.1.2 训练 GAN

公式 15.4 是一个比我们之前见过的更复杂的损失函数；判别器参数 $\boldsymbol{\phi}$ 被调整以最小化损失函数，而生成参数 $\boldsymbol{\theta}$ 被调整以最大化损失函数。GAN 训练被描述为一个**极小极大博弈**（minimax game）；生成器试图找到新方法来欺骗判别器，判别器又反过来寻找新方法来区分生成样本和真实样本。从技术上讲，解是一个**纳什均衡**（Nash equilibrium）——优化算法搜索一个同时是一个函数的最小值和另一个函数的最大值的位置。如果训练按计划进行，则收敛后 $\mathbf{g}[\mathbf{z}, \boldsymbol{\theta}]$ 将从与数据相同的分布中采样，而 $\text{sig}[\text{f}[\bullet, \boldsymbol{\phi}]]$ 将处于随机水平（即 0.5）。

为了训练 GAN，我们可以将公式 15.4 分为两个损失函数：

$$
\begin{aligned}
L[\boldsymbol{\phi}] &= \sum_j -\log\Big[1 - \text{sig}[\text{f}[\mathbf{g}[\mathbf{z}_j, \boldsymbol{\theta}], \boldsymbol{\phi}]]\Big] - \sum_i \log\Big[\text{sig}[\text{f}[\mathbf{x}_i, \boldsymbol{\phi}]]\Big] \\
L[\boldsymbol{\theta}] &= \sum_j \log\Big[1 - \text{sig}[\text{f}[\mathbf{g}[\mathbf{z}_j, \boldsymbol{\theta}], \boldsymbol{\phi}]]\Big],
\end{aligned}
\tag{15.5}
$$

其中我们将第二个函数乘以负一以转换为最小化问题，并去掉了不依赖于 $\boldsymbol{\theta}$ 的第二项。最小化第一个损失函数训练判别器。最小化第二个训练生成器。

在每一步中，我们从基础分布中抽取一批潜变量 $\mathbf{z}_j$，并将它们通过生成器创建样本 $\mathbf{x}_j^* = \mathbf{g}[\mathbf{z}_j, \boldsymbol{\theta}]$。然后选择一批真实训练样本 $\mathbf{x}_i$。有了这两批数据，我们就可以对每个损失函数执行一步或多步梯度下降（图 15.2）。

<div align="center">

![图 15.2](/figures/ch15/GanGaussMotivation.png)

</div>

> **图 15.2** GAN 损失函数。潜变量 $\mathbf{z}_j$ 从基础分布中抽取，并通过生成器创建样本 $\mathbf{x}^*$。一批样本 $\{\mathbf{x}_j^*\}$ 和一批真实样本 $\{\mathbf{x}_i\}$ 被传递给判别器，判别器为每个样本分配一个为真实的概率。判别器参数 $\boldsymbol{\phi}$ 被修改以对真实样本赋予高概率、对生成样本赋予低概率。生成器参数 $\boldsymbol{\theta}$ 被修改以"欺骗"判别器，使其对生成样本赋予高概率。

### 15.1.3 深度卷积 GAN

**深度卷积 GAN**（deep convolutional GAN）或 **DCGAN** 是一种早期专门用于生成图像的 GAN 架构（图 15.3）。生成器 $\mathbf{g}[\mathbf{z}, \boldsymbol{\theta}]$ 的输入是一个从均匀分布中采样的 100 维潜变量 $\mathbf{z}$。首先通过线性变换将其映射为具有 1024 个通道的 $4\times4$ 空间表示。接着是四个卷积层，每层使用分数步长卷积（即步长为 0.5 的卷积）将分辨率翻倍。在最后一层，$64\times64\times3$ 的信号通过 tanh 函数生成范围在 $[-1, 1]$ 内的图像 $\mathbf{x}^*$。判别器 $\text{f}[\bullet, \boldsymbol{\phi}]$ 是一个标准的卷积网络，其中最后一个卷积层将大小缩减为 $1\times1$ 并只有一个通道。这个单一数值通过 sigmoid 函数 $\text{sig}[\bullet]$ 产生输出概率。

训练完成后，判别器被丢弃。为了创建新样本，从基础分布中抽取潜变量 $\mathbf{z}$ 并通过生成器。示例结果见图 15.4。

<div align="center">

![图 15.3](/figures/ch15/GanDCGANArch.png)

</div>

> **图 15.3** DCGAN 架构。在生成器中，100 维潜变量 $\mathbf{z}$ 从均匀分布中抽取，通过线性变换映射为具有 1024 个通道的 $4\times4$ 表示。然后通过一系列卷积层逐步上采样表示并减少通道数。最后是一个 tanh 函数，将 $64\times64\times3$ 的表示映射到固定范围以表示图像。判别器由标准卷积网络组成，将输入分类为真实样本或生成样本。

<div align="center">

![图 15.4](/figures/ch15/GANDCGANResults.png)

</div>

> **图 15.4** DCGAN 模型合成的图像。a) 在人脸数据集上训练的 DCGAN 的随机样本。b) 使用 ImageNet 数据库的随机样本（见图 10.15）。c) 从 LSUN 场景理解数据集中抽取的随机样本。改编自 Radford et al. (2015)。

### 15.1.4 训练 GAN 的困难

理论上，GAN 相当简单。然而，GAN 训练起来是出了名地困难。例如，要让 DCGAN 可靠地训练，需要 (i) 在生成器和判别器中都使用步长卷积进行上采样和下采样；(ii) 除了生成器的最后一层和判别器的第一层外，都使用 BatchNorm；(iii) 在判别器中使用 leaky ReLU 激活函数（图 3.13）；(iv) 使用 Adam 优化器但动量系数比通常更低。这是不寻常的。大多数深度学习模型对这类选择相对鲁棒。

一个常见的失败模式是生成器产生看似合理的样本，但这些样本只代表数据的一个子集（例如，对于人脸，它可能永远不会生成有胡须的面孔）。这被称为**模式丢失**（mode dropping）。一个极端版本是生成器完全或基本上忽略潜变量 $\mathbf{z}$，将所有样本坍缩到一个或几个点；这被称为**模式坍缩**（mode collapse）（图 15.5）。

<div align="center">

![图 15.5](/figures/ch15/GANModeCollapse.png)

</div>

> **图 15.5** 模式坍缩。在 LSUN 场景理解数据集上使用与 DCGAN 具有相似参数数量和层数的 MLP 生成器训练的 GAN 合成的图像。样本质量很低，且许多样本相似。改编自 Arjovsky et al. (2017)。

## 15.2 提高稳定性

要理解*为什么* GAN 难以训练，有必要确切地理解损失函数代表*什么*。

### 15.2.1 GAN 损失函数的分析

如果我们将公式 15.5 第一行中的两个求和分别除以真实样本数 $I$ 和生成样本数 $J$，则损失函数可以写成期望的形式：

$$
\begin{aligned}
L[\boldsymbol{\phi}] &= -\frac{1}{J}\sum_{j=1}^{J}\left(\log\Big[1 - \text{sig}[\text{f}[\mathbf{x}_j^*, \boldsymbol{\phi}]]\Big]\right) - \frac{1}{I}\sum_{i=1}^{I}\left(\log\Big[\text{sig}[\text{f}[\mathbf{x}_i, \boldsymbol{\phi}]]\Big]\right) \\
&\approx -\mathbb{E}_{\mathbf{x}^*}\Big[\log\big[1 - \text{sig}[\text{f}[\mathbf{x}^*, \boldsymbol{\phi}]]\big]\Big] - \mathbb{E}_{\mathbf{x}}\Big[\log\big[\text{sig}[\text{f}[\mathbf{x}, \boldsymbol{\phi}]]\big]\Big] \\
&= -\int Pr(\mathbf{x}^*)\log\Big[1 - \text{sig}[\text{f}[\mathbf{x}^*, \boldsymbol{\phi}]]\Big]d\mathbf{x}^* - \int Pr(\mathbf{x})\log\Big[\text{sig}[\text{f}[\mathbf{x}, \boldsymbol{\phi}]]\Big]d\mathbf{x},
\end{aligned}
\tag{15.6}
$$

其中 $Pr(\mathbf{x}^*)$ 是生成样本上的概率分布，$Pr(\mathbf{x})$ 是真实样本上的真实概率分布。

当 $I = J$ 时，对于来源未知的样本 $\tilde{\mathbf{x}}$，最优判别器为：

$$
Pr(\text{real}|\tilde{\mathbf{x}}) = \text{sig}[\text{f}[\tilde{\mathbf{x}}, \boldsymbol{\phi}]] = \frac{Pr(\tilde{\mathbf{x}}|\text{real})}{Pr(\tilde{\mathbf{x}}|\text{generated}) + Pr(\tilde{\mathbf{x}}|\text{real})} = \frac{Pr(\mathbf{x})}{Pr(\mathbf{x}^*) + Pr(\mathbf{x})},
\tag{15.7}
$$

其中在右侧，我们将 $\tilde{\mathbf{x}}$ 与生成分布 $Pr(\mathbf{x}^*)$ 和真实分布 $Pr(\mathbf{x})$ 进行比较。将其代入公式 15.6，我们得到：

$$
\begin{aligned}
L[\boldsymbol{\phi}] &= -\int Pr(\mathbf{x}^*)\log\Big[1 - \text{sig}[\text{f}[\mathbf{x}^*, \boldsymbol{\phi}]]\Big]d\mathbf{x}^* - \int Pr(\mathbf{x})\log\Big[\text{sig}[\text{f}[\mathbf{x}, \boldsymbol{\phi}]]\Big]d\mathbf{x} \\
&= -\int Pr(\mathbf{x}^*)\log\left[\frac{Pr(\mathbf{x}^*)}{Pr(\mathbf{x}^*) + Pr(\mathbf{x})}\right]d\mathbf{x}^* - \int Pr(\mathbf{x})\log\left[\frac{Pr(\mathbf{x})}{Pr(\mathbf{x}^*) + Pr(\mathbf{x})}\right]d\mathbf{x}.
\end{aligned}
\tag{15.8}
$$

忽略加法和乘法常数，这就是合成分布 $Pr(x^*)$ 和真实分布 $Pr(x)$ 之间的 **Jensen-Shannon 散度**（Jensen-Shannon divergence）：

$$
\begin{aligned}
D_{JS}\Big[Pr(\mathbf{x}^*) \| Pr(\mathbf{x})\Big] &\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad\qquad (15.9)\\
&= \frac{1}{2}D_{KL}\left[Pr(\mathbf{x}^*)\left\|\frac{Pr(\mathbf{x}^*) + Pr(\mathbf{x})}{2}\right.\right] + \frac{1}{2}D_{KL}\left[Pr(\mathbf{x})\left\|\frac{Pr(\mathbf{x}^*) + Pr(\mathbf{x})}{2}\right.\right] \\
&= \underbrace{\frac{1}{2}\int Pr(\mathbf{x}^*)\log\left[\frac{2Pr(\mathbf{x}^*)}{Pr(\mathbf{x}^*) + Pr(\mathbf{x})}\right]d\mathbf{x}^*}_{\text{质量}} + \underbrace{\frac{1}{2}\int Pr(\mathbf{x})\log\left[\frac{2Pr(\mathbf{x})}{Pr(\mathbf{x}^*) + Pr(\mathbf{x})}\right]d\mathbf{x}}_{\text{覆盖性}}.
\end{aligned}
$$

其中 $D_{KL}[\bullet\|\bullet]$ 是 **Kullback-Leibler 散度**。

第一项表明，如果在样本密度 $Pr(\mathbf{x}^*)$ 高的地方，混合分布 $(Pr(\mathbf{x}^*) + Pr(\mathbf{x}))/2$ 也具有高概率，则距离会很小。换言之，它惩罚有样本 $\mathbf{x}^*$ 但没有真实样本 $\mathbf{x}$ 的区域；它强制**质量**（quality）。第二项表明，如果在真实密度 $Pr(\mathbf{x})$ 高的地方，混合分布 $(Pr(\mathbf{x}^*) + Pr(\mathbf{x}))/2$ 也具有高概率，则距离会很小。换言之，它惩罚有真实样本但没有生成样本的区域。它强制**覆盖性**（coverage）。参考公式 15.6，我们看到第二项不依赖于生成器，因此生成器不关心覆盖性；它只需要生成一个看起来像训练样本子集的样本就满足了。这就是模式丢失的推定原因。

<div align="center">

![图 15.6](/figures/ch15/GanGradients.png)

</div>

> **图 15.6** GAN 损失函数的问题。如果生成样本（橙色箭头）容易与真实样本（青色箭头）区分，则判别器（sigmoid）在样本位置处的斜率可能非常小；因此，更新生成器参数的梯度可能很小。

### 15.2.2 梯度消失

在上一节中，我们看到当判别器最优时，损失函数最大化生成样本和真实样本之间分布的距离度量。然而，使用概率分布之间的距离作为优化 GAN 的准则存在一个潜在问题。如果概率分布完全不相交，这个距离就是最大的，生成器的任何微小变化都不会减少损失。当我们考虑原始公式时也能看到同样的现象；如果判别器能完美地分离生成样本和真实样本，则对生成数据的微小改变不会改变分类得分（图 15.6）。

不幸的是，生成样本和真实样本的分布可能*确实*是不相交的；生成样本位于一个子空间中，该子空间的大小与潜变量 $\mathbf{z}$ 相同，而真实样本由于产生数据的物理过程也位于一个低维子空间中（图 1.9）。这些子空间之间可能几乎没有重叠，结果就是非常小的梯度或没有梯度。

图 15.7 提供了支持这一假说的经验证据。如果冻结 DCGAN 生成器并反复更新判别器以改善其分类性能，生成器的梯度会下降。简而言之，判别器和生成器的质量之间存在非常微妙的平衡；如果判别器变得太好，生成器的训练更新就会被削弱。

<div align="center">

![图 15.7](/figures/ch15/GanGaussMotivationProb.png)

</div>

> **图 15.7** DCGAN 生成器中的梯度消失。生成器在 1、10 和 25 个 epoch 后被冻结，判别器继续训练。生成器的梯度迅速下降（注意对数刻度）；如果判别器变得太精确，生成器的梯度就会消失。改编自 Arjovsky & Bottou (2017)。

### 15.2.3 Wasserstein 距离

前面几节表明 (i) GAN 损失可以解释为概率分布之间的距离，(ii) 当生成样本太容易与真实样本区分时，该距离的梯度变为零。显而易见的前进方向是选择具有更好性质的距离度量。

**Wasserstein** 距离或（对于离散分布）**推土机距离**（earth mover's distance）是将概率质量从一个分布传输到另一个分布以创建后者所需的工作量。这里，"工作"定义为质量乘以移动距离。这听起来立即更有前景；即使分布不相交，Wasserstein 距离也是良定义的，并且随着分布相互接近而平滑下降。

### 15.2.4 离散分布的 Wasserstein 距离

Wasserstein 距离最容易通过离散分布来理解（图 15.8）。考虑定义在 $K$ 个区间上的分布 $Pr(x = i)$ 和 $q(x = j)$。假设将第一个分布中区间 $i$ 的一个单位质量移动到第二个分布中区间 $j$ 的代价为 $C_{ij}$；这个代价可以是索引之间的绝对差 $|i - j|$。移动的量构成**传输计划**（transport plan），存储在矩阵 $\mathbf{P}$ 中。

Wasserstein 距离定义为：

$$
D_w\Big[Pr(x)\|q(x)\Big] = \min_{\mathbf{P}}\left[\sum_{i,j}P_{ij}\cdot|i-j|\right],
\tag{15.10}
$$

服从约束条件：

$$
\begin{aligned}
\sum_j P_{ij} &= Pr(x = i) &\qquad \text{初始分布 } Pr(x) \\
\sum_i P_{ij} &= q(x = j) &\qquad \text{初始分布 } q(x) \\
P_{ij} &\geq 0 &\qquad \text{非负质量。}
\end{aligned}
\tag{15.11}
$$

换言之，Wasserstein 距离是将一个分布的质量映射到另一个分布的约束最小化问题的解。这不太方便，因为每次我们想计算距离时都必须对元素 $P_{ij}$ 求解这个最小化问题。幸运的是，对于小规模方程组，这是一个容易求解的标准问题。它是**原始形式**（primal form）的**线性规划**（linear programming）问题：

$$
\begin{array}{ll}
\textbf{原始形式} & \textbf{对偶形式} \\
\text{minimize} \quad \mathbf{c}^T\mathbf{p}, & \text{maximize} \quad \mathbf{b}^T\mathbf{f}, \\
\text{such that} \quad \mathbf{Ap} = \mathbf{b} & \text{such that} \quad \mathbf{A}^T\mathbf{f} \leq \mathbf{c} \\
\text{and} \quad \mathbf{p} \geq \mathbf{0} &
\end{array}
$$

其中 $\mathbf{p}$ 包含决定移动质量的向量化元素 $P_{ij}$，$\mathbf{c}$ 包含距离，$\mathbf{Ap} = \mathbf{b}$ 包含初始分布约束，$\mathbf{p} \geq 0$ 确保移动的质量非负。

与所有线性规划问题一样，存在一个具有相同解的等价**对偶问题**（dual problem）。这里，我们关于应用于初始分布的变量 $\mathbf{f}$ 最大化，受限于依赖距离 $\mathbf{c}$ 的约束。该对偶问题的解为：

$$
D_w\Big[Pr(x)\|q(x)\Big] = \max_{\mathbf{f}}\left[\sum_i Pr(x=i)f_i - \sum_j q(x=j)f_j\right],
\tag{15.12}
$$

服从约束条件：

$$
|f_{i+1} - f_i| < 1.
\tag{15.13}
$$

换言之，我们对一组新变量 $\{f_i\}$ 进行优化，其中相邻值的变化不能超过一。

<div align="center">

![图 15.8](/figures/ch15/GANWassersteinDist.png)

</div>

> **图 15.8** Wasserstein 或推土机距离。a) 考虑离散分布 $Pr(x = i)$。b) 我们希望移动概率质量以创建目标分布 $q(x = j)$。c) 传输计划 $\mathbf{P}$ 标识将从 $i$ 移动多少质量到 $j$。例如，青色高亮方块 $p_{54}$ 表示将从 $i = 5$ 移动到 $j = 4$ 的质量。传输计划的元素必须非负，对 $j$ 的求和必须为 $Pr(x = i)$，对 $i$ 的求和必须为 $q(x = j)$。因此 $\mathbf{P}$ 是一个联合概率分布。d) 元素 $i$ 和 $j$ 之间的距离矩阵。最优传输计划 $\mathbf{P}$ 最小化 $\mathbf{P}$ 和距离矩阵的逐元素乘积之和（称为 Wasserstein 距离）。因此，$\mathbf{P}$ 的元素倾向于靠近对角线（距离代价最低处）分布。改编自 Hermann (2017)。

### 15.2.5 连续分布的 Wasserstein 距离

将这些结果推广到连续多维情形，原始形式（公式 15.10）的等价物为：

$$
D_w\Big[Pr(\mathbf{x}), q(\mathbf{x})\Big] = \min_{\pi[\bullet,\bullet]}\left[\iint \pi(\mathbf{x}_1, \mathbf{x}_2)\cdot\|\mathbf{x}_1 - \mathbf{x}_2\|d\mathbf{x}_1 d\mathbf{x}_2\right],
\tag{15.14}
$$

服从与公式 15.11 类似的约束条件，其中传输计划 $\pi(\mathbf{x}_1, \mathbf{x}_2)$ 表示从位置 $\mathbf{x}_1$ 移动到 $\mathbf{x}_2$ 的质量。对偶形式（公式 15.12）的等价物为：

$$
D_w\Big[Pr(\mathbf{x}), q(\mathbf{x})\Big] = \max_{\text{f}[\mathbf{x}]}\left[\int Pr(\mathbf{x})\text{f}[\mathbf{x}]d\mathbf{x} - \int q(\mathbf{x})\text{f}[\mathbf{x}]d\mathbf{x}\right],
\tag{15.15}
$$

服从约束条件：函数 $\text{f}[\mathbf{x}]$ 的 **Lipschitz 常数**（Lipschitz constant）小于一（即函数的绝对梯度小于一）。

### 15.2.6 Wasserstein GAN 损失函数

在神经网络的背景下，我们通过优化神经网络 $\text{f}[\mathbf{x}, \boldsymbol{\phi}]$ 中的参数 $\boldsymbol{\phi}$ 来在函数 $\text{f}[\mathbf{x}]$ 的空间上最大化，并使用生成样本 $\mathbf{x}_j^*$ 和真实样本 $\mathbf{x}_i$ 来近似这些积分：

$$
\begin{aligned}
L[\boldsymbol{\phi}] &= \sum_j \text{f}[\mathbf{x}_j^*, \boldsymbol{\phi}] - \sum_i \text{f}[\mathbf{x}_i, \boldsymbol{\phi}] \\
&= \sum_j \text{f}[\mathbf{g}[\mathbf{z}_j, \boldsymbol{\theta}], \boldsymbol{\phi}] - \sum_i \text{f}[\mathbf{x}_i, \boldsymbol{\phi}],
\end{aligned}
\tag{15.16}
$$

其中我们必须约束神经网络判别器 $\text{f}[\mathbf{x}_i, \boldsymbol{\phi}]$ 在每个位置 $\mathbf{x}$ 处的绝对梯度范数小于一：

$$
\left|\frac{\partial \text{f}[\mathbf{x}, \boldsymbol{\phi}]}{\partial \mathbf{x}}\right| < 1.
\tag{15.17}
$$

实现这一点的一种方法是将判别器权重裁剪到一个小范围内（例如 $\pm 0.01$）。另一种方法是**梯度惩罚 Wasserstein GAN**（gradient penalty Wasserstein GAN）或 **WGAN-GP**，它添加一个随着梯度范数偏离一而增加的正则化项。

## 15.3 渐进式生长、小批量判别和截断

Wasserstein 公式使 GAN 训练更加稳定。然而，生成高质量图像需要更多的技术。我们现在回顾**渐进式生长**（progressive growing）、**小批量判别**（minibatch discrimination）和**截断**（truncation），它们都能改善输出质量。

在**渐进式生长**中（图 15.9），我们首先训练一个合成 $4\times4$ 图像的 GAN，使用与 DCGAN 类似的架构。然后在生成器上添加后续层，对表示进行上采样并执行进一步处理以创建 $8\times8$ 图像。判别器也有额外层添加，使其能够接收更高分辨率的图像并将其分类为生成样本或真实样本。在实践中，更高分辨率的层随时间逐渐"淡入"；最初，更高分辨率的图像是前一结果的上采样版本，通过残差连接传递，新层逐渐接管。这个过程继续创建 $16\times16$ 图像，依此类推。通过这种方式，可以训练出生成非常逼真的高分辨率图像的 GAN。图 15.9d 展示了从同一潜变量在不同阶段生成的递增分辨率的图像。

<div align="center">

![图 15.9](/figures/ch15/GANProgressive.png)

</div>

> **图 15.9** 渐进式生长。a) 生成器最初被训练创建非常小的 $4\times4$ 图像，判别器识别这些图像是合成的还是下采样的真实图像。b) 在这个低分辨率训练终止后，后续层被添加到生成器以生成 $8\times8$ 图像。类似的层也被添加到判别器以进行下采样。c) 这个过程继续创建 $16\times16$ 图像，依此类推。通过这种方式，可以训练出生成非常逼真的高分辨率图像的 GAN。d) 从同一潜变量在不同阶段生成的递增分辨率的图像。改编自 Wolf (2021)，使用 Karras et al. (2018) 的方法。

**小批量判别**确保样本具有足够的多样性，从而有助于防止模式坍缩。这可以通过计算合成数据和真实数据的小批量上的特征统计量来实现。这些统计量可以被总结并作为特征图添加（通常在判别器末端附近）。这允许判别器向生成器发送信号，鼓励合成数据中包含与原始数据集中类似的变化量。

另一个改善生成效果的技巧是**截断**（truncation，图 15.10），即在采样时只选择高概率（即接近均值）的潜变量 $\mathbf{z}$。这减少了样本的变化但提高了质量。精心的归一化和正则化方案也能改善样本质量。使用这些方法的组合，GAN 可以合成多样且逼真的图像（图 15.11）。在潜空间中平滑移动有时也能产生从一个合成图像到另一个的逼真插值（图 15.12）。

<div align="center">

![图 15.10](/figures/ch15/GANTruncation.png)

</div>

> **图 15.10** 截断。通过拒绝距均值超过 $\tau$ 个标准差的潜变量 $\mathbf{z}$ 的样本，可以在 GAN 样本的质量和多样性之间进行权衡。a) 如果阈值较大（$\tau = 2.0$），样本视觉上多样但可能有缺陷。b-c) 随着阈值降低，平均视觉质量提高，但多样性下降。d) 阈值非常小时，样本看起来几乎相同。通过巧妙选择该阈值，可以提高 GAN 结果的平均质量。改编自 Brock et al. (2019)。

<div align="center">

![图 15.11](/figures/ch15/GANProgressiveResults.png)

</div>

> **图 15.11** 方法组合。在 CELEBA-HQ 数据集上训练时，GAN 可以生成逼真的人脸图像，在 LSUN 类别上训练时可以生成更复杂、更多变的物体。改编自 Karras et al. (2018)。

<div align="center">

![图 15.12](/figures/ch15/GANProgressiveInterp.png)

</div>

> **图 15.12** 遍历渐进式 GAN 在 LSUN 汽车上训练的潜空间。在潜空间中移动产生平滑变化的汽车图像。这通常只在短轨迹上有效；最终，潜变量会移动到产生不逼真图像的位置。改编自 Karras et al. (2018)。

## 15.4 条件生成

GAN 生成逼真的图像，但不指定其属性：我们无法选择发色、种族或年龄，也无需为每种特征组合训练单独的 GAN。**条件生成**（conditional generation）模型为我们提供了这种控制能力。

### 15.4.1 条件 GAN

**条件 GAN**（conditional GAN）将属性向量 $\mathbf{c}$ 同时传递给生成器和判别器，它们现在分别写作 $\mathbf{g}[\mathbf{z}, \mathbf{c}, \boldsymbol{\theta}]$ 和 $\text{f}[\mathbf{x}, \mathbf{c}, \boldsymbol{\phi}]$。生成器旨在将潜变量 $\mathbf{z}$ 转换为具有正确属性 $\mathbf{c}$ 的数据样本 $\mathbf{x}$。判别器的目标是区分 (i) 具有目标属性的生成样本和 (ii) 具有真实属性的真实样本（图 15.13a）。

对于生成器，属性 $\mathbf{c}$ 可以附加到潜向量 $\mathbf{z}$ 上。对于判别器，如果数据是一维的，可以将其附加到输入。如果数据是图像，属性可以线性变换为二维表示，并作为额外通道附加到判别器输入或其某个中间隐藏层。

### 15.4.2 辅助分类器 GAN

**辅助分类器 GAN**（auxiliary classifier GAN）或 **ACGAN** 通过要求判别器正确预测属性来简化条件生成（图 15.13b）。对于具有 $C$ 个类别的离散属性，判别器以真实/合成图像为输入，有 $C + 1$ 个输出；第一个通过 sigmoid 函数预测样本是生成的还是真实的。其余输出通过 softmax 函数预测数据属于 $C$ 个类别中每个类别的概率。用该方法训练的网络可以从 ImageNet 合成多个类别（图 15.14）。

<div align="center">

![图 15.13](/figures/ch15/GanConditional.png)

</div>

> **图 15.13** 条件生成。a) 条件 GAN 的生成器还接收描述图像某些方面的属性向量 $\mathbf{c}$。和往常一样，判别器接收真实样本或生成样本，但现在它也接收属性向量；这鼓励样本既逼真又与属性兼容。b) 辅助分类器 GAN (ACGAN) 的生成器接受离散属性变量。判别器必须同时 (i) 确定其输入是真实的还是合成的，以及 (ii) 正确识别类别。c) InfoGAN 将潜变量分为噪声 $\mathbf{z}$ 和未指定的随机属性 $\mathbf{c}$。判别器必须区分其输入是否为真实的，并重建这些属性。在实践中，这意味着变量 $\mathbf{c}$ 对应于数据具有现实世界可解释性的显著方面（即潜空间是解缠的）。

<div align="center">

![图 15.14](/figures/ch15/GANACGANResults.png)

</div>

> **图 15.14** 辅助分类器 GAN。生成器接受类别标签和潜变量。判别器必须同时识别数据点是否真实*且*预测类别标签。该模型在十个 ImageNet 类别上训练。从左到右：帝王蝶、金翅雀、雏菊、红脚鹬和灰鲸的生成示例。改编自 Odena et al. (2017)。

### 15.4.3 InfoGAN

条件 GAN 和 ACGAN 都生成具有预定属性的样本。相比之下，**InfoGAN**（图 15.13c）尝试自动识别重要属性。生成器接受由随机噪声变量 $\mathbf{z}$ 和*随机*属性变量 $\mathbf{c}$ 组成的向量。判别器既预测图像是否为真实的，又估计属性变量。

其洞察在于，可解释的现实世界特征应该最容易预测，因此会被表示在属性变量 $\mathbf{c}$ 中。$\mathbf{c}$ 中的属性可以是离散的（使用二元或多类交叉熵损失）或连续的（使用最小二乘损失）。离散变量识别数据中的类别，连续变量识别渐变的变化模式（图 15.15）。

<div align="center">

![图 15.15](/figures/ch15/GanInfoGAN.png)

</div>

> **图 15.15** InfoGAN 用于 MNIST。a) MNIST 数据库的训练样本，由 $28\times28$ 像素的手写数字图像组成。b) 第一个属性 $c_1$ 是离散的，有 10 个类别；每列展示用其中一个类别生成的样本。InfoGAN 恢复了十个数字。属性向量 $c_2$ 和 $c_3$ 是连续的。c) 从左到右，每列代表不同的 $c_2$ 值，保持其他潜变量不变。该属性似乎对应于字符的方向。d) 第三个属性似乎对应于笔画的粗细。改编自 Chen et al. (2016b)。

## 15.5 图像翻译

虽然对抗判别器最初是在 GAN 的场景中用于生成随机样本，但它也可以用作在将一个数据示例翻译为另一个的任务中偏向真实性的先验。这最常见于图像，比如将灰度图像翻译为彩色、将噪声图像翻译为干净图像、将模糊图像翻译为清晰图像，或将草图翻译为逼真照片。

本节讨论三种使用不同数量标注数据的图像翻译模型。Pix2Pix 模型使用前/后配对图像进行训练。带有对抗损失的模型使用前/后配对作为主模型，但同时利用判别器中的非配对"后"图像。CycleGAN 模型使用非配对图像。

### 15.5.1 Pix2Pix

Pix2Pix 模型（图 15.16）是一个网络 $\hat{\mathbf{x}} = \mathbf{g}[\mathbf{c}, \boldsymbol{\theta}]$，使用 U-Net（图 11.10）将一幅图像 $\mathbf{c}$ 映射为不同风格的图像，参数为 $\boldsymbol{\theta}$。一个典型的用例是着色，其中输入 $\mathbf{c}$ 是灰度图像，输出 $\mathbf{g}[\mathbf{c}, \boldsymbol{\theta}]$ 是彩色图像。输出应该与输入相似，这通过惩罚输入 $\mathbf{c}$ 和真实输出 $\mathbf{x}$ 之间的 $\ell_1$ 范数 $\|\mathbf{x} - \mathbf{g}[\mathbf{c}, \boldsymbol{\theta}]\|_1$ 的**内容损失**（content loss）来鼓励。

然而，输出图像也应该看起来像输入的逼真转换。这通过使用对抗判别器 $\text{f}[\mathbf{c}, \mathbf{x}, \boldsymbol{\phi}]$ 来鼓励，该判别器接收前后图像 $\mathbf{c}$ 和 $\mathbf{x}$。在每一步中，判别器试图区分真实的前/后配对和前/合成配对。在这些能被成功区分的范围内，向 U-Net 提供反馈信号以使其输出更逼真。由于内容损失确保了大尺度图像结构正确，判别器主要需要确保局部纹理是合理的。为此，**PatchGAN** 损失基于纯卷积分类器。在最后一层，每个隐藏单元指示其感受野内的区域是真实的还是合成的。这些响应被平均以提供最终输出。

可以将该模型理解为一个条件 GAN，其中 U-Net 是生成器，且以图像而非标签为条件。但注意，U-Net 输入不包含噪声，因此不是传统意义上的"生成器"。有趣的是，原始作者尝试过在 U-Net 输入 $\mathbf{c}$ 之外添加噪声 $\mathbf{z}$。然而，网络只是学会了忽略它。

<div align="center">

![图 15.16](/figures/ch15/GANPix2Pix_C.png)

</div>

> **图 15.16** Pix2Pix 模型。a) 模型使用 U-Net（见图 11.10）将输入图像翻译为不同风格的预测。在此例中，将灰度图像映射为合理的彩色版本。U-Net 用两个损失训练。首先，内容损失鼓励输出图像与输入图像具有相似的结构。其次，对抗损失鼓励灰度/彩色图像对在每个局部区域与真实配对不可区分。该框架可以适应许多任务，包括 b) 将地图翻译为卫星图像，c) 将草图转换为逼真照片，d) 着色，e) 将标签图转换为逼真的建筑立面。改编自 Isola et al. (2017)。

### 15.5.2 对抗损失

Pix2Pix 模型中的判别器试图区分图像翻译任务中的前/后配对是否合理。这有一个缺点：我们需要真实的前/后配对来利用判别器损失。幸运的是，有一种更简单的方法可以在监督学习中利用对抗判别器的能力，而无需额外的标注训练数据。

**对抗损失**（adversarial loss）在判别器能区分监督网络的输出和其输出域中的真实样本时添加惩罚。相应地，监督模型改变其预测以减小这个惩罚。这可以在整个输出的尺度上进行，也可以在补丁级别进行，如 Pix2Pix 算法。这有助于改善复杂结构化输出的**真实性**（realism）。然而，它不一定在原始损失函数方面带来更好的解。

**超分辨率 GAN**（super-resolution GAN）或 **SRGAN** 使用了这种方法（图 15.17）。主模型由带残差连接的卷积网络组成，接收低分辨率图像并通过上采样层将其转换为高分辨率图像。网络用三个损失训练。内容损失衡量输出和真实高分辨率图像之间的平方差。**VGG 损失**或**感知损失**（perceptual loss）将合成和真实输出通过 VGG 网络并衡量其激活之间的平方差。这鼓励图像在语义上与目标相似。最后，对抗损失使用判别器来区分这是真实的高分辨率图像还是上采样的图像。这鼓励输出与真实样本不可区分。

<div align="center">

![图 15.17](/figures/ch15/GanSuperRes.png)

</div>

> **图 15.17** 超分辨率生成对抗网络（SRGAN）。a) 训练带残差连接的卷积网络将图像分辨率提高四倍。模型有鼓励内容接近真实高分辨率图像的损失。然而，它还包括对抗损失，惩罚能被真实高分辨率图像区分的结果。b) 使用双三次插值上采样的图像。c) 使用 SRGAN 上采样的图像。d) 使用双三次插值上采样的图像。e) 使用 SRGAN 上采样的图像。改编自 Ledig et al. (2017)。

### 15.5.3 CycleGAN

对抗损失假设主监督网络有标注的前/后图像。**CycleGAN** 处理的是我们有两组不同风格的数据但*没有*匹配配对的情况。例如将照片转换为莫奈的艺术风格。存在许多照片和许多莫奈画作，但它们之间没有对应关系。CycleGAN 利用的想法是，将图像在一个方向转换（例如照片→莫奈）然后再转换回来应该恢复原始图像。

CycleGAN 的损失函数是三个损失的加权和（图 15.18）。内容损失鼓励前后图像相似，基于 $\ell_1$ 范数。对抗损失使用判别器来鼓励输出与目标域的真实样本不可区分。最后，**循环一致性损失**（cycle-consistency loss）鼓励映射是可逆的。这里同时训练两个模型。一个从第一个域映射到第二个域，另一个从相反方向映射。如果翻译后的图像能够成功地被翻译回原始域中的图像，则循环一致性损失会很低。该模型结合这三个损失来训练网络将图像从一种风格翻译到另一种风格，以及反向翻译。

<div align="center">

![图 15.18](/figures/ch15/GANCycleGAN.png)

</div>

> **图 15.18** CycleGAN。两个模型同时训练。第一个 $\mathbf{c}' = \mathbf{g}[\mathbf{c}_j, \boldsymbol{\theta}]$ 将第一种风格（马）的图像 $\mathbf{c}$ 翻译为第二种风格（斑马）的图像 $\mathbf{c}'$。第二个模型 $\mathbf{c} = \mathbf{g}'[\mathbf{c}', \boldsymbol{\theta}]$ 学习相反的映射。循环一致性损失惩罚两个模型不能成功地将图像转换到另一个域再转换回来的情况。此外，两个对抗损失鼓励翻译后的图像看起来像目标域的逼真样本（此处仅展示斑马方向）。两个内容损失鼓励每次映射前后图像的细节和布局相似（即斑马与马处于相同的位置和姿势，背景相同，反之亦然）。改编自 Zhu et al. (2017)。

## 15.6 StyleGAN

StyleGAN 是一种更现代的 GAN，它将数据集中的变化分解为有意义的组件，每个组件由潜变量的一个子集控制。特别是，StyleGAN 在不同尺度上控制输出图像，并将风格与噪声分离。对于人脸图像，大尺度变化包括脸型和头部姿势，中尺度变化包括面部特征的形状和细节，细尺度变化包括发色和肤色。风格组件代表对人类来说显著的图像方面，而噪声方面代表不重要的变化，如头发的精确位置、胡茬、雀斑或皮肤毛孔。

到目前为止我们看到的 GAN 都从一个从标准基础分布中采样的潜变量 $\mathbf{z}$ 开始。它通过一系列卷积层来产生输出图像。然而，生成器的潜变量输入可以 (i) 在架构中的不同点引入，(ii) 以不同方式修改这些点的当前表示。StyleGAN 巧妙地做出这些选择以控制尺度并将风格与噪声分离（图 15.19）。

StyleGAN 的主要生成分支从一个学习的常量 $4\times4$ 表示（具有 512 个通道）开始。通过一系列卷积层逐步上采样表示以生成最终分辨率的图像。在每个尺度上引入两组代表风格和噪声的随机潜变量；越接近输出，它们代表越精细的尺度细节。

代表噪声的潜变量是独立采样的高斯向量 $\mathbf{z}_1, \mathbf{z}_2 \ldots$，在主生成管道中每次卷积操作后以加性方式注入。它们与添加点处的主表示具有相同的空间大小，但乘以学习的逐通道缩放因子 $\boldsymbol{\psi}_1, \boldsymbol{\psi}_2 \ldots$，因此对每个通道的贡献不同。随着网络分辨率的增加，该噪声贡献于更细的尺度。

代表风格的潜变量从 $1\times1\times512$ 的噪声张量开始，通过七层全连接网络创建一个中间变量 $\mathbf{w}$。这允许网络去相关风格的各个方面，使得 $\mathbf{w}$ 的每个维度可以表示一个独立的现实世界因素，如头部姿势或发色。这个变量 $\mathbf{w}$ 被线性变换为 $2\times1\times512$ 的张量 $\mathbf{y}$，用于在噪声添加后设置主分支中表示的逐通道均值和方差。这被称为**自适应实例归一化**（adaptive instance normalization，图 11.14e）。一系列向量 $\mathbf{y}_1, \mathbf{y}_2, \ldots$ 以这种方式在主分支的多个不同点注入，因此相同的风格在不同尺度上产生贡献。图 15.20 展示了在不同尺度上操纵风格和噪声向量的示例。

<div align="center">

![图 15.19](/figures/ch15/GanStyleGANArch.png)

</div>

> **图 15.19** StyleGAN。主管道（中间行）从一个常量学习表示（灰色方框）开始。通过一系列卷积层逐步上采样以创建输出。噪声（顶行）通过周期性地添加带有逐通道缩放 $\boldsymbol{\psi}_\bullet$ 的高斯变量 $\mathbf{z}_\bullet$ 在不同尺度上添加。高斯风格变量 $\mathbf{z}$ 通过全连接网络创建中间变量 $\mathbf{w}$（底行）。它用于在管道中不同点设置每个通道的均值和方差。

<div align="center">

![图 15.20](/figures/ch15/GANStyleGANResults_C.png)

</div>

> **图 15.20** StyleGAN 结果。前四列展示了在不同尺度上风格的系统变化。第五列展示了增加噪声幅度的效果。最后两列展示了在两个不同尺度上使用不同噪声向量的效果。

## 15.7 总结

GAN 学习一个生成器网络，将随机噪声转换为与训练集不可区分的数据。为此，使用一个试图区分真实样本和生成样本的判别器网络来训练生成器。然后更新生成器，使其创建的数据被判别器识别为更"真实"的。这一想法的原始公式有一个缺陷，即当容易判断样本是真实的还是生成的时，训练信号很弱。这导致了 Wasserstein GAN，它提供了更一致的训练信号。

我们回顾了用于生成图像的卷积 GAN 以及一系列改善生成图像质量的技巧，包括渐进式生长、小批量判别和截断。条件 GAN 架构引入了一个辅助向量，允许控制输出（例如物体类别的选择）。图像翻译任务保留了图像形式的条件信息，但不使用随机噪声。GAN 判别器现在作为一个偏向"逼真"外观图像的额外损失项。最后，我们描述了 StyleGAN，它将噪声在不同尺度上策略性地注入生成器，以控制风格和噪声。
