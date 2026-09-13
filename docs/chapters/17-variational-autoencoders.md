# 第17章 变分自编码器

生成对抗网络学习一种机制，生成与训练样本 $\{\mathbf{x}_i\}$ 不可区分的新样本。相比之下，与归一化流类似，**变分自编码器**（variational autoencoders），简称 **VAE**，是**概率生成模型**；它们旨在学习数据上的分布 $Pr(\mathbf{x})$（见图 14.2）。训练完成后，可以从该分布中抽取（生成）样本。然而，VAE 的性质意味着不幸的是*不能*精确地评估新样本 $\mathbf{x}^*$ 的概率。

常见的说法是把 VAE 当作 $Pr(\mathbf{x})$ 的模型本身来讨论，但这实际上是有误导性的；VAE 是一种神经架构，旨在帮助*学习* $Pr(\mathbf{x})$ 的模型。$Pr(\mathbf{x})$ 的最终模型既不包含"变分"部分，也不包含"自编码器"部分，或许更适合称为**非线性潜变量模型**（nonlinear latent variable model）。

本章首先介绍一般的潜变量模型，然后考虑非线性潜变量模型的特定情形。我们将看到，该模型的最大似然学习并不直观。然而，可以定义似然的一个下界，VAE 架构使用蒙特卡罗（采样）方法来近似这个下界。本章最后介绍 VAE 的若干应用。

## 17.1 潜变量模型

潜变量模型采用一种间接方法来描述多维变量 $\mathbf{x}$ 上的概率分布 $Pr(\mathbf{x})$。它不是直接写出 $Pr(\mathbf{x})$ 的表达式，而是对数据 $\mathbf{x}$ 和一个不可观测的**隐变量**（hidden variable）或**潜变量**（latent variable）$\mathbf{z}$ 的联合分布 $Pr(\mathbf{x},\mathbf{z})$ 建模。然后将 $Pr(\mathbf{x})$ 的概率描述为该联合概率的**边缘化**（marginalization），即：

$$
Pr(\mathbf{x}) = \int Pr(\mathbf{x},\mathbf{z})d\mathbf{z}.
\tag{17.1}
$$

通常，联合概率 $Pr(\mathbf{x},\mathbf{z})$ 使用条件概率的规则分解为数据关于潜变量的**似然**（likelihood）项 $Pr(\mathbf{x}|\mathbf{z})$ 和**先验**（prior）$Pr(\mathbf{z})$：

$$
Pr(\mathbf{x}) = \int Pr(\mathbf{x}|\mathbf{z})Pr(\mathbf{z})d\mathbf{z}.
\tag{17.2}
$$

这是描述 $Pr(\mathbf{x})$ 的一种相当间接的方法，但它之所以有用，是因为 $Pr(\mathbf{x}|\mathbf{z})$ 和 $Pr(\mathbf{z})$ 的相对简单的表达式就能定义复杂的分布 $Pr(\mathbf{x})$。

### 17.1.1 示例：高斯混合模型

在一维高斯混合模型（图 17.1a）中，潜变量 $z$ 是离散的，先验 $Pr(z)$ 是一个分类分布（图 5.9），对每个可能的值有一个概率 $\lambda_n$。给定潜变量 $z$ 取值为 $n$ 时，数据 $x$ 的似然 $Pr(x|z=n)$ 服从均值为 $\mu_n$、方差为 $\sigma_n^2$ 的正态分布：

$$
\begin{aligned}
Pr(z=n) &= \lambda_n \\
Pr(x|z=n) &= \text{Norm}_x\left[\mu_n, \sigma_n^2\right].
\end{aligned}
\tag{17.3}
$$

如等式 17.2 所述，概率 $Pr(x)$ 由对潜变量 $z$ 的边缘化给出（图 17.1b）。这里，潜变量是离散的，所以对其可能取值求和：

$$
\begin{aligned}
Pr(x) &= \sum_{n=1}^{N} Pr(x, z=n) \\
&= \sum_{n=1}^{N} Pr(x|z=n) \cdot Pr(z=n) \\
&= \sum_{n=1}^{N} \lambda_n \cdot \text{Norm}_x\left[\mu_n, \sigma_n^2\right].
\end{aligned}
\tag{17.4}
$$

通过简单的似然和先验表达式，我们描述了一个复杂的多模态概率分布。

> **图 17.1** 高斯混合模型（MoG）。a) MoG 将一个复杂的概率分布（青色曲线）描述为高斯分量（虚线曲线）的加权和。b) 这个和是连续观测数据 $x$ 与离散潜变量 $z$ 之间的联合密度 $Pr(x,z)$ 的边缘化。

## 17.2 非线性潜变量模型

在非线性潜变量模型中，数据 $\mathbf{x}$ 和潜变量 $\mathbf{z}$ 都是连续且多维的。先验 $Pr(\mathbf{z})$ 是一个标准多元正态分布：

$$
Pr(\mathbf{z}) = \text{Norm}_{\mathbf{z}}[\mathbf{0}, \mathbf{I}].
\tag{17.5}
$$

似然 $Pr(\mathbf{x}|\mathbf{z},\boldsymbol{\phi})$ 也是正态分布；其均值是潜变量的一个非线性函数 $\mathbf{f}[\mathbf{z},\boldsymbol{\phi}]$，协方差是球形的 $\sigma^2\mathbf{I}$：

$$
Pr(\mathbf{x}|\mathbf{z},\boldsymbol{\phi}) = \text{Norm}_{\mathbf{x}}\left[\mathbf{f}[\mathbf{z},\boldsymbol{\phi}], \sigma^2\mathbf{I}\right].
\tag{17.6}
$$

函数 $\mathbf{f}[\mathbf{z},\boldsymbol{\phi}]$ 由一个参数为 $\boldsymbol{\phi}$ 的深度网络描述。潜变量 $\mathbf{z}$ 的维度低于数据 $\mathbf{x}$。模型 $\mathbf{f}[\mathbf{z},\boldsymbol{\phi}]$ 描述数据的重要方面，其余未建模的方面归因于噪声 $\sigma^2\mathbf{I}$。

数据概率 $Pr(\mathbf{x}|\boldsymbol{\phi})$ 通过对潜变量 $\mathbf{z}$ 边缘化获得：

$$
\begin{aligned}
Pr(\mathbf{x}|\boldsymbol{\phi}) &= \int Pr(\mathbf{x},\mathbf{z}|\boldsymbol{\phi})d\mathbf{z} \\
&= \int Pr(\mathbf{x}|\mathbf{z},\boldsymbol{\phi}) \cdot Pr(\mathbf{z})d\mathbf{z} \\
&= \int \text{Norm}_{\mathbf{x}}\left[\mathbf{f}[\mathbf{z},\boldsymbol{\phi}], \sigma^2\mathbf{I}\right] \cdot \text{Norm}_{\mathbf{z}}\left[\mathbf{0}, \mathbf{I}\right] d\mathbf{z}.
\end{aligned}
\tag{17.7}
$$

这可以看作球形高斯的无限加权和（即无限混合），其中权重是 $Pr(\mathbf{z})$，均值是网络输出 $\mathbf{f}[\mathbf{z},\boldsymbol{\phi}]$（图 17.2）。

> **图 17.2** 非线性潜变量模型。一个复杂的二维密度 $Pr(\mathbf{x})$（右）是通过对联合分布 $Pr(\mathbf{x},z)$（左）在潜变量 $z$ 上的边缘化来创建的；为了创建 $Pr(\mathbf{x})$，我们对 $z$ 维度上的三维体积进行积分。对于每个 $z$，$\mathbf{x}$ 上的分布是一个球形高斯（展示了两个切片），其均值 $\mathbf{f}[z,\boldsymbol{\phi}]$ 是 $z$ 的非线性函数且依赖于参数 $\boldsymbol{\phi}$。分布 $Pr(\mathbf{x})$ 是这些高斯的加权和。

### 17.2.1 生成

可以使用**祖先采样**（ancestral sampling）生成新样本 $\mathbf{x}^*$（图 17.3）。我们从先验 $Pr(\mathbf{z})$ 中抽取 $\mathbf{z}^*$，并将其通过网络 $\mathbf{f}[\mathbf{z}^*,\boldsymbol{\phi}]$ 来计算似然 $Pr(\mathbf{x}|\mathbf{z}^*,\boldsymbol{\phi})$（等式 17.6）的均值，然后从中抽取 $\mathbf{x}^*$。由于先验和似然都是正态分布，因此这很直观。

> **图 17.3** 从非线性潜变量模型生成。a) 从潜变量的先验概率 $Pr(z)$ 中抽取样本 $z^*$。b) 从 $Pr(\mathbf{x}|z^*,\boldsymbol{\phi})$ 中抽取样本 $\mathbf{x}^*$。这是一个球形高斯，其均值是 $z^*$ 的非线性函数 $\mathbf{f}[\bullet, \boldsymbol{\phi}]$ 且方差固定为 $\sigma^2\mathbf{I}$。c) 如果重复多次这个过程，就恢复了密度 $Pr(\mathbf{x}|\boldsymbol{\phi})$。

## 17.3 训练

为了训练模型，我们最大化训练数据集 $\{\mathbf{x}_i\}_{i=1}^{I}$ 上关于模型参数的对数似然。为简单起见，我们假设似然表达式中的方差项 $\sigma^2$ 已知，专注于学习 $\boldsymbol{\phi}$：

$$
\hat{\boldsymbol{\phi}} = \underset{\boldsymbol{\phi}}{\text{argmax}}\left[\sum_{i=1}^{I}\log\left[Pr(\mathbf{x}_i|\boldsymbol{\phi})\right]\right],
\tag{17.8}
$$

其中：

$$
Pr(\mathbf{x}_i|\boldsymbol{\phi}) = \int \text{Norm}_{\mathbf{x}_i}\left[\mathbf{f}[\mathbf{z},\boldsymbol{\phi}], \sigma^2\mathbf{I}\right] \cdot \text{Norm}_{\mathbf{z}}\left[\mathbf{0}, \mathbf{I}\right] d\mathbf{z}.
\tag{17.9}
$$

不幸的是，这个积分是不可处理的（intractable）。对于该积分不存在封闭形式的表达式，也没有简便方法对特定值 $\mathbf{x}$ 进行求值。

### 17.3.1 证据下界（ELBO）

为了取得进展，我们定义对数似然的一个**下界**（lower bound）。这是一个函数，对于给定的 $\boldsymbol{\phi}$ 值，它始终小于或等于对数似然，并且还依赖于另一组参数 $\boldsymbol{\theta}$。最终，我们将构建一个网络来计算这个下界并对其进行优化。为了定义这个下界，我们需要 **Jensen 不等式**（Jensen's inequality）。

### 17.3.2 Jensen 不等式

Jensen 不等式指出，**凹函数**（concave function）$g[\bullet]$ 作用于数据 $y$ 的期望时，大于或等于函数作用于数据后的期望：

$$
g\left[\mathbb{E}[y]\right] \geq \mathbb{E}\left[g[y]\right].
\tag{17.10}
$$

在此情形下，凹函数是对数函数，因此有：

$$
\log\left[\mathbb{E}[y]\right] \geq \mathbb{E}\left[\log[y]\right],
\tag{17.11}
$$

或者将期望写成完整表达式：

$$
\log\left[\int Pr(y)y\,dy\right] \geq \int Pr(y)\log[y]\,dy.
\tag{17.12}
$$

这在图 17.4-17.5 中有所探讨。事实上，稍微更一般的陈述也成立：

$$
\log\left[\int Pr(y)h[y]\,dy\right] \geq \int Pr(y)\log[h[y]]\,dy.
\tag{17.13}
$$

其中 $h[y]$ 是 $y$ 的函数。这是因为 $h[y]$ 是另一个具有新分布的随机变量。由于我们从未指定 $Pr(y)$，因此该关系仍然成立。

> **图 17.4** Jensen 不等式（离散情形）。对数函数（黑色曲线）是一个凹函数；你可以在曲线上任意两点之间画一条直线，这条直线始终在曲线下方。因此，对数函数上六个点的任何凸组合（正权重且和为一的加权和）都必须落在曲线下方的灰色区域中。这里，我们对这些点等权平均，得到青色点。由于该点在曲线下方，$\log[\mathbb{E}[y]] > \mathbb{E}[\log[y]]$。

> **图 17.5** Jensen 不等式（连续情形）。对于凹函数，计算分布 $Pr(y)$ 的期望并将其通过函数，得到的结果大于或等于先将变量 $y$ 通过函数再计算新变量的期望。对于对数函数，有 $\log[\mathbb{E}[y]] \geq \mathbb{E}[\log[y]]$。

### 17.3.3 推导下界

我们现在使用 Jensen 不等式来推导对数似然的下界。首先，将对数似然乘以并除以关于潜变量的任意概率分布 $q(\mathbf{z})$：

$$
\begin{aligned}
\log[Pr(\mathbf{x}|\boldsymbol{\phi})] &= \log\left[\int Pr(\mathbf{x},\mathbf{z}|\boldsymbol{\phi})d\mathbf{z}\right] \\
&= \log\left[\int q(\mathbf{z})\frac{Pr(\mathbf{x},\mathbf{z}|\boldsymbol{\phi})}{q(\mathbf{z})}d\mathbf{z}\right],
\end{aligned}
\tag{17.14}
$$

然后对对数函数使用 Jensen 不等式（等式 17.12）来求下界：

$$
\log\left[\int q(\mathbf{z})\frac{Pr(\mathbf{x},\mathbf{z}|\boldsymbol{\phi})}{q(\mathbf{z})}d\mathbf{z}\right] \geq \int q(\mathbf{z})\log\left[\frac{Pr(\mathbf{x},\mathbf{z}|\boldsymbol{\phi})}{q(\mathbf{z})}\right]d\mathbf{z},
\tag{17.15}
$$

其中右边的项称为**证据下界**（evidence lower bound）或 **ELBO**。之所以得此名称，是因为 $Pr(\mathbf{x}|\boldsymbol{\phi})$ 在贝叶斯法则（等式 17.19）的语境中被称为**证据**（evidence）。

实际上，分布 $q(\mathbf{z})$ 有参数 $\boldsymbol{\theta}$，因此 ELBO 可以写为：

$$
\text{ELBO}[\boldsymbol{\theta},\boldsymbol{\phi}] = \int q(\mathbf{z}|\boldsymbol{\theta})\log\left[\frac{Pr(\mathbf{x},\mathbf{z}|\boldsymbol{\phi})}{q(\mathbf{z}|\boldsymbol{\theta})}\right]d\mathbf{z}.
\tag{17.16}
$$

为了学习非线性潜变量模型，我们将这个量作为 $\boldsymbol{\phi}$ 和 $\boldsymbol{\theta}$ 的函数来最大化。计算这个量的神经架构就是 VAE。

> **图 17.6** 证据下界（ELBO）。目标是最大化关于参数 $\boldsymbol{\phi}$ 的对数似然 $\log[Pr(\mathbf{x}|\boldsymbol{\phi})]$（黑色曲线）。ELBO 是一个处处位于对数似然下方的函数。它是 $\boldsymbol{\phi}$ 和第二组参数 $\boldsymbol{\theta}$ 的函数。固定 $\boldsymbol{\theta}$，我们得到 $\boldsymbol{\phi}$ 的一条曲线（两条彩色曲线对应 $\boldsymbol{\theta}$ 的不同值）。因此，我们可以通过改进 a) 新参数 $\boldsymbol{\theta}$（从彩色曲线移到彩色曲线）或 b) 原参数 $\boldsymbol{\phi}$（沿当前彩色曲线移动）来增加对数似然。

## 17.4 ELBO 的性质

初次遇到 ELBO 时，它是一个有些神秘的对象，因此我们现在提供一些关于其性质的直觉。考虑原始的数据对数似然是参数 $\boldsymbol{\phi}$ 的函数，我们想要找到其最大值。对于任意固定的 $\boldsymbol{\theta}$，ELBO 仍然是参数的函数，但必须位于原始似然函数的下方。当我们改变 $\boldsymbol{\theta}$ 时，我们修改了这个函数，根据选择不同，下界可能靠近或远离对数似然。当我们改变 $\boldsymbol{\phi}$ 时，我们沿着下界函数移动（图 17.6）。

### 17.4.1 下界的紧致性

当对于固定的 $\boldsymbol{\phi}$ 值，ELBO 与对数似然函数重合时，下界是**紧致的**（tight）。为了找到使下界紧致的分布 $q(\mathbf{z}|\boldsymbol{\theta})$，我们利用条件概率的定义来分解 ELBO 中对数项的分子：

$$
\begin{aligned}
\text{ELBO}[\boldsymbol{\theta},\boldsymbol{\phi}] &= \int q(\mathbf{z}|\boldsymbol{\theta})\log\left[\frac{Pr(\mathbf{x},\mathbf{z}|\boldsymbol{\phi})}{q(\mathbf{z}|\boldsymbol{\theta})}\right]d\mathbf{z} \\
&= \int q(\mathbf{z}|\boldsymbol{\theta})\log\left[\frac{Pr(\mathbf{z}|\mathbf{x},\boldsymbol{\phi})Pr(\mathbf{x}|\boldsymbol{\phi})}{q(\mathbf{z}|\boldsymbol{\theta})}\right]d\mathbf{z} \\
&= \int q(\mathbf{z}|\boldsymbol{\theta})\log\left[Pr(\mathbf{x}|\boldsymbol{\phi})\right]d\mathbf{z} + \int q(\mathbf{z}|\boldsymbol{\theta})\log\left[\frac{Pr(\mathbf{z}|\mathbf{x},\boldsymbol{\phi})}{q(\mathbf{z}|\boldsymbol{\theta})}\right]d\mathbf{z} \\
&= \log\left[Pr(\mathbf{x}|\boldsymbol{\phi})\right] + \int q(\mathbf{z}|\boldsymbol{\theta})\log\left[\frac{Pr(\mathbf{z}|\mathbf{x},\boldsymbol{\phi})}{q(\mathbf{z}|\boldsymbol{\theta})}\right]d\mathbf{z} \\
&= \log\left[Pr(\mathbf{x}|\boldsymbol{\phi})\right] - \text{D}_{KL}\left[q(\mathbf{z}|\boldsymbol{\theta})\,\|\,Pr(\mathbf{z}|\mathbf{x},\boldsymbol{\phi})\right].
\end{aligned}
\tag{17.17}
$$

其中，第三行和第四行之间，第一个积分消失了，因为 $\log[Pr(\mathbf{x}|\boldsymbol{\phi})]$ 不依赖于 $\mathbf{z}$，且概率分布 $q(\mathbf{z}|\boldsymbol{\theta})$ 的积分为一。在最后一行，我们使用了 **Kullback-Leibler（KL）散度**的定义。

这个等式表明，ELBO 是原始对数似然减去 KL 散度 $\text{D}_{KL}[q(\mathbf{z}|\boldsymbol{\theta})\|Pr(\mathbf{z}|\mathbf{x},\boldsymbol{\phi})]$。KL 散度衡量分布之间的"距离"，只能取非负值。因此 ELBO 是 $\log[Pr(\mathbf{x}|\boldsymbol{\phi})]$ 的下界。当 $q(\mathbf{z}|\boldsymbol{\theta}) = Pr(\mathbf{z}|\mathbf{x},\boldsymbol{\phi})$ 时，KL 距离为零，下界是**紧致的**。这是潜变量 $\mathbf{z}$ 在给定观测数据 $\mathbf{x}$ 下的**后验分布**（posterior distribution）；它指示哪些潜变量值可能对该数据点负责（图 17.7）。

> **图 17.7** 潜变量上的后验分布。a) 后验分布 $Pr(z|\mathbf{x}^*,\boldsymbol{\phi})$ 是在给定数据点 $\mathbf{x}^*$ 时，潜变量 $z$ 的取值上的分布。我们通过贝叶斯法则 $Pr(z|\mathbf{x}^*,\boldsymbol{\phi}) \propto Pr(\mathbf{x}^*|z,\boldsymbol{\phi})Pr(z)$ 来计算。b) 右侧第一项（似然）通过评估 $\mathbf{x}^*$ 对于每个 $z$ 值在对称高斯下的概率来计算。这里，它更可能由 $z_1$ 而非 $z_2$ 产生。第二项是潜变量上的先验概率 $Pr(z)$。将这两个因子相乘并归一化使分布和为一，就得到后验 $Pr(z|\mathbf{x}^*,\boldsymbol{\phi})$。

### 17.4.2 ELBO 作为重构损失减去与先验的 KL 距离

等式 17.16 和 17.17 是表达 ELBO 的两种不同方式。第三种方式是将下界视为重构误差减去与先验的距离：

$$
\begin{aligned}
\text{ELBO}[\boldsymbol{\theta},\boldsymbol{\phi}] &= \int q(\mathbf{z}|\boldsymbol{\theta})\log\left[\frac{Pr(\mathbf{x},\mathbf{z}|\boldsymbol{\phi})}{q(\mathbf{z}|\boldsymbol{\theta})}\right]d\mathbf{z} \\
&= \int q(\mathbf{z}|\boldsymbol{\theta})\log\left[\frac{Pr(\mathbf{x}|\mathbf{z},\boldsymbol{\phi})Pr(\mathbf{z})}{q(\mathbf{z}|\boldsymbol{\theta})}\right]d\mathbf{z} \\
&= \int q(\mathbf{z}|\boldsymbol{\theta})\log\left[Pr(\mathbf{x}|\mathbf{z},\boldsymbol{\phi})\right]d\mathbf{z} + \int q(\mathbf{z}|\boldsymbol{\theta})\log\left[\frac{Pr(\mathbf{z})}{q(\mathbf{z}|\boldsymbol{\theta})}\right]d\mathbf{z} \\
&= \int q(\mathbf{z}|\boldsymbol{\theta})\log\left[Pr(\mathbf{x}|\mathbf{z},\boldsymbol{\phi})\right]d\mathbf{z} - \text{D}_{KL}\left[q(\mathbf{z}|\boldsymbol{\theta})\,\|\,Pr(\mathbf{z})\right],
\end{aligned}
\tag{17.18}
$$

其中联合分布 $Pr(\mathbf{x},\mathbf{z}|\boldsymbol{\phi})$ 在第一行和第二行之间被分解为条件概率 $Pr(\mathbf{x}|\mathbf{z},\boldsymbol{\phi})Pr(\mathbf{z})$，最后一行再次使用了 KL 散度的定义。

在这个表述中，第一项衡量潜变量和数据之间 $Pr(\mathbf{x}|\mathbf{z},\boldsymbol{\phi})$ 的平均一致性。这衡量的是**重构精度**（reconstruction accuracy）。第二项衡量辅助分布 $q(\mathbf{z}|\boldsymbol{\theta})$ 与先验匹配的程度。这个表述就是变分自编码器中使用的表述。

## 17.5 变分近似

我们在等式 17.17 中看到，当 $q(\mathbf{z}|\boldsymbol{\theta})$ 是后验 $Pr(\mathbf{z}|\mathbf{x},\boldsymbol{\phi})$ 时，ELBO 是紧致的。原则上，我们可以使用贝叶斯法则计算后验：

$$
Pr(\mathbf{z}|\mathbf{x},\boldsymbol{\phi}) = \frac{Pr(\mathbf{x}|\mathbf{z},\boldsymbol{\phi})Pr(\mathbf{z})}{Pr(\mathbf{x}|\boldsymbol{\phi})},
\tag{17.19}
$$

但实际上这是不可处理的，因为我们无法计算分母中的证据项 $Pr(\mathbf{x}|\boldsymbol{\phi})$（见第 17.3 节）。

一种解决方案是做**变分近似**（variational approximation）：我们为 $q(\mathbf{z}|\boldsymbol{\theta})$ 选择一个简单的参数化形式，并用它来近似真实后验。这里，我们选择一个均值为 $\boldsymbol{\mu}$、对角协方差为 $\boldsymbol{\Sigma}$ 的多元正态分布。这不会总是很好地匹配后验，但对于某些 $\boldsymbol{\mu}$ 和 $\boldsymbol{\Sigma}$ 的值会比其他值更好。在训练中，我们将找到与真实后验 $Pr(\mathbf{z}|\mathbf{x})$ "最接近"的正态分布（图 17.8）。这对应于最小化等式 17.17 中的 KL 散度，将图 17.6 中的彩色曲线上移。

由于 $q(\mathbf{z}|\boldsymbol{\theta})$ 的最优选择是后验 $Pr(\mathbf{z}|\mathbf{x})$，且这取决于数据样本 $\mathbf{x}$，变分近似也应如此，因此我们选择：

$$
q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta}) = \text{Norm}_{\mathbf{z}}\left[\mathbf{g}_{\boldsymbol{\mu}}[\mathbf{x},\boldsymbol{\theta}], \mathbf{g}_{\boldsymbol{\Sigma}}[\mathbf{x},\boldsymbol{\theta}]\right],
\tag{17.20}
$$

其中 $\mathbf{g}[\mathbf{x},\boldsymbol{\theta}]$ 是第二个神经网络，参数为 $\boldsymbol{\theta}$，预测正态变分近似的均值 $\boldsymbol{\mu}$ 和方差 $\boldsymbol{\Sigma}$。

> **图 17.8** 变分近似。后验 $Pr(\mathbf{z}|\mathbf{x}^*,\boldsymbol{\phi})$ 无法以封闭形式计算。变分近似选择一族分布 $q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})$（这里是高斯分布），试图找到这个族中与真实后验最接近的成员。a) 有时近似效果很好（青色曲线），与真实后验（橙色曲线）很接近。b) 然而，如果后验是多模态的（如图 17.7），那么高斯近似就会很差。

## 17.6 变分自编码器

最后，我们可以描述 VAE。我们构建一个计算 ELBO 的网络：

$$
\text{ELBO}[\boldsymbol{\theta},\boldsymbol{\phi}] = \int q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})\log\left[Pr(\mathbf{x}|\mathbf{z},\boldsymbol{\phi})\right]d\mathbf{z} - \text{D}_{KL}\left[q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})\,\|\,Pr(\mathbf{z})\right],
\tag{17.21}
$$

其中分布 $q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})$ 是等式 17.20 中的近似。

第一项仍然涉及一个不可处理的积分，但由于它是关于 $q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})$ 的**期望**，我们可以通过采样来近似它。对于任意函数 $\text{a}[\bullet]$，有：

$$
\mathbb{E}_{\mathbf{z}}\left[\text{a}[\mathbf{z}]\right] = \int \text{a}[\mathbf{z}]q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})d\mathbf{z} \approx \frac{1}{N}\sum_{n=1}^{N}\text{a}[\mathbf{z}_n^*],
\tag{17.22}
$$

其中 $\mathbf{z}_n^*$ 是从 $q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})$ 中抽取的第 $n$ 个样本。这称为**蒙特卡罗估计**（Monte Carlo estimate）。作为非常粗略的估计，我们可以只使用从 $q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})$ 中抽取的单个样本 $\mathbf{z}^*$：

$$
\text{ELBO}[\boldsymbol{\theta},\boldsymbol{\phi}] \approx \log\left[Pr(\mathbf{x}|\mathbf{z}^*,\boldsymbol{\phi})\right] - \text{D}_{KL}\left[q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})\,\|\,Pr(\mathbf{z})\right].
\tag{17.23}
$$

第二项是变分分布 $q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta}) = \text{Norm}_{\mathbf{z}}[\boldsymbol{\mu},\boldsymbol{\Sigma}]$ 和先验 $Pr(\mathbf{z}) = \text{Norm}_{\mathbf{z}}[\mathbf{0},\mathbf{I}]$ 之间的 KL 散度。两个正态分布之间的 KL 散度可以以封闭形式计算。对于一个分布具有参数 $\boldsymbol{\mu},\boldsymbol{\Sigma}$ 而另一个是标准正态的特殊情形，它由下式给出：

$$
\text{D}_{KL}\left[q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})\,\|\,Pr(\mathbf{z})\right] = \frac{1}{2}\left(\text{Tr}[\boldsymbol{\Sigma}] + \boldsymbol{\mu}^T\boldsymbol{\mu} - D_{\mathbf{z}} - \log\left[\det[\boldsymbol{\Sigma}]\right]\right).
\tag{17.24}
$$

其中 $D_{\mathbf{z}}$ 是潜空间的维度。

### 17.6.1 VAE 算法

总结一下，我们旨在构建一个计算数据点 $\mathbf{x}$ 的证据下界的模型。然后使用优化算法在数据集上最大化该下界，从而提高对数似然。为了计算 ELBO，我们：

- 使用网络 $\mathbf{g}[\mathbf{x},\boldsymbol{\theta}]$ 计算该数据点变分后验分布 $q(\mathbf{z}|\boldsymbol{\theta},\mathbf{x})$ 的均值 $\boldsymbol{\mu}$ 和方差 $\boldsymbol{\Sigma}$，
- 从该分布中抽取样本 $\mathbf{z}^*$，以及
- 使用等式 17.23 计算 ELBO。

相关的架构如图 17.9 所示。现在应该清楚为什么称之为变分自编码器了。它是**变分的**，因为它计算了后验的高斯近似。它是**自编码器**，因为它从数据点 $\mathbf{x}$ 开始，计算一个低维的潜向量 $\mathbf{z}$，然后使用这个向量尽可能精确地重建数据点 $\mathbf{x}$。在这个语境中，网络 $\mathbf{g}[\mathbf{x},\boldsymbol{\theta}]$ 从数据到潜变量的映射称为**编码器**（encoder）；网络 $\mathbf{f}[\mathbf{z},\boldsymbol{\phi}]$ 从潜变量到数据的映射称为**解码器**（decoder）。

VAE 将 ELBO 计算为 $\boldsymbol{\phi}$ 和 $\boldsymbol{\theta}$ 的函数。为了最大化这个下界，我们将小批量样本通过网络并使用 SGD 或 Adam 等优化算法更新这些参数。ELBO 关于参数的梯度照常使用自动微分计算。在这个过程中，我们既在彩色曲线之间移动（改变 $\boldsymbol{\theta}$），又沿着曲线移动（改变 $\boldsymbol{\phi}$），如图 17.10 所示。在此过程中，参数 $\boldsymbol{\phi}$ 发生变化，使非线性潜变量模型为数据分配更高的似然。

> **图 17.9** 变分自编码器。编码器 $\mathbf{g}[\mathbf{x},\boldsymbol{\theta}]$ 接收训练样本 $\mathbf{x}$，预测变分分布 $q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})$ 的参数 $\boldsymbol{\mu},\boldsymbol{\Sigma}$。我们从该分布中采样，然后使用解码器 $\mathbf{f}[\mathbf{z},\boldsymbol{\phi}]$ 预测数据 $\mathbf{x}$。损失函数是负 ELBO，取决于预测的准确性以及变分分布 $q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})$ 与先验 $Pr(\mathbf{z})$ 的相似程度（等式 17.21）。

> **图 17.10** VAE 在每次迭代中更新决定下界的两个因素。解码器的参数 $\boldsymbol{\phi}$ 和编码器的参数 $\boldsymbol{\theta}$ 都被操纵以增加这个下界。

## 17.7 重参数化技巧

还有一个复杂之处；网络涉及一个采样步骤，很难对这个随机分量进行微分。然而，通过这个步骤进行微分对于更新网络中它之前的参数 $\boldsymbol{\theta}$ 是必要的。

幸运的是，有一个简单的解决方案；我们可以将随机部分移到网络的一个分支中，该分支从 $\text{Norm}_{\boldsymbol{\epsilon}}[\mathbf{0},\mathbf{I}]$ 中抽取样本 $\boldsymbol{\epsilon}^*$，然后使用关系式：

$$
\mathbf{z}^* = \boldsymbol{\mu} + \boldsymbol{\Sigma}^{1/2}\boldsymbol{\epsilon}^*,
\tag{17.25}
$$

从目标高斯分布中抽样。现在我们可以照常计算导数，因为反向传播算法不需要通过随机分支传递。这被称为**重参数化技巧**（reparameterization trick）（图 17.11）。

> **图 17.11** 重参数化技巧。在原始架构（图 17.9）中，我们无法轻易地通过采样步骤进行反向传播。重参数化技巧将采样步骤从主管道中移出；我们从标准正态分布中抽样，并将其与预测的均值和协方差结合以获得变分分布的样本。

## 17.8 应用

变分自编码器有许多用途，包括去噪、异常检测和压缩。本节回顾图像数据的几个应用。

### 17.8.1 近似样本概率

在第 17.3 节中，我们论证了用 VAE 无法评估样本的概率，该模型将这个概率描述为：

$$
\begin{aligned}
Pr(\mathbf{x}) &= \int Pr(\mathbf{x}|\mathbf{z})Pr(\mathbf{z})d\mathbf{z} \\
&= \mathbb{E}_{\mathbf{z}}\left[Pr(\mathbf{x}|\mathbf{z})\right] \\
&= \mathbb{E}_{\mathbf{z}}\left[\text{Norm}_{\mathbf{x}}[\mathbf{f}[\mathbf{z},\boldsymbol{\phi}], \sigma^2\mathbf{I}]\right].
\end{aligned}
\tag{17.26}
$$

原则上，我们可以使用等式 17.22，从 $Pr(\mathbf{z}) = \text{Norm}_{\mathbf{z}}[\mathbf{0},\mathbf{I}]$ 中抽取样本来*近似*这个概率，计算：

$$
Pr(\mathbf{x}) \approx \frac{1}{N}\sum_{n=1}^{N}Pr(\mathbf{x}|\mathbf{z}_n).
\tag{17.27}
$$

然而，维度灾难意味着几乎所有我们抽取的 $\mathbf{z}_n$ 值都会具有非常低的概率 $Pr(\mathbf{x}|\mathbf{z}_n)$；我们需要抽取大量样本才能得到可靠的估计。更好的方法是使用**重要性采样**（importance sampling）。这里，我们从辅助分布 $q(\mathbf{z})$ 中采样 $\mathbf{z}$，评估 $Pr(\mathbf{x}|\mathbf{z}_n)$，并用新分布下的概率 $q(\mathbf{z})$ 重新缩放结果值：

$$
\begin{aligned}
Pr(\mathbf{x}) &= \int Pr(\mathbf{x}|\mathbf{z})Pr(\mathbf{z})d\mathbf{z} \\
&= \int \frac{Pr(\mathbf{x}|\mathbf{z})Pr(\mathbf{z})}{q(\mathbf{z})}q(\mathbf{z})d\mathbf{z} \\
&= \mathbb{E}_{q(\mathbf{z})}\left[\frac{Pr(\mathbf{x}|\mathbf{z})Pr(\mathbf{z})}{q(\mathbf{z})}\right] \\
&\approx \frac{1}{N}\sum_{n=1}^{N}\frac{Pr(\mathbf{x}|\mathbf{z}_n)Pr(\mathbf{z}_n)}{q(\mathbf{z}_n)},
\end{aligned}
\tag{17.28}
$$

其中现在样本从 $q(\mathbf{z})$ 中抽取。如果 $q(\mathbf{z})$ 接近 $\mathbf{z}$ 空间中 $Pr(\mathbf{x}|\mathbf{z})$ 具有高似然的区域，那么我们将把采样集中在相关空间区域上，从而更高效地估计 $Pr(\mathbf{x})$。

我们尝试积分的乘积 $Pr(\mathbf{x}|\mathbf{z})Pr(\mathbf{z})$（由贝叶斯法则）与后验分布 $Pr(\mathbf{z}|\mathbf{x})$ 成正比。因此，辅助分布 $q(\mathbf{z})$ 的合理选择是编码器计算的变分后验 $q(\mathbf{z}|\mathbf{x})$。

这样，我们可以近似新样本的概率。有了足够多的样本，这将提供比下界更好的估计，可以用来通过评估测试数据的对数似然来评价模型质量。或者，它可以用作判断新样本是否属于该分布或是否异常的标准。

> **图 17.12** 从在 CELEBA 上训练的标准 VAE 采样。每列中抽取一个潜变量 $\mathbf{z}^*$，通过模型预测均值 $\mathbf{f}[\mathbf{z}^*,\boldsymbol{\phi}]$，然后加上独立的高斯噪声（见图 17.3）。a) 一组样本，是 b) 预测均值和 c) 球形高斯噪声向量的和。加噪声前图像看起来太平滑，加噪声后又太嘈杂。这是典型的，通常展示无噪声版本，因为噪声被认为代表图像中未建模的方面。改编自 Dorta et al. (2018)。d) 现在可以使用层次先验、专门的架构和仔细的正则化从 VAE 生成高质量图像。改编自 Vahdat & Kautz (2020)。

### 17.8.2 生成

VAE 构建了一个概率模型，很容易从该模型中采样，方法是从潜变量上的先验 $Pr(\mathbf{z})$ 中抽样，将结果通过解码器 $\mathbf{f}[\mathbf{z},\boldsymbol{\phi}]$，并根据 $Pr(\mathbf{x}|\mathbf{f}[\mathbf{z},\boldsymbol{\phi}])$ 添加噪声。不幸的是，来自普通 VAE 的样本通常质量较低（图 17.12a-c）。这部分是因为朴素的球形高斯噪声模型，部分是因为先验和变分后验使用的高斯模型。一种提高生成质量的技巧是从**聚合后验**（aggregated posterior）$q(\mathbf{z}|\boldsymbol{\theta}) = (1/I)\sum_i q(\mathbf{z}|\mathbf{x}_i,\boldsymbol{\theta})$ 而非先验中采样；这是所有样本的平均后验，是潜空间中真实分布的更具代表性的高斯混合。

现代 VAE 可以产生高质量样本（图 17.12d），但只能通过使用层次先验和专门的网络架构与正则化技术实现。扩散模型（第 18 章）可以视为具有层次先验的 VAE。它们也能创建非常高质量的样本。

### 17.8.3 重合成

VAE 也可以用来修改真实数据。数据点 $\mathbf{x}$ 可以通过 (i) 取编码器预测的分布的均值，或者 (ii) 通过优化过程找到使后验概率最大的潜变量 $\mathbf{z}$（贝叶斯法则告诉我们后验概率与 $Pr(\mathbf{x}|\mathbf{z})Pr(\mathbf{z})$ 成正比）来投影到潜空间中。

在图 17.13 中，多个标记为"中性"或"微笑"的图像被投影到潜空间中。表示这种变化的向量通过取这两组均值的差来估计。第二个向量估计表示"嘴巴闭合"与"嘴巴张开"。

现在，感兴趣的图像被投影到潜空间，然后通过加减这些向量来修改表示。为了生成中间图像，使用**球面线性插值**（spherical linear interpolation）或 *Slerp* 而非线性插值。在三维空间中，这就是沿球面表面插值与在球体内直接挖隧道之间的区别。

编码（并可能修改）输入数据然后再解码的过程被称为**重合成**（resynthesis）。这也可以用 GAN 和归一化流来完成。然而，在 GAN 中没有编码器，因此必须使用单独的过程来找到与观测数据对应的潜变量。

> **图 17.13** 重合成。左侧的原始图像使用编码器投影到潜空间，预测的高斯均值被选为表示该图像。网格中心左侧的图像是输入的重构。其他图像是在潜空间中沿表示微笑/中性（水平）和嘴巴张开/闭合（垂直）方向操纵后的重构。改编自 White (2016)。

### 17.8.4 解缠

在上述重合成示例中，表示可解释属性的空间方向必须使用标注的训练数据来估计。其他工作试图改善潜空间的特性，使其坐标方向对应于真实世界的属性。当每个维度代表一个独立的真实世界因素时，潜空间被描述为**解缠的**（disentangled）。例如，在建模人脸图像时，我们可能希望将头部姿态或发色作为独立因素发现出来。

鼓励解缠的方法通常基于 (i) 后验 $q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})$ 关于潜变量 $\mathbf{z}$ 的分布，或 (ii) 聚合后验 $q(\mathbf{z}|\boldsymbol{\theta}) = (1/I)\sum_i q(\mathbf{z}|\mathbf{x}_i,\boldsymbol{\theta})$，向损失函数添加正则化项：

$$
L_{\text{new}} = -\text{ELBO}[\boldsymbol{\theta},\boldsymbol{\phi}] + \lambda_1 \mathbb{E}_{Pr(\mathbf{x})}\left[\text{r}_1\left[q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})\right]\right] + \lambda_2 \text{r}_2\left[q(\mathbf{z}|\boldsymbol{\theta})\right].
\tag{17.29}
$$

其中正则化项 $\text{r}_1[\bullet]$ 是后验的函数，权重为 $\lambda_1$。项 $\text{r}_2[\bullet]$ 是聚合后验的函数，权重为 $\lambda_2$。

例如，**beta VAE** 上调了 ELBO 中的第二项（等式 17.18）：

$$
\text{ELBO}[\boldsymbol{\theta},\boldsymbol{\phi}] \approx \log\left[Pr(\mathbf{x}|\mathbf{z}^*,\boldsymbol{\phi})\right] - \beta \cdot \text{D}_{KL}\left[q(\mathbf{z}|\mathbf{x},\boldsymbol{\theta})\,\|\,Pr(\mathbf{z})\right],
\tag{17.30}
$$

其中 $\beta > 1$ 决定先验 $Pr(\mathbf{z})$ 的偏差相对于重构误差的权重有多大。由于先验通常是具有球形协方差矩阵的多元正态分布，其维度是独立的，上调该项鼓励后验分布的相关性更小。另一个变体是**总相关 VAE**（total correlation VAE），它添加了一个减少潜空间中变量总相关性的项（图 17.14），并最大化一小部分潜变量与观测之间的互信息。

> **图 17.14** 总相关 VAE 中的解缠。VAE 模型被修改使得损失函数鼓励潜变量的总相关性被最小化，从而鼓励解缠。在椅子图像数据集上训练时，几个潜维度具有清晰的真实世界解释，包括 a) 旋转、b) 整体大小和 c) 椅腿（转椅 vs 普通椅）。在每种情况下，中间列展示模型的样本，向左或向右移动时，我们在潜空间中减去或添加一个坐标向量。改编自 Chen et al. (2018d)。

## 17.9 总结

VAE 是一种帮助学习数据 $\mathbf{x}$ 上非线性潜变量模型的架构。该模型可以通过从潜变量中采样、将结果通过深度网络、然后添加独立的高斯噪声来生成新样本。

无法以封闭形式计算数据点的似然，这给最大似然训练带来了困难。然而，我们可以定义似然的下界并最大化这个下界。不幸的是，为了使下界紧致，我们需要计算给定观测数据时潜变量的后验概率，而这也是不可处理的。解决方案是做变分近似。这是一个更简单的分布（通常是高斯分布），近似后验，其参数由第二个编码器网络计算。

为了从 VAE 创建高质量样本，似乎有必要用比高斯先验和后验更复杂的概率分布来建模潜空间。一种选择是使用层次先验（其中一个潜变量生成另一个）。下一章讨论扩散模型，它产生非常高质量的样本，并可以视为层次 VAE。
