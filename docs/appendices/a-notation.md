# 附录A 符号表

本附录详细说明本书中使用的符号。这些符号大体遵循计算机科学中的标准惯例，但由于深度学习涉及许多不同领域，因此在此做完整说明。此外，本书有一些独特的符号惯例，包括函数的表示法以及参数和变量之间的系统性区分。

## 标量、向量、矩阵和张量

标量（scalar）用小写或大写字母表示，如 $a, A, \alpha$。列向量（column vector）（即一维数组）用小写粗体字母表示，如 $\mathbf{a}, \boldsymbol{\phi}$，行向量为列向量的转置 $\mathbf{a}^T, \boldsymbol{\phi}^T$。矩阵（matrix）和张量（tensor）（即二维和 $N$ 维数组）均用大写粗体字母表示，如 $\mathbf{B}, \boldsymbol{\Phi}$。

## 变量和参数

变量（variable）（通常是函数的输入、输出或中间计算结果）始终用罗马字母表示，如 $a, \mathbf{b}, \mathbf{C}$。参数（parameter）（函数或概率分布的内部量）始终用希腊字母表示，如 $\alpha, \boldsymbol{\beta}, \boldsymbol{\Gamma}$。通用的未指定参数用 $\boldsymbol{\phi}$ 表示。这一区分在全书中保持一致，唯一的例外是强化学习中的策略（policy），按照惯例用 $\pi$ 表示。

## 集合

集合（set）用花括号表示，因此 $\{0, 1, 2\}$ 表示数字 0、1 和 2。记号 $\{0, 1, 2, \ldots\}$ 表示非负整数集。有时我们需要指定一组变量，$\{\mathbf{x}_i\}_{i=1}^{I}$ 表示 $I$ 个变量 $\mathbf{x}_1, \ldots, \mathbf{x}_I$。当不需要指定集合中有多少项时，简写为 $\{\mathbf{x}_i\}$。记号 $\{\mathbf{x}_i, \mathbf{y}_i\}_{i=1}^{I}$ 表示 $I$ 对 $\mathbf{x}_i, \mathbf{y}_i$。命名集合的惯例是使用花体字母。特别地，$\mathcal{B}_t$ 用于表示训练中第 $t$ 次迭代时批次中的索引集合。集合 $\mathcal{S}$ 中元素的数量用 $|\mathcal{S}|$ 表示。

集合 $\mathbb{R}$ 表示实数集。集合 $\mathbb{R}^+$ 表示非负实数集。记号 $\mathbb{R}^D$ 表示包含实数的 $D$ 维向量的集合。记号 $\mathbb{R}^{D_1 \times D_2}$ 表示维度为 $D_1 \times D_2$ 的矩阵的集合。记号 $\mathbb{R}^{D_1 \times D_2 \times D_3}$ 表示大小为 $D_1 \times D_2 \times D_3$ 的张量的集合，依此类推。

记号 $[a, b]$ 表示从 $a$ 到 $b$ 的实数，包括 $a$ 和 $b$ 本身。当方括号被替换为圆括号时，意味着相邻的值不包含在集合中。例如，集合 $(-\pi, \pi]$ 表示从 $-\pi$ 到 $\pi$ 的实数，但不包括 $-\pi$。

集合的成员关系用符号 $\in$ 表示，因此 $x \in \mathbb{R}^+$ 表示变量 $x$ 是一个非负实数，记号 $\boldsymbol{\Sigma} \in \mathbb{R}^{D \times D}$ 表示 $\boldsymbol{\Sigma}$ 是一个大小为 $D \times D$ 的矩阵。有时，我们需要系统性地遍历集合中的每个元素，记号 $\forall \{1, \ldots, K\}$ 表示"对所有"从 1 到 $K$ 的整数。

## 函数

函数表示为一个名称后跟方括号，方括号中包含函数的参数。例如，$\log[x]$ 返回变量 $x$ 的对数。当函数返回一个向量时，它用粗体书写并以小写字母开头。例如，函数 $\mathbf{y} = \mathbf{mlp}[\mathbf{x}, \boldsymbol{\phi}]$ 返回一个向量 $\mathbf{y}$，并以向量参数 $\mathbf{x}$ 和 $\boldsymbol{\phi}$ 作为输入。当函数返回一个矩阵或张量时，它用粗体书写并以大写字母开头。例如，函数 $\mathbf{Y} = \mathbf{Sa}[\mathbf{X}, \boldsymbol{\phi}]$ 返回一个矩阵 $\mathbf{Y}$，并以参数 $\mathbf{X}$ 和 $\boldsymbol{\phi}$ 作为输入。当我们想故意不指定函数的参数时，使用圆点符号（例如 $\mathbf{mlp}[\bullet, \boldsymbol{\phi}]$）。

## 最小化和最大化

以下几个特殊函数在全书中反复使用：

- 函数 $\min_x [\text{f}[x]]$ 返回函数 $\text{f}[x]$ 在变量 $x$ 的所有可能值上的最小值。这一记号常在不指定如何找到这个最小值的情况下使用。
- 函数 $\text{argmin}_x [\text{f}[x]]$ 返回使 $\text{f}[x]$ 最小化的 $x$ 的值，因此如果 $y = \text{argmin}_x [\text{f}[x]]$，则 $\min_x [\text{f}[x]] = \text{f}[y]$。
- 函数 $\max_x [\text{f}[x]]$ 和 $\text{argmax}_x [\text{f}[x]]$ 对最大化函数执行等价操作。

## 概率分布

概率分布应写为 $Pr(x = a)$，表示随机变量 $x$ 取值 $a$。然而，这种记法较为繁琐。因此，我们通常简化为 $Pr(x)$，其中 $x$ 根据方程的含义表示随机变量或其取值。$y$ 在给定 $x$ 条件下的条件概率写为 $Pr(y|x)$。$y$ 和 $x$ 的联合概率写为 $Pr(y, x)$。这两种形式可以组合，因此 $Pr(\mathbf{y}|\mathbf{x}, \boldsymbol{\phi})$ 表示在已知 $\mathbf{x}$ 和 $\boldsymbol{\phi}$ 的条件下变量 $\mathbf{y}$ 的概率。类似地，$Pr(\mathbf{y}, \mathbf{x}|\boldsymbol{\phi})$ 表示在已知 $\boldsymbol{\phi}$ 的条件下变量 $\mathbf{y}$ 和 $\mathbf{x}$ 的概率。当我们需要对同一变量使用两个概率分布时，第一个分布写为 $Pr(x)$，第二个写为 $q(x)$。关于概率分布的更多信息可以在附录C中找到。

## 渐近符号

*渐近符号*（asymptotic notation）用于比较随着输入大小 $D$ 增加时不同算法所做的工作量。虽然可以用多种方式来表达，但本书仅使用*大O符号*（big-O notation），它表示算法计算量增长的上界。如果存在常数 $c > 0$ 和整数 $n_0$ 使得对所有 $n > n_0$ 都有 $\text{f}[n] < c \cdot \text{g}[n]$，则函数 $\text{f}[n]$ 是 $\mathcal{O}[\text{g}[n]]$。

这一符号给出了算法最坏情况运行时间的界。例如，当我们说求逆一个 $D \times D$ 矩阵的复杂度是 $\mathcal{O}[D^3]$，意思是一旦 $D$ 足够大，计算量的增长不会超过某个常数乘以 $D^3$。这使我们了解到求逆不同大小矩阵的可行性。如果 $D = 10^3$，那么求逆它可能需要 $10^9$ 量级的运算。

## 杂项

数学方程中的小圆点旨在提高阅读的便利性，没有实际含义（或仅暗示乘法）。例如，$\alpha \cdot \text{f}[x]$ 与 $\alpha \text{f}[x]$ 相同，但更易于阅读。为避免歧义，点积写为 $\mathbf{a}^T\mathbf{b}$（参见附录B.3.4节）。左箭头符号 $\leftarrow$ 表示赋值，因此 $x \leftarrow x + 2$ 表示我们将当前值的 $x$ 加上 2。
