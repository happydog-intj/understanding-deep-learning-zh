# 第7章 梯度与初始化

第 6 章介绍了迭代优化算法。这些算法是用于求函数最小值的通用方法。在神经网络的场景下，它们寻找使损失最小化的参数，从而使模型能够从输入准确预测训练输出。基本方法是随机选择初始参数，然后进行一系列小的调整，使损失平均而言逐步降低。每次调整都基于损失对当前位置处参数的梯度。

本章讨论两个神经网络特有的问题。首先，我们考虑如何高效地计算梯度。这是一个严峻的挑战，因为撰写本书时最大的模型拥有约 $10^{12}$ 个参数，而训练算法的每次迭代都需要计算每个参数的梯度。其次，我们考虑如何初始化参数。如果初始化不当，初始损失及其梯度可能会极端地大或极端地小。在任何一种情况下，都会阻碍训练过程。

## 7.1 问题定义

考虑一个网络 $\text{f}[\mathbf{x}, \boldsymbol{\phi}]$，它具有多元输入 $\mathbf{x}$、参数 $\boldsymbol{\phi}$，以及三个隐藏层 $\mathbf{h}_1$、$\mathbf{h}_2$ 和 $\mathbf{h}_3$：

$$\begin{aligned}
\mathbf{h}_1 &= \text{a}[\boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0 \mathbf{x}] \\
\mathbf{h}_2 &= \text{a}[\boldsymbol{\beta}_1 + \boldsymbol{\Omega}_1 \mathbf{h}_1] \\
\mathbf{h}_3 &= \text{a}[\boldsymbol{\beta}_2 + \boldsymbol{\Omega}_2 \mathbf{h}_2] \\
\text{f}[\mathbf{x}, \boldsymbol{\phi}] &= \boldsymbol{\beta}_3 + \boldsymbol{\Omega}_3 \mathbf{h}_3,
\end{aligned} \tag{7.1}$$

其中函数 $\text{a}[\bullet]$ 对输入的每个元素分别施加激活函数。模型参数 $\boldsymbol{\phi} = \{\boldsymbol{\beta}_0, \boldsymbol{\Omega}_0, \boldsymbol{\beta}_1, \boldsymbol{\Omega}_1, \boldsymbol{\beta}_2, \boldsymbol{\Omega}_2, \boldsymbol{\beta}_3, \boldsymbol{\Omega}_3\}$ 由每层之间的偏置向量 $\boldsymbol{\beta}_k$ 和权重矩阵 $\boldsymbol{\Omega}_k$ 组成（图 7.1）。

我们还有单个损失项 $\ell_i$，它返回给定模型预测 $\text{f}[\mathbf{x}_i, \boldsymbol{\phi}]$ 与真实标签 $y_i$ 时训练输入 $\mathbf{x}_i$ 的负对数似然。例如，这可以是最小二乘损失 $\ell_i = (\text{f}[\mathbf{x}_i, \boldsymbol{\phi}] - y_i)^2$。总损失是所有训练数据上这些项的求和：

$$L[\boldsymbol{\phi}] = \sum_{i=1}^{I} \ell_i. \tag{7.2}$$

训练神经网络最常用的优化算法是随机梯度下降（SGD），它按如下方式更新参数：

$$\boldsymbol{\phi}_{t+1} \longleftarrow \boldsymbol{\phi}_t - \alpha \sum_{i \in \mathcal{B}_t} \frac{\partial \ell_i[\boldsymbol{\phi}_t]}{\partial \boldsymbol{\phi}}, \tag{7.3}$$

其中 $\alpha$ 是学习率，$\mathcal{B}_t$ 包含迭代 $t$ 的批次索引。为了计算该更新，我们需要计算导数：

$$\frac{\partial \ell_i}{\partial \boldsymbol{\beta}_k} \qquad \text{和} \qquad \frac{\partial \ell_i}{\partial \boldsymbol{\Omega}_k}, \tag{7.4}$$

对每层 $k \in \{0, 1, \ldots, K\}$ 的参数 $\{\boldsymbol{\beta}_k, \boldsymbol{\Omega}_k\}$ 以及批次中的每个索引 $i$。本章第一部分描述*反向传播算法*（backpropagation algorithm），它能高效地计算这些导数。

在本章第二部分，我们考虑在训练开始前如何初始化网络参数。我们描述选择初始权重 $\boldsymbol{\Omega}_k$ 和偏置 $\boldsymbol{\beta}_k$ 的方法，以使训练保持稳定。

## 7.2 计算导数

损失的导数告诉我们，当对参数做出微小改变时，损失如何变化。优化算法利用这一信息来调整参数以使损失减小。*反向传播算法*计算这些导数。其数学细节较为复杂，因此我们先做两个直观观察。

**观察 1：** $\boldsymbol{\Omega}_k$ 中的每个权重将源隐藏单元处的激活值相乘，并将结果加到目标隐藏单元上。因此，对权重的任何微小改变的效果都会被源隐藏单元的激活值放大或衰减。由此，我们需要对批次中的每个数据样本运行网络，并存储所有隐藏单元的激活值。这被称为*前向传播*（forward pass）（图 7.1）。所存储的激活值随后将用于计算梯度。

**观察 2：** 偏置或权重的微小改变会引起后续网络中的一连串变化。该改变修改其目标隐藏单元的值。这进而改变后续层中隐藏单元的值，然后再改变更后面一层的隐藏单元，如此传递，直到改变模型输出，最终改变损失。

因此，要知道改变一个参数如何影响损失，我们还需要知道后续每个隐藏层的变化如何传递给其后继者。在考虑同一层或更早层的其他参数时，也需要这些同样的量。因此我们可以一次性计算并重复利用它们。例如，考虑计算馈入隐藏层 $\mathbf{h}_3$、$\mathbf{h}_2$ 和 $\mathbf{h}_1$ 的权重微小改变的效果：

- 要计算馈入隐藏层 $\mathbf{h}_3$ 的权重或偏置的微小改变如何影响损失，我们需要知道 (i) 层 $\mathbf{h}_3$ 的变化如何改变模型输出 $\text{f}$，以及 (ii) 输出的变化如何改变损失 $\ell$（图 7.2a）。
- 要计算馈入隐藏层 $\mathbf{h}_2$ 的权重或偏置的微小改变如何影响损失，我们需要知道 (i) 层 $\mathbf{h}_2$ 的变化如何影响 $\mathbf{h}_3$，(ii) $\mathbf{h}_3$ 如何改变模型输出，以及 (iii) 输出如何改变损失（图 7.2b）。
- 要计算馈入隐藏层 $\mathbf{h}_1$ 的权重或偏置的微小改变如何影响损失，我们需要知道 (i) 层 $\mathbf{h}_1$ 的变化如何影响层 $\mathbf{h}_2$，(ii) 层 $\mathbf{h}_2$ 的变化如何影响层 $\mathbf{h}_3$，(iii) 层 $\mathbf{h}_3$ 如何改变模型输出，以及 (iv) 模型输出如何改变损失（图 7.2c）。

<div align="center">

![图 7.1](/figures/ch07/Train2BP1.png)

</div>

> **图 7.1** 反向传播前向传播。目标是计算损失 $\ell$ 关于每个权重（箭头）和偏置（未显示）的导数。换言之，我们想知道每个参数的微小改变将如何影响损失。每个权重将其源处的隐藏单元激活值相乘，并将结果贡献给目标处的隐藏单元。因此，对权重的任何微小改变的效果都会被源隐藏单元的激活值缩放。例如，蓝色权重作用于第 1 层的第二个隐藏单元；如果该单元的激活值翻倍，则对蓝色权重微小改变的效果也会翻倍。因此，要计算权重的导数，我们需要计算并存储隐藏层处的激活值。这被称为前向传播，因为它涉及按顺序运行网络方程。

<div align="center">

![图 7.2](/figures/ch07/Train2BP2.png)

</div>

> **图 7.2** 反向传播的反向传播。a) 要计算馈入层 $\mathbf{h}_3$ 的权重（蓝色箭头）的改变如何影响损失，我们需要知道 $\mathbf{h}_3$ 中的隐藏单元如何改变模型输出 $\text{f}$，以及 $\text{f}$ 如何改变损失（橙色箭头）。b) 要计算馈入 $\mathbf{h}_2$ 的权重（蓝色箭头）的微小改变如何影响损失，我们需要知道 (i) $\mathbf{h}_2$ 中的隐藏单元如何改变 $\mathbf{h}_3$，(ii) $\mathbf{h}_3$ 如何改变 $\text{f}$，以及 (iii) $\text{f}$ 如何改变损失（橙色箭头）。c) 类似地，要计算馈入 $\mathbf{h}_1$ 的权重（蓝色箭头）的微小改变如何影响损失，我们需要知道 $\mathbf{h}_1$ 如何改变 $\mathbf{h}_2$，以及这些改变如何通过网络传播到损失（橙色箭头）。反向传播先在网络末端计算导数，然后反向回溯以利用这些计算中固有的冗余性。

当我们反向遍历网络时，可以发现大多数所需的项已在前一步中计算过，无需重复计算。以这种方式反向遍历网络来计算导数，被称为*反向传播*（backward pass）。

反向传播背后的思想相对容易理解。然而，推导需要矩阵微积分，因为偏置项和权重项分别是向量和矩阵。为了帮助理解其底层机制，下一节先用一个更简单的标量参数玩具模型来推导反向传播。然后我们在 7.4 节将同样的方法应用于深度神经网络。

## 7.3 玩具示例

考虑一个模型 $\text{f}[x, \boldsymbol{\phi}]$，它有八个标量参数 $\boldsymbol{\phi} = \{\beta_0, \omega_0, \beta_1, \omega_1, \beta_2, \omega_2, \beta_3, \omega_3\}$，由函数 $\sin[\bullet]$、$\exp[\bullet]$ 和 $\cos[\bullet]$ 复合而成：

$$\text{f}[x, \boldsymbol{\phi}] = \beta_3 + \omega_3 \cdot \cos\Big[\beta_2 + \omega_2 \cdot \exp\big[\beta_1 + \omega_1 \cdot \sin[\beta_0 + \omega_0 \cdot x]\big]\Big], \tag{7.5}$$

以及最小二乘损失函数 $L[\boldsymbol{\phi}] = \sum_i \ell_i$，其中各项为：

$$\ell_i = (\text{f}[x_i, \boldsymbol{\phi}] - y_i)^2, \tag{7.6}$$

其中 $x_i$ 是第 $i$ 个训练输入，$y_i$ 是第 $i$ 个训练输出。你可以将其想象为一个简单的神经网络，每层有一个输入、一个输出、一个隐藏单元，以及各层之间分别使用 $\sin[\bullet]$、$\exp[\bullet]$ 和 $\cos[\bullet]$ 作为激活函数。

我们的目标是计算导数：

$$\frac{\partial \ell_i}{\partial \beta_0},\quad \frac{\partial \ell_i}{\partial \omega_0},\quad \frac{\partial \ell_i}{\partial \beta_1},\quad \frac{\partial \ell_i}{\partial \omega_1},\quad \frac{\partial \ell_i}{\partial \beta_2},\quad \frac{\partial \ell_i}{\partial \omega_2},\quad \frac{\partial \ell_i}{\partial \beta_3},\quad \text{和} \quad \frac{\partial \ell_i}{\partial \omega_3}. \tag{7.7}$$

当然，我们可以手动推导这些导数的表达式并直接计算它们。但是，其中一些表达式相当复杂。例如：

$$\begin{aligned}
\frac{\partial \ell_i}{\partial \omega_0} &= -2\Big(\beta_3 + \omega_3 \cdot \cos\big[\beta_2 + \omega_2 \cdot \exp[\beta_1 + \omega_1 \cdot \sin[\beta_0 + \omega_0 \cdot x_i]]\big] - y_i\Big) \\
&\quad \cdot \omega_1 \omega_2 \omega_3 \cdot x_i \cdot \cos[\beta_0 + \omega_0 \cdot x_i] \cdot \exp\big[\beta_1 + \omega_1 \cdot \sin[\beta_0 + \omega_0 \cdot x_i]\big] \\
&\quad \cdot \sin\Big[\beta_2 + \omega_2 \cdot \exp\big[\beta_1 + \omega_1 \cdot \sin[\beta_0 + \omega_0 \cdot x_i]\big]\Big].
\end{aligned} \tag{7.8}$$

这类表达式既难以正确推导和编码，又未能利用其中固有的冗余性；注意上式中三个指数项是相同的。

反向传播算法是一种高效的方法，可同时计算所有这些导数。它包含 (i) 前向传播，在其中我们计算并存储一系列中间值和网络输出；以及 (ii) 反向传播，在其中我们从网络末端开始计算每个参数的导数，并在向起点回溯时重复利用先前的计算。

<div align="center">

![图 7.3](/figures/ch07/Train2BP3.png)

</div>

> **图 7.3** 反向传播前向传播。我们依次计算并存储每个中间变量，直到最终计算出损失。

**前向传播：** 我们将损失的计算视为一系列计算步骤：

$$\begin{aligned}
f_0 &= \beta_0 + \omega_0 \cdot x_i \\
h_1 &= \sin[f_0] \\
f_1 &= \beta_1 + \omega_1 \cdot h_1 \\
h_2 &= \exp[f_1] \\
f_2 &= \beta_2 + \omega_2 \cdot h_2 \\
h_3 &= \cos[f_2] \\
f_3 &= \beta_3 + \omega_3 \cdot h_3 \\
\ell_i &= (f_3 - y_i)^2.
\end{aligned} \tag{7.9}$$

我们计算并存储中间变量 $f_k$ 和 $h_k$ 的值（图 7.3）。

**反向传播 #1：** 现在我们计算 $\ell_i$ 关于这些中间变量的导数，但按逆序进行：

$$\frac{\partial \ell_i}{\partial f_3},\quad \frac{\partial \ell_i}{\partial h_3},\quad \frac{\partial \ell_i}{\partial f_2},\quad \frac{\partial \ell_i}{\partial h_2},\quad \frac{\partial \ell_i}{\partial f_1},\quad \frac{\partial \ell_i}{\partial h_1},\quad \text{和} \quad \frac{\partial \ell_i}{\partial f_0}. \tag{7.10}$$

第一个导数很直接：

$$\frac{\partial \ell_i}{\partial f_3} = 2(f_3 - y_i). \tag{7.11}$$

下一个导数可以用链式法则计算：

$$\frac{\partial \ell_i}{\partial h_3} = \frac{\partial f_3}{\partial h_3} \frac{\partial \ell_i}{\partial f_3}. \tag{7.12}$$

左边询问当 $h_3$ 改变时 $\ell_i$ 如何变化。右边告诉我们可以将其分解为 (i) 当 $h_3$ 改变时 $f_3$ 如何变化，以及 (ii) 当 $f_3$ 改变时 $\ell_i$ 如何变化。在原始方程中，$h_3$ 改变 $f_3$，$f_3$ 改变 $\ell_i$，这些导数代表了这条链的效应。注意这些导数中的第二个已经在前一步中计算过了，另一个是 $\beta_3 + \omega_3 \cdot h_3$ 关于 $h_3$ 的导数，即 $\omega_3$。

<div align="center">

![图 7.4](/figures/ch07/Train2BPIntuitions.png)

</div>

> **图 7.4** 反向传播的反向传播 #1。我们从函数末端反向计算损失关于中间量的导数 $\partial \ell_i / \partial f_k$ 和 $\partial \ell_i / \partial h_k$。每个导数由前一个导数乘以 $\partial f_k / \partial h_k$ 或 $\partial h_k / \partial f_{k-1}$ 形式的项得到。

我们以这种方式继续，计算输出关于这些中间量的导数（图 7.4）：

$$\begin{aligned}
\frac{\partial \ell_i}{\partial f_2} &= \frac{\partial h_3}{\partial f_2}\left(\frac{\partial f_3}{\partial h_3}\frac{\partial \ell_i}{\partial f_3}\right) \\
\frac{\partial \ell_i}{\partial h_2} &= \frac{\partial f_2}{\partial h_2}\left(\frac{\partial h_3}{\partial f_2}\frac{\partial f_3}{\partial h_3}\frac{\partial \ell_i}{\partial f_3}\right) \\
\frac{\partial \ell_i}{\partial f_1} &= \frac{\partial h_2}{\partial f_1}\left(\frac{\partial f_2}{\partial h_2}\frac{\partial h_3}{\partial f_2}\frac{\partial f_3}{\partial h_3}\frac{\partial \ell_i}{\partial f_3}\right) \\
\frac{\partial \ell_i}{\partial h_1} &= \frac{\partial f_1}{\partial h_1}\left(\frac{\partial h_2}{\partial f_1}\frac{\partial f_2}{\partial h_2}\frac{\partial h_3}{\partial f_2}\frac{\partial f_3}{\partial h_3}\frac{\partial \ell_i}{\partial f_3}\right) \\
\frac{\partial \ell_i}{\partial f_0} &= \frac{\partial h_1}{\partial f_0}\left(\frac{\partial f_1}{\partial h_1}\frac{\partial h_2}{\partial f_1}\frac{\partial f_2}{\partial h_2}\frac{\partial h_3}{\partial f_2}\frac{\partial f_3}{\partial h_3}\frac{\partial \ell_i}{\partial f_3}\right).
\end{aligned} \tag{7.13}$$

在每种情况下，括号内的量已在上一步中计算过，最后一项具有简单的表达式。这些方程体现了上一节的观察 2（图 7.2）；如果我们按逆序计算，就可以重复利用之前计算过的导数。

**反向传播 #2：** 最后，我们考虑当参数 $\{\beta_k\}$ 和 $\{\omega_k\}$ 改变时损失 $\ell_i$ 如何变化。再次应用链式法则（图 7.5）：

$$\begin{aligned}
\frac{\partial \ell_i}{\partial \beta_k} &= \frac{\partial f_k}{\partial \beta_k} \frac{\partial \ell_i}{\partial f_k} \\
\frac{\partial \ell_i}{\partial \omega_k} &= \frac{\partial f_k}{\partial \omega_k} \frac{\partial \ell_i}{\partial f_k}.
\end{aligned} \tag{7.14}$$

在每种情况下，右边的第二项已在方程 7.13 中计算过。当 $k > 0$ 时，$f_k = \beta_k + \omega_k \cdot h_k$，因此：

$$\frac{\partial f_k}{\partial \beta_k} = 1 \qquad \text{和} \qquad \frac{\partial f_k}{\partial \omega_k} = h_k. \tag{7.15}$$

<div align="center">

![图 7.5](/figures/ch07/Train2BPIntuitions2.png)

</div>

> **图 7.5** 反向传播的反向传播 #2。最后，我们计算 $\partial \ell_i / \partial \beta_k$ 和 $\partial \ell_i / \partial \omega_k$。每个导数由 $\partial \ell_i / \partial f_k$ 乘以 $\partial f_k / \partial \beta_k$ 或 $\partial f_k / \partial \omega_k$ 得到。

这与上一节的观察 1 一致；权重 $\omega_k$ 变化的效果正比于源变量 $h_k$ 的值（前向传播中已存储）。从 $f_0 = \beta_0 + \omega_0 \cdot x_i$ 得到的最后的导数是：

$$\frac{\partial f_0}{\partial \beta_0} = 1 \qquad \text{和} \qquad \frac{\partial f_0}{\partial \omega_0} = x_i. \tag{7.16}$$

反向传播比单独计算各导数（如方程 7.8）既简单又高效。

## 7.4 反向传播算法

现在我们对三层网络（图 7.1）重复这一过程。直觉和大部分代数运算是相同的。主要区别在于中间变量 $\mathbf{f}_k$、$\mathbf{h}_k$ 是向量，偏置 $\boldsymbol{\beta}_k$ 是向量，权重 $\boldsymbol{\Omega}_k$ 是矩阵，而且我们使用 ReLU 函数而非 $\cos[\bullet]$ 等简单代数函数。

**前向传播：** 我们将网络写为一系列顺序计算：

$$\begin{aligned}
\mathbf{f}_0 &= \boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0 \mathbf{x}_i \\
\mathbf{h}_1 &= \text{a}[\mathbf{f}_0] \\
\mathbf{f}_1 &= \boldsymbol{\beta}_1 + \boldsymbol{\Omega}_1 \mathbf{h}_1 \\
\mathbf{h}_2 &= \text{a}[\mathbf{f}_1] \\
\mathbf{f}_2 &= \boldsymbol{\beta}_2 + \boldsymbol{\Omega}_2 \mathbf{h}_2 \\
\mathbf{h}_3 &= \text{a}[\mathbf{f}_2] \\
\mathbf{f}_3 &= \boldsymbol{\beta}_3 + \boldsymbol{\Omega}_3 \mathbf{h}_3 \\
\ell_i &= \text{l}[\mathbf{f}_3, y_i].
\end{aligned} \tag{7.17}$$

<div align="center">

![图 7.6](/figures/ch07/Train2ReLUDeriv.png)

</div>

> **图 7.6** 修正线性单元的导数。修正线性单元（橙色曲线）在输入小于零时返回零，否则返回输入本身。它的导数（青色曲线）在输入小于零时返回零（因为此处斜率为零），在输入大于零时返回一（因为此处斜率为一）。

其中 $\mathbf{f}_{k-1}$ 表示第 $k$ 个隐藏层的预激活值（即经过 ReLU 函数 $\text{a}[\bullet]$ 之前的值），$\mathbf{h}_k$ 包含第 $k$ 个隐藏层的激活值（即 ReLU 函数之后的值）。项 $\text{l}[\mathbf{f}_3, y_i]$ 表示损失函数（例如最小二乘损失或二元交叉熵损失）。在前向传播中，我们逐步完成这些计算并存储所有中间量。

**反向传播 #1：** 现在让我们考虑当预激活值 $\mathbf{f}_0$、$\mathbf{f}_1$、$\mathbf{f}_2$ 改变时损失如何变化。应用链式法则，损失 $\ell_i$ 关于 $\mathbf{f}_2$ 的导数表达式为：

$$\frac{\partial \ell_i}{\partial \mathbf{f}_2} = \frac{\partial \mathbf{h}_3}{\partial \mathbf{f}_2} \frac{\partial \mathbf{f}_3}{\partial \mathbf{h}_3} \frac{\partial \ell_i}{\partial \mathbf{f}_3}. \tag{7.18}$$

右边三项的大小分别为 $D_3 \times D_3$、$D_3 \times D_f$ 和 $D_f \times 1$，其中 $D_3$ 是第三层隐藏单元数，$D_f$ 是模型输出 $\mathbf{f}_3$ 的维度。

类似地，我们可以计算当 $\mathbf{f}_1$ 和 $\mathbf{f}_0$ 改变时损失如何变化：

$$\frac{\partial \ell_i}{\partial \mathbf{f}_1} = \frac{\partial \mathbf{h}_2}{\partial \mathbf{f}_1} \frac{\partial \mathbf{f}_2}{\partial \mathbf{h}_2} \left(\frac{\partial \mathbf{h}_3}{\partial \mathbf{f}_2} \frac{\partial \mathbf{f}_3}{\partial \mathbf{h}_3} \frac{\partial \ell_i}{\partial \mathbf{f}_3}\right) \tag{7.19}$$

$$\frac{\partial \ell_i}{\partial \mathbf{f}_0} = \frac{\partial \mathbf{h}_1}{\partial \mathbf{f}_0} \frac{\partial \mathbf{f}_1}{\partial \mathbf{h}_1} \left(\frac{\partial \mathbf{h}_2}{\partial \mathbf{f}_1} \frac{\partial \mathbf{f}_2}{\partial \mathbf{h}_2} \frac{\partial \mathbf{h}_3}{\partial \mathbf{f}_2} \frac{\partial \mathbf{f}_3}{\partial \mathbf{h}_3} \frac{\partial \ell_i}{\partial \mathbf{f}_3}\right). \tag{7.20}$$

注意在每种情况下，括号内的项已在前一步中计算过。通过反向遍历网络，我们可以重复利用先前的计算。

此外，各项本身都很简单。反向计算方程 7.18 右边的各项，我们有：

- 损失 $\ell_i$ 关于网络输出 $\mathbf{f}_3$ 的导数 $\partial \ell_i / \partial \mathbf{f}_3$ 取决于损失函数，但通常具有简单的形式。

- 网络输出关于隐藏层 $\mathbf{h}_3$ 的导数 $\partial \mathbf{f}_3 / \partial \mathbf{h}_3$ 为：

$$\frac{\partial \mathbf{f}_3}{\partial \mathbf{h}_3} = \frac{\partial}{\partial \mathbf{h}_3}(\boldsymbol{\beta}_3 + \boldsymbol{\Omega}_3 \mathbf{h}_3) = \boldsymbol{\Omega}_3^T. \tag{7.21}$$

如果你不熟悉矩阵微积分，这个结果可能不太直观。

- 激活函数输出 $\mathbf{h}_3$ 关于其输入 $\mathbf{f}_2$ 的导数 $\partial \mathbf{h}_3 / \partial \mathbf{f}_2$ 取决于激活函数。它将是一个对角矩阵，因为每个激活值只取决于对应的预激活值。对于 ReLU 函数，对角项在 $\mathbf{f}_2$ 小于零的地方为零，其余地方为一（图 7.6）。我们不实际乘以这个矩阵，而是提取对角项作为向量 $\mathbb{I}[\mathbf{f}_2 > 0]$ 并进行逐元素乘法，这样更高效。

方程 7.19 和 7.20 右边的项具有类似的形式。当我们反向遍历网络时，交替执行 (i) 乘以权重矩阵 $\boldsymbol{\Omega}_k^T$ 的转置，以及 (ii) 基于馈入隐藏层的输入 $\mathbf{f}_{k-1}$ 进行阈值化。这些输入在前向传播中已经存储过。

**反向传播 #2：** 现在我们知道如何计算 $\partial \ell_i / \partial \mathbf{f}_k$，可以集中精力计算损失关于权重和偏置的导数。要计算损失关于偏置 $\boldsymbol{\beta}_k$ 的导数，我们再次使用链式法则：

$$\begin{aligned}
\frac{\partial \ell_i}{\partial \boldsymbol{\beta}_k} &= \frac{\partial \mathbf{f}_k}{\partial \boldsymbol{\beta}_k} \frac{\partial \ell_i}{\partial \mathbf{f}_k} \\
&= \frac{\partial}{\partial \boldsymbol{\beta}_k}(\boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_k) \frac{\partial \ell_i}{\partial \mathbf{f}_k} \\
&= \frac{\partial \ell_i}{\partial \mathbf{f}_k},
\end{aligned} \tag{7.22}$$

这已在方程 7.18 和 7.19 中计算过。

类似地，权重矩阵 $\boldsymbol{\Omega}_k$ 的导数为：

$$\begin{aligned}
\frac{\partial \ell_i}{\partial \boldsymbol{\Omega}_k} &= \frac{\partial \mathbf{f}_k}{\partial \boldsymbol{\Omega}_k} \frac{\partial \ell_i}{\partial \mathbf{f}_k} \\
&= \frac{\partial}{\partial \boldsymbol{\Omega}_k}(\boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_k) \frac{\partial \ell_i}{\partial \mathbf{f}_k} \\
&= \frac{\partial \ell_i}{\partial \mathbf{f}_k} \mathbf{h}_k^T.
\end{aligned} \tag{7.23}$$

从第二行到第三行的推导并不显然。然而，这个结果很有意义。最终结果是一个与 $\boldsymbol{\Omega}_k$ 大小相同的矩阵。它线性地依赖于 $\mathbf{h}_k$（在原表达式中 $\mathbf{h}_k$ 被 $\boldsymbol{\Omega}_k$ 所乘）。这也与最初的直觉一致：$\boldsymbol{\Omega}_k$ 中权重的导数正比于它们所乘的隐藏单元值 $\mathbf{h}_k$。回忆一下，我们已在前向传播中计算过这些值。

### 7.4.1 反向传播算法总结

我们现在简要总结最终的反向传播算法。考虑一个深度神经网络 $\text{f}[\mathbf{x}_i, \boldsymbol{\phi}]$，它接收输入 $\mathbf{x}_i$，有 $K$ 个使用 ReLU 激活的隐藏层，以及单个损失项 $\ell_i = \text{l}[\text{f}[\mathbf{x}_i, \boldsymbol{\phi}], \mathbf{y}_i]$。反向传播的目标是计算损失关于偏置 $\boldsymbol{\beta}_k$ 和权重 $\boldsymbol{\Omega}_k$ 的导数 $\partial \ell_i / \partial \boldsymbol{\beta}_k$ 和 $\partial \ell_i / \partial \boldsymbol{\Omega}_k$。

**前向传播：** 我们计算并存储以下量：

$$\begin{aligned}
\mathbf{f}_0 &= \boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0 \mathbf{x}_i \\
\mathbf{h}_k &= \text{a}[\mathbf{f}_{k-1}] \qquad k \in \{1, 2, \ldots, K\} \\
\mathbf{f}_k &= \boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_k. \qquad k \in \{1, 2, \ldots, K\}
\end{aligned} \tag{7.24}$$

**反向传播：** 我们从损失函数 $\ell_i$ 关于网络输出 $\mathbf{f}_K$ 的导数 $\partial \ell_i / \partial \mathbf{f}_K$ 开始，反向遍历网络：

$$\begin{aligned}
\frac{\partial \ell_i}{\partial \boldsymbol{\beta}_k} &= \frac{\partial \ell_i}{\partial \mathbf{f}_k} \qquad &k \in \{K, K-1, \ldots, 1\} \\
\frac{\partial \ell_i}{\partial \boldsymbol{\Omega}_k} &= \frac{\partial \ell_i}{\partial \mathbf{f}_k} \mathbf{h}_k^T \qquad &k \in \{K, K-1, \ldots, 1\} \\
\frac{\partial \ell_i}{\partial \mathbf{f}_{k-1}} &= \mathbb{I}[\mathbf{f}_{k-1} > 0] \odot \left(\boldsymbol{\Omega}_k^T \frac{\partial \ell_i}{\partial \mathbf{f}_k}\right), \qquad &k \in \{K, K-1, \ldots, 1\}
\end{aligned} \tag{7.25}$$

其中 $\odot$ 表示逐元素乘法，$\mathbb{I}[\mathbf{f}_{k-1} > 0]$ 是一个向量，在 $\mathbf{f}_{k-1}$ 大于零的位置包含 1，其余位置为 0。最后，我们计算第一组偏置和权重的导数：

$$\begin{aligned}
\frac{\partial \ell_i}{\partial \boldsymbol{\beta}_0} &= \frac{\partial \ell_i}{\partial \mathbf{f}_0} \\
\frac{\partial \ell_i}{\partial \boldsymbol{\Omega}_0} &= \frac{\partial \ell_i}{\partial \mathbf{f}_0} \mathbf{x}_i^T.
\end{aligned} \tag{7.26}$$

我们对批次中的每个训练样本计算这些导数，并将它们求和以得到 SGD 更新所需的梯度。

注意反向传播算法极为高效；前向传播和反向传播中计算量最大的步骤都是矩阵乘法（分别乘以 $\boldsymbol{\Omega}$ 和 $\boldsymbol{\Omega}^T$），这只需要加法和乘法。然而，它的内存效率不高；前向传播中的中间值必须全部存储，这可能限制我们能训练的模型大小。

### 7.4.2 自动微分

虽然理解反向传播算法很重要，但在实践中你不太可能需要自己编写它。现代深度学习框架如 PyTorch 和 TensorFlow 能根据模型规格自动计算导数。这被称为*自动微分*（algorithmic differentiation）。

框架中的每个功能组件（线性变换、ReLU 激活、损失函数）都知道如何计算自身的导数。例如，PyTorch 的 ReLU 函数 $\mathbf{z}_{out} = \textbf{relu}[\mathbf{z}_{in}]$ 知道如何计算其输出 $\mathbf{z}_{out}$ 关于输入 $\mathbf{z}_{in}$ 的导数。类似地，线性函数 $\mathbf{z}_{out} = \boldsymbol{\beta} + \boldsymbol{\Omega}\mathbf{z}_{in}$ 知道如何计算输出 $\mathbf{z}_{out}$ 关于输入 $\mathbf{z}_{in}$ 以及参数 $\boldsymbol{\beta}$ 和 $\boldsymbol{\Omega}$ 的导数。自动微分框架还知道网络中操作的序列，因而拥有执行前向传播和反向传播所需的全部信息。

这些框架利用了现代图形处理单元（GPU）的大规模并行性。前向传播和反向传播中的矩阵乘法等计算天然适合并行化。此外，如果模型和前向传播中的中间结果不超过可用内存，还可以对整个批次并行执行前向传播和反向传播。

由于训练算法现在并行处理整个批次，输入变成了多维*张量*（tensor）。在这个语境下，张量可以看作矩阵向任意维度的推广。因此，向量是一维张量，矩阵是二维张量，三维网格的数字是三维张量。到目前为止，训练数据都是一维的，所以反向传播的输入将是一个二维张量，其中第一个维度索引批次元素，第二个维度索引数据维度。在后续章节中，我们将遇到更复杂的结构化输入数据。例如，当输入是 RGB 图像时，原始数据样本是三维的（高度 × 宽度 × 通道）。此时，学习框架的输入将是一个四维张量，其中额外的维度索引批次元素。

### 7.4.3 扩展到任意计算图

我们描述的反向传播是在一个自然顺序结构的深度神经网络中进行的；我们依次计算中间量 $\mathbf{f}_0$、$\mathbf{h}_1$、$\mathbf{f}_1$、$\mathbf{h}_2$、$\ldots$、$\mathbf{f}_k$。然而，模型不必局限于顺序计算。在本书后面，我们将遇到具有分支结构的模型。例如，我们可能取某一隐藏层的值，通过两个不同的子网络处理，然后再合并。

幸运的是，只要计算图是无环的，反向传播的思想仍然适用。现代自动微分框架（如 PyTorch 和 TensorFlow）能够处理任意的无环计算图。

## 7.5 参数初始化

反向传播算法计算随机梯度下降和 Adam 所使用的导数来训练模型。现在我们讨论在训练开始前如何初始化参数。要理解这为何至关重要，考虑在前向传播中，每组预激活值 $\mathbf{f}_k$ 的计算如下：

$$\begin{aligned}
\mathbf{f}_k &= \boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_k \\
&= \boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \text{a}[\mathbf{f}_{k-1}],
\end{aligned} \tag{7.27}$$

其中 $\text{a}[\bullet]$ 施加 ReLU 函数，$\boldsymbol{\Omega}_k$ 和 $\boldsymbol{\beta}_k$ 分别是权重和偏置。假设我们将所有偏置初始化为零，将 $\boldsymbol{\Omega}_k$ 的元素按均值为零、方差为 $\sigma^2$ 的正态分布初始化。考虑两种情况：

- 如果方差 $\sigma^2$ 非常小（例如 $10^{-5}$），则 $\boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_k$ 的每个元素将是 $\mathbf{h}_k$ 各元素的加权和，而权重非常小；结果可能具有比输入更小的幅度。此外，ReLU 函数将小于零的值截断，因此 $\mathbf{h}_k$ 的值域将是 $\mathbf{f}_{k-1}$ 的一半。结果是，随着网络深度的推进，隐藏层预激活值的幅度会越来越小。

- 如果方差 $\sigma^2$ 非常大（例如 $10^5$），则 $\boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_k$ 的每个元素将是 $\mathbf{h}_k$ 各元素的加权和，而权重非常大；结果可能具有比输入大得多的幅度。ReLU 函数将值域减半，但如果 $\sigma^2$ 足够大，预激活值的幅度仍会随着网络深度的推进而越来越大。

在这两种情况下，预激活值可能变得如此之小或如此之大，以至于无法用有限精度浮点运算来表示。

即使前向传播是可处理的，同样的逻辑也适用于反向传播。每次梯度更新（方程 7.25）都包含乘以 $\boldsymbol{\Omega}^T$。如果 $\boldsymbol{\Omega}$ 的值初始化不合理，梯度幅度可能在反向传播过程中不可控地减小或增大。这两种情况分别被称为*梯度消失问题*（vanishing gradient problem）和*梯度爆炸问题*（exploding gradient problem）。前者导致模型更新变得极其微小，后者导致更新变得不稳定。

### 7.5.1 前向传播的初始化

我们现在给出上述论点的数学版本。考虑相邻预激活值 $\mathbf{f}$ 和 $\mathbf{f}'$ 之间的计算，它们的维度分别为 $D_h$ 和 $D_{h'}$：

$$\begin{aligned}
\mathbf{h} &= \text{a}[\mathbf{f}] \\
\mathbf{f}' &= \boldsymbol{\beta} + \boldsymbol{\Omega}\mathbf{h}
\end{aligned} \tag{7.28}$$

其中 $\mathbf{h}$ 表示激活值，$\boldsymbol{\Omega}$ 和 $\boldsymbol{\beta}$ 分别表示权重和偏置，$\text{a}[\bullet]$ 是激活函数。

假设输入层 $\mathbf{f}$ 的预激活值 $f_j$ 具有方差 $\sigma_f^2$。考虑将偏置 $\beta_i$ 初始化为零，权重 $\Omega_{ij}$ 按均值为零、方差为 $\sigma_\Omega^2$ 的正态分布初始化。现在我们推导后续层预激活值 $\mathbf{f}'$ 的均值和方差的表达式。

中间值 $f_i'$ 的期望（均值）$\mathbb{E}[f_i']$ 为：

$$\begin{aligned}
\mathbb{E}[f_i'] &= \mathbb{E}\left[\beta_i + \sum_{j=1}^{D_h} \Omega_{ij} h_j\right] \\
&= \mathbb{E}[\beta_i] + \sum_{j=1}^{D_h} \mathbb{E}[\Omega_{ij} h_j] \\
&= \mathbb{E}[\beta_i] + \sum_{j=1}^{D_h} \mathbb{E}[\Omega_{ij}] \, \mathbb{E}[h_j] \\
&= 0 + \sum_{j=1}^{D_h} 0 \cdot \mathbb{E}[h_j] = 0,
\end{aligned} \tag{7.29}$$

其中 $D_h$ 是输入层 $\mathbf{h}$ 的维度。我们使用了期望的运算规则，并假设隐藏单元 $h_j$ 和网络权重 $\Omega_{ij}$ 在第二行和第三行之间是独立的。

利用这个结果，我们可以看到预激活值 $f_i'$ 的方差 $\sigma_{f_i'}^2$ 为：

$$\begin{aligned}
\sigma_{f_i'}^2 &= \mathbb{E}[f_i'^2] - \mathbb{E}[f_i']^2 \\
&= \mathbb{E}\left[\left(\beta_i + \sum_{j=1}^{D_h} \Omega_{ij} h_j\right)^2\right] - 0 \\
&= \mathbb{E}\left[\left(\sum_{j=1}^{D_h} \Omega_{ij} h_j\right)^2\right] \\
&= \sum_{j=1}^{D_h} \mathbb{E}\left[\Omega_{ij}^2\right] \mathbb{E}\left[h_j^2\right] \\
&= \sum_{j=1}^{D_h} \sigma_\Omega^2 \mathbb{E}\left[h_j^2\right] = \sigma_\Omega^2 \sum_{j=1}^{D_h} \mathbb{E}\left[h_j^2\right],
\end{aligned} \tag{7.30}$$

其中我们使用了方差恒等式 $\sigma^2 = \mathbb{E}[(z - \mathbb{E}[z])^2] = \mathbb{E}[z^2] - \mathbb{E}[z]^2$。我们再次假设权重 $\Omega_{ij}$ 和隐藏单元 $h_j$ 的分布在第三行和第四行之间是独立的。

假设前一层的预激活值 $f_j$ 的分布关于零对称，则其中一半会被 ReLU 函数截断，二阶矩 $\mathbb{E}[h_j^2]$ 将是 $f_j$ 方差 $\sigma_f^2$ 的一半：

$$\sigma_{f_i'}^2 = \sigma_\Omega^2 \sum_{j=1}^{D_h} \frac{\sigma_f^2}{2} = \frac{1}{2} D_h \sigma_\Omega^2 \sigma_f^2. \tag{7.31}$$

<div align="center">

![图 7.7](/figures/ch07/Train2Exploding.png)

</div>

> **图 7.7** 权重初始化。考虑一个有 50 个隐藏层、每层 $D_h = 100$ 个隐藏单元的深度网络。网络有 100 维输入 $\mathbf{x}$、单一固定目标 $y = 0$，使用最小二乘损失函数。偏置向量 $\boldsymbol{\beta}_k$ 初始化为零，权重矩阵 $\boldsymbol{\Omega}_k$ 用均值为零、五种不同方差 $\sigma_\Omega^2 \in \{0.001, 0.01, 0.02, 0.1, 1.0\}$ 的正态分布初始化。a) 隐藏单元激活值的方差随网络层数变化。对于 He 初始化（$\sigma_\Omega^2 = 2/D_h = 0.02$），方差是稳定的。但对于更大的值，方差快速增加；对于更小的值，方差快速减小（注意纵轴为对数刻度）。b) 反向传播中梯度的方差（实线）延续了这一趋势；如果初始化时使用大于 0.02 的值，梯度幅度在反向传播过程中快速增加。如果使用更小的值，则幅度减小。这分别被称为梯度爆炸和梯度消失问题。

这意味着，如果我们希望后续预激活值 $\mathbf{f}'$ 的方差 $\sigma_{f'}^2$ 与原始预激活值 $\mathbf{f}$ 在前向传播中的方差 $\sigma_f^2$ 相同，我们应该设置：

$$\sigma_\Omega^2 = \frac{2}{D_h}, \tag{7.32}$$

其中 $D_h$ 是权重所作用的原始层的维度。这被称为 *He 初始化*（He initialization）。

### 7.5.2 反向传播的初始化

类似的论证建立了梯度 $\partial l / \partial \mathbf{f}_k$ 的方差在反向传播中如何变化。在反向传播中，我们乘以权重矩阵的转置 $\boldsymbol{\Omega}^T$（方程 7.25），因此等价的表达式变为：

$$\sigma_\Omega^2 = \frac{2}{D_{h'}}, \tag{7.33}$$

其中 $D_{h'}$ 是权重馈入的目标层的维度。

### 7.5.3 同时满足前向和反向传播的初始化

如果权重矩阵 $\boldsymbol{\Omega}$ 不是方阵（即相邻两层的隐藏单元数不同，$D_h$ 和 $D_{h'}$ 不同），则无法同时满足方程 7.32 和 7.33。一种可行的折中方案是使用均值 $(D_h + D_{h'})/2$ 作为项数的代理，由此得到：

$$\sigma_\Omega^2 = \frac{4}{D_h + D_{h'}}. \tag{7.34}$$

图 7.7 展示了经验上，当参数被适当初始化时，前向传播中隐藏单元的方差和反向传播中梯度的方差都保持稳定。

## 7.6 训练代码示例

本书的主要侧重点是科学性的；这不是一本实现深度学习模型的指南。不过，在图 7.8 中，我们展示了实现本书到目前为止所探讨的思想的 PyTorch 代码。该代码定义了一个神经网络并初始化权重。它创建了随机的输入和输出数据集，并定义了最小二乘损失函数。模型使用带动量的 SGD 在数据上训练，批次大小为 10，共训练 100 个 epoch。学习率初始为 0.01，每 10 个 epoch 减半。

要点在于，虽然深度学习背后的思想相当复杂，但实现却相对简单。例如，反向传播的所有细节都隐藏在一行代码中：`loss.backward()`。

> **图 7.8** 在随机数据上训练两层网络的示例代码。

## 7.7 总结

前一章介绍了随机梯度下降（SGD），一种旨在求函数最小值的迭代优化算法。在神经网络的场景下，该算法寻找使损失函数最小化的参数。SGD 依赖于损失函数关于参数的梯度，而这些参数必须在优化开始前被初始化。本章针对深度神经网络解决了这两个问题。

梯度必须为大量参数、批次中的每个成员以及 SGD 的每次迭代进行评估。因此，梯度计算的效率至关重要，为此引入了反向传播算法。仔细的参数初始化同样重要。前向传播中隐藏单元激活值的幅度可能随网络层数呈指数级增长或衰减。反向传播中梯度的幅度也是如此，这些现象分别被称为梯度消失和梯度爆炸问题。两者都会阻碍训练，但可以通过适当的初始化来避免。

至此，我们已经定义了模型和损失函数，也能训练模型了。下一章讨论如何度量模型性能。
