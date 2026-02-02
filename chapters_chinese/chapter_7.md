# 第7章

*页码：110–131*

---

## 第7章  
梯度与参数初始化  

第6章介绍了迭代优化算法。这类算法是用于寻找函数极小值的通用方法。在神经网络的语境中，它们用于搜寻使损失函数最小化的模型参数，从而使模型能够根据输入准确预测训练输出。其基本思路是：首先随机初始化参数，然后通过一系列微小调整逐步降低平均损失。每次调整均依据当前参数位置处损失函数关于参数的梯度。

本章讨论两个专属于神经网络的关键问题。第一，我们探讨如何高效地计算梯度。这是一项严峻挑战：截至本书撰写之时，规模最大的模型拥有约 $10^{12}$ 个参数，而训练算法在每一次迭代中都必须为每个参数计算梯度。第二，我们探讨参数的初始化策略。若初始化不当，初始损失及其梯度可能变得极大或极小；无论哪种情况，都会严重阻碍训练过程。

### 7.1 问题定义  

考虑一个具有多元输入 $\mathbf{x}$、参数 $\boldsymbol{\phi}$ 和三层隐藏层 $\mathbf{h}_1$、$\mathbf{h}_2$、$\mathbf{h}_3$ 的网络 $f[\mathbf{x}, \boldsymbol{\phi}]$：

$$
\begin{aligned}
\mathbf{h}_1 &= \mathbf{a}[\boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0 \mathbf{x}], \\
\mathbf{h}_2 &= \mathbf{a}[\boldsymbol{\beta}_1 + \boldsymbol{\Omega}_1 \mathbf{h}_1], \\
\mathbf{h}_3 &= \mathbf{a}[\boldsymbol{\beta}_2 + \boldsymbol{\Omega}_2 \mathbf{h}_2], \\
f[\mathbf{x}, \boldsymbol{\phi}] &= \boldsymbol{\beta}_3 + \boldsymbol{\Omega}_3 \mathbf{h}_3, \quad \text{(7.1)}
\end{aligned}
$$

其中函数 $\mathbf{a}[\bullet]$ 对输入向量（或矩阵）的每个元素独立施加激活函数。模型参数 $\boldsymbol{\phi} = \{\boldsymbol{\beta}_0, \boldsymbol{\Omega}_0, \boldsymbol{\beta}_1, \boldsymbol{\Omega}_1, \boldsymbol{\beta}_2, \boldsymbol{\Omega}_2, \boldsymbol{\beta}_3, \boldsymbol{\Omega}_3\}$ 由各层之间的偏置向量 $\boldsymbol{\beta}_k$ 和权重矩阵 $\boldsymbol{\Omega}_k$ 构成（见图7.1）。

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社。

7.2 计算导数　97  
我们还定义了各个样本的损失项 $\ell_i$，它返回模型对训练输入 $x_i$ 的预测 $f[x_i,\phi]$ 下，真实标签 $y_i$ 的负对数似然值。例如，该损失项可以是均方误差损失 $\ell_i = (f[x_i,\phi] - y_i)^2$。总损失为所有训练样本损失项之和：  
$$
L[\phi] = \sum_{i=1}^I \ell_i. \tag{7.2}
$$  

训练神经网络最常用的优化算法是随机梯度下降（Stochastic Gradient Descent, SGD），其参数更新规则为：  
$$
\phi_{t+1} \leftarrow \phi_t - \alpha \sum_{i \in \mathcal{B}_t} \frac{\partial \ell_i[\phi_t]}{\partial \phi}, \tag{7.3}
$$  
其中 $\alpha$ 为学习率，$\mathcal{B}_t$ 表示第 $t$ 次迭代所用的小批量（batch）样本索引集合。为执行该更新，我们需要计算如下导数：  
$$
\frac{\partial \ell_i}{\partial \beta_k} \quad \text{和} \quad \frac{\partial \ell_i}{\partial \Omega_k}, \tag{7.4}
$$  
其中 $\{\beta_k, \Omega_k\}$ 为第 $k$ 层（$k \in \{0,1,\dots,K\}$）的参数，而 $i$ 遍历当前小批量中的每个样本索引。本章第一部分将介绍反向传播（backpropagation）算法——一种高效计算上述导数的方法。  

本章第二部分则讨论网络训练开始前的参数初始化策略。我们将介绍如何选取初始权重 $\Omega_k$ 和偏置 $\beta_k$，以确保训练过程稳定。  

7.2 计算导数  
损失函数关于参数的导数揭示了：当参数发生微小变化时，损失函数随之变化的方向与幅度。各类优化算法正是利用这一信息来调整参数，从而降低损失值。反向传播算法即用于计算这些导数。其数学细节较为复杂，因此我们首先提出两个直观观察，以帮助理解。  

**观察 1**：每个权重（即 $\Omega_k$ 中的某个元素）将某一源隐层单元的激活值相乘，并将结果加至下一层中某一目标隐层单元。因此，该权重的任意微小变动，其影响均会因源隐层单元的激活值而被放大或衰减。故而，我们需对小批量中每个数据样本执行一次前向传播（forward pass），并缓存所有隐层单元的激活值（见图 7.1）。这些缓存的激活值将在后续用于梯度计算。  

**观察 2**：偏置或权重的微小变动，会在后续网络中引发一系列连锁反应（ripple effect），即逐层传播的数值变化。该变化首先修改其目标隐层单元的取值，进而影响更深层的计算。  

草稿：如有勘误，请发送邮件至 udlbookmail@gmail.com。

98 7 梯度与参数初始化  
图7.1　反向传播的前向传播过程。目标是计算损失函数 $\ell$ 关于每个权重（图中箭头所示）和偏置（图中未标出）的导数。换言之，我们希望了解每个参数的微小变化将如何影响损失值。每个权重将其源端隐单元的激活值相乘，并将结果贡献至其目标端隐单元。因此，该权重的任意微小变化所产生的效应，都将按源端隐单元的激活值进行缩放。例如，蓝色权重作用于第1层的第二个隐单元；若该隐单元的激活值翻倍，则对该蓝色权重施加微小扰动所引起的效应也将随之翻倍。因此，为计算权重的导数，我们需要计算并存储各隐层的激活值。这一过程称为**前向传播**（forward pass），因其涉及按顺序执行网络的前向计算方程。

该隐单元的变化，又会改变后续层中隐单元的取值；后者进一步引发再下一层隐单元的变化，依此类推，直至模型输出发生变化，最终导致损失值改变。

因此，若要获知某一参数的变动如何影响损失，我们还需掌握：后续每一隐层的变动又将如何依次影响其后继层。当考察同一层或更早层中的其他参数时，所需计算的正是完全相同的这些量。由此可知，我们可以一次性计算这些量，并在后续重复使用。例如，考虑分别计算输入至隐层 $h_3$、$h_2$ 和 $h_1$ 的权重所受微小扰动对损失的影响：

- 为计算输入至隐层 $h_3$ 的某个权重或偏置的微小变化对损失的影响，我们需要知道：(i) 隐层 $h_3$ 的变化如何影响模型输出 $f$；(ii) 该模型输出的变化又如何影响损失 $\ell$（见图7.2a）。  
- 为计算输入至隐层 $h_2$ 的某个权重或偏置的微小变化对损失的影响，我们需要知道：(i) 隐层 $h_2$ 的变化如何影响 $h_3$；(ii) $h_3$ 的变化如何影响模型输出；(iii) 该模型输出的变化如何影响损失 $\ell$（见图7.2b）。  
- 为计算输入至隐层 $h_1$ 的某个权重或偏置的微小变化对损失的影响，我们需要知道：(i) 隐层 $h_1$ 的变化如何影响 $h_2$；(ii) 隐层 $h_2$ 的变化如何影响 $h_3$；(iii) 隐层 $h_3$ 的变化如何影响模型输出；(iv) 模型输出的变化如何影响损失 $\ell$（见图7.2c）。

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

7.2 计算导数　99  
图7.2　反向传播的反向传递过程。  
a) 为计算输入至隐层 $h_3$ 的某个权重（蓝色箭头）发生微小变化时对损失函数的影响，我们需要知道：该隐层 $h_3$ 中的隐单元如何影响模型输出 $f$，以及 $f$ 又如何影响损失（橙色箭头）。  
b) 为计算输入至隐层 $h_2$ 的某个权重（蓝色箭头）发生微小变化时对损失函数的影响，我们需要知道：(i) 隐层 $h_2$ 中的隐单元如何影响 $h_3$；(ii) $h_3$ 如何影响 $f$；(iii) $f$ 如何影响损失（橙色箭头）。  
c) 类似地，为计算输入至隐层 $h_1$ 的某个权重（蓝色箭头）发生微小变化时对损失函数的影响，我们需要知道 $h_1$ 如何影响 $h_2$，以及这些影响如何逐层向前传播直至损失函数（橙色箭头）。反向传递过程首先在网络末端计算导数，再逐步向后回溯，从而充分利用这些导数计算中固有的冗余性。  

草稿：如有勘误，请发送邮件至 udlbookmail@gmail.com。

100 7 梯度与初始化  
当我们沿网络反向传播时，会发现所需计算的大多数项在前一步中已经完成，因此无需重复计算。以这种方式沿网络反向推进以计算导数的过程，称为**反向传播过程（backward pass）**。

反向传播（backpropagation）背后的思想相对容易理解；然而，其严格推导需要用到矩阵微积分，这是因为偏置项为向量、权重项为矩阵。为帮助读者深入理解其内在机理，下一节将首先针对一个仅含标量参数的简化玩具模型推导反向传播；随后，我们将在第 7.4 节中将相同方法推广至深度神经网络。

7.3 玩具示例  
考虑一个模型 $ f[x,\phi] $，其包含八个标量参数 $ \phi = \{\beta_0, \omega_0, \beta_1, \omega_1, \beta_2, \omega_2, \beta_3, \omega_3\} $，该模型由函数 $ \sin[\cdot] $、$ \exp[\cdot] $ 和 $ \cos[\cdot] $ 复合而成：

$$
f[x,\phi] = \beta_3 + \omega_3 \cdot \cos\!\left[ \beta_2 + \omega_2 \cdot \exp\!\left[ \beta_1 + \omega_1 \cdot \sin[\beta_0 + \omega_0 \cdot x] \right] \right], \tag{7.5}
$$

并采用最小二乘损失函数 $ L[\phi] = \sum_i \ell_i $，其中各项定义为：

$$
\ell_i = \big(f[x_i,\phi] - y_i\big)^2, \tag{7.6}
$$

其中，如惯例所示，$ x_i $ 表示第 $ i $ 个训练输入，$ y_i $ 表示第 $ i $ 个训练输出。可将此模型视作一个结构简单的神经网络：它具有单个输入、单个输出、每层仅含一个隐单元，且相邻层之间分别采用不同的激活函数——$ \sin[\cdot] $、$ \exp[\cdot] $ 和 $ \cos[\cdot] $。

我们的目标是计算如下八个偏导数：

$$
\frac{\partial \ell_i}{\partial \beta_0},\ 
\frac{\partial \ell_i}{\partial \omega_0},\ 
\frac{\partial \ell_i}{\partial \beta_1},\ 
\frac{\partial \ell_i}{\partial \omega_1},\ 
\frac{\partial \ell_i}{\partial \beta_2},\ 
\frac{\partial \ell_i}{\partial \omega_2},\ 
\frac{\partial \ell_i}{\partial \beta_3},\ 
\frac{\partial \ell_i}{\partial \omega_3}. \tag{7.7}
$$

当然，我们完全可以手动推导这些导数的解析表达式并直接计算。但其中某些表达式极为复杂。例如：

$$
\begin{aligned}
\frac{\partial \ell_i}{\partial \omega_0} = &-2 \Big( \beta_3 + \omega_3 \cdot \cos\!\big[ \beta_2 + \omega_2 \cdot \exp\!\big[ \beta_1 + \omega_1 \cdot \sin[\beta_0 + \omega_0 \cdot x_i] \big] \big] - y_i \Big) \\
&\cdot \omega_1 \omega_2 \omega_3 \cdot x_i \cdot \cos[\beta_0 + \omega_0 \cdot x_i] \cdot \exp\!\big[ \beta_1 + \omega_1 \cdot \sin[\beta_0 + \omega_0 \cdot x_i] \big] \\
&\cdot \sin\!\big[ \beta_2 + \omega_2 \cdot \exp\!\big[ \beta_1 + \omega_1 \cdot \sin[\beta_0 + \omega_0 \cdot x_i] \big] \big]. \tag{7.8}
\end{aligned}
$$

此类表达式不仅推导繁琐、编程实现时极易出错，而且未能利用其内在的冗余性——注意，上式中出现了三个完全相同的指数项。

反向传播算法正是一种高效地一次性计算所有这些导数的方法。该算法包含两个阶段：（i）**前向传播过程（forward pass）**，即逐层计算并缓存一系列中间变量及网络最终输出；（ii）**反向传播过程（backward pass）**，即……

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议保护。（©）麻省理工学院出版社。

7.3 简单示例 101  
图 7.3 反向传播的前向传递过程。我们依次计算并存储每个中间变量，直至最终算出损失值。  
接着，我们从网络末端开始，逐层计算各参数的导数，并在向网络起始端回溯的过程中复用先前已计算的结果。  

**前向传递**：我们将损失值的计算视为一系列连续运算：  
$$
f_0 = \beta_0 + \boldsymbol{\omega}_0 \cdot \mathbf{x}_i \\
h_1 = \sin[f_0] \\
f_1 = \beta_1 + \boldsymbol{\omega}_1 \cdot \mathbf{h}_1 \\
h_2 = \exp[f_1] \\
f_2 = \beta_2 + \boldsymbol{\omega}_2 \cdot \mathbf{h}_2 \\
h_3 = \cos[f_2] \\
f_3 = \beta_3 + \boldsymbol{\omega}_3 \cdot \mathbf{h}_3 \\
\ell_i = (f_3 - y_i)^2. \tag{7.9}
$$  
我们计算并存储所有中间变量 $ f_k $ 和 $ h_k $ 的值（见图 7.3）。  

**反向传递 #1**：接下来，我们按逆序计算损失 $ \ell_i $ 关于这些中间变量的偏导数：  
$$
\frac{\partial \ell_i}{\partial f_3},\; \frac{\partial \ell_i}{\partial h_3},\; \frac{\partial \ell_i}{\partial f_2},\; \frac{\partial \ell_i}{\partial h_2},\; \frac{\partial \ell_i}{\partial f_1},\; \frac{\partial \ell_i}{\partial h_1},\; \frac{\partial \ell_i}{\partial f_0}. \tag{7.10}
$$  
其中第一个偏导数可直接得出：  
$$
\frac{\partial \ell_i}{\partial f_3} = 2(f_3 - y_i). \tag{7.11}
$$  
下一个偏导数则可通过链式法则计算：  
$$
\frac{\partial \ell_i}{\partial h_3} = \frac{\partial f_3}{\partial h_3} \cdot \frac{\partial \ell_i}{\partial f_3}. \tag{7.12}
$$  
左侧表示当 $ h_3 $ 变化时 $ \ell_i $ 的变化率；右侧则将其分解为两部分：(i) $ f_3 $ 随 $ h_3 $ 的变化率，以及 (ii) $ \ell_i $ 随 $ f_3 $ 的变化率。在原始方程中，$ h_3 $ 影响 $ f_3 $，而 $ f_3 $ 进而影响 $ \ell_i $。  

草稿：请将勘误发送至 udlbookmail@gmail.com。

102 7 梯度与初始化  
图7.4　反向传播：反向传递阶段 #1。我们从函数末端开始逆向计算损失 ℓ 关于中间变量的偏导数 ∂ℓ/∂fₖ 和 ∂ℓ/∂hₖ。每个偏导数均通过将前一个偏导数乘以形如 ∂fₖ/∂hₖ 或 ∂hₖ/∂fₖ₋₁ 的项得到，这些项表征了链式求导过程中的影响。注意，我们已预先计算出上述两个偏导数中的第二个；而另一个则是 β₃ + ω₃ · h₃ 关于 h₃ 的导数，即 ω₃。

我们依此方式继续推进，依次计算输出关于这些中间变量的偏导数（见图7.4）：

$$
\frac{\partial \ell_i}{\partial f_2} = \frac{\partial h_3}{\partial f_2} \frac{\partial f_3}{\partial h_3} \frac{\partial \ell_i}{\partial f_3}
$$

$$
\frac{\partial \ell_i}{\partial h_2} = \frac{\partial f_2}{\partial h_2} \frac{\partial h_3}{\partial f_2} \frac{\partial f_3}{\partial h_3} \frac{\partial \ell_i}{\partial f_3}
$$

$$
\frac{\partial \ell_i}{\partial f_1} = \frac{\partial h_2}{\partial f_1} \frac{\partial f_2}{\partial h_2} \frac{\partial h_3}{\partial f_2} \frac{\partial f_3}{\partial h_3} \frac{\partial \ell_i}{\partial f_3}
$$

$$
\frac{\partial \ell_i}{\partial h_1} = \frac{\partial f_1}{\partial h_1} \frac{\partial h_2}{\partial f_1} \frac{\partial f_2}{\partial h_2} \frac{\partial h_3}{\partial f_2} \frac{\partial f_3}{\partial h_3} \frac{\partial \ell_i}{\partial f_3}
$$

$$
\frac{\partial \ell_i}{\partial f_0} = \frac{\partial h_1}{\partial f_0} \frac{\partial f_1}{\partial h_1} \frac{\partial h_2}{\partial f_1} \frac{\partial f_2}{\partial h_2} \frac{\partial h_3}{\partial f_2} \frac{\partial f_3}{\partial h_3} \frac{\partial \ell_i}{\partial f_3}. \tag{7.13}
$$

在每种情形下，方括号内的各项均已在前一步骤中完成计算，而最后一项则具有简洁明了的表达式。这些等式体现了上一节中提出的“观察2”（见图7.2）：若按逆序计算导数，则可复用先前已计算出的导数值。

反向传递阶段 #2：最后，我们考察当参数 {βₖ} 和 {ωₖ} 发生变化时，损失 ℓᵢ 如何随之改变。我们再次应用链式法则（见图7.5）：

$$
\frac{\partial \ell_i}{\partial \beta_k} = \frac{\partial f_k}{\partial \beta_k} \frac{\partial \ell_i}{\partial f_k}
$$

$$
\frac{\partial \ell_i}{\partial \omega_k} = \frac{\partial f_k}{\partial \omega_k} \frac{\partial \ell_i}{\partial f_k}. \tag{7.14}
$$

在每种情形下，右侧第二项已在式(7.13)中计算得出。当 k > 0 时，有 fₖ = βₖ + ωₖ · hₖ，因此：

$$
\frac{\partial f_k}{\partial \beta_k} = 1 \quad \text{且} \quad \frac{\partial f_k}{\partial \omega_k} = h_k. \tag{7.15}
$$

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

7.4 反向传播算法　103  
图7.5　反向传播的反向传递步骤 #2。最后，我们计算偏导数 $\partial\ell/\partial\beta_{k}$ 和 $\partial\ell/\partial\omega_{k}$。每个偏导数均通过将中间项 $\partial\ell/\partial f_k$ 分别乘以对应的 $\partial f_k/\partial\beta_k$ 或 $\partial f_k/\partial\omega_k$ 得到。  

这与上一节中的“观察1”一致：权重 $\omega_k$ 的变化所造成的影响，正比于其源变量 $h_k$ 的取值（该值已在前向传递过程中被缓存）。由表达式 $f_0 = \beta_0 + \omega_0 \cdot x_i$ 所得的最终偏导数为：  

**Notebook 7.1**  
**反向传播**  
**intoymodel**  

$$
\frac{\partial f_0}{\partial \beta_0} = 1 \quad \text{且} \quad \frac{\partial f_0}{\partial \omega_0} = x_i. \tag{7.16}
$$  

相较于逐项计算导数（如式(7.8)所示），反向传播不仅更简洁，而且效率更高。  

7.4　反向传播算法  
现在，我们将上述过程推广至一个三层网络（见图7.1）。其核心思想及大部分代数推导均保持一致；主要区别在于：中间变量 $f_k$、$h_k$ 均为向量，偏置项 $\beta_k$ 为向量，权重 $\Omega_k$ 为矩阵，且激活函数采用 ReLU 函数，而非余弦等简单代数函数（如 $\cos[\cdot]$）。  

**前向传递**：我们将网络表示为一系列顺序计算：  

$$
\begin{aligned}
\mathbf{f}_0 &= \boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0 \mathbf{x}_i \\
\mathbf{h}_1 &= \mathbf{a}[\mathbf{f}_0] \\
\mathbf{f}_1 &= \boldsymbol{\beta}_1 + \boldsymbol{\Omega}_1 \mathbf{h}_1 \\
\mathbf{h}_2 &= \mathbf{a}[\mathbf{f}_1] \\
\mathbf{f}_2 &= \boldsymbol{\beta}_2 + \boldsymbol{\Omega}_2 \mathbf{h}_2 \\
\mathbf{h}_3 &= \mathbf{a}[\mathbf{f}_2] \\
\mathbf{f}_3 &= \boldsymbol{\beta}_3 + \boldsymbol{\Omega}_3 \mathbf{h}_3 \\
\ell_i &= l[\mathbf{f}_3, y_i], \tag{7.17}
\end{aligned}
$$  

¹ 注意：我们实际上并不需要损失函数 $\ell_i$ 关于各层激活值 $h_k$ 的偏导数 $\partial \ell_i / \partial h_k$。在最终的反向传播算法中，这些偏导数不会被显式计算。  

草稿：如有勘误，请发送邮件至 udlbookmail@gmail.com。

104 7 梯度与初始化  
图7.6 修正线性单元（ReLU）的导数。修正线性单元（橙色曲线）在输入小于零时输出零，否则输出输入本身。其导数（青色曲线）在输入小于零时为零（此处斜率为零），在输入大于零时为一（此处斜率为一）。

其中，$ f^{k-1} $ 表示第 $ k $ 个隐含层的预激活值（即 ReLU 函数 $ a[\cdot] $ 作用前的值），而 $ h^k $ 包含第 $ k $ 个隐含层的激活值（即经过 ReLU 函数之后的值）。项 $ \ell[f_3, y_i] $ 表示损失函数（例如最小二乘损失或二元交叉熵损失）。在前向传播过程中，我们依次执行这些计算，并存储所有中间量。

**反向传播步骤 #1：** 现在考虑当预激活值 $ f_0 $、$ f_1 $、$ f_2 $ 发生变化时，损失如何随之改变。应用链式法则，损失 $ \ell_i $ 关于 $ f_2 $ 的导数表达式为：

**附录 B.5 矩阵微积分 2**

$$
\frac{\partial \ell_i}{\partial f_2} = \frac{\partial h_3}{\partial f_2} \frac{\partial f_3}{\partial h_3} \frac{\partial \ell_i}{\partial f_3}. \tag{7.18}
$$

右侧三项的维度分别为 $ D_3 \times D_2 $、$ D_3 \times D_3 $ 和 $ D_3 \times 1 $，其中 $ D_3 $ 是第三层隐含单元的数量，$ D_f $ 是模型输出 $ f_3 $ 的维度。

类似地，我们可计算损失关于 $ f_1 $ 和 $ f_0 $ 的变化率：

$$
\frac{\partial \ell_i}{\partial f_1} = \frac{\partial h_2}{\partial f_1} \frac{\partial f_2}{\partial h_2} \frac{\partial h_3}{\partial f_2} \frac{\partial f_3}{\partial h_3} \frac{\partial \ell_i}{\partial f_3}, \tag{7.19}
$$

$$
\frac{\partial \ell_i}{\partial f_0} = \frac{\partial h_1}{\partial f_0} \frac{\partial f_1}{\partial h_1} \frac{\partial h_2}{\partial f_1} \frac{\partial f_2}{\partial h_2} \frac{\partial h_3}{\partial f_2} \frac{\partial f_3}{\partial h_3} \frac{\partial \ell_i}{\partial f_3}. \tag{7.20}
$$

注意，在每种情况下，括号内的项均已在前一步中计算完成。通过**习题 7.3** 所述方式，沿网络反向遍历，即可复用先前的计算结果。此外，各项本身形式简单。沿式 (7.18) 右侧从右至左反向推导，我们有：

- 损失 $ \ell_i $ 关于网络输出 $ f_3 $ 的导数 $ \partial \ell_i / \partial f_3 $ 取决于所选损失函数，但通常具有简洁的形式；  
- 网络输出 $ f_3 $ 关于隐含层 $ h_3 $ 的导数 $ \partial f_3 / \partial h_3 $ 为：

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

7.4 反向传播算法　105  
$$
\frac{\partial \mathbf{f}_3}{\partial \mathbf{h}_3} = \frac{\partial}{\partial \mathbf{h}_3} (\boldsymbol{\beta}_3 + \boldsymbol{\Omega}_3 \mathbf{h}_3) = \boldsymbol{\Omega}_3^\top. \tag{7.21}
$$  

若你不熟悉矩阵微积分，该结果并不显然。详见习题 7.6。

- 激活函数输出 $\mathbf{h}_3$ 关于其输入 $\mathbf{f}_2$ 的导数 $\partial \mathbf{h}_3 / \partial \mathbf{f}_2$ 取决于所采用的激活函数。由于每个激活值仅依赖于对应的预激活值（pre-activation），该导数必为对角矩阵。对于 ReLU 函数，其对角线元素在 $\mathbf{f}_2$ 的各分量小于零处为 0，其余位置为 1（见图 7.6）。习题 7.7–7.8 对此进行了进一步探讨。为提升计算效率，我们不显式构造并乘以此对角矩阵，而是将其对角线元素提取为向量 $\mathbf{I}[\mathbf{f}_2 > 0]$，再执行逐元素（pointwise）乘法。

公式 (7.19) 和 (7.20) 右侧各项具有相似结构。当沿网络反向传播时，我们交替执行以下两步操作：(i) 左乘权重矩阵 $\boldsymbol{\Omega}_k^\top$ 的转置；(ii) 基于隐层输入 $\mathbf{f}_{k-1}$ 进行阈值化（即 ReLU 的导数作用）。这些输入 $\mathbf{f}_{k-1}$ 在前向传播过程中已被缓存。

**第二次反向传播**：现已掌握如何计算 $\partial \ell / \partial \mathbf{f}_k$，接下来可集中求解损失函数关于权重与偏置的导数。为计算损失函数关于偏置 $\boldsymbol{\beta}_k$ 的导数，再次应用链式法则：

$$
\frac{\partial \ell_i}{\partial \boldsymbol{\beta}_k} = \frac{\partial \mathbf{f}_k}{\partial \boldsymbol{\beta}_k} \frac{\partial \ell_i}{\partial \mathbf{f}_k}
= \frac{\partial}{\partial \boldsymbol{\beta}_k} (\boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_k) \, \frac{\partial \ell_i}{\partial \mathbf{f}_k}
= \frac{\partial \ell_i}{\partial \mathbf{f}_k}, \tag{7.22}
$$

其中 $\partial \ell_i / \partial \mathbf{f}_k$ 已在公式 (7.18) 和 (7.19) 中求得。

类似地，权重矩阵 $\boldsymbol{\Omega}_k$ 的导数为：

$$
\frac{\partial \ell_i}{\partial \boldsymbol{\Omega}_k} = \frac{\partial \mathbf{f}_k}{\partial \boldsymbol{\Omega}_k} \frac{\partial \ell_i}{\partial \mathbf{f}_k}
= \frac{\partial}{\partial \boldsymbol{\Omega}_k} (\boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_k) \, \frac{\partial \ell_i}{\partial \mathbf{f}_k}
= \frac{\partial \ell_i}{\partial \mathbf{f}_k} \, \mathbf{h}_k^\top. \tag{7.23}
$$

从第二行到第三行的推导同样不显然，详见习题 7.9。但最终结果符合直觉：最后一行所得矩阵与 $\boldsymbol{\Omega}_k$ 同维，且线性依赖于 $\mathbf{h}_k$——而 $\mathbf{h}_k$ 正是原始表达式中与 $\boldsymbol{\Omega}_k$ 相乘的隐层输出。这也印证了最初的直观理解：$\boldsymbol{\Omega}_k$ 中各权重的梯度应正比于其所乘隐单元 $\mathbf{h}_k$ 的取值。需注意，这些 $\mathbf{h}_k$ 值已在前向传播中完成计算。

草稿：勘误请发送至 udlbookmail@gmail.com。

106 7 梯度与初始化  
7.4.1 反向传播算法总结  

我们现在简要总结最终的反向传播算法。考虑一个深度神经网络 $ f[\mathbf{x}_i, \boldsymbol{\phi}] $，其输入为 $ \mathbf{x}_i $，包含 $ K $ 个隐含层，各层激活函数均为 ReLU，并具有单个损失项 $ \ell_i = \ell[f[\mathbf{x}_i, \boldsymbol{\phi}], \mathbf{y}_i] $。反向传播的目标是计算损失 $ \ell_i $ 关于偏置 $ \boldsymbol{\beta}_k $ 和权重 $ \boldsymbol{\Omega}_k $ 的导数 $ \partial \ell_i / \partial \boldsymbol{\beta}_k $ 与 $ \partial \ell_i / \partial \boldsymbol{\Omega}_k $。

**前向传播**：我们计算并存储以下量：  
$$
\mathbf{f}_0 = \boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0 \mathbf{x}_i
$$  
$$
\mathbf{h}_k = a[\mathbf{f}_{k-1}], \quad k \in \{1,2,\dots,K\}
$$  
$$
\mathbf{f}_k = \boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_k, \quad k \in \{1,2,\dots,K\} \tag{7.24}
$$  

**反向传播**：我们从损失函数 $ \ell_i $ 关于网络输出 $ \mathbf{f}_K $ 的导数 $ \partial \ell_i / \partial \mathbf{f}_K $ 出发，沿网络逐层回溯：  
$$
\frac{\partial \ell_i}{\partial \boldsymbol{\beta}_k} = \frac{\partial \ell_i}{\partial \mathbf{f}_k}, \quad k \in \{K,K-1,\dots,1\}
$$  
$$
\frac{\partial \ell_i}{\partial \boldsymbol{\Omega}_k} = \frac{\partial \ell_i}{\partial \mathbf{f}_k} \mathbf{h}_k^\top, \quad k \in \{K,K-1,\dots,1\}
$$  
$$
\frac{\partial \ell_i}{\partial \mathbf{f}_{k-1}} = \mathbf{I}[\mathbf{f}_{k-1} > \mathbf{0}] \odot \left( \boldsymbol{\Omega}_k^\top \frac{\partial \ell_i}{\partial \mathbf{f}_k} \right), \quad k \in \{K,K-1,\dots,1\} \tag{7.25}
$$  
其中 $ \odot $ 表示逐元素乘法（Hadamard 积），而 $ \mathbf{I}[\mathbf{f}_{k-1} > \mathbf{0}] $ 是一个指示向量：当 $ \mathbf{f}_{k-1} $ 的对应分量大于零时取值为 1，否则为 0。最后，我们计算关于第一组偏置和权重的导数：  
$$
\frac{\partial \ell_i}{\partial \boldsymbol{\beta}_0} = \frac{\partial \ell_i}{\partial \mathbf{f}_0}
$$  
$$
\frac{\partial \ell_i}{\partial \boldsymbol{\Omega}_0} = \frac{\partial \ell_i}{\partial \mathbf{f}_0} \mathbf{x}_i^\top \tag{7.26}
$$  

我们对批量（batch）中每个训练样本计算上述导数，并将结果累加，从而获得随机梯度下降（SGD）更新所需的梯度。  

需注意，反向传播算法在**计算效率**上极高；前向与反向传播中最耗时的运算均为矩阵乘法（分别涉及 $ \boldsymbol{\Omega}_k $ 与 $ \boldsymbol{\Omega}_k^\top $），仅需加法与乘法操作。然而，该算法在**内存效率**方面表现不佳：前向传播中所有中间变量均需保存，这会限制可训练模型的规模。  

7.4.2 算法微分  

尽管理解反向传播算法至关重要，但在实际工程中，你几乎无需手动实现它。现代深度学习框架（如 PyTorch）  

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（C）麻省理工学院出版社。

7.5 参数初始化　107  
而 TensorFlow 则能在给定模型定义的前提下自动计算导数。  
这被称为**算法微分**（algorithmic differentiation）。  

框架中的每个函数式组件（如线性变换、ReLU 激活函数、损失函数）均内置了其自身导数的计算逻辑。例如，PyTorch 中的 ReLU 函数 $ z_{\text{out}} = \text{relu}(z_{\text{in}}) $ 知道如何计算其输出 $ z_{\text{out}} $ 关于输入 $ z_{\text{in}} $ 的导数；类似地，一个线性函数 $ z_{\text{out}} = \beta + \Omega z_{\text{in}} $ 也能够计算其输出 $ z_{\text{out}} $ 关于输入 $ z_{\text{in}} $ 以及关于参数 $ \beta $ 和 $ \Omega $ 的导数。此外，算法微分框架还掌握网络中各运算的执行顺序，因而具备完成前向传播与反向传播所需的全部信息。  

这些框架充分利用了现代图形处理器（GPU）所具备的大规模并行计算能力。诸如矩阵乘法（在前向与反向传播中均频繁出现）等运算天然适合并行化处理。此外，只要模型本身及其前向传播过程中产生的中间结果未超出可用内存容量，就可对整个批量（batch）同时执行前向与反向传播。  

由于训练算法如今以并行方式处理整个批量，输入数据便成为一种多维张量（tensor）。在此语境下，张量可视为矩阵向任意维度的推广：向量是 1 维张量，矩阵是 2 维张量，而 3 维张量则是一个三维数值网格。此前，训练数据均为 1 维，因此反向传播的输入为一个 2 维张量，其中第一维索引批量中的样本，第二维索引数据特征维度。在后续章节中，我们将遇到结构更为复杂的输入数据。例如，在以 RGB 图像为输入的模型中，原始数据样本本身即为 3 维（高度 × 宽度 × 通道数），此时学习框架的输入则为一个 4 维张量，额外增加的一维用于索引批量中的样本。  

7.4.3 推广至任意计算图  
我们此前描述的反向传播，针对的是天然具有顺序结构的深度神经网络：我们依次计算中间量 $ f_0, h_1, f_1, h_2, \dots, f_k $。然而，模型并不局限于顺序计算。本书后续章节将介绍具有分支结构的模型。例如，我们可能将某隐层的输出值分别送入两个不同的子网络进行处理，再将结果重新合并。  
**习题 7.12–7.13**  

幸运的是，只要计算图是**无环的**（acyclic），反向传播的核心思想依然成立。当前主流的算法微分框架（如 PyTorch 和 TensorFlow）均能处理任意无环计算图。  

7.5 参数初始化  
反向传播算法所计算的导数，被随机梯度下降（SGD）和 Adam 等优化算法用于模型训练。接下来，我们讨论在开始训练之前应如何初始化模型参数。理解该步骤的重要性至关重要：在前向传播过程中，每一层的预激活值 $ f_k $ 均按如下方式计算：  
$$
f_k = \cdots
$$  
草稿：如有勘误，请发送邮件至 udlbookmail@gmail.com。

108 7 梯度与参数初始化  
$$
\mathbf{f}_k = \boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_k \\
= \boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \, a[\mathbf{f}_{k-1}], \tag{7.27}
$$  
其中 $a[\cdot]$ 表示逐元素应用的 ReLU 函数，$\boldsymbol{\Omega}_k$ 和 $\boldsymbol{\beta}_k$ 分别为第 $k$ 层的权重矩阵和偏置向量。假设我们将所有偏置初始化为零，并将 $\boldsymbol{\Omega}_k$ 的各元素独立采样自均值为 0、方差为 $\sigma^2$ 的正态分布。考虑以下两种情形：  

- 若方差 $\sigma^2$ 极小（例如 $10^{-5}$），则 $\boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_k$ 的每个元素均为 $\mathbf{h}_k$ 的加权和，且权重极小；其结果的幅值很可能小于输入。此外，ReLU 函数会将负值截断为零，因此 $\mathbf{h}_k$ 的取值范围仅为 $\mathbf{f}_{k-1}$ 的一半。于是，随着网络深度增加，隐层中前激活值（pre-activation）的幅值将逐层衰减。  

- 若方差 $\sigma^2$ 极大（例如 $10^5$），则 $\boldsymbol{\beta}_k + \boldsymbol{\Omega}_k \mathbf{h}_k$ 的每个元素同样是 $\mathbf{h}_k$ 的加权和，但此时权重极大；其结果的幅值很可能远超输入。尽管 ReLU 函数使输入的有效取值范围减半，但只要 $\sigma^2$ 足够大，前激活值的幅值仍将在网络前向传播过程中逐层放大。  

在这两种情形下，前激活值可能变得过小或过大，以致无法用有限精度的浮点数准确表示。  

即使前向传播过程在数值上尚可处理，上述分析同样适用于反向传播过程。每次梯度更新（式 (7.25)）均涉及左乘权重矩阵的转置 $\boldsymbol{\Omega}^\top$。若 $\boldsymbol{\Omega}$ 的初始值设置不合理，则反向传播过程中梯度幅值可能失控地衰减或增长。前者称为**梯度消失问题**（vanishing gradient problem），后者称为**梯度爆炸问题**（exploding gradient problem）。在梯度消失情形下，模型参数的更新量趋近于零；而在梯度爆炸情形下，更新过程则变得不稳定。  

### 7.5.1 前向传播的参数初始化  

下面我们以更严谨的数学形式重述上述论证。考虑相邻两层前激活值 $\mathbf{f}$ 与 $\mathbf{f}'$，其维度分别为 $D_h$ 和 $D_{h'}$：  
$$
\mathbf{h} = a[\mathbf{f}], \\
\mathbf{f}' = \boldsymbol{\beta} + \boldsymbol{\Omega} \mathbf{h}, \tag{7.28}
$$  
其中 $\mathbf{h}$ 表示激活值，$\boldsymbol{\Omega}$ 和 $\boldsymbol{\beta}$ 分别为权重矩阵与偏置向量，$a[\cdot]$ 为激活函数。  

假设输入层的前激活值 $\mathbf{f}$ 中各元素具有方差 $\sigma_f^2$。现设偏置 $\boldsymbol{\beta}$ 全部初始化为零，权重 $\boldsymbol{\Omega}_{ij}$ 独立采样自均值为 0、方差为 $\sigma_\Omega^2$ 的正态分布。接下来，我们推导后续层前激活值 $\mathbf{f}'$ 的均值与方差表达式。  

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

7.5 参数初始化　109  
中间变量 $ f'_i $ 的期望（均值）$ \mathbb{E}[f'_i] $ 为：附录 C.2  
$$
\mathbb{E}[f'_i] = \mathbb{E}\left[ \beta_i + \sum_{j=1}^{D_h} \Omega_{ij} h_j \right]
= \mathbb{E}[\beta_i] + \sum_{j=1}^{D_h} \mathbb{E}[\Omega_{ij} h_j]
= \mathbb{E}[\beta_i] + \sum_{j=1}^{D_h} \mathbb{E}[\Omega_{ij}]\,\mathbb{E}[h_j]
= 0 + 0 \cdot \mathbb{E}[h_j] = 0, \tag{7.29}
$$  
其中 $ D_h $ 为输入层 $ \mathbf{h} $ 的维度。我们使用了期望运算的代数规则（附录 C.2.1），并在第二行到第三行之间假设隐单元 $ h_j $ 与网络权重 $ \Omega_{ij} $ 的分布相互独立。  

利用该结果，可得预激活值 $ f'_i $ 的方差 $ \sigma^2_{f'} $：  
$$
\sigma^2_{f'} = \mathbb{E}\big[(f'_i)^2\big] - \big(\mathbb{E}[f'_i]\big)^2
= \mathbb{E}\left[ \left( \beta_i + \sum_{j=1}^{D_h} \Omega_{ij} h_j \right)^2 \right] - 0
= \mathbb{E}\left[ \left( \sum_{j=1}^{D_h} \Omega_{ij} h_j \right)^2 \right]
= \sum_{j=1}^{D_h} \mathbb{E}\big[\Omega_{ij}^2\big]\,\mathbb{E}\big[h_j^2\big]
= \sum_{j=1}^{D_h} \sigma^2_\Omega\,\mathbb{E}\big[h_j^2\big]
= \sigma^2_\Omega \sum_{j=1}^{D_h} \mathbb{E}\big[h_j^2\big], \tag{7.30}
$$  
其中我们使用了方差恒等式 $ \sigma^2_z = \mathbb{E}\big[(z - \mathbb{E}[z])^2\big] = \mathbb{E}[z^2] - \big(\mathbb{E}[z]\big)^2 $（附录 C.2.3）。在第三行到第四行之间，我们再次假设权重 $ \Omega_{ij} $ 与隐单元 $ h_j $ 的分布相互独立。  

进一步假设前一层预激活值 $ f_j $ 的分布关于零点对称，则 ReLU 函数将截断其中一半的预激活值；此时隐单元的二阶矩 $ \mathbb{E}[h_j^2] $ 将等于 $ f_j $ 方差 $ \sigma^2_f $ 的一半（参见习题 7.14）：  
$$
\sigma^2_{f'} = \sigma^2_\Omega \sum_{j=1}^{D_h} \mathbb{E}[h_j^2]
= \sigma^2_\Omega \sum_{j=1}^{D_h} \frac{\sigma^2_f}{2}
= \sigma^2_\Omega \cdot D_h \cdot \frac{\sigma^2_f}{2}. \tag{7.31}
$$  

草稿：请将勘误发送至 udlbookmail@gmail.com。

110 7 梯度与参数初始化  
图7.7 权重初始化。考虑一个具有50个隐藏层的深度网络，每层包含 $ D = 100 $ 个隐藏单元。该网络接收一个 $ 100 $ 维输入 $ \mathbf{x} $，其各分量独立采样自标准正态分布；设定单一固定目标值 $ y = 0 $；损失函数采用最小二乘形式。偏置向量 $ \boldsymbol{\beta}_k $ 全部初始化为零；权重矩阵 $ \boldsymbol{\Omega}_k $ 则从均值为零、方差 $ \sigma^2_\Omega $ 取以下五种不同值的正态分布中采样：$ \sigma^2_\Omega \in \{0.001,\, 0.01,\, 0.02,\, 0.1,\, 1.0\} $。  
a) 前向传播过程中各隐藏单元激活值的方差随网络层数的变化关系。对于 He 初始化（$ \sigma^2_\Omega = 2/D = 0.02 $），该方差保持稳定；而当初始方差取值更大时，方差迅速增长；取值更小时，则迅速衰减（注意纵轴为对数刻度）。  
b) 反向传播过程中梯度的方差（实线）延续了这一趋势：若以大于 $ 0.02 $ 的方差初始化权重，则随着梯度沿网络反向传递，其幅值将急剧增大；若以小于 $ 0.02 $ 的方差初始化，则梯度幅值将急剧衰减。这两种现象分别被称为**梯度爆炸问题**（exploding gradient problem）和**梯度消失问题**（vanishing gradient problem）。  

由此可得：若希望后续层预激活值 $ f' $ 的方差 $ \sigma^2_{f'} $ 在前向传播中与原始层预激活值 $ f $ 的方差 $ \sigma^2_f $ 保持一致，则应令：  
$$
\sigma^2_\Omega = \frac{2}{D_h}, \tag{7.32}
$$  
其中 $ D_h $ 表示权重所作用的原始层的维度。该策略称为 **He 初始化**（He initialization）。  

7.5.2 反向传播的初始化策略  
类似地，我们可推导梯度 $ \partial l / \partial \mathbf{f}_k $ 在反向传播过程中方差的变化规律。在反向传播中，我们需乘以权重矩阵的转置 $ \boldsymbol{\Omega}^\top $（见式 (7.25)），因此对应的表达式变为：  

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

7.6 示例训练代码　111  
$$
\sigma^2 = \frac{2}{\Omega D_{h'}}, \tag{7.33}
$$  
其中 $D_{h'}$ 表示权重所连接的层的维度。  

7.5.3 前向传播与反向传播均适用的初始化方法  
若权重矩阵 $\Omega$ 不是方阵（即相邻两层的隐单元数量不同，因而 $D_h$ 与 $D_{h'}$ 不相等），则无法同时满足式（7.32）和式（7.33）所要求的方差条件。一种可行的折中方案是：以平均值 $(D_h + D_{h'})/2$ 作为项数的代理，从而得到：  
$$
\sigma^2 = \frac{4}{\Omega (D_h + D_{h'})}. \tag{7.34}
$$  
图 7.7 通过实验表明，当参数经适当初始化后，前向传播中隐单元的方差以及反向传播中梯度的方差均能保持稳定。  

**习题 7.15**  
**Notebook 7.3：初始化**  

7.6 示例训练代码  
本书的核心定位是科学性探讨；它并非深度学习模型实现的实践指南。尽管如此，图 7.8 中仍给出了一个 PyTorch 实现代码，用以落实本书迄今所探讨的核心思想。该代码定义了一个神经网络，并对权重进行了初始化；生成了随机的输入与输出数据集，并定义了一个最小二乘损失函数；随后，模型在数据上采用动量随机梯度下降法（SGD with momentum）进行训练：批量大小为 10，共训练 100 轮（epochs）；学习率初始值设为 0.01，并每 10 轮衰减为原来的一半。  

关键启示在于：尽管深度学习背后的理论思想相当复杂，其实际实现却相对简洁。例如，反向传播的所有细节均被封装在单行代码 `loss.backward()` 中。  

7.7 小结  
上一章介绍了随机梯度下降（Stochastic Gradient Descent, SGD）——一种旨在寻找函数最小值的迭代优化算法。在神经网络语境下，该算法用于搜寻使损失函数最小化的模型参数。SGD 的执行依赖于损失函数关于各参数的梯度；而这些参数必须在优化开始前完成初始化。本章即针对深度神经网络，系统地解决了上述两个关键问题：梯度计算与参数初始化。  

对于每个批量样本、每次 SGD 迭代，都需要对海量参数分别求取梯度。因此，高效、准确地计算梯度至关重要。  

*草稿：勘误请发送至 udlbookmail@gmail.com。*

112 7 梯度与参数初始化  
```python
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# 定义输入维度、隐藏层维度、输出维度
D_i, D_k, D_o = 10, 40, 5

# 构建含两个隐藏层的模型
model = nn.Sequential(
    nn.Linear(D_i, D_k),
    nn.ReLU(),
    nn.Linear(D_k, D_k),
    nn.ReLU(),
    nn.Linear(D_k, D_o)
)

# 使用 He 初始化（即 Kaiming 正态初始化）对权重进行初始化
def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_normal_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)

model.apply(weights_init)

# 选用最小二乘损失函数
criterion = nn.MSELoss()

# 构造随机梯度下降（SGD）优化器，并设置初始学习率与动量
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# 学习率调度器：每经过 10 个 epoch，学习率衰减为当前值的一半
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# 生成 100 个随机数据点，并封装为 DataLoader
x = torch.randn(100, D_i)
y = torch.randn(100, D_o)
data_loader = DataLoader(TensorDataset(x, y), batch_size=10, shuffle=True)

# 在数据集上训练 100 轮（epoch）
for epoch in range(100):
    epoch_loss = 0.0
    # 遍历每个 mini-batch
    for i, data in enumerate(data_loader):
        # 提取当前 batch 的输入与标签
        x_batch, y_batch = data
        # 清零参数梯度
        optimizer.zero_grad()
        # 前向传播
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        # 反向传播
        loss.backward()
        # 执行 SGD 参数更新
        optimizer.step()
        # 累计统计本 epoch 的损失
        epoch_loss += loss.item()
    # 输出当前 epoch 的平均损失
    print(f'Epoch {epoch:5d}, loss {epoch_loss:.3f}')

    # 通知调度器考虑更新学习率
    scheduler.step()
```

图 7.8 在随机数据上训练两层网络的示例代码。  
本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

注释 113  
计算效率至关重要，为此引入了反向传播算法。  
参数的谨慎初始化同样极为关键。隐藏单元激活值的幅值在前向传播过程中可能呈指数级衰减或增长；同样地，在反向传播过程中，梯度幅值也会表现出类似行为，分别被称为**梯度消失问题**与**梯度爆炸问题**。二者均会严重阻碍模型训练，但可通过恰当的初始化策略予以规避。  

至此，我们已明确定义了模型结构与损失函数，从而可针对特定任务开展模型训练。下一章将讨论如何评估模型性能。  

---

**注释**  

**反向传播**：在计算图中高效复用部分计算结果以求取梯度的思想曾被多次独立发现，包括 Werbos（1974）、Bryson 等（1979）、LeCun（1985）以及 Parker（1985）。然而，该思想最具影响力的阐述出自 Rumelhart 等（1985）及 Rumelhart 等（1986），后者亦首次提出“反向传播”（backpropagation）这一术语。后一工作开启了神经网络研究在 20 世纪 80 至 90 年代的新阶段——人们首次得以切实可行地训练含隐藏层的网络。然而，后续进展陷入停滞（事后看来），主要原因在于训练数据匮乏、计算能力有限，以及广泛采用 S 型（sigmoid）激活函数。自然语言处理与计算机视觉等领域直至 Krizhevsky 等（2012）取得突破性的图像分类成果之后，才真正开始普遍采用神经网络模型，由此 ushered in（开启）了深度学习的现代纪元。  

现代深度学习框架（如 PyTorch 和 TensorFlow）中反向传播的实现，是**反向模式自动微分**（reverse-mode algorithmic differentiation）的一个典型范例。这与**正向模式自动微分**（forward-mode algorithmic differentiation）相区别：后者在沿计算图前向遍历时即累积链式法则导出的导数（参见习题 7.13）。关于自动微分的更多内容，可参考 Griewank & Walther（2008）及 Baydin 等（2018）。  

**初始化方法**：He 初始化由 He 等（2015）首次提出。它在 Glorot（又称 Xavier）初始化（Glorot & Bengio，2010）基础上发展而来，二者高度相似，但 He 初始化额外考虑了 ReLU 层的影响，因此其方差缩放因子相差两倍。本质上相同的方法其实更早由 LeCun 等（2012）提出，但动机略有不同：当时采用的是 S 型激活函数，其天然具备归一化各层输出范围的能力，因而有助于防止隐藏单元幅值呈指数增长。然而，若预激活值过大，则会落入 S 型函数的饱和平坦区，导致梯度极小。因此，权重的合理初始化依然至关重要。Klambauer 等（2017）提出了缩放指数线性单元（scaled exponential linear unit, SeLU），并证明：在一定输入范围内，该激活函数可促使网络各层激活值自动收敛至均值为 0、方差为 1 的分布。  

另一种截然不同的思路是：先将数据前向通过网络，再依据经验观测到的方差进行归一化。“层序单位方差初始化”（Layer-sequential unit variance initialization，Mishkin & Matas，2016）即属此类方法，其将权重矩阵初始化为正交矩阵。GradInit（Zhu 等，2021）则对初始权重进行随机化，并在训练初期将其暂时固定，同时为每个权重矩阵学习一组非负缩放因子；这些因子的选择目标是在给定学习率下最大化损失下降量，且满足最大梯度范数约束。**激活归一化**（Activation normalization，ActNorm）则在每一隐藏单元处、每层网络之后引入可学习的缩放与偏移参数。具体做法是：先将一个初始批次数据送入网络，据此选定偏移量与缩放因子，使得激活值的均值为 0、方差为 1；此后，这些额外参数便作为模型的一部分参与联合学习。  

草稿：如有勘误，请发送邮件至 udlbookmail@gmail.com。

114 7 梯度与参数初始化  
与上述方法密切相关的是批归一化（BatchNorm，Ioffe & Szegedy，2015）等方案，该方法在每一层的前向传播过程中，将当前批次数据的方差进行归一化处理。BatchNorm 及其变体将在第 11 章中详细讨论。此外，针对特定网络架构还提出了若干专用的初始化方案，例如：面向卷积神经网络的卷积正交初始化器（ConvolutionOrthogonal initializer，Xiao 等，2018a）；面向残差网络的 Fixup 初始化（Zhang 等，2019a）；以及面向 Transformer 架构的 TFixup（Huang 等，2020a）和 DTFixup（Xu 等，2021b）。

降低内存开销：神经网络训练对内存资源要求极高。在前向传播过程中，我们必须为批次中的每个样本同时存储模型参数以及各隐藏单元的预激活值（pre-activations）。两种可显著降低内存需求的方法是梯度检查点技术（gradient checkpointing，Chen 等，2016a）与微批次（micro-batching，Huang 等，2019）。在梯度检查点技术中，仅每隔 $N$ 层才保存一次前向传播的激活值；而在反向传播阶段，则从最近的检查点出发，重新计算缺失的中间激活值。由此，我们可在计算代价仅为两次前向传播（见习题 7.11）的前提下，大幅削减内存占用。在微批次方法中，原始批次被划分为若干更小的子批次，各子批次分别计算梯度，并在应用于网络之前进行聚合。另一种截然不同的思路是构建可逆网络（reversible network，例如 Gomez 等，2017），其中前一层的激活值可由当前层的激活值精确重构，因而前向传播过程中无需缓存任何中间结果（参见第 16 章）。Sohoni 等（2019）对各类降低内存需求的技术进行了系统综述。

分布式训练：对于规模足够大的模型，单个处理器可能无法满足其内存需求或总训练耗时要求。此时必须采用分布式训练，即在多个处理器上并行执行训练过程。实现并行化的方式主要有多种。在**数据并行**（data parallelism）中，每个处理器（或节点）均保存完整的模型副本，但仅处理批次的一个子集（参见 Xing 等，2015；Li 等，2020b）。各节点计算出的梯度被集中聚合后，再分发回所有节点，以确保各模型副本保持一致——此即所谓**同步训练**（synchronous training）。然而，梯度聚合与再分发所需的同步操作可能成为性能瓶颈，由此催生了**异步训练**（asynchronous training）的思想。例如，在 Hogwild! 算法（Recht 等，2011）中，任一节点一旦完成梯度计算，便立即用于更新中心模型；更新后的模型随即被重新分发至该节点。这意味着任意时刻各节点所持模型版本可能存在细微差异，导致梯度更新存在“陈旧性”（staleness）；但在实践中该方法仍表现良好。其他去中心化方案亦已被提出。例如，在 Zhang 等（2016a）的工作中，各节点以环状拓扑结构相互更新模型参数。  

需注意，数据并行方法仍隐含一个前提：整个模型必须能完整装入单个节点的内存。相比之下，**流水线模型并行**（pipeline model parallelism）将网络的不同层分别部署于不同节点之上，因而无此限制。在朴素实现中，首个节点先对批次数据在前几层执行前向传播，并将结果传递给下一节点；后者继而在后续几层上执行前向传播，依此类推；反向传播则按相反顺序逐层更新梯度。该方法的明显缺陷在于：每个计算节点在大部分时间处于空闲状态。为此，研究者提出了多种优化方案，其核心思想是令各节点按序处理微批次（micro-batches），从而缓解该低效问题（例如 Huang 等，2019；Narayanan 等，2021a）。最后，在**张量模型并行**（tensor model parallelism）中，单个网络层内部的计算被跨节点分布执行（例如 Shoeybi 等，2019）。Narayanan 等（2021b）提供了一份关于分布式训练方法的优秀综述，他们融合张量并行、流水线并行与数据并行三种策略，在 3072 块 GPU 上成功训练了一个含一万亿参数的语言模型。

习题  
习题 7.1 考虑一个两层网络，每层包含两个隐藏单元，其定义如下：  
本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

注释 115  
$$
y = \phi_0 + \phi_1 a\!\left[\psi_{01} + \psi_{11} x\right] + \psi_{21} a\!\left[\theta_{02} + \theta_{12} x\right] \\
+ \phi_2 a\!\left[\psi_{02} + \psi_{12} x\right] + \psi_{22} a\!\left[\theta_{02} + \theta_{12} x\right], \quad (7.35)
$$  
其中函数 $a[\bullet]$ 为 ReLU 函数。请直接（即不使用反向传播算法）计算输出 $y$ 关于全部 13 个参数 $\phi_\bullet$、$\theta_{\bullet\bullet}$ 和 $\psi_{\bullet\bullet}$ 的偏导数。ReLU 函数关于其输入的导数 $\partial a[z]/\partial z$ 为指示函数 $I[z > 0]$，即当自变量大于零时返回 1，否则返回 0（见图 7.6）。

**习题 7.2**  
求式 (7.13) 中五条导数链各自最后一项的表达式。

**习题 7.3**  
式 (7.20) 中各项的维度（大小）各是多少？

**习题 7.4**  
对最小二乘损失函数  
$$
\ell_i = \left(y_i - f[x_i,\phi]\right)^2 \quad (7.36)
$$  
计算导数 $\partial \ell_i / \partial f[x_i,\phi]$。

**习题 7.5**  
对二分类损失函数  
$$
\ell_i = -(1-y_i)\log\!\left[1-\mathrm{sig}\!\left(f[x_i,\phi]\right)\right] - y_i \log\!\left[\mathrm{sig}\!\left(f[x_i,\phi]\right)\right], \quad (7.37)
$$  
计算导数 $\partial \ell_i / \partial f[x_i,\phi]$。其中函数 $\mathrm{sig}[\bullet]$ 为逻辑 Sigmoid 函数，定义为：  
$$
\mathrm{sig}[z] = \frac{1}{1+\exp[-z]}. \quad (7.38)
$$

**习题 7.6\***  
设 $z = \beta + \Omega h$，试证明：  
$$
\frac{\partial z}{\partial h} = \Omega^\top, \quad (7.39)
$$  
其中 $\partial z/\partial h$ 是一个矩阵，其第 $i$ 列第 $j$ 行元素为 $\partial z_i / \partial h_j$。为此，请先推导出各分量 $\partial z_i / \partial h_j$ 的表达式，再分析矩阵 $\partial z/\partial h$ 所应具有的形式。

**习题 7.7**  
考虑以逻辑 Sigmoid 函数（见式 7.38）作为激活函数的情形，即 $h = \mathrm{sig}[f]$。试计算该激活函数的导数 $\partial h / \partial f$。当输入分别取（i）很大的正值和（ii）很大的负值时，该导数将如何变化？

**习题 7.8**  
考虑分别采用（i）Heaviside 函数与（ii）矩形函数作为激活函数的情形：  
$$
\mathrm{Heaviside}[z] = 
\begin{cases}
0 & z < 0, \\
1 & z \geq 0,
\end{cases}
\quad (7.40)
$$  
草稿：勘误请发送至 udlbookmail@gmail.com。

116 7 梯度与初始化  
图7.9 问题7.12与问题7.13所对应的计算图。改编自Domke（2010）。  

且  
$$
\mathrm{rect}[z] = 
\begin{cases}
0 & z < 0 \\
1 & 0 \leq z \leq 1 \\
0 & z > 1
\end{cases}
\quad (7.41)
$$  

讨论为何这类函数在采用基于梯度的优化方法训练神经网络时会带来困难。  

**问题7.9\*** 考虑一个损失函数 $\ell[f]$，其中 $f = \beta + \Omega h$。我们希望分析当 $\Omega$ 发生变化时，损失 $\ell$ 如何随之变化；该变化率可用一个矩阵表示，其第 $i$ 行、第 $j$ 列元素为偏导数 $\partial \ell / \partial \Omega_{ij}$。请推导出 $\partial f_i / \partial \Omega_{ij}$ 的表达式，并利用链式法则证明：  
$$
\frac{\partial \ell}{\partial \Omega} = \frac{\partial \ell}{\partial f} \, h^\top. \quad (7.42)
$$  

**问题7.10\*** 推导采用带泄露线性整流单元（leaky ReLU）激活函数的网络在反向传播算法中反向传递阶段的更新方程。leaky ReLU 定义如下：  
$$
a[z] = \mathrm{ReLU}[z] = 
\begin{cases}
\alpha \cdot z & z < 0 \\
z & z \geq 0
\end{cases}
\quad (7.43)
$$  
其中 $\alpha$ 是一个较小的正常数（通常取值为 $0.1$）。  

**问题7.11** 考虑使用梯度检查点（gradient checkpointing）技术训练一个包含五十层的网络。假设在前向传播过程中，我们仅在每第十个隐层处保存其预激活值（pre-activation）。请说明在此设定下如何计算各参数的梯度。  

**问题7.12\*** 本题探讨在一般有向无环计算图（acyclic computational graph）上计算导数的方法。考虑如下函数：  
$$
y = \exp\!\left( \exp[x] + \exp[x]^2 \right) + \sin\!\left( \exp[x] + \exp[x]^2 \right). \quad (7.44)
$$  
我们可以将其分解为一系列中间计算步骤，使得：  

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

注释 117  
$$
f_1 = \exp[x] \\
f_2 = f_1^2 \\
f_3 = f_1 + f_2 \\
f_4 = \exp[f_3] \\
f_5 = \sin[f_3] \\
y = f_4 + f_5. \tag{7.45}
$$  
对应的计算图如图 7.9 所示。请使用反向模式自动微分（reverse-mode differentiation）计算导数 $\partial y/\partial x$。换言之，需按如下顺序依次计算：  
$$
\frac{\partial y}{\partial f_5},\; \frac{\partial y}{\partial f_4},\; \frac{\partial y}{\partial f_3},\; \frac{\partial y}{\partial f_2},\; \frac{\partial y}{\partial f_1},\; \frac{\partial y}{\partial x}, \tag{7.46}
$$  
每一步均利用链式法则，复用此前已计算出的导数值。

**习题 7.13∗**  
对习题 7.12 中定义的同一函数，试采用前向模式自动微分（forward-mode differentiation）计算导数 $\partial y/\partial x$。即按如下顺序依次计算：  
$$
\frac{\partial f_1}{\partial x},\; \frac{\partial f_2}{\partial x},\; \frac{\partial f_3}{\partial x},\; \frac{\partial f_4}{\partial x},\; \frac{\partial f_5}{\partial x},\; \frac{\partial y}{\partial x}, \tag{7.47}
$$  
每一步同样利用链式法则，复用此前已计算出的导数值。  
为何在深度神经网络中计算参数梯度时通常不采用前向模式自动微分？

**习题 7.14**  
考虑一个随机变量 $a$，其方差为 $\mathrm{Var}[a] = \sigma^2$，且其分布关于均值 $\mathbb{E}[a] = 0$ 对称。证明：若将该变量通过 ReLU 函数变换：  
$$
b = \mathrm{ReLU}[a] =
\begin{cases}
0, & a < 0 \\
a, & a \geq 0
\end{cases} \tag{7.48}
$$  
则变换后变量的二阶矩满足 $\mathbb{E}[b^2] = \sigma^2 / 2$。

**习题 7.15**  
若将网络中所有权重和偏置项均初始化为零，你预期会发生什么现象？

**习题 7.16**  
请基于图 7.8 中的代码，在 PyTorch 中实现该模型，并绘制训练损失随训练轮数（epoch）变化的曲线图。

**习题 7.17**  
修改图 7.8 中的代码，使其适用于二分类任务。你需要完成以下三项调整：（i）将目标标签 $y$ 改为二值型；（ii）修改网络结构，使其输出值位于区间 $[0, 1]$ 内；（iii）相应地更换损失函数。

勘误反馈请发送至：udlbookmail@gmail.com。