# 第4章：最终输出是这些隐单元的线性组合（公式4.9）。

*页码：60–62*

46 4 深度神经网络  
考察这些方程，可引出另一种理解网络如何构建日益复杂函数的视角（图4.5）：  
**Notebook4.2**  
**截断函数（Clipping functions）**  

1. 第一层的三个隐单元 $h_1$、$h_2$ 和 $h_3$ 按常规方式计算：先对输入构造线性函数，再将结果通过 ReLU 激活函数（式(4.7)）。  
2. 第二层的预激活值（pre-activations）通过对上述三个隐单元再次构造三个新的线性函数得到（即式(4.8)中激活函数的输入参数）。此时，我们实质上已构建了一个具有三个输出的浅层网络；我们已计算出三个分段线性函数，且这些函数在线性区域之间的“连接点”（joints）位置完全一致（参见图3.6）。  
3. 在第二隐层，对每个函数再次应用 ReLU 激活函数 $a[\cdot]$（式(4.8)），从而对各函数进行截断，并在每条函数曲线上引入新的“连接点”。  
4. 最终输出是这些隐单元的线性组合（式(4.9)）。  

综上所述，我们既可将每一层理解为对输入空间进行“折叠”，也可理解为生成新函数——这些函数经截断（从而产生新的分段区域）后再被重新组合。前一种观点强调输出函数中各部分之间的依赖关系，但未体现截断如何生成新的连接点；后一种观点则恰好相反。最终，这两种描述都仅能部分揭示深度神经网络的工作机理。尽管如此，我们必须始终牢记：这本质上仍不过是一个将输入 $\mathbf{x}$ 映射到输出 $y'$ 的数学方程。事实上，我们可以将式(4.7)–(4.9)合并，得到如下单一表达式：  

$$
\begin{aligned}
y' =\ & \phi'_0 + \phi'_1 a\big[\psi_{10} + \psi_{11} a[\theta_{10} + \theta_{11} x] + \psi_{12} a[\theta_{20} + \theta_{21} x] + \psi_{13} a[\theta_{30} + \theta_{31} x]\big] \\
& + \phi'_2 a\big[\psi_{20} + \psi_{21} a[\theta_{10} + \theta_{11} x] + \psi_{22} a[\theta_{20} + \theta_{21} x] + \psi_{23} a[\theta_{30} + \theta_{31} x]\big] \\
& + \phi'_3 a\big[\psi_{30} + \psi_{31} a[\theta_{10} + \theta_{11} x] + \psi_{32} a[\theta_{20} + \theta_{21} x] + \psi_{33} a[\theta_{30} + \theta_{31} x]\big],
\end{aligned}
$$  
(4.10)  

诚然，该表达式理解起来相当困难。  

### 4.3.1 超参数  
我们可以将深度网络结构扩展至两个以上隐层；现代网络可能拥有逾百层，且每层包含数千个隐单元。每层中隐单元的数量称为网络的**宽度（width）**，隐层的总数则称为网络的**深度（depth）**。所有隐单元的总数则是衡量网络**容量（capacity）** 的一个指标。  

我们将网络层数记为 $K$，第 $k$ 层的隐单元数记为 $D_k$（$k = 1,2,\dots,K$）。这些量均属于**超参数（hyperparameters）**：它们是在学习模型参数（即斜率与截距项）之前预先选定的量。对于固定的超参数（例如 $K = 2$ 层，且每层 $D_k = 3$ 个隐单元），模型所刻画的是一个函数族（family of functions），而具体参数则决定了该族中的某一个特定函数。因此，若同时考虑超参数，则可将神经网络视为表示一类“函数族的族”（a family of families of functions），用以建立输入到输出之间的映射关系。  

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社（MIT Press）。

4.3 深度神经网络　47  
图4.5　图4.4中深度网络的计算过程。  
a–c）第二隐含层的输入（即预激活值）是三个分段线性函数，其线性区段之间的“拐点”位于相同位置（参见图3.6）。  
d–f）每个分段线性函数均经ReLU激活函数截断至零。  
g–i）这些被截断后的函数分别以参数 $\phi'_1$、$\phi'_2$ 和 $\phi'_3$ 加权。  
j）最后，将加权后的截断函数求和，并加上一个偏置项 $\phi'_0$，该偏置用于调控整体输出的高度。  
（交互式图表）  
草稿：如有勘误，请发送至 udlbookmail@gmail.com。

48 4 深度神经网络  
图4.6　矩阵表示法：网络具有 $D_i = 3$ 维输入 $\mathbf{x}$、$D_o = 2$ 维输出 $\mathbf{y}$，以及 $K = 3$ 个隐含层 $\mathbf{h}_1$、$\mathbf{h}_2$ 和 $\mathbf{h}_3$，其维度分别为 $D_1 = 4$、$D_2 = 2$ 和 $D_3 = 3$。权重存储于矩阵 $\boldsymbol{\Omega}_k$ 中，用于将前一层的激活值相乘，从而生成下一层的预激活值（pre-activations）。例如，计算从第1层激活值 $\mathbf{h}_1$ 到第2层预激活值 $\mathbf{h}_2$ 的权重矩阵 $\boldsymbol{\Omega}_1$ 具有 $2 \times 4$ 的维度；它作用于第1层的4个隐单元，并生成第2层两个隐单元的输入。偏置项存储于向量 $\boldsymbol{\beta}_k$ 中，其维度与所注入的目标层维度一致。例如，偏置向量 $\boldsymbol{\beta}_2$ 的长度为3，因为第3层隐含层 $\mathbf{h}_3$ 包含3个隐单元。

4.4 矩阵表示法  
我们已知，深度神经网络由线性变换与激活函数交替构成（参见附录B.3“矩阵”）。因此，公式(4.7)–(4.9)亦可等价地以矩阵形式表述如下：

$$
\begin{bmatrix}
\mathbf{h}_1 \\
\mathbf{h}_2 \\
\mathbf{h}_3
\end{bmatrix}
=
a\left(
\begin{bmatrix}
\boldsymbol{\theta}_{10} \\
\boldsymbol{\theta}_{20} \\
\boldsymbol{\theta}_{30}
\end{bmatrix}
+
\begin{bmatrix}
\boldsymbol{\theta}_{11} \\
\boldsymbol{\theta}_{21} \\
\boldsymbol{\theta}_{31}
\end{bmatrix}
\mathbf{x}
\right),
\quad\text{(4.11)}
$$

$$
\begin{bmatrix}
\mathbf{h}'_1 \\
\mathbf{h}'_2 \\
\mathbf{h}'_3
\end{bmatrix}
=
a\left(
\begin{bmatrix}
\boldsymbol{\psi}_{10} \\
\boldsymbol{\psi}_{20} \\
\boldsymbol{\psi}_{30}
\end{bmatrix}
+
\begin{bmatrix}
\boldsymbol{\psi}_{11} & \boldsymbol{\psi}_{12} & \boldsymbol{\psi}_{13} \\
\boldsymbol{\psi}_{21} & \boldsymbol{\psi}_{22} & \boldsymbol{\psi}_{23} \\
\boldsymbol{\psi}_{31} & \boldsymbol{\psi}_{32} & \boldsymbol{\psi}_{33}
\end{bmatrix}
\begin{bmatrix}
\mathbf{h}_1 \\
\mathbf{h}_2 \\
\mathbf{h}_3
\end{bmatrix}
\right),
\quad\text{(4.12)}
$$

以及

$$
\mathbf{y}' = \phi'\!\left(
\boldsymbol{\phi}'_0 +
\begin{bmatrix}
\boldsymbol{\phi}'_1 & \boldsymbol{\phi}'_2 & \boldsymbol{\phi}'_3
\end{bmatrix}
\begin{bmatrix}
\mathbf{h}'_1 \\
\mathbf{h}'_2 \\
\mathbf{h}'_3
\end{bmatrix}
\right).
\quad\text{(4.13)}
$$

本作品采用知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。© MIT出版社。