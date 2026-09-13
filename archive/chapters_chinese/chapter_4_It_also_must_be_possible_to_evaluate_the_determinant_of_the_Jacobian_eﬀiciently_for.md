# 第4章：还必须能够高效地计算雅可比矩阵的行列式

*页码：323–340*

16.3 可逆网络层　309  
$$
\left| \frac{\partial f[z,\phi]}{\partial z} \right| = 
\left| \frac{\partial f_1[z,\phi_1]}{\partial z} \right| \cdot
\left| \frac{\partial f_2[f_1,\phi_2]}{\partial f_1} \right| \cdots
\left| \frac{\partial f_{K-1}[f_{K-2},\phi_{K-1}]}{\partial f_{K-2}} \right| \cdot
\left| \frac{\partial f_K[f_{K-1},\phi_K]}{\partial f_{K-1}} \right|.
\tag{16.7}
$$

对逆映射而言，其雅可比矩阵的绝对行列式可通过将同一规则应用于公式（16.5）求得，其值恰为前向映射中绝对行列式的倒数（参见习题 16.3）。

我们使用包含 $I$ 个训练样本 $\{x_i\}$ 的数据集，以负对数似然为优化目标来训练归一化流模型：

$$
\hat{\phi} = \arg\max_{\phi} \prod_{i=1}^{I} \Pr(z_i) \cdot \left| \frac{\partial f[z_i,\phi]}{\partial z_i} \right|^{-1},
$$

等价地，

$$
\hat{\phi} = \arg\min_{\phi} \sum_{i=1}^{I} \left[ \log \left| \frac{\partial f[z_i,\phi]}{\partial z_i} \right| - \log \Pr(z_i) \right], \tag{16.8}
$$

其中 $z_i = f^{-1}[x_i,\phi]$，$\Pr(z_i)$ 在基础分布（base distribution）下计算，而绝对行列式 $\left| \frac{\partial f[z_i,\phi]}{\partial z_i} \right|$ 则由公式（16.7）给出。

16.2.2 网络层的设计要求  

归一化流的理论框架十分简洁。然而，要使其具备实际应用价值，我们需要设计满足以下四条性质的可逆神经网络层 $f_k$：

1. **表达能力充分性**：所有网络层组合起来，必须具备足够强的表达能力，能够将多元标准正态分布映射为任意目标概率密度函数。

2. **可逆性**：各网络层本身必须是可逆的；即每个层都需定义一个从任意输入点到输出点的唯一、一一对应的映射（双射，bijection）。若多个不同输入被映射至同一输出，则其逆映射将不唯一，从而产生歧义（参见附录 B.1：双射）。

3. **逆运算高效性**：必须能高效地计算每一层的逆变换。因为在每次似然函数评估过程中均需执行该操作；而训练过程又需反复进行大量似然评估，因此逆变换必须存在闭式解（closed-form solution），或至少具备快速算法支持。

4. **雅可比行列式高效可计算性**：必须能高效地计算前向映射或逆映射的雅可比矩阵的行列式（绝对值）。

16.3 可逆网络层  

接下来，我们将介绍若干适用于此类模型的可逆网络层（亦称“流”）。我们首先从线性流与逐元素流（elementwise flows）入手：这两类流易于求逆，且其雅可比行列式亦可解析计算；但单独使用时，二者均不足以刻画基础分布的任意复杂变换。不过，它们构成了耦合流（coupling flows）、自回归流（autoregressive flows）及残差流（residual flows）的基本构建模块——而这三类流均具备更强的表达能力。

草稿：如有勘误，请发送邮件至 udlbookmail@gmail.com。

310 16 归一化流  
16.3.1 线性流  

线性流的形式为 $ f[\mathbf{h}] = \boldsymbol{\beta} + \boldsymbol{\Omega}\mathbf{h} $。若矩阵 $ \boldsymbol{\Omega} $ 可逆，则该线性变换亦可逆。对于 $ \boldsymbol{\Omega} \in \mathbb{R}^{D \times D} $，求逆运算的时间复杂度为 $ O[D^3] $。附录 A  
雅可比行列式即为 $ \boldsymbol{\Omega} $ 的行列式，其计算复杂度同样为 $ O[D^3] $。大 $ O $ 记号  
这意味着，随着维度 $ D $ 的增加，线性流的计算开销将显著上升。  

若矩阵 $ \boldsymbol{\Omega} $ 具有特殊结构，则求逆与行列式计算可更高效，但变换的表达能力也随之减弱。附录 B.4  
矩阵类型  
例如，对角矩阵的求逆与行列式计算仅需 $ O[D] $ 时间，但其各输出分量彼此独立——即隐变量 $ \mathbf{h} $ 的各元素之间不发生交互。正交矩阵的求逆也更为高效，且其行列式恒为 $ \pm 1 $，但无法对各个维度进行独立缩放。三角矩阵则更具实用性：其可通过“回代法”（back-substitution）完成求逆，时间复杂度为 $ O[D^2] $；而其行列式则等于主对角线元素的乘积。  

一种兼顾通用性、高效可逆性及雅可比行列式易计算性的线性流构造方法，是直接以 $ \boldsymbol{\Omega} $ 的 LU 分解形式进行参数化。换言之，我们采用如下形式：  
$$
\boldsymbol{\Omega} = \mathbf{P}\mathbf{L}(\mathbf{U} + \mathbf{D}), \tag{16.9}
$$  
其中 $ \mathbf{P} $ 是预设的置换矩阵，$ \mathbf{L} $ 是下三角矩阵，$ \mathbf{U} $ 是主对角线全为零的上三角矩阵，而 $ \mathbf{D} $ 是对角矩阵，用于补足 $ \mathbf{U} $ 所缺失的对角元素。该形式可在 $ O[D^2] $ 时间内完成求逆；其对数行列式则等于 $ \mathbf{L} $ 与 $ \mathbf{D} $ 主对角线上各元素绝对值的对数之和。  

遗憾的是，线性流的表达能力并不充分。当线性函数 $ f[\mathbf{h}] = \boldsymbol{\beta} + \boldsymbol{\Omega}\mathbf{h} $ 作用于服从正态分布 $ \mathrm{Norm}[\boldsymbol{\mu}, \boldsymbol{\Sigma}] $ 的输入 $ \mathbf{h} $ 时，输出仍服从正态分布，其均值与协方差分别为 $ \boldsymbol{\beta} + \boldsymbol{\Omega}\boldsymbol{\mu} $ 和 $ \boldsymbol{\Omega}\boldsymbol{\Sigma}\boldsymbol{\Omega}^\top $。习题 16.5–16.6  
因此，仅靠线性流无法将正态分布映射为任意目标概率密度函数。  

16.3.2 逐元素流  

由于线性流表达能力不足，我们必须转向非线性流。其中最简单的一类是**逐元素流**（elementwise flows），它对输入向量的每个分量独立施加一个逐点非线性函数 $ f[\cdot,\boldsymbol{\phi}] $（含参数 $ \boldsymbol{\phi} $），即：  
$$
f[\mathbf{h}] = \big[ f[h_1,\boldsymbol{\phi}],\, f[h_2,\boldsymbol{\phi}],\, \dots,\, f[h_D,\boldsymbol{\phi}] \big]^\top. \tag{16.10}
$$  

其雅可比矩阵 $ \partial f[\mathbf{h}]/\partial \mathbf{h} $ 为对角矩阵，因为第 $ d $ 个输入分量 $ h_d $ 仅影响第 $ d $ 个输出分量。因此，其行列式即为对角线上各元素的乘积：  
$$
\left| \frac{\partial f[\mathbf{h}]}{\partial \mathbf{h}} \right| = \prod_{d=1}^D \left| \frac{\partial f[h_d]}{\partial h_d} \right|. \tag{16.11}
$$  

函数 $ f[\cdot,\boldsymbol{\phi}] $ 可以是固定的可逆非线性函数（如带泄露的 ReLU，见图 3.13），此时不含可学习参数；习题 16.7  
也可采用任意参数化的可逆函数。  

本作品遵循知识共享署名-非商业-禁止演绎（CC-BY-NC-ND）许可协议。（C）麻省理工学院出版社。

16.3 可逆网络层　311  
图16.5　分段线性映射。通过将输入域 $ h \in [0,1] $ 划分为 $ K $ 个等宽区域（此处 $ K = 5 $），可构造一个可逆的分段线性映射 $ h' = f[h,\phi] $。每个区域具有一个斜率参数 $ \phi_k $。  
a）若这些参数均为正数且总和为 1，则  
b）该函数即为可逆函数，并将输入域映射至输出域 $ h' \in [0,1] $。  

可逆的一一映射。一个简单示例是具有 $ K $ 个区间的分段线性函数（见图16.5），其将区间 $[0,1]$ 映射到自身：  
$$
f[h,\phi] = \sum_{k=1}^{b-1} \phi_k + (hK - b + 1)\phi_b, \tag{16.12}
$$  
其中参数 $ \phi_1, \phi_2, \dots, \phi_K $ 均为正数且满足 $ \sum_{k=1}^K \phi_k = 1 $，而 $ b = \lfloor Kh \rfloor + 1 $ 表示包含 $ h $ 所在的区间索引。第一项为所有前序区间的参数之和，第二项则表示 $ h $ 在当前区间内所处位置的比例。该函数易于求逆，且其梯度几乎处处可计算。  
习题 16.8–16.9  

存在大量构造光滑函数的类似方案，通常采用样条函数（splines），并通过参数约束确保函数单调，从而保证其可逆性。  

逐元素流（elementwise flows）是非线性的，但不混合输入维度，因此无法在变量间建立相关性。当与线性流（会混合维度）交替使用时，则可建模更复杂的变换。然而在实践中，逐元素流通常作为更复杂层（如耦合流）的组成部分加以使用。  

16.3.3　耦合流  

耦合流将输入 $ h $ 划分为两部分，记为 $ h = [h_1^\top, h_2^\top]^\top $，并定义流变换 $ f[h,\phi] $ 为：  

草稿：请将勘误发送至 udlbookmail@gmail.com。

312 16 归一化流  
图 16.6 耦合流。a) 输入（橙色向量）被划分为 $h_1$ 和 $h_2$。输出（青色向量）的第一部分 $h'_1$ 直接复制自 $h_1$；第二部分 $h'_2$ 则通过对 $h_2$ 应用一个可逆变换 $g[\cdot,\phi]$ 得到，其中参数 $\phi$ 本身是 $h_1$ 的（未必可逆）函数。b) 在逆映射中，有 $h_1 = h'_1$。这使得我们能够先计算出参数 $\phi[h_1]$，再对 $h'_2$ 应用逆变换 $g^{-1}[h'_2,\phi]$，从而恢复出 $h_2$。

$$
h'_1 = h_1 \tag{16.13}
$$
$$
h'_2 = g\bigl[h_2,\phi[h_1]\bigr].
$$

此处，$g[\cdot,\phi]$ 是一个逐元素流（或其它可逆层），其参数 $\phi[h_1]$ 本身是输入 $h_1$ 的非线性函数（见图 16.6）。函数 $\phi[\cdot]$ 通常为某种形式的神经网络，且无需满足可逆性要求。原始变量可通过如下方式恢复：

$$
h_1 = h'_1 \tag{16.14}
$$
$$
h_2 = g^{-1}\bigl[h'_2,\phi[h_1]\bigr].
$$

若 $g[\cdot,\phi]$ 为逐元素流，则其雅可比矩阵为下三角矩阵：左上角为单位矩阵，右下角为该逐元素变换的导数矩阵。其行列式即为所有对角线元素的乘积。

逆变换与雅可比行列式均可高效计算；但该方法仅以依赖于前半部分参数的方式变换后半部分参数。为实现更通用的变换，可在各层之间借助置换矩阵（参见附录 B.4.4）对 $h$ 的各元素进行随机重排，从而使每个变量最终均受到其余所有变量的影响。

**置换矩阵**  
在实际应用中，这类置换矩阵往往难以通过学习获得。因此，通常采用随机初始化并随后冻结其参数的方式处理。对于图像等结构化数据，通道维度被划分为两半 $h_1$ 和 $h_2$，并在各层之间使用 $1\times1$ 卷积实现通道置换。

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

16.3 可逆网络层　313  
图16.7　自回归流（Autoregressive flows）。输入 $ \mathbf{h} $（橙色列）与输出 $ \mathbf{h}' $（青色列）被划分为各自的维度分量（此处为四维）。  
a) 输出 $ h'_1 $ 是输入 $ \mathbf{h} $ 的一个可逆变换；$ h'_2 $ 是输入 $ h_1 $ 的可逆函数，其参数依赖于 $ h_1 $；$ h'_3 $ 是输入 $ h_2 $ 的可逆函数，其参数依赖于先前的输入 $ h_1 $ 和 $ h_2 $；依此类推。所有输出彼此独立，因此可并行计算。  
b) 自回归流的逆变换采用与耦合流（coupling flows）类似的方法进行计算。但需注意：为计算 $ h_2 $，必须已知 $ h_1 $；为计算 $ h_3 $，必须已知 $ h_1 $ 和 $ h_2 $；依此类推。因此，其逆变换无法并行计算。  

16.3.4　自回归流  
自回归流是耦合流的一种推广形式，它将每个输入维度视为一个独立的“块”（见图16.7）。该方法基于输入 $ \mathbf{h} $ 的前 $ d-1 $ 个维度来计算输出 $ \mathbf{h}' $ 的第 $ d $ 个维度：  
$$
h'_d = g\big(h_d,\, \phi[\mathbf{h}_{1:d-1}]\big). \tag{16.15}
$$  
其中，函数 $ g[\cdot,\cdot] $ 称为**变换器（transformer）**¹，而参数 $ \phi $、$ \phi[h_1] $、$ \phi[h_1,h_2] $、… 则称为**条件器（conditioners）**。与耦合流类似，变换器 $ g[\cdot,\phi] $ 必须是可逆的；而条件器 $ \phi[\cdot] $ 可以采用任意形式，通常由神经网络实现。若变换器与条件器均具备足够的表达能力，则自回归流即为**通用近似器（universal approximator）**——亦即，它能够表示任意概率分布。  

借助带有适当掩码（mask）的网络，可并行计算输出 $ \mathbf{h}' $ 的全部分量：掩码确保位于位置 $ d $ 处的参数 $ \phi $ 仅依赖于此前的输入分量。  

¹ 此处的“transformer”与第12章所讨论的“Transformer 层”无关。  
草稿：勘误请发送至 udlbookmail@gmail.com。

314 16 归一化流  
位置。这被称为**掩码自回归流**（masked autoregressive flow）。其原理与掩码自注意力机制（第12.7.2节）非常相似：连接输入与先前输出的边被剪除（pruned）。

该变换的逆向计算效率较低。考虑前向映射：  
$$
\begin{aligned}
h'_1 &= g_1\big(h_1,\phi\big) \\
h'_2 &= g_2\big(h_2,\phi[h_1]\big) \\
h'_3 &= g_3\big(h_3,\phi[h_{1:2}]\big) \\
h'_4 &= g_4\big(h_4,\phi[h_{1:3}]\big). \quad (16.16)
\end{aligned}
$$  
其逆变换需按顺序执行，所依据的原则与耦合流（coupling flows）类似：  
$$
\begin{aligned}
h_1 &= g_1^{-1}\big(h'_1,\phi\big) \\
h_2 &= g_2^{-1}\big(h'_2,\phi[h_1]\big) \\
h_3 &= g_3^{-1}\big(h'_3,\phi[h_{1:2}]\big) \\
h_4 &= g_4^{-1}\big(h'_4,\phi[h_{1:3}]\big). \quad (16.17)
\end{aligned}
$$  
由于 $h_d$ 的计算依赖于 $h_{1:d-1}$（即截至目前已获得的部分结果），因此无法并行执行。故当输入维度较大时，逆变换耗时显著。

**自回归流**  

16.3.5 逆自回归流（Inverse Autoregressive Flows）  
掩码自回归流是在归一化（即逆向）方向上定义的。这一设定对于高效计算似然函数（从而实现模型学习）是必需的。然而，采样过程需在前向（生成）方向上进行，此时每一层中各变量必须依次计算，因而速度较慢。若将自回归流用于前向（生成）变换，则采样高效，但似然计算（及模型训练）变得缓慢。此类结构被称为**逆自回归流**（inverse autoregressive flow）。

一种兼顾快速学习与快速（但近似）采样的技巧是：首先构建一个掩码自回归流以学习目标分布（作为“教师”模型），再利用该教师模型训练一个逆自回归流（作为“学生”模型），从而实现高效采样。该方法要求采用一种不同的归一化流建模形式——即从另一个函数（而非原始样本集）中学习（参见第16.5.3节）。

16.3.6 残差流：iRevNet  
残差流的设计灵感源自残差网络（ResNets）。它将输入划分为两部分 $ \mathbf{h} = [\mathbf{h}_1^\top, \mathbf{h}_2^\top]^\top $（划分方式与耦合流相同），并定义输出为：  

本作品遵循知识共享署名—非商业性使用—禁止演绎（CC-BY-NC-ND）许可协议。（C）麻省理工学院出版社。

16.3 可逆网络层　315  
图16.8　残差流（Residual Flows）。  
a) 通过将输入拆分为 $h_1$ 和 $h_2$，并构建两个残差层，计算一个可逆函数。在第一层中，对 $h_2$ 进行处理，并将结果加至 $h_1$；在第二层中，对上一步的结果进行处理，并将结果加至 $h_2$。  
b) 在反向机制中，函数按相反顺序执行，加法运算相应地变为减法运算。

$$
h'_1 = h_1 + f_1[h_2, \phi_1] \\
h'_2 = h_2 + f_2[h'_1, \phi_2], \tag{16.18}
$$

其中 $f_1[\cdot,\phi_1]$ 和 $f_2[\cdot,\phi_2]$ 是两个函数，二者本身未必可逆（见图16.8）。其逆变换可通过逆序计算得到：

$$
h_2 = h'_2 - f_2[h'_1, \phi_2] \\
h_1 = h'_1 - f_1[h_2, \phi_1]. \tag{16.19}
$$

与耦合流（coupling flows）类似，这种按块划分的方式限制了所能表示的变换族。因此，各层之间会对输入变量进行置换（permutation），以确保变量能够以任意方式混合。

该形式易于求逆，但对于一般函数 $f_1[\cdot,\phi_1]$ 和 $f_2[\cdot,\phi_2]$，尚无高效方法计算其雅可比矩阵（Jacobian）。该形式有时被用于**习题16.10**中，以节省残差网络训练时的内存开销：由于网络整体可逆，在前向传播过程中无需显式存储每一层的激活值。

16.3.7　残差流与压缩映射：iResNet  
另一种利用残差网络的方法是借助巴拿赫不动点定理（Banach fixed-point theorem），亦称压缩映射定理（contraction mapping theorem）——该定理指出：任一压缩映射均存在唯一不动点。压缩映射 $f[\cdot]$ 满足如下性质：  

（此处原文未完，但根据上下文及技术惯例，应接续定义压缩性条件，例如 $\|f(x) - f(y)\| \leq \gamma \|x - y\|$，其中 $0 < \gamma < 1$；但因原文截断，译文严格遵循所给内容，不作补充。）

草稿：请将勘误发送至 udlbookmail@gmail.com。

316 16 归一化流  
图 16.9 压缩映射（Contraction Mappings）。若一个函数在所有位置的绝对斜率均小于 1，则对该函数进行迭代时，结果将收敛至一个不动点 $ f[z] = z $。  
a) 从初始点 $ z_0 $ 出发，我们计算 $ z_1 = f[z_0] $；随后将 $ z_1 $ 再次代入函数并继续迭代。最终，该过程将收敛至满足 $ f[z] = z $ 的点（即函数曲线与虚线所示的对角恒等线 $ y = z $ 相交之处）。  
b) 此性质可用于求解形如 $ y = z + f[z] $ 的方程，以对给定值 $ y^* $ 反解出对应的 $ z^* $：注意到函数 $ y^* - f[z] $ 的不动点（即橙色曲线与虚线恒等线的交点）恰好位于满足 $ y^* = z + f[z] $ 的位置。

$$
\text{dist}\big[f[z'], f[z]\big] < \beta \cdot \text{dist}\big[z', z\big] \quad \forall z, z', \tag{16.20}
$$

其中 $ \text{dist}[\cdot,\cdot] $ 是某一距离函数，且 $ 0 < \beta < 1 $。具备该性质的函数称为压缩映射；当对其反复迭代（即持续将输出作为下一次的输入）时，结果必收敛至一个不动点，满足 $ f[z] = z $（见图 16.9）。为理解其原理，可考虑将该函数同时作用于当前点 $ z $ 与不动点 $ z^* $：不动点保持不变，而两点之间的距离则严格缩小，因此当前点必然不断趋近于不动点。

该定理可用于求解如下形式的方程：

$$
y = z + f[z] \tag{16.21}
$$

前提是 $ f[z] $ 是一个压缩映射。换言之，它可用于对给定输出值 $ y^* $，反推出对应的输入 $ z^* $。具体做法是：任选初始点 $ z_0 $，然后迭代执行 $ z_{k+1} = y^* - f[z_k] $。该迭代过程的不动点即满足 $ z + f[z] = y^* $（见图 16.9b）。

附录 B.1.1 同样的原理亦可用于对残差网络层 $ h' = h + f[h, \phi] $ 进行求逆，只要确保 $ f[h, \phi] $ 是一个压缩映射即可。实践中，这等价于要求其 Lipschitz 常数小于 1。若假设激活函数的斜率不大于 1，则该条件进一步等价于要求 $ f[h, \phi] $ 关于 $ h $ 的雅可比矩阵的最大奇异值小于 1。

附录 B.3.7 奇异值  

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（C）麻省理工学院出版社。

16.4 多尺度流　317  
每个权重矩阵 $\Omega$ 的谱范数必须小于 1。一种粗略的实现方式是通过对权重 $\Omega$ 进行裁剪（clipping），以确保其绝对值足够小。  

雅可比行列式本身难以直接计算，但其对数可通过一系列技巧进行近似。  
$$
\log \left| I + \frac{\partial f[h,\phi]}{\partial h} \right| = \operatorname{trace} \left[ \log \left( I + \frac{\partial f[h,\phi]}{\partial h} \right) \right] 
= \sum_{k=1}^\infty \frac{(-1)^{k-1}}{k} \operatorname{trace} \left[ \left( \frac{\partial f[h,\phi]}{\partial h} \right)^k \right], \tag{16.22}
$$  
其中第一行利用了恒等式 $\log |A| = \operatorname{trace}[\log A]$，第二行则将其展开为幂级数。  

即使对该级数进行截断，计算各项中矩阵幂的迹仍具有较高的计算开销（参见附录 B.3.8「迹的估计」）。因此，我们采用 Hutchinson 迹估计器（Hutchinson’s trace estimator）对其进行近似。考虑一个均值为 $0$、协方差矩阵为单位阵 $I$ 的标准正态随机向量 $\epsilon$。矩阵 $A$ 的迹可估计为：  
$$
\begin{aligned}
\operatorname{trace}[A] &= \operatorname{trace}\left[ A\, \mathbb{E}\!\left[ \epsilon \epsilon^\top \right] \right] \\
&= \operatorname{trace}\left[ \mathbb{E}\!\left[ A \epsilon \epsilon^\top \right] \right] \\
&= \mathbb{E}\!\left[ \operatorname{trace}\left[ A \epsilon \epsilon^\top \right] \right] \\
&= \mathbb{E}\!\left[ \operatorname{trace}\left[ \epsilon^\top A \epsilon \right] \right] \\
&= \mathbb{E}\!\left[ \epsilon^\top A \epsilon \right], \tag{16.23}
\end{aligned}
$$  
其中：第一行成立是因为 $\mathbb{E}[\epsilon \epsilon^\top] = I$；第二行由期望算子的线性性质导出；第三行源于迹算子的线性性；第四行利用了迹在循环置换下的不变性；最后一行成立是因为第四行括号内的表达式现在是一个标量。我们通过从分布 $\Pr(\epsilon)$ 中采样独立同分布样本 $\epsilon_i$ 来估计该迹：  
$$
\operatorname{trace}[A] = \mathbb{E}\!\left[ \epsilon^\top A \epsilon \right] 
\approx \frac{1}{I} \sum_{i=1}^I \epsilon_i^\top A \epsilon_i. \tag{16.24}
$$  
借助该方法，我们即可近似计算泰勒展开式（式 (16.22)）中各幂次项的迹，并最终求得对数概率。  

16.4 多尺度流  
在标准化流（normalizing flows）中，隐空间 $z$ 的维度必须与数据空间 $x$ 完全一致；然而，我们已知自然数据集往往可由更少的潜在变量加以刻画。  
草稿：请将勘误发送至 udlbookmail@gmail.com。

318 16 归一化流  
图 16.10 多尺度流。在归一化流中，隐变量空间 $ \mathbf{z} $ 的维度必须与模型密度的维度一致。然而，该空间可被划分为若干分量，并在不同网络层中逐步引入。这一设计既加速了密度估计，也提升了采样效率。在逆向（即生成）过程中，图中黑色箭头方向反转；且每个模块的最后一部分将跳过后续处理步骤。例如，函数 $ f^{-1}_{[3,\phi]} $ 仅作用于前三个分块，而第四个分块则直接作为 $ \mathbf{z}_4 $ 输出，并依据基础分布进行评估。

在某一层，我们终究需要引入所有这些变量；但若将它们全部贯穿整个网络，则效率低下。由此催生了多尺度流（见图 16.10）的思想。  
在生成方向上，多尺度流将隐向量划分为 $ \mathbf{z} = [\mathbf{z}_1, \mathbf{z}_2, \dots, \mathbf{z}_N] $。第一部分 $ \mathbf{z}_1 $ 首先经由一系列可逆层处理，这些层均保持与 $ \mathbf{z}_1 $ 相同的维度；直至某一时刻，$ \mathbf{z}_2 $ 被拼接并融合至第一部分；此过程持续进行，直至网络输出维度与数据 $ \mathbf{x} $ 一致。在归一化（即推断）方向上，网络从 $ \mathbf{x} $ 的完整维度开始运行；当处理至 $ \mathbf{z}_n $ 被引入的位置时，该部分即与基础分布进行比对。

16.5 应用场景  
接下来，我们将介绍归一化流的三类典型应用：首先讨论概率密度建模；其次介绍用于图像合成的 GLOW 模型；最后探讨利用归一化流近似其他分布的方法。

16.5.1 密度建模  
在本书所讨论的四类生成模型中，归一化流是唯一能够精确计算新样本对数似然值的模型。生成对抗网络（Generative Adversarial Networks, GANs）  

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社。

16.5 应用 319  
图 16.11 密度建模。a）人工构造的二维数据样本；b）使用 iResNet 建模得到的密度；c–d）第二个示例。改编自 Behrmann 等（2019）  

生成对抗网络（GAN）等模型本身不具备概率解释性，而变分自编码器（VAE）与扩散模型（diffusion model）均只能提供似然函数的下界估计。² 图 16.11 展示了利用 i-ResNet 在两个简单玩具问题中估计出的概率分布。密度估计的一个典型应用是异常检测：首先使用归一化流（normalizing flow）模型对干净数据集的数据分布进行建模；随后，将新输入样本中概率值显著偏低者标记为离群点（outlier）。但需谨慎对待该方法，因为可能存在某些高概率离群点，并未落入典型集（typical set）内（参见图 8.13）。  

16.5.2 合成（图像生成）  

生成流（Generative Flow），亦称 GLOW，是一种可生成高保真度图像（见图 16.12）的归一化流模型，其设计融合了本章介绍的诸多关键技术思想。GLOW 最便于从归一化方向（即从数据 $x$ 映射至潜变量 $z$ 的方向）加以理解。该模型以一个尺寸为 $256 \times 256 \times 3$ 的张量作为输入，其中每个元素代表一幅 RGB 图像的一个像素。模型采用耦合层（coupling layer）结构：在每一层中，通道维度被划分为两半；其中后一半通道在每个空间位置上接受一个不同的仿射变换（affine transformation），而该仿射变换的参数则由一个作用于另一半通道上的二维卷积神经网络计算得出。耦合层交替地与 $1 \times 1$ 卷积层堆叠；这些 $1 \times 1$ 卷积被参数化为 LU 分解形式，从而实现通道间的混合。此外，模型周期性地执行下采样操作：将每个 $2 \times 2$ 的局部图像块合并为单个空间位置，同时将通道数增至原来的四倍。GLOW 是一种多尺度流（multi-scale flow）架构，部分通道会在若干层之后被周期性地剥离，直接构成潜变量向量 $z$ 的一部分。由于图像数据本质上是离散的（源于 RGB 值的量化），为防止训练过程中似然函数无界增长，需在输入端加入噪声——此技术称为去量化（dequantization）。  

为生成更逼真的图像，GLOW 模型并非直接从基础密度 $p(z)$ 中采样，而是从其幂次形式 $p(z)^\alpha$（其中 $\alpha > 0$）中采样。该策略倾向于选择更靠近密度中心而非尾部区域的样本，其思想与 GAN 中的截断技巧（truncation trick）类似。  

² 扩散模型所给出的似然下界在实际计算中甚至可能超过归一化流所能获得的精确似然值，但其数据生成速度却显著更慢（参见第 18 章）。  
草稿：勘误请发送至 udlbookmail@gmail.com。

320 16 归一化流  
图16.12 基于CelebA HQ数据集（Karras 等，2018）训练的GLOW模型所生成的样本。这些样本质量尚可，但生成对抗网络（GAN）和扩散模型可产生更优的结果。改编自Kingma与Dhariwal（2018）。  
图16.13 利用GLOW模型进行插值。左右两侧图像为真实人物图像；中间各图则是先将真实图像投影至隐空间，再在隐空间中线性插值，最后将插值所得点反投影回图像空间而得到。改编自Kingma与Dhariwal（2018）。  
本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

16.6 小结　321  
（见图15.10）。值得注意的是，其生成样本质量不及生成对抗网络（GAN）或扩散模型（diffusion models）。目前尚不清楚这一差距源于可逆层本身存在的根本性限制，还是仅仅因为学界在该方向投入的研究精力相对较少。  

图16.13展示了使用GLOW模型进行插值的一个示例：首先沿归一化方向（normalizing direction）将两张真实图像分别变换为两个潜在向量；随后对这两个潜在向量进行线性插值，得到中间点；最后再沿生成方向（generative direction）利用网络将这些中间点映射回图像空间。最终结果是一组在两张原始真实图像之间实现自然、逼真过渡的插值图像。  

16.5.3 近似其他密度模型  
归一化流还可用于学习生成服从某一既定密度分布的样本——该密度易于评估（evaluation），但难以直接采样（sampling）。在此类任务中，我们将归一化流 $ P_r(x \mid \phi) $ 视为“学生模型”（student），而将目标密度 $ q(x) $ 视为“教师模型”（teacher）。  

为推进训练，我们从学生模型中生成样本 $ x_i = f[z_i, \phi] $。由于这些样本由我们自身生成，因此已知其对应的潜在变量 $ z_i $，且无需执行逆变换即可计算其在学生模型下的似然值。由此，我们甚至可以采用诸如掩码自回归流（masked autoregressive flow）等逆变换较慢的模型结构。我们定义一个基于反向KL散度（reverse KL divergence）的损失函数，以促使学生模型与教师模型的似然分布尽可能一致，并据此训练学生模型（见图16.14）：  
**问题16.11**  
$$
\hat{\phi} = \arg\min_{\phi} \, \mathrm{KL}\!\left[ \frac{1}{I}\sum_{i=1}^{I} \delta\!\left( x - f[z_i,\phi] \right) \,\middle|\!\middle|\, q(x) \right]. \tag{16.25}
$$  

该方法与归一化流的典型用法形成鲜明对比：后者通常用于构建数据 $ x_i $ 的概率模型 $ P_r(x_i, \phi) $，其中数据来自未知分布，仅能通过有限样本观测；此时训练依赖于最大似然估计，即最小化前向KL散度中的交叉熵项（参见第5.7节）：  
$$
\hat{\phi} = \arg\min_{\phi} \, \mathrm{KL}\!\left[ \frac{1}{I}\sum_{i=1}^{I} \delta[x - x_i] \,\middle|\!\middle|\, P_r(x_i, \phi) \right]. \tag{16.26}
$$  

借助上述技巧，归一化流亦可用于变分自编码器（VAE）中建模后验分布（详见第17章）。  

16.6 小结  
归一化流通过对基分布（通常为标准正态分布）施加一系列变换，构造出新的密度函数。其核心优势在于：既能精确计算任意样本的似然值，又能高效生成新样本。然而，其架构受到严格约束——每一层变换都必须是可逆的：前向变换用于生成样本，反向变换则用于计算似然。  

此外，雅可比行列式（Jacobian）必须能够被高效计算，这是似然评估的关键环节；而在密度学习过程中，该计算需反复执行。然而，可逆层的设计本身即构成一项挑战。  

草稿：如有勘误，请发送邮件至 udlbookmail@gmail.com。

322 16 归一化流  
图 16.14 密度模型近似。a）训练数据；b）通常，我们调整流模型的参数，以最小化训练数据分布到流模型分布的 KL 散度。这等价于最大似然拟合（第 5.7 节）；c）另一种方法是调整流参数 $\phi$，以最小化流采样结果 $x_i = f[z_i, \phi]$ 到 d）目标密度分布的 KL 散度。  
即使在雅可比行列式无法被高效估计的情况下，这类方法本身仍具有实用价值；它们将训练一个 $K$ 层网络所需的内存开销从 $O[K]$ 降低至 $O[1]$。  
本章回顾了可逆网络层（亦称“流”）。我们首先讨论了线性流与逐元素流——它们结构简单，但表达能力不足；随后介绍了更复杂的流结构，例如耦合流、自回归流和残差流。最后，我们展示了归一化流如何用于似然估计、图像生成以及图像插值，并可用于近似其他任意分布。  

注释  
归一化流最早由 Rezende & Mohamed（2015）提出，但其思想渊源可追溯至 Tabak & Vanden-Eijnden（2010）、Tabak & Turner（2013）以及 Rippel & Adams（2013）的研究工作。关于归一化流的综述性文献可见 Kobyzev 等（2020）及 Papamakarios 等（2021）。Kobyzev 等（2020）对各类归一化流模型进行了定量对比分析。  
本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）MIT 出版社。

注释 323  
许多归一化流（normalizing flow）方法。他们得出结论：Flow++ 模型（一种耦合流，采用新颖的逐元素变换及其他创新设计）在当时性能最优。  

可逆网络层（Invertible network layers）：可逆层可降低反向传播算法的内存需求；前向传播过程中无需再存储激活值，因为这些值可在反向传播中重新计算。除本章已讨论的常规网络层与残差层（Gomez 等，2017；Jacobsen 等，2018）外，可逆层还已被拓展应用于图神经网络（Li 等，2021a）、循环神经网络（MacKay 等，2018）、掩码卷积（masked convolutions）（Song 等，2019）、U-Net（Brügger 等，2019；Etmann 等，2020）以及 Transformer（Mangalam 等，2022）。  

径向流与平面流（Radial and planar flows）：原始归一化流论文（Rezende & Mohamed，2015）采用了平面流（沿特定维度对分布进行收缩或扩张）与径向流（围绕某一点对分布进行扩张或收缩）。这些流的逆变换难以显式解析求解，但它们适用于近似采样速度缓慢、或仅能以未知缩放因子形式评估似然函数的分布（见图 16.14）。  

应用领域：包括图像生成（Ho 等，2019；Kingma & Dhariwal，2018）、噪声建模（Abdelhamed 等，2019）、视频生成（Kumar 等，2019b）、音频生成（Esling 等，2019；Kim 等，2018；Prenger 等，2019）、图结构生成（Madhawa 等，2019）、图像分类（Kim 等，2021；Mackowiak 等，2021）、图像隐写（Lu 等，2021）、超分辨率重建（Yu 等，2020；Wolf 等，2021；Liang 等，2021）、风格迁移（An 等，2021）、运动风格迁移（Wen 等，2021）、3D 形状建模（Paschalidou 等，2021）、图像压缩（Zhang 等，2021b）、sRGB 到 RAW 图像转换（Xing 等，2021）、图像去噪（Liu 等，2021b）、异常检测（Yu 等，2021）、图像到图像翻译（Ardizzone 等，2020）、在不同分子干预条件下合成细胞显微图像（Yang 等，2021），以及光传输模拟（Müller 等，2019b）。对于所有基于图像数据的应用，由于输入为量化后的离散值，因此在训练前必须添加噪声（参见 Theis 等，2016）。  

Rezende & Mohamed（2015）利用归一化流对变分自编码器（VAE）中的后验分布进行建模。Abdal 等（2021）则使用归一化流建模 StyleGAN 潜在空间中各属性的分布，并据此修改真实图像中指定的属性。Wolf 等（2021）利用归一化流学习给定干净图像条件下含噪输入图像的条件分布，从而合成可用于训练去噪或超分辨率模型的噪声数据。  

归一化流还在物理学（Kanwar 等，2020；Köhler 等，2020；Noé 等，2019；Wirnsberger 等，2020；Wong 等，2020）、自然语言处理（Tran 等，2019；Ziegler & Rush，2019；Zhou 等，2019；He 等，2018；Jin 等，2019）以及强化学习（Schroecker 等，2019；Haarnoja 等，2018a；Mazoure 等，2020；Ward 等，2019；Touati 等，2020）等领域展现出多样化的应用价值。  

线性流（Linear flows）：对角线性流可表示如批归一化（BatchNorm）（Dinh 等，2016）与激活归一化（ActNorm）（Kingma & Dhariwal，2018）等归一化变换。Tomczak & Welling（2016）研究了组合三角矩阵及采用 Householder 变换参数化的正交变换。Kingma & Dhariwal（2018）提出了第 16.5.2 节所述的 LU 分解参数化方法。Hoogeboom 等（2019b）则提出改用 QR 分解，该方法无需预设置换矩阵。卷积是深度学习中广泛使用的线性变换（见图 10.4），但其逆变换与行列式计算并不直接。Kingma & Dhariwal（2018）采用 $1\times1$ 卷积，这实质上是在每个空间位置独立施加一个完整的线性变换。Zheng 等（2017）引入了 ConvFlow，但将其限制于一维卷积。Hoogeboom 等（2019b）则提供了更通用的二维卷积建模方案：或通过堆叠掩码自回归卷积实现，或在傅里叶域中进行操作。  

草稿：请将勘误发送至 udlbookmail@gmail.com。

324 16 归一化流  
逐元素流与耦合函数：逐元素流（elementwise flows）对每个变量独立地应用相同的变换函数（但各变量对应不同的参数）。此类流亦可用于构建耦合流（coupling flows）和自回归流（autoregressive flows）中的耦合函数，此时其参数依赖于先前的变量。为保证可逆性，这些函数必须是单调函数。

加性耦合函数（Dinh 等，2015）仅对变量施加一个偏移量；仿射耦合函数（affine coupling functions）则对变量进行缩放并添加偏移量，被 Dinh 等（2015）、Dinh 等（2016）、Kingma & Dhariwal（2018）、Kingma 等（2016）以及 Papamakarios 等（2017）所采用。Ziegler & Rush（2019）提出了非线性平方流（nonlinear squared flow），该流定义为一个具有五个参数的、可逆的多项式比值函数。连续混合累积分布函数（continuous mixture CDFs，Ho 等，2019）则基于 $K$ 个逻辑斯谛（logistic）分布的混合模型的累积分布函数（CDF）实施单调变换，并在其后复合一个逆逻辑 Sigmoid 函数，再进行缩放与偏移。

分段线性耦合函数（图 16.5）由 Müller 等（2019b）提出。此后，基于三次样条（cubic splines，Durkan 等，2019a）和有理二次样条（rational quadratic splines，Durkan 等，2019b）的系统相继被提出。Huang 等（2018a）引入了神经自回归流（neural autoregressive flows），其中变换函数由一个神经网络表示，该网络需输出单调函数；一个充分条件是：所有权重均为正，且激活函数本身单调。然而，强制约束网络权重为正会显著增加训练难度，因此催生了无约束单调神经网络（unconstrained monotone neural networks，Wehenkel & Louppe，2019）：这类网络首先建模严格正函数，再通过数值积分获得单调函数。Jaini 等（2019）则基于一个经典结论——任一单变量正多项式均可表示为若干多项式之平方和——构造出可解析积分的正函数。最后，Dinh 等（2019）研究了分段单调耦合函数（piecewise monotonic coupling functions）。

耦合流（Coupling flows）：Dinh 等（2015）首次提出耦合流，将输入维度均分为两半（图 16.6）。Dinh 等（2016）进一步提出 RealNVP 模型，其对图像输入按交替像素或通道块进行划分。Das 等（2019）则提出依据导数幅值选择待传播部分的特征。Dinh 等（2016）将多尺度流（multi-scale flows，即维度逐步引入的流）解释为一类特殊的耦合流，其中参数 $\phi$ 不依赖于数据另一半。Kruse 等（2021）引入了一种层次化耦合流（hierarchical formulation of coupling flows），其中每一划分均递归地二分为两个子划分。GLOW（图 16.12–16.13）由 Kingma & Dhariwal（2018）设计，采用耦合流架构；同样基于耦合流的模型还包括 NICE（Dinh 等，2015）、RealNVP（Dinh 等，2016）、FloWaveNet（Kim 等，2018）、WaveGlow（Prenger 等，2019）以及 Flow++（Ho 等，2019）。

自回归流（Autoregressive flows）：Kingma 等（2016）将自回归模型用于归一化流。Germain 等（2015）开发了一种通用的掩码机制（masking mechanism），用以屏蔽先前变量的影响；Papamakarios 等（2017）利用该机制，在掩码自回归流（masked autoregressive flows）中实现了前向传播所有输出的并行计算。Kingma 等（2016）还提出了逆自回归流（inverse autoregressive flow）。Parallel WaveNet（Van den Oord 等，2018）通过对原始 WaveNet（Van den Oord 等，2016a）——一种面向音频生成的不同类型生成模型——进行知识蒸馏（knowledge distillation），将其转化为逆自回归流，从而大幅加速采样过程（参见图 16.14c–d）。

残差流（Residual flows）：残差流基于残差网络（residual networks，He 等，2016a）。RevNets（Gomez 等，2017）与 iRevNets（Jacobsen 等，2018）将输入划分为两个部分（图 16.8），每部分分别经由一个残差网络处理。此类网络具备可逆性，但其雅可比行列式（Jacobian determinant）难以高效计算。残差连接可被视作常微分方程（ordinary differential equation, ODE）的一种离散化形式；这一视角启发了若干新型可逆网络架构（Chang 等，2018，2019a）。然而，这些网络的雅可比行列式仍无法高效计算。Behrmann 等（2019）指出：若网络的 Lipschitz 常数小于 1，则可通过不动点迭代法实现网络求逆。由此发展出 iResNet，其雅可比矩阵对数行列式（log determinant of the Jacobian）可借助 Hutchinson 迹估计器（Hutchinson’s trace estimator）进行估算。  
本作品受 Creative Commons CC-BY-NC-ND 许可协议保护。（C）MIT 出版社。

注释 325  
估计器（Hutchinson，1989）。Chen 等（2019）通过采用俄罗斯轮盘（Russian Roulette）估计器，消除了式（16.22）中幂级数截断所引入的偏差。  

无穷小流（Infinitesimal flows）：若残差网络可视为常微分方程（ODE）的一种离散化，则下一步自然逻辑便是直接用 ODE 表示变量的变化。Chen 等（2018e）提出了神经 ODE（Neural ODE），并利用标准的 ODE 前向与反向传播方法。此时，计算似然函数不再需要显式计算雅可比矩阵；取而代之的是另一个 ODE，其中对数概率的变化由前向传播导数的迹（trace）决定。Grathwohl 等（2019）采用 Hutchinson 估计器来估计该迹，并进一步简化了该过程。Finlay 等（2020）在损失函数中引入正则化项以降低训练难度；Dupont 等（2019）则扩展了模型表示能力，使神经 ODE 能够刻画更广泛的微分同胚类（diffeomorphisms）。Tzen & Raginsky（2019）以及 Peluchetti & Favaro（2020）则将 ODE 替换为随机微分方程（SDE）。  

普适性（Universality）：普适性是指归一化流（normalizing flow）能够以任意精度逼近任意概率分布的能力。某些流结构（例如平面流 planar flow、逐元素流 elementwise flow）并不具备该性质。而当耦合函数采用神经单调网络（Huang 等，2018a）、基于单调多项式的构造（Jaini 等，2020）或基于样条（splines）的构造（Kobyzev 等，2020）时，自回归流（autoregressive flows）可被证明具有普适性。对于维度 $ D $，一系列 $ D $ 层耦合流即可构成一个自回归流。其原理在于：每层将变量划分为两部分 $ \mathbf{h}_1 $ 和 $ \mathbf{h}_2 $，其中 $ \mathbf{h}_2 $ 仅依赖于此前的变量（见图 16.6）。因此，若在每一层将 $ \mathbf{h}_1 $ 的维度逐次增加 1，则最终可复现一个自回归流，从而获得普适性。目前尚不清楚耦合流是否能在少于 $ D $ 层的情况下实现普适性；但在实践中（例如 GLOW 模型），即使不强制引入这种自回归结构，耦合流仍表现出优异性能。  

其他工作：归一化流领域的活跃研究方向还包括：离散流（discrete flows）（Hoogeboom 等，2019a；Tran 等，2019）、定义在非欧流形（non-Euclidean manifolds）上的归一化流（Gemici 等，2016；Wang & Wang，2019），以及等变流（equivariant flows）（Köhler 等，2020；Rezende 等，2019），后者旨在构建对某类变换族保持不变的概率密度。  

习题  

**习题 16.1**  
考虑对定义在 $ z \in [0,1] $ 上的均匀基分布进行变换，变换函数为 $ x = f(z) = z^2 $。求变换后分布 $ \Pr(x) $ 的表达式。  

**习题 16.2\***  
考虑对标准正态分布  
$$
\Pr(z) = \frac{1}{\sqrt{2\pi}} \exp\!\left[-\frac{z^2}{2}\right], \tag{16.27}
$$  
应用如下变换函数：  
$$
x = f(z) = \frac{1}{1 + \exp[-z]}. \tag{16.28}
$$  
求变换后分布 $ \Pr(x) $ 的表达式。  

**习题 16.3\***  
写出逆映射 $ z = f^{-1}(x,\phi) $ 的雅可比矩阵及其绝对行列式的表达式，形式分别类比式（16.6）和式（16.7）。  

草稿：勘误请发送至 udlbookmail@gmail.com。

326 16 归一化流（Normalizing Flows）

**习题 16.4**  
请手动计算下列矩阵的逆矩阵及其行列式：

$$
\Omega_1 =
\begin{bmatrix}
2 & 0 & 0 & 0 \\
0 & -5 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 2
\end{bmatrix}, \quad
\Omega_2 =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
2 & 4 & 0 & 0 \\
1 & -1 & 2 & 0 \\
4 & -2 & -2 & 1
\end{bmatrix}.
\tag{16.29}
$$

**习题 16.5**  
考虑一个均值为 $\mu$、协方差为 $\Sigma$ 的随机变量 $z$，其经仿射变换 $x = Az + b$ 后得到新变量 $x$。试证明：$x$ 的期望值为 $A\mu + b$，且其协方差为 $A\Sigma A^\top$。

**习题 16.6\***  
设 $x = f[z] = Az + b$，且 $z \sim \mathrm{Norm}_z[\mu, \Sigma]$。利用如下变量变换关系式，证明 $x \sim \mathrm{Norm}_x[A\mu + b,\, A\Sigma A^\top]$：

$$
\mathrm{Pr}(x) = \mathrm{Pr}(z) \cdot \left| \frac{\partial f[z]}{\partial z} \right|^{-1}.
\tag{16.30}
$$

**习题 16.7**  
Leaky ReLU 函数定义为：

$$
\mathrm{LReLU}[z] =
\begin{cases}
0.1z, & z < 0, \\
z,    & z \geq 0.
\end{cases}
\tag{16.31}
$$

请写出 Leaky ReLU 的反函数表达式；并针对多元变量 $z = [z_1, z_2, \dots, z_D]^\top$ 的逐元素变换 $x = f[z]$，其中

$$
f[z] = \big[ \mathrm{LReLU}[z_1],\, \mathrm{LReLU}[z_2],\, \dots,\, \mathrm{LReLU}[z_D] \big]^\top,
\tag{16.32}
$$

写出其雅可比矩阵绝对值的倒数 $\left| \frac{\partial f[z]}{\partial z} \right|^{-1}$ 的表达式。

**习题 16.8**  
考虑将式 (16.12) 中定义的分段线性函数 $f[h, \phi]$ 在定义域 $h' \in [0,1]$ 上逐元素作用于输入向量 $h = [h_1, h_2, \dots, h_D]^\top$，即令

$$
f[h] = \big[ f[h_1, \phi],\, f[h_2, \phi],\, \dots,\, f[h_D, \phi] \big]^\top.
$$

试求雅可比矩阵 $\partial f[h] / \partial h$ 的表达式，并计算其行列式。

**习题 16.9\***  
考虑基于等间距分桶上平方根函数锥组合（conical combination）构造的逐元素流：

$$
h'_b = f[h, \phi] = \sqrt{K h - b + 1}\, \phi_b + \sum_{k=1}^{b-1} \sqrt{K h - k + 1}\, \phi_k,
\tag{16.33}
$$

其中 $b = \lfloor K h \rfloor + 1$ 表示 $h$ 所落入的桶编号，参数 $\phi_k > 0$ 且满足 $\sum_k \phi_k = 1$。取 $K = 5$，$\phi_1 = 0.1$、$\phi_2 = 0.2$、$\phi_3 = 0.5$、$\phi_4 = 0.1$、$\phi_5 = 0.1$，请绘制函数 $f[h, \phi]$ 及其反函数 $f^{-1}[h', \phi]$ 的图像。

**习题 16.10**  
请画出图 16.8 所示残差流（residual flow）前向映射的雅可比矩阵结构图（标出零元素位置），分别考虑以下两种情形：  
(i) $f_1[\cdot, \phi_1]$ 和 $f_2[\cdot, \phi_2]$ 均为全连接神经网络；  
(ii) $f_1[\cdot, \phi_1]$ 和 $f_2[\cdot, \phi_2]$ 均为逐元素流（elementwise flow）。

**习题 16.11\***  
请显式写出式 (16.25) 中 KL 散度的表达式。为何即使我们仅能以某个比例因子 $\kappa$ 确定概率密度 $q(x)$（即只能计算 $\kappa\, q(x)$），该 KL 散度仍可被有效优化？为最小化该损失函数，网络是否必须可逆？请阐明你的理由。

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社（MIT Press）。