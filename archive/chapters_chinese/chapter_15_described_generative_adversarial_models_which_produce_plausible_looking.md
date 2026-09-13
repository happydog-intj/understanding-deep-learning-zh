# 第15章：介绍生成对抗模型，该模型可生成看似真实的样本

*页码：363–387*

第18章  
扩散模型  

第15章介绍了生成对抗网络（GAN），这类模型能够生成视觉上逼真的样本，但并未在数据空间上定义一个显式的概率分布。第16章讨论了标准化流（normalizing flows）：这类模型确实定义了这样的概率分布，但对网络架构施加了严格约束——每一层必须可逆，且其雅可比矩阵的行列式需易于计算。第17章引入了变分自编码器（VAE），它同样具备坚实的概率建模基础，但数据似然（data likelihood）的精确计算是难解的（intractable），因而需借助一个基于编码器的下界（lower bound）进行近似。

本章介绍扩散模型（diffusion models）。与标准化流类似，扩散模型也是概率模型，定义了一个从隐变量到观测数据的非线性映射，且二者维度相同；与变分自编码器类似，它也通过一个编码器将数据映射至隐空间，并基于该编码器构造似然函数的下界来近似数据似然。然而，在扩散模型中，该编码器是预先设定（prespecified）的；学习目标仅在于构建一个解码器，使其成为该预设编码过程的逆过程，并可用于生成新样本。扩散模型训练简便，且能生成极高保真度的样本，其真实感甚至超越生成对抗网络所生成的样本。读者在阅读本章前，应已熟悉第17章所述的变分自编码器。

18.1 概述  
扩散模型由一个编码器和一个解码器组成。编码器接收一个数据样本 $ \mathbf{x} $，并将其经由一系列中间隐变量 $ \mathbf{z}_1, \dots, \mathbf{z}_T $ 逐步变换；解码器则执行逆向过程：它以 $ \mathbf{z}_T $ 为起点，依次反向映射经过 $ \mathbf{z}_{T-1}, \dots, \mathbf{z}_1 $，最终（重新）生成一个数据点 $ \mathbf{x} $。在编码器与解码器中，所有映射均为随机过程（stochastic），而非确定性函数。

编码器是预先设定的：它通过一系列步骤，逐步将输入与白噪声样本相混合（见图18.1）。当步数足够多时，最终隐变量 $ \mathbf{z}_T $ 的条件分布 $ q(\mathbf{z}_T \mid \mathbf{x}) $ 及其边缘分布 $ q(\mathbf{z}_T) $ 均收敛于标准正态分布。由于该编码过程完全预设，所有待学习的参数均位于解码器中。

在解码器中，一系列神经网络被联合训练，用以逐级实现从 $ \mathbf{z}_t $ 到 $ \mathbf{z}_{t-1} $ 的逆向映射。  

草稿：请将勘误发送至 udlbookmail@gmail.com。

350 18 扩散模型  
图18.1 扩散模型。编码器（前向过程，亦称扩散过程）将输入 $x$ 经过一系列潜在变量 $z_1, \dots, z_T$ 进行映射。该过程是预先设定的，逐步将数据与噪声混合，直至最终仅剩纯噪声。解码器（反向过程）则通过学习获得，它将数据沿潜在变量序列逆向传递，在每一阶段逐步去除噪声。训练完成后，新样本的生成方式为：从噪声向量 $z_T$ 中采样，并将其输入解码器。

相邻的两个潜在变量 $z_t$ 与 $z_{t-1}$ 构成一对。损失函数促使每个网络尽可能地逆转对应的编码器步骤。其结果是，表示中的噪声被逐级去除，最终保留一个视觉上逼真的数据样本。为生成一个新的数据样本 $x$，我们首先从 $q(z_T)$ 中采样，再将其送入解码器。

在第18.2节中，我们将详细考察编码器。其性质并不直观，但对学习算法至关重要。第18.3节讨论解码器。第18.4节推导训练算法，第18.5节则对其加以重构，使其更具实用性。第18.6节介绍实现细节，包括如何使生成过程以文本提示为条件。

18.2 编码器（前向过程）  
扩散过程（或称前向过程）¹（见图18.2）将一个数据样本 $x$ 映射为一系列中间变量 $z_1, z_2, \dots, z_T$，这些变量均与 $x$ 具有相同维度，映射规则如下：

$$
z_1 = \sqrt{1 - \beta_1} \cdot x + \sqrt{\beta_1} \cdot \epsilon_1 \tag{18.1}
$$
$$
z_t = \sqrt{1 - \beta_t} \cdot z_{t-1} + \sqrt{\beta_t} \cdot \epsilon_t \quad \forall t \in \{2, \dots, T\},
$$

其中 $\epsilon_t$ 是从标准正态分布中采样的噪声。第一项衰减了当前数据及其此前已加入的全部噪声；第二项则引入额外噪声。超参数 $\beta_t \in [0,1]$ 控制噪声混合的速度，统称为**噪声调度表**（noise schedule）。前向过程亦可等价地表示为：

¹ 注：此处术语命名与标准化流（normalizing flows）相反——在标准化流中，“逆向映射”指从数据到潜在变量的映射，而“前向映射”则指从潜在变量返回数据的映射。  
本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社。

18.2 编码器（前向过程） 351  
图 18.2 前向过程。  
a) 我们考虑一维数据 $x$，并引入 $T = 100$ 个隐变量 $z_1, \dots, z_{100}$，且每一步的 $\beta_t = 0.03$。图中顶部一行展示了三个初始值 $x$（灰色、青色和橙色）。这些初始值依次经由 $z_1, \dots, z_{100}$ 传播。在每一步中，当前变量按比例衰减为 $\sqrt{1-\beta_t}$ 倍，并叠加均值为 0、方差为 $\beta_t$ 的高斯噪声（见公式 18.1）。因此，这三个样本在传播过程中逐渐混入噪声，并整体趋向于零。  
b) 条件概率 $\Pr(z_1|x)$ 和 $\Pr(z_t|z_{t-1})$ 均为正态分布：其均值略小于当前点（即更靠近零），方差固定为 $\beta_t$（见公式 18.2）。  

$$
q(z_1|x) = \mathrm{Norm}\!\left(\sqrt{1-\beta_1}\,x,\, \beta_1\mathbf{I}\right) \tag{18.2}
$$  
$$
q(z_t|z_{t-1}) = \mathrm{Norm}\!\left(\sqrt{1-\beta_t}\,z_{t-1},\, \beta_t\mathbf{I}\right) \quad \forall t \in \{2,\dots,T\}.
$$  

这是一个马尔可夫链，因为 $z_t$ 的概率完全由前一时刻的变量 $z_{t-1}$ 决定。当步数 $T$ 足够大时，原始数据的所有痕迹均被消除，此时 $q(z_T|x) = q(z_T)$ 退化为标准正态分布。  

**问题 18.1**  
给定输入 $x$，所有隐变量 $z_1, z_2, \dots, z_T$ 的联合分布为：  

$$
q(z_{1\ldots T}|x) = q(z_1|x)\prod_{t=2}^{T} q(z_t|z_{t-1}). \tag{18.3}
$$  

² 我们采用记号 $q(z_t|z_{t-1})$ 而非 $\Pr(z_t|z_{t-1})$，以与前一章中变分自编码器（VAE）编码器部分所使用的符号保持一致。  
草稿：请将勘误发送至 udlbookmail@gmail.com。

352 18 扩散模型  
图18.3 扩散核。a）点 $x^*=2.0$ 通过隐变量沿式（18.1）进行传播（图中以五条灰色路径示意）。扩散核 $q(z_t \mid x^*)$ 表示在起始点为 $x^*$ 的前提下，关于变量 $z_t$ 的概率分布。该分布具有闭式解，且为正态分布：其均值随 $t$ 增大而趋近于零，方差则随 $t$ 增大而增大。热力图展示了每个变量上的 $q(z_t \mid x^*)$；青色线条表示均值上下各两个标准差的范围。b）显式绘出 $t=20$、$40$、$80$ 时的扩散核 $q(z_t \mid x^*)$。在实际应用中，扩散核使我们能够直接对与给定 $x^*$ 对应的隐变量 $z_t$ 进行采样，而无需计算中间变量 $z_1,\dots,z_{t-1}$。当 $t$ 变得非常大时，扩散核收敛为标准正态分布。  

图18.4 边缘分布。a）给定初始密度 $\Pr(x)$（最上方一行），扩散过程在经过隐变量 $z_t$ 时逐步模糊该分布，并将其逐渐推向标准正态分布。热力图中每一后续水平行代表一个边缘分布 $q(z_t)$。b）顶部图像显示初始分布 $\Pr(x)$；其余两张图像则分别显示边缘分布 $q(z_{20})$ 和 $q(z_{60})$。  

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社。

18.2 编码器（前向过程） 353  
18.2.1 扩散核 $ q(\mathbf{z}_t \mid \mathbf{x}) $  

为训练解码器以逆转该过程，我们对同一输入样本 $ \mathbf{x} $ 在时刻 $ t $ 处使用多个采样点 $ \mathbf{z}_t $。然而，当 $ t $ 较大时，若按式 (18.1) 顺序生成这些采样点，计算开销将十分高昂。幸运的是，$ q(\mathbf{z}_t \mid \mathbf{x}) $ 存在闭式表达式，使我们能够直接从初始数据点 $ \mathbf{x} $ 采样出 $ \mathbf{z}_t $，而无需显式计算中间变量 $ \mathbf{z}_1, \dots, \mathbf{z}_{t-1} $。这一表达式被称为**扩散核**（见图 18.3）。  

为推导 $ q(\mathbf{z}_t \mid \mathbf{x}) $ 的表达式，考虑前向过程的前两步：  
$$
\mathbf{z}_1 = \sqrt{1 - \beta_1}\, \mathbf{x} + \sqrt{\beta_1}\, \boldsymbol{\epsilon}_1, \\
\mathbf{z}_2 = \sqrt{1 - \beta_2}\, \mathbf{z}_1 + \sqrt{\beta_2}\, \boldsymbol{\epsilon}_2. \tag{18.4}
$$  

将第一式代入第二式，得：  
$$
\mathbf{z}_2 = \sqrt{1 - \beta_2}\left( \sqrt{1 - \beta_1}\, \mathbf{x} + \sqrt{\beta_1}\, \boldsymbol{\epsilon}_1 \right) + \sqrt{\beta_2}\, \boldsymbol{\epsilon}_2 \tag{18.5}
$$  
$$
= \sqrt{(1 - \beta_2)(1 - \beta_1)}\, \mathbf{x} + \sqrt{1 - \beta_2}\sqrt{\beta_1}\, \boldsymbol{\epsilon}_1 + \sqrt{\beta_2}\, \boldsymbol{\epsilon}_2
$$  
$$
= \sqrt{(1 - \beta_2)(1 - \beta_1)}\, \mathbf{x} + \sqrt{1 - (1 - \beta_2)(1 - \beta_1)}\, \boldsymbol{\epsilon}, \tag{18.6}
$$  
其中 $ \boldsymbol{\epsilon} $ 同样服从标准正态分布。  

最后两项分别为均值为零、方差分别为 $ 1 - \beta_2 - (1 - \beta_2)(1 - \beta_1) $ 和 $ \beta_2 $ 的独立正态分布采样。该和的均值为零（参见习题 18.2），其方差等于各分量方差之和，故可合并为单个标准正态扰动项（如式 (18.6) 所示）。  

若继续将该式代入 $ \mathbf{z}_3 $ 的表达式，并依此类推，可归纳证明：  
$$
\mathbf{z}_t = \sqrt{\alpha_t}\, \mathbf{x} + \sqrt{1 - \alpha_t}\, \boldsymbol{\epsilon}, \tag{18.7}
$$  
其中 $ \alpha_t = \prod_{s=1}^t (1 - \beta_s) $。等价地，可将其写为概率形式：  
$$
q(\mathbf{z}_t \mid \mathbf{x}) = \mathcal{N}\!\left( \sqrt{\alpha_t}\, \mathbf{x},\, (1 - \alpha_t)\mathbf{I} \right). \tag{18.8}
$$  

对于任意起始数据点 $ \mathbf{x} $，变量 $ \mathbf{z}_t $ 均服从具有已知均值与方差的正态分布。因此，若不关心经由中间变量 $ \mathbf{z}_1, \dots, \mathbf{z}_{t-1} $ 演化的具体路径，则从 $ q(\mathbf{z}_t \mid \mathbf{x}) $ 中采样将极为简便。  

18.2.2 边缘分布 $ q(\mathbf{z}_t) $  

边缘分布 $ q(\mathbf{z}_t) $ 表示：在所有可能的起始点 $ \mathbf{x} $ 及其各自对应的扩散路径下，观测到 $ \mathbf{z}_t $ 取某特定值的概率。  

草案：勘误请发送至 udlbookmail@gmail.com。

354 18 扩散模型  
点（图18.4）。它可通过考虑联合分布 $q(\mathbf{x}, \mathbf{z}_{1\ldots t})$ 并对除 $\mathbf{z}_t$ 外的所有变量进行边缘化而得到：  
**边缘化**（附录 C.1.2）  
$$
q(\mathbf{z}_t) = \int\!\!\int q(\mathbf{z}_{1\ldots t}, \mathbf{x}) \, d\mathbf{z}_{1\ldots t-1} \, d\mathbf{x} \\
= \int\!\!\int q(\mathbf{z}_{1\ldots t} \mid \mathbf{x}) \, \mathrm{Pr}(\mathbf{x}) \, d\mathbf{z}_{1\ldots t-1} \, d\mathbf{x}, \quad \text{(18.9)}
$$  
其中 $q(\mathbf{z}_{1\ldots t} \mid \mathbf{x})$ 在式（18.3）中已定义。  

然而，由于我们现在已获得一个“跳过”中间变量的扩散核 $q(\mathbf{z}_t \mid \mathbf{x})$ 的显式表达式，因此也可等价地写作：  
$$
q(\mathbf{z}_t) = \int q(\mathbf{z}_t \mid \mathbf{x}) \, \mathrm{Pr}(\mathbf{x}) \, d\mathbf{x}. \quad \text{(18.10)}
$$  

因此，若我们反复从数据分布 $\mathrm{Pr}(\mathbf{x})$ 中采样，并在每个样本上叠加扩散核 $q(\mathbf{z}_t \mid \mathbf{x})$，所得结果即为边缘分布 $q(\mathbf{z}_t)$（图18.4；参见 Notebook 18.1）。但该边缘分布无法以闭式表达，因为我们并不知晓原始数据分布 $\mathrm{Pr}(\mathbf{x})$。  

**18.2.3 条件分布 $q(\mathbf{z}_{t-1} \mid \mathbf{z}_t)$**  
我们将条件概率 $q(\mathbf{z}_t \mid \mathbf{z}_{t-1})$ 定义为混合过程（式（18.2））。  
**贝叶斯法则**（附录 C.1.4）  
为逆转该过程，我们应用贝叶斯法则：  
$$
q(\mathbf{z}_{t-1} \mid \mathbf{z}_t) = \frac{q(\mathbf{z}_t \mid \mathbf{z}_{t-1}) \, q(\mathbf{z}_{t-1})}{q(\mathbf{z}_t)}. \quad \text{(18.11)}
$$  
该式不可解析求解，因为我们无法计算边缘分布 $q(\mathbf{z}_{t-1})$。  

对于本例这一简单的 1D 情形，可对 $q(\mathbf{z}_{t-1} \mid \mathbf{z}_t)$ 进行数值计算（图18.5）。一般而言，其形式较为复杂，但在许多情况下，可用正态分布对其进行良好近似。这一点至关重要，因为我们在构建解码器时，将使用正态分布来近似反向过程。  

**18.2.4 条件扩散分布 $q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x})$**  
尚需考虑与编码器相关的最后一个分布。前文指出，由于未知边缘分布 $q(\mathbf{z}_{t-1})$，我们无法求得条件分布 $q(\mathbf{z}_{t-1} \mid \mathbf{z}_t)$。然而，若已知起始变量 $\mathbf{x}$，则我们便知晓其前一时刻的分布 $q(\mathbf{z}_{t-1} \mid \mathbf{x})$。这正是扩散核（图18.3），且服从正态分布。  

因此，条件扩散分布 $q(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \mathbf{x})$ 可以闭式形式精确计算（图18.6）。该分布用于训练解码器：它表示在已知当前隐变量 $\mathbf{z}_t$ 和训练样本 $\mathbf{x}$ 的前提下，$\mathbf{z}_{t-1}$ 的分布。  

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（C）麻省理工学院出版社。

18.2 编码器（前向过程） 355  
图 18.5 条件分布 $ q(z_{t-1} \mid z_t) $。  
a) 边缘密度 $ q(z_t) $，其中标出了三个点 $ z_t^* $。  
b) 概率 $ q(z_{t-1} \mid z_t^*) $（青色曲线）通过贝叶斯定理计算，正比于 $ q(z_t^* \mid z_{t-1})\,q(z_{t-1}) $。一般而言，该分布并非正态分布（上图），尽管在多数情况下正态分布是一个良好的近似（下图两幅）。第一项似然 $ q(z_t^* \mid z_{t-1}) $ 关于 $ z_{t-1} $ 是正态分布的（式 (18.2)），其均值略远离零点，且比 $ z_t^* $ 更远（棕色曲线）。第二项为边缘密度 $ q(z_{t-1}) $（灰色曲线）。  

图 18.6 条件分布 $ q(z_{t-1} \mid z_t, x) $。  
a) 针对 $ x^* = -2.1 $ 的扩散核，其中标出了三个点 $ z_t^* $。  
b) 概率 $ q(z_{t-1} \mid z_t^*, x^*) $ 通过贝叶斯定理计算，正比于 $ q(z_t^* \mid z_{t-1})\,q(z_{t-1} \mid x^*) $。该分布为正态分布，且可解析求解（closed-form solution）。第一项似然 $ q(z_t^* \mid z_{t-1}) $ 关于 $ z_{t-1} $ 是正态分布的（式 (18.2)），其均值略远离零点，且比 $ z_t^* $ 更远（棕色曲线）。第二项为扩散核 $ q(z_{t-1} \mid x^*) $（灰色曲线）。  

草稿：如有勘误，请发送至 udlbookmail@gmail.com。

356 18 扩散模型  
数据样本 $x$（当然，我们在训练时正是使用该样本）。为推导出 $q(z_{t-1} \mid z_t, x)$ 的表达式，我们从贝叶斯定理出发：  

$$
q(z_{t-1} \mid z_t, x) = \frac{q(z_t \mid z_{t-1}, x)\, q(z_{t-1} \mid x)}{q(z_t \mid x)} \tag{18.12}
$$  

$$
\propto q(z_t \mid z_{t-1})\, q(z_{t-1} \mid x) 
= \operatorname{Norm}\!\left[z_t; \sqrt{1-\beta_t}\,z_{t-1},\, \beta_t \mathbf{I}_N\right] 
\cdot \operatorname{Norm}\!\left[z_{t-1}; \sqrt{\alpha_{t-1}}\,x,\, (1-\alpha_{t-1})\mathbf{I}\right]
$$  

$$
\propto \operatorname{Norm}\!\left[z_{t-1}; \frac{1}{\sqrt{1-\beta_t}}\,z_t,\, \frac{\beta_t}{1-\beta_t}\mathbf{I}\right] 
\cdot \operatorname{Norm}\!\left[z_{t-1}; \sqrt{\alpha_{t-1}}\,x,\, (1-\alpha_{t-1})\mathbf{I}\right],
$$  

其中，第一行至第二行之间，我们利用了扩散过程的马尔可夫性质：$q(z_t \mid z_{t-1}, x) = q(z_t \mid z_{t-1})$，因为关于 $z_t$ 的全部信息均已由 $z_{t-1}$ 捕获。第三行至第四行之间，则应用了高斯变量变换恒等式（见附录 C.3.4）：  

**高斯变量变换恒等式**  
$$
\operatorname{Norm}[A\mathbf{w} + \mathbf{v},\, \mathbf{B}] \propto \operatorname{Norm}\!\left[\mathbf{w};\, (\mathbf{A}^\top \mathbf{B}^{-1}\mathbf{A})^{-1} \mathbf{A}^\top \mathbf{B}^{-1}\mathbf{v},\, (\mathbf{A}^\top \mathbf{B}^{-1}\mathbf{A})^{-1}\right], \tag{18.13}
$$  

将第一个分布重写为关于 $z_{t-1}$ 的形式。随后，我们再应用第二个高斯恒等式（习题 18.4–18.5）：  

$$
\operatorname{Norm}[\mathbf{a},\, \mathbf{A}] \cdot \operatorname{Norm}[\mathbf{b},\, \mathbf{B}] \propto 
\operatorname{Norm}\!\left[\mathbf{w};\, (\mathbf{A}^{-1} + \mathbf{B}^{-1})^{-1}(\mathbf{A}^{-1}\mathbf{a} + \mathbf{B}^{-1}\mathbf{b}),\, (\mathbf{A}^{-1} + \mathbf{B}^{-1})^{-1}\right], \tag{18.14}
$$  

以合并两个关于 $z_{t-1}$ 的正态分布，最终得到：  

**习题 18.6**  
$$
q(z_{t-1} \mid z_t, x) = \operatorname{Norm}\!\left[z_{t-1};\, \frac{1 - \alpha_{t-1}}{1 - \alpha_t} \frac{1}{\sqrt{1 - \beta_t}} z_t + \frac{\alpha_{t-1} \beta_t}{1 - \alpha_t} x,\, \frac{\beta_t (1 - \alpha_{t-1})}{1 - \alpha_t} \mathbf{I}\right]. \tag{18.15}
$$  

注意：式 (18.12)、(18.13) 和 (18.14) 中出现的比例常数必然相互抵消，因为最终结果本身已是正确归一化的概率分布。  

### 18.3 解码器模型（逆向过程）  
当我们训练一个扩散模型时，实际学习的是其逆向过程。换言之，我们学习一系列从潜在变量 $z_T$ 回溯至 $z_{T-1}$、再从 $z_{T-1}$ 回溯至 $z_{T-2}$，依此类推，直至抵达原始数据 $x$ 的概率映射。扩散过程的真实逆向分布 $q(z_{t-1} \mid z_t)$ 是高度复杂的多峰分布（见图 18.5），其形态依赖于数据分布 $\Pr(x)$。我们以正态分布对其进行近似：  

$$
\Pr(z_T) = \operatorname{Norm}[0,\, \mathbf{I}] \\
\Pr(z_{t-1} \mid z_t, \boldsymbol{\phi}_t) = \operatorname{Norm}\!\left[z_{t-1};\, f_t[z_t, \boldsymbol{\phi}_t],\, \sigma_t^2 \mathbf{I}\right] \\
\Pr(x \mid z_1, \boldsymbol{\phi}_1) = \operatorname{Norm}\!\left[x;\, f_x[z_1, \boldsymbol{\phi}_1],\, \sigma_1^2 \mathbf{I}\right], \tag{18.16}
$$  

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（C）麻省理工学院出版社。

18.4 训练　357  
其中，$f[z_t,\phi_t]$ 是一个神经网络，用于计算从隐变量 $z_t$ 到前一时刻隐变量 $z_{t-1}$ 的估计映射所对应的正态分布的均值。各项 $\{\sigma_t^2\}$ 是预先设定的。若扩散过程中的超参数 $\beta_t$ 接近于零（且时间步总数 $T$ 较大），则该正态近似将是合理的。  

我们通过祖先采样（ancestral sampling）从分布 $P_r(x)$ 中生成新的样本。首先从先验分布 $P_r(z_T)$ 中采样得到 $z_T$；接着从条件分布 $P_r(z_{T-1} \mid z_T, \phi_T)$ 中采样得到 $z_{T-1}$，再从 $P_r(z_{T-2} \mid z_{T-1}, \phi_{T-1})$ 中采样得到 $z_{T-2}$，依此类推，直至最终从 $P_r(x \mid z_1, \phi_1)$ 中采样生成观测数据 $x$。

18.4 训练  
观测变量 $x$ 与隐变量序列 $\{z_t\}$ 的联合分布为：  
$$
P_r(x, z_{1\ldots T} \mid \phi_{1\ldots T}) = P_r(x \mid z_1, \phi_1) \prod_{t=2}^{T} P_r(z_{t-1} \mid z_t, \phi_t) \cdot P_r(z_T). \tag{18.17}
$$  

观测数据的似然函数 $P_r(x \mid \phi_{1\ldots T})$ 可通过对隐变量进行边缘化（marginalization）获得（参见附录 C.1.2）：  
$$
P_r(x \mid \phi_{1\ldots T}) = \int P_r(x, z_{1\ldots T} \mid \phi_{1\ldots T}) \, dz_{1\ldots T}. \tag{18.18}
$$  

为训练模型，我们需关于参数 $\phi_{1\ldots T}$ 最大化训练数据集 $\{x_i\}$ 的对数似然：  
$$
\hat{\phi}_{1\ldots T} = \arg\max_{\phi_{1\ldots T}} \left[ \sum_{i=1}^{I} \log P_r(x_i \mid \phi_{1\ldots T}) \right]. \tag{18.19}
$$  

由于式 (18.18) 中的边缘化积分不可解析求解，我们无法直接最大化该目标函数。因此，我们采用 Jensen 不等式构造似然函数的一个下界（lower bound），并如变分自编码器（VAE）中那样（参见第 17.3.1 节），对该下界关于参数 $\phi_{1\ldots T}$ 进行优化。

18.4.1 证据下界（ELBO）  
为推导该下界，我们在对数似然中乘以并除以编码器分布 $q(z_{1\ldots T} \mid x)$，并应用 Jensen 不等式（参见第 17.3.2 节）：  
$$
\begin{aligned}
\log\big[P_r(x \mid \phi_{1\ldots T})\big] 
&= \log \left[ \int P_r(x, z_{1\ldots T} \mid \phi_{1\ldots T}) \, dz_{1\ldots T} \right] \\
&= \log \left[ \int q(z_{1\ldots T} \mid x) \frac{P_r(x, z_{1\ldots T} \mid \phi_{1\ldots T})}{q(z_{1\ldots T} \mid x)} \, dz_{1\ldots T} \right] \\
&\ge \int q(z_{1\ldots T} \mid x) \log \frac{P_r(x, z_{1\ldots T} \mid \phi_{1\ldots T})}{q(z_{1\ldots T} \mid x)} \, dz_{1\ldots T}. \tag{18.20}
\end{aligned}
$$  

草稿：请将勘误发送至 udlbookmail@gmail.com。

358 18 扩散模型  
由此可得证据下界（Evidence Lower Bound，ELBO）：  
$$
\mathrm{ELBO}_{\phi_{1\ldots T}} = \int q(z_{1\ldots T} \mid x) \log \frac{\Pr(x, z_{1\ldots T} \mid \phi_{1\ldots T})}{q(z_{1\ldots T} \mid x)} \, dz_{1\ldots T}. \tag{18.21}
$$  

在变分自编码器（VAE）中，编码器 $q(z \mid x)$ 通过逼近隐变量的后验分布，使该下界尽可能紧致；而解码器则负责最大化该下界（见图 17.10）。  
在扩散模型中，全部优化工作均由解码器承担，因为编码器本身无参数可学习。解码器通过以下两种方式使下界更紧致：(i) 调整自身参数，使得固定形式的编码器近似逼近后验分布 $\Pr(z_{1\ldots T} \mid x, \phi_{1\ldots T})$；(ii) 在该下界目标上联合优化其自身参数（参见图 17.6）。

18.4.2 ELBO 的简化  

接下来，我们将 ELBO 中的对数项推导为最终待优化的形式。首先，将式 (18.17) 和式 (18.3) 分别代入分子与分母的定义：  
$$
\log \frac{\Pr(x, z_{1\ldots T} \mid \phi_{1\ldots T})}{q(z_{1\ldots T} \mid x)} 
= \log \frac{\Pr(x \mid z_1, \phi_1) \prod_{t=2}^T \Pr(z_{t-1} \mid z_t, \phi_t) \cdot \Pr(z_T)}{q(z_1 \mid x) \prod_{t=2}^T q(z_t \mid z_{t-1})}
$$  
$$
= \log \Pr(x \mid z_1, \phi_1) + \sum_{t=2}^T \log \frac{\Pr(z_{t-1} \mid z_t, \phi_t)}{q(z_t \mid z_{t-1})} + \log \frac{\Pr(z_T)}{q(z_T \mid x)}. \tag{18.22}
$$  

接着展开第二项分母：  
$$
q(z_t \mid z_{t-1}) = q(z_t \mid z_{t-1}, x) = \frac{q(z_{t-1} \mid z_t, x)\, q(z_t \mid x)}{q(z_{t-1} \mid x)}, \tag{18.23}
$$  
其中第一个等号成立，是因为关于变量 $z_t$ 的全部信息均已包含于 $z_{t-1}$ 中，因此额外以数据 $x$ 为条件是冗余的；第二个等号则是贝叶斯定理（Bayes’ rule）的直接应用（参见附录 C.1.4）。

将式 (18.23) 代入后得：  
$$
\log \frac{\Pr(x, z_{1\ldots T} \mid \phi_{1\ldots T})}{q(z_{1\ldots T} \mid x)} 
= \log \Pr(x \mid z_1, \phi_1) + \sum_{t=2}^T \log \frac{\Pr(z_{t-1} \mid z_t, \phi_t)\, q(z_{t-1} \mid x)}{q(z_{t-1} \mid z_t, x)\, q(z_t \mid x)} + \log \frac{\Pr(z_T)}{q(z_T \mid x)}
$$  
$$
= \log \Pr(x \mid z_1, \phi_1) + \sum_{t=2}^T \log \frac{\Pr(z_{t-1} \mid z_t, \phi_t)}{q(z_{t-1} \mid z_t, x)} + \log \frac{\Pr(z_T)}{q(z_T \mid x)}
$$  
$$
\approx \log \Pr(x \mid z_1, \phi_1) + \sum_{t=2}^T \log \frac{\Pr(z_{t-1} \mid z_t, \phi_t)}{q(z_{t-1} \mid z_t, x)}. \tag{18.24}
$$  

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（C）麻省理工学院出版社。

18.4 训练　359  
其中，比值乘积 $q(z_{t-1} \mid x)/q(z_t \mid x)$ 中除两项外的所有项在第二行与第三行之间相互抵消，最终仅余下 $q(z_1 \mid x)$ 和 $q(z_T \mid x)$。第三行的最后一项近似为 $\log[1] = 0$，这是因为前向过程的结果 $q(z_T \mid x)$ 是一个标准正态分布，故其等于先验分布 $\Pr(z_T)$。

因此，简化后的证据下界（ELBO）为：  

$$
\mathrm{ELBO}_\phi = \mathbb{E}_{q(z_{1\ldots T} \mid x)} \left[ \log \frac{\Pr(x, z_{1\ldots T} \mid \phi)}{q(z_{1\ldots T} \mid x)} \right] \tag{18.25}
$$

$$
= \int q(z_{1\ldots T} \mid x) \log \frac{\Pr(x, z_{1\ldots T} \mid \phi)}{q(z_{1\ldots T} \mid x)} \, dz_{1\ldots T}
$$

$$
\approx \int q(z_{1\ldots T} \mid x) \left[ \log \Pr(x \mid z_1, \phi_1) + \sum_{t=2}^T \log \frac{\Pr(z_{t-1} \mid z_t, \phi_t)}{q(z_{t-1} \mid z_t, x)} \right] dz_{1\ldots T}
$$

$$
= \mathbb{E}_{q(z_1 \mid x)} \left[ \log \Pr(x \mid z_1, \phi_1) \right] - \sum_{t=2}^T \mathbb{E}_{q(z_t \mid x)} \left[ D_{\mathrm{KL}} \Big( q(z_{t-1} \mid z_t, x) \,\|\, \Pr(z_{t-1} \mid z_t, \phi_t) \Big) \right],
$$

其中，我们在第二行与第三行之间对 $q(z_{1\ldots T} \mid x)$ 中无关变量进行了边缘化，并利用了 KL 散度的定义（参见习题 18.7）。  
附录 C.5.1　KL 散度  

18.4.3　ELBO 的分析  

ELBO 中的第一项概率由式 (18.16) 定义：  

$$
\Pr(x \mid z_1, \phi_1) = \mathcal{N}\big( f_1[z_1, \phi_1],\, \sigma_x^2 I \big), \tag{18.26}
$$  

该式等价于变分自编码器（VAE）中的重构项。当模型预测与观测数据越吻合时，ELBO 值越大。与 VAE 类似，我们将通过蒙特卡洛估计（参见式 (17.22)–(17.23)）来近似该对数项的期望值，即使用从 $q(z_1 \mid x)$ 中采样的样本来估计该期望。  

ELBO 中的 KL 散度项用于衡量 $\Pr(z_{t-1} \mid z_t, \phi_t)$ 与 $q(z_{t-1} \mid z_t, x)$ 之间的距离；这两者分别由式 (18.16) 和式 (18.15) 定义：  

$$
\Pr(z_{t-1} \mid z_t, \phi_t) = \mathcal{N}\Big( f_t[z_t, \phi_t],\, \sigma_t^2 I \Big), \tag{18.27}
$$  

$$
q(z_{t-1} \mid z_t, x) = \mathcal{N}\Bigg( \sqrt{1 - \beta_t}\, z_t + \frac{\beta_t}{\sqrt{1 - \alpha_{t-1}}}\, x,\; \frac{\beta_t (1 - \alpha_{t-1})}{\alpha_t}\, I \Bigg).
$$  

两个正态分布之间的 KL 散度具有闭式表达式。此外，该表达式中许多项并不依赖于参数 $\phi$（参见习题 18.8），因而可进一步简化为均值之差的平方加上一个常数 $C$：  

$$
D_{\mathrm{KL}} \Big( q(z_{t-1} \mid z_t, x) \,\|\, \Pr(z_{t-1} \mid z_t, \phi_t) \Big) = \tag{18.28}
$$  

$$
\frac{1}{2\sigma_t^2} \left\| \sqrt{1 - \beta_t}\, z_t + \frac{\beta_t}{\sqrt{1 - \alpha_{t-1}}}\, x - f_t[z_t, \phi_t] \right\|^2 + C.
$$  

附录 C.5.4　正态分布间的 KL 散度  
习题 18.8  

草稿：请将勘误发送至 udlbookmail@gmail.com。

360 第18章 扩散模型  
图18.7 拟合模型。  
a) 可通过从标准正态分布 $ P_r(\mathbf{z}) $（最底行）中采样，然后依次从条件分布 $ P_r(\mathbf{z}_{T-1} \,|\, \mathbf{z}_T) = \mathrm{Norm}_{\mathbf{z}_{T-1}}\!\big[\mathbf{f}_T[\mathbf{z}_T,\boldsymbol{\phi}_T],\,\sigma_T^2\mathbf{I}\big] $ 中采样 $ \mathbf{z}_{T-1} $，依此类推，直至得到原始数据 $ \mathbf{x} $（图中展示了五条采样路径）。所估计的边缘密度（热力图）由这些样本聚合而成，其形态与真实边缘密度（图18.4）高度相似。  
b) 所估计的条件分布 $ P_r(\mathbf{z}_{t-1} \,|\, \mathbf{z}_t) $（棕色曲线）是对扩散模型真实后验分布 $ q(\mathbf{z}_{t-1} \,|\, \mathbf{z}_t) $（青色曲线，见图18.5）的一个合理近似。此外，所估计模型与真实模型的边缘分布 $ P_r(\mathbf{z}_t) $ 与 $ q(\mathbf{z}_t) $（分别为深蓝色与灰色曲线）也十分接近。

18.4.4 扩散损失函数  
为拟合模型，我们关于参数 $ \boldsymbol{\phi}_{1\ldots T} $ 最大化证据下界（ELBO）。等价地，我们将该优化问题转化为最小化问题：在目标函数前添加负号，并以蒙特卡洛采样近似其中的期望项，从而得到如下损失函数：  

$$
\begin{aligned}
\mathcal{L}[\boldsymbol{\phi}_{1\ldots T}] 
&= -\sum_{i=1}^I \left( 
\underbrace{\log \mathrm{Norm}\!\big[\mathbf{f}_1[\mathbf{z}_{i1},\boldsymbol{\phi}_1],\,\sigma_1^2\mathbf{I}\big]}_{\text{重构项}} \right. \\
&\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\

18.5 损失函数的重参数化　361  
图18.8　拟合模型结果。青色与棕色曲线分别为原始密度与估计密度，分别对应图18.4和图18.7顶部各行。竖直条形表示从模型中采样的分箱样本：首先从先验分布 $P_r(z_T)$ 中采样，再如图18.7中所示沿五条路径反向传播，依次经过变量 $z_{T-1}, z_{T-2}, \dots$。

18.4.5　训练流程  
该损失函数可用于为每个扩散时间步长单独训练一个网络。其目标是最小化对前一时刻隐变量的估计值 $f_t[z_t, \phi_t]$ 与该隐变量在给定真实去噪数据 $x$ 条件下的最可能取值之间的差异。  
图18.7与图18.8展示了前述简单一维示例中拟合所得的逆向过程。该模型的训练步骤如下：（i）从原始密度中采集大量样本 $x$ 构成训练数据集；（ii）利用扩散核，对每个时间步 $t$ 预测对应的潜在变量 $z_t$ 的多个实现；（iii）训练模型 $f_t[z_t, \phi_t]$，使其最小化式（18.29）所定义的损失函数。这些模型采用非参数形式（即输入输出均为一维的查表法映射），但在更典型的情形下，它们将由深度神经网络实现。  

18.5　损失函数的重参数化  
尽管式（18.29）所定义的损失函数可直接使用，但实践表明，扩散模型在采用另一种参数化方式时性能更优：此时损失函数被重新设计，使模型的目标变为预测叠加至原始数据样本上、从而生成当前变量的噪声成分。第18.5.1节讨论对目标项（即式（18.29）第二行前两项）的重参数化；第18.5.2节则讨论对网络输出项（即式（18.29）第二行最后一项）的重参数化。  

18.5.1　目标项的重参数化  
原始扩散更新公式为：  
$$
z_t = \sqrt{\alpha_t} \cdot x + \sqrt{1 - \alpha_t} \cdot \epsilon. \tag{18.30}
$$  
由此可知，式（18.28）中的数据项 $x$ 可表示为扩散后图像减去其上所叠加的噪声：  
$$
x = \frac{1}{\sqrt{\alpha_t}} \cdot z_t - \frac{\sqrt{1 - \alpha_t}}{\sqrt{\alpha_t}} \cdot \epsilon. \tag{18.31}
$$  
将式（18.31）代入式（18.29）中的目标项，可得：  

草稿：请将勘误发送至 udlbookmail@gmail.com。

362 18 扩散模型  
$$
\frac{\sqrt{1-\alpha_{t-1}}}{\sqrt{1-\alpha_t}} \sqrt{\frac{1-\beta_t}{1-\alpha_{t-1}}} \, z + \frac{\alpha_{t-1} \beta_t}{1-\alpha_t} x \tag{18.32}
$$  
$$
= \frac{\sqrt{1-\alpha_{t-1}}}{\sqrt{1-\alpha_t}} \sqrt{\frac{1-\beta_t}{1-\alpha_{t-1}}} \, z + \frac{\alpha_{t-1} \beta_t}{1-\alpha_t} \left( \sqrt{\frac{1}{\alpha_t}} \, z - \sqrt{\frac{1-\alpha_t}{\alpha_t}} \, \epsilon \right)
$$  
$$
= \frac{\sqrt{1-\alpha_{t-1}}}{\sqrt{1-\alpha_t}} \sqrt{\frac{1-\beta_t}{1-\alpha_{t-1}}} \, z + \frac{\beta_t}{\sqrt{1-\alpha_t}} \sqrt{\frac{1}{1-\beta_t}} \, z - \frac{\beta_t}{\sqrt{1-\alpha_t}} \sqrt{\frac{1-\alpha_t}{1-\beta_t}} \, \epsilon,
$$  
其中，我们在第二行与第三行之间利用了关系式 $\alpha_t / \alpha_{t-1} = 1 - \beta_t$。  

**习题 18.9**  
进一步化简可得：  
$$
\frac{\sqrt{1-\alpha_{t-1}}}{\sqrt{1-\alpha_t}} \sqrt{\frac{1-\beta_t}{1-\alpha_{t-1}}} \, z + \frac{\alpha_{t-1} \beta_t}{1-\alpha_t} x \tag{18.33}
$$  
$$
= \left[ \frac{\sqrt{1-\alpha_{t-1}}}{\sqrt{1-\alpha_t}} \sqrt{\frac{1-\beta_t}{1-\alpha_{t-1}}} + \frac{\beta_t}{\sqrt{1-\alpha_t}} \sqrt{\frac{1}{1-\beta_t}} \right] z - \frac{\beta_t}{\sqrt{1-\alpha_t}} \sqrt{\frac{1-\alpha_t}{1-\beta_t}} \, \epsilon
$$  
$$
= \left[ \frac{(1-\alpha_{t-1})(1-\beta_t) + \beta_t}{(1-\alpha_t)\sqrt{1-\beta_t}} \right] z - \frac{\beta_t}{\sqrt{1-\alpha_t}} \sqrt{\frac{1-\alpha_t}{1-\beta_t}} \, \epsilon
$$  
$$
= \frac{(1-\alpha_{t-1})(1-\beta_t) + \beta_t}{(1-\alpha_t)\sqrt{1-\beta_t}} \, z - \frac{\beta_t}{\sqrt{1-\alpha_t}} \sqrt{\frac{1-\alpha_t}{1-\beta_t}} \, \epsilon
$$  
$$
= \frac{1-\alpha_t}{(1-\alpha_t)\sqrt{1-\beta_t}} \, z - \frac{\beta_t}{\sqrt{1-\alpha_t}} \sqrt{\frac{1-\alpha_t}{1-\beta_t}} \, \epsilon
$$  
$$
= \frac{1}{\sqrt{1-\beta_t}} \, z - \frac{\beta_t}{\sqrt{1-\alpha_t}} \sqrt{\frac{1-\alpha_t}{1-\beta_t}} \, \epsilon,
$$  
其中，我们在第二行与第三行之间将第一项的分子与分母同乘以 $\sqrt{1-\beta_t}$，随后展开各项，并在第三行与第四行之间对第一项的分子进行了化简。  

**习题 18.10**  
将上述结果代回损失函数（式 (18.29)），可得：  
$$
\mathcal{L}[\phi] = -\sum_{i=1}^I \log \operatorname{Norm}\left( f_1[z_{i1}, \phi_1], \sigma_1^2 I \right) \tag{18.34}
$$  
$$
+ \sum_{t=2}^T \frac{1}{2\sigma_t^2} \left\| \frac{1}{\sqrt{1-\beta_t}} z_{it} - \frac{\sqrt{1-\alpha_t}}{\sqrt{\beta_t (1-\beta_t)}} \, \epsilon_{it} - f_t[z_{it}, \phi_t] \right\|^2.
$$  

**18.5.2 网络参数化的重参数化**  
现在，我们将模型 $ \hat{z}_{t-1} = f_t[z_t, \phi_t] $ 替换为一个新模型 $ \hat{\epsilon} = g_t[z_t, \phi_t] $，该模型直接预测用于构造 $ z_t $ 的噪声 $ \epsilon $（即与原始数据 $ x $ 混合生成 $ z_t $ 的噪声）：  
$$
f_t[z_t, \phi_t] = \frac{1}{\sqrt{1-\beta_t}} z_t - \frac{\beta_t}{\sqrt{1-\alpha_t}} \sqrt{\frac{1-\alpha_t}{1-\beta_t}} \, g_t[z_t, \phi_t]. \tag{18.35}
$$  

本作品采用知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。© MIT 出版社。

18.6 实现　363  
将新模型代入式（18.34）可得如下优化准则：  
\[
L[\phi ]= \sum_{i=1}^{I} \left[ -\log \mathrm{Norm}\!\left( x_i \,\middle|\, f[z_{i1},\phi_1],\,\sigma_1^2 I \right) \right] + \sum_{t=2}^{T} \frac{\beta_t^2}{(1-\alpha_t)(1-\beta_t)^2\sigma_t^2} \left\| g[z_{it},\phi_t] - \epsilon_{it} \right\|^2 \tag{18.36}
\]

根据第5.3.1节，对数正态分布项可重写为最小二乘损失加上一个常数 \(C_i\)：  
\[
L[\phi ]= \sum_{i=1}^{I} \frac{1}{2\sigma_1^2} \left\| x_i - f[z_{i1},\phi_1] \right\|^2 + \sum_{t=2}^{T} \frac{\beta_t^2}{(1-\alpha_t)(1-\beta_t)^2\sigma_t^2} \left\| g[z_{it},\phi_t] - \epsilon_{it} \right\|^2 + C_i \tag{18.37}
\]

将式（18.31）与式（18.35）分别给出的 \(x_i\) 与 \(f[z_{i1},\phi_1]\) 的定义代入上式，第一项可简化为：  
\[
\frac{1}{2\sigma_1^2} \left\| x_i - f_1[z_{i1},\phi_1] \right\|^2 = \frac{1}{2\sigma_1^2} \left\| \sqrt{1-\alpha_1}\sqrt{1-\beta_1}\,g_1[z_{i1},\phi_1] - \sqrt{1-\alpha_1}\sqrt{1-\beta_1}\,\epsilon_{i1} \right\|^2 \tag{18.38}
\]

将该结果代回最终损失函数，得到：  
\[
L[\phi ] = \sum_{i=1}^{I} \sum_{t=1}^{T} \frac{\beta_t^2}{(1-\alpha_t)(1-\beta_t)^2\sigma_t^2} \left\| g[z_{it},\phi_t] - \epsilon_{it} \right\|^2 \tag{18.39}
\]  
其中已忽略加性常数 \(C_i\)。

在实际应用中，通常忽略各时间步可能不同的缩放因子，从而获得更简洁的形式：  
\[
L[\phi ] = \sum_{i=1}^{I} \sum_{t=1}^{T} \left\| g_t[z_{it},\phi_t] - \epsilon_{it} \right\|^2 \tag{18.40}
\]  
\[
= \sum_{i=1}^{I} \sum_{t=1}^{T} \left\| g_t\!\left[ \sqrt{\alpha_t}\cdot x_i + \sqrt{1-\alpha_t}\cdot \epsilon_{it},\,\phi_t \right] - \epsilon_{it} \right\|^2,
\]  
其中第二行已利用扩散核（式（18.30））将 \(z_{it}\) 重写。

18.6 实现  
上述推导直接导出两个清晰明确的算法：用于模型训练的算法（算法18.1）和用于采样的算法（算法18.2）。训练算法具有以下优势：（i）实现简单；（ii）天然具备数据增强能力——我们可在每个时间步对每个原始数据点 \(x_i\) 重复使用任意多次，并每次配以不同的噪声实现 \(\epsilon\)。而采样算法的缺点在于其需串行调用大量神经网络 \(g_t[z_{it},\phi_t]\)，因此计算耗时较长。  

草稿：勘误请发送至 udlbookmail@gmail.com。

364 18 扩散模型  
**算法 18.1：扩散模型训练**  
输入：训练数据 $ \mathbf{x} $  
输出：模型参数 $ \boldsymbol{\phi}_t $  

```
repeat
    for i ∈ B do                      // 对批数据中每个训练样本索引
        t ∼ Uniform[1, ..., T]         // 随机采样时间步
        ϵ ∼ Norm[0, I]                 // 采样高斯噪声
        ℓ_i = || g_t(√α_t x_i + √(1−α_t) ϵ, ϕ_t) − ϵ ||²  // 计算单个样本损失
    end for
    累积批内损失并执行梯度更新
until 收敛
```

**算法 18.2：采样**  
输入：训练好的模型 $ g_t[\cdot, \boldsymbol{\phi}_t] $  
输出：生成样本 $ \mathbf{x} $  

```
z_T ∼ Norm[0, I]                                      // 采样最终隐变量
for t = T ... 2 do
    ϵ̂_t ∼ Norm[ (z_t − √(1−β_t) g_t[z_t, ϕ_t]) / √β_t , I ]   // 预测去噪方向（即后验分布 p(z_{t−1}|z_t) 的均值）
    z_{t−1} = √(1−β_t) g_t[z_t, ϕ_t] + √β_t ϵ̂_t                // 重参数化采样 z_{t−1}
    z_{t−1} = ẑ_{t−1} + σ_t ϵ                                 // 向前一隐变量添加噪声（若 σ_t > 0）
end for
x = √(1−β_1) g_1[z_1, ϕ_1] − √(β_1/(1−β_1)) g_1[z_1, ϕ_1]     // 从 z_1 无噪声生成样本 x
```

**18.6.1 在图像上的应用**  
扩散模型在图像建模任务中取得了巨大成功。在此类任务中，我们需要构建一类模型，使其能够接收含噪图像，并预测在每一步中被加入的噪声。实现该图像到图像映射任务最自然的网络架构是 U-Net（见图 11.10）。然而，实际应用中扩散步数 $ T $ 可能非常大，若为每个时间步单独训练并存储一个 U-Net，则效率极低。解决方案是仅训练一个共享的 U-Net，并额外将表征当前时间步的预设向量作为输入（见图 18.9）。实践中，该时间嵌入向量会被调整尺寸，以匹配 U-Net 各阶段的通道数，并用于在每个空间位置对特征表示进行偏移（offset）和/或缩放（scale）。

需要设置大量时间步的原因在于：当超参数 $ \beta_t $ 趋近于零时，条件转移概率 $ q(\mathbf{z}_{t-1} \mid \mathbf{z}_t) $ 将越来越接近正态分布，从而与解码器分布 $ p_\theta(\mathbf{z}_{t-1} \mid \mathbf{z}_t, \boldsymbol{\phi}_t) $ 的形式相匹配。但这也导致采样过程变得缓慢——为生成高质量图像，我们可能需运行 U-Net 模型多达 $ T = 1000 $ 步。

**18.6.2 加速生成过程**  
损失函数（式 18.40）要求前向扩散核满足形式 $ q(\mathbf{z}_t \mid \mathbf{x}) = \mathrm{Norm}\big[\sqrt{\alpha_t}\,\mathbf{x},\, (1-\alpha_t)\mathbf{I}\big] $。对于任意满足该形式的前向过程，该损失函数均有效。  

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

18.6 实现　365  
图18.9　扩散模型中用于图像处理的U-Net结构。该网络旨在预测被添加至图像中的噪声。其结构包含一个编码器（encoder）和一个解码器（decoder）：编码器逐步降低空间尺度并增加通道数，而解码器则逐步提升空间尺度并减少通道数。编码器各层的特征表示与其在解码器中对应层级的特征表示进行通道拼接（concatenation）。相邻特征表示之间的连接由残差块（residual block）构成，并周期性地引入全局自注意力机制（global self-attention），使得每个空间位置均能与所有其他空间位置交互。整个U-Net网络在所有时间步上共享权重；具体实现方式为：将正弦时间嵌入（sinusoidal time embedding，见图12.5）送入一个浅层神经网络，再将所得结果加到U-Net每一阶段中每个空间位置的通道特征上。

基于这一关系，存在一族彼此兼容的扩散过程。这些过程均采用相同的损失函数进行优化，但在前向过程（forward process）的具体规则上各不相同，且相应地，在反向过程（reverse process）中利用估计噪声 $g[z_t, \phi_t]$ 从 $z_t$ 预测 $z_{t-1}$ 的策略亦各不相同（见图18.10）。

该族中包括去噪扩散隐式模型（Denoising Diffusion Implicit Models, DDIM），其自第一步（从 $x$ 到 $z_1$）之后即不再具有随机性；还包括加速采样模型（accelerated sampling models），其前向过程仅定义于时间步序列的一个子集之上。这使得反向过程可跳过若干中间时间步，从而显著提升采样效率：当前向过程不再具有随机性时，仅需50个时间步即可生成高质量样本（参见笔记本18.4）。  
尽管该方法比此前快得多，但仍慢于大多数其他生成模型。  

**扩散模型族**  

18.6.3　条件生成  
若数据附带标签 $c$，则可利用这些标签对生成过程施加控制。此类条件信息有时可提升生成对抗网络（GAN）的生成质量；我们同样预期其在扩散模型中亦具类似效果——若已知图像所含内容的先验信息，则去噪任务将更为容易。扩散模型中一种典型的条件合成方法是分类器引导（classifier guidance）。该方法对从 $z_t$ 到 $z_{t-1}$ 的去噪更新步骤加以修正，以融入类别信息 $c$。在实践中，这意味着在更新过程中额外引入一项依赖于 $c$ 的梯度项。  

草稿：请将勘误发送至 udlbookmail@gmail.com。

366 18 扩散模型  
图 18.10 与同一模型兼容的不同扩散过程。  
a) 重参数化模型的五条采样轨迹，叠加于真实边缘分布之上。首行为 $P_r(\mathbf{x})$，后续各行依次为 $q(\mathbf{x}_t)$。  
b) 重参数化模型生成样本的直方图，与真实密度曲线 $P_r(\mathbf{x})$ 并列绘制。同一训练好的模型可兼容一族扩散模型（及其对应的反向更新过程），其中包括去噪扩散隐式模型（DDIM）——该模型是确定性的，且在每一步中不添加噪声。  
c) DDIM 的五条轨迹。  
d) DDIM 生成样本的直方图。同一模型还可兼容加速扩散模型，后者通过跳过若干推理步骤以提升采样速度。  
e) 加速模型的五条轨迹。  
f) 加速模型生成样本的直方图。  

将该修正项加入算法 18.2 的最终更新步骤，得到：  
$$
\mathbf{z}_{t-1} = \hat{\mathbf{z}}_{t-1} + \sigma_t^2 \frac{\partial \log P_r(c|\mathbf{z}_t)}{\partial \mathbf{z}_t} + \sigma_t \boldsymbol{\epsilon}. \tag{18.41}
$$  
新增项依赖于一个基于潜在变量 $\mathbf{z}_t$ 的分类器 $P_r(c|\mathbf{z}_t)$ 的梯度。该分类器将 U-Net 下采样分支提取的特征映射至类别 $c$。与 U-Net 类似，该分类器通常在所有时间步上共享，并以时间步 $t$ 作为输入。由此，从 $\mathbf{z}_t$ 到 $\mathbf{z}_{t-1}$ 的更新过程将使类别 $c$ 更有可能出现。  

无分类器引导（classifier-free guidance）则避免单独学习分类器 $P_r(c|\mathbf{z}_t)$，而是将类别信息直接融入主模型 $g[\mathbf{z}_t, \boldsymbol{\phi}_t, c]$ 中。实践中，这通常体现为：将基于类别 $c$ 的嵌入向量以类似于时间步嵌入的方式，逐层注入 U-Net（参见图 18.9）。该模型通过随机丢弃类别信息，在条件目标与无条件目标上进行联合训练。  

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社。

18.6 实现　367  
图18.11　基于文本提示的级联条件生成。a）采用由一系列U-Net构成的扩散模型生成一幅 $64 \times 64$ 的图像；b）该生成过程以语言模型计算所得的句子嵌入（sentence embedding）为条件；c）进一步生成更高分辨率的 $256 \times 256$ 图像，其生成过程同时以低分辨率图像和文本编码为条件；d）该过程重复进行，最终生成 $1024 \times 1024$ 的图像；e）最终图像序列。改编自 Saharia 等（2022b）。  

在训练过程中，该模型可学习到对条件信息的灵活权衡。因此，在测试阶段，它既能生成无条件样本，也能生成条件样本，或二者任意加权组合的样本。这带来了一个出人意料的优势（见习题18.12）：若条件信息的权重设置过高，模型倾向于生成质量极高但略显刻板（stereotypical）的样本——这一现象在某种程度上类似于GAN中截断技巧（truncation trick）的作用（参见图15.10）。  

18.6.4　提升生成质量  
与其他生成模型类似，要获得最高质量的生成结果，需在基础模型之上综合应用多种技巧与扩展方法。首先，研究发现，除估计反向过程的均值外，同时估计其方差 $\sigma_t^2$（即各时间步的噪声尺度）亦有助于提升性能。  

草稿：勘误请发送至 udlbookmail@gmail.com。

368 18 扩散模型  
（图18.7中所示的布朗正态分布）。这一改进在步数较少的采样过程中尤为显著。第二，可在前向过程中调整噪声调度（noise schedule），使每个时间步 $ t $ 对应的噪声方差 $ \beta_t $ 动态变化，这同样有助于提升生成质量。第三，为生成高分辨率图像，采用级联式扩散模型（cascade of diffusion models）：首个模型生成低分辨率图像（可能以类别信息为条件）；后续各扩散模型则依次生成更高分辨率的图像。这些后续模型以低分辨率图像为条件——具体做法是将其缩放至对应尺寸，并拼接至构成U-Net的各层特征图上，同时亦可融合其他类别信息（见图18.11）。

综合运用上述全部技术，即可生成极高保真度的图像。图18.12展示了基于ImageNet类别标签为条件所生成的图像示例。尤为令人印象深刻的是，同一模型能够学会生成如此多样化的类别图像。图18.13则展示了另一类模型所生成的图像：该模型经训练后可接受由BERT等语言模型编码的文本描述作为条件输入，且该文本嵌入方式与时间步嵌入方式完全相同（参见图18.9和图18.11）。由此生成的图像不仅高度逼真，且与文本描述严格一致。由于扩散模型本质上具有随机性，因此可针对同一段文本描述生成多张不同的图像。

18.7 小结  
扩散模型通过一系列潜在变量对数据样本进行映射，其核心机制是在每一步中将当前表征与随机噪声反复混合。经过足够多的步骤后，该表征将退化为不可区分于白噪声的状态。由于每一步的扰动幅度极小，反向去噪过程在每个时间步均可近似为一个正态分布，并由深度学习模型进行预测。其损失函数基于证据下界（Evidence Lower Bound, ELBO），最终可简化为一种简洁的最小二乘形式。

在图像生成任务中，每个去噪步骤均通过U-Net实现，因此相较于其他生成模型，采样速度较慢。为提升生成效率，可将扩散模型重构为确定性形式，此时减少采样步数仍能取得良好效果。此外，已有多种方法被提出用于将生成过程以类别标签、图像或文本信息为条件。将这些方法有机结合，即可实现极具表现力的文本到图像合成（text-to-image synthesis）。

注释  
去噪扩散概率模型（Denoising Diffusion Probabilistic Models）由Sohl-Dickstein等人（2015）首次提出；早期与之相关的工作则基于分数匹配（score-matching），由Song与Ermon（2019）完成。Ho等人（2020）生成了可与GAN相媲美的图像样本，从而激发了学界对该领域的广泛关注。本章的大部分内容阐述——包括原始公式推导及重参数化处理——均源自该论文。Dhariwal与Nichol（2021）进一步提升了生成图像的质量，并首次通过Fréchet Inception Distance（FID）指标在定量层面证实：扩散模型生成的图像质量已超越GAN模型。  
本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社。

注释 369  
图 18.12 利用分类器引导实现的条件生成。图像样本分别以不同 ImageNet 类别为条件生成。同一模型可生成高质量、类别差异显著的图像样本。改编自 Dhariwal & Nichol（2021）。  
图 18.13 利用文本提示实现的条件生成。基于级联式生成框架合成的图像，其生成过程以由大型语言模型编码的文本提示为条件。该随机模型能够生成多种与提示相容的不同图像；模型具备物体计数能力，并能将文本融入图像之中。改编自 Saharia 等（2022b）。  
草稿：请将勘误发送至 udlbookmail@gmail.com。

370 18 扩散模型  
在图像生成领域，条件图像合成的当前最优性能由 Karras 等人（2022）实现。关于去噪扩散模型的综述可参见 Croitoru 等人（2022）、Cao 等人（2022）、Luo（2022）以及 Yang 等人（2022）。

**图像应用**：扩散模型的应用包括文本到图像生成（Nichol 等，2022；Ramesh 等，2022；Saharia 等，2022b）、图像到图像任务（如着色、修复、扩图与复原（Saharia 等，2022a））、超分辨率重建（Saharia 等，2022c）、图像编辑（Hertz 等，2022；Meng 等，2021）、对抗扰动去除（Nie 等，2022）、语义分割（Baranchuk 等，2022），以及医学影像（Song 等，2021b；Chung & Ye，2022；Chung 等，2022；Peng 等，2022；Xie & Li，2022；Luo 等，2022），其中扩散模型有时被用作先验模型。

**其他数据类型**：扩散模型亦已拓展至视频数据（Ho 等，2022b；Harvey 等，2022；Yang 等，2022；Höppe 等，2022；Voleti 等，2022），用于生成、过去/未来帧预测及帧间插值。它们还被用于三维形状生成（Zhou 等，2021；Luo & Hu，2021）；近期更提出一种仅借助二维文本到图像扩散模型即可生成三维模型的新技术（Poole 等，2023）。Austin 等（2021）与 Hoogeboom 等（2021）研究了面向离散数据的扩散模型；Kong 等（2021）与 Chen 等（2021d）则将扩散模型应用于音频数据。

**去噪之外的替代方案**：本章所述扩散模型通过向数据中逐步添加噪声，并构建模型以渐进式地去除噪声。然而，使用噪声退化图像并非必要。Rissanen 等（2022）设计了一种逐级模糊图像的方法；Bansal 等（2022）进一步证明，相同的思想可推广至一大类确定性退化操作，包括掩码（masking）、形变（morphing）、模糊（blurring）和像素化（pixelating）等。

**与其他生成模型的比较**：扩散模型所合成图像的质量高于其他生成模型，且训练过程简单。可将其视为分层变分自编码器（hierarchical VAE）（Vahdat & Kautz，2020；Sønderby 等，2016b）的一种特例——其编码器固定不变，且隐空间维度与原始数据一致。扩散模型本质上是概率模型，但其基本形式仅能计算数据点对数似然（log-likelihood）的一个下界。不过，Kingma 等（2021）指出，该下界在测试数据上的表现优于归一化流（normalizing flows）与自回归模型（autoregressive models）所给出的精确对数似然值。扩散模型的对数似然可通过转化为常微分方程（ODE）进行计算（Song 等，2021c），或通过以扩散准则训练连续归一化流模型（continuous normalizing flow model）来获得（Lipman 等，2022）。扩散模型的主要缺点在于采样速度慢，且其隐空间缺乏语义可解释性。

**质量提升方法**：已有多种技术被提出以提升生成图像质量。其中包括第 18.5 节所述网络的重参数化（reparameterization），以及对损失函数中后续各项赋予相等权重（Ho 等，2020）。Choi 等（2022）随后系统研究了损失函数中各项的不同加权策略。

Kingma 等（2021）通过学习去噪过程中的权重参数 $\beta_t$ 来提升模型在测试集上的对数似然。相反，Nichol & Dhariwal（2021）则通过在每个时间步额外学习去噪估计的方差 $\sigma^2_t$（除均值外），从而改善模型性能。Bao 等（2022）进一步展示了如何在模型训练完成后学习这些方差参数。

Ho 等（2022a）提出了级联（cascaded）方法以生成极高分辨率图像（图 18.11）。为防止低分辨率图像中的伪影被传递至高分辨率结果中，他们引入了噪声条件增强（noise conditioning augmentation）：即在每次训练迭代中，向低分辨率条件图像添加噪声。此举降低了训练过程对低分辨率图像细节精度的依赖。该策略在推理阶段同样适用，此时通过遍历不同噪声水平并选择最优值来确定最佳噪声强度。

本作品受知识共享署名-非商业-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社。

注释 371  
**提升采样速度**：扩散模型的主要缺陷之一是训练和采样耗时较长。Stable Diffusion（Rombach 等，2022）采用一个传统自编码器将原始数据投影至一个更小的潜在空间，随后在该低维空间中执行扩散过程。该方法的优势在于：一方面降低了扩散过程所需处理的训练数据维度；另一方面使扩散模型得以扩展应用于其他类型的数据（如文本、图结构等）。Vahdat 等（2021）亦采用了类似思路。  

Song 等（2021a）指出，存在一大类扩散过程均与标准训练目标兼容。其中多数过程为非马尔可夫型（即当前扩散步不仅依赖于前一步的结果）。这类模型之一即为**去噪扩散隐式模型**（Denoising Diffusion Implicit Model, DDIM），其更新过程完全确定性（无随机性）（见图 18.10c）。该模型允许采用更大的步长（见图 18.10e）进行采样，而不会引入显著误差。本质上，它将原模型转化为一个常微分方程（ODE），其轨迹曲率较低，因而可直接应用高效的常微分方程数值求解方法。  

Song 等（2021c）提出将底层的随机微分方程（SDE）转换为一个**概率流 ODE**（Probability Flow ODE），该 ODE 具有与原始 SDE 过程完全相同的边缘分布。Vahdat 等（2021）、Xiao 等（2022b）以及 Karras 等（2022）均利用常微分方程求解技术加速合成过程。Karras 等（2022）系统评估了不同时间离散化策略对采样质量的影响，并筛选出性能最优的采样时间表（sampler schedule）。上述及其他改进措施共同显著减少了合成过程中所需的迭代步数。  

采样缓慢的根本原因在于：需执行大量细粒度的扩散步骤，以确保后验分布 $q(z_{t-1} \mid z_t)$ 接近高斯分布（见图 18.5），从而保证解码器中所假设的高斯先验成立。若在每一步去噪过程中采用能刻画更复杂分布的模型，则可天然减少所需扩散步数。为此，Xiao 等（2022b）探索了使用条件生成对抗网络（Conditional GAN）建模，Gao 等（2021）则尝试了条件能量模型（Conditional Energy-Based Models）。尽管这些模型无法精确描述原始数据分布，但足以准确预测（远为简单得多的）逆向扩散步骤。  

Salimans & Ho（2022）通过知识蒸馏（distillation）技术，将相邻的多个去噪步骤合并为单步操作，从而加速合成。Dockhorn 等（2022）则在扩散过程中引入动量（momentum）机制，使采样轨迹更为平滑，进而支持更粗粒度的步进采样。  

**条件生成**：Dhariwal & Nichol（2021）提出了**分类器引导**（Classifier Guidance）方法——训练一个独立分类器，在每个去噪步骤中识别当前正在合成对象所属的类别，并利用该信息对去噪更新施加偏差，使其朝向指定类别收敛。该方法效果良好，但额外训练分类器开销较大。**无分类器引导**（Classifier-Free Guidance）（Ho & Salimans，2022）则通过一种类似 Dropout 的机制，在部分训练样本中随机丢弃类别标签，从而实现条件去噪模型与无条件去噪模型的联合训练。该技术可灵活调节条件分量与无条件分量的相对权重：当条件分量权重过高时，模型倾向于生成更具典型性与真实感的样本。  

对图像进行条件控制的标准做法是：将（缩放后的）条件图像沿 U-Net 的不同层级进行拼接（concatenation）。例如，该方法已被用于超分辨率任务中的级联生成流程（Ho 等，2022a）。Choi 等（2021）则提出一种在无条件扩散模型中实现图像条件控制的新方法：通过匹配潜在变量，使模型隐状态与条件图像的潜在表示对齐。对文本进行条件控制的标准做法是：先将文本嵌入（text embedding）经线性变换映射至与 U-Net 层特征尺寸一致，再以与时间嵌入（time embedding）相同的方式将其逐元素加至对应层特征表示之上（见图 18.9）。  

现有扩散模型还可借助一种称为**控制网络**（Control Network）（Zhang & Agrawala，2023）的神经网络架构，针对边缘图（edge maps）、关节点位置（joint positions）、语义分割图（segmentation）、深度图（depth maps）等各类条件信号进行微调适配。  

**文本到图像生成**：在扩散模型兴起之前，最先进的文本到图像系统主要基于 Transformer 架构（例如 Ramesh 等，2021）。GLIDE（Nichol 等，2022）与 DALL·E 2（Ramesh 等，2022）均以 CLIP 模型（Radford 等，2021）生成的嵌入作为条件输入。  

草稿：勘误请发送至 udlbookmail@gmail.com。

372 18 扩散模型  
该模型可为文本与图像数据生成联合嵌入表示。Imagen（Saharia 等，2022b）表明，来自大规模语言模型的文本嵌入可进一步提升生成效果（参见图 18.13）。同一批作者还提出了一个评测基准 DrawBench，旨在评估模型在色彩渲染、物体数量、空间关系及其他语义特征方面的建模能力。Feng 等（2022）则开发了一个面向中文的文本到图像生成模型。  

与其他模型的关联：本章将扩散模型描述为一种分层变分自编码器（hierarchical variational autoencoder），因为该视角与本书其他章节内容联系最为紧密。然而，扩散模型亦与随机微分方程（可参考图 18.5 中的采样路径）以及得分匹配（score matching）（Song & Ermon，2019，2020）密切相关。Song 等（2021c）提出了一种基于随机微分方程的统一框架，同时涵盖去噪解释与得分匹配解释。此外，扩散模型与标准化流（normalizing flows）亦存在紧密联系（Zhang & Chen，2021）。Yang 等（2022）综述了扩散模型与其他生成式建模方法之间的关系。  

习题  
习题 18.1 证明：若 $\mathrm{Cov}[x_{t-1}] = I$，且采用如下更新规则：  
$$
x_t = \sqrt{1 - \beta_t} \cdot x_{t-1} + \sqrt{\beta_t} \cdot \epsilon_t, \tag{18.42}
$$  
则 $\mathrm{Cov}[x_t] = I$，即方差保持不变。  

习题 18.2 考虑变量：  
$$
z = a \cdot \epsilon_1 + b \cdot \epsilon_2, \tag{18.43}
$$  
其中 $\epsilon_1$ 与 $\epsilon_2$ 均独立服从均值为 0、方差为 1 的标准正态分布。证明：  
$$
\mathbb{E}[z] = 0, \quad \mathrm{Var}[z] = a^2 + b^2, \tag{18.44}
$$  
因此，等价地可令 $z = \sqrt{a^2 + b^2} \cdot \epsilon$，其中 $\epsilon$ 同样服从标准正态分布。  

习题 18.3 将式 (18.5) 所示过程继续展开，证明：  
$$
z_3 = \sqrt{(1-\beta_3)(1-\beta_2)(1-\beta_1)} \cdot x + \sqrt{1-(1-\beta_3)(1-\beta_2)(1-\beta_1)} \cdot \epsilon', \tag{18.45}
$$  
其中 $\epsilon'$ 服从标准正态分布。  

习题 18.4∗ 证明如下关系式：  
$$
\mathrm{Norm}_w[Aw, B] \propto \mathrm{Norm}\!\left( (A^\top B^{-1} A)^{-1} A^\top B^{-1} v,\, (A^\top B^{-1} A)^{-1} \right). \tag{18.46}
$$  

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）MIT 出版社。

注释 373  
习题 18.5∗ 证明如下关系式：  
$$
\mathrm{Norm}[a,A] \, \mathrm{Norm}[b,B] \propto \mathrm{Norm}\Bigl( (A^{-1}+B^{-1})^{-1}(A^{-1}a + B^{-1}b),\; (A^{-1}+B^{-1})^{-1} \Bigr).
\tag{18.47}
$$  

习题 18.6∗ 推导公式 (18.15)。  

习题 18.7∗ 由公式 (18.25) 的第二行推导出其第三行。  

习题 18.8∗ 在 $D$ 维空间中，均值分别为 $a$ 和 $b$、协方差矩阵分别为 $A$ 和 $B$ 的两个正态分布之间的 KL 散度为：  
$$
D_{\mathrm{KL}}\Bigl[ \mathrm{Norm}[a,A] \,\bigm|\!\bigm|\, \mathrm{Norm}[b,B] \Bigr] = \frac{1}{2}\left( \mathrm{tr}\bigl[ B^{-1}A \bigr] - D + (a-b)^\top B^{-1}(a-b) + \log\!\left[ \frac{|B|}{|A|} \right] \right).
\tag{18.48}
$$  
将公式 (18.27) 中的定义代入该表达式，并证明其中唯一依赖于参数 $\phi$ 的项即为公式 (18.28) 中的第一项。  

习题 18.9∗ 若 $\alpha_t = \prod_{s=1}^{t} (1-\beta_s)$，则证明：  
$$
\frac{\alpha_t}{\alpha_{t-1}} = 1 - \beta_t.
\tag{18.49}
$$  

习题 18.10∗ 若 $\alpha_t = \prod_{s=1}^{t} (1-\beta_s)$，则证明：  
$$
\frac{(1-\alpha_{t-1})(1-\beta_t) + \beta_t}{(1-\alpha_t)\sqrt{1-\beta_t}} = \frac{1}{\sqrt{1-\beta_t}}.
\tag{18.50}
$$  

习题 18.11∗ 证明公式 (18.38)。  

习题 18.12 无分类器引导（classifier-free guidance）使我们能够生成给定类别下更具刻板印象（stereotyped）、更“典型”（canonical）的图像。在介绍 Transformer 解码器、生成对抗网络（GAN）以及 GLOW 算法时，我们也曾讨论过若干用于减少输出变化性、从而生成更具刻板印象输出的方法。这些方法分别是什么？你是否认为，以这种方式限制生成模型的输出是不可避免的？  

草稿：请将勘误发送至 udlbookmail@gmail.com。