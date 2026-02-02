# 第5章

*第70–73页*

---

第5章  
损失函数  

前三章分别介绍了线性回归、浅层神经网络和深度神经网络。这些模型均属于一类将输入映射到输出的函数族，其中具体的函数由模型参数 $\phi$ 决定。在训练这些模型时，我们的目标是寻找一组参数 $\phi$，使其对当前任务而言能实现从输入到输出的最优映射。本章即定义何为“最优映射”。

该定义依赖于一个训练数据集 $\{x_i, y_i\}$，即若干输入/输出配对样本。**损失函数**（loss function）或**代价函数**（cost function）$L[\phi]$ 返回一个标量值，用于量化模型预测 $f[x_i, \phi]$ 与其对应真实标签（ground-truth output）$y_i$ 之间的偏差程度。在训练过程中，我们寻求使损失最小化的参数 $\phi$，从而令模型对训练输入的映射尽可能贴近其真实输出。我们在第2章中已见过一种损失函数示例：**最小二乘损失函数**（least squares loss function），它适用于目标变量为实数 $y \in \mathbb{R}$ 的单变量回归问题，其计算方式为模型预测 $f[x_i, \phi]$ 与真实值 $y_i$ 之间偏差的平方和。

附录A  
集合  

本章提供一个统一框架：一方面为实值输出场景下选用最小二乘准则提供理论依据；另一方面支持构建适用于其他预测类型（如分类任务）的损失函数。我们依次讨论以下情形：**二分类**（binary classification），其中预测结果 $y \in \{0,1\}$ 属于两个互斥类别之一；**多类分类**（multiclass classification），其中预测结果 $y \in \{1,2,\dots,K\}$ 属于 $K$ 个类别之一；以及更复杂的预测任务。在接下来的两章中，我们将聚焦于**模型训练**（model training）——其核心目标正是求解使上述损失函数最小化的参数值。

5.1 最大似然估计  

本节提出一种构造损失函数的系统性方法。考虑一个以参数 $\phi$ 为变量、由输入 $x$ 计算输出的模型 $f[x,\phi]$。此前，我们默认该模型直接输出预测值 $y$；而此处我们将视角转向概率建模：将模型视为计算一个**条件概率分布** $\Pr(y|x)$，即在给定输入 $x$ 的前提下，对所有可能输出 $y$ 所赋予的概率分布（参见附录C.1.3：条件概率）。损失函数的设计目标，是促使每个训练样本的真实输出 $y_i$ 在由对应输入 $x_i$ 所导出的条件分布 $\Pr(y_i|x_i)$ 中具有较高的概率（见图5.1）。

本作品采用知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。©麻省理工学院出版社（MIT Press）。

5.1 最大似然估计　57  
图5.1　对输出变量建模的概率分布。  
a) 回归任务：目标是基于训练数据 $\{x_i, y_i\}$（橙色点）从输入 $x$ 预测实值输出 $y$。对每个输入值 $x$，机器学习模型预测一个关于输出 $y \in \mathbb{R}$ 的条件概率分布 $\mathrm{Pr}(y \mid x)$（青色曲线分别表示 $x = 2.0$ 和 $x = 7.0$ 时对应的分布）。最小化损失函数等价于最大化训练样本输出 $y_i$ 在由对应输入 $x_i$ 所预测的分布下的概率。  
b) 在分类任务中，需预测离散类别 $y \in \{1,2,3,4\}$，因此采用离散概率分布；模型对每个输入 $x$ 均预测一个在 $y$ 的四个可能取值上的不同直方图（即概率质量函数）。  
c) 预测计数型输出 $y \in \{0,1,2,\ldots\}$，以及  
d) 预测方向型输出 $y \in (-\pi, \pi]$ 时，我们分别采用定义在正整数集和圆形域（circular domain）上的概率分布。  

草稿：如有勘误，请发送邮件至 udlbookmail@gmail.com。

58 5 损失函数  
5.1.1 输出分布的计算  
这种视角的转变引出了一个关键问题：模型 $ f[\mathbf{x}, \boldsymbol{\phi}] $ 究竟应如何被适配，以输出一个概率分布？解决方案十分简洁。首先，我们在输出域 $ y $ 上选定一个参数化分布 $ \mathrm{Pr}(y \mid \boldsymbol{\theta}) $；随后，利用神经网络计算该分布的一个或多个参数 $ \boldsymbol{\theta} $。

例如，假设预测域为实数集，即 $ y \in \mathbb{R} $。此时可选用定义在 $ \mathbb{R} $ 上的单变量正态分布。该分布由均值 $ \mu $ 和方差 $ \sigma^2 $ 定义，故参数向量为 $ \boldsymbol{\theta} = \{\mu, \sigma^2\} $。机器学习模型可直接预测均值 $ \mu $，而方差 $ \sigma^2 $ 则可视为一个未知常量。

5.1.2 最大似然准则  
此时，模型对每个训练输入 $ \mathbf{x}_i $ 均输出一组不同的分布参数 $ \boldsymbol{\theta}_i = f[\mathbf{x}_i, \boldsymbol{\phi}] $。每个观测到的训练输出 $ y_i $ 应在其对应分布 $ \mathrm{Pr}(y_i \mid \boldsymbol{\theta}_i) $ 下具有较高的概率。因此，我们选取模型参数 $ \boldsymbol{\phi} $，使其最大化全部 $ I $ 个训练样本的联合概率：

$$
\hat{\boldsymbol{\phi}} = \underset{\boldsymbol{\phi}}{\operatorname{argmax}} \left[ \prod_{i=1}^{I} \mathrm{Pr}(y_i \mid \mathbf{x}_i) \right]
$$

$$
= \underset{\boldsymbol{\phi}}{\operatorname{argmax}} \left[ \prod_{i=1}^{I} \mathrm{Pr}(y_i \mid \boldsymbol{\theta}_i) \right]
$$

$$
= \underset{\boldsymbol{\phi}}{\operatorname{argmax}} \left[ \prod_{i=1}^{I} \mathrm{Pr}(y_i \mid f[\mathbf{x}_i, \boldsymbol{\phi}]) \right]. \tag{5.1}
$$

该联合概率项称为参数的**似然（likelihood）**，因此式 (5.1) 被称为**最大似然准则（maximum likelihood criterion）**¹。

此处我们隐含地作出了两项假设。第一，假设数据是**同分布的（identically distributed）**——即各数据点对应的输出 $ y_i $ 的概率分布形式完全相同；第二，假设给定输入 $ \mathbf{x}_i $ 后，输出 $ y_i $ 的条件分布 $ \mathrm{Pr}(y_i \mid \mathbf{x}_i) $ 相互**独立**，从而训练数据的总似然可分解为：

$$
\mathrm{Pr}(y_1, y_2, \dots, y_I \mid \mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_I) = \prod_{i=1}^{I} \mathrm{Pr}(y_i \mid \mathbf{x}_i). \tag{5.2}
$$

换言之，我们假设训练数据满足**独立同分布（independent and identically distributed, i.i.d.）** 条件。

¹ 条件概率 $ \mathrm{Pr}(z \mid \boldsymbol{\psi}) $ 可从两种角度理解：若将其视为关于 $ z $ 的函数，则它是一个概率分布，其对 $ z $ 的求和（或积分）为 1；若将其视为关于 $ \boldsymbol{\psi} $ 的函数，则称之为**似然函数（likelihood）**，其一般并不满足归一化（即对 $ \boldsymbol{\psi} $ 的求和或积分不一定为 1）。

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议保护。（©）麻省理工学院出版社。

5.1 最大似然估计　59  
图5.2　对数变换。  
a）对数函数是单调递增的：若 $ z > z' $，则 $ \log[z] > \log[z'] $。由此可知，任意函数 $ g[z] $ 的最大值点与其对数变换 $ \log[g[z]] $ 的最大值点位置完全相同。  
b）函数 $ g[z] $。  
c）该函数的对数形式 $ \log[g[z]] $。在 $ g[z] $ 上所有具有正斜率的位置，经对数变换后仍保持正斜率；所有具有负斜率的位置，变换后仍保持负斜率。最大值点的位置保持不变。  

5.1.3　最大化对数似然  
最大似然准则（式（5.1））在实际应用中并不便利。每个因子 $ \Pr(y_i \,|\, f[\mathbf{x}_i,\boldsymbol{\phi}]) $ 的取值可能都很小，因此大量此类因子的乘积可能极其微小，以致难以用有限精度的浮点算术准确表示。幸运的是，我们可以等价地最大化似然函数的对数：

$$
\hat{\boldsymbol{\phi}} = \underset{\boldsymbol{\phi}}{\operatorname{argmax}} \prod_{i=1}^I \Pr(y_i \,|\, f[\mathbf{x}_i,\boldsymbol{\phi}])
$$

$$
= \underset{\boldsymbol{\phi}}{\operatorname{argmax}} \log \left( \prod_{i=1}^I \Pr(y_i \,|\, f[\mathbf{x}_i,\boldsymbol{\phi}]) \right)
$$

$$
= \underset{\boldsymbol{\phi}}{\operatorname{argmax}} \sum_{i=1}^I \log \big[ \Pr(y_i \,|\, f[\mathbf{x}_i,\boldsymbol{\phi}]) \big]. \tag{5.3}
$$

该对数似然准则与原始最大似然准则等价，原因在于对数函数是单调递增函数：若 $ z > z' $，则 $ \log[z] > \log[z'] $，反之亦然（见图5.2）。因此，当我们调整模型参数 $ \boldsymbol{\phi} $ 以提升对数似然准则时，原始的最大似然准则也同步得到提升；同理，两个准则的全局最大值点必然重合，故所得最优模型参数 $ \hat{\boldsymbol{\phi}} $ 在两种准则下完全一致。此外，对数似然准则还具备一项实用优势：它将乘积运算转化为求和运算，从而避免了有限精度表示下的数值下溢问题。

勘误请发送至：udlbookmail@gmail.com。