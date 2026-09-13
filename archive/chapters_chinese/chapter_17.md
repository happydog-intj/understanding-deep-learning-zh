# 第17章

*页码：341–362*

---

第 17 章  
变分自编码器  

生成对抗网络（GAN）学习一种生成样本的机制，使得这些样本无法与训练样本集 $\{x_i\}$ 相区分。相比之下，与归一化流（normalizing flows）类似，变分自编码器（Variational Autoencoders, VAE）属于概率生成模型；其目标是学习数据上的概率分布 $P_r(\mathbf{x})$（参见图 14.2）。训练完成后，可从此分布中采样（即生成）新样本。然而，由于 VAE 自身的结构特性，**无法精确计算新样本 $\mathbf{x}^*$ 的概率值**。人们常将 VAE 本身称作 $P_r(\mathbf{x})$ 的“模型”，但这种说法具有误导性：VAE 实质上是一种神经网络架构，其设计初衷是辅助学习 $P_r(\mathbf{x})$ 的模型；而最终用于刻画 $P_r(\mathbf{x})$ 的模型既不包含“变分”部分，也不含“自编码器”结构，更准确的描述应为**非线性潜在变量模型**（nonlinear latent variable model）。

本章首先概述潜在变量模型的一般形式，随后聚焦于非线性潜在变量模型这一特例。我们将看到，对该模型进行最大似然估计并非易事。尽管如此，我们仍可为其似然函数构造一个下界（lower bound），而 VAE 架构则借助蒙特卡洛（采样）方法对该下界进行近似。最后，本章将介绍 VAE 的若干典型应用。

17.1 潜在变量模型  

潜在变量模型采用间接方式来刻画多维变量 $\mathbf{x}$ 上的概率分布 $P_r(\mathbf{x})$。它并不直接写出 $P_r(\mathbf{x})$ 的显式表达式，而是建模数据 $\mathbf{x}$ 与一个不可观测的隐变量（或称潜在变量）$\mathbf{z}$ 的联合分布 $P_r(\mathbf{x}, \mathbf{z})$（参见附录 C.1.2）。进而，通过将该联合概率关于 $\mathbf{z}$ 边缘化（marginalization），得到 $P_r(\mathbf{x})$ 的表达式：

$$
P_r(\mathbf{x}) = \int P_r(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}. \tag{17.1}
$$

（参见附录 C.1.3）

通常，联合概率 $P_r(\mathbf{x}, \mathbf{z})$ 利用条件概率法则分解为两部分：一是以潜在变量 $\mathbf{z}$ 为条件的数据似然项 $P_r(\mathbf{x} \mid \mathbf{z})$，二是潜在变量的先验分布 $P_r(\mathbf{z})$：

328 17 变分自编码器  
\[
P_r(\mathbf{x}) = \int P_r(\mathbf{x}|\mathbf{z})\,P_r(\mathbf{z})\,\mathrm{d}\mathbf{z}. \tag{17.2}
\]  
这是一种描述 \(P_r(\mathbf{x})\) 的间接方法，但十分有用，因为即使 \(P_r(\mathbf{x}|\mathbf{z})\) 和 \(P_r(\mathbf{z})\) 具有相对简单的表达式，它们的组合仍可定义出复杂的分布 \(P_r(\mathbf{x})\)。

17.1.1 示例：高斯混合模型  
在一维高斯混合模型（图 17.1a）中，隐变量 \(\mathbf{z}\) 是离散的，其先验分布 \(P_r(\mathbf{z})\) 为分类分布（参见图 5.9），对 \(\mathbf{z}\) 的每一个可能取值 \(n\)，对应一个概率 \(\lambda_n\)。  
给定隐变量 \(\mathbf{z} = n\) 时，观测数据 \(\mathbf{x}\) 的似然 \(P_r(\mathbf{x}|\mathbf{z} = n)\) 服从均值为 \(\mu_n\)、方差为 \(\sigma_n^2\) 的正态分布：  
\[
P_r(\mathbf{z} = n) = \lambda_n
\]  
\[
P_r(\mathbf{x}|\mathbf{z} = n) = \mathrm{Norm}_\mathbf{x}\!\left[\mu_n,\,\sigma_n^2\right]. \tag{17.3}
\]  

如公式 (17.2) 所示，\(P_r(\mathbf{x})\) 通过对隐变量 \(\mathbf{z}\) 边缘化得到（图 17.1b）。此处隐变量是离散的，因此我们对其所有可能取值求和以完成边缘化：  
\[
P_r(\mathbf{x}) = \sum_{n=1}^{N} P_r(\mathbf{x},\,\mathbf{z} = n)
\]  
\[
= \sum_{n=1}^{N} P_r(\mathbf{x}|\mathbf{z} = n)\cdot P_r(\mathbf{z} = n)
\]  
\[
= \sum_{n=1}^{N} \lambda_n \cdot \mathrm{Norm}_\mathbf{x}\!\left[\mu_n,\,\sigma_n^2\right]. \tag{17.4}
\]  
借助似然与先验的简单表达式，我们即可刻画出一种复杂的多峰概率分布。

17.2 非线性隐变量模型  
在非线性隐变量模型中，观测数据 \(\mathbf{x}\) 和隐变量 \(\mathbf{z}\) 均为连续型且多维。其先验分布 \(P_r(\mathbf{z})\) 是标准多元正态分布（参见附录 C.3.2）：  
\[
P_r(\mathbf{z}) = \mathrm{Norm}_\mathbf{z}\!\left[\mathbf{0},\,\mathbf{I}\right]. \tag{17.5}
\]  
似然 \(P_r(\mathbf{x}|\mathbf{z},\boldsymbol{\phi})\) 同样服从正态分布；其均值是隐变量 \(\mathbf{z}\) 的一个非线性函数 \(f[\mathbf{z},\boldsymbol{\phi}]\)，协方差矩阵为球形结构 \(\sigma^2\mathbf{I}\)：  

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（C）麻省理工学院出版社。

17.2 非线性潜在变量模型　329  
图17.1　高斯混合模型（MoG）。a）MoG 将一个复杂的概率分布（青色曲线）表示为若干高斯成分（虚线曲线）的加权和。b）该加权和即为观测连续数据 $x$ 与离散潜在变量 $z$ 的联合密度 $\Pr(x,z)$ 关于 $z$ 的边缘化结果。  
$$
\Pr(x|z,\phi)=\mathrm{Norm}\left[f[z,\phi],\sigma^2\mathbf{I}\right]. \tag{17.6}
$$  
函数 $f[z,\phi]$ 由一个具有参数 $\phi$ 的深度网络描述。潜在变量 $z$ 的维度低于观测数据 $x$。模型 $f[z,\phi]$ 刻画了数据中关键的结构信息，而其余未被建模的部分则归因于噪声项 $\sigma^2\mathbf{I}$。  
**Notebook 17.1**  

数据的概率密度 $\Pr(x|\phi)$ 可通过对潜在变量 $z$ 边缘化得到：  
**潜在变量模型**  
$$
\Pr(x|\phi) = \int \Pr(x,z|\phi)\,dz  
= \int \Pr(x|z,\phi)\cdot\Pr(z)\,dz  
= \int \mathrm{Norm}\left[f[z,\phi],\sigma^2\mathbf{I}\right] \cdot \mathrm{Norm}[0,\mathbf{I}]\,dz. \tag{17.7}
$$  
该式可理解为一种无限加权和（即无限混合），其中每个成分均为球形高斯分布，其权重为 $\Pr(z)$，均值为网络输出 $f[z,\phi]$（见图17.2）。  

17.2.1　生成过程  
可通过祖先采样（ancestral sampling）（见图17.3）生成新的样本 $x^*$（参见附录 C.4.2）。我们首先从先验分布 $\Pr(z)$ 中采样得到 $z^*$，再将其输入网络 $f[z^*,\phi]$，以计算似然 $\Pr(x|z^*,\phi)$ 的均值（见公式 17.6），最后从此似然分布中采样得到 $x^*$。由于先验分布与似然分布均为正态分布，该过程实现起来十分直接。  

*草稿：如有勘误，请发送邮件至 udlbookmail@gmail.com。*

330 17 变分自编码器  
图17.2 非线性潜在变量模型。右侧的复杂二维密度 $ \Pr(\mathbf{x}) $ 是通过对联合分布 $ \Pr(\mathbf{x},\mathbf{z}) $（左侧）关于潜在变量 $ \mathbf{z} $ 进行边缘化得到的；为获得 $ \Pr(\mathbf{x}) $，需沿维度 $ \mathbf{z} $ 对三维体积分。对每个 $ \mathbf{z} $，$ \mathbf{x} $ 上的分布是一个球形高斯分布（图中展示了两个截面），其均值为 $ \mathbf{f}[\mathbf{z},\boldsymbol{\phi}] $，该均值是 $ \mathbf{z} $ 的非线性函数，且依赖于参数 $ \boldsymbol{\phi} $。分布 $ \Pr(\mathbf{x}) $ 即为这些高斯分布的加权和。  

图17.3 非线性潜在变量模型的采样生成过程。  
a) 我们从潜在变量的先验概率分布 $ \Pr(\mathbf{z}) $ 中抽取一个样本 $ \mathbf{z}^* $；  
b) 然后从条件分布 $ \Pr(\mathbf{x}\,|\,\mathbf{z}^*,\boldsymbol{\phi}) $ 中抽取一个样本 $ \mathbf{x}^* $。该条件分布是一个球形高斯分布，其均值为 $ \mathbf{z}^* $ 的非线性函数 $ \mathbf{f}[\cdot,\boldsymbol{\phi}] $，方差固定为 $ \sigma^2\mathbf{I} $；  
c) 若重复此过程多次，则可重构出密度 $ \Pr(\mathbf{x}\,|\,\boldsymbol{\phi}) $。  

本作品采用知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。© MIT出版社。

17.3 训练　331  
图17.4　詹森不等式（离散情形）。对数函数（黑色曲线）是凹函数；在该曲线上任取两点并连接成一条直线，这条直线始终位于曲线的下方。由此可知，对数函数上任意六个点所构成的凸组合（即权重为正、且权重之和为1的加权和）必落在曲线下的灰色区域内。此处，我们对各点赋予相等权重（即取算术平均），从而得到青色点。由于该点位于曲线下方，故有  
$$
\log[\mathbb{E}[y]] > \mathbb{E}[\log[y]].
$$

17.3 训练  
为训练模型，我们在训练数据集 $\{x_i\}_{i=1}^I$ 上关于模型参数最大化对数似然函数。为简化起见，我们假设似然表达式中的方差项 $\sigma^2$ 已知，并专注于学习参数 $\phi$：  
$$
\hat{\phi} = \arg\max_{\phi} \sum_{i=1}^{I} \log \Pr(x_i \mid \phi), \tag{17.8}
$$  
其中：  
$$
\Pr(x_i \mid \phi) = \int \mathrm{Norm}\big[f[z,\phi],\, \sigma^2 \mathbf{I}\big] \cdot \mathrm{Norm}[0,\, \mathbf{I}]\, dz. \tag{17.9}
$$  

遗憾的是，该积分不可解析求解：既不存在闭式表达式，也无法针对特定 $x_i$ 高效地数值计算。

17.3.1 证据下界（ELBO）  
为推进求解，我们定义对数似然的一个下界。该下界函数对于给定的 $\phi$ 值恒小于或等于对数似然，并且还依赖于另一组参数 $\theta$。最终，我们将构建一个神经网络来计算该下界并对其进行优化。为定义该下界，我们需要借助詹森不等式。

17.3.2 詹森不等式  
詹森不等式指出：若 $g[\cdot]$ 是一个凹函数，则其作用于数据 $y$ 的期望值的结果，大于或等于该函数作用于数据后再取期望的结果：  
> **附录 B.1.2　凹函数**  
> （草稿：勘误请发送至 udlbookmail@gmail.com）

332 17 变分自编码器  
图17.5 詹森不等式（连续情形）。对于一个凹函数，先对分布 $\mathrm{Pr}(y)$ 计算其期望值、再将该期望值代入函数中，所得结果大于或等于先将变量 $y$ 经函数变换、再对新变量计算期望值。以对数函数为例，我们有：  
$$
\log[\mathbb{E}[y]] \geq \mathbb{E}[\log[y]].
$$  
图中左侧对应于该不等式的左侧，右侧则对应于右侧。一种理解方式是：我们正在对定义在 $y \in [0,1]$ 上的橙色分布中的点作凸组合；依据图17.4的逻辑，该凸组合必然位于曲线之下。另一种理解是：凹函数会相对压缩 $y$ 的高值部分、而保留低值部分，因此若先将 $y$ 通过该函数变换，再取期望，所得结果会更低。

$$
g[\mathbb{E}[y]] \geq \mathbb{E}\big[g[y]\big]. \tag{17.10}
$$  

本例中，该凹函数即为对数函数，故有：  
**习题17.2–17.3**  

$$
\log\big[\mathbb{E}[y]\big] \geq \mathbb{E}\big[\log[y]\big], \tag{17.11}
$$  

或将其期望表达式完全展开，可得：  

$$
\log\!\left[\int \mathrm{Pr}(y)\,y\,\mathrm{d}y\right] \geq \int \mathrm{Pr}(y)\,\log[y]\,\mathrm{d}y. \tag{17.12}
$$  

这一关系在图17.4–17.5中进行了可视化阐释。事实上，一个稍更一般的结论也成立：  

$$
\log\!\left[\int \mathrm{Pr}(y)\,h[y]\,\mathrm{d}y\right] \geq \int \mathrm{Pr}(y)\,\log[h[y]]\,\mathrm{d}y, \tag{17.13}
$$  

其中 $h[y]$ 是关于 $y$ 的任意函数。该式成立的原因在于：$h[y]$ 本身构成一个新的随机变量，具有其自身的分布；而由于我们在推导中从未对 $\mathrm{Pr}(y)$ 做出任何特定假设，因此该关系恒成立。  

本作品采用知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。© MIT出版社。

17.3 训练 333  
图 17.6 证据下界（ELBO）。目标是关于参数 $\phi$ 最大化对数似然 $\log[\Pr(\mathbf{x}|\phi)]$（黑色曲线）。ELBO 是一个处处位于对数似然下方的函数，它同时依赖于 $\phi$ 和另一组参数 $\theta$。当 $\theta$ 固定时，ELBO 成为仅关于 $\phi$ 的函数（图中两条彩色曲线分别对应不同的 $\theta$ 值）。因此，我们可通过以下任一方式提升对数似然：a) 针对新引入的参数 $\theta$ 优化 ELBO（即在不同彩色曲线之间切换）；或 b) 针对原始参数 $\phi$ 优化 ELBO（即沿当前彩色曲线移动）。

17.3.3 下界推导  
我们现在利用詹森不等式（Jensen’s inequality）推导对数似然的下界。首先，在对数似然中乘以并除以一个关于隐变量 $\mathbf{z}$ 的任意概率分布 $q(\mathbf{z})$：

$$
\log[\Pr(\mathbf{x}|\phi)] = \log \left[ \int \Pr(\mathbf{x},\mathbf{z}|\phi)\,d\mathbf{z} \right] \\
= \log \left[ \int q(\mathbf{z}) \frac{\Pr(\mathbf{x},\mathbf{z}|\phi)}{q(\mathbf{z})} \,d\mathbf{z} \right], \quad \text{(17.14)}
$$

接着，应用对数函数的詹森不等式（式 (17.12)），得到一个下界：

$$
\log \left[ \int q(\mathbf{z}) \frac{\Pr(\mathbf{x},\mathbf{z}|\phi)}{q(\mathbf{z})} \,d\mathbf{z} \right] 
\ge \int q(\mathbf{z}) \log \left[ \frac{\Pr(\mathbf{x},\mathbf{z}|\phi)}{q(\mathbf{z})} \right] d\mathbf{z}, \quad \text{(17.15)}
$$

式 (17.15) 右侧称为**证据下界**（evidence lower bound, ELBO）。该名称源于贝叶斯法则（式 (17.19)）中 $\Pr(\mathbf{x}|\phi)$ 被称为“证据”（evidence）。  

在实际应用中，分布 $q(\mathbf{z})$ 本身具有参数 $\theta$，因此 ELBO 可写作：

$$
\mathrm{ELBO}[\theta,\phi] = \int q(\mathbf{z}|\theta) \log \left[ \frac{\Pr(\mathbf{x},\mathbf{z}|\phi)}{q(\mathbf{z}|\theta)} \right] d\mathbf{z}. \quad \text{(17.16)}
$$

草稿：请将勘误发送至 udlbookmail@gmail.com。

334 17 变分自编码器  
为学习该非线性潜在变量模型，我们需将该量关于参数 $\phi$ 和 $\theta$ 同时最大化。用于计算该量的神经网络架构即为变分自编码器（VAE）。

17.4 证据下界（ELBO）的性质  
初见 ELBO 时，它显得颇为神秘；因此，我们接下来对其性质提供若干直观解释。回顾原始数据的对数似然函数，它是关于参数 $\phi$ 的函数，而我们的目标是求其最大值。对于任意固定的 $\theta$，ELBO 仍是关于参数的函数，但该函数恒位于原始似然函数下方。当我们改变 $\theta$ 时，便相应地调整了这一下界函数；具体而言，根据 $\theta$ 的选取，该下界可能趋近或远离对数似然函数。而当我们改变 $\phi$ 时，则是在该下界函数曲线上沿其移动（见图 17.6）。

17.4.1 边界的紧致性  
当对某一固定 $\phi$ 值，ELBO 与对数似然函数完全重合时，称该边界是“紧致的”（tight）。为找出使边界紧致的分布 $q(z|\theta)$，我们利用条件概率定义，对 ELBO 中对数项的分子进行因式分解（参见附录 C.1.3）：  

$$
\begin{aligned}
\mathrm{ELBO}[\theta,\phi] &= \int q(z|\theta)\log\frac{\Pr(x,z|\phi)}{q(z|\theta)}\,dz \\
&= \int q(z|\theta)\log\frac{\Pr(z|x,\phi)\Pr(x|\phi)}{q(z|\theta)}\,dz \\
&= \int q(z|\theta)\log\Pr(x|\phi)\,dz + \int q(z|\theta)\log\frac{\Pr(z|x,\phi)}{q(z|\theta)}\,dz \\
&= \log\Pr(x|\phi) + \int q(z|\theta)\log\frac{\Pr(z|x,\phi)}{q(z|\theta)}\,dz \\
&= \log\Pr(x|\phi) - D_{\mathrm{KL}}\big[q(z|\theta)\,\|\,\Pr(z|x,\phi)\big]. \tag{17.17}
\end{aligned}
$$

此处，第三行至第四行中第一个积分消失，是因为 $\log[\Pr(x|\phi)]$ 与 $z$ 无关，而概率分布 $q(z|\theta)$ 在全空间上的积分为 1（参见附录 C.5.1）。最后一行则直接应用了 Kullback–Leibler（KL）散度的定义。

该等式表明：ELBO 等于原始对数似然减去 KL 散度 $D_{\mathrm{KL}}[q(z|\theta)\,\|\,\Pr(z|x,\phi)]$。KL 散度用于度量两个分布之间的“距离”，其值恒为非负。由此可知，ELBO 确实是对 $\log[\Pr(x|\phi)]$ 的一个下界。当且仅当 $q(z|\theta) = \Pr(z|x,\phi)$ 时，KL 散度为零，此时边界达到紧致。该分布即为在观测数据 $x$ 给定条件下，关于潜在变量 $z$ 的后验分布；它揭示了哪些潜在变量取值可能生成了当前数据点（见图 17.7）。

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社。

17.4 ELBO 的性质　335  
图 17.7　隐变量上的后验分布。  
a) 后验分布 $\Pr(z \mid x^*, \phi)$ 表示能够生成观测数据点 $x^*$ 的隐变量 $z$ 的取值分布。我们通过贝叶斯法则计算该分布：  
$$
\Pr(z \mid x^*, \phi) \propto \Pr(x^* \mid z, \phi)\,\Pr(z).
$$  
b) 我们通过将 $x^*$ 与每个 $z$ 值所对应的对称高斯分布进行比较，来计算等式右侧第一项（即似然项）$\Pr(x^* \mid z, \phi)$。此处，$x^*$ 更可能由 $z_1$ 而非 $z_2$ 生成。第二项为隐变量的先验概率 $\Pr(z)$。将这两项因子相乘并归一化，使整个分布之和为 1，即可得到后验分布 $\Pr(z \mid x^*, \phi)$。  

17.4.2　ELBO 可表示为重构损失减去到先验分布的 KL 散度  
式 (17.16) 和式 (17.17) 是 ELBO 的两种不同表达形式。第三种表达方式是将该下界理解为重构误差减去变分分布到先验分布的 KL 散度：  
$$
\begin{aligned}
\mathrm{ELBO}[\theta,\phi] 
&= \int q(z \mid \theta)\,\log\frac{\Pr(x,z \mid \phi)}{q(z \mid \theta)}\,dz \\
&= \int q(z \mid \theta)\,\log\frac{\Pr(x \mid z,\phi)\,\Pr(z)}{q(z \mid \theta)}\,dz \\
&= \int q(z \mid \theta)\,\log\big[\Pr(x \mid z,\phi)\big]\,dz + \int q(z \mid \theta)\,\log\frac{\Pr(z)}{q(z \mid \theta)}\,dz \\
&= \int q(z \mid \theta)\,\log\big[\Pr(x \mid z,\phi)\big]\,dz - D_{\mathrm{KL}}\big[q(z \mid \theta) \,\|\, \Pr(z)\big], \tag{17.18}
\end{aligned}
$$  
其中，从第一行到第二行，我们将联合分布 $\Pr(x,z \mid \phi)$ 分解为条件概率与先验的乘积 $\Pr(x \mid z,\phi)\,\Pr(z)$；最后一行再次使用了 KL 散度的定义。  

习题 17.4  
草稿：如有勘误，请发送邮件至 udlbookmail@gmail.com。

336 17 变分自编码器  
在此形式化表达中，第一项衡量隐变量 $z$ 与观测数据 $x$ 之间的平均一致性 $\Pr(x|z,\phi)$，即重构精度；第二项则衡量辅助分布 $q(z|\theta)$ 与先验分布 $\Pr(z)$ 的匹配程度。该形式化表达正是变分自编码器（VAE）所采用的。

17.5 变分近似  
我们在式（17.17）中已知：当 $q(z|\theta)$ 恰好等于后验分布 $\Pr(z|x,\phi)$ 时，证据下界（ELBO）达到紧界。理论上，我们可借助贝叶斯定理计算该后验分布：  
\[
\Pr(z|x,\phi) = \frac{\Pr(x|z,\phi)\Pr(z)}{\Pr(x|\phi)}, \tag{17.19}
\]  
但在实际中，该式不可行，因为分母中的证据项 $\Pr(x|\phi)$ 无法解析计算（参见第 17.3 节）。

一种解决方案是采用**变分近似**：我们为 $q(z|\theta)$ 选取一个结构简单、参数化的分布形式，并用它来近似真实的后验分布。此处，我们选择均值为 $\mu$、协方差矩阵为对角阵 $\Sigma$ 的多元正态分布（参见附录 C.3.2）。该分布未必能完美拟合后验分布，但对某些 $\mu$ 和 $\Sigma$ 的取值，其拟合效果将优于其他取值。在训练过程中，我们将寻找“最接近”真实后验分布 $\Pr(z|x)$ 的正态分布（图 17.8）。这等价于最小化式（17.17）中的 KL 散度，并将图 17.6 中的彩色曲线整体上移。

由于使 $q(z|\theta)$ 最优的选取即为后验分布 $\Pr(z|x)$，而该后验又依赖于具体的数据样本 $x$，因此变分近似也应具有相同的数据依赖性。于是我们设定：  
\[
q(z|x,\theta) = \mathcal{N}\big[ g_\mu[x,\theta],\, g_\Sigma[x,\theta] \big], \tag{17.20}
\]  
其中 $g[x,\theta]$ 是另一个以 $\theta$ 为参数的神经网络，用于预测变分近似正态分布的均值 $\mu$ 和协方差 $\Sigma$。

17.6 变分自编码器  
最后，我们完整描述变分自编码器（VAE）。我们构建一个网络，用于计算 ELBO：  
\[
\mathrm{ELBO}[\theta,\phi] = \int q(z|x,\theta)\log \Pr(x|z,\phi)\,dz - D_{\mathrm{KL}}\big[ q(z|x,\theta) \,\|\, \Pr(z) \big], \tag{17.21}
\]  
其中分布 $q(z|x,\theta)$ 即为式（17.20）所定义的近似分布。

（参见附录 C.2）第一项仍包含一个难以解析计算的积分；但由于它是关于 $q(z|x,\theta)$ 的期望，我们可通过采样对其进行近似。对于任意函数 $a[\cdot]$，均有：  
\[
\mathbb{E}_{z \sim q(z|x,\theta)}\big[a[z]\big] \approx \frac{1}{L}\sum_{l=1}^L a[z^{(l)}], \quad \text{其中 } z^{(l)} \sim q(z|x,\theta).
\]  

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（C）麻省理工学院出版社。

17.6 变分自编码器　337  
图17.8　变分近似。后验分布 $\Pr(z\,|\,x^*,\phi)$ 无法以闭式解形式计算。变分近似方法选取一族分布 $q(z\,|\,x,\theta)$（此处为高斯分布），并试图在该族中找到最接近真实后验分布的成员。  
a) 有时，近似分布（青色曲线）效果良好，与真实后验分布（橙色曲线）非常接近。  
b) 然而，若后验分布呈多峰形态（如图17.7所示），则高斯近似的效果将显著变差。  

$$
\mathbb{E}_{z \sim q(z\,|\,x,\theta)}\big[a[z]\big] = \int a[z]\,q(z\,|\,x,\theta)\,dz \approx \frac{1}{N}\sum_{n=1}^{N} a[z_n^*], \tag{17.22}
$$

其中 $z_n^*$ 是从 $q(z\,|\,x,\theta)$ 中抽取的第 $n$ 个样本。该估计称为蒙特卡洛估计（Monte Carlo estimate）。  
对于粗略估计，我们可仅使用单一样本 $z^*$（从 $q(z\,|\,x,\theta)$ 中抽取）：

$$
\mathrm{ELBO}[\theta,\phi] \approx \log \Pr(x\,|\,z^*,\phi) - D_{\mathrm{KL}}\!\left[q(z\,|\,x,\theta)\,\middle\|\,\Pr(z)\right]. \tag{17.23}
$$

第二项为变分分布 $q(z\,|\,x,\theta) = \mathcal{N}_z[\mu,\Sigma]$ 与先验分布 $\Pr(z) = \mathcal{N}_z[0,I]$ 之间的 KL 散度（参见附录 C.5.4）。两个正态分布之间的 KL 散度可解析求得；特别地，当其中一个分布参数为 $\mu,\Sigma$、另一个为标准正态分布时，其表达式为：

$$
D_{\mathrm{KL}}\!\left[q(z\,|\,x,\theta)\,\middle\|\,\Pr(z)\right] = \frac{1}{2}\Big(\mathrm{Tr}[\Sigma] + \mu^\top\mu - D_z - \log \det[\Sigma]\Big). \tag{17.24}
$$

其中 $D_z$ 表示潜在空间的维度。

17.6.1　VAE 算法  
综上所述，我们的目标是构建一个模型，用于计算某一点 $x$ 的证据下界（ELBO）。随后，我们采用优化算法最大化该下界，以更新参数 $\theta$ 和 $\phi$。  

草稿：请将勘误发送至 udlbookmail@gmail.com。

338 17 变分自编码器  
图 17.9 变分自编码器。编码器 $g[x,\theta]$ 接收一个训练样本 $x$，并预测变分分布 $q(z|x,\theta)$ 的参数 $\mu$ 和 $\Sigma$。我们从此分布中采样得到隐变量 $z$，再利用解码器 $f[z,\phi]$ 预测原始数据 $x$。损失函数为负的证据下界（ELBO），其值取决于该预测的准确性，以及变分分布 $q(z|x,\theta)$ 与先验分布 $\mathrm{Pr}(z)$ 的接近程度（见公式 17.21）。

数据集，从而提升对数似然。为计算 ELBO，我们执行以下步骤：  
- 利用网络 $g[x,\theta]$ 计算当前数据点 $x$ 对应的变分后验分布 $q(z|\theta,x)$ 的均值 $\mu$ 和协方差矩阵 $\Sigma$；  
- 从此分布中抽取一个样本 $z^*$；  
- 利用公式 17.23 计算 ELBO。

对应的网络架构如图 17.9 所示。此时，我们便不难理解为何该模型被称为“变分自编码器”：称其为“变分”的，是因为它通过高斯分布近似后验分布；称其为“自编码器”的，是因为它从输入数据点 $x$ 出发，经由网络映射得到一个低维潜在向量 $z$，再利用该向量尽可能精确地重构原始数据点 $x$。在此框架下，网络 $g[x,\theta]$ 实现的数据到潜在变量的映射称为**编码器**（encoder），而网络 $f[z,\phi]$ 实现的潜在变量到数据的映射则称为**解码器**（decoder）。

变分自编码器（VAE）将 ELBO 视为关于参数 $\phi$ 和 $\theta$ 的函数。为最大化该下界，我们采用小批量（mini-batch）方式将样本送入网络，并借助随机梯度下降（SGD）、Adam 等优化算法更新这些参数。ELBO 关于各参数的梯度，仍按常规方式通过自动微分（automatic differentiation）计算。在该优化过程中，我们既在图 17.10 中沿不同颜色的曲线移动（即调整 $\theta$），也在每条曲线上沿其方向移动（即调整 $\phi$）。此过程使得参数 $\phi$ 不断更新，以在非线性潜在变量模型中赋予观测数据更高的似然。

本作品遵循知识共享署名—非商业性使用—禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

17.7 重参数化技巧　339  
图17.10　变分自编码器（VAE）在每次迭代中同时更新决定证据下界（ELBO）的两个因子：解码器参数 $\phi$ 和编码器参数 $\theta$，以最大化该下界。  

图17.11　重参数化技巧。在原始网络结构（图17.9）中，我们无法便捷地对采样步骤进行反向传播。重参数化技巧将采样操作从主计算流中移出：我们从标准正态分布 $\mathcal{N}(0, I)$ 中采样一个噪声变量 $\epsilon^*$，再将其与模型预测的均值 $\mu$ 和协方差矩阵的平方根 $\Sigma^{1/2}$ 组合，从而得到变分分布的一个样本。  

17.7 重参数化技巧  
还存在一个额外的难点：网络中包含一个采样步骤，而对该随机组件直接求导是困难的。然而，为更新位于该采样步骤之前网络层中的参数 $\theta$，必须能够完成穿过该步骤的梯度反向传播。  

幸运的是，存在一种简洁的解决方案：我们将随机性部分移至网络的一条分支中，该分支从标准正态分布 $\mathcal{N}(0, I)$ 中采样一个样本 $\epsilon^*$，然后利用如下关系式（习题17.5）：  
$$
z^* = \mu + \Sigma^{1/2} \epsilon^*, \tag{17.25}
$$  
生成目标高斯分布的一个样本。此时，我们可以像往常一样计算所有导数（参见 Notebook 17.2），因为反向传播算法无需沿该随机分支向下传递梯度。  

重参数化技巧  
这一方法被称为**重参数化技巧**（见图17.11）。  

草稿：勘误请发送至 udlbookmail@gmail.com。

340 17 变分自编码器  
17.8 应用场景  
变分自编码器（VAE）具有多种用途，包括去噪、异常检测以及数据压缩。本节将回顾其在图像数据上的若干典型应用。  

17.8.1 样本概率的近似估计  
在第17.3节中，我们指出：无法直接对变分自编码器中的样本计算精确的概率值，因为VAE将该概率定义为如下积分形式：  
\[
\Pr(\mathbf{x}) = \int \Pr(\mathbf{x}|\mathbf{z})\Pr(\mathbf{z})\,d\mathbf{z}
\]  
\[
= \mathbb{E}_{\mathbf{z}}\big[\Pr(\mathbf{x}|\mathbf{z})\big]
\]  
\[
= \mathbb{E}_{\mathbf{z}}\Big[\mathcal{N}\big[\mathbf{f}[\mathbf{z},\boldsymbol{\phi}],\sigma^2\mathbf{I}\big]\Big]. \tag{17.26}
\]  

理论上，我们可依据公式(17.22)通过从先验分布 $\Pr(\mathbf{z}) = \mathcal{N}[\mathbf{0},\mathbf{I}]$ 中采样 $N$ 个隐变量 $\mathbf{z}_n$，并计算如下平均值来近似该概率：  
\[
\Pr(\mathbf{x}) \approx \frac{1}{N}\sum_{n=1}^{N} \Pr(\mathbf{x}|\mathbf{z}_n). \tag{17.27}
\]  

然而，受“维数灾难”影响，所采样的绝大多数 $\mathbf{z}_n$ 值对应的条件概率 $\Pr(\mathbf{x}|\mathbf{z}_n)$ 都极低；为获得可靠的估计，需采样数量庞大的样本。一种更优的方法是采用**重要性采样（importance sampling）**。此时，我们从一个辅助分布 $q(\mathbf{z})$ 中采样 $\mathbf{z}_n$，计算 $\Pr(\mathbf{x}|\mathbf{z}_n)$，再按新分布下的概率密度 $q(\mathbf{z}_n)$ 对结果进行重加权：  
\[
\Pr(\mathbf{x}) = \int \Pr(\mathbf{x}|\mathbf{z})\Pr(\mathbf{z})\,d\mathbf{z}
\]  
\[
= \int \frac{\Pr(\mathbf{x}|\mathbf{z})\Pr(\mathbf{z})}{q(\mathbf{z})}\,q(\mathbf{z})\,d\mathbf{z}
\]  
\[
= \mathbb{E}_{q(\mathbf{z})}\left[ \frac{\Pr(\mathbf{x}|\mathbf{z})\Pr(\mathbf{z})}{q(\mathbf{z})} \right]
\]  
\[
\approx \frac{1}{N}\sum_{n=1}^{N} \frac{\Pr(\mathbf{x}|\mathbf{z}_n)\Pr(\mathbf{z}_n)}{q(\mathbf{z}_n)}, \tag{17.28}
\]  
其中所有 $\mathbf{z}_n$ 均从辅助分布 $q(\mathbf{z})$ 中采样得到。若 $q(\mathbf{z})$ 的支撑区域与 $\Pr(\mathbf{x}|\mathbf{z})$ 取值较高的区域高度重合，则采样将集中于对概率估计真正关键的隐空间区域，从而显著提升 $\Pr(\mathbf{x})$ 的估计效率。  

待积分的乘积项 $\Pr(\mathbf{x}|\mathbf{z})\Pr(\mathbf{z})$（即被积函数）正比于后验分布 $\Pr(\mathbf{z}|\mathbf{x})$（由贝叶斯定理可知）。因此，一个合理且自然的辅助分布 $q(\mathbf{z})$ 选择，便是由编码器所计算出的**变分后验分布** $q(\mathbf{z}|\mathbf{x})$。  

> **Notebook 17.3**：重要性采样  

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

17.8 应用　341  
图17.12　在CELEBA数据集上训练的标准变分自编码器（VAE）所生成的采样结果。每一列中，首先从潜在空间中抽取一个潜在变量 $ z^* $，再将其输入模型以预测均值 $ f[z^*, \phi] $，最后叠加独立的高斯噪声（参见图17.3）。  
a) 为最终生成的样本集合；  
b) 为预测所得的均值图像；  
c) 为球形高斯噪声向量。  
在添加噪声之前，图像显得过于平滑；而添加噪声之后，图像又显得过于嘈杂。这是典型现象；通常仅展示无噪声版本，因为该噪声被视作表征图像中未被模型捕获的部分。引自Dorta等（2018）。  
d) 当前已可通过引入分层先验、专用网络架构及精细的正则化策略，利用VAE生成高质量图像。引自Vahdat与Kautz（2020）。  

通过这种方式，我们可近似估计新样本出现的概率。当采样数量足够大时，该估计将优于证据下界（ELBO），并可用于评估模型质量——例如，通过计算测试数据的对数似然值来衡量。此外，该估计亦可作为判别准则：用于判断新样本是否属于该分布，抑或属于异常样本。  

17.8.2 生成  
VAE构建了一个概率模型，从中采样十分简便：只需从潜在变量的先验分布 $ \Pr(z) $ 中采样，将采样结果送入解码器 $ f[z, \phi] $，再依据条件分布 $ \Pr(x \mid f[z, \phi]) $ 添加相应噪声即可。  
草稿：如有勘误，请发送至 udlbookmail@gmail.com。

342 17 变分自编码器  
标准变分自编码器（vanilla VAE）通常生成质量较低（图17.12a–c）。这在一定程度上源于其过于简化的球形高斯噪声模型，另一部分原因则在于先验分布与变分后验分布均采用高斯建模。一种提升生成质量的技巧是：不从先验分布 $ p(z) $ 中采样，而是从**聚合后验分布**（aggregated posterior）中采样，即  
$$
q(z|\theta) = \frac{1}{I}\sum_{i=1}^I q(z|x_i,\theta),
$$  
该分布为所有训练样本对应后验分布的平均值，本质上是一个高斯混合分布，因而更能准确刻画潜在空间中真实隐变量的分布。

现代VAE虽可生成高质量样本（图17.12d），但必须依赖分层先验（hierarchical priors）、专用网络架构以及特定的正则化技术。扩散模型（第18章）可被视作具有分层先验的VAE，同样能生成极高保真度的样本。

17.8.3 重合成（Resynthesis）  
VAE还可用于对真实数据进行编辑修改。给定一个数据点 $ x $，可通过以下两种方式将其投影至潜在空间：（i）直接取编码器所预测分布的均值；或（ii）通过优化方法求解使后验概率 $ \Pr(z|x) $ 最大的潜在变量 $ z $；根据贝叶斯定理，该后验概率正比于 $ \Pr(x|z)\Pr(z) $。

在图17.13中，若干标注为“中性”或“微笑”的人脸图像被分别投影至潜在空间。表征“中性→微笑”这一语义变化的方向向量，通过计算这两组图像在潜在空间中各自均值之差而估计得到；同理，另一方向向量则用于表征“嘴闭合”与“嘴张开”的差异。

随后，待处理图像被投影至潜在空间，并通过沿上述方向向量加减相应位移来修改其潜在表示。为生成中间过渡图像，采用**球面线性插值**（spherical linear interpolation，简称 Slerp），而非普通线性插值。  

**习题17.6**  
在三维空间中，这相当于沿球面表面插值与直接穿过球体内部挖掘一条直线隧道之间的区别。

将输入数据先编码（并可能对其进行修改），再解码重建的过程称为**重合成**（resynthesis）。该技术同样适用于生成对抗网络（GAN）和归一化流（normalizing flows）。然而，在GAN中并不存在显式编码器，因此需借助独立的优化过程，以寻找与观测数据相对应的潜在变量。

17.8.4 解耦（Disentanglement）  
在前述重合成示例中，表征可解释语义属性的方向需借助带标签的训练数据进行估计。另一类研究工作则致力于改进潜在空间的结构特性，使其坐标轴方向天然对应现实世界中的独立语义因子。当每个维度均表征一个独立的现实世界因素时，该潜在空间即被称为**解耦的**（disentangled）。例如，在人脸图像建模任务中，我们期望潜在空间的不同维度能分别对应头部姿态、发色等彼此独立的属性。

为促进解耦性，典型方法是在损失函数中引入额外的正则化项，其设计依据通常为：（i）单个样本的后验分布 $ q(z|x,\theta) $，或（ii）聚合后验分布 $ q(z|\theta)=\frac{1}{I}\sum_{i=1}^I q(z|x_i,\theta) $：  
$$
\mathcal{L}_{\text{new}} = -\text{ELBO}[\theta,\phi] + \lambda_1 \mathbb{E}_{\Pr(x)}\left[r_1\big(q(z|x,\theta)\big)\right] + \lambda_2\, r_2\big(q(z|\theta)\big). \tag{17.29}
$$  
本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社。

17.9 小结 343  
图 17.13　重合成（Resynthesis）。左侧原始图像经编码器投影至潜在空间，其预测高斯分布的均值被选作该图像在潜在空间中的表示。网格中左中部图像为对输入图像的重构结果。其余图像则是在潜在空间中沿表征“微笑/中性”（水平方向）和“张嘴/闭嘴”（垂直方向）的语义轴进行操控后所得的重构结果。改编自 White（2016）。

此处正则化项 $ r_1[\cdot] $ 是后验分布的函数，并以权重 $ \lambda_1 $ 加权。  
$ r_2[\cdot] $ 是聚合后验分布（aggregated posterior）的函数，并以权重 $ \lambda_2 $ 加权。

例如，β-VAE 对 ELBO（式（17.18））中的第二项进行了加权提升：  
$$
\mathrm{ELBO}[\theta,\phi] \approx \log \Pr(x|z^*,\phi) - \beta \cdot D_{\mathrm{KL}}\!\left[q(z|x,\theta)\,\middle|\!\middle|\,\Pr(z)\right], \tag{17.30}
$$  
其中 $ \beta > 1 $ 控制着偏离先验分布 $ \Pr(z) $ 的惩罚力度相对于重构误差的比重。由于先验分布通常取为具有球形协方差矩阵的多元正态分布，其各维度相互独立。因此，提升该项的权重有助于促使后验分布各维度之间相关性降低。另一变体为总相关性 VAE（Total Correlation VAE），它在目标函数中额外引入一项以降低潜在空间中各变量之间的总相关性（见图 17.14），并最大化潜在变量的一个小子集与观测数据之间的互信息。

17.9 小结  
变分自编码器（VAE）是一种用于学习关于观测数据 $ x $ 的非线性潜在变量模型的架构。该模型可通过从潜在变量中采样、将采样结果送入深度网络、再叠加独立高斯噪声的方式生成新样本。

草稿：如有勘误，请发送邮件至 udlbookmail@gmail.com。

344 第17章 变分自编码器  
图17.14 总相关性变分自编码器（Total Correlation VAE）中的解耦性。该VAE模型经过修改，使其损失函数鼓励隐变量的总相关性（total correlation）最小化，从而促进隐表示的解耦。当在椅子图像数据集上进行训练时，若干隐变量维度展现出清晰的现实世界语义解释，包括：a) 旋转角度、b) 整体尺寸、c) 座椅腿结构（可旋转底座 vs. 普通固定底座）。在每幅子图中，中间一列展示模型生成的样本；从左至右移动时，对应于在隐空间中沿某一坐标轴方向减去或加上一个单位向量。改编自Chen等（2018d）。

数据点的似然函数无法以闭式形式计算，这给基于最大似然的训练带来了困难。然而，我们可以定义似然的一个下界，并最大化该下界。遗憾的是，为使该下界足够紧致（tight），我们需要计算给定观测数据下隐变量的后验概率，而该后验本身同样难以解析求解。解决方法是引入变分近似（variational approximation）：即采用一种更简单的分布（通常为高斯分布）来近似真实后验，其参数则由第二个编码器网络（即推断网络）计算得出。

若要从VAE中生成高质量样本，似乎有必要采用比标准高斯先验与高斯后验更为复杂的概率分布来建模隐空间。一种可行方案是使用分层先验（hierarchical priors），即一个隐变量生成另一个隐变量。下一章将介绍扩散模型（diffusion models），该类模型能够生成极高保真度的样本，且可被视作一类分层VAE。

注释  
变分自编码器（VAE）最初由Kingma与Welling（2014）提出。关于VAE的全面导论可参见Kingma等（2019）。

应用领域：VAE及其各类变体已被广泛应用于图像处理（Kingma & Welling, 2014；Gregor等, 2016；Gulrajani等, 2016；Akuzawa等, 2018）、语音处理（Hsu等, 2017b）、文本建模（Bowman等, 2015；Hu等, 2017；Xu等, 2020）、分子建模（Gómez-Bombarelli等,  
本作品遵循知识共享署名-非商业-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

注释 345  
2018；Sultan 等，2018）、图结构（Kipf & Welling，2016；Simonovsky & Komodakis，2018）、机器人学（Hernández 等，2018；Inoue 等，2018；Park 等，2018）、强化学习（Heess 等，2015；Van Hoof 等，2016）、3D 场景（Eslami 等，2016，2018；Rezende Jimenez 等，2016）以及手写体建模（Chung 等，2015）。

应用包括重合成与插值（White，2016；Bowman 等，2015）、协同过滤（Liang 等，2018）以及数据压缩（Gregor 等，2016）。Gómez-Bombarelli 等（2018）利用变分自编码器（VAE）构建化学结构的连续表征，并进一步针对期望性质对其进行优化。Ravanbakhsh 等（2017）则模拟天文观测数据，用于校准测量系统。

与其他模型的关系：自编码器（autoencoder）（Rumelhart 等，1985；Hinton & Salakhutdinov，2006）将数据经编码器映射至一个瓶颈层（bottleneck layer），再通过解码器将其重构。该瓶颈层与 VAE 中的潜在变量类似，但其设计动机不同：此处的目标并非学习概率分布，而是构造一种能捕捉数据本质的低维表征。自编码器亦有多种应用，例如去噪（Vincent 等，2008）和异常检测（Zong 等，2018）。

若编码器与解码器均为线性变换，则自编码器即等价于主成分分析（PCA）。因此，非线性自编码器可视为主成分分析的一种推广形式。此外，还存在若干概率化版本的 PCA：概率主成分分析（Probabilistic PCA，Tipping & Bishop，1999）在重构中加入球形高斯噪声以构建概率模型；而因子分析（factor analysis）则采用对角高斯噪声（参见 Rubin & Thayer，1982）。若将上述概率化变体中的编码器与解码器扩展为非线性形式，便回归到变分自编码器框架。

架构变体：条件变分自编码器（conditional VAE）（Sohn 等，2015）将类别信息 $c$ 同时输入编码器与解码器。其结果是潜在空间无需再编码类别信息。例如，在 MNIST 数据上以数字标签为条件时，潜在变量可能仅编码数字的方向与宽度，而非数字本身的类别。Sønderby 等（2016a）提出了阶梯式变分自编码器（ladder variational autoencoder），该模型通过一个依赖于数据的近似似然项递归地修正生成分布。

似然函数的改进：其他研究工作探索了更复杂的似然模型 $P_r(x \mid z)$。PixelVAE（Gulrajani 等，2016）在输出变量上采用自回归模型；Dorta 等（2018）不仅建模了解码器输出的均值，还对其协方差进行了建模；Lamb 等（2016）则通过引入额外的正则化项来提升重构质量——这些正则化项鼓励重构图像在某图像分类模型中间层激活空间中与原始图像保持相似。该模型有助于保留语义信息，并被用于生成图 17.13 所示结果。Larsen 等（2016）则采用对抗损失（adversarial loss）进行重构，同样显著改善了结果。

潜在空间、先验分布与后验分布：针对后验分布的变分近似，已有大量不同形式被深入研究，包括标准化流（normalizing flows）（Rezende & Mohamed，2015；Kingma 等，2016）、有向图模型（directed graphical models）（Maaløe 等，2016）、无向模型（undirected models）（Vahdat 等，2020）以及面向时序数据的递归模型（Gregor 等，2016，2019）。另有学者探讨了离散潜在空间的使用（van den Oord 等，2017；Razavi 等，2019b；Rolfe，2017；Vahdat 等，2018a,b）。例如，Razavi 等（2019b）采用向量量化（vector quantized）潜在空间，并以自回归模型（公式 12.15）对先验分布建模。该方法采样速度较慢，但可刻画极为复杂的分布。

草稿：请将勘误发送至 udlbookmail@gmail.com。

346 17 变分自编码器（VAE）  
蒋等人（Jiang et al., 2016）采用高斯混合模型（mixture of Gaussians）作为后验分布，从而支持聚类任务。这是一种层次化隐变量模型（hierarchical latent variable model），通过引入一个离散隐变量来增强后验分布的表达能力。其他研究者（Salimans et al., 2015；Ranganath et al., 2016；Maaløe et al., 2016；Vahdat & Kautz, 2020）则尝试了基于连续变量的层次化模型。这类模型与扩散模型（第18章）存在紧密联系。

与其他模型的结合：Gulrajani 等人（2016）将 VAE 与自回归模型（autoregressive model）相结合，以生成更逼真的图像。Chung 等人（2015）则将 VAE 与循环神经网络（RNN）结合，用于建模随时间变化的观测数据。

如前所述，对抗性损失（adversarial losses）已被直接用于指导似然项（likelihood term）的设计。然而，也有其他模型以不同方式融合了生成对抗网络（GAN）与 VAE 的思想。Makhzani 等人（2015）在隐空间中引入对抗性损失；其核心思想是：判别器（discriminator）应确保聚合后验分布 $ q(z) = \mathbb{E}_{x\sim p_{\text{data}}(x)}[q(z|x)] $ 无法被区分于先验分布 $ \Pr(z) $。Tolstikhin 等人（2018）将该思想推广至更广泛的先验与聚合后验之间的距离度量族。Dumoulin 等人（2017）提出了对抗式学习推断（adversarially learned inference），其使用对抗性损失来区分两组隐变量/观测数据对：其中一组隐变量采样自隐后验分布 $ q(z|x) $，另一组则采样自先验分布 $ \Pr(z) $。Larsen 等人（2016）、Brock 等人（2016）以及 Hsu 等人（2017a）也分别提出了若干 VAE 与 GAN 的混合架构。

后验坍缩（Posterior collapse）：训练过程中一个潜在问题是后验坍缩，即编码器始终预测出与先验分布一致的结果。这一现象最早由 Bowman 等人（2015）指出；可通过在训练初期逐步增大 KL 散度项（即鼓励后验分布 $ q(z|x) $ 与先验分布 $ \Pr(z) $ 尽可能接近的正则项）的权重加以缓解。此外，Razavi 等人（2019a）及 Lucas 等人（2019b, a）也提出了多种防止后验坍缩的方法；而采用离散隐空间（Van Den Oord et al., 2017）亦部分源于对此问题的考量。

重建图像模糊（Blurry reconstructions）：Zhao 等人（2017c）指出，重建图像模糊的部分原因在于高斯噪声，以及变分近似所诱导的次优后验分布。值得注意的是，目前一些最优的合成结果恰恰来自两类模型：一类是采用复杂自回归模型建模的离散隐空间（Razavi et al., 2019b），另一类则是采用层次化隐空间的模型（Vahdat & Kautz, 2020；参见图 17.12d）。图 17.12a–c 展示的是在 CELEBA 数据集（Liu et al., 2015）上训练的 VAE；图 17.12d 则展示了一个在 CELEBA HQ 数据集（Karras et al., 2018）上训练的层次化 VAE。

其他问题：Chen 等人（2017）发现，当采用更复杂的似然项（例如 PixelCNN（Van den Oord et al., 2016c））时，模型输出可能完全不再依赖于隐变量。他们将此现象称为“信息偏好问题”（information preference problem）。Zhao 等人（2017b）提出的 InfoVAE 通过额外引入一项最大化隐变量与观测变量之间互信息（mutual information）的正则项，对该问题进行了修正。

VAE 的另一问题是隐空间中可能存在“空洞”（holes），即某些区域不对应任何现实样本。Xu 等人（2020）提出了约束后验 VAE（constrained posterior VAE），通过添加一项正则化项，有助于抑制隐空间中的空旷区域，从而提升从真实样本出发的插值质量。

隐表示解耦（Disentangling latent representation）：实现隐表示解耦的方法包括 β-VAE（Higgins et al., 2017）及其他方法（例如 Kim & Mnih, 2018；Kumar et al.,  

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社。

注释 347  
图 17.15 期望最大化（EM）算法。EM 算法交替调整辅助参数 $\theta$（在不同颜色的曲线上移动）和模型参数 $\phi$（沿各颜色曲线移动），直至达到极大值点。这些调整步骤分别称为 E 步（Expectation Step）和 M 步（Maximization Step）。由于 E 步使用后验分布 $\Pr(h|x,\phi)$ 作为变分分布 $q(h|x,\theta)$，因此该下界是紧的（tight），即每次执行完 E 步后，各颜色曲线均与黑色似然曲线相切。

Chen 等（2018d）进一步对证据下界（ELBO）进行分解，揭示其中存在一项用于度量潜在变量间总相关性（total correlation）的项——即聚合后验分布（aggregate posterior）与其各边缘分布乘积之间的距离。他们据此提出“总相关性 VAE”（Total Correlation VAE），旨在最小化该项。Factor VAE（Kim & Mnih，2018）则采用另一种方法来最小化总相关性。Mathieu 等（2019）讨论了影响表征解耦（disentangling representations）的关键因素。

重参数化技巧（Reparameterization Trick）：考虑计算某个函数关于某概率分布的期望值，而该分布本身又依赖于某些参数。重参数化技巧用于计算该期望值关于这些参数的导数。本章将该技巧引入为一种可对采样过程（用以近似期望）进行求导的方法；尽管存在其他替代方案（参见习题 17.5），但重参数化技巧所得估计量通常具有较低方差。这一问题详见 Rezende 等（2014）、Kingma 等（2015）以及 Roeder 等（2017）。

下界与 EM 算法：变分自编码器（VAE）的训练基于优化证据下界（Evidence Lower Bound，常简称为 ELBO、变分下界或负变分自由能）。Hoffman & Johnson（2016）及 Lücke 等（2020）以多种等价形式重新表达了该下界，从而更清晰地揭示其性质。另有研究致力于使该下界更紧（Burda 等，2016；Li & Turner，2016；Bornschein 等，2016；Masrani 等，2019）。例如，Burda 等（2016）提出一种改进的下界，其目标函数由从近似后验分布中抽取的多个重要性加权（importance-weighted）样本构造而成。当变分分布 $q(z|\theta)$ 与真实后验 $\Pr(z|x,\phi)$ 完全匹配时，ELBO 达到紧界。这正是期望最大化（EM）算法（Dempster 等，1977）的理论基础：我们交替执行以下两步：（i）选取 $\theta$，使得 $q(z|\theta) = \Pr(z|x,\phi)$；（ii）更新 $\phi$ 以最大化下界（见图 17.15）。该方法适用于高斯混合模型等可解析计算后验分布的情形。遗憾的是，对于非线性潜在变量模型，后验分布无法闭式求解，因此该方法不可行。

习题  
习题 17.1：构建一个含 $n=5$ 个成分的一维高斯混合模型，共需多少个参数？  

草稿：勘误请发送至 udlbookmail@gmail.com。

348 17 变分自编码器  
各分量（见公式 17.4）？请说明每个参数可能取值的范围。  

**习题 17.2**  
若一个函数的二阶导数处处小于或等于零，则该函数为凹函数。试证明函数 $ g[x] = \log[x] $ 满足这一性质。  

**习题 17.3**  
对于凸函数，詹森不等式（Jensen’s inequality）的方向相反：  
$$
g\left[\mathbb{E}[y]\right] \leq \mathbb{E}\left[g[y]\right]. \tag{17.31}
$$  
若一个函数的二阶导数处处大于或等于零，则该函数为凸函数。试证明函数 $ g[x] = x^{2n} $ 对任意 $ n \in \{1,2,3,\ldots\} $ 均为凸函数。并利用该结论及詹森不等式，证明分布 $ \Pr(x) $ 的均值的平方 $ \left(\mathbb{E}[x]\right)^2 $ 必不大于其二阶矩 $ \mathbb{E}[x^2] $。  

**习题 17.4\***  
试证明：公式 17.18 所定义的证据下界（ELBO）可等价地从变分分布 $ q(z|x) $ 与真实后验分布 $ \Pr(z|x,\phi) $ 之间的 KL 散度导出：  
$$
D_{\mathrm{KL}}\left[ q(z|x) \,\middle\|\, \Pr(z|x,\phi) \right] = \int q(z|x)\,\log\frac{q(z|x)}{\Pr(z|x,\phi)}\,dz. \tag{17.32}
$$  
推导时请首先应用贝叶斯定理（公式 17.19）。  

**习题 17.5**  
重参数化技巧（reparameterization trick）用于计算函数 $ f[x] $ 关于分布 $ \Pr(x|\phi) $ 的期望的导数：  
$$
\frac{\partial}{\partial\phi}\,\mathbb{E}_{\Pr(x|\phi)}\left[f[x]\right], \tag{17.33}
$$  
其中导数是相对于该期望所依赖的分布 $ \Pr(x|\phi) $ 的参数 $ \phi $ 计算的。试证明该导数亦可表示为：  
$$
\frac{\partial}{\partial\phi}\,\mathbb{E}_{\Pr(x|\phi)}\left[f[x]\right] 
= \mathbb{E}_{\Pr(x|\phi)}\left[ f[x]\,\frac{\partial}{\partial\phi}\log\Pr(x|\phi) \right] 
\approx \frac{1}{I}\sum_{i=1}^{I} f[x_i]\,\frac{\partial}{\partial\phi}\log\Pr(x_i|\phi). \tag{17.34}
$$  
该方法称为 REINFORCE 算法，或称得分函数估计器（score function estimator）。  

**习题 17.6**  
为何在潜空间中对两点进行插值时，采用球面线性插值（spherical linear interpolation）优于常规线性插值？提示：参见图 8.13。  

**习题 17.7\***  
推导适用于一维高斯混合模型（含 $ N $ 个分量）的 EM 算法。具体需完成以下两步：（i）针对数据点 $ x $，求出关于隐变量 $ z \in \{1,2,\ldots,N\} $ 的后验分布 $ \Pr(z|x) $ 的表达式；（ii）基于所有数据点的后验分布，给出证据下界（ELBO）更新公式的表达式。推导过程中需引入拉格朗日乘子，以确保各高斯分量的权重 $ \lambda_1,\ldots,\lambda_N $ 之和为 1。  

本作品受知识共享署名—非商业性使用—禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社。