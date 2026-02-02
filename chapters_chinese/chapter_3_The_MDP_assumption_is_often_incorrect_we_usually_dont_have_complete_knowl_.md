# 第3章：马尔可夫决策过程（MDP）假设通常不成立；我们通常并不具备完全的状态知识

*页码：403–415*

19.5 策略梯度方法　389  
根本性问题在于：同一网络既通过最大化操作（`argmax`）选择目标动作，又负责更新对应的状态–动作值。双 Q 学习（Double Q-Learning）通过同时训练两个独立模型 $q_1[s_t, a_t, \pi]$ 与 $q_2[s_t, a_t, \pi]$ 来解决该问题：

$$
\begin{aligned}
q_1[s_t, a_t] &\leftarrow q_1[s_t, a_t] + \alpha \left( r[s_t, a_t] + \gamma \cdot q_2\!\left[s_{t+1}, \arg\max_a q_1[s_{t+1}, a]\right] - q_1[s_t, a_t] \right), \\
q_2[s_t, a_t] &\leftarrow q_2[s_t, a_t] + \alpha \left( r[s_t, a_t] + \gamma \cdot q_1\!\left[s_{t+1}, \arg\max_a q_2[s_{t+1}, a]\right] - q_2[s_t, a_t] \right).
\end{aligned}
\tag{19.20}
$$

此时，目标动作的选择与目标值本身被解耦，从而有助于抑制此类偏差。在实践中，新采样得到的转移元组 $\langle s, a, r, s' \rangle$ 被随机分配给两个模型之一进行更新。该方法即称为**双 Q 学习**（Double Q-Learning）。而**双深度 Q 网络**（Double Deep Q-Networks，简称 Double DQNs）则进一步采用两个深度神经网络 $q[s_t, a_t, \phi_1]$ 与 $q[s_t, a_t, \phi_2]$ 来分别估计动作值，其参数更新规则为：

$$
\begin{aligned}
\phi_1 &\leftarrow \phi_1 + \alpha \left( r[s_t, a_t] + \gamma \cdot q\!\left[s_{t+1}, \arg\max_a q[s_{t+1}, a, \phi_1], \phi_2\right] - q[s_t, a_t, \phi_1] \right) \frac{\partial q[s_t, a_t, \phi_1]}{\partial \phi_1}, \\
\phi_2 &\leftarrow \phi_2 + \alpha \left( r[s_t, a_t] + \gamma \cdot q\!\left[s_{t+1}, \arg\max_a q[s_{t+1}, a, \phi_2], \phi_1\right] - q[s_t, a_t, \phi_2] \right) \frac{\partial q[s_t, a_t, \phi_2]}{\partial \phi_2}.
\end{aligned}
\tag{19.21}
$$

19.5 策略梯度方法  
Q 学习首先估计动作值函数，再基于这些估计值更新策略。与此相反，基于策略的方法则直接学习一个随机策略 $\pi[a_t \mid s_t, \theta]$。该策略是一个以可训练参数 $\theta$ 为变量的函数，将状态 $s_t$ 映射为动作 $a_t$ 上的概率分布 $\Pr(a_t \mid s_t)$，我们可从此分布中采样得到具体动作。在马尔可夫决策过程（MDP）中，总存在一个最优的确定性策略。然而，采用随机策略仍有以下三个重要原因：  

1. 随机策略天然有利于对状态–动作空间的探索；我们无需在每个时间步都强制采取当前估计下最优的动作。  
2. 修改随机策略时，损失函数的变化是平滑的。这意味着即使奖励信号是离散的，我们仍可使用梯度下降类优化算法。这类似于在（离散）分类任务中采用最大似然估计：当模型参数调整以提升真实类别概率时，损失函数也随之平滑变化。  
3. MDP 的基本假设往往并不成立；我们通常无法获得关于系统状态的完整信息。例如，考虑一个在环境中导航的智能体，其仅能观测邻近区域（参见图 19.4）。若两个位置外观完全相同，但其周围潜在的奖励结构存在差异，则随机策略允许智能体在不同时间步尝试不同动作，直至该观测歧义得以消除。  

草稿：勘误请发送至 udlbookmail@gmail.com。

390 第19章 强化学习  
19.5.1 梯度更新公式的推导  

考虑马尔可夫决策过程（MDP）中的一条轨迹 $\tau = [s_1, a_1, s_2, a_2, \dots, s_T, a_T]$。该轨迹的概率 $\Pr(\tau \mid \theta)$ 同时依赖于状态转移函数 $\Pr(s_{t+1} \mid s_t, a_t)$ 和当前的随机策略 $\pi[a_t \mid s_t, \theta]$：  

$$
\Pr(\tau \mid \theta) = \Pr(s_1) \prod_{t=1}^T \pi[a_t \mid s_t, \theta]\, \Pr(s_{t+1} \mid s_t, a_t). \tag{19.22}
$$  

策略梯度算法的目标是在大量此类轨迹上最大化期望回报 $r[\tau]$：  

$$
\theta^* = \arg\max_\theta \mathbb{E}_\tau\big[ r[\tau] \big] = \arg\max_\theta \left[ \int \Pr(\tau \mid \theta)\, r[\tau]\, d\tau \right], \tag{19.23}
$$  

其中，回报 $r[\tau]$ 定义为该轨迹上所有即时奖励之和。  

为最大化该目标量，我们采用梯度上升更新规则：  

$$
\theta \leftarrow \theta + \alpha \cdot \frac{\partial}{\partial \theta} \int \Pr(\tau \mid \theta)\, r[\tau]\, d\tau  
= \theta + \alpha \cdot \int \frac{\partial \Pr(\tau \mid \theta)}{\partial \theta}\, r[\tau]\, d\tau, \tag{19.24}
$$  

式中 $\alpha$ 为学习率。  

我们希望用经验观测到的轨迹集合来近似上述积分。这些轨迹服从分布 $\Pr(\tau \mid \theta)$；因此，为推进推导，我们在被积函数中同时乘以并除以该分布：  

$$
\theta \leftarrow \theta + \alpha \cdot \int \frac{\partial \Pr(\tau \mid \theta)}{\partial \theta}\, r[\tau]\, d\tau  
= \theta + \alpha \cdot \int \Pr(\tau \mid \theta) \cdot \frac{1}{\Pr(\tau \mid \theta)} \cdot \frac{\partial \Pr(\tau \mid \theta)}{\partial \theta}\, r[\tau]\, d\tau  
\approx \theta + \alpha \cdot \frac{1}{I} \sum_{i=1}^I \frac{1}{\Pr(\tau_i \mid \theta)} \cdot \frac{\partial \Pr(\tau_i \mid \theta)}{\partial \theta}\, r[\tau_i]. \tag{19.25}
$$  

该公式具有直观解释（见图19.15）：参数更新 $\theta$ 的方向是提升所观测轨迹 $\tau_i$ 的似然概率 $\Pr(\tau_i \mid \theta)$，其调整幅度正比于该轨迹对应的回报 $r[\tau_i]$。然而，更新还通过该轨迹的原始出现概率 $\Pr(\tau_i \mid \theta)$ 进行归一化，以补偿不同轨迹在采样过程中出现频率不均等的事实。若某轨迹本就高频出现且带来高回报，则无需大幅调整参数；而最大幅度的更新将来自那些虽罕见但能产生高额回报的轨迹。  

我们可借助似然比恒等式（likelihood ratio identity）进一步简化该表达式：  

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（C）麻省理工学院出版社。

19.5 策略梯度方法　391  
图19.15　策略梯度。同一策略下生成的五个轨迹（亮度越高表示累积奖励越高）。轨迹1、2和3始终获得较高奖励，但此类轨迹在当前策略下已频繁出现，因此无需调整策略。相反，轨迹4获得较低奖励，故应修改策略以避免生成类似轨迹。轨迹5虽获得高奖励，却属罕见情形——根据式(19.25)，它将导致策略参数发生最大幅度的更新。

$$
\frac{\partial \log[f[z]]}{\partial z} = \frac{1}{f[z]} \cdot \frac{\partial f[z]}{\partial z}, \quad \text{(19.26)}
$$

由此可得参数更新公式：

$$
\theta \leftarrow \theta + \alpha \cdot \frac{1}{I} \sum_{i=1}^{I} r[\tau_i] \frac{\partial \log \Pr(\tau_i \mid \theta)}{\partial \theta}. \quad \text{(19.27)}
$$

轨迹 $\tau$ 的对数概率 $\log[\Pr(\tau \mid \theta)]$ 可表示为：

$$
\log[\Pr(\tau \mid \theta)] = \log \left[ \Pr(s_1) \prod_{t=1}^{T} \pi[a_t \mid s_t, \theta] \, \Pr(s_{t+1} \mid s_t, a_t) \right], \quad \text{(19.28)}
$$

$$
= \log \Pr(s_1) + \sum_{t=1}^{T} \log \pi[a_t \mid s_t, \theta] + \sum_{t=1}^{T} \log \Pr(s_{t+1} \mid s_t, a_t),
$$

注意到仅中间项（即策略产生的动作对数概率）依赖于参数 $\theta$，因此可将式(19.27)重写为：

$$
\theta \leftarrow \theta + \alpha \cdot \frac{1}{I} \sum_{i=1}^{I} \sum_{t=1}^{T} r[\tau_i] \frac{\partial \log \pi[a_{it} \mid s_{it}, \theta]}{\partial \theta}, \quad \text{(19.29)}
$$

其中 $s_{it}$ 表示第 $i$ 个 episode 中时刻 $t$ 所处的状态，$a_{it}$ 表示该 episode 中时刻 $t$ 所采取的动作。注意，在此形式中，描述状态转移过程的项 $\Pr(s_{t+1} \mid s_t, a_t)$ 已完全消失。由此可见，该参数更新规则并不要求状态演化过程满足马尔可夫性。

进一步地，利用如下恒等式可对式(19.29)作简化：

$$
r[\tau_i] = \sum_{t=1}^{T} r_{i,t+1} = \sum_{k=1}^{t} r_{i,k+1} + \sum_{k=t}^{T} r_{i,k+1}. \quad \text{(19.30)}
$$

草稿：勘误请发送至 udlbookmail@gmail.com。

392 19 强化学习  
其中 $ r_{it} $ 表示第 $ i $ 个回合（episode）中时刻 $ t $ 所获得的奖励。可以（非显然地）证明：第一项（即时刻 $ t $ 之前的所有奖励）对时刻 $ t $ 的参数更新无影响，因此可将更新公式写作：  
$$
\theta \leftarrow \theta + \alpha \cdot \frac{1}{I} \sum_{i=1}^{I} \sum_{t=1}^{T} \frac{\partial \log \pi[a_{it} \mid s_{it}, \theta]}{\partial \theta} \sum_{k=t}^{T} r_{ik}. \tag{19.31}
$$  

19.5.2 REINFORCE 算法  
REINFORCE 是一种早期的策略梯度算法，它利用上述结论，并引入了折扣因子（discounting）。该算法属于蒙特卡洛（Monte Carlo）方法：基于当前策略 $ \pi[a \mid s, \theta] $ 生成若干回合 $ \tau_i = [s_{i1}, a_{i1}, r_{i2}, s_{i2}, a_{i2}, r_{i3}, \dots, r_{iT}] $。对于离散动作空间，该策略可由一个神经网络 $ \pi[s \mid \theta] $ 实现——该网络以当前状态 $ s $ 作为输入，为每个可能的动作输出一个标量值；这些输出再经 softmax 函数归一化，形成动作概率分布，并在每个时间步从中采样得到具体动作。  

对每个回合 $ i $，我们遍历其中每一步 $ t $，并计算从时刻 $ t $ 开始的子轨迹 $ \tau_{it} $ 的经验折扣回报（empirical discounted return）：  
$$
r[\tau_{it}] = \sum_{k=t+1}^{T} \gamma^{k-1} r_{ik}, \tag{19.32}
$$  
随后，针对每个回合 $ i $ 中的每个时间步 $ t $，执行如下参数更新：  
$$
\theta \leftarrow \theta + \alpha \cdot \frac{\partial \log \pi[a_{it} \mid s_{it}, \theta]}{\partial \theta} \, r[\tau_{it}] \quad \forall\, i,t, \tag{19.33}
$$  
其中 $ \pi[a_{it} \mid s_{it}, \theta] $ 表示在当前状态 $ s_{it} $ 和参数 $ \theta $ 下，神经网络输出动作 $ a_{it} $ 的概率，$ \alpha $ 为学习率。  

19.5.3 基线（Baselines）  
策略梯度方法具有较高的方差；往往需要大量回合才能获得导数的稳定估计。降低该方差的一种方法是：从轨迹回报 $ r[\tau_{it}] $ 中减去一个基线项 $ b $：  
$$
\theta \leftarrow \theta + \alpha \cdot \frac{1}{I} \sum_{i=1}^{I} \sum_{t=1}^{T} \frac{\partial \log \pi[a_{it} \mid s_{it}, \theta]}{\partial \theta} \, \big(r[\tau_{it}] - b\big). \tag{19.34}
$$  

习题 19.6：只要基线 $ b $ 不依赖于所采取的动作，  
本作品遵循知识共享署名—非商业性使用—禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

19.5 策略梯度方法　393  
图19.16　利用控制变量（control variates）降低估计量的方差。  
a) 考虑仅用少量样本估计 $ \mathbb{E}[a] $。该估计量（即样本均值）将随样本数量及其方差而变化。  
b) 现在考虑观测另一变量 $ b $，它与 $ a $ 协变，且满足 $ \mathbb{E}[b] = 0 $，同时与 $ a $ 具有相同的方差。  
c) $ a - b $ 的样本方差远小于 $ a $ 的样本方差，但其期望值仍满足 $ \mathbb{E}[a - b] = \mathbb{E}[a] $，因此我们获得了一个方差更低的无偏估计量。  

$$
\mathbb{E}_\tau \left[ \sum_{t=1}^T \frac{\partial \log \pi_\theta (a_t \mid s_t)}{\partial \theta} \cdot b \right] = 0, \tag{19.35}
$$

于是期望值保持不变。然而，若该基线（baseline）与引入不确定性的无关因子协变，则减去该基线即可降低方差（见图19.16）。这是**控制变量法**（method of control variates）的一个特例（参见习题19.7）。  

**习题19.7**  
这引出了一个问题：我们应如何选取 $ b $？可通过写出方差表达式、对其关于 $ b $ 求导、令导数为零并求解，从而得到使方差最小化的 $ b $ 值：  

**习题19.8**  
$$
b = \frac{\sum_{i=1}^I \sum_{t=1}^T \left( \frac{\partial \log \pi_\theta (a_{it} \mid s_{it})}{\partial \theta} \right)^2 r[\tau_i]}{\sum_{i=1}^I \sum_{t=1}^T \left( \frac{\partial \log \pi_\theta (a_{it} \mid s_{it})}{\partial \theta} \right)^2}. \tag{19.36}
$$

在实际应用中，该式常被近似为：  
$$
b = \frac{1}{I} \sum_{i=1}^I r[\tau_i]. \tag{19.37}
$$

减去这一基线，可消除如下情形所引入的方差：所有轨迹的回报 $ r[\tau_i] $ 均高于典型水平，但这仅是因为这些轨迹恰好经过了某些状态——无论采取何种动作，这些状态本身即具有高于平均值的回报。  

草稿：如有勘误，请发送邮件至 udlbookmail@gmail.com。

394 第19章 强化学习  
19.5.4 依赖状态的基线（State-dependent baselines）  
更优的选择是采用一个依赖于当前状态 $ s $ 的基线 $ b[s] $：  

$$
\theta \leftarrow \theta + \alpha \cdot \frac{1}{I} \sum_{i=1}^{I} \sum_{t=1}^{T_i} \frac{\partial \log \pi_\theta[s_{it}, a_{it}]}{\partial \theta} \left( r[\tau_i] - b[s_{it}] \right). \tag{19.38}
$$  

此处，我们旨在补偿因某些状态天然具有比其他状态更高的整体回报（无论采取何种动作）而引入的方差。  

一种合理的选择是基于当前状态的期望未来奖励，即状态值函数 $ v[s] $。此时，经验观测到的回报与基线之间的差值被称为**优势估计（advantage estimate）**。由于我们处于蒙特卡洛（Monte Carlo）框架下，该基线可由一个参数化的神经网络建模：$ b[s] = v[s, \phi] $，其中 $ \phi $ 为网络参数；我们可通过最小二乘损失函数，利用观测到的回报对该网络进行拟合：  

$$
L[\phi] = \frac{1}{I} \sum_{i=1}^{I} \sum_{t=1}^{T_i} \left( v[s_{it}, \phi] - \sum_{j=t}^{T_i} r_{ij+1} \right)^2. \tag{19.39}
$$  

19.6 Actor-Critic 方法  
Actor-Critic 算法是一类时序差分（Temporal Difference, TD）策略梯度算法。它可在每个时间步更新策略网络的参数，这与蒙特卡洛 REINFORCE 算法形成鲜明对比——后者必须等待一个或多个完整回合（episode）结束之后才能更新参数。  

在 TD 方法中，我们无法直接获得沿轨迹的全部未来奖励 $ r[\tau] = \sum_{k=t}^{T} r_k $。Actor-Critic 算法以**当前观测到的即时奖励**加上**下一状态的折扣价值**来近似该未来奖励总和：  

$$
r[\tau] \approx r_{it} + \gamma \cdot v[s_{i,t+1}, \phi]. \tag{19.40}
$$  

其中，值函数 $ v[s_{i,t+1}, \phi] $ 由另一个具有参数 $ \phi $ 的神经网络估计得到。  

将式 (19.40) 代入式 (19.38)，可得参数更新公式：  

$$
\theta \leftarrow \theta + \alpha \cdot \frac{1}{I} \sum_{i=1}^{I} \sum_{t=1}^{T_i} \frac{\partial \log \mathrm{Pr}(a_{it} \mid s_{it}, \theta)}{\partial \theta} \left( r_{it} + \gamma \cdot v[s_{i,t+1}, \phi] - v[s_{it}, \phi] \right). \tag{19.41}
$$  

与此同时，我们通过自举（bootstrapping）方式，利用如下损失函数更新参数 $ \phi $：  

$$
L[\phi] = \sum_{i=1}^{I} \sum_{t=1}^{T_i} \left( r_{it} + \gamma \cdot v[s_{i,t+1}, \phi] - v[s_{it}, \phi] \right)^2. \tag{19.42}
$$  

用于预测 $ \mathrm{Pr}(a \mid s_t) $ 的策略网络 $ \pi_\theta[s_t] $ 被称为 **Actor**；用于估计值函数 $ v[s_t, \phi] $ 的网络则被称为 **Critic**。实践中，Actor 和 Critic 常共享同一神经网络的底层结构（即采用共享主干网络）。  

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社（MIT Press）。

19.7 离线强化学习　395  
图19.17　决策变换器（Decision Transformer）。决策变换器将离线强化学习建模为一个序列预测任务。其输入是一组状态、动作和“剩余回报”（即当前时刻至回合结束的累计奖励）的序列，其中每一项均被映射为固定维度的嵌入向量。在每个时间步，网络预测下一个动作。在测试阶段，“剩余回报”是未知的；实践中通常先给出一个初始估计值，再根据后续实际观测到的奖励逐步减去。

该网络结构同时包含策略网络（actor）与价值网络（critic），并具有两组输出：一组用于预测策略，另一组用于预测状态值（或状态-动作值）。

需要注意的是，尽管 actor-critic 方法理论上可在每个时间步更新策略参数，但在实际应用中极少如此操作。代理（agent）通常会先在多个时间步内收集一批经验数据，再统一进行一次策略更新。

19.7 离线强化学习  
与环境交互是强化学习的核心环节。然而，在某些场景下，让一个未经训练的代理直接进入环境探索不同动作的效果并不现实。这可能是因为环境中的异常行为具有危险性（例如自动驾驶车辆控制），也可能是因为数据采集耗时或成本高昂（例如金融交易）。

但在这两类情形中，我们仍有可能从人类操作员处获取历史交互数据。离线强化学习（Offline RL），亦称批式强化学习（Batch RL），旨在仅通过观察过去的历史序列 $s_1, a_1, r_2, s_2, a_2, r_3, \dots$ 来学习如何在未来的回合中采取最大化累积奖励的动作，而**完全不与环境发生任何在线交互**。它不同于模仿学习（imitation learning）——后者是一种相关但有本质区别的技术：（i）无法访问奖励信号；（ii）目标是复现历史操作员的表现，而非超越之。

尽管目前已存在基于 Q 学习与策略梯度的离线强化学习方法，  
草稿：请将勘误发送至 udlbookmail@gmail.com。

396 第19章 强化学习  
这一范式开辟了全新的可能性。具体而言，我们可以将其视为一个序列学习问题，其目标是：在给定状态、奖励与动作的历史序列的前提下，预测下一个动作。决策变换器（Decision Transformer）利用变换器解码器框架（第12.7节）完成此类预测（图19.17）。  

然而，其目标是依据未来奖励来预测动作，而标准的状态–动作–奖励序列 $s, a, r$ 并未显式包含未来奖励信息。因此，决策变换器将时刻 $t$ 的奖励 $r_t$ 替换为“剩余回报”（returns-to-go）$R_{t:T} = \sum_{t'=t}^{T} r_{t'}$（即从 $t$ 到终止时刻 $T$ 所观测到的未来奖励之和）。其余框架与标准变换器解码器高度相似：状态、动作与剩余回报均通过可学习的映射转换为固定维度的嵌入向量。对于Atari游戏，状态嵌入可通过一个类似于图19.14所示的卷积网络实现；而动作与剩余回报的嵌入则可采用与词嵌入（图12.9）相同的方式进行学习。该变换器采用掩码自注意力机制（masked self-attention）与位置编码（position embeddings）进行训练。  

这种建模方式在训练阶段自然合理，但在推理阶段却引发了一个难题：我们无法预先获知剩余回报。该问题可通过如下方式解决——在初始步骤中设定期望的总回报值，并在后续每一步根据实际收到的奖励递减该值。例如，在Atari游戏中，期望总回报即为获胜所需的总得分。  

此外，决策变换器亦可基于在线交互经验进行微调，从而随时间持续学习。其显著优势在于摒弃了传统强化学习的大部分复杂机制及其固有的不稳定性，转而采用标准的监督学习范式。变换器能够从海量数据中学习，并在长时序上下文中整合信息，从而使“时间信用分配问题”（temporal credit assignment problem）更具可解性。这代表了强化学习领域一个极具吸引力的新方向。  

19.8 小结  
强化学习是一种面向马尔可夫决策过程（Markov decision processes）及类似系统的序列化决策框架。本章回顾了强化学习的表格化方法，包括：  
- 动态规划（dynamic programming）：适用于环境模型已知的情形；  
- 蒙特卡洛方法（Monte Carlo methods）：通过运行多个完整回合（episodes），依据所获奖励更新动作价值函数与策略；  
- 时序差分方法（temporal difference methods）：在回合进行过程中即对动作价值等量进行增量式更新。  

深度Q学习（Deep Q-Learning）是一种时序差分方法，它利用深度神经网络为每个状态预测所有可能动作的Q值（即动作价值）。该方法已成功训练智能体在Atari 2600游戏中达到接近人类水平的表现。策略梯度方法（policy gradient methods）则直接优化策略本身，而非为动作赋值；当环境部分可观测时，其所生成的随机性策略尤为重要。但这类方法的梯度更新具有较高噪声，为此研究者已提出多种改进技术以降低其方差。  

离线强化学习（offline reinforcement learning）适用于无法与环境交互、仅能依赖历史数据进行学习的场景。决策变换器借助深度学习领域的最新进展，构建状态–动作–奖励序列的模型，并预测能够最大化累积奖励的动作。  

本作品受知识共享署名–非商业性使用–禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社。

注释 397  
注释  

Sutton 与 Barto（2018）深入阐述了表格型强化学习方法。Li（2017）、Arulkumaran 等（2017）、François-Lavet 等（2018）以及 Wang 等（2022c）均提供了深度强化学习的综述性介绍。Graesser 与 Keng（2019）是一份极佳的入门资料，其中包含 Python 代码实现。

深度强化学习中的里程碑成果：强化学习领域绝大多数标志性成就均诞生于视频游戏或现实世界中的博弈场景，原因在于这些环境具有约束性强、动作空间有限且规则固定等特点。Deep Q-Learning（Mnih 等，2015）在一组 Atari 游戏基准测试中达到了人类水平的性能。AlphaGo（Silver 等，2016）击败了围棋世界冠军；而围棋此前曾被普遍认为是计算机极难掌握的博弈。Berner 等（2019）构建了一套系统，在五对五多人在线战术竞技游戏《Defense of the Ancients 2》（DOTA2）中战胜了世界冠军战队，该游戏要求玩家之间进行高度协同合作。Ye 等（2021）开发的系统仅需极少样本数据即可在 Atari 游戏中超越人类表现（相较以往系统需远超人类的经验量）。近期，Cicero 系统在需要自然语言协商与玩家间复杂协作的棋盘游戏《Diplomacy》（外交）中展现出人类水平的性能（FAIR，2022）。

强化学习亦已成功应用于组合优化问题（参见 Mazyavkina 等，2021）。例如，Kool 等（2019）训练出一个模型，在旅行商问题（TSP）上表现与最优启发式算法相当。近期，AlphaTensor（Fawzi 等，2022）将矩阵乘法建模为一种博弈，并通过学习以更少的标量乘法次数实现更快的矩阵乘法运算。鉴于深度学习严重依赖于矩阵乘法，这成为人工智能领域自我改进的首批实例之一。

经典强化学习方法：马尔可夫决策过程（MDP）理论的早期奠基性工作可追溯至 Thompson（1933）与 Thompson（1935）。贝尔曼递推关系式（Bellman recursions）由 Bellman（1966）提出。Howard（1960）引入了策略迭代（policy iteration）算法。Sutton 与 Barto（2018）指出，Andreae（1969）的工作首次采用 MDP 形式化框架来描述强化学习。

强化学习的现代发展阶段，可追溯至 Sutton（1984）与 Watkins（1989）的博士论文。Sutton（1988）提出了“时序差分学习”（temporal difference learning）这一术语。Watkins（1989）及 Watkins 与 Dayan（1992）提出了 Q-Learning，并借助巴拿赫不动点定理（Banach’s theorem）证明：由于贝尔曼算子（Bellman operator）是一种压缩映射（contraction mapping），Q-Learning 必收敛至唯一不动点。Watkins（1989）首次明确建立了动态规划（dynamic programming）与强化学习之间的联系。SARSA 算法由 Rummery 与 Niranjan（1994）提出。Gordon（1995）提出了拟合 Q 学习（fitted Q-learning），即利用机器学习模型预测每个状态–动作对的行动值（action value）。Riedmiller（2005）进一步提出神经拟合 Q 学习（neural-fitted Q-learning），使用神经网络一次性从输入状态预测所有可能动作对应的行动值。关于蒙特卡洛方法的早期研究由 Singh 与 Sutton（1996）开展，而“探索性起始”（exploring starts）算法则由 Sutton 与 Barto（1999）提出。需注意，以上仅为逾半个世纪研究工作的极其简略的概述；更为详尽的论述可参阅 Sutton 与 Barto（2018）。

深度 Q 网络（Deep Q-Networks）：深度 Q-Learning 由 Mnih 等（2015）提出，其思想直接承袭自神经拟合 Q 学习。该方法充分利用当时卷积神经网络（CNN）所取得的突破性进展，构建出一种拟合 Q 学习框架，使其在 Atari 游戏基准测试中达到人类水平性能。然而，深度 Q-Learning 面临所谓“致命三重奏”（deadly triad）问题（Sutton & Barto，2018）：任何同时包含以下三项要素的学习方案均可能导致训练不稳定——（i）自举法（bootstrapping）、（ii）离策略学习（off-policy learning）以及（iii）函数近似（function approximation）。此后大量研究致力于提升训练稳定性。Mnih 等（2015）引入经验回放缓冲区（experience replay buffer）（Lin，1992），随后 Schaul 等（2016）对其加以改进，赋予重要性更高的经验元组更高采样优先级，从而加快学习速度；该改进被称为“优先经验回放”（prioritized experience replay）。

草稿：请将勘误发送至 udlbookmail@gmail.com。

398 第19章 强化学习  
原始的Q学习论文通过拼接四帧图像，使网络能够观测物体的运动速度，从而使底层过程更接近完全可观测。Hausknecht与Stone（2015）提出了深度循环Q学习（deep recurrent Q-learning），采用循环神经网络架构，每次仅输入单帧图像，因其可“记忆”先前状态。Van Hasselt（2010）指出，由于Q学习中`max`操作的存在，状态值存在系统性高估问题，并由此提出双Q学习（double Q-learning）：同时训练两个模型以缓解该问题。该方法随后被应用于深度Q学习场景（Van Hasselt等，2016），但其有效性此后受到质疑（Hessel等，2018）。Wang等（2016）提出了深度对偶网络（deep dueling networks），其中同一网络的两个输出头分别预测：（i）状态值（state value）和（ii）各动作的优势函数（advantage function，即相对于状态值的动作相对价值）。其核心直觉在于：在某些情形下，真正重要的是状态本身的价值，而具体采取哪个动作影响甚微；将这两类估计解耦，有助于提升训练稳定性。

Fortunato等（2018）提出了含噪声深度Q网络（noisy deep Q-networks），即在Q网络的部分权重上乘以随机噪声，从而为预测引入随机性，以鼓励探索行为。随着策略逐步收敛至合理水平，网络可自主学习降低噪声幅度。分布式的DQN（Distributional DQN，Bellemare等，2017a；Dabney等，2018，延续Morimura等，2010的工作）旨在估计回报（return）的完整分布信息，而非仅估计其期望值。这使得网络有可能规避最坏情形下的结果，同时亦能提升性能——因为预测更高阶矩（higher moments）可提供更丰富的训练信号。Rainbow（Hessel等，2018）整合了六项针对原始深度Q学习算法的改进，包括对偶网络、分布式DQN与含噪声DQN等，从而在ATARI基准测试中同时提升了训练速度与最终性能。

策略梯度（Policy gradients）：Williams（1992）提出了REINFORCE算法。“策略梯度方法”（policy gradient method）这一术语最早见于Sutton等（1999）。Konda与Tsitsiklis（1999）提出了Actor-Critic算法。利用不同基线（baseline）降低方差的方法，参见Greensmith等（2004）及Peters与Schaal（2008）。后续研究进一步指出，值函数基线（value baseline）主要作用在于削弱更新步长的激进程度（aggressiveness），而非显著降低其方差（Mei等，2022）。策略梯度方法已被拓展用于生成确定性策略（Silver等，2014；Lillicrap等，2016；Fujimoto等，2018）。最直接的方法是对所有可能动作进行最大化，但若动作空间为连续型，则每一步均需执行一次优化过程。深度确定性策略梯度算法（Deep Deterministic Policy Gradient, DDPG；Lillicrap等，2016）则沿动作值函数关于动作的梯度方向更新策略（这隐含地采用了Actor-Critic框架）。

现代策略梯度方法：我们此前从参数更新角度引入了策略梯度。然而，它亦可被视作一种基于重要性采样（importance sampling）的代理损失（surrogate loss）优化方法——该损失函数以当前策略参数所生成的轨迹（trajectories）所对应的期望奖励为基础。这一视角允许我们合法地执行多次优化步。但该做法可能导致策略更新幅度过大。在监督学习中，“跨步过大”（overstepping）仅为次要问题，因为后续轨迹可予以校正；而在强化学习中，过大的更新会直接影响后续的数据采集过程，甚至造成严重破坏。

为此，学界已提出多种方法以约束此类更新。自然策略梯度（Natural policy gradients；Kakade，2001）基于自然梯度（natural gradients；Amari，1998），即利用费舍尔信息矩阵（Fisher information matrix）修正梯度下降方向。该方法可提供更优的更新路径，从而降低陷入局部平坦区域（local plateaus）的风险。然而，在参数量庞大的模型中，费舍尔矩阵的实际计算并不可行。在可信域策略优化（Trust-Region Policy Optimization, TRPO；Schulman等，2015）中，代理目标函数在旧策略与新策略之间的KL散度（KL divergence）约束下被最大化。Schulman等（2017）则提出一种更简洁的公式化形式：将KL散度作为正则化项引入目标函数。该正则化权重依据KL散度与预设目标值（即期望策略改变程度）之间的距离动态调整。近端策略优化（Proximal Policy Optimization, PPO；Schulman等，2017）则是一种更为简明的方法：通过裁剪（clipping）损失函数，确保策略更新幅度保持较小。

Actor-Critic：在第19.6节所述的Actor-Critic算法（Konda & Tsitsiklis，1999）中，Critic采用的是单步估计器（1-step estimator）。此外，也可使用`k`步估计器（`k`-step estimator，即  

This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

注释 399  
观察 $k$ 步折扣奖励，并用状态值估计近似后续奖励。随着 $k$ 增大，该估计的方差增大，但偏差减小。广义优势估计（Generalized Advantage Estimation, GAE；Schulman 等，2016）将来自多步的估计加权组合，并通过一个单一参数化权重项来权衡偏差与方差。  
Mnih 等（2016）提出了异步 Actor-Critic（Asynchronous Actor-Critic, A3C），其中多个智能体在并行环境中独立运行，并共同更新同一组参数。策略函数与价值函数均每隔 $T$ 个时间步，基于混合的 $k$-步回报进行更新。Wang 等（2017）则提出若干方法，旨在提升异步 Actor-Critic 的效率。  
软 Actor-Critic（Soft Actor-Critic, SAC；Haarnoja 等，2018b）在损失函数中引入熵项，以鼓励探索并缓解过拟合——因为该机制促使策略输出更均匀的概率分布，从而降低其置信度。  

**离线强化学习（Offline RL）**：在离线强化学习中，策略仅通过观察其他智能体的行为（包括其所获得的奖励）进行学习，而无法主动干预或改变环境交互过程。它与模仿学习（Imitation Learning）密切相关，后者的目标是在无奖励信号可用的前提下复现另一智能体的行为（参见 Hussein 等，2017）。一种常见思路是将离线 RL 视为一种特殊的离策略强化学习（off-policy RL）。然而在实践中，观测数据所对应的策略分布与实际部署策略之间存在显著的分布偏移（distributional shift），这会导致动作价值函数被过度乐观地高估，进而引发性能严重下降（参见 Fujimoto 等，2019；Kumar 等，2019a；Agarwal 等，2020）。保守 Q 学习（Conservative Q-Learning, CQL；Kumar 等，2020b）通过对 Q 值施加正则化，学习具有保守性的、对价值函数的下界估计。决策 Transformer（Decision Transformer；Chen 等，2021c）是一种简洁高效的离线学习方法，充分利用了已被深入研究的自注意力（self-attention）架构；该模型后续还可通过在线训练进一步微调（Zheng 等，2022）。  

**强化学习与聊天机器人**：聊天机器人可采用一种称为“基于人类反馈的强化学习”（Reinforcement Learning from Human Feedback, RLHF）的技术进行训练（Christiano 等，2018；Stiennon 等，2020）。例如，InstructGPT（ChatGPT 的前身；Ouyang 等，2022）起始于一个标准的 Transformer 解码器模型，随后基于由人工标注员撰写的提示–响应对（prompt–response pairs）对其进行微调。在此阶段，模型被优化以预测真实响应中的下一个词。  
遗憾的是，此类高质量人工标注数据的生产成本极高，难以大规模获取以支撑高性能模型训练。为解决这一问题，人工标注员转而对模型生成的若干候选响应进行偏好排序。这类（成本显著更低的）偏好数据被用于训练一个奖励模型（reward model）：这是一个独立的 Transformer 网络，输入为提示与模型响应，输出为一个标量，表征该响应的质量高低。最后，前述已微调的聊天机器人模型将借助该奖励模型作为监督信号，进一步训练以生成更高奖励的响应。此处无法直接使用标准梯度下降法，因为聊天机器人输出涉及采样过程，其不可导；因此，模型采用近端策略优化（Proximal Policy Optimization, PPO）——一种可解析求导的策略梯度方法——来提升所获奖励。  

**强化学习的其他研究方向**：强化学习是一个极为广阔的领域，其体量之大足以单独成书；本节文献综述仅作极粗略的概览。我们尚未讨论的其他重要方向包括：  
- **基于模型的强化学习（Model-Based RL）**：显式建模状态转移概率与奖励函数（参见 Moerland 等，2023）。该范式支持前向规划，且同一动力学模型可在不同奖励结构下复用；  
- **混合方法（Hybrid Methods）**：如 AlphaGo（Silver 等，2016）与 MuZero（Schrittwieser 等，2020），它们分别构建独立模型以刻画状态动态、策略及未来位置的价值；  
- **探索策略**：本章仅介绍了若干简单方法，例如 $\varepsilon$-贪心策略、含噪声的 Q 学习（noisy Q-learning），以及通过添加熵项惩罚过度自信的策略；  
- **内在动机（Intrinsic Motivation）**：指为鼓励探索而人为增设奖励信号的方法，从而赋予智能体某种“好奇心”（参见 Barto，2013；Aubret 等，2019）；  
- **分层强化学习（Hierarchical Reinforcement Learning）**：将最终目标分解为若干子任务进行求解（参见 Pateria 等，2021）；  
- **多智能体强化学习（Multi-Agent Reinforcement Learning）**：研究多个智能体共存于共享环境中的情形，该场景既可能呈现竞争性，也可能体现合作性（参见 Zhang 等，2021a）。  

草稿：如有勘误，请发送至 udlbookmail@gmail.com。

400 19 强化学习  
习题  

**习题 19.1**  
图 19.18 展示了一个示例马尔可夫奖励过程（Markov reward process）中的一条单一轨迹。已知折扣因子 $\gamma = 0.9$，请计算该轨迹中每一步的回报（return）。

**习题 19.2\***  
证明策略改进定理（policy improvement theorem）。考虑从策略 $\pi$ 变更为策略 $\pi'$：对于状态 $s_t$，新策略 $\pi'$ 选择使期望回报最大化的动作，即  
$$
\pi'[a_t \mid s_t] \leftarrow \arg\max_{a_t} \left[ r[s_t, a_t] + \gamma \cdot \sum_{s_{t+1}} \Pr(s_{t+1} \mid s_t, a_t)\, v[s_{t+1} \mid \pi] \right]. \tag{19.43}
$$  
而对于所有其他状态，$\pi'$ 与 $\pi$ 完全一致。试证明：原始策略 $\pi$ 在状态 $s_t$ 处的状态值 $v[s_t \mid \pi]$ 必然小于或等于新策略 $\pi'$ 在该状态下的状态值 $v[s_t \mid \pi']$（此处记号 $v[s_t \mid \pi']$ 表示在 $s_t$ 处采用 $\pi'$，此后仍沿用 $\pi$ 的混合策略）：  
$$
v[s_t \mid \pi] \leq q\big[s_t,\, \pi'[a_t \mid s_t] \,\big|\, \pi\big] \\
= \mathbb{E}_{\pi'}\!\left[ r_{t+1} + \gamma \cdot v[s_{t+1} \mid \pi] \right]. \tag{19.44}
$$  
**提示**：可从将 $v[s_{t+1} \mid \pi]$ 表达为关于新策略 $\pi'$ 的形式入手。

**习题 19.3**  
当状态值函数与策略按图 19.10a 所示初始化时，请证明：经过两次迭代——（i）策略评估（policy evaluation，即基于当前所有状态值同步更新全部状态值，并以新值覆盖旧值），以及（ii）策略改进（policy improvement）——后，其结果恰好如图 19.10b 所示。状态转移规则如下：策略所指定方向的动作获得一半概率；剩余一半概率则在其余所有合法动作间均分。奖励函数定义为：当企鹅离开一个洞穴（hole）时，无论执行何种动作，均返回 $-2$；当企鹅离开鱼形方块（fish tile）时，无论执行何种动作，均返回 $+3$，且此时情节（episode）终止，因此鱼形方块对应的状态值为 $+3$。设折扣因子 $\gamma = 0.9$。

**习题 19.4**  
Boltzmann 策略通过将动作概率 $\pi[a \mid s]$ 建立在当前状态–动作价值函数 $q[s,a]$ 的基础上，在探索（exploration）与利用（exploitation）之间取得平衡：  
$$
\pi[a \mid s] = \frac{\exp\!\big(q[s,a]/\tau\big)}{\sum_{a'} \exp\!\big(q[s,a']/\tau\big)}. \tag{19.45}
$$  
请解释温度参数 $\tau$ 如何调节以优先侧重探索或利用。

**习题 19.5\***  
当学习率 $\alpha = 1$ 时，Q 学习（Q-Learning）的更新式为：  
$$
f\big(q[s,a]\big) = r[s,a] + \gamma \cdot \max_a q[s',a], \tag{19.46}
$$  
其中 $s$ 为当前状态，$s'$ 为后续状态。请证明该映射 $f$ 是一个压缩映射（contraction mapping）（参见公式 16.30），即满足：  
$$
\left\| f\big(q_1[s,a]\big) - f\big(q_2[s,a]\big) \right\|_\infty < \left\| q_1[s,a] - q_2[s,a] \right\|_\infty \quad \forall\, q_1, q_2. \tag{19.47}
$$  
此处 $\|\cdot\|_\infty$ 表示 $\ell^\infty$ 范数（见附录 B.3.2）。由此可知，根据巴拿赫不动点定理（Banach’s fixed-point theorem），该映射存在唯一不动点，且迭代更新最终必然收敛。

本作品受知识共享署名–非商业性使用–禁止演绎（CC-BY-NC-ND）许可协议保护。（©）麻省理工学院出版社（MIT Press）。

注释 401  
图 19.18 一条在马尔可夫决策过程（MRP）中的轨迹。企鹅抵达第一条鱼所在方格时获得奖励 $+1$，落入洞中时获得奖励 $-2$，抵达第二条鱼所在方格时获得奖励 $+1$。折扣因子 $\gamma$ 为 $0.9$。

**习题 19.6**  
证明：  
$$
\mathbb{E}_\tau \left[ \frac{\partial}{\partial \theta} \log \Pr(\tau \mid \theta) \, b \right] = 0, \tag{19.48}
$$  
其中 $b$ 不依赖于轨迹 $\tau$，因此加入一个与轨迹无关的基线（baseline）项不会改变策略梯度更新的期望值。

**习题 19.7\***  
假设我们希望利用样本 $a_1, a_2, \dots, a_I$ 来估计量 $ \mathbb{E}[a] $。同时，我们还拥有一组配对样本 $b_1, b_2, \dots, b_I$，它们与 $a$ 协变，且满足 $\mathbb{E}[b] = \mu_b$。我们定义一个新变量：  
$$
a' = a - c(b - \mu_b). \tag{19.49}
$$  
证明：当常数 $c$ 被适当地选取时，有 $\mathrm{Var}[a'] \leq \mathrm{Var}[a]$。并求出使方差最小的最优 $c$ 的表达式。

**习题 19.8**  
式 (19.34) 中梯度估计可写作：  
$$
\mathbb{E}_\tau \left[ g[\theta] \, (r[\tau] - b) \right], \tag{19.50}
$$  
其中  
$$
g[\theta,\tau] = \sum_{t=1}^T \frac{\partial \log \Pr(a_t \mid s_t, \theta)}{\partial \theta}, \tag{19.51}
$$  
且  
$$
r[\tau] = \sum_{k=t}^T r_k. \tag{19.52}
$$  
证明：使该梯度估计方差最小的基线值 $b$ 由下式给出：  
$$
b = \frac{\mathbb{E}_\tau \left[ g[\theta,\tau]^2 \, r[\tau] \right]}{\mathbb{E}_\tau \left[ g[\theta,\tau]^2 \right]}. \tag{19.53}
$$  
你将需要用到式 (19.35) 的结果。

勘误建议请发送至：udlbookmail@gmail.com。