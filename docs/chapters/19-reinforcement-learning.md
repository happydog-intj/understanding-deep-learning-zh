# 第19章 强化学习

前几章介绍了生成模型，这些模型从一组无标签样本中学习。本章考虑**强化学习**（reinforcement learning），其中模型通过与环境的一系列交互来学习。在每个时间步，模型接收关于环境状态的信息，并据此选择一个动作。这个动作改变环境的状态并产生奖励。我们的目标是学习一种策略（即选择动作的规则），使得期望的累积奖励最大化。

强化学习在研发能够击败人类玩家的 Atari 游戏代理和 AlphaGo 围棋系统方面发挥了重要作用。它也被用于训练大型语言模型聊天机器人，其中奖励来自人类反馈，目标是使模型生成更有帮助的回复。

本章首先形式化强化学习问题并描述其关键概念——马尔可夫决策过程、回报、策略和值函数。然后分别讨论表格方法（动态规划、蒙特卡洛方法和时序差分方法）、拟合 Q 学习（使用深度网络近似值函数）、策略梯度方法以及 Actor-Critic 方法。最后简要介绍离线强化学习。

## 19.1 马尔可夫决策过程、回报和策略

我们从马尔可夫决策过程的组成部分入手，逐步构建完整框架。

### 19.1.1 马尔可夫过程

**马尔可夫过程**（Markov process）或**马尔可夫链**（Markov chain）产生一个随机的状态序列 $s_1, s_2, s_3, \ldots$，其中每个后续状态 $s_{t+1}$ 仅依赖于当前状态 $s_t$，而不依赖于更早的历史状态。这被称为**马尔可夫性质**（Markov property）。**状态转移概率**（state transition probabilities）$Pr(s_{t+1}|s_t)$ 定义了从一个状态转移到另一个状态的概率（图 19.1）。

> **图 19.1** 马尔可夫过程。a) 企鹅从左上角出发，在 $4\times4$ 网格上随机行走。在每个时间步，它以相等概率移动到相邻的格子。b) 转移概率 $Pr(s_{t+1}|s_t)$；以状态 6 为例，它有四个邻居（状态 2、5、7 和 10），各有 0.25 的概率。c-d) 从左上角出发的两条可能的轨迹。

### 19.1.2 马尔可夫奖励过程

**马尔可夫奖励过程**（Markov reward process）为马尔可夫过程增加了奖励 $r_{t+1}$，该奖励在从状态 $s_t$ 转移到状态 $s_{t+1}$ 时产生。奖励可以是确定性的也可以是随机的。**回报**（return）$G_t$ 是从时间步 $t$ 之后累积的**折扣奖励**（discounted rewards）之和（图 19.2）：

$$
G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k+1},
\tag{19.1}
$$

其中 $\gamma \in [0,1]$ 是**折扣因子**（discount factor），它控制我们对未来奖励的重视程度。折扣因子确保在无限时间范围内回报的总和是有限的。

> **图 19.2** 马尔可夫奖励过程。a) 企鹅在网格上行走。有些格子包含洞（奖励 $-2$）和鱼（奖励 $+3$），空格子的奖励为零。b) 一条可能的轨迹。c) 该轨迹的回报是折扣奖励之和。

> **图 19.3** 马尔可夫决策过程（MDP）。企鹅现在可以选择动作（上、下、左、右），但网格是湿滑的，所以它只有概率 0.5 按照选择的方向移动，其余概率均匀分配给其他有效方向。MDP 产生一个由状态、动作和奖励组成的序列。

### 19.1.3 马尔可夫决策过程

**马尔可夫决策过程**（Markov decision process）或 **MDP** 在每个时间步增加了一组可能的**动作**（actions）。动作 $a_t$ 改变了转移概率，现在写作 $Pr(s_{t+1}|s_t, a_t)$。奖励也可以依赖于动作，现在写作 $Pr(r_{t+1}|s_t, a_t)$。MDP 产生一个由状态 $s_t$、动作 $a_t$ 和奖励 $r_{t+1}$（在后续时间步接收）组成的序列 $(s_1, a_1, r_2), (s_2, a_2, r_3), (s_3, a_3, r_4)\ldots$（图 19.3）。执行这些动作的实体被称为**智能体**（agent）。

### 19.1.4 部分可观测马尔可夫决策过程

在**部分可观测马尔可夫决策过程**（partially observable Markov decision process）或 **POMDP** 中，状态不是直接可见的（图 19.4）。智能体接收的是从 $Pr(o_t|s_t)$ 中采样的观测 $o_t$。因此，POMDP 产生一个由状态、观测、动作、奖励组成的序列 $s_1, o_1, a_1, r_2, s_2, o_2, a_2, r_3, o_3, a_3, s_3, r_4, \ldots$。通常每个观测都与某些状态更兼容，但不足以唯一确定状态。

> **图 19.4** 部分可观测马尔可夫决策过程（POMDP）。在 POMDP 中，智能体无法获得完整的状态信息。这里，企鹅不知道当前状态，只能看到附近的格子（虚线框）。不幸的是，真实状态（3）与状态 9 中看到的情况无法区分。在第一种情况下，向右移动会导致掉入洞中（奖励 $-2$），而在后一种情况下，会到达鱼（奖励 $+3$）。

### 19.1.5 策略

确定智能体在每个状态下的动作的规则被称为**策略**（policy）（图 19.5）。策略可以是**随机的**（stochastic，策略为每个状态的动作定义一个分布）或**确定性的**（deterministic，智能体在给定状态下总是采取相同的动作）。随机策略 $\pi[a|s]$ 返回状态 $s$ 下每个可能动作 $a$ 的概率分布，从中采样新动作。确定性策略 $\pi[a|s]$ 对被选择的动作 $a$ 返回 1，其他动作返回 0。**平稳**（stationary）策略仅依赖于当前状态。**非平稳**（non-stationary）策略还依赖于时间步。

环境和智能体形成一个循环（图 19.6）。智能体接收上一时间步的状态 $s_t$ 和奖励 $r_t$。基于此，它可以修改策略 $\pi[a_t|s_t]$（如果需要），并选择下一个动作 $a_t$。环境然后按照 $Pr(s_{t+1}|s_t, a_t)$ 推进到下一个状态，并按照 $Pr(r_{t+1}|s_t, a_t)$ 发出奖励。

> **图 19.5** 策略。a) 确定性策略在每个状态下总是选择相同的动作（由箭头指示）。有些策略比其他策略更好。这个策略不是最优的，但总体上能将企鹅从左上角引导到右下角的奖励处。b) 这个策略更加随机。c) 随机策略为每个状态的动作定义一个概率分布（由箭头大小表示概率）。这样的优势是智能体能更充分地探索各个状态，在部分可观测的 MDP 中可能是最优性能所必需的。

> **图 19.6** 强化学习循环。智能体在时间 $t$ 根据状态 $s_t$ 按照策略 $\pi[a_t|s_t]$ 采取动作 $a_t$。这触发了新状态 $s_{t+1}$ 的生成（通过状态转移函数）和奖励 $r_{t+1}$（通过奖励函数）的产生。两者都反馈给智能体，然后智能体选择新的动作。

## 19.2 期望回报

上一节介绍了马尔可夫决策过程和智能体按照策略执行动作的概念。我们希望选择一个使期望回报最大化的策略。在本节中，我们将这个概念精确化。为此，我们为每个状态 $s_t$ 和状态-动作对 $\{s_t, a_t\}$ 赋予一个**值**（value）。

### 19.2.1 状态值和动作值

回报 $G_t$ 依赖于状态 $s_t$ 和策略 $\pi[a|s]$。从这个状态出发，智能体将经历一系列状态，采取动作并接收奖励。由于策略 $\pi[a_t|s_t]$、状态转移 $Pr(s_{t+1}|s_t, a_t)$ 和奖励 $Pr(r_{t+1}|s_t, a_t)$ 都可能是随机的，因此每次从同一起点出发的序列都可能不同。

我们可以通过考虑**期望回报** $v[s_t|\pi]$ 来衡量一个状态在给定策略 $\pi$ 下有多"好"。这是从该状态出发平均能获得的回报，被称为**状态值**（state value）或**状态-值函数**（state-value function）（图 19.7a）：

$$
v[s_t|\pi] = \mathbb{E}\left[G_t|s_t, \pi\right].
\tag{19.2}
$$

非形式化地说，状态值告诉我们如果从这个状态开始并此后遵循指定策略，平均能期望获得的**长期**奖励。对于即将带来大奖励的后续转移的状态，它的值最高（假设折扣因子 $\gamma$ 小于 1）。

类似地，**动作值**（action value）或**状态-动作值函数**（state-action value function）$q[s_t, a_t|\pi]$ 是在状态 $s_t$ 执行动作 $a_t$ 的期望回报（图 19.7b）：

$$
q[s_t, a_t|\pi] = \mathbb{E}\left[G_t|s_t, a_t, \pi\right].
\tag{19.3}
$$

动作值告诉我们如果从这个状态开始、采取这个动作、然后此后遵循指定策略，平均能期望获得的长期奖励。通过这个量，强化学习算法将未来奖励与当前动作联系起来（即解决**时间信用分配问题**）。

> **图 19.7** 状态值和动作值。a) 状态 $s_t$ 的值 $v[s_t|\pi]$（每个位置处的数字）是该状态在给定策略 $\pi$（灰色箭头）下的期望回报。它是从该状态出发的多条轨迹上获得的折扣奖励的平均和。这里，离鱼越近的状态越有价值。b) 状态 $s_t$ 中动作 $a_t$ 的值 $q[s_t, a_t, \pi]$（每个位置/状态对应四个数字，对应四个动作）是在该特定动作被采取的条件下的期望回报。在这种情况下，离鱼越近值越大，朝向鱼的方向的动作值也越大。c) 如果我们知道一个状态的动作值，就可以修改策略使其选择这些值的最大值对应的动作（面板 b 中的红色数字）。

### 19.2.2 最优策略

我们希望找到使期望回报最大化的策略。对于 MDP（但不适用于 POMDP），总存在一个确定性的平稳策略能最大化每个状态的值。如果我们知道这个最优策略，就能得到**最优状态-值函数** $v^*[s_t]$：

$$
v^*[s_t] = \max_{\pi}\left[\mathbb{E}\left[G_t|s_t, \pi\right]\right].
\tag{19.4}
$$

类似地，最优状态-动作值函数在最优策略下获得：

$$
q^*[s_t, a_t] = \max_{\pi}\left[\mathbb{E}\left[G_t|s_t, a_t, \pi\right]\right].
\tag{19.5}
$$

反过来，如果我们知道最优动作值 $q^*[s_t, a_t]$，就可以通过选择具有最高值的动作 $a_t$ 来推导最优策略（图 19.7c）：

$$
\pi[a_t|s_t] \leftarrow \mathop{\text{argmax}}_{a_t}\left[q^*[s_t, a_t]\right].
\tag{19.6}
$$

实际上，一些强化学习算法就是基于交替估计动作值和策略的（参见 19.3 节）。

### 19.2.3 贝尔曼方程

我们可能不知道任何策略的状态值 $v[s_t]$ 或动作值 $q[s_t, a_t]$。然而，我们知道它们必须彼此一致，且容易写出它们之间的关系。状态值 $v[s_t]$ 可以通过对动作值 $q[s_t, a_t]$ 取加权和得到，权重是策略下采取该动作的概率 $\pi[a_t|s_t]$（图 19.8）：

$$
v[s_t] = \sum_{a_t}\pi[a_t|s_t]q[s_t, a_t].
\tag{19.7}
$$

类似地，一个动作的值是采取该动作产生的即时奖励 $r_{t+1} = r[s_t, a_t]$，加上处于后续状态 $s_{t+1}$ 的值 $v[s_{t+1}]$ 乘以折扣因子 $\gamma$（图 19.9）。由于 $s_{t+1}$ 的分配不是确定性的，我们按照转移概率 $Pr(s_{t+1}|s_t, a_t)$ 对值 $v[s_{t+1}]$ 加权：

$$
q[s_t, a_t] = r[s_t, a_t] + \gamma \cdot \sum_{s_{t+1}}Pr(s_{t+1}|s_t, a_t)v[s_{t+1}].
\tag{19.8}
$$

将公式 19.8 代入公式 19.7，得到时间 $t$ 和 $t+1$ 之间状态值的关系：

$$
v[s_t] = \sum_{a_t}\pi[a_t|s_t]\left(r[s_t, a_t] + \gamma \cdot \sum_{s_{t+1}}Pr(s_{t+1}|s_t, a_t)v[s_{t+1}]\right).
\tag{19.9}
$$

类似地，将公式 19.7 代入公式 19.8，得到时间 $t$ 和 $t+1$ 之间动作值的关系：

$$
q[s_t, a_t] = r[s_t, a_t] + \gamma \cdot \sum_{s_{t+1}}Pr(s_{t+1}|s_t, a_t)\left(\sum_{a_{t+1}}\pi[a_{t+1}|s_{t+1}]q[s_{t+1}, a_{t+1}]\right).
\tag{19.10}
$$

后面两个关系就是**贝尔曼方程**（Bellman equations），是许多强化学习方法的基础。简而言之，它们要求状态（动作）值必须是自洽的。因此，当我们更新一个状态（动作）值的估计时，会产生连锁反应，导致对所有其他值的修改。

> **图 19.8** 状态值和动作值之间的关系。状态 6 的值 $v[s_t=6]$ 是状态 6 处动作值 $q[s_t=6, a_t]$ 的加权和，权重是策略概率 $\pi[a_t|s_t=6]$。

> **图 19.9** 动作值和状态值之间的关系。在状态 6 采取动作 2 的值 $q[s_t=6, a_t=2]$ 是采取该动作的奖励 $r[s_t=6, a_t=2]$ 加上后续状态中折扣值 $v[s_{t+1}]$ 的加权和，权重是转移概率 $Pr(s_{t+1}|s_t=6, a_t=2)$。贝尔曼方程将这个关系与图 19.8 的关系串联起来，将当前和下一时刻的 (i) 状态值和 (ii) 动作值联系起来。

## 19.3 表格强化学习

**表格强化学习算法**（tabular RL algorithms，即不依赖函数近似的算法）分为**基于模型的**（model-based）方法和**无模型的**（model-free）方法。**基于模型的方法**显式使用 MDP 结构（转移矩阵 $Pr(s_{t+1}|s_t, a_t)$ 和奖励结构 $r[s, a]$）来寻找最优策略。如果这些已知，这是一个可以用**动态规划**（dynamic programming）解决的直接优化问题。如果未知，原则上可以从观测到的 MDP 轨迹中估计。

相反，**无模型方法**假设底层 MDP 的转移矩阵和奖励结构是未知的。这些方法分为两类：

1. **值估计**（value estimation）方法估计最优状态-动作值函数，然后根据每个状态中具有最大值的动作来分配策略。
2. **策略估计**（policy estimation）方法使用梯度下降技术直接估计最优策略，不需要估计模型或值函数的中间步骤。

在每个类别中，**蒙特卡洛**（Monte Carlo）方法通过模拟策略在 MDP 中的多条轨迹来收集如何改进该策略的信息。有时在更新策略之前模拟很多轨迹是不可行或不实际的。**时序差分**（temporal difference, TD）方法在智能体遍历 MDP **的同时**更新策略。

我们现在简要描述动态规划方法、蒙特卡洛值估计方法和 TD 值估计方法。第 19.4 节描述了深度网络如何用于 TD 值估计方法。我们在第 19.5 节回到策略估计。

### 19.3.1 动态规划

动态规划算法假设我们拥有转移和奖励结构的**完美**知识。在这方面，它们区别于大多数观察智能体与环境交互以间接收集这些信息的强化学习算法。

状态值 $v[s]$ 被任意初始化（通常为零）。确定性策略 $\pi[a|s]$ 也被初始化（例如，为每个状态随机选择一个动作）。算法然后交替进行：迭代计算当前策略的状态值（**策略评估**）和改进该策略（**策略改进**）。

**策略评估：** 我们遍历所有状态 $s_t$，更新它们的值：

$$
v[s_t] \leftarrow \sum_{a_t}\pi[a_t|s_t]\left(r[s_t, a_t] + \gamma \cdot \sum_{s_{t+1}}Pr(s_{t+1}|s_t, a_t)v[s_{t+1}]\right),
\tag{19.11}
$$

其中 $s_{t+1}$ 是后续状态，$Pr(s_{t+1}|s_t, a_t)$ 是状态转移概率。每次更新使 $v[s_t]$ 与后续状态 $s_{t+1}$ 的值一致，使用的是状态值的贝尔曼方程（公式 19.9）。这被称为**自举**（bootstrapping）。

**策略改进：** 为了更新策略，我们贪心地选择使每个状态的值最大化的动作：

$$
\pi[a_t|s_t] \leftarrow \mathop{\text{argmax}}_{a_t}\left[r[s_t, a_t] + \gamma \cdot \sum_{s_{t+1}}Pr(s_{t+1}|s_t, a_t)v[s_{t+1}]\right].
\tag{19.12}
$$

根据**策略改进定理**，这保证能改进策略。这两个步骤迭代直到策略收敛（图 19.10）。

这种方法有许多变体。在**策略迭代**（policy iteration）中，策略评估步骤迭代到收敛后再进行策略改进。值可以就地更新也可以同步更新。在**值迭代**（value iteration）中，策略评估过程只遍历一次值就进行策略改进。**异步**（asynchronous）动态规划算法不需要在每一步系统地遍历所有值，而是可以按任意顺序更新状态子集。

> **图 19.10** 动态规划。a) 状态值初始化为零，策略（箭头）随机选择。b) 状态值更新为与其邻居一致（公式 19.11，显示两次迭代后的结果）。策略更新为将智能体移向具有最高值的状态（公式 19.12）。c) 经过多次迭代，算法收敛到最优策略，其中企鹅试图避开洞并到达鱼。

### 19.3.2 蒙特卡洛方法

与动态规划算法不同，蒙特卡洛方法不需要知道 MDP 的转移概率和奖励结构。相反，它们通过从 MDP 中重复采样轨迹并观察奖励来获取经验。它们交替进行动作值的计算（基于这些经验）和策略的更新（基于动作值）。

为了估计动作值 $q[s, a]$，运行一系列**回合**（episodes）。每个回合从给定的状态和动作开始，然后遵循当前策略，产生一系列动作、状态和奖励（图 19.11a）。给定状态-动作对在当前策略下的动作值被估计为每次该对出现时其后的经验回报（即时间折扣的奖励累积和）的平均值（图 19.11b）。然后通过在每个状态选择具有最大值的动作来更新策略（图 19.11c）：

$$
\pi[a|s] \leftarrow \mathop{\text{argmax}}_{a}\left[q[s, a]\right].
\tag{19.13}
$$

> **图 19.11** 蒙特卡洛方法。a) 策略（箭头）随机初始化。MDP 被反复模拟，这些回合的轨迹被存储（橙色和棕色路径代表两条轨迹）。b) 动作值基于这些轨迹上观察到的回报的平均值来经验性估计。在这种情况下，动作值初始都为零，且仅在观察到某动作的地方进行了更新。c) 然后可以根据获得最好（或最不差）奖励的动作来更新策略。

这是一种**在策略**（on-policy）方法；当前最优策略被用来引导智能体通过环境。这个策略基于观测到的动作值，但当然，无法估计那些从未使用过的动作的值，也没有什么能鼓励算法去探索这些动作。一种解决方案是使用**探索性起始**（exploring starts），即以所有可能的状态-动作对发起回合，使得每种组合至少被观察一次。然而，如果状态数量很大或起始点不可控制，这是不切实际的。另一种方法是使用 **$\epsilon$-贪心**（epsilon greedy）策略，其中以概率 $\epsilon$ 随机选择动作，以概率 $1-\epsilon$ 选择最优动作。$\epsilon$ 的选择权衡了**利用**（exploitation）和**探索**（exploration）。这里，在策略方法将从 epsilon-贪心家族中寻找最优策略，但这通常**不是**总体最优策略。

相反，在**离策略**（off-policy）方法中，最优策略 $\pi$（**目标策略**，target policy）是从不同的**行为策略**（behavior policy）$\pi'$ 生成的回合中学习的。通常，目标策略是确定性的，而行为策略是随机的（例如，epsilon-贪心策略）。因此，行为策略可以探索环境，而学习到的目标策略保持高效。一些离策略方法显式使用**重要性采样**（importance sampling）来利用策略 $\pi'$ 的样本估计策略 $\pi$ 下的动作值。其他方法，如 Q 学习（在下一节描述），基于贪心动作估计值，尽管这不一定是被选择的动作。

### 19.3.3 时序差分方法

动态规划方法使用自举过程来更新值，使其在当前策略下自洽。蒙特卡洛方法通过采样 MDP 来获取信息。**时序差分**（temporal difference, TD）方法结合了自举和采样。然而，与蒙特卡洛方法不同的是，它们在智能体遍历 MDP 的**过程中**而不是之后更新值和策略。

**SARSA**（State-Action-Reward-State-Action）是一种在策略算法，更新规则为：

$$
q[s_t, a_t] \leftarrow q[s_t, a_t] + \alpha\left(r[s_t, a_t] + \gamma \cdot q[s_{t+1}, a_{t+1}] - q[s_t, a_t]\right),
\tag{19.14}
$$

其中 $\alpha \in \mathbb{R}^+$ 是学习率。括号中的项被称为 **TD 误差**（TD error），衡量的是估计的动作值 $q[s_t, a_t]$ 与走一步后的估计 $r[s_t, a_t] + \gamma \cdot q[s_{t+1}, a_{t+1}]$ 之间的一致性。

相反，**Q 学习**（Q-Learning）是一种离策略算法，更新规则为（图 19.12）：

$$
q[s_t, a_t] \leftarrow q[s_t, a_t] + \alpha\left(r[s_t, a_t] + \gamma \cdot \max_{a}\left[q[s_{t+1}, a]\right] - q[s_t, a_t]\right),
\tag{19.15}
$$

其中每一步的动作选择来自不同的行为策略 $\pi'$。在两种情况下，策略都通过在每个状态取动作值的最大值来更新（公式 19.13）。可以证明这些更新是**收缩映射**（contraction mappings）（见公式 16.20）；假设每个状态-动作对被访问无限次，动作值最终会收敛。

> **图 19.12** Q 学习。a) 智能体从状态 $s_t$ 开始，按照策略采取动作 $a_t=2$。它没有在冰上滑倒，所以向下移动，离开原始状态获得奖励 $r[s_t, a_t]=0$。b) 新状态的最大动作值（这里为 0.43）。c) 原始状态中动作 2 的动作值根据后续状态最大动作值的当前估计、奖励、折扣因子 $\gamma=0.9$ 和学习率 $\alpha=0.1$ 更新为 1.12。这改变了原始状态处最高的动作值，因此策略也随之改变。

## 19.4 拟合 Q 学习

上述表格蒙特卡洛和 TD 算法反复遍历整个 MDP 并更新动作值。然而，这仅在状态-动作空间较小时才可行。不幸的是，这种情况很少出现；即使在棋盘这样受约束的环境中，也有超过 $10^{40}$ 种可能的合法状态。

在**拟合 Q 学习**（fitted Q-learning）中，动作值的离散表示 $q[s_t, a_t]$ 被机器学习模型 $q[\mathbf{s}_t, a_t, \boldsymbol{\phi}]$ 替代，其中状态现在由向量 $\mathbf{s}_t$ 而不仅仅是索引表示。然后我们定义一个基于相邻动作值一致性的最小二乘损失（类似于 Q 学习的损失，见公式 19.15）：

$$
L[\boldsymbol{\phi}] = \left(r[\mathbf{s}_t, a_t] + \gamma \cdot \max_{a}\left[q[\mathbf{s}_{t+1}, a, \boldsymbol{\phi}]\right] - q[\mathbf{s}_t, a_t, \boldsymbol{\phi}]\right)^2,
\tag{19.16}
$$

由此得到更新规则：

$$
\boldsymbol{\phi} \leftarrow \boldsymbol{\phi} + \alpha\left(r[\mathbf{s}_t, a_t] + \gamma \cdot \max_{a}\left[q[\mathbf{s}_{t+1}, a, \boldsymbol{\phi}]\right] - q[\mathbf{s}_t, a_t, \boldsymbol{\phi}]\right)\frac{\partial q[\mathbf{s}_t, a_t, \boldsymbol{\phi}]}{\partial \boldsymbol{\phi}}.
\tag{19.17}
$$

拟合 Q 学习与 Q 学习的不同之处在于不再保证收敛。参数的改变同时影响目标 $r[\mathbf{s}_t, a_t] + \gamma \cdot \max_{a_{t+1}}[q[\mathbf{s}_{t+1}, a_{t+1}, \boldsymbol{\phi}]]$（最大值可能改变）和预测 $q[\mathbf{s}_t, a_t, \boldsymbol{\phi}]$。理论和实验都表明这可能损害收敛性。

### 19.4.1 用于玩 ATARI 游戏的深度 Q 网络

深度网络非常适合从高维状态空间进行预测，因此是拟合 Q 学习中模型的自然选择。原则上，它们可以同时接受状态和动作作为输入来预测值，但在实践中，网络只接受状态并同时预测每个动作的值。

**深度 Q 网络**（Deep Q-Network）是一个突破性的强化学习架构，利用深度网络学习玩 ATARI 2600 游戏。观测数据包括 $220\times160$ 的图像，每个像素有 128 种可能的颜色（图 19.13）。图像被调整为 $84\times84$ 大小，只保留亮度值。不幸的是，完整状态无法从单帧观察到。例如，游戏物体的速度是未知的。为了帮助解决这个问题，网络在每个时间步输入最近四帧来组成 $\mathbf{s}_t$。这些帧通过三个卷积层，然后是一个全连接层来预测每个动作的值（图 19.14）。

对标准训练过程做了几项修改。首先，奖励（由游戏分数驱动）被裁剪为负变化 $-1$、正变化 $+1$。这补偿了不同游戏之间分数的巨大差异，并允许使用相同的学习率。其次，系统利用了**经验回放**（experience replay）。不是基于当前步骤的元组 $\langle\mathbf{s}_t, a_t, r_{t+1}, \mathbf{s}_{t+1}\rangle$ 更新网络，而是将最近的所有元组存储在缓冲区中。从该缓冲区中随机采样批次来生成训练数据。这种方法多次复用数据样本，并减少了由相邻帧相似性引起的批次内相关性。

最后，拟合 Q 网络中的收敛问题通过将目标参数固定为值 $\boldsymbol{\phi}^-$ 并定期更新来解决。这给出了更新规则：

$$
\boldsymbol{\phi} \leftarrow \boldsymbol{\phi} + \alpha\left(r[\mathbf{s}_t, a_t] + \gamma \cdot \max_{a}\left[q[\mathbf{s}_{t+1}, a, \boldsymbol{\phi}^-]\right] - q[\mathbf{s}_t, a_t, \boldsymbol{\phi}]\right)\frac{\partial q[\mathbf{s}_t, a_t, \boldsymbol{\phi}]}{\partial \boldsymbol{\phi}}.
\tag{19.18}
$$

现在网络不再追逐一个移动的目标，更不容易发生振荡。

使用这些启发式方法和 $\epsilon$-贪心策略，深度 Q 网络在 49 个游戏上的表现达到了与专业游戏测试人员相当的水平，使用的是相同的网络架构（每个游戏单独训练）。值得注意的是，训练过程非常耗费数据。学习每个游戏大约需要 38 个整天的游戏经验。在某些游戏中，算法超越了人类表现。而在"Montezuma's Revenge"等其他游戏上，几乎没有任何进展。这个游戏的特点是奖励稀疏，且有多个外观完全不同的屏幕。

> **图 19.13** Atari 基准测试。Atari 基准测试由 49 个 Atari 2600 游戏组成，包括 Breakout（图中所示）、Pong 以及各种射击、平台和其他类型的游戏。a-d) 即使是单屏游戏，由于物体速度未知，状态也不是完全可观测的。因此，通常使用多个相邻帧（这里为四帧）来表示状态。e) 动作模拟用户通过摇杆的输入。f) 有 18 个动作，对应八个移动方向或不移动，以及每种情况下按钮是否按下。

> **图 19.14** 深度 Q 网络架构。输入 $\mathbf{s}_t$ 由 ATARI 游戏的四个相邻帧组成。每帧调整为 $84\times84$ 并转换为灰度。这些帧作为四个通道表示，经过 $8\times8$ 步长 4 的卷积处理，然后是 $4\times4$ 步长 2 的卷积，再接两个全连接层。最终输出预测该状态下 18 个动作中每个动作的动作值 $q[\mathbf{s}_t, a_t]$。

### 19.4.2 双重 Q 学习和双重深度 Q 网络

Q 学习的一个潜在缺陷是更新中对动作的最大化操作：

$$
q[s_t, a_t] \leftarrow q[s_t, a_t] + \alpha\left(r[s_t, a_t] + \gamma \cdot \max_{a}\left[q[s_{t+1}, a]\right] - q[s_t, a_t]\right)
\tag{19.19}
$$

会导致估计的动作值 $q[s_t, a_t]$ 产生系统性偏差。考虑两个提供相同平均奖励的动作，但一个是随机的，另一个是确定性的。随机奖励大约有一半的时间会超过平均值，且会被最大化操作选中，导致相应的动作值 $q[s_t, a_t]$ 被高估。类似的论证也适用于网络 $q[\mathbf{s}_t, a_t, \boldsymbol{\phi}]$ 的随机不准确性或 Q 函数的随机初始化。

根本问题是同一个网络既选择目标（通过最大化操作）又更新值。**双重 Q 学习**（Double Q-Learning）通过同时训练两个模型 $q_1[s_t, a_t, \pi_1]$ 和 $q_2[s_t, a_t, \pi_2]$ 来解决这个问题：

$$
q_1[s_t, a_t] \leftarrow q_1[s_t, a_t] + \alpha\left(r[s_t, a_t] + \gamma \cdot q_2\left[s_{t+1}, \mathop{\text{argmax}}_{a}\left[q_1[s_{t+1}, a]\right]\right] - q_1[s_t, a_t]\right)
$$

$$
q_2[s_t, a_t] \leftarrow q_2[s_t, a_t] + \alpha\left(r[s_t, a_t] + \gamma \cdot q_1\left[s_{t+1}, \mathop{\text{argmax}}_{a}\left[q_2[s_{t+1}, a]\right]\right] - q_2[s_t, a_t]\right).
\tag{19.20}
$$

现在目标的选择和目标本身被解耦，这有助于防止这些偏差。在实践中，新的元组 $\langle s, a, r, s'\rangle$ 被随机分配给其中一个模型进行更新。这被称为**双重 Q 学习**。**双重深度 Q 网络**（double DQNs）使用深度网络 $q[\mathbf{s}_t, a_t, \boldsymbol{\phi}_1]$ 和 $q[\mathbf{s}_t, a_t, \boldsymbol{\phi}_2]$ 来估计动作值，更新变为：

$$
\boldsymbol{\phi}_1 \leftarrow \boldsymbol{\phi}_1 + \alpha\left(r[\mathbf{s}_t, a_t] + \gamma \cdot q\left[\mathbf{s}_{t+1}, \mathop{\text{argmax}}_{a}\left[q[\mathbf{s}_{t+1}, a, \boldsymbol{\phi}_1]\right], \boldsymbol{\phi}_2\right] - q[\mathbf{s}_t, a_t, \boldsymbol{\phi}_1]\right)\frac{\partial q[\mathbf{s}_t, a_t, \boldsymbol{\phi}_1]}{\partial \boldsymbol{\phi}_1}
$$

$$
\boldsymbol{\phi}_2 \leftarrow \boldsymbol{\phi}_2 + \alpha\left(r[\mathbf{s}_t, a_t] + \gamma \cdot q\left[\mathbf{s}_{t+1}, \mathop{\text{argmax}}_{a}\left[q[\mathbf{s}_{t+1}, a, \boldsymbol{\phi}_2]\right], \boldsymbol{\phi}_1\right] - q[\mathbf{s}_t, a_t, \boldsymbol{\phi}_2]\right)\frac{\partial q[\mathbf{s}_t, a_t, \boldsymbol{\phi}_2]}{\partial \boldsymbol{\phi}_2}.
\tag{19.21}
$$

## 19.5 策略梯度方法

Q 学习先估计动作值，然后利用它们来更新策略。相反，**基于策略的方法**（policy-based methods）直接学习随机策略 $\pi[a_t|\mathbf{s}_t, \boldsymbol{\theta}]$。这是一个以可训练参数 $\boldsymbol{\theta}$ 为参数的函数，将状态 $\mathbf{s}_t$ 映射到动作 $a_t$ 的分布 $Pr(a_t|\mathbf{s}_t)$，可以从中采样。在 MDP 中，总存在一个最优的确定性策略。然而，有三个理由使用随机策略：

1. 随机策略自然有助于空间的探索；我们不必在每个时间步都采取最优动作。
2. 当我们修改随机策略时，损失平滑变化。这意味着即使奖励是离散的，我们也可以使用梯度下降方法。这类似于在（离散的）分类问题中使用最大似然。当模型参数改变时，损失平滑变化以使真实类别更可能。
3. MDP 假设通常是不正确的；我们通常不具有状态的完整知识。例如，考虑一个智能体在只能观察附近位置的环境中导航（例如，图 19.4）。如果两个位置看起来相同但附近的奖励结构不同，随机策略允许采取不同动作的可能性，直到这种模糊性被消除。

### 19.5.1 梯度更新的推导

考虑通过 MDP 的一条轨迹 $\boldsymbol{\tau} = [\mathbf{s}_1, a_1, \mathbf{s}_2, a_2, \ldots, \mathbf{s}_T, a_T, \mathbf{s}_{T+1}]$。这条轨迹的概率 $Pr(\boldsymbol{\tau}|\boldsymbol{\theta})$ 取决于状态演化函数 $Pr(\mathbf{s}_{t+1}|\mathbf{s}_t, a_t)$ 和当前随机策略 $\pi[a_t|\mathbf{s}_t, \boldsymbol{\theta}]$：

$$
Pr(\boldsymbol{\tau}|\boldsymbol{\theta}) = Pr(\mathbf{s}_1)\prod_{t=1}^{T}\pi[a_t|\mathbf{s}_t, \boldsymbol{\theta}]Pr(\mathbf{s}_{t+1}|\mathbf{s}_t, a_t).
\tag{19.22}
$$

策略梯度算法旨在最大化许多此类轨迹上的期望回报。这些轨迹从分布 $Pr(\boldsymbol{\tau}|\boldsymbol{\theta})$ 中采样，因此我们乘以并除以这个分布：

$$
\boldsymbol{\theta} = \mathop{\text{argmax}}_{\boldsymbol{\theta}}\left[\mathbb{E}_{\boldsymbol{\tau}}\left[r[\boldsymbol{\tau}]\right]\right] = \mathop{\text{argmax}}_{\boldsymbol{\theta}}\left[\int Pr(\boldsymbol{\tau}|\boldsymbol{\theta})r[\boldsymbol{\tau}]d\boldsymbol{\tau}\right],
\tag{19.23}
$$

其中 $r[\boldsymbol{\tau}]$ 是沿轨迹获得的所有奖励之和。为了最大化这个量，我们使用梯度上升更新：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot \frac{\partial}{\partial\boldsymbol{\theta}}\int Pr(\boldsymbol{\tau}|\boldsymbol{\theta})r[\boldsymbol{\tau}]d\boldsymbol{\tau} = \boldsymbol{\theta} + \alpha \cdot \int\frac{\partial Pr(\boldsymbol{\tau}|\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}r[\boldsymbol{\tau}]d\boldsymbol{\tau},
\tag{19.24}
$$

其中 $\alpha$ 是学习率。我们希望用经验观测轨迹上的求和来近似这个积分。这些轨迹是从分布 $Pr(\boldsymbol{\tau}|\boldsymbol{\theta})$ 中抽取的，所以我们乘以并除以这个分布：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot \int Pr(\boldsymbol{\tau}|\boldsymbol{\theta})\frac{1}{Pr(\boldsymbol{\tau}|\boldsymbol{\theta})}\frac{\partial Pr(\boldsymbol{\tau}|\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}r[\boldsymbol{\tau}]d\boldsymbol{\tau} \approx \boldsymbol{\theta} + \alpha \cdot \frac{1}{I}\sum_{i=1}^{I}\frac{1}{Pr(\boldsymbol{\tau}_i|\boldsymbol{\theta})}\frac{\partial Pr(\boldsymbol{\tau}_i|\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}r[\boldsymbol{\tau}_i].
\tag{19.25}
$$

这个方程有一个简单的解释（图 19.15）；更新改变参数 $\boldsymbol{\theta}$ 以增加观测轨迹 $\boldsymbol{\tau}_i$ 的似然 $Pr(\boldsymbol{\tau}_i|\boldsymbol{\theta})$，增幅正比于该轨迹的回报 $r[\boldsymbol{\tau}_i]$。然而，它也按轨迹本身的概率进行归一化，以补偿某些轨迹比其他轨迹更频繁出现的事实。如果一条轨迹已经很常见且产生高回报，则不需要太大改变。最大的更新来自那些不常见但产生大回报的轨迹。

> **图 19.15** 策略梯度。同一策略的五个回合（更亮表示更高奖励）。轨迹 1、2 和 3 持续产生较高的奖励，但类似轨迹在这个策略下已经频繁出现，所以无需改变。相反，轨迹 4 获得的奖励很低，因此应该修改策略以避免产生类似轨迹。轨迹 5 获得了高奖励**且**不常见。这将导致公式 19.25 下策略的最大变化。

我们可以使用**似然比恒等式**（likelihood ratio identity）来简化这个表达式：

$$
\frac{\partial\log[f[z]]}{\partial z} = \frac{1}{f[z]}\frac{\partial f[z]}{\partial z},
\tag{19.26}
$$

由此得到更新：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot \frac{1}{I}\sum_{i=1}^{I}\frac{\partial\log\left[Pr(\boldsymbol{\tau}_i|\boldsymbol{\theta})\right]}{\partial\boldsymbol{\theta}}r[\boldsymbol{\tau}_i].
\tag{19.27}
$$

轨迹的对数概率 $\log[Pr(\boldsymbol{\tau}|\boldsymbol{\theta})]$ 由下式给出：

$$
\log[Pr(\boldsymbol{\tau}|\boldsymbol{\theta})] = \log\left[Pr(\mathbf{s}_1)\prod_{t=1}^{T}\pi[a_t|\mathbf{s}_t, \boldsymbol{\theta}]Pr(\mathbf{s}_{t+1}|\mathbf{s}_t, a_t)\right]
$$

$$
= \log\left[Pr(\mathbf{s}_1)\right] + \sum_{t=1}^{T}\log\left[\pi[a_t|\mathbf{s}_t, \boldsymbol{\theta}]\right] + \sum_{t=1}^{T}\log\left[Pr(\mathbf{s}_{t+1}|\mathbf{s}_t, a_t)\right],
\tag{19.28}
$$

注意到只有中间项依赖于 $\boldsymbol{\theta}$，我们可以将公式 19.27 的更新重写为：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot \frac{1}{I}\sum_{i=1}^{I}\sum_{t=1}^{T}\frac{\partial\log\left[\pi[a_{it}|\mathbf{s}_{it}, \boldsymbol{\theta}]\right]}{\partial\boldsymbol{\theta}}r[\boldsymbol{\tau}_i],
\tag{19.29}
$$

其中 $\mathbf{s}_{it}$ 是第 $i$ 个回合中时间 $t$ 的状态，$a_{it}$ 是第 $i$ 个回合中时间 $t$ 采取的动作。注意与状态演化 $Pr(\mathbf{s}_{t+1}|\mathbf{s}_t, a_t)$ 相关的项消失了。由此可知，这个参数更新不假设马尔可夫时间演化过程。

我们可以进一步注意到：

$$
r[\boldsymbol{\tau}_i] = \sum_{k=1}^{T}r_{i,k+1} = \sum_{k=1}^{t}r_{i,k+1} + \sum_{k=t}^{T}r_{i,k+1},
\tag{19.30}
$$

其中 $r_{it}$ 是第 $i$ 个回合中时间 $t$ 的奖励。可以证明（非显然地），第一项（时间 $t$ 之前的奖励）不影响从时间 $t$ 开始的更新，因此我们可以写为：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot \frac{1}{I}\sum_{i=1}^{I}\sum_{t=1}^{T}\frac{\partial\log\left[\pi[a_{it}|\mathbf{s}_{it}, \boldsymbol{\theta}]\right]}{\partial\boldsymbol{\theta}}\sum_{k=t}^{T}r_{i,k+1}.
\tag{19.31}
$$

### 19.5.2 REINFORCE 算法

**REINFORCE** 是一种早期的策略梯度算法，利用了上述结果并引入了折扣。它是一种蒙特卡洛方法，基于当前策略 $\pi[a|\mathbf{s}, \boldsymbol{\theta}]$ 生成回合 $\boldsymbol{\tau}_i = [\mathbf{s}_{i1}, a_{i1}, r_{i2}, \mathbf{s}_{i2}, a_{i2}, r_{i3}, \ldots, r_{iT}]$。对于离散动作，这个策略可以由神经网络 $\pi[\mathbf{s}, \boldsymbol{\theta}]$ 确定，它接收当前状态 $\mathbf{s}$ 并为每个可能的动作返回一个输出。这些输出通过 softmax 函数创建动作的分布，在每个时间步从中采样。

对于每个回合 $i$，我们遍历每个时间步 $t$ 并计算从时间 $t$ 开始的部分轨迹 $\boldsymbol{\tau}_{it}$ 的经验折扣回报：

$$
r[\boldsymbol{\tau}_{it}] = \sum_{k=t+1}^{T}\gamma^{k-t-1}r_{i,k},
\tag{19.32}
$$

然后我们对每条轨迹中的每个时间步 $t$ 更新参数：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot \frac{\partial\log\left[\pi_{a_{it}}[\mathbf{s}_{it}, \boldsymbol{\theta}]\right]}{\partial\boldsymbol{\theta}}r[\boldsymbol{\tau}_{it}] \qquad \forall\ i, t,
\tag{19.33}
$$

其中 $\pi_{a_t}[\mathbf{s}_t, \boldsymbol{\theta}]$ 是给定当前状态 $\mathbf{s}_t$ 和参数 $\boldsymbol{\theta}$，神经网络产生 $a_t$ 的概率，$\alpha$ 是学习率。

### 19.5.3 基线

策略梯度方法具有高方差；可能需要很多回合才能获得稳定的导数估计。减少方差的一种方法是从轨迹回报 $r[\boldsymbol{\tau}]$ 中减去一个**基线**（baseline）$b$：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot \frac{1}{I}\sum_{i=1}^{I}\sum_{t=1}^{T}\frac{\partial\log\left[\pi_{a_{it}}[\mathbf{s}_{it}, \boldsymbol{\theta}]\right]}{\partial\boldsymbol{\theta}}\left(r[\boldsymbol{\tau}_{it}] - b\right).
\tag{19.34}
$$

只要基线 $b$ 不依赖于动作：

$$
\mathbb{E}_{\boldsymbol{\tau}}\left[\sum_{t=1}^{T-1}\frac{\partial\log\left[\pi_{a_{it}}[\mathbf{s}_{it}, \boldsymbol{\theta}]\right]}{\partial\boldsymbol{\theta}} \cdot b\right] = 0,
\tag{19.35}
$$

期望值不会改变。然而，如果基线与增加不确定性的无关因素协变，则减去它可以降低方差（图 19.16）。这是**控制变量**（control variates）方法的一个特例。

> **图 19.16** 使用控制变量降低估计方差。a) 考虑从少量样本估计 $\mathbb{E}[a]$。估计值（样本均值）会因样本数量和方差而变化。b) 现在考虑观察另一个与 $a$ 协变的变量 $b$，且 $\mathbb{E}[b] = 0$，方差与 $a$ 相同。c) $a - b$ 的样本方差远小于 $a$，但期望值 $\mathbb{E}[a-b] = \mathbb{E}[a]$，所以我们得到了一个方差更低的估计量。

这引出了一个问题：我们应该如何选择 $b$。我们可以找到使方差最小化的 $b$ 值，对方差写出关于 $b$ 的表达式，取导数令其为零，求解得到：

$$
b = \frac{\sum_i\sum_{t=1}^{T}\left(\partial\log\left[\pi_{a_{it}}[\mathbf{s}_{it}, \boldsymbol{\theta}]\right]/\partial\boldsymbol{\theta}\right)^2 r[\boldsymbol{\tau}_{it}]}{\sum_i\sum_{t=1}^{T}\left(\partial\log\left[\pi_{a_{it}}[\mathbf{s}_{it}, \boldsymbol{\theta}]\right]/\partial\boldsymbol{\theta}\right)^2}.
\tag{19.36}
$$

在实践中，这通常被近似为：

$$
b = \frac{1}{I}\sum_i r[\boldsymbol{\tau}_i].
\tag{19.37}
$$

减去这个基线可以消除以下情况引起的方差：所有轨迹的回报 $r[\boldsymbol{\tau}_i]$ 都高于典型值，这仅仅是因为它们碰巧经过了具有较高平均回报的状态，而与采取**什么**动作无关。

### 19.5.4 状态相关基线

一个更好的选择是使用依赖于当前状态 $\mathbf{s}_{it}$ 的基线 $b[\mathbf{s}_{it}]$。

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot \frac{1}{I}\sum_{i=1}^{I}\sum_{t=1}^{T}\frac{\partial\log\left[\pi_{a_{it}}[\mathbf{s}_{it}, \boldsymbol{\theta}]\right]}{\partial\boldsymbol{\theta}}\left(r[\boldsymbol{\tau}_{it}] - b[\mathbf{s}_{it}]\right).
\tag{19.38}
$$

在这里，我们补偿的是由于某些状态具有更大的总体回报而引入的方差，而不管我们采取什么动作。

一个明智的选择是基于当前状态的期望未来奖励，即状态值 $v[\mathbf{s}]$。在这种情况下，经验观测的回报与基线之间的差异被称为**优势估计**（advantage estimate）。由于我们处于蒙特卡洛上下文中，这可以由参数为 $\boldsymbol{\phi}$ 的神经网络 $b[\mathbf{s}] = v[\mathbf{s}, \boldsymbol{\phi}]$ 参数化，我们可以使用最小二乘损失将其拟合到观测回报：

$$
L[\boldsymbol{\phi}] = \sum_{i=1}^{I}\sum_{t=1}^{T}\left(v[\mathbf{s}_{it}, \boldsymbol{\phi}] - \sum_{j=t}^{T}r_{i,j+1}\right)^2.
\tag{19.39}
$$

## 19.6 Actor-Critic 方法

Actor-Critic 算法是时序差分（TD）策略梯度算法。它们可以在每一步更新策略网络的参数。这与蒙特卡洛 REINFORCE 算法形成对比，后者**必须**等待一个或多个回合完成后才能更新参数。

在 TD 方法中，我们无法获得未来奖励 $r[\boldsymbol{\tau}_t] = \sum_{k=t}^{T}r_k$。Actor-Critic 算法用当前观测奖励加上下一个状态的折扣值来近似所有未来奖励之和：

$$
r[\boldsymbol{\tau}_{it}] \approx r_{i,t+1} + \gamma \cdot v[\mathbf{s}_{i,t+1}, \boldsymbol{\phi}].
\tag{19.40}
$$

这里值 $v[\mathbf{s}_{i,t+1}, \boldsymbol{\phi}]$ 由参数为 $\boldsymbol{\phi}$ 的第二个神经网络估计。将其代入公式 19.38 给出更新：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \alpha \cdot \frac{1}{I}\sum_{i=1}^{I}\sum_{t=1}^{T}\frac{\partial\log\left[Pr(a_{it}|\mathbf{s}_{it}, \boldsymbol{\theta})\right]}{\partial\boldsymbol{\theta}}\left(r_{i,t+1} + \gamma \cdot v[\mathbf{s}_{i,t+1}, \boldsymbol{\phi}] - v[\mathbf{s}_{i,t}, \boldsymbol{\phi}]\right).
\tag{19.41}
$$

同时，我们通过自举更新参数 $\boldsymbol{\phi}$，使用损失函数：

$$
L[\boldsymbol{\phi}] = \sum_{i=1}^{I}\sum_{t=1}^{T}\left(r_{i,t+1} + \gamma \cdot v[\mathbf{s}_{i,t+1}, \boldsymbol{\phi}] - v[\mathbf{s}_{i,t}, \boldsymbol{\phi}]\right)^2.
\tag{19.42}
$$

预测 $Pr(a|\mathbf{s}_t)$ 的策略网络 $\pi[\mathbf{s}_t, \boldsymbol{\theta}]$ 被称为 **Actor**（演员）。值网络 $v[\mathbf{s}_t, \boldsymbol{\phi}]$ 被称为 **Critic**（评论家）。通常，同一个网络同时表示 Actor 和 Critic，有两组输出分别预测策略和值。注意，虽然 Actor-Critic 方法可以在每一步更新策略参数，但在实践中很少这样做。智能体通常收集大量经验后才更新策略。

> **图 19.17** 决策 Transformer。决策 Transformer 将离线强化学习视为序列预测任务。输入是状态、动作和剩余回报（回合中剩余的奖励）的序列，每个都映射到固定大小的嵌入。在每个时间步，网络预测下一个动作。在测试时，剩余回报是未知的；实际中使用一个初始估计，然后减去后续观测到的奖励。

## 19.7 离线强化学习

与环境的交互是强化学习的核心。然而，有些场景下将一个幼稚的智能体送入环境中探索不同动作的效果是不切实际的。这可能是因为环境中的错误行为是危险的（例如，驾驶自动驾驶汽车），或者因为数据收集耗时或成本高昂（例如，进行金融交易）。

然而，在这两种情况下都可以从人类代理中收集历史数据。**离线强化学习**（offline RL）或**批量强化学习**（batch RL）旨在通过观察过去的序列 $\mathbf{s}_1, a_1, r_2, \mathbf{s}_2, a_2, r_3, \ldots$ 学习如何采取能最大化未来回合奖励的动作，而无需与环境交互。它区别于**模仿学习**（imitation learning），后者 (i) 无法获得奖励且 (ii) 试图复制而非改进历史代理的表现。

虽然有基于 Q 学习和策略梯度的离线强化学习方法，但这一范式开启了新的可能性。特别是，我们可以将其视为一个序列学习问题，其中目标是在给定状态、奖励和动作的历史的情况下预测下一个动作。**决策 Transformer**（decision transformer）利用 Transformer 解码器框架（第 12.7 节）来做出这些预测（图 19.17）。

然而，目标是基于**未来奖励**来预测动作，而这些在标准的 $\mathbf{s}, a, r$ 序列中是无法获得的。因此，决策 Transformer 将奖励 $r_t$ 替换为**剩余回报**（returns-to-go）$R_{t:T} = \sum_{t'=t}^{T}r_{t'}$（即经验观测的未来奖励之和）。其余框架与标准 Transformer 解码器非常相似。状态、动作和剩余回报通过学习的映射转换为固定大小的嵌入。对于 Atari 游戏，状态嵌入可以通过类似于图 19.14 的卷积网络转换。动作和剩余回报的嵌入可以用与词嵌入（图 12.9）相同的方式学习。Transformer 使用掩码自注意力和位置嵌入进行训练。

这种方法在训练时很自然，但在推理时会遇到困难，因为我们不知道剩余回报。这可以通过使用期望的总回报在第一步初始化，然后随着奖励的获得递减来解决。例如，在 Atari 游戏中，期望的总回报就是获胜所需的总分。

决策 Transformer 还可以从在线经验中进行微调，因此可以随时间学习。它们的优势在于可以摒弃大部分强化学习机制及其相关的不稳定性，用标准的监督学习替代。Transformer 可以从海量数据中学习并整合长时间上下文中的信息（使时间信用分配问题更加可控）。这代表了强化学习的一个引人入胜的新方向。

## 19.8 总结

强化学习是一种针对马尔可夫决策过程和类似系统的顺序决策框架。本章回顾了表格方法的强化学习，包括动态规划（环境模型已知的情况）、蒙特卡洛方法（运行多个回合，随后根据获得的奖励更新动作值和策略）以及时序差分方法（在回合进行中更新值）。

深度 Q 学习是一种时序差分方法，使用深度神经网络预测每个状态的动作值。它可以训练智能体在 Atari 2600 游戏上达到接近人类的水平。策略梯度方法直接优化策略而非为动作赋值。它们产生随机策略，这在环境部分可观测时非常重要。更新是有噪声的，已经引入了许多改进来降低其方差。

离线强化学习用于无法与环境交互但必须从历史数据中学习的场景。决策 Transformer 利用深度学习的最新进展来构建状态-动作-奖励序列的模型，并预测能最大化奖励的动作。
