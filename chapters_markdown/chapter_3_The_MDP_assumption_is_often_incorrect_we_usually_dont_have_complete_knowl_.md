# Chapter 3: The MDP assumption is often incorrect; we usually don’t have complete knowl-

*Pages: 403-415*

---

19.5 Policy gradient methods 389
The underlying problem is that the same network both selects the target (by the
maximizationoperation)andupdatesthevalue. DoubleQ-Learningtacklesthisproblem
by training two models q [s ,a ,π ] and q [s ,a ,π ] simultaneously:
1 t t 1 2 t t 2
(cid:16) (cid:20) h i(cid:21) (cid:17)
q [s ,a ] ← q [s ,a ]+α r[s ,a ]+γ·q s ,argmax q [s ,a] −q [s ,a ]
1 t t 1 t t t t 2 t+1 1 t+1 1 t t
(cid:16) (cid:20) a h i(cid:21) (cid:17)
q [s ,a ] ← q [s ,a ]+α r[s ,a ]+γ·q s ,argmax q [s ,a] −q [s ,a ] .
2 t t 2 t t t t 1 t+1 2 t+1 2 t t
a
(19.20)
Now the choice of the target and the target itself are decoupled, which helps prevent
these biases. In practice, new tuples <s,a,r,s′> are randomly assigned to update one
modeloranother. ThisisknownasdoubleQ-learning. DoubledeepQ-networksordouble
DQNs use deep networks q[s ,a ,ϕ ] and q[s ,a ,ϕ ] to estimate the action values, and
t t 1 t t 2
the update becomes:
(cid:18) (cid:20) h i (cid:21) (cid:19)
∂q[s ,a ,ϕ ]
ϕ ←ϕ +α r[s ,a ]+γ·q s ,argmax q[s ,a,ϕ ] ,ϕ −q[s ,a ,ϕ ] t t 1
1 1 t t t+1 t+1 1 2 t t 1 ∂ϕ
(cid:18) (cid:20) a h i (cid:21) (cid:19) 1
∂q[s ,a ,ϕ ]
ϕ ←ϕ +α r[s ,a ]+γ·q s ,argmax q[s ,a,ϕ ] ,ϕ −q[s ,a ,ϕ ] t t 2 .
2 2 t t t+1 t+1 2 1 t t 2 ∂ϕ
a 2
(19.21)
19.5 Policy gradient methods
Q-learning estimates the action values first and then uses these to update the policy.
Conversely, policy-based methods directly learn a stochastic policy π[a |s ,θ]. This is a
t t
functionwithtrainableparametersθthatmapsastates toadistributionPr(a |s )over
t t t
actionsa fromwhichwecansample. InMDPs,thereisalwaysanoptimaldeterministic
t
policy. However, there are three reasons to use a stochastic policy:
1. Astochasticpolicynaturallyhelpswithexplorationofthespace;wearenotobliged
to take the best action at each time step.
2. Thelosschangessmoothlyaswemodifyastochasticpolicy. Thismeanswecanuse
gradient descent methods even though the rewards are discrete. This is similar to
using maximum likelihood in (discrete) classification problems. The loss changes
smoothly as the model parameters change to make the true class more likely.
3. The MDP assumption is often incorrect; we usually don’t have complete knowl-
edge of the state. For example, consider an agent navigating in an environment
where it can only observe nearby locations (e.g., figure 19.4). If two locations look
identical, but the nearby reward structure is different, a stochastic policy allows
the possibility of taking different actions until this ambiguity is resolved.
Draft: please send errata to udlbookmail@gmail.com.

390 19 Reinforcement learning
19.5.1 Derivation of gradient update
Consider a trajectory τ =[s ,a ,s ,a ,...,s ,a ] through an MDP. The probability of
1 1 2 2 T T
thistrajectoryPr(τ|θ)dependsonboththestateevolutionfunctionPr(s |s ,a )and
t+1 t t
the current stochastic policy π[a |s ,θ]:
t t
YT
Pr(τ|θ) = Pr(s ) π[a |s ,θ]Pr(s |s ,a ). (19.22)
1 t t t+1 t t
t=1
Policy gradient algorithms aim to maximize the expected return r[τ] over many such
trajectories:
(cid:20) h i(cid:21) (cid:20)Z (cid:21)
θ =argmax E r[τ] =argmax Pr(τ|θ)r[τ]dτ , (19.23)
τ
θ θ
where the return is the sum of all the rewards received along the trajectory.
To maximize this quantity, we use the gradient ascent update:
Z
∂
θ ← θ+α· Pr(τ|θ)r[τ]dτ
∂θ
Z
∂Pr(τ|θ)
= θ+α· r[τ]dτ, (19.24)
∂θ
where α is the learning rate.
We want to approximate this integral with a sum over empirically observed trajecto-
ries. These are drawn from the distribution Pr(τ|θ), so to make progress, we multiply
and divide the integrand by this distribution:
Z
∂Pr(τ|θ)
θ ← θ+α· r[τ]dτ
∂θ
Z
1 ∂Pr(τ|θ)
= θ+α· Pr(τ|θ) r[τ]dτ
Pr(τ|θ) ∂θ
1 XI 1 ∂Pr(τ |θ)
≈ θ+α· i r[τ ]. (19.25)
I Pr(τ |θ) ∂θ i
i
i=1
This equation has a simple interpretation (figure 19.15); the update changes the pa-
rametersθ toincreasethelikelihoodPr(τ |θ)ofanobservedtrajectoryτ inproportion
i i
to the reward r[τ ] from that trajectory. However, it also normalizes by the probabil-
i
ity of observing that trajectory in the first place to compensate for the fact that some
trajectories are observed more often than others. If a trajectory is already common and
yields high rewards, then we don’t need to change much. The biggest updates will come
from trajectories that are uncommon but create large rewards.
We can simplify this expression using the likelihood ratio identity:
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

19.5 Policy gradient methods 391
Figure 19.15 Policy gradients. Five
episodesforthesamepolicy(brighterin-
dicates higher reward). Trajectories 1,
2, and 3 generate consistently high re-
wards, but similar trajectories already
frequently occur with this policy, so
there is no need to change. Conversely,
trajectory 4 receives low rewards, so the
policy should be modified to avoid pro-
ducing similar trajectories. Trajectory 5
receives high rewards and is unusual.
Thiswillcausethelargestchangetothe
policy under equation 19.25.
∂log[f[z]] 1 ∂f[z]
= , (19.26)
∂z f[z] ∂z
which yields the update:
(cid:2) (cid:3)
1 XI ∂log Pr(τ |θ)
θ ←θ+α· i r[τ ]. (19.27)
I ∂θ i
i=1
The log probability log[Pr(τ|θ)] of a trajectory is given by:
h YT i
log[Pr(τ|θ)] = log Pr(s ) π[a |s ,θ]Pr(s |s ,a ) (19.28)
1 t t t+1 t t
t=1
(cid:2) (cid:3) XT (cid:2) (cid:3) XT (cid:2) (cid:3)
= log Pr(s ) + log π[a |s ,θ] + log Pr(s |s ,a ) ,
1 t t t+1 t t
t=1 t=1
and noting that only the center term depends on θ, we can rewrite the update from
equation 19.27 as:
(cid:2) (cid:3)
1 XI XT ∂log π[a |s ,θ]
θ ← θ+α· it it r[τ ], (19.29)
I ∂θ i
i=1 t=1
where s is the state at time t in episode i, and a is the action taken at time t in
it it
episode i. Note that the terms relating to the state evolution Pr(s |s ,a ) disappear
t+1 t t
from this formulation. It follows that this parameter update does not assume a Markov
time evolution process.
We can further simplify this by noting that:
XT Xt XT
r[τ ]= r = r + r , (19.30)
i i,t+1 i,k+1 i,k+1
t=1 k=1 k=t
Draft: please send errata to udlbookmail@gmail.com.

392 19 Reinforcement learning
where r is the reward at time t in the ith episode. It can (non-obviously) be proved
it
thatthefirstterm(therewardsbeforetimet)doesnotaffecttheupdatefromtime t, so
we can write:
(cid:2) (cid:3)
1 XI XT ∂log π[a |s ,θ] XT
θ ← θ+α· it it r . (19.31)
I ∂θ i,k+1
i=1 t=1 k=t
19.5.2 REINFORCE algorithm
REINFORCE is an early policy gradient algorithm that exploits this result and in-
corporates discounting. It is a Monte Carlo method that generates episodes τ =
i
[s ,a ,r ,s ,a ,r ,...,r ] based on the current policy π[a|s,θ]. For discrete ac-
i1 i1 i2 i2 i2 i3 iT
tions, this policy could be determined by a neural network π[s|θ], which takes the cur-
rent state s and returns one output for each possible action. These outputs are passed
through a softmax function to create a distribution over actions, which is sampled at
each time step.
Foreachepisodei,weloopthrougheachsteptandcalculatetheempiricaldiscounted
return for the partial trajectory τ that starts at time t:
it
XT
r[τ ]=
γk−1r
, (19.32)
it i,k
k=t+1
and then we update the parameters for each time step t in each trajectory:
(cid:2) (cid:3)
∂log π [s ,θ]
θ ←θ+α· ait it r[τ ] ∀i,t, (19.33)
∂θ it
whereπ [s ,θ]istheprobabilityofa producedbytheneuralnetworkgiventhecurrent
at t t
state s and parameters θ, and α is the learning rate.
t
19.5.3 Baselines
Policy gradient methods exhibit high variance; many episodes may be needed to get
stable estimates of the derivatives. One way to reduce this variance is to subtract a
baseline b from the trajectory returns r[τ]:
(cid:2) (cid:3)
1
XI XT
∂log π [s ,θ]
θ ←θ+α· ait it (r[τ ]−b). (19.34)
I ∂θ it
i=1 t=1
Problem19.6 As long as the baseline b doesn’t depend on the actions:
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

19.5 Policy gradient methods 393
Figure19.16Decreasingvarianceofestimatesusingcontrolvariates. a)Consider
tryingtoestimateE[a]fromasmallnumberofsamples. Theestimate(themean
ofthesamples)willvarybasedonthenumberofsamplesandthevarianceofthose
samples. b) Now consider observing another variable b that co-varies with a and
has E[b]=0 and the same variance as a. c) The variance of the samples of a−b
is much less than that of a, but the expected value E[a−b]=E[a], so we get an
estimator with lower variance.
" (cid:2) (cid:3) #
XT
∂log π [s ,θ]
E ait it ·b =0, (19.35)
τ ∂θ
t=1
andtheexpectedvaluewillnotchange. However,ifthebaselineco-varieswithirrelevant Notebook19.5
Controlvariates
factorsthatadduncertainty,thensubtractingitreducesthevariance(figure19.16). This
is a special case of the method of control variates (see problem 19.7).
Problem19.7
This raises the question of how we should choose b. We can find the value of b that
minimizes the variance by writing an expression for the variance, taking the derivative
with respect to b, setting the result to zero, and solving to yield:
Problem19.8
P (cid:0) (cid:2) (cid:3) (cid:1)
X T ∂log π [s ,θ] /∂θ 2 r[τ ]
b= tP=1 (cid:0) a(cid:2)it it (cid:3) (cid:1) it . (19.36)
T ∂log π [s ,θ] /∂θ 2
i t=1 ait it
In practice, this is often approximated as:
X
1
b= r[τ ]. (19.37)
I i
i
Subtracting this baseline factors out variance that might occur when the returns r[τ ]
i
from all trajectories are greater than is typical but only because they happen to pass
through states with higher than average returns whatever actions are taken.
Draft: please send errata to udlbookmail@gmail.com.

394 19 Reinforcement learning
19.5.4 State-dependent baselines
A better option is to use a baseline b[s ] that depends on the current state s .
it it
(cid:2) (cid:3)
1
XI XT
∂log π [s ,θ]
θ ←θ+α· ait it (r[τ ]−b[s ]). (19.38)
I ∂θ it it
i=1 t=1
Here,wearecompensatingforvarianceintroducedbysomestateshavinggreateroverall
returns than others, whichever actions we take.
A sensible choice is the expected future reward based on the current state, which is
justthestatevaluev[s]. Inthiscase, thedifferencebetweentheempiricallyobservedre-
wardsandthebaselineisknownastheadvantageestimate. SinceweareinaMonteCarlo
context,thiscanbeparameterizedbyaneuralnetworkb[s]=v[s,ϕ]withparametersϕ,
which we can fit to the observed returns using least squares loss:
0 1
2
XI XT XT
L[ϕ]= @ v[s ,ϕ]− r A . (19.39)
it i,j+1
i=1 t=1 j=t
19.6 Actor-critic methods
Actor-critic algorithms are temporal difference (TD) policy gradient algorithms. They
can update the parameters of the policy network at each step. This contrasts with
the Monte Carlo REINFORCE algorithm, which must wait for one or more episodes to
complete before updating the parameters. P
In the TD approach, we do not have access to the future rewards r[τ ] = T r
t k=t k
along this trajectory. Actor-critic algorithms approximate the sum over all the future
rewards with the observed current reward plus the discounted value of the next state:
r[τ ]≈r +γ·v[s ,ϕ]. (19.40)
it i,t+1 i,t+1
Here the value v[s ,ϕ] is estimated by a second neural network with parameters ϕ.
i,t+1
Substituting this into equation 19.38 gives the update:
(cid:2) (cid:3)
1 XI XT ∂log Pr(a |s ,θ)] (cid:16) (cid:17)
θ ←θ+α· it it r +γ·v[s ,ϕ]−v[s ,ϕ] . (19.41)
I ∂θ i,t+1 i,t+1 i,t
i=1 t=1
Concurrently, we update the parameters ϕ by bootstrapping using the loss function:
XI XT
L[ϕ]= (r +γ·v[s ,ϕ]−v[s ,ϕ])2. (19.42)
i,t+1 i,t+1 i,t
i=1 t=1
The policy network π[s ,θ] that predicts Pr(a|s ) is termed the actor. The value
t t
network v[s ,ϕ] is termed the critic. Often the same network represents both actor and
t
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

19.7 Offline reinforcement learning 395
Figure 19.17 Decision transformer. The decision transformer treats offline rein-
forcementlearningasasequencepredictiontask. Theinputisasequenceofstates,
actions, and returns-to-go (remaining rewards in the episode), each of which is
mapped to a fixed-size embedding. At each time step, the network predicts the
nextaction. Duringtesting,thereturns-to-goareunknown;inpractice,aninitial
estimate is made from which subsequent observed rewards are subtracted.
the critic, with two sets of outputs that predict the policy and the values, respectively.
Note that although actor-critic methods can update the policy parameters at each step,
this is rarely done in practice. The agent typically collects a batch of experience over
many time steps before the policy is updated.
19.7 Offline reinforcement learning
Interactionwiththeenvironmentisatthecoreofreinforcementlearning. However,there
are some scenarios where it is not practical to send a naïve agent into an environment
to explore the effect of different actions. This may be because erratic behavior in the
environment is dangerous (e.g., driving autonomous vehicles) or because data collection
is time-consuming or expensive (e.g., making financial trades).
However, it is possible to gather historical data from human agents in both cases.
OfflineRLorbatchRLaimstolearnhowtotakeactionsthatmaximizerewardsonfuture
episodes by observing past sequences s ,a ,r ,s ,a ,r ,..., without ever interacting
1 1 2 2 2 3
with the environment. It is distinct from imitation learning, a related technique that (i)
does not have access to the rewards and (ii) attempts to replicate the performance of a
historical agent rather than improve it.
Although there are offline RL methods based on Q-Learning and policy gradients,
Draft: please send errata to udlbookmail@gmail.com.

396 19 Reinforcement learning
this paradigm opens up new possibilities. In particular, we can treat this as a sequence
learning problem, in which the goal is to predict the next action, given the history of
states, rewards, and actions. The decision transformer exploits a transformer decoder
framework (section 12.7) to make these predictions (figure 19.17).
However, the goal is to predict actions based on future rewards, and these are not
captured in a standard s,a,r sequenceP. Hence, the decision transformer replaces the re-
wardr t withthereturns-to-goR t:T = T t′=t r t′ (i.e.,thesumoftheempiricallyobserved
future rewards). The remaining framework is very similar to a standard transformer
decoder. The states, actions, and returns-to-go are converted to fixed-size embeddings
via learned mappings. For Atari games, the state embedding might be converted via a
convolutional network similar to that in figure 19.14. The embeddings for the actions
andreturns-to-gocanbelearnedinthesamewayaswordembeddings(figure12.9). The
transformer is trained with masked self-attention and position embeddings.
This formulation is natural during training but poses a quandary during inference
becausewedon’tknowthereturns-to-go. Thiscanberesolvedbyusingthedesiredtotal
return at the first step and decrementing this as rewards are received. For example, in
an Atari game, the desired total return would be the total score required to win.
Decision transformers can also be fine-tuned from online experience and hence learn
over time. They have the advantage of dispensing with most of the reinforcement learn-
ing machinery and its associated instability and replacing this with standard supervised
learning. Transformers can learn from enormous quantities of data and integrate infor-
mationacrosslargetimecontexts(makingthetemporalcreditassignmentproblemmore
tractable). This represents an intriguing new direction for reinforcement learning.
19.8 Summary
Reinforcement learning is a sequential decision-making framework for Markov decision
processes and similar systems. This chapter reviewed tabular approaches to RL, includ-
ing dynamic programming (in which the environment model is known), Monte Carlo
methods (in which multiple episodes are run and the action values and policy subse-
quently changed based on the rewards received), and temporal difference methods (in
which these values are updated while the episode is ongoing).
Deep Q-Learning is a temporal difference method where deep neural networks are
used to predict the action value for every state. It can train agents to perform Atari
2600 games at a level similar to humans. Policy gradient methods directly optimize the
policy rather than assigning values to actions. They produce stochastic policies, which
areimportantwhen theenvironmentis partiallyobservable. Theupdatesare noisy, and
many refinements have been introduced to reduce their variance.
Offline reinforcement learning is used when we cannot interact with the environment
but must learn from historical data. The decision transformer leverages recent progress
in deep learning to build a model of the state-action-reward sequence and predict the
actions that will maximize the rewards.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 397
Notes
Sutton & Barto (2018) cover tabular reinforcement learning methods in depth. Li (2017),
Arulkumaran et al. (2017), François-Lavet et al. (2018), and Wang et al. (2022c) all provide
overviews of deep reinforcement learning. Graesser & Keng (2019) is an excellent introductory
resource that includes Python code.
Landmarksindeepreinforcementlearning: Mostlandmarkachievementsofreinforcement
learninghavebeenineithervideogamesorreal-worldgamessincetheseprovideconstraineden-
vironmentswithlimited actionsand fixedrules. DeepQ-Learning(Mnihetal.,2015)achieved
human-level performance across a benchmark of ATARI games. AlphaGo (Silver et al., 2016)
beattheworldchampionatGo. Thisgamewaspreviouslyconsideredverydiﬀicultforcomput-
erstoplay. Berneretal.(2019)builtasystemthatbeattheworldchampionteaminthefivevs.
five-playergameDefense of the Ancients 2,whichrequirescooperationacrossplayers. Yeetal.
(2021)builtasystemthatcouldbeathumansonAtarigameswithlimiteddata(incontrastto
previous systems, which need much more experience than humans). More recently, the Cicero
system demonstrated human-level performance in the game Diplomacy which requires natural
language negotiations and coordination between players (FAIR, 2022).
RLhasalsobeenappliedsuccessfullytocombinatorialoptimizationproblems(seeMazyavkina
et al., 2021). For example, Kool et al. (2019) learned a model that performed similarly to the
best heuristics for the traveling salesman problem. Recently, AlphaTensor (Fawzi et al., 2022)
treatedmatrixmultiplicationasagameandlearnedfasterwaystomultiplymatricesusingfewer
multiplication operations. Since deep learning relies heavily on matrix multiplication, this is
one of the first examples of self-improvement in AI.
Classicalreinforcementlearningmethods: VeryearlycontributionstothetheoryofMDPs
weremadebyThompson(1933)andThompson(1935). TheBellmanrecursionswereintroduced
byBellman(1966). Howard(1960)introducedpolicyiteration. Sutton&Barto(2018)identify
the work of Andreae (1969) as being the first to describe RL using the MDP formalism.
The modern era of reinforcement learning arguably originated in the Ph.D. theses of Sutton
(1984) and Watkins (1989). Sutton (1988) introduced the term temporal difference learning.
Watkins (1989) and Watkins & Dayan (1992) introduced Q-Learning and showed that it con-
verges to a fixed point by Banach’s theorem because the Bellman operator is a contraction
mapping. Watkins (1989) made the first explicit connection between dynamic programming
and reinforcement learning. SARSA was developed by Rummery & Niranjan (1994). Gordon
(1995) introduced fitted Q-learning in which a machine learning model is used to predict the
action value for each state-action pair. Riedmiller (2005) introduced neural-fitted Q-learning,
which used a neural network to predict all the action values at once from a state. Early work
on Monte Carlo methods was carried out by Singh & Sutton (1996), and the exploring starts
algorithm was introduced by Sutton & Barto (1999). Note that this is an extremely cursory
summary of more than fifty years of work. A much more thorough treatment can be found in
Sutton & Barto (2018).
DeepQ-Networks: DeepQ-LearningwasdevisedbyMnihetal.(2015)andisanintellectual
descendent of neural-fitted Q-learning. It exploited the then-recent successes of convolutional
networks to develop a fitted Q-Learning method that could achieve human-level performance
onabenchmarkofATARIgames. DeepQ-Learningsuffersfromthedeadly triad issue(Sutton
&Barto,2018): trainingcanbeunstableinanyschemethatincorporates(i)bootstrapping,(ii)
off-policylearning,and(iii)functionapproximation. Muchsubsequentworkhasaimedtomake
training more stable. Mnih et al. (2015) introduced the experience replay buffer (Lin, 1992),
which was subsequently improved by Schaul et al. (2016) to favor more important tuples and
hence increase learning speed. This is termed prioritized experience replay.
Draft: please send errata to udlbookmail@gmail.com.

398 19 Reinforcement learning
The original Q-Learning paper concatenated four frames so the network could observe the
velocities of objects and make the underlying process closer to fully observable. Hausknecht &
Stone(2015)introduceddeeprecurrentQ-learning,whichusedarecurrentnetworkarchitecture
that only ingested a single frame at a time because it could “remember” the previous states.
Van Hasselt (2010) identified the systematic overestimation of the state values due to the max
operation and proposed double Q-Learning in which two models are trained simultaneously to
remedy this. This was subsequently applied in the context of deep Q-learning (Van Hasselt
et al., 2016), although its eﬀicacy has since been questioned (Hessel et al., 2018). Wang et al.
(2016) introduced deep dueling networks in which two heads of the same network predict (i)
thestatevalueand(ii)theadvantage(relativevalue)ofeachaction. Theintuitionhereisthat
sometimes it is the state value that is important, and it doesn’t matter much which action is
taken, and decoupling these estimates improves stability.
Fortunato et al. (2018) introduced noisy deep Q-Networks, in which some weights in the Q-
Network are multiplied by noise to add stochasticity to the predictions and encourage explo-
ration. Thenetworkcanlearntodecreasethemagnitudesofthenoiseovertimeasitconverges
to a sensible policy. Distributional DQN (Bellemare et al., 2017a; Dabney et al., 2018 follow-
ing Morimura et al., 2010) aims to estimate more complete information about the distribution
of returns than just the expectation. This potentially allows the network to mitigate against
worst-caseoutcomesandcanalsoimproveperformance,aspredictinghighermomentsprovides
arichertrainingsignal. Rainbow(Hesseletal.,2018)combinedsiximprovementstotheoriginal
deep Q-learning algorithm, including dueling networks, distributional DQN, and noisy DQN,
to improve both the training speed and the final performance on the ATARI benchmark.
Policygradients: Williams(1992)introducedtheREINFORCEalgorithm. Theterm“policy
gradientmethod”datestoSuttonetal.(1999). Konda&Tsitsiklis(1999)introducedtheactor-
criticalgorithm. DecreasingthevariancebyusingdifferentbaselinesisdiscussedinGreensmith
et al. (2004) and Peters & Schaal (2008). It has since been argued that the value baseline
primarilyreducestheaggressivenessoftheupdatesratherthantheirvariance(Meietal.,2022).
Policygradientshavebeenadaptedtoproducedeterministicpolicies(Silveretal.,2014;Lillicrap
et al., 2016; Fujimoto et al., 2018). The most direct approach is to maximize over the possible
actions, but if the action space is continuous, this requires an optimization procedure at each
step. The deep deterministic policy gradient algorithm (Lillicrap et al., 2016) moves the policy
inthedirectionofthegradientoftheactionvalue(implyingtheuseofanactor-criticmethod).
Modernpolicygradients: Weintroducedpolicygradientsintermsoftheparameterupdate.
However,theycanalsobeviewedasoptimizingasurrogatelossbasedonimportancesampling
oftheexpectedrewards,usingtrajectoriesfromthecurrentpolicyparameters. Thisviewallows
ustotakemultipleoptimizationstepsvalidly. However,thiscancauseverylargepolicyupdates.
Oversteppingisaminorprobleminsupervisedlearning,asthetrajectorycanbecorrectedlater.
However, in RL, it affects future data collection and can be extremely destructive.
Several methods have been proposed to moderate these updates. Natural policy gradients
(Kakade, 2001) are based on natural gradients (Amari, 1998), which modify the descent di-
rection by the Fisher information matrix. This provides a better update which is less likely to
get stuck in local plateaus. However, the Fisher matrix is impractical to compute in models
with many parameters. In trust-region policy optimization or TRPO (Schulman et al., 2015),
thesurrogateobjectiveismaximizedsubjecttoaconstraintontheKLdivergencebetweenthe
old and new policies. Schulman et al. (2017) propose a simpler formulation in which this KL
divergence appears as a regularization term. The regularization weight is adapted based on
the distance between the KL divergence and a target indicating how much we want the policy
to change. Proximal policy optimization or PPO (Schulman et al., 2017) is an even simpler
approach in which the loss is clipped to ensure smaller updates.
Actor-critic: Intheactor-criticalgorithm(Konda&Tsitsiklis,1999)describedinsection19.6,
the critic used a 1-step estimator. It’s also possible to use k-step estimators (in which we
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 399
observe k discounted rewards and approximate subsequent rewards with an estimate of the
state value). As k increases, the variance of the estimate increases, but the bias decreases.
Generalizedadvantageestimation(Schulmanetal.,2016)weightstogetherestimatesfrommany
stepsandparameterizestheweightingbyasingletermthattradesoffthebiasandthevariance.
Mnih et al. (2016) introduced asynchronous actor-critic or A3C in which multiple agents are
run independently in parallel environments and update the same parameters. Both the policy
and value function are updated every T time steps using a mix of k-step returns. Wang et al.
(2017) introduced several methods designed to make asynchronous actor-critic more eﬀicient.
Soft actor-critic (Haarnoja et al., 2018b) adds an entropy term to the cost function, which
encourages exploration and reduces overfitting as the policy is encouraged to be less confident.
Offline RL: In offline reinforcement learning, the policy is learned by observing the behavior
of other agents, including the rewards they receive, without the ability to change the policy. It
isrelatedtoimitationlearning,wherethegoalistocopythebehaviorofanotheragentwithout
access to rewards (see Hussein et al., 2017). One approach is to treat offline RL in the same
way as off-policy reinforcement learning. However, in practice, the distributional shift between
the observed and applied policy manifests in overly optimistic estimates of the action value
and poor performance (see Fujimoto et al., 2019; Kumar et al., 2019a; Agarwal et al., 2020).
Conservative Q-learning (Kumar et al., 2020b) learns conservative, lower-bound estimates of
the value function by regularizing the Q-values. The decision transformer (Chen et al., 2021c)
is a simple approach to offline learning that takes advantage of the well-studied self-attention
architecture. It can subsequently be fine-tuned with online training (Zheng et al., 2022).
Reinforcement learning and chatbots: Chatbots can be trained using a technique known
asreinforcementlearningwithhumanfeedbackorRLHF(Christianoetal.,2018;Stiennonetal.,
2020). Forexample,InstructGPT(theforerunnerofChatGPT,Ouyangetal.,2022)startswith
astandardtransformerdecodermodel. Thisisthenfine-tunedbasedonprompt-responsepairs
where the response was written by human annotators. During this training step, the model is
optimized to predict the next word in the ground truth response.
Unfortunately, such training data are expensive to produce in suﬀicient quantities to support
high-quality performance. To resolve this problem, human annotators then indicate which of
several model responses they prefer. These (much cheaper) data are used to train a reward
model. This is a second transformer network that ingests the prompt and model response and
returns a scalar indicating how good the response is. Finally, the fine-tuned chatbot model is
furthertrainedtoproducehighrewardsusingtherewardmodelassupervision. Here,standard
gradientdescentcannotbeusedasit’snotpossibletocomputederivativesthroughthesampling
procedureinthechatbotoutput. Hence,themodelistrainedwithproximalpolicyoptimization
(a policy gradient method where the derivatives are tractable) to generate higher rewards.
Other areas of RL: Reinforcement learning is an enormous area, which easily justifies its
own book, and this literature review is extremely superficial. Other notable areas of RL that
we have not discussed include model-based RL, in which the state transition probabilities and
reward functions are modeled (see Moerland et al., 2023). This allows forward planning and
has the advantage that the same model can be reused for different reward structures. Hybrid
methods such as AlphaGo (Silver et al., 2016) and MuZero (Schrittwieser et al., 2020) have
separate models for the dynamics of the states, the policy, and the value of future positions.
This chapter has only discussed simple methods for exploration, like the epsilon-greedy ap-
proach, noisy Q-learning, and adding an entropy term to penalize overconfident policies. In-
trinsicmotivationreferstomethodsthataddrewardsforexplorationandthusimbuetheagent
with“curiosity”(seeBarto,2013;Aubretetal.,2019). Hierarchical reinforcement learning(see
Pateriaetal.,2021)referstomethodsthatbreakdownthefinalobjectiveintosub-tasks. Multi-
agent reinforcement learning(seeZhangetal.,2021a)considersthecasewheremultipleagents
coexist in a shared environment. This may be in either a competitive or cooperative context.
Draft: please send errata to udlbookmail@gmail.com.

400 19 Reinforcement learning
Problems
Problem19.1Figure19.18showsasingletrajectorythroughanexampleMarkovrewardprocess.
Calculate the return for each step in the trajectory given that the discount factor γ is 0.9.
Problem 19.2∗ Prove the policy improvement theorem. Consider changing from policy π to
policy π′, where for state s the new policy π′ chooses the action that maximizes the expected
t
return:
(cid:20) (cid:21)
X
π ′ [a |s ]←argmax r[s ,a ]+γ· Pr(s |s ,a )v[s |π] . (19.43)
t t t t t+1 t t t+1
at
st+1
and for all other states, the policies are the same. Show that the value v[s |π] for the original
t
policy must be less than or equal to v[s |π′] for the new policy (notation indicates using π′ for
t
state s and π thereafter):
t
h (cid:12) i
(cid:12)
v[s |π] ≤ q s ,π ′ [a |s ](cid:12)π
t t t t
h i
= E π′ r t+1 +γ·v[s t+1 |π] . (19.44)
Hint: Start by writing the term v[s |π] in terms of the new policy.
t+1
Problem 19.3 Show that when the state values and policy are initialized as in figure 19.10a,
they become those in figure 19.10b after two iterations of (i) policy evaluation (in which all
states are updated based on their current values and then replace the previous ones) and (ii)
policy improvement. The state transition allots half the probability to the direction the policy
indicates and divides the remaining probability equally between the other valid actions. The
rewardfunctionreturns-2irrespectiveoftheactionwhenthepenguinleavesahole. Thereward
functionreturns+3regardlessoftheactionwhenthepenguinleavesthefishtileandtheepisode
ends, so the fish tile has a value of +3. Assume a discount factor of γ =0.9.
Problem 19.4 TheBoltzmann policystrikesabalancebetweenexplorationandexploitationby
basing the action probabilities π[a|s] on the current state-action reward function q[s,a]:
(cid:2) (cid:3)
exp q[s,a]/τ
π[a|s]= P (cid:2) (cid:3). (19.45)
exp q[s,a′]/τ
a′
Explainhowthetemperatureparameterτ canbevariedtoprioritizeexplorationorexploitation.
Problem 19.5∗ When the learning rate α is one, the Q-Learning update is given by:
(cid:2) (cid:3) (cid:2) (cid:3)
f q[s,a] =r[s,a]+γ·max q[s ′ ,a] . (19.46)
a
where s is a state and s′ is the subsequent state. Show that this is a contraction mapping
(equation 16.30) so that:
(cid:12)(cid:12) (cid:12)(cid:12) (cid:12)(cid:12) (cid:12)(cid:12)
(cid:12)(cid:12)(cid:2) (cid:3) (cid:2) (cid:3)(cid:12)(cid:12) (cid:12)(cid:12) (cid:12)(cid:12)
(cid:12) (cid:12) (cid:12) (cid:12)f q 1 [s,a] −f q 2 [s,a] (cid:12) (cid:12) (cid:12) (cid:12) < (cid:12) (cid:12) (cid:12) (cid:12)q 1 [s,a]−q 2 [s,a] (cid:12) (cid:12) (cid:12) (cid:12) ∀q 1 ,q 2 . (19.47)
∞ ∞
AppendixB.3.2 where||•|| ∞representstheℓ∞norm. ItfollowsthatafixedpointwillexistbyBanach’stheorem
Vectornorms and that the updates will eventually converge.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 401
Figure 19.18 One trajectory through an
MRP. The penguin receives a reward
of +1 when it reaches the first fish
tile,−2whenitfallsinthehole,and+1
forreachingthesecondfishtile. Thedis-
count factor γ is 0.9.
Problem 19.6 Show that: (cid:20) (cid:21)
(cid:2) (cid:3)
∂
E log Pr(τ|θ) b =0, (19.48)
τ ∂θ
where b does not depend on τ, so adding a baseline update doesn’t change the expected policy
gradient update.
Problem 19.7∗ Suppose that we want to estimate a quantity E[a] from samples a ,a ...a .
1 2 I
Consider that we also have paired samples b ,b ...b that are samples that co-vary with a
1 2 I
where E[b]=µ . We define a new variable:
b
a ′ =a−c(b−µ ). (19.49)
b
Show that Var[a′] ≤ Var[a] when the constant c is chosen judiciously. Find an expression for
the optimal value of c.
Problem 19.8 The estimate of the gradient in equation 19.34 can be written as:
h i
E g[θ](r[τ ]−b) , (19.50)
τ t
where
(cid:2) (cid:3)
XT ∂log Pr(a |s ,θ)]
t t
g[θ,τ]= , (19.51)
∂θ
t=1
and
XT
r[τ]= r . (19.52)
k
k=t
Show that the value of b that minimizes the variance of the gradient estimate is given by:
(cid:2) (cid:3)
E g[θ,τ]2r[τ]
b= (cid:2) (cid:3) . (19.53)
E g[θ,τ]2
You will need to use the result from equation 19.35.
Draft: please send errata to udlbookmail@gmail.com.