# Chapter 2: Policy estimation approaches directly estimate the optimal policy using a gradient

*Pages: 396-402*

---

382 19 Reinforcement learning
0 1
X X
v[s ]= π[a |s ] @ r[s ,a ]+γ· Pr(s |s ,a )v[s ] A . (19.9)
t t t t t t+1 t t t+1
at st+1
Similarly, substituting equation 19.7 into equation 19.8 provides a relation between the
action value at time t and t+1:
0 1
X X
q[s ,a ]=r[s ,a ]+γ· Pr(s |s ,a ) @ π[a |s ]q[s ,a ] A . (19.10)
t t t t t+1 t t t+1 t+1 t+1 t+1
st+1 at+1
The latter two relations are the Bellman equations and are the backbone of many
RL methods. In short, they say that the state (action) values have to be self-consistent.
Consequently, when we update an estimate of one state (action) value, this will have a
ripple effect that causes modifications to all the others.
19.3 Tabular reinforcement learning
TabularRLalgorithms(i.e.,thosethatdon’trelyonfunctionapproximation)aredivided
into model-based and model-free methods. Model-based methods4 use the MDP structure
explicitly and find the best policy from the transition matrix Pr(s |s ,a ) and reward
t+1 t t
structure r[s,a]. If these are known, this is a straightforward optimization problem that
canbetackledusingdynamic programming. Iftheyareunknown,theycan(inprinciple)
be estimated from observed MDP trajectories.5
Conversely, model-free methods assume that the transition matrix and reward struc-
ture of the underlying MDP are unknown. These methods fall into two families:
1. Value estimation approaches estimate the optimal state-action value function and
thenassignthepolicyaccordingtotheactionineachstatewiththegreatestvalue.
2. Policy estimation approaches directly estimate the optimal policy using a gradient
descenttechniquewithouttheintermediatestepsofestimatingthemodelorvalues.
Within each family, Monte Carlo methods simulate many trajectories through the MDP
for a given policy to gather information about how to improve this policy. Sometimes
it is not feasible or practical to simulate many trajectories before updating the policy.
Temporaldifference(TD)methodsupdatethepolicywhiletheagenttraversestheMDP.
We now briefly describe dynamic programming methods, Monte Carlo value esti-
mation methods, and TD value estimation methods. Section 19.4 describes how deep
networks have been used in TD value estimation methods. We return to policy estima-
tion in section 19.5.
4ThetermmodelrefersheretotheMDPandnotamachinelearningmodel.
5InRL,atrajectoryisanobservedsequenceofstates,rewards,andactions. Arolloutisasimulated
trajectory. Anepisodeisatrajectorythatstartsinaninitialstateandendsinaterminalstate(e.g.,a
fullgameofchessstartingfromthestandardopeningpositionandendinginawin,lose,ordraw.)
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

19.3 Tabular reinforcement learning 383
Figure 19.10 Dynamic programming. a) The state values are initialized to zero,
and the policy (arrows) is chosen randomly. b) The state values are updated to
be consistent with their neighbors (equation 19.11, shown after two iterations).
The policy is updated to move the agent to states with the highest value (equa-
tion 19.12). c) After several iterations, the algorithm converges to the optimal
policy, in which the penguin tries to avoid the holes and reach the fish.
19.3.1 Dynamic programming
Dynamic programming algorithms assume we have perfect knowledge of the transition
and reward structure. In this respect, they are distinguished from most RL algorithms
which observe the agent interacting with the environment to gather information about
these quantities indirectly.
The state values v[s] are initialized arbitrarily (usually to zero). The deterministic
policy π[a|s] is also initialized (e.g., by choosing a random action for each state). The
algorithm then alternates between iteratively computing the state values for the current
policy (policy evaluation) and improving that policy (policy improvement).
Policy evaluation: We sweep through the states s , updating their values:
t
0 1
X X
v[s ]← π[a |s ] @ r[s ,a ]+γ· Pr(s |s ,a )v[s ] A , (19.11)
t t t t t t+1 t t t+1
at st+1
where s is the successor state and Pr(s |s ,a ) is the state transition probability.
t+1 t+1 t t
Each update makes v[s ] consistent with the value at the successor state s using the
t t+1
Bellman equation for state values (equation 19.9). This is termed bootstrapping.
Policy improvement: To update the policy, we greedily choose the action that maxi-
mizes the value for each state:
(cid:20) (cid:21)
X
π[a |s ]←argmax r[s ,a ]+γ· Pr(s |s ,a )v[s ] . (19.12)
t t t t t+1 t t t+1
at
st+1
This is guaranteed to improve the policy according to the policy improvement theorem.
Draft: please send errata to udlbookmail@gmail.com.

384 19 Reinforcement learning
Figure 19.11 Monte Carlo methods. a) The policy (arrows) is initialized ran-
domly. The MDP is repeatedly simulated, and the trajectories of these episodes
are stored (orange and brown paths represent two trajectories). b) The action
values are empirically estimated based on the observed returns averaged over
these trajectories. In this case, the action values were all initially zero and have
been updated where an action was observed. c) The policy can then be updated
according to the action which received the best (or least bad) reward.
These two steps are iterated until the policy converges (figure 19.10).
Problems19.2–19.3
Therearemanyvariationsofthisapproach. Inpolicy iteration, thepolicyevaluation
stepisiterateduntilconvergencebeforepolicyimprovement. Thevaluescanbeupdated
either in place or synchronously in each sweep. In value iteration, the policy evaluation
Notebook19.2
proceduresweepsthroughthevaluesjustoncebeforepolicyimprovement. Asynchronous
Dynamic
programming dynamic programming algorithms don’t have to systematically sweep through all the
values at each step but can update a subset of the states in place in an arbitrary order.
19.3.2 Monte Carlo methods
Unlikedynamicprogrammingalgorithms,MonteCarlomethodsdon’tassumeknowledge
oftheMDP’stransitionprobabilitiesandrewardstructure. Instead,theygainexperience
by repeatedly sampling trajectories from the MDP and observing the rewards. They
alternate between computing the action values (based on this experience) and updating
the policy (based on the action values).
To estimate the action values q[s,a], a series of episodes are run. Each starts with
a given state and action and thereafter follows the current policy, producing a series of
actions, states, and rewards (figure 19.11a). The action value for a given state-action
pair under the current policy is estimated as the average of the empirical returns (i.e.,
cumulative sums of time-discounted rewards) that follow each time this pair occurs (fig-
ure19.11b). Thenthepolicyisupdatedbychoosingtheactionwiththemaximumvalue
at every state (figure 19.11c):
h i
π[a|s]←argmax q[s,a] . (19.13)
a
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

19.3 Tabular reinforcement learning 385
Thisisanon-policymethod;thecurrentbestpolicyisusedtoguidetheagentthrough
the environment. This policy is based on the observed action values in every state, but
of course, it’s not possible to estimate the value of actions that haven’t been used, and
there is nothing to encourage the algorithm to explore these. One solution is to use
exploring starts. Here,episodeswithallpossiblestate-actionpairsareinitiated,soevery
combination is observed at least once. However, this is impractical if the number of
states is large or the starting point cannot be controlled. A different approach is to
Problem19.4
use an epsilon greedy policy, in which a random action is taken with probability ϵ, and
the optimal action is allotted the remaining probability 1−ϵ. The choice of ϵ trades off
exploitation and exploration. Here, an on-policy method will seek the best policy from
this epsilon-greedy family, which will not generally be the best overall policy.
Conversely, in off-policy methods, the optimal policy π (the target policy) is learned
based on episodes generated by a different behavior policy π′. Typically, the target
policy is deterministic, and the behavior policy is stochastic (e.g., an epsilon-greedy
policy). Hence, the behavior policy can explore the environment, but the learned target
Notebook19.3
policy remains eﬀicient. Some off-policy methods explicitly use importance sampling
MonteCarlo
(section 17.8.1) to estimate the action value under policy π using samples from π′. methods
Others, such as Q-learning (described in the next section), estimate the values based
on the greedy action, even though this is not necessarily what was chosen.
19.3.3 Temporal difference methods
Dynamic programming methods use a bootstrapping process to update the values to
make them self-consistent under the current policy. Monte Carlo methods sample the
MDP to acquire information. Temporal difference (TD) methods combine both boot-
strapping and sampling. However, unlike Monte Carlo methods, they update the values
and policy while the agent traverses the states of the MDP instead of afterward.
SARSA (State-Action-Reward-State-Action) is an on-policy algorithm with update:
(cid:16) (cid:17)
q[s ,a ]←q[s ,a ]+α r[s ,a ]+γ·q[s ,a ]−q[s ,a ] , (19.14)
t t t t t t t+1 t+1 t t
where α ∈ R+ is the learning rate. The bracketed term is called the TD error and
measures the consistency between the estimated action value q[s ,a ] and the esti-
t t
mate r[s ,a ]+γ·q[s ,a ] after taking a single step.
t t t+1 t+1
By contrast, Q-Learning is an off-policy algorithm with update (figure 19.12):
(cid:16) (cid:2) (cid:3) (cid:17)
q[s ,a ]←q[s ,a ]+α r[s ,a ]+γ·max q[s ,a] −q[s ,a ] , (19.15)
t t t t t t t+1 t t
a
wherenowthechoiceofactionateachstepisderivedfromadifferentbehaviorpolicyπ′. Notebook19.4
Temporaldifference
In both cases, the policy is updated by taking the maximum of the action values
methods
at each state (equation 19.13). It can be shown that these updates are contraction
mappings(seeequation16.20); theactionvalueswilleventuallyconverge,assumingthat
Problem19.5
every state-action pair is visited an infinite number of times.
Draft: please send errata to udlbookmail@gmail.com.

386 19 Reinforcement learning
Figure 19.12 Q-learning. a) The agent starts in state s and takes action a =2
t t
accordingtothepolicy. Itdoesnotslipontheice,soitmovesdownward,receiving
reward r[s ,a ]=0 for leaving the original state. b) The maximum action value
t t
at the new state is found (here 0.43). c) The action value for action 2 in the
original state is updated to 1.12 based on the current estimate of the maximum
action value at the subsequent state, the reward, discount factor γ = 0.9, and
learningrateα=0.1. Thischangesthehighestactionvalueattheoriginalstate,
so the policy changes.
19.4 Fitted Q-learning
The tabular Monte Carlo and TD algorithms described above repeatedly traverse the
entire MDP and update the action values. However, this is only practical if the state-
action space is small. Unfortunately, this is rarely the case; even for the constrained
environment of a chessboard, there are more than 1040 possible legal states.
InfittedQ-learning,thediscreterepresentationq[s ,a ]oftheactionvaluesisreplaced
t t
by a machine learning model q[s ,a ,ϕ], where now the state is represented by a vector
t t
s ratherthanjustanindex. Wethendefinealeastsquareslossbasedontheconsistency
t
of adjacent action values (similar to the loss in Q-learning, see equation 19.15):
(cid:18) h i (cid:19)
2
L[ϕ]= r[s ,a ]+γ·max q[s ,a,ϕ] −q[s ,a ,ϕ] , (19.16)
t t t+1 t t
a
which in turn leads to the update:
(cid:18) h i (cid:19)
∂q[s ,a ,ϕ]
ϕ←ϕ+α r[s ,a ]+γ·max q[s ,a,ϕ] −q[s ,a ,ϕ] t t . (19.17)
t t a t+1 t t ∂ϕ
Fitted Q-learning differs from Q-Learning in that convergence is no longer guar-
anteed. A change to the parameters potentially modifies both the target r[s ,a ]+γ ·
t t
max [q[s ,a ,ϕ]](themaximumvaluemaychange)andthepredictionq[s ,a ,ϕ].
at+1 t+1 t+1 t t
This can be shown both theoretically and empirically to damage convergence.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

19.4 Fitted Q-learning 387
Figure 19.13 Atari Benchmark. The Atari benchmark consists of 49 Atari 2600
games, including Breakout (pictured), Pong, and various shoot-em-up, platform,
and other types of games. a-d) Even for games with a single screen, the state
is not fully observable from a single frame because the velocity of the objects is
unknown. Consequently, it is usual to use several adjacent frames (here, four)
to represent the state. e) The action simulates the user input via a joystick. f)
There are eighteen actions corresponding to eight directions of movement or no
movement, and for each of these nine cases, the button being pressed or not.
19.4.1 Deep Q-networks for playing ATARI games
Deep networks are ideally suited to making predictions from a high-dimensional state
space, so they are a natural choice for the model in fitted Q-learning. In principle, they
could take both state and action as input and predict the values, but in practice, the
network takes only the state and simultaneously predicts the values for each action.
The Deep Q-Network was a breakthrough reinforcement learning architecture that
exploited deep networks to learn to play ATARI 2600 games. The observed data com-
prises 220×160 images with 128 possible colors at each pixel (figure 19.13). This was
reshaped to size 84×84, and only the brightness value was retained. Unfortunately, the
full state is not observable from a single frame. For example, the velocity of game ob-
jects is unknown. To help resolve this problem, the network ingests the last four frames
at each time step to form s . It maps these frames through three convolutional layers
t
followed by a fully connected layer to predict the value of every action (figure 19.14).
Several modifications were made to the standard training procedure. First, the re-
wards (which were driven by the score in the game) were clipped to −1 for a negative
change and +1 for a positive change. This compensates for the wide variation in scores
between different games and allows the same learning rate to be used. Second, the
system exploited experience replay. Rather than update the network based on the tu-
ple<s ,a ,r ,s >atthecurrentsteporwithabatchofthelastI tuples, allrecent
t t t+1 t+1
Draft: please send errata to udlbookmail@gmail.com.

388 19 Reinforcement learning
Figure 19.14 Deep Q-network architec-
ture. The input s consists of four adja-
t
cent frames of the ATARI game. Each
is resized to 84×84 and converted to
grayscale. These frames are represented
asfourchannelsandprocessedbyan8×8
convolutionwithstridefour,followedby
a4×4convolutionwithstride2,followed
by two fully connected layers. The final
output predicts the action value q[s ,a ]
t t
for each of the 18 actions in this state.
...
tuples were stored in a buffer. This buffer was sampled randomly to generate a batch
at each step. This approach reuses data samples many times and reduces correlations
between the samples in the batch that arise due to the similarity of adjacent frames.
Finally,theissueofconvergenceinfittedQ-Networkswastackledbyfixingthetarget
−
parameters to values ϕ and only updating them periodically. This gives the update:
(cid:18) h i (cid:19)
∂q[s ,a ,ϕ]
ϕ←ϕ+α r[s ,a ]+γ·max q[s ,a,ϕ − ] −q[s ,a ,ϕ] t t . (19.18)
t t a t+1 t t ∂ϕ
Now the network no longer chases a moving target and is less prone to oscillation.
Using these and other heuristics and with an ϵ-greedy policy, Deep Q-Networks per-
formedatalevelcomparabletoaprofessionalgametesteracrossasetof49gamesusing
the same network architecture (trained separately for each game). It should be noted
thatthetrainingprocesswasdata-intensive. Ittookaround38fulldaysofexperienceto
learneachgame. Insomegames, thealgorithmexceeded humanperformance. On other
games like “Montezuma’s Revenge,” it barely made any progress. This game features
sparse rewards and multiple screens with quite different appearances.
19.4.2 Double Q-learning and double deep Q-networks
OnepotentialflawofQ-Learningisthatthemaximizationovertheactionsintheupdate:
(cid:16) (cid:2) (cid:3) (cid:17)
q[s ,a ]←q[s ,a ]+α r[s ,a ]+γ·max q[s ,a] −q[s ,a ] (19.19)
t t t t t t t+1 t t
a
leads to a systematic bias in the estimated action values q[s ,a ]. Consider two actions
t t
that provide the same average reward, but one is stochastic and the other deterministic.
The stochastic reward will exceed the average roughly half of the time and be chosen
by the maximum operation, causing the corresponding action value q[s ,a ] to be over-
t t
estimated. Asimilarargumentcanbemadeaboutrandominaccuraciesintheoutputof
the network q[s ,a ,ϕ] or random initializations of the q-function.
t t
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.