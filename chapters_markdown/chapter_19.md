# Chapter 19

*Pages: 388-395*

---

Chapter 19
Reinforcement learning
Reinforcement learning (RL) is a sequential decision-making framework in which agents
learntoperformactionsinanenvironmentwiththegoalofmaximizingreceivedrewards.
For example, an RL algorithm might control the moves (actions) of a character (the
agent) in a video game (the environment), aiming to maximize the score (the reward).
In robotics, an RL algorithm might control the movements (actions) of a robot (the
agent) in the world (the environment) to perform a task (earning a reward). In finance,
an RL algorithm might control a virtual trader (the agent) who buys or sells assets (the
actions) on a financial exchange (the environment) to maximize profit (the reward).
Consider learning to play chess. Here, there is a reward of +1, −1, or 0 at the end of
thegameiftheagentwins,loses,ordrawsand0ateveryothertimestep. Thisillustrates
the challenges of RL. First, the reward is sparse; here, we must play an entire game to
receive feedback. Second, the reward is temporally offset from the action that caused it;
adecisiveadvantagemightbegainedthirtymovesbeforevictory. Wemustassociatethe
reward with this critical action. This is termed the temporal credit assignment problem.
Third, the environment is stochastic; the opponent doesn’t always make the same move
in the same situation, so it’s hard to know if an action was truly good or just lucky.
Finally, the agent must balance exploring the environment (e.g., trying new opening
moves) with exploiting what it already knows (e.g., sticking to a previously successful
opening). This is termed the exploration-exploitation trade-off.
Reinforcementlearningisanoverarchingframeworkthatdoesnotnecessarilyrequire
deep learning. However, in practice, state-of-the-art systems often use deep networks.
They encode the environment (the video game display, robot sensors, financial time
series,orchessboard)andmapthisdirectlyorindirectlytothenextaction(figure1.13).
19.1 Markov decision processes, returns, and policies
Reinforcementlearningmapsobservationsofanenvironmenttoactions,aimingtomaxi-
mizeanumericalquantitythatisconnectedtotherewardsreceived. Inthemostcommon
case, we learn a policy that maximizes the expected return in a Markov decision process.
This section explains these terms.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

19.1 Markov decision processes, returns, and policies 375
Figure19.1Markovprocess. AMarkovprocessconsistsofasetofstatesandtran-
sitionprobabilitiesPr(s |s )thatdefinetheprobabilityofmovingtostates
t+1 t t+1
given the current state is s . a) The penguin can visit 16 different positions
t
(states)ontheice. b)Theiceisslippery,soateachtime,ithasanequalproba-
bility of moving to any adjacent state. For example, in position 6, it has a 25%
chance of moving to states 2, 5, 7, and 10. A trajectory τ =[s ,s ,s ,...] from
1 2 3
this process consists of a sequence of states.
19.1.1 Markov process
A Markov process assumes that the world is always in one of a set of possible states.
The word Markov implies that the probability of being in a state depends only on the
previousstateandnotonthestatesbefore. Thechangesbetweenstatesarecapturedby
thetransitionprobabilitiesPr(s |s )ofmovingtothenextstates giventhecurrent
t+1 t t+1
state s , where t indexes the time step. Hence, a Markov process is an evolving system
t
that produces a sequence s ,s ,s ... of states (figure 19.1).
1 2 3
19.1.2 Markov reward process
AMarkovrewardprocessextendstheMarkovprocesstoincludeadistributionPr(r |s )
t+1 t
overthepossiblerewardsr receivedatthenexttimestep,giventhatweareinstates .
t+1 t
Problem19.1
Thisproducesasequences ,r ,s ,r ,s ,r ...ofstatesandtheassociatedrewards(fig-
1 2 2 3 3 4
ure 19.2). The Markov reward process also includes a discount factor γ ∈ (0,1] that is
used to compute the return G at time t:
t
X∞
G = γkr . (19.1)
t t+k+1
k=0
Thereturnisthesumofthecumulativediscountedfuturerewards;itmeasuresthefuture
benefitofbeingonthistrajectory. Adiscountfactoroflessthanonemakesrewardsthat
are closer in time more valuable than rewards that are further away.
Draft: please send errata to udlbookmail@gmail.com.

376 19 Reinforcement learning
Figure 19.2 Markov reward process. This associates a distribution Pr(r |s )
t+1 t
of rewards r with each state s . a) Here, the rewards are deterministic; the
t+1 t
penguin will receive a reward of +1 if it lands on a fish and 0 otherwise. The
trajectoryτ nowconsistsofasequences ,r ,s ,r ,s ,r ...ofalternatingstates
1 2 2 3 3 4
and rewards, terminating after eight steps. The return G of the sequence is the
t
sumofdiscountedfuturerewards,herewithdiscountfactorγ =0.9. b-c)Asthe
penguin proceeds along the trajectory and gets closer to reaching the rewards,
the return increases.
Figure 19.3 Markov decision process. a) The agent (penguin) can perform one
of a set of actions in each state. b) Here, the four actions correspond to moving
up, right, down, and left. c) For any state (here, state 6), the action changes
the probability of moving to the next state. The penguin moves in the intended
direction with 50% probability, but the ice is slippery, so it may slide to one
of the other adjacent positions with equal probability. Accordingly, in panel
(a), the action taken (gray arrows) doesn’t always line up with the trajectory
(orangeline). Ingeneral,theactioncanalsoinfluencetheprobabilityofreceiving
rewards, but in this example the reward is the same regardless of the action,
so Pr(r |s ,a ) = Pr(r |s ). The trajectory τ from an MDP consists of a
t+1 t t t+1 t
sequence s ,a ,r ,s ,a ,r ,s ,a ,r ... of alternating states s , actions a , and
1 1 2 2 2 3 3 3 4 t t
rewards, r . Note that here the penguin receives the reward when it leaves a
t+1
state with a fish (i.e., the reward is received for passing through the fish square,
regardless of whether the penguin arrived there intentionally or not).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

19.1 Markov decision processes, returns, and policies 377
Figure 19.4PartiallyobservableMarkov
decision process (POMDP). In a
POMDP,theagentdoesnothaveaccess
to the entire state. Here, the penguin
does not know the current state and
can only see tiles in the vicinity (dashed
box). Unfortunately, the true state
(three) is indistinguishable from what it
wouldseeinstatenine. Inthefirstcase,
moving right leads to the hole in the ice
(with -2 reward) and, in the latter, to
the fish (with +3 reward).
Figure 19.5Policies. a)Adeterministicpolicyalwayschoosesthesameactionin
eachstate(indicatedbyarrow). Somepoliciesarebetterthanothers. Thispolicy
isnotoptimalbutstillgenerallysteersthepenguinfromtop-lefttobottom-right
wheretherewardlies. b)Thispolicyismorerandom. c)Astochasticpolicyhas
a probability distribution over actions for each state (probability indicated by
size of arrows). This has the advantage that the agent explores the states more
thoroughlyandcanbenecessaryforoptimalperformanceinpartiallyobservable
Markov decision processes.
Figure 19.6 Reinforcement learning
loop. The agent takes an action a at
t
time t based on the state s , according
t
to the policy π[a |s ]. This triggers
t t
the generation of a new state s (via
t+1
the state transition function) and a
reward r (via the reward function).
t+1
Both are passed back to the agent,
which then chooses a new action.
Draft: please send errata to udlbookmail@gmail.com.

378 19 Reinforcement learning
19.1.3 Markov decision process
AMarkov decision processorMDPaddsasetofpossibleactionsateachtimestep. The
action a changes the transition probabilities, which are now written as Pr(s |s ,a ).
t t+1 t t
The rewards can also depend on the action and are now written as Pr(r |s ,a ). An
t+1 t t
MDP produces a sequence (s ,a ,r ),(s ,a ,r ),(s ,a ,r )... of states s , actions a ,
1 1 2 2 2 3 3 3 4 t t
andrewardsr whicharereceivedatthesubsequenttime-step(figure19.3). Theentity
t+1
that performs the actions is known as the agent.
19.1.4 Partially observable Markov decision process
In a partially observable Markov decision process or POMDP, the state is not directly
visible (figure 19.4). Instead, the agent receives an observation o drawn from Pr(o |s ).
t t t
Hence,aPOMDPgeneratesasequences ,o ,a ,r ,s ,o ,a ,r ,o ,a ,s ,r ,...ofstates,
1 1 1 2 2 2 2 3 3 3 3 4
observations,actions,andrewards. Ingeneral,eachobservationwillbemorecompatible
with some states than others but insuﬀicient to identify the state uniquely.
19.1.5 Policy
The rules that determine the agent’s action for each state are known as the policy (fig-
ure 19.5). This may be stochastic (the policy defines a distribution over actions for each
state) or deterministic (the agent always takes the same action in a given state). A
stochastic policy π[a|s] returns a probability distribution over each possible action a for
states,fromwhichanewactionissampled. Adeterministicpolicyπ[a|s]returnsonefor
the action a that is chosen for state s and zero otherwise. A stationary policy depends
only on the current state. A non-stationary policy also depends on the time step.
The environment and the agent form a loop (figure 19.6). The agent receives the
Notebook19.1
state s and reward r from the last time step. Based on this, it can modify the policy
Markovdecision t t
processes π[a t |s t ] if desired and choose the next action a t . The environment then advances to the
next state according to Pr(s |s ,a ) and issues a reward according to Pr(r |s ,a ).
t+1 t t t+1 t t
19.2 Expected return
The previous section introduced the Markov decision process and the idea of an agent
carrying out actions according to a policy. We want to choose a policy that maximizes
the expected return. In this section, we make this idea mathematically precise. To do
that, we assign a value to each state s and state-action pair {s ,a }.
t t t
19.2.1 State and action values
The return G depends on the state s and the policy π[a|s]. From this state, the
t t
agent will pass through a sequence of states, taking actions and receiving rewards. This
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

19.2 Expected return 379
Figure19.7Stateandactionvalues. a)Thevaluev[s |π]ofastates (numberat
t t
each position) is the expected return for this state for a given policy π (gray ar-
rows). Itistheaveragesumofdiscountedrewardsreceivedovermanytrajectories
started from this state. Here, states closer to the fish are more valuable. b) The
value q[s ,a ,π] of an action a in state s (four numbers at each position/state
t t t t
corresponding to four actions) is the expected return given that this particular
actionistakeninthisstate. Inthiscase,itgetslargeraswegetclosertothefish
and is larger for actions that head in the direction of the fish. c) If we know the
action values at a state, then the policy can be modified so that it chooses the
maximum of these values (red numbers in panel b).
sequencedifferseverytimetheagentstartsinthesameplacesince,ingeneral,thepolicy
π[a |s ], the state transitions Pr(s |s ,a ), and the rewards issued Pr(r |s ,a ) are
t t t+1 t t t+1 t t
all stochastic.
We can characterize how “good” a state is under a given policy π by considering
the expected return v[s |π]. This is the return that would be received on average from AppendixC.2
t Expectation
sequences that start from this state and is termed the state value or state-value function
(figure 19.7a):
h i
v[s |π]=E G |s ,π . (19.2)
t t t
Informally, the state value tells us the long-term reward we can expect on average if
we start in this state and follow the specified policy thereafter. It is highest for states
where it’s probable that subsequent transitions will bring large rewards soon (assuming
the discount factor γ is less than one).
Similarly, the action value or state-action value function q[s ,a |π] is the expected
t t
return from executing action a in state s (figure 19.7b):
t t
h i
q[s ,a |π]=E G |s ,a ,π . (19.3)
t t t t t
Theactionvaluetellsusthelong-termrewardwecanexpectonaverageifwestartinthis
state, take this action, and follow the specified policy thereafter. Through this quantity,
reinforcementlearningalgorithmsconnectfuturerewardstocurrentactions(i.e.,resolve
the temporal credit assignment problem).
Draft: please send errata to udlbookmail@gmail.com.

380 19 Reinforcement learning
19.2.2 Optimal policy
We want a policy that maximizes the expected return. For MDPs (but not POMDPs),
thereisalwaysadeterministic,stationarypolicythatmaximizesthevalueofeverystate.
If we know this optimal policy, then we get the optimal state-value function v∗[s ]:
t
(cid:20) h i(cid:21)
v ∗ [s ]=max E G |s ,π . (19.4)
t t t
π
Similarly, the optimal state-action value function is obtained under the optimal policy:
h h ii
q ∗ [s ,a ]=max E G |s ,a ,π . (19.5)
t t t t t
π
Turningthisonitshead,ifweknewtheoptimalaction-valuesq∗[s ,a ],thenwecould
t t
derivetheoptimalpolicybychoosingtheactiona withthehighestvalue(figure19.7c):1
t
h i
π[a |s ]←argmax q ∗ [s ,a ] . (19.6)
t t t t
at
Indeed, some reinforcement learning algorithms are based on alternately estimating the
action values and the policy (see section 19.3).
19.2.3 Bellman equations
Wemaynotknowthestatevaluesv[s ]oractionvaluesq[s ,a ]foranypolicy.2 However,
t t t
we know that they must be consistent with one another, and it’s easy to write relations
between these quantities. The state value v[s ] can be found by taking a weighted sum
t
of the action values q[s ,a ], where the weights depend on the probability under the
t t
policy π[a |s ] of taking that action (figure 19.8):
t t
X
v[s ]= π[a |s ]q[s ,a ]. (19.7)
t t t t t
at
Similarly, the value of an action is the immediate reward r = r[s ,a ] generated by
t+1 t t
takingtheaction,plusthevaluev[s ]ofbeinginthesubsequentstates discounted
t+1 t+1
by γ (figure 19.9).3 Since the assignment of s is not deterministic, we weight the
t+1
values v[s ] according to the transition probabilities Pr(s |s ,a ):
t+1 t+1 t t
X
q[s ,a ]=r[s ,a ]+γ· Pr(s |s ,a )v[s ]. (19.8)
t t t t t+1 t t t+1
st+1
Substituting equation 19.8 into equation 19.7 provides a relation between the state
value at time t and t+1:
1Thenotationπ[at |st]←ainequations19.6,19.12,and19.13meanssetπ[at |s]tooneforactiona
andπ[at |s]tozeroforotheractions.
2Forsimplicity,wewilljustwritev[st]andq[st,at]insteadofv[st |π]andq[st,at |π]fromnowon.
3Wealsoassumefromnowonthattherewardsaredeterministicandcanbewrittenasr[st,at].
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

19.2 Expected return 381
Figure 19.8 Relationship between state values and action values. The value of
state six v[s =6] is a weighted sum of the action values q[s =6,a ] at state six,
t t t
where the weights are the policy probabilities π[a |s =6] of taking that action.
t t
Figure19.9Relationshipbetweenactionvaluesandstatevalues. Thevalueq[s =
t
6,a =2]oftakingactiontwoinstatesixistherewardr[s =6,a =2]fromtaking
t t t
that action plus a weighted sum of the discounted values v[s ] of being in
t+1
successor states, where the weights are the transition probabilities Pr(s |s =
t+1 t
6,a =2). The Bellman equations chain this relation with that of figure 19.8 to
t
link the current and next (i) state values and (ii) action values.
Draft: please send errata to udlbookmail@gmail.com.