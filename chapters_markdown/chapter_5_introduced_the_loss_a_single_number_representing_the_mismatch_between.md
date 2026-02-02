# Chapter 5: introduced the loss — a single number representing the mismatch between

*Pages: 91-109*

---

Chapter 6
Fitting models
Chapters 3 and 4 described shallow and deep neural networks. These represent families
of piecewise linear functions, where the parameters determine the particular function.
Chapter 5 introduced the loss — a single number representing the mismatch between
the network predictions and the ground truth for a training set.
The loss depends on the network parameters, and this chapter considers how to find
the parameter values that minimize this loss. This is known as learning the network’s
parameters or simply as training or fitting the model. The process is to choose initial
parameter values and then iterate the following two steps: (i) compute the derivatives
(gradients) of the loss with respect to the parameters, and (ii) adjust the parameters
based on the gradients to decrease the loss. After many iterations, we hope to reach the
overall minimum of the loss function.
This chapter tackles the second of these steps; we consider algorithms that adjust
theparameterstodecreasetheloss. Chapter7discusseshowtoinitializetheparameters
and compute the gradients for neural networks.
6.1 Gradient descent
To fit a model, we need a training set {x ,y } of input/output pairs. We seek param-
i i
eters ϕ for the model f[x ,ϕ] that map the inputs x to the outputs y as closely as
i i i
possible. To this end, we define a loss function L[ϕ] that returns a single number that
quantifies the mismatch in this mapping. The goal of an optimization algorithm is to
find parameters ϕˆ that minimize the loss:
h i
ϕˆ =argmin L[ϕ] . (6.1)
ϕ
Therearemanyfamiliesofoptimizationalgorithms,butthestandardmethodsfortrain-
ingneuralnetworksareiterative. Thesealgorithmsinitializetheparametersheuristically
and then adjust them repeatedly in such a way that the loss decreases.
Draft: please send errata to udlbookmail@gmail.com.

78 6 Fitting models
The simplest method in this class is gradient descent. This starts with initial param-
eters ϕ=[ϕ ,ϕ ,...,ϕ ]T and iterates two steps:
0 1 N
Step 1. Compute the derivatives of the loss with respect to the parameters:
2 3
∂L
6∂ϕ07
6 ∂L 7
∂L = 6 6 ∂ϕ . 1 7 7. (6.2)
∂ϕ 6 4 . . 7 5
∂L
∂ϕN
Step 2. Update the parameters according to the rule:
∂L
ϕ←−ϕ−α· , (6.3)
∂ϕ
where the positive scalar α determines the magnitude of the change.
The first step computes the gradient of the loss function at the current position. This
determines the uphill direction of the loss function. The second step moves a small
distance α downhill (hence the negative sign). The parameter α may be fixed (in which
Notebook6.1
case, we call it a learning rate), or we may perform a line search where we try several
Linesearch
values of α to find the one that most decreases the loss.
At the minimum of the loss function, the surface must be flat (or we could improve
furtherbygoingdownhill). Hence,thegradientwillbezero,andtheparameterswillstop
changing. In practice, we monitor the gradient magnitude and terminate the algorithm
when it becomes too small.
6.1.1 Linear regression example
Considerapplyinggradientdescenttothe1Dlinearregressionmodelfromchapter2. The
modelf[x,ϕ]mapsascalarinputxtoascalaroutputyandhasparametersϕ=[ϕ ,ϕ ]T,
0 1
which represent the y-intercept and the slope:
y = f[x,ϕ]
= ϕ +ϕ x. (6.4)
0 1
Given a dataset {x ,y } containing I input/output pairs, we choose the least squares
i i
loss function:
XI XI
L[ϕ] = ℓ = (f[x ,ϕ]−y )2
i i i
i=1 i=1
XI
= (ϕ +ϕ x −y )2, (6.5)
0 1 i i
i=1
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

6.1 Gradient descent 79
Figure6.1Gradientdescentforthelinearregressionmodel. a)TrainingsetofI =
12 input/output pairs {x ,y }. b) Loss function showing iterations of gradient
i i
descent. We start at point 0 and move in the steepest downhill direction until
we can improve no further to arrive at point 1. We then repeat this procedure.
We measure the gradient at point 1 and move downhill to point 2 and so on. c)
This can be visualized better as a heatmap, where the brightness represents the
loss. After only four iterations, we are already close to the minimum. d) The
modelwiththeparametersatpoint0(lightestline)describesthedataverybadly,
buteachsuccessiveiterationimprovesthefit. Themodelwiththeparametersat
point 4 (darkest line) is already a reasonable description of the training data.
Draft: please send errata to udlbookmail@gmail.com.

80 6 Fitting models
where the term ℓ = (ϕ +ϕ x −y )2 is the individual contribution to the loss from
i 0 1 i i
the ith training example.
Thederivativeofthelossfunctionwithrespecttotheparameterscanbedecomposed
into the sum of the derivatives of the individual contributions:
XI XI
∂L ∂ ∂ℓ
= ℓ = i, (6.6)
∂ϕ ∂ϕ i ∂ϕ
i=1 i=1
where these are given by:
Problem6.1
2 3 " #
∂ℓ i = 4∂ ∂ ϕ ℓi 05 = 2(ϕ 0 +ϕ 1 x i −y i ) . (6.7)
∂ϕ
∂
∂
ϕ
ℓi
1
2x
i
(ϕ
0
+ϕ
1
x
i
−y
i
)
Figure 6.1 shows the progression of this algorithm as we iteratively compute the
Notebook6.2
derivativesaccordingtoequations6.6and6.7andthenupdatetheparametersusingthe
Gradientdescent
rule in equation 6.3. In this case, we have used a line search procedure to find the value
of α that decreases the loss the most at each iteration.
6.1.2 Gabor model example
Loss functions for linear regression problems (figure 6.1c) always have a single well-
defined global minimum. More formally, they are convex, which means that every chord
Problem6.2
(line segment between two points on the surface) lies above the function and does not
intersect it. Convexity implies that wherever we initialize the parameters, we are bound
to reach the minimum if we keep walking downhill; the training procedure can’t fail.
Unfortunately, loss functions for most nonlinear models, including both shallow and
deepnetworks, are non-convex. Visualizingneural networkloss functions ischallenging
duetothenumberofparameters. Hence,wefirstexploreasimplernonlinearmodelwith
two parameters to gain insight into the properties of non-convex loss functions:
(cid:18) (cid:19)
(ϕ +0.06·ϕ x)2
f[x,ϕ]=sin[ϕ +0.06·ϕ x]·exp − 0 1 . (6.8)
0 1 32.0
This Gabor model maps scalar input x to scalar output y and consists of a sinusoidal
Problems6.3–6.5
component (creating an oscillatory function) multiplied by a negative exponential com-
ponent (causing the amplitude to decrease as we move from the center). It has two
parameters ϕ = [ϕ ,ϕ ]T, where ϕ ∈ R determines the mean position of the function
0 1 0
and ϕ ∈R+ stretches or squeezes it along the x-axis (figure 6.2).
1
Consider a training set of I examples {x ,y } (figure 6.3). The least squares loss
i i
function for I training examples is defined as:
XI
L[ϕ]= (f[x ,ϕ]−y )2. (6.9)
i i
i=1
Once more, the goal is to find the parameters ϕˆ that minimize this loss.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

6.1 Gradient descent 81
Figure 6.2 Gabor model. This nonlinear model maps scalar input x to scalar
output y and has parameters ϕ = [ϕ ,ϕ ]T. It describes a sinusoidal function
0 1
that decreases in amplitude with distance from its center. Parameter ϕ ∈ R
0
determines the position of the center. As ϕ increases, the function moves left.
0
Parameter ϕ ∈R+ squeezes the function along the x-axis relative to the center.
1
As ϕ increases, the function narrows. a–c) Model with different parameters.
1
(Interactive figure)
Figure 6.3 Training data for fitting the
Gabormodel. Thetrainingdatasetcon-
tains28input/outputexamples{x ,y }.
i i
These data were created by uniformly
sampling x ∈ [−15,15], passing the
i
samplesthroughaGabormodelwithpa-
rameters ϕ = [0.0,16.6]T, and adding
normally distributed noise.
6.1.3 Local minima and saddle points
Figure 6.4 depicts the loss function associated with the Gabor model for this dataset.
There are numerous local minima (cyan circles). Here the gradient is zero, and the loss
Problem6.6
increases if we move in any direction, but we are not at the overall minimum of the
function. Thepointwiththelowestlossisknownastheglobal minimumandisdepicted
by the gray circle.
If we start in a random position and use gradient descent to go downhill, there is
Problems6.7–6.8
no guarantee that we will wind up at the global minimum and find the best parameters
(figure 6.5a). It’s equally or even more likely that the algorithm will terminate in one
of the local minima. Furthermore, there is no way of knowing whether there is a better
solution elsewhere.
Draft: please send errata to udlbookmail@gmail.com.

82 6 Fitting models
Figure6.4LossfunctionfortheGabormodel. a)Thelossfunctionisnon-convex,
withmultiplelocalminima(cyancircles)inadditiontotheglobalminimum(gray
circle). It also contains saddle points where the gradient is locally zero, but the
function increases in one direction and decreases in the other. The blue cross is
an example of a saddle point; the function decreases as we move horizontally in
either direction but increases as we move vertically. b–f) Models associated with
the different minima. In each case, there is no small change that decreases the
loss. Panel (c) shows the global minimum, which has a loss of 0.64. (Interactive
figure)
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

6.2 Stochastic gradient descent 83
Figure 6.5 Gradient descent vs. stochastic gradient descent. a) Gradient descent
with line search. As long as the gradient descent algorithm is initialized in the
right “valley” of the loss function (e.g., points 1 and 3), the parameter estimate
willmovesteadilytowardtheglobalminimum. However,ifitisinitializedoutside
this valley (e.g., point 2), it will descend toward one of the local minima. b)
Stochasticgradientdescentaddsnoisetotheoptimizationprocess,soitispossible
toescapefromthewrongvalley(e.g.,point2)andstillreachtheglobalminimum.
Inaddition,thelossfunctioncontainssaddlepoints(e.g.,thebluecrossinfigure6.4).
Here, the gradient is zero, but the function increases in some directions and decreases
in others. If the current parameters are not exactly at the saddle point, then gradient
descent can escape by moving downhill. However, the surface near the saddle point is
flat, soit’shardtobesurethattraininghasn’t converged; ifweterminatethe algorithm
when the gradient is small, we may erroneously stop near a saddle point.
6.2 Stochastic gradient descent
TheGabormodelhastwoparameters,sowecouldfindtheglobalminimumbyeither(i)
exhaustively searching the parameter space or (ii) repeatedly starting gradient descent
from different positions and choosing the result with the lowest loss. However, neural
network models can have millions of parameters, so neither approach is practical. In
short, using gradient descent to find the global optimum of a high-dimensional loss
functionischallenging. Wecanfindaminimum, butthereisnowaytotellwhetherthis
Draft: please send errata to udlbookmail@gmail.com.

84 6 Fitting models
Figure 6.6 Alternative view of SGD for the Gabor model with a batch size of
three. a) Loss function for the entire training dataset. At each iteration, there
isaprobabilitydistributionofpossibleparameterchanges(insetshowssamples).
Thesecorrespondtodifferentchoicesofthethreebatchelements. b)Lossfunction
for one possible batch. The SGD algorithm moves in the downhill direction on
this function for a distance that is determined by the learning rate and the local
gradient magnitude. The current model (dashed function in inset) changes to
better fit the batch data (solid function). c) A different batch creates a different
loss function and results in a different update. d) For this batch, the algorithm
movesdownhillwithrespecttothebatchlossfunctionbutuphillwithrespectto
thegloballossfunctioninpanel(a). ThisishowSGDcanescapelocalminima.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

6.2 Stochastic gradient descent 85
is the global minimum or even a good one.
Oneofthemainproblemsisthatthefinaldestinationofagradientdescentalgorithm
Notebook6.3
isentirelydeterminedbythestartingpoint. Stochasticgradientdescent(SGD)attempts
Stochastic
to remedy this problem by adding some noise to the gradient at each step. The solution gradientdescent
still moves downhill on average, but at any given iteration, the direction chosen is not
necessarily in the steepest downhill direction. Indeed, it might not be downhill at all.
The SGD algorithm has the possibility of moving temporarily uphill and hence jumping
from one “valley” of the loss function to another (figure 6.5b).
6.2.1 Batches and epochs
The mechanism for introducing randomness is simple. At each iteration, the algorithm
chooses a random subset of the training data and computes the gradient from these
examplesalone. Thissubsetisknownasaminibatchorbatchforshort. Theupdaterule
for the model parameters ϕ at iteration t is hence:
t
X
∂ℓ [ϕ ]
ϕ ←−ϕ −α· i t , (6.10)
t+1 t ∂ϕ
i∈B
t
where B is a set containing the indices of the input/output pairs in the current batch
t
and, as before, ℓ is the loss due to the ith pair. The term α is the learning rate, and
i
together with the gradient magnitude, determines the distance moved at each iteration.
The learning rate is chosen at the start of the procedure and does not depend on the
local properties of the function.
Thebatchesareusuallydrawnfromthedatasetwithoutreplacement. Thealgorithm
works through the training examples until it has used all the data, at which point it
Problem6.9
starts sampling from the full training dataset again. A single pass through the entire
training dataset is referred to as an epoch. A batch may be as small as a single example
or as large as the whole dataset. The latter case is called full-batch gradient descent and
is identical to regular (non-stochastic) gradient descent.
An alternative interpretation of SGD is that it computes the gradient of a different
loss function at each iteration; the loss function depends on both the model and the
training data and hence will differ for each randomly selected batch. In this view,
SGD performs deterministic gradient descent on a constantly changing loss function
(figure 6.6). However, despite this variability, the expected loss and expected gradients
at any point remain the same as for gradient descent.
6.2.2 Properties of stochastic gradient descent
SGD has several attractive features. First, although it adds noise to the trajectory, it
still improves the fit to a subset of the data at each iteration. Hence, the updates tend
to be sensible even if they are not optimal. Second, because it draws training examples
without replacement and iterates through the dataset, the training examples all still
contribute equally. Third, it is less computationally expensive to compute the gradient
Draft: please send errata to udlbookmail@gmail.com.

86 6 Fitting models
fromjustasubsetofthetrainingdata. Fourth,itcan(inprinciple)escapelocalminima.
Fifth, it reduces the chances of getting stuck near saddle points; it is likely that at least
some of the possible batches will have a significant gradient at any point on the loss
function. Finally, there is some evidence that SGD finds parameters for neural networks
that cause them to generalize well to new data in practice (see section 9.2).
SGD does not necessarily “converge” in the traditional sense. However, the hope is
that when we are close to the global minimum, all the data points will be well described
by the model. Consequently, the gradient will be small, whichever batch is chosen, and
the parameters will cease to change much. In practice, SGD is often applied with a
learning rate schedule. The learning rate α starts at a high value and is decreased by a
constantfactoreveryN epochs. Thelogicisthatintheearlystagesoftraining,wewant
the algorithm to explore the parameter space, jumping from valley to valley to find a
sensibleregion. Inlaterstages,weareroughlyintherightplaceandaremoreconcerned
with fine-tuning the parameters, so we decrease α to make smaller changes.
6.3 Momentum
A common modification to stochastic gradient descent is to add a momentum term. We
update the parameters with a weighted combination of the gradient computed from the
current batch and the direction moved in the previous step:
X
∂ℓ [ϕ ]
m ← β·m +(1−β) i t
t+1 t ∂ϕ
i∈B
t
ϕ ← ϕ −α·m , (6.11)
t+1 t t+1
where m is the momentum (which drives the update at iteration t), β ∈[0,1) controls
t
the degree to which the gradient is smoothed over time, and α is the learning rate.
Therecursiveformulationofthemomentumcalculationmeansthatthegradientstep
is an infinite weighted sum of all the previous gradients, where the weights get smaller
as we move back in time. The effective learning rate increases if all these gradients
Problem6.10
are aligned over multiple iterations but decreases if the gradient direction repeatedly
changes as the terms in the sum cancel out. The overall effect is a smoother trajectory
and reduced oscillatory behavior in valleys (figure 6.7).
6.3.1 Nesterov accelerated momentum
ThemomentumtermcanbeconsideredacoarsepredictionofwheretheSGDalgorithm
Notebook6.4
will move next. Nesterov accelerated momentum (figure 6.8) computes the gradients at
Momentum
this predicted point rather than at the current point:
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

6.3 Momentum 87
Figure 6.7 Stochastic gradient descent with momentum. a) Regular stochastic
descent takes a very indirect path toward the minimum. b) With a momentum
term, the change at the current step is a weighted combination of the previ-
ous change and the gradient computed from the batch. This smooths out the
trajectory and increases the speed of convergence.
Figure 6.8Nesterovacceleratedmomen-
tum. The solution has traveled along
the dashed line to arrive at point 1. A
traditional momentum update measures
the gradient at point 1, moves some dis-
tance in this direction to point 2, and
thenaddsthemomentumtermfromthe
previous iteration (i.e., in the same di-
rection as the dashed line), arriving at
point 3. The Nesterov momentum up-
date first applies the momentum term
(moving from point 1 to point 4) and
then measures the gradient and applies
an update to arrive at point 5.
Draft: please send errata to udlbookmail@gmail.com.

88 6 Fitting models
X ∂ℓ [ϕ −αβ·m ]
m ← β·m +(1−β) i t t
t+1 t ∂ϕ
i∈B
t
ϕ ← ϕ −α·m , (6.12)
t+1 t t+1
where now the gradients are evaluated at ϕ −αβ·m . One way to think about this is
t t
that the gradient term now corrects the path provided by momentum alone.
6.4 Adam
Gradient descent with a fixed step size has the following undesirable property: it makes
large adjustments to parameters associated with large gradients (where perhaps we
should be more cautious) and small adjustments to parameters associated with small
gradients (where perhaps we should explore further). When the gradient of the loss
surface is much steeper in one direction than another, it is diﬀicult to choose a learning
rate that (i) makes good progress in both directions and (ii) is stable (figures 6.9a–b).
A straightforward approach is to normalize the gradients so that we move a fixed
distance (governed by the learning rate) in each direction. To do this, we first measure
the gradient m and the pointwise squared gradient v :
t+1 t+1
∂L[ϕ ]
m ← t
t+1 ∂ϕ
(cid:18) (cid:19)
∂L[ϕ ] 2
v ← t . (6.13)
t+1 ∂ϕ
Then we apply the update rule:
m
ϕ ← ϕ −α· √ t+1 , (6.14)
t+1 t v +ϵ
t+1
where the square root and division are both pointwise, α is the learning rate, and ϵ is a
small constant that prevents division by zero when the gradient magnitude is zero. The
term v is the squared gradient, and the positive root of this is used to normalize the
t+1
gradient itself, so all that remains is the sign in each coordinate direction. The result is
that the algorithm moves a fixed distance α along each coordinate, where the direction
is determined by whichever way is downhill (figure 6.9c). This simple algorithm makes
good progress in both directions but will not converge unless it happens to land exactly
at the minimum. Instead, it will bounce back and forth around the minimum.
Adaptive moment estimation, or Adam, takes this idea and adds momentum to both
the estimate of the gradient and the squared gradient:
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

6.4 Adam 89
Figure 6.9 Adaptive moment estimation (Adam). a) This loss function changes
quickly in the vertical direction but slowly in the horizontal direction. If we run
full-batch gradient descent with a learning rate that makes good progress in the
vertical direction, then the algorithm takes a long time to reach the final hor-
izontal position. b) If the learning rate is chosen so that the algorithm makes
good progress in the horizontal direction, it overshoots in the vertical direction
andbecomesunstable. c)Astraightforwardapproachistomoveafixeddistance
alongeachaxisateachstepsothatwemovedownhillinbothdirections. Thisis
accomplishedbynormalizingthegradientmagnitudeandretainingonlythesign.
However,thisdoesnotusuallyconvergetotheexactminimumbutinsteadoscil-
lates back and forth around it (here between the last two points). d) The Adam
algorithmusesmomentuminboththeestimatedgradientandthenormalization
term, which creates a smoother path.
Draft: please send errata to udlbookmail@gmail.com.

90 6 Fitting models
∂L[ϕ ]
m ← β·m +(1−β) t
t+1 t ∂ϕ
(cid:18) (cid:19)
∂L[ϕ ] 2
v ← γ·v +(1−γ) t , (6.15)
t+1 t ∂ϕ
where β and γ are the momentum coeﬀicients for the two statistics.
Using momentum is equivalent to taking a weighted average over the history of each
of these statistics. At the start of the procedure, all the previous measurements are
effectively zero, resulting in unrealistically small estimates. Consequently, we modify
these statistics using the rule:
m
m˜ ← t+1
t+1 1−βt+1
v
v˜ ← t+1 . (6.16)
t+1 1−γt+1
Since β and γ are in the range [0,1), the terms with exponents t+1 become smaller
with each time step, the denominators become closer to one, and this modification has
a diminishing effect.
Finally, we update the parameters as before, but with the modified terms:
m˜
ϕ ← ϕ −α· p t+1 . (6.17)
t+1 t v˜ +ϵ
t+1
The result is an algorithm that can converge to the overall minimum and makes good
Notebook6.5
progress in every direction in the parameter space. Note that Adam is usually used in a
Adam
stochasticsettingwherethegradientsandtheirsquaresarecomputedfrommini-batches:
X
∂ℓ [ϕ ]
m ← β·m +(1−β) i t
t+1 t ∂ϕ
i∈B
t !
X 2
∂ℓ [ϕ ]
v ← γ·v +(1−γ) i t , (6.18)
t+1 t ∂ϕ
i∈B
t
and so the trajectory is noisy in practice.
As we shall see in chapter 7, the gradient magnitudes of neural network parameters
can depend on their depth in the network. Adam helps compensate for this tendency
and balances out changes across the different layers. In practice, Adam also has the
advantage of being less sensitive to the initial learning rate because it avoids situations
like those in figures 6.9a–b, so it doesn’t need complex learning rate schedules.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

6.5 Training algorithm hyperparameters 91
6.5 Training algorithm hyperparameters
The choices of learning algorithm, batch size, learning rate schedule, and momentum
coeﬀicients are all considered hyperparameters of the training algorithm; these directly
affectthefinalmodelperformancebutaredistinctfromthemodelparameters. Choosing
thesecanbemoreartthanscience,andit’scommontotrainmanymodelswithdifferent
hyperparameters and choose the best one. This is known as hyperparameter search. We
return to this issue in chapter 8.
6.6 Summary
Thischapterdiscussedmodeltraining. Thisproblemwasframedasfindingparametersϕ
thatcorrespondedtotheminimumofalossfunctionL[ϕ]. Thegradientdescentmethod
measures the gradient of the loss function for the current parameters (i.e., how the loss
changeswhenwemakeasmallchangetotheparameters). Thenitmovestheparameters
in the direction that decreases the loss fastest. This is repeated until convergence.
For nonlinear functions, the loss function may have both local minima (where gradi-
entdescentgetstrapped)andsaddlepoints(wheregradientdescentmayappeartohave
converged but has not). Stochastic gradient descent helps mitigate these problems.1 At
each iteration, we use a different random subset of the data (a batch) to compute the
gradient. This adds noise to the process and helps prevent the algorithm from getting
trapped in a sub-optimal region of parameter space. Each iteration is also computation-
ally cheaper since it only uses a subset of the data. We saw that adding a momentum
term makes convergence more eﬀicient. Finally, we introduced the Adam algorithm.
The ideas in this chapter apply to optimizing any model. The next chapter tackles
two aspects of training specific to neural networks. First, we address how to compute
the gradients of the loss with respect to the parameters of a neural network. This is
accomplished using the famous backpropagation algorithm. Second, we discuss how to
initialize the network parameters before optimization begins. Without careful initializa-
tion, the gradients used by the optimization can become extremely large or extremely
small, which can hinder the training process.
Notes
Optimization algorithms: Optimization algorithms are used extensively throughout engi-
neering, and it is generally more typical to use the term objective function rather than loss
functionorcostfunction. GradientdescentwasinventedbyCauchy(1847),andstochasticgra-
dient descent dates back to at least Robbins & Monro (1951). A modern compromise between
the two is stochastic variance-reduced descent (Johnson & Zhang, 2013), in which the full gra-
dient is computed periodically, with stochastic updates interspersed. Reviews of optimization
algorithms for neural networks can be found in Ruder (2016), Bottou et al. (2018), and Sun
(2020). Bottou(2012)discussesbestpracticeforSGD,includingshufflingwithoutreplacement.
1Chapter 20 discusses the extent to which saddle points and local minima really are problems in
deeplearning. Inpractice,deepnetworksaresurprisinglyeasytotrain.
Draft: please send errata to udlbookmail@gmail.com.

92 6 Fitting models
Convexity, minima, and saddle points: A function is convex if every chord (line segment
between two points on the surface) lies above the function and does not intersect it. This can
be tested algebraically by considering the Hessian matrix (the matrix of second derivatives):
2 3
∂2L ∂2L ... ∂2L
6 6 6 ∂ ∂ 2 ϕ L 2 0 ∂ϕ ∂ 0 2 ∂ L ϕ1 ... ∂ϕ ∂ 0 2 ∂ L ϕN7 7 7
H[ϕ]=
6
6 6
4
∂ϕ1
. .
.
∂ϕ0 ∂ϕ
. .
.
2
1 ...
∂ϕ1
. .
.
∂ϕN7
7 7
5
. (6.19)
∂2L ∂2L ... ∂2L
∂ϕN∂ϕ0 ∂ϕN∂ϕ1 ∂ϕ2
N
AppendixB.3.7 If the Hessian matrix is positive definite (has positive eigenvalues) for all possible parameter
Eigenvalues values, then the function is convex; the loss function will look like a smooth bowl (as in fig-
ure6.1c),sotrainingwillberelativelyeasy. Therewillbeasingleglobalminimumandnolocal
minima or saddle points.
For any loss function, the eigenvalues of the Hessian matrix at places where the gradient is
zero allow us to classify this position as (i) a minimum (the eigenvalues are all positive), (ii)
a maximum (the eigenvalues are all negative), or (iii) a saddle point (positive eigenvalues are
associated with directions in which we are at a minimum and negative ones with directions
where we are at a maximum).
Line search: Gradientdescentwithafixedstepsizeisineﬀicientbecausethedistancemoved
dependsentirelyonthemagnitudeofthegradient. Itmovesalongdistancewhenthefunction
is changing fast (where perhaps it should be more cautious) but a short distance when the
functionischangingslowly(whereperhapsitshouldexplorefurther). Forthisreason,gradient
descent methods are usually combined with a line search procedure in which we sample the
function along the desired direction to try to find the optimal step size. One such approach
is bracketing (figure 6.10). Another problem with gradient descent is that it tends to lead to
ineﬀicient oscillatory behavior when descending valleys (e.g., path 1 in figure 6.5a).
Beyondgradientdescent: Numerousalgorithmshavebeendevelopedthatremedytheprob-
lemsofgradientdescent. MostnotableistheNewtonmethod,whichtakesthecurvatureofthe
surface into account using the inverse of the Hessian matrix; if the gradient of the function is
changing quickly, then it applies a more cautious update. This method eliminates the need for
line search and does not suffer from oscillatory behavior. However, it has its own problems; in
its simplest form, it moves toward the nearest extremum, but this may be a maximum if we
are closer to the top of a hill than we are to the bottom of a valley. Moreover, computing the
Problem6.11
inverse Hessian is intractable when the number of parameters is large, as in neural networks.
Properties of SGD: The limit of SGD as the learning rate tends to zero is a stochastic
differential equation. Jastrzębski et al. (2018) showed that this equation relies on the learning-
ratetobatchsizeratioandthatthereisarelationbetweenthelearningratetobatchsizeratio
and the width of the minimum found. Wider minima are considered more desirable; if the loss
function for test data is similar, then small errors in the parameter estimates will have little
effect on test performance. He et al. (2019) prove a generalization bound for SGD that has a
positive correlation with the ratio of batch size to learning rate. They train a large number of
models on different architectures and datasets and find empirical evidence that test accuracy
improveswhentheratioofbatchsizetolearningrateislow. Smithetal.(2018)andGoyaletal.
(2018)alsoidentifiedtheratioofbatchsizetolearningrateasbeingimportantforgeneralization
(see figure 20.10).
Momentum: TheideaofusingmomentumtospeedupoptimizationdatestoPolyak(1964).
Goh (2017) presents an in-depth discussion of the properties of momentum. The Nesterov
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 93
Figure6.10Linesearchusingthebracketingapproach. a)Thecurrentsolutionis
atpositiona(orangepoint),andwewishtosearchtheregion[a,d](grayshaded
area). Wedefinetwopointsb,cinteriortothesearchregionandevaluatetheloss
functionatthesepoints. HereL[b]>L[c],soweeliminatetherange[a,b]. b)We
now repeat this procedure in the refined search region and find that L[b]<L[c],
so we eliminate the range [c,d]. c) We repeat this process until this minimum is
closely bracketed.
acceleratedgradientmethodwasintroducedbyNesterov(1983). Nesterovmomentumwasfirst
applied in the context of stochastic gradient descent by Sutskever et al. (2013).
Adaptive training algorithms: AdaGrad (Duchi et al., 2011) is an optimization algorithm
that addresses the possibility that some parameters may have to move further than others by
assigning a different learning rate to each parameter. AdaGrad uses the cumulative squared
gradient for each parameter to attenuate its learning rate. This has the disadvantage that the
learningratesdecreaseovertime,andlearningcanhaltbeforetheminimumisfound. RMSProp
(Hintonetal.,2012a)andAdaDelta(Zeiler,2012)modifiedthisalgorithmtohelppreventthese
problems by recursively updating the squared gradient term.
By far the most widely used adaptive training algorithm is adaptive moment optimization or
Adam (Kingma & Ba, 2015). This combines the ideas of momentum (in which the gradient
vector is averaged over time) and AdaGrad, AdaDelta, and RMSProp (in which a smoothed
squared gradient term is used to modify the learning rate for each parameter). The original
paper on the Adam algorithm provided a convergence proof for convex loss functions, but a
counterexample was identified by Reddi et al. (2018), who developed a modification of Adam
called AMSGrad, which does converge. Of course, in deep learning, the loss functions are non-
convex, and Zaheer et al. (2018) subsequently developed an adaptive algorithm called YOGI
and proved that it converges in this scenario. Regardless of these theoretical objections, the
original Adam algorithm works well in practice and is widely used, not least because it works
well over a broad range of hyperparameters and makes rapid initial progress.
Onepotentialproblemwithadaptivetrainingalgorithmsisthatthelearningratesarebasedon
accumulated statistics of the observed gradients. At the start of training, when there are few
samples, these statistics may be very noisy. This can be remedied by learning rate warm-up
(Goyaletal.,2018),inwhichthelearningratesaregraduallyincreasedoverthefirstfewthou-
sand iterations. An alternative solution is rectified Adam (Liu et al., 2021a), which gradually
Draft: please send errata to udlbookmail@gmail.com.

94 6 Fitting models
changes the momentum term over time in a way that helps avoid high variance. Dozat (2016)
incorporated Nesterov momentum into the Adam algorithm.
SGD vs. Adam: There has been a lively discussion about the relative merits of SGD and
Adam. Wilsonetal.(2017)providedevidencethatSGDwithmomentumcanfindlowerminima
than Adam, which generalizes better over a variety of deep learning tasks. However, this is
strange since SGD is a special case of Adam (when β = 0,γ = 1) once the modification
term (equation 6.16) becomes one, which happens quickly. It is hence more likely that SGD
outperforms Adam when we use Adam’s default hyperparameters. Loshchilov & Hutter (2019)
proposed AdamW, which substantially improves the performance of Adam in the presence of
L2regularization(seesection9.1). Choietal.(2019)provideevidencethatifwesearchforthe
best Adam hyperparameters, it performs just as well as SGD and converges faster. Keskar &
Socher(2017)proposedamethodcalledSWATSthatstartsusingAdam(tomakerapidinitial
progress) and then switches to SGD (to get better final generalization performance).
Exhaustive search: All the algorithms discussed in this chapter are iterative. A completely
different approach is to quantize the network parameters and exhaustively search the resulting
discretized parameter space using SAT solvers (Mézard & Mora, 2009). This approach has
the potential to find the global minimum and provide a guarantee that there is no lower loss
elsewhere but is only practical for very small models.
Problems
Problem 6.1 Show that the derivatives of the least squares loss function in equation 6.5 are
given by the expressions in equation 6.7.
Problem 6.2 A surface is guaranteed to be convex if the eigenvalues of the Hessian H[ϕ] are
positive everywhere. In this case, the surface has a unique minimum, and optimization is easy.
Find an algebraic expression for the Hessian matrix,
2 3
∂2L ∂2L
H[ϕ]=
4 ∂ϕ2
0
∂ϕ0∂ϕ15
, (6.20)
∂2L ∂2L
∂ϕ1∂ϕ0 ∂ϕ2
1
AppendixB.3.7 for the linear regression model (equation 6.5). Prove that this function is convex by showing
Eigenvalues that the eigenvalues are always positive. This can be done by showing that both the trace and
the determinant of the matrix are positive.
AppendixB.3.8
Trace Problem 6.3 Computethe derivativesof theleast squares loss L[ϕ] withrespect to the param-
eters ϕ and ϕ for the Gabor model (equation 6.8).
AppendixB.3.8 0 1
Determinant
Problem 6.4∗ The logistic regression model uses a linear function to assign an input x to one
of two classes y ∈{0,1}. For a 1D input and a 1D output, it has two parameters, ϕ and ϕ ,
0 1
and is defined by:
Pr(y=1|x)=sig[ϕ +ϕ x], (6.21)
0 1
where sig[•] is the logistic sigmoid function:
1
sig[z]= . (6.22)
1+exp[−z]
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 95
Figure 6.11 Three 1D loss functions for problem 6.6.
(i) Plot y against x for this model for different values of ϕ and ϕ and explain the qualitative
0 1
meaning of each parameter. (ii) What is a suitable loss function for this model? (iii) Compute
the derivatives of this loss function with respect to the parameters. (iv) Generate ten data
points from a normal distribution with mean -1 and standard deviation 1 and assign them the
label y = 0. Generate another ten data points from a normal distribution with mean 1 and
standard deviation 1 and assign these the label y =1. Plot the loss as a heatmap in terms of
the two parameters ϕ and ϕ . (v) Is this loss function convex? How could you prove this?
0 1
Problem 6.5∗ Compute the derivatives of the least squares loss with respect to the ten param-
eters of the simple neural network model introduced in equation 3.1:
f[x,ϕ]=ϕ +ϕ a[θ +θ x]+ϕ a[θ +θ x]+ϕ a[θ +θ x]. (6.23)
0 1 10 11 2 20 21 3 30 31
Think carefully about what the derivative of the ReLU function a[•] will be.
Problem6.6Whichofthefunctionsinfigure6.11isconvex? Justifyyouranswer. Characterize
each of the points 1–7 as (i) a local minimum, (ii) the global minimum, or (iii) neither.
Problem 6.7∗ Thegradientdescenttrajectoryforpath1infigure6.5aoscillatesbackandforth
ineﬀicientlyasitmovesdownthevalleytowardtheminimum. It’salsonotablethatitturnsat
right angles to the previous direction at each step. Provide a qualitative explanation for these
phenomena. Propose a solution that might help prevent this behavior.
Problem 6.8∗ Can (non-stochastic) gradient descent with a fixed learning rate escape local
minima?
Problem 6.9 Werunthestochasticgradientdescentalgorithmfor1,000iterationsonadataset
of size 100 with a batch size of 20. For how many epochs did we train the model?
Problem 6.10 Show that the momentum term m (equation 6.11) is an infinite weighted sum
t
ofthegradientsatthepreviousiterationsandderiveanexpressionforthecoeﬀicients(weights)
of that sum.
Problem 6.11 WhatdimensionswilltheHessianhaveifthemodelhasonemillionparameters?
Draft: please send errata to udlbookmail@gmail.com.