# Chapter 4: To perform inference for a new test example x, return either the full distribu-

*Pages: 74-90*

---

60 5 Loss functions
5.1.4 Minimizing negative log-likelihood
Finally, we note that, by convention, model fitting problems are framed in terms of
minimizing a loss. To convert the maximum log-likelihood criterion to a minimization
problem, we multiply by minus one, which gives us the negative log-likelihood criterion:
" #
XI h i
ϕˆ = argmin − log Pr(y |f[x ,ϕ])
i i
ϕ
h i=i1
= argmin L[ϕ] , (5.4)
ϕ
which is what forms the final loss function L[ϕ].
5.1.5 Inference
The network no longer directly predicts the outputs y but instead determines a proba-
bility distribution over y. When we perform inference, we often want a point estimate
rather than a distribution, so we return the maximum of the distribution:
h i
yˆ =argmax Pr(y|f[x,ϕˆ]) . (5.5)
y
It is usually possible to find an expression for this in terms of the distribution parame-
ters θ predicted by the model. For example, in the univariate normal distribution, the
maximum occurs at the mean µ.
5.2 Recipe for constructing loss functions
The recipe for constructing loss functions for training data {x ,y } using the maximum
i i
likelihood approach is hence:
1. Choose a suitable probability distribution Pr(y|θ) defined over the domain of the
predictions y with distribution parameters θ.
2. Set the machine learning model f[x,ϕ] to predict one or more of these parameters,
so θ =f[x,ϕ] and Pr(y|θ)=Pr(y|f[x,ϕ]).
3. To train the model, find the network parameters ϕˆ that minimize the negative
log-likelihood loss function over the training dataset pairs {x ,y }:
i i
" #
h i XI h i
ϕˆ =argmin L[ϕ] =argmin − log Pr(y |f[x ,ϕ]) . (5.6)
i i
ϕ ϕ
i=1
4. To perform inference for a new test example x, return either the full distribu-
tion Pr(y|f[x,ϕˆ]) or the value where this distribution is maximized.
We devote most of the rest of this chapter to constructing loss functions for common
prediction types using this recipe.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

5.3 Example 1: univariate regression 61
Figure 5.3 The univariate normal distri-
bution (also known as the Gaussian dis-
tribution) is defined on the real line z ∈
R and has parameters µ and σ2. The
mean µ determines the position of the
peak. The positive root of the vari-
ance σ2 (the standard deviation) de-
termines the width of the distribution.
Since the total probability density sums
to one, the peak becomes higher as the
variance decreases and the distribution
becomes narrower.
5.3 Example 1: univariate regression
Westartbyconsideringunivariateregressionmodels. Herethegoalistopredictasingle
scalar output y ∈ R from input x using a model f[x,ϕ] with parameters ϕ. Following
therecipe, wechooseaprobabilitydistributionovertheoutputdomainy. Weselectthe
univariate normal distribution (figure 5.3), which is defined over y ∈ R. This has two
parameters (mean µ and variance σ2) and has a probability density function:
(cid:20) (cid:21)
1 (y−µ)2
Pr(y|µ,σ2)= √ exp − . (5.7)
2πσ2 2σ2
Second,wesetthemachinelearningmodelf[x,ϕ]tocomputeoneormoreoftheparam-
eters of this distribution. Here, we just compute the mean so µ=f[x,ϕ]:
(cid:20) (cid:21)
1 (y−f[x,ϕ])2
Pr(y|f[x,ϕ],σ2)= √ exp − . (5.8)
2πσ2 2σ2
We aim to find the parameters ϕ that make the training data {x ,y } most probable
i i
under this distribution (figure 5.4). To accomplish this, we choose a loss function L[ϕ]
based on the negative log-likelihood:
XI (cid:2) (cid:3)
L[ϕ] = − log Pr(y |f[x ,ϕ],σ2)
i i
i=1
(cid:20) (cid:20) (cid:21)(cid:21)
XI 1 (y −f[x ,ϕ])2
= − log √ exp − i i . (5.9)
2πσ2 2σ2
i=1
When we train the model, we seek parameters ϕˆ that minimize this loss.
Draft: please send errata to udlbookmail@gmail.com.

62 5 Loss functions
5.3.1 Least squares loss function
Now let’s perform some algebraic manipulations on the loss function. We seek:
" (cid:20) (cid:20) (cid:21)(cid:21)#
XI 1 (y −f[x ,ϕ])2
ϕˆ = argmin − log √ exp − i i
ϕ 2πσ2 2σ2
" i=1 (cid:18) (cid:20) (cid:21) (cid:19)#
XI 1 (y −f[x ,ϕ])2
= argmin − log √ − i i
ϕ 2πσ2 2σ2
" i=1 #
XI (y −f[x ,ϕ])2
= argmin − − i i
2σ2
ϕ
" i=1 #
XI
= argmin (y −f[x ,ϕ])2 , (5.10)
i i
ϕ
i=1
where we removed the first term between the second and third lines because it doesn’t
dependonϕ. Weremovedthedenominatorbetweenthethirdandfourthlines,asthisis
just a constant positive scaling factor that does not affect the position of the minimum.
The result of these manipulations is the least squares loss function that we originally
introduced when we discussed linear regression in chapter 2:
XI (cid:0) (cid:1)
L[ϕ]= y −f[x ,ϕ] 2 . (5.11)
i i
i=1
Weseethattheleastsquareslossfunctionfollowsnaturallyfromtheassumptionsthatthe
Notebook5.1
predictionsare(i)independentand(ii)drawnfromanormaldistributionwithmeanµ=
Leastsquares
loss f[x i ,ϕ] (figure 5.4).
5.3.2 Inference
The network no longer directly predicts y but instead predicts the mean µ = f[x,ϕ] of
the normal distribution over y. When we perform inference, we usually want a single
“best” point estimate yˆ, so we take the maximum of the predicted distribution:
h i
yˆ=argmax Pr(y|f[x,ϕˆ],σ2) . (5.12)
y
Fortheunivariatenormaldistribution,themaximumpositionisdeterminedbythemean
parameter µ (figure 5.3). This is precisely what the model computed, so yˆ=f[x,ϕˆ].
5.3.3 Estimating variance
To formulate the least squares loss function, we assumed that the network predicted the
mean of a normal distribution. The final expression in equation 5.11 (perhaps surpris-
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

5.3 Example 1: univariate regression 63
Figure 5.4 Equivalence of least squares and maximum likelihood loss for the
normal distribution. a) Consider the linear model from figure 2.2. The least
squarescriterionminimizesthesumofthesquaresofthedeviations(dashedlines)
between the model prediction f[x ,ϕ] (green line) and the true output values y
i i
(orange points). Here the fit is good, so these deviations are small (e.g., for the
two highlighted points). b) For these parameters, the fit is bad, and the squared
deviations are large. c) The least squares criterion follows from the assumption
that the model predicts the mean of a normal distribution over the outputs and
that we maximize the probability. For the first case, the model fits well, so the
probability Pr(y |x ) of the data (horizontal orange dashed lines) is large (and
i i
thenegativelogprobabilityissmall). d)Forthesecondcase,themodelfitsbadly,
so the probability is small and the negative log probability is large.
Draft: please send errata to udlbookmail@gmail.com.

64 5 Loss functions
ingly) does not depend on the variance σ2. However, there is nothing to stop us from
treatingσ2 asalearnedparameterandminimizingequation5.9withrespecttoboththe
model parameters ϕ and the distribution variance σ2:
" (cid:20) (cid:20) (cid:21)(cid:21)#
XI 1 (y −f[x ,ϕ])2
ϕˆ,σˆ2 =argmin − log √ exp − i i . (5.13)
ϕ,σ2
i=1
2πσ2 2σ2
Ininference, themodelpredictsthemeanµ=f[x,ϕˆ]fromtheinput, andwelearnedthe
variance σˆ2 during the training process. The former is the best prediction. The latter
tells us about the uncertainty of the prediction.
5.3.4 Heteroscedastic regression
Themodelaboveassumesthatthevarianceofthedataisconstanteverywhere. However,
this might be unrealistic. When the uncertainty of the model varies as a function of the
input data, we refer to this as heteroscedastic (as opposed to homoscedastic, where the
uncertainty is constant).
A simple way to model this is to train a neural network f[x,ϕ] that computes both
the mean and the variance. For example, consider a shallow network with two outputs.
We denote the first output as f [x,ϕ] and use this to predict the mean, and we denote
1
the second output as f [x,ϕ] and use it to predict the variance.
2
There is one complication; the variance must be positive, but we can’t guarantee
that the network will always produce a positive output. To ensure that the computed
variance is positive, we pass the second network output through a function that maps
an arbitrary value to a positive one. A suitable choice is the squaring function, giving:
µ = f [x,ϕ]
1
σ2 = f [x,ϕ]2, (5.14)
2
which results in the loss function:
" (cid:18) " # (cid:19)#
XI 1 (y −f [x ,ϕ])2
ϕˆ =argmin − log p − i 1 i . (5.15)
ϕ i=1 2πf 2 [x i ,ϕ]2 2f 2 [x i ,ϕ]2
Homoscedastic and heteroscedastic models are compared in figure 5.5.
5.4 Example 2: binary classification
Inbinary classification, thegoalistoassignthedataxtooneoftwodiscreteclassesy ∈
{0,1}. In this context, we refer to y as a label. Examples of binary classification include
(i) predicting whether a restaurant review is positive (y = 1) or negative (y = 0) from
text data x and (ii) predicting whether a tumor is present (y = 1) or absent (y = 0)
from an MRI scan x.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

5.4 Example 2: binary classification 65
Figure 5.5 Homoscedastic vs. heteroscedastic regression. a) A shallow neural
network for homoscedastic regression predicts just the mean µ of the output
distribution from the input x. b) The result is that while the mean (blue line)
is a piecewise linear function of the input x, the variance is constant everywhere
(arrows and gray region show ±2 standard deviations). c) A shallow neural
network for heteroscedastic regression also predicts the variance σ2 (or, more
precisely, computes its square root, which we then square). d) The standard
deviation now also becomes a piecewise linear function of the input x.
Figure 5.6 Bernoulli distribution. The
Bernoulli distribution is defined on the
domain z ∈ {0,1} and has a single pa-
rameter λ that denotes the probability
of observing z = 1. It follows that the
probability of observing z=0 is 1−λ.
Draft: please send errata to udlbookmail@gmail.com.

66 5 Loss functions
Figure 5.7 Logistic sigmoid function.
This function maps the real line z ∈
R to numbers between zero and one,
sosig[z]∈[0,1]. Aninputof0ismapped
to 0.5. Negative inputs are mapped to
numbers below 0.5, and positive inputs
to numbers above 0.5.
Onceagain,wefollowtherecipefromsection5.2toconstructthelossfunction. First,
we choose a probability distribution over the output space y ∈{0,1}. A suitable choice
is the Bernoulli distribution, which is defined on the domain {0,1}. This has a single
parameterλ∈[0,1]thatrepresentstheprobabilitythatytakesthevalueone(figure5.6):
(
1−λ y =0
Pr(y|λ)= , (5.16)
λ y =1
which can equivalently be written as:
Pr(y|λ)=(1−λ)1−y·λy.
(5.17)
Second, we set the machine learning model f[x,ϕ] to predict the single distribution
parameterλ. However,λcanonlytakevaluesintherange[0,1],andwecannotguarantee
thatthenetworkoutputwilllieinthisrange. Consequently,wepassthenetworkoutput
through a function that maps the real numbers R to [0,1]. A suitable function is the
logistic sigmoid (figure 5.7):
Problem5.1
1
sig[z]= . (5.18)
1+exp[−z]
Hence, we predict the distribution parameter as λ=sig[f[x,ϕ]]. The likelihood is now:
Pr(y|x)=(1−sig[f[x,ϕ]])1−y·sig[f[x,ϕ]]y.
(5.19)
This is depicted in figure 5.8 for a shallow neural network model. The loss function is
the negative log-likelihood of the training set:
XI h i h i
L[ϕ]= −(1−y )log 1−sig[f[x ,ϕ]] −y log sig[f[x ,ϕ]] . (5.20)
i i i i
i=1
Forreasonstobeexplainedinsection5.7,thisisknownasthebinary cross-entropy loss.
Notebook5.2
Binary The transformed model output sig[f[x,ϕ]] predicts the parameter λ of the Bernoulli
cross-entropyloss distribution. This represents the probability that y = 1, and it follows that 1 − λ
Problem5.2 represents the probability that y =0. When we perform inference, we may want a point
estimate of y, so we set y =1 if λ>0.5 and y =0 otherwise.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

5.5 Example 3: multiclass classification 67
Figure 5.8 Binary classification model. a) The network output is a piecewise
linearfunctionthatcantakearbitraryrealvalues. b)Thisistransformedbythe
logistic sigmoid function, which compresses these values to the range [0,1]. c)
The transformed output predicts the probability λ that y = 1 (solid line). The
probability that y = 0 is hence 1−λ (dashed line). For any fixed x (vertical
slice), we retrieve the two values of a Bernoulli distribution similar to that in
figure 5.6. The loss function favors model parameters that produce large values
of λ at positions x that are associated with positive examples y =1 and small
i i
values of λ at positions associated with negative examples y =0.
i
Figure 5.9Categoricaldistribution. The
categoricaldistributionassignsprobabil-
itiestoK>2categories,withassociated
probabilities λ ,λ ,...,λ . Here, there
1 2 K
arefivecategories,so K =5. Toensure
that this is a valid probability distribu-
tion, each parameter λ must lie in the
k
range [0,1], and all K parameters must
sum to one.
5.5 Example 3: multiclass classification
ThegoalofmulticlassclassificationistoassignaninputdataexamplextooneofK >2
classes,soy ∈{1,2,...,K}. Real-worldexamplesinclude(i)predictingwhichofK =10
digitsy ispresentinanimagexofahandwrittennumberand(ii)predictingwhichofK
possible words y follows an incomplete sentence x.
We once more follow the recipe from section 5.2. We first choose a distribution
over the prediction space y. In this case, we have y ∈ {1,2,...,K}, so we choose
the categorical distribution (figure 5.9), which is defined on this domain. This has K
parameters λ ,λ ,...,λ , which determine the probability of each category:
1 2 K
Draft: please send errata to udlbookmail@gmail.com.

68 5 Loss functions
Figure 5.10 Multiclass classification for K=3 classes. a) The network has three
piecewise linear outputs, which can take arbitrary values. b) After the softmax
function,theseoutputsareconstrainedtobenon-negativeandsumtoone. Hence,
foragiveninputx,wecomputevalidparametersforthecategoricaldistribution:
any vertical slice of this plot produces three values that sum to one and would
form the heights of the bars in a categorical distribution similar to figure 5.9.
Pr(y =k)=λ . (5.21)
k
The parameters are constrained to take values between zero and one, and they must
collectively sum to one to ensure a valid probability distribution.
Then we use a network f[x,ϕ] with K outputs to compute these K parameters from
the input x. Unfortunately, the network outputs will not necessarily obey the afore-
mentioned constraints. Consequently, we pass the K outputs of the network through a
function that ensures these constraints are respected. A suitable choice is the softmax
function (figure 5.10). This takes an arbitrary vector of length K and returns a vector
of the same length but where the elements are now in the range [0,1] and sum to one.
The kth output of the softmax function is:
exp[z ]
softmax [z]= P k , (5.22)
k K
k′=1
exp[z k′]
where the exponential functions ensure positivity, and the sum in the denominator en-
AppendixB.1.3
sures that the K numbers sum to one.
Exponential
function The likelihood that input x has label y =k (figure 5.10) is hence:
h i
Pr(y =k|x)=softmax f[x,ϕ] . (5.23)
k
The loss function is the negative log-likelihood of the training data:
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

5.6 Multiple outputs 69
XI h h ii
L[ϕ] = − log softmax f[x ,ϕ]
yi i
i=1 " #!
XI XK
= − f
yi
[x
i
,ϕ]−log exp[ f k′[x
i
,ϕ]] , (5.24)
i=1 k′=1
wheref
yi
[x,ϕ]andf k′[x,ϕ]denotethey
i
th andk′th outputsofthenetwork, respectively.
For reasons that will be explained in section 5.7, this is known as the multiclass cross-
entropy loss.
The transformed model output represents a categorical distribution over possible
classes y ∈{1,2,...,K}. For a point estimate, we take the most probable category yˆ= Notebook5.3
(cid:2) (cid:3) Multiclass
argmax Pr(y =k|f[x,ϕˆ]) . This corresponds to whichever curve is highest for that cross-entropyloss
k
value of x in figure 5.10.
5.5.1 Predicting other data types
In this chapter, we have focused on regression and classification because these problems
are widespread. However, to make different types of predictions, we simply choose an
appropriatedistributionoverthatdomainandapplytherecipeinsection5.2. Figure5.11
enumerates a series of probability distributions and their prediction domains. Some of
Problems5.3–5.6
these are explored in the problems at the end of the chapter.
5.6 Multiple outputs
Often, we wish to make more than one prediction with the same model, so the target
output y is a vector. For example, we might want to predict a molecule’s melting
and boiling point (a multivariate regression problem, figure 1.2b) or the object class at
every point in an image (a multivariate classification problem, figure 1.4a). While it
is possible to define multivariate probability distributions and use a neural network to
modeltheirparametersasafunctionoftheinput,itismoreusualtotreateachprediction
as independent.
Independence implies that we treat the probability Pr(y|f[x,ϕ]) as a product of
univariate terms for each element y ∈y: AppendixC.1.5
d Independence
Y
Pr(y|f[x,ϕ])= Pr(y |f [x,ϕ]), (5.25)
d d
d
where f [x,ϕ] is the dth set of network outputs, which describe the parameters of the
d
distribution over y . For example, to predict multiple continuous variables y ∈ R, we
d d
useanormaldistributionforeachy ,andthenetworkoutputsf [x,ϕ]predictthemeans
d d
ofthese distributions. Topredict multiplediscrete variablesy ∈{1,2,...,K}, weuse a
d
Draft: please send errata to udlbookmail@gmail.com.

70 5 Loss functions
Data Type Domain Distribution Use
univariate, continuous, y∈R univariate regression
unbounded normal
univariate, continuous, y∈R Laplace robust
unbounded or t-distribution regression
univariate, continuous, y∈R mixture of multimodal
unbounded Gaussians regression
univariate, continuous, y∈R+ exponential predicting
bounded below or gamma magnitude
univariate, continuous, y∈[0,1] beta predicting
bounded proportions
multivariate, continuous, y∈RK multivariate multivariate
unbounded normal regression
univariate, continuous, y∈(−π,π] von Mises predicting
circular direction
univariate, discrete, y∈{0,1} Bernoulli binary
binary classification
univariate, discrete, y∈{1,2,...,K} categorical multiclass
bounded classification
univariate, discrete, y∈{0,1,2,3,...} Poisson predicting
bounded below event counts
multivariate, discrete, y∈Perm[1,2,...,K] Plackett-Luce ranking
permutation
Figure 5.11 Distributions for loss functions for different prediction types.
categorical distribution for each y . Here, each set of network outputs f [x,ϕ] predicts
d d
the K values that contribute to the categorical distribution for y .
d
Whenweminimizethenegativelogprobability,thisproductbecomesasumofterms:
XI h i XI X h i
L[ϕ]=− log Pr(y |f[x ,ϕ]) =− log Pr(y |f [x ,ϕ]) . (5.26)
i i id d i
i=1 i=1 d
where y is the dth output from the ith training example.
id
Tomaketwoormorepredictiontypessimultaneously,wesimilarlyassumetheerrors
in each are independent. For example, to predict wind direction and strength, we might
Problems5.7–5.10
choose the von Mises distribution (defined on circular domains) for the direction and
the exponential distribution (defined on positive real numbers) for the strength. The
independence assumption implies that the joint likelihood of the two predictions is the
product of individual likelihoods. These terms will become additive when we compute
the negative log-likelihood.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

5.7 Cross-entropy loss 71
Figure 5.12Cross-entropymethod. a)Empiricaldistributionoftrainingsamples
(arrows denote Dirac delta functions). b) Model distribution (a normal distribu-
tion with parameters θ = {µ,σ2}). In the cross-entropy approach, we minimize
thedistance(KLdivergence)betweenthesetwodistributionsasafunctionofthe
model parameters θ.
5.7 Cross-entropy loss
In this chapter, wedeveloped loss functions that minimize negative log-likelihood. How-
ever, the term cross-entropy loss is also commonplace. In this section, we describe the
cross-entropy loss and show that it is equivalent to using negative log-likelihood.
Thecross-entropylossisbasedontheideaoffindingparametersθ thatminimizethe
distancebetweentheempiricaldistributionq(y)oftheobserveddatay andamodeldis-
tribution Pr(y|θ) (figure 5.12). The distance between two probability distributions q(z)
AppendixC.5.1
and p(z) can be evaluated using the Kullback-Leibler (KL) divergence:
KLDivergence
Z Z
(cid:2) (cid:3) ∞ (cid:2) (cid:3) ∞ (cid:2) (cid:3)
D q||p = q(z)log q(z) dz− q(z)log p(z) dz. (5.27)
KL
−∞ −∞
Now consider that we observe an empirical data distribution at points {y }I . We
i i=1
can describe this as a weighted sum of point masses:
XI
1
q(y)= δ[y−y ], (5.28)
I i
i=1
where δ[•] is the Dirac delta function. We want to minimize the KL divergence between AppendixB.1.3
the model distribution Pr(y|θ) and this empirical distribution: Diracdelta
function
(cid:20)Z Z (cid:21)
∞ (cid:2) (cid:3) ∞ (cid:2) (cid:3)
θˆ = argmin q(y)log q(y) dy− q(y)log Pr(y|θ) dy
θ (cid:20) −Z∞ −∞(cid:21)
∞ (cid:2) (cid:3)
= argmin − q(y)log Pr(y|θ) dy , (5.29)
θ −∞
Draft: please send errata to udlbookmail@gmail.com.

72 5 Loss functions
where the first term disappears, as it has no dependence on θ. The remaining second
term is known as the cross-entropy. It can be interpreted as the amount of uncertainty
that remains in one distribution after taking into account what we already know from
the other. Now, we substitute in the definition of q(y) from equation 5.28:
" ! #
Z
∞ XI (cid:2) (cid:3)
1
θˆ = argmin − δ[y−y ] log Pr(y|θ) dy
I i
θ −∞
" i=1 #
XI (cid:2) (cid:3)
1
= argmin − log Pr(y |θ)
I i
θ
" i=1 #
XI (cid:2) (cid:3)
= argmin − log Pr(y |θ) . (5.30)
i
θ
i=1
The product of the two terms in the first line corresponds to pointwise multiplying the
point masses in figure 5.12a with the logarithm of the distribution in figure 5.12b. We
are left with a finite set of weighted probability masses centered on the data points. In
the last line, we have eliminated the constant scaling factor 1/I, as this does not affect
the position of the minimum.
Inmachinelearning,thedistributionparametersθarecomputedbythemodelf[x ,ϕ],
i
so we have:
" #
XI (cid:2) (cid:3)
ϕˆ =argmin − log Pr(y |f[x ,ϕ]) . (5.31)
i i
ϕ
i=1
This is precisely the negative log-likelihood criterion from the recipe in section 5.2.
It follows that the negative log-likelihood criterion (from maximizing the data likeli-
hood) and the cross-entropy criterion (from minimizing the distance between the model
and empirical data distributions) are equivalent.
5.8 Summary
We previously considered neural networks as directly predicting outputs y from data x.
In this chapter, we shifted perspective to think about neural networks as computing the
parameters θ of probability distributions Pr(y|θ) over the output space. This led to a
principled approach to building loss functions. We selected model parameters ϕ that
maximized the likelihood of the observed data under these distributions. We saw that
this is equivalent to minimizing the negative log-likelihood.
The least squares criterion for regression is a natural consequence of this approach;
it follows from the assumption that y is normally distributed and that we are predicting
the mean. We also saw how the regression model could be (i) extended to estimate the
uncertainty over the prediction and (ii) extended to make that uncertainty dependent
ontheinput(theheteroscedasticmodel). Weappliedthesameapproachtoboth binary
and multiclass classification and derived loss functions for each. We discussed how to
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 73
tackle more complex data types and how to deal with multiple outputs. Finally, we
argued that cross-entropy is an equivalent way to think about fitting models.
In previous chapters, we developed neural network models. In this chapter, we de-
veloped loss functions for deciding how well a model describes the training data for a
given set of parameters. The next chapter considers model training, in which we aim to
find the model parameters that minimize this loss.
Notes
Losses based on the normal distribution: Nix & Weigend (1994) and Williams (1996)
investigated heteroscedastic nonlinear regression in which both the mean and the variance of
the output are functions of the input. In the context of unsupervised learning, Burda et al.
(2016)usealossfunctionbasedonamultivariatenormaldistributionwithdiagonalcovariance,
andDortaetal. (2018)usea lossfunction basedon anormal distributionwith fullcovariance.
Robust regression: Qietal.(2020)investigatethepropertiesofregressionmodelsthatmin-
imize mean absolute error rather than mean squared error. This loss function follows from
assuming a Laplace distribution over the outputs and estimates the median output for a given
input rather than the mean. Barron (2019) presents a loss function that parameterizes the de-
gree of robustness. When interpreted in a probabilistic context, it yields a family of univariate
probability distributions that includes the normal and Cauchy distributions as special cases.
Estimating quantiles: Sometimes, we may not want to estimate the mean or median in a
regression task but may instead want to predict a quantile. For example, this is useful for risk
models, where we want to know that the true value will be less than the predicted value 90%
of the time. This is known as quantile regression (Koenker & Hallock, 2001). This could be
done by fitting a heteroscedastic regression model and then estimating the quantile based on
the predicted normal distribution. Alternatively, the quantiles can be estimated directly using
quantile loss (also known as pinball loss). In practice, this minimizes the absolute deviations
of the data from the model but weights the deviations in one direction more than the other.
Recentworkhasinvestigatedsimultaneouslypredictingmultiplequantilestogetanideaofthe
overall distribution shape (Rodrigues & Pereira, 2020).
Class imbalance and focal loss: Lin et al. (2017c) address data imbalance in classification
problems. Ifthenumberofexamplesforsomeclassesismuchgreaterthanforothers,thenthe
standardmaximumlikelihoodlossdoesnotworkwell;themodelmayconcentrateonbecoming
more confident about well-classified examples from the dominant classes and classify less well-
represented classes poorly. Lin et al. (2017c) introduce focal loss, which adds a single extra
parameter that down-weights the effect of well-classified examples to improve performance.
Learning to rank: Cao et al. (2007), Xia et al. (2008), and Chen et al. (2009) all used the
Plackett-Lucemodelinlossfunctionsforlearningtorankdata. Thisisthelistwiseapproachto
learningtorankasthemodelingestsanentirelistofobjectstoberankedatonce. Alternative
approaches are the pointwise approach, in which the model ingests a single object, and the
pairwise approach, where the model ingests pairs of objects. Chen et al. (2009) summarize
different approaches for learning to rank.
Other data types: Fan et al. (2020) use a loss based on the beta distribution for predicting
values between zero and one. Jacobs et al. (1991) and Bishop (1994) investigated mixture
density networks for multimodal data. These model the output as a mixture of Gaussians
Draft: please send errata to udlbookmail@gmail.com.

74 5 Loss functions
Figure 5.13 The von Mises distribu-
tion is defined over the circular do-
main (−π,π]. It has two parameters.
The mean µ determines the position
of the peak. The concentration κ >
0 acts like th√e inverse of the vari-
ance. Hence 1/ κ is roughly equivalent
to the standard deviation in a normal
distribution.
(see figure 5.14) that is conditional on the input. Prokudin et al. (2018) used the von Mises
distributiontopredictdirection(seefigure5.13). Fallahetal.(2009)constructedlossfunctions
forpredictioncountsusingthePoissondistribution(seefigure5.15). Ngetal.(2017)usedloss
functions based on the gamma distribution to predict duration.
Non-probabilistic approaches: It is not strictly necessary to adopt the probabilistic ap-
proachdiscussedinthischapter,butthishasbecomethedefaultinrecentyears;anylossfunc-
tion that aims to reduce the distance between the model output and the training outputs will
suﬀice,anddistancecanbedefinedinanywaythatseemssensible. Thereareseveralwell-known
non-probabilistic machinelearning models for classification, including support vectormachines
(Vapnik,1995;Cristianini&Shawe-Taylor,2000),whichusehinge loss,andAdaBoost(Freund
& Schapire, 1997), which uses exponential loss.
Problems
Problem 5.1 Show that the logistic sigmoid function sig[z] becomes 0 as z → −∞, is 0.5
when z=0, and becomes 1 when z→∞, where:
1
sig[z]= . (5.32)
1+exp[−z]
Problem 5.2 The loss L for binary classification for a single training pair {x,y} is:
h i h i
L=−(1−y)log 1−sig[f[x,ϕ]] −ylog sig[f[x,ϕ]] , (5.33)
wheresig[•]isdefinedinequation5.32. Plotthislossasafunctionofthetransformednetwork
output sig[f[x,ϕ]]∈[0,1] (i) when the training label y=0 and (ii) when y=1.
Problem5.3∗ Supposewewanttobuildamodelthatpredictsthedirectiony inradiansofthe
prevailing wind based on local measurements of barometric pressure x. A suitable distribution
over circular domains is the von Mises distribution (figure 5.13):
(cid:2) (cid:3)
exp κcos[y−µ]
Pr(y|µ,κ)= , (5.34)
2π·Bessel [κ]
0
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 75
Figure 5.14 Multimodal data and mixture of Gaussians density. a) Example
training data where, for intermediate values of the input x, the corresponding
output y follows one of two paths. For example, at x = 0, the output y might
be roughly −2 or +3 but is unlikely to be between these values. b) The mixture
of Gaussians is a probability model suited to this kind of data. As the name
suggests, the model is a weighted sum (solid cyan curve) of two or more normal
distributionswithdifferentmeansandvariances(here,twonormaldistributions,
dashed blue and orange curves). When the means are far apart, this forms a
multimodal distribution. c) When the means are close, the mixture can model
unimodal but non-normal densities.
whereµisameasureofthemeandirectionandκisameasureofconcentration(i.e.,theinverse
of the variance). The term Bessel [κ] is a modified Bessel function of the first kind of order 0.
0
Use the recipe from section 5.2 to develop a loss function for learning the parameter µ of a
model f[x,ϕ] to predict the most likely wind direction. Your solution should treat the concen-
tration κ as constant. How would you perform inference?
Problem 5.4∗ Sometimes, the outputs y for input x are multimodal (figure 5.14a); there is
morethanonevalidpredictionforagiveninput. Here,wemightuseaweightedsumofnormal
componentsasthedistributionovertheoutput. ThisisknownasamixtureofGaussiansmodel.
For example, a mixture of two Gaussians has parameters θ={λ,µ ,σ2,µ ,σ2}:
1 1 2 2
(cid:20) (cid:21) (cid:20) (cid:21)
λ −(y−µ )2 1−λ −(y−µ )2
Pr(y|λ,µ ,µ ,σ2,σ2)= p exp 1 + p exp 2 , (5.35)
1 2 1 2 2πσ2 2σ2 2πσ2 2σ2
1 1 2 2
where λ ∈ [0,1] controls the relative weight of the two components, which have means µ ,µ
1 2
and variances σ2, σ2, respectively. This model can represent a distribution with two peaks
1 2
(figure 5.14b) or a distribution with one peak but a more complex shape (figure 5.14c).
Usetherecipefromsection5.2toconstructalossfunctionfortrainingamodelf[x,ϕ]thattakes
input x, has parameters ϕ, and predicts a mixture of two Gaussians. The loss should be based
on I training data pairs {x ,y }. What problems do you foresee when performing inference?
i i
Problem5.5Considerextendingthemodelfromproblem5.3topredictthewinddirectionusing
a mixture of two von Mises distributions. Write an expression for the likelihood Pr(y|θ) for
this model. How many outputs will the network need to produce?
Draft: please send errata to udlbookmail@gmail.com.

76 5 Loss functions
Figure 5.15 Poisson distribution. This discrete distribution is defined over non-
negative integers z ∈ {0,1,2,...}. It has a single parameter λ ∈ R+, which is
knownastherateandisthemeanofthedistribution. a–c)Poissondistributions
with rates of 1.4, 2.8, and 6.0, respectively.
Problem 5.6 Consider building a model to predict the number of pedestrians y ∈ {0,1,2,...}
that will pass a given point in the city in the next minute, based on data x that contains
information about the time of day, the longitude and latitude, and the type of neighborhood.
A suitable distribution for modeling counts is the Poisson distribution (figure 5.15). This has
a single parameter λ > 0 called the rate that represents the mean of the distribution. The
distribution has probability density function:
λke−λ
Pr(y=k)= . (5.36)
k!
Design a loss function for this model assuming we have access to I training pairs {x ,y }.
i i
Problem 5.7 Consider a multivariate regression problem where we predict ten outputs, so y∈
R10, and model each with an independent normal distribution where the means µ are pre-
d
dicted by the network, and variances σ2 are constant. Write an expression for the likeli-
hood Pr(y|f[x,ϕ]). Show that minimizing the negative log-likelihood of this model is still
equivalent to minimizing a sum of squared terms if we don’t estimate the variance σ2.
Problem 5.8∗ Construct a loss function for making multivariate predictions y ∈ RDo based
on independent normal distributions with different variances σ2 for each dimension. Assume
d
a heteroscedastic model so that both the means µ and variances σ2 vary as a function of the
d d
data.
Problem 5.9∗ Consider a multivariate regression problem in which we predict the height of a
person in meters and their weight in kilos from data x. Here, the units take quite different
ranges. What problems do you see this causing? Propose two solutions to these problems.
Problem 5.10 Extend the model from problem 5.3 to predict both the wind direction and the
wind speed and define the associated loss function.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.