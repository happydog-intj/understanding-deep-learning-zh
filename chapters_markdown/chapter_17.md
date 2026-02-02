# Chapter 17

*Pages: 341-362*

---

Chapter 17
Variational autoencoders
Generative adversarial networks learn a mechanism for creating samples that cannot
be distinguished from the training examples {x }. In contrast, like normalizing flows,
i
variational autoencoders, or VAEs, are probabilistic generative models; they aim to learn
adistributionPr(x)overthedata(seefigure14.2). Aftertraining,itispossibletodraw
(generate)samplesfromthisdistribution. However,thepropertiesoftheVAEmeanthat
it is unfortunately not possible to evaluate the probability of new examples x∗ exactly.
ItiscommontotalkabouttheVAEasifitisthemodelofPr(x),butthisismislead-
ing; the VAE is a neural architecture that is designed to help learn the model for Pr(x).
ThefinalmodelforPr(x)containsneitherthe“variational”northe“autoencoder”parts
and might be better described as a nonlinear latent variable model.
This chapter starts by introducing latent variable models in general and then con-
siders the specific case of the nonlinear latent variable model. It will become clear that
maximum likelihood learning of this model is not straightforward. Nevertheless, it is
possible to define a lower bound on the likelihood, and the VAE architecture approxi-
mates this bound using a Monte Carlo (sampling) method. The chapter concludes by
presenting several applications of the VAE.
17.1 Latent variable models
Latent variable models take an indirect approach to describing a probability distribu-
tionPr(x)overamulti-dimensionalvariablex. Insteadofdirectlywritingtheexpression
forPr(x),theymodelajointdistributionPr(x,z)ofthedataxandanunobservedhid-
AppendixC.1.2
denorlatentvariablez. TheythendescribetheprobabilityofPr(x)asamarginalization
Marginalization
of this joint probability so that:
Z
Pr(x)= Pr(x,z)dz. (17.1)
AppendixC.1.3
Typically,thejointprobabilityPr(x,z)isbrokendownusingtherulesofconditional Conditional
probabilityintothelikelihoodofthedatawithrespecttothelatentvariablestermPr(x|z) probability
and the prior Pr(z):
Draft: please send errata to udlbookmail@gmail.com.

328 17 Variational autoencoders
Z
Pr(x)= Pr(x|z)Pr(z)dz. (17.2)
ThisisaratherindirectapproachtodescribingPr(x), butitisusefulbecauserelatively
simple expressions for Pr(x|z) and Pr(z) can define complex distributions Pr(x).
17.1.1 Example: mixture of Gaussians
In a 1D mixture of Gaussians (figure 17.1a), the latent variable z is discrete, and the
prior Pr(z) is a categorical distribution (figure 5.9) with one probability λ for each
n
Problem17.1 possible value of z. The likelihood Pr(x|z = n) of the data x given that the latent
variable z takes value n is normally distributed with mean µ and variance σ2:
n n
Pr(z =n) = λ
n (cid:2) (cid:3)
Pr(x|z =n) = Norm µ ,σ2 . (17.3)
x n n
Asinequation17.2,theprobabilityPr(x)isgivenbythemarginalizationoverthelatent
variablez (figure17.1b). Here,thelatentvariableisdiscrete,sowesumoveritspossible
values to marginalize:
XN
Pr(x) = Pr(x,z =n)
n=1
XN
= Pr(x|z =n)·Pr(z =n)
n=1
XN (cid:2) (cid:3)
= λ ·Norm µ ,σ2 . (17.4)
n x n n
n=1
Fromsimpleexpressionsforthelikelihoodandprior,wedescribeacomplexmulti-modal
probability distribution.
17.2 Nonlinear latent variable model
In the nonlinear latent variable model, both the data x and the latent variable z are
AppendixC.3.2
continuous and multivariate. The prior Pr(z) is a standard multivariate normal:
Multivariate
normal
Pr(z)=Norm [0,I]. (17.5)
z
The likelihood Pr(x|z,ϕ) is also normally distributed; its mean is a nonlinear func-
tion f[z,ϕ] of the latent variable, and its covariance σ2I is spherical:
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

17.2 Nonlinear latent variable model 329
Figure 17.1 Mixture of Gaussians (MoG). a) The MoG describes a complex
probabilitydistribution(cyancurve)asaweightedsumofGaussiancomponents
(dashedcurves). b)Thissumisthemarginalizationof thejointdensity Pr(x,z)
between the continuous observed data x and a discrete latent variable z.
h i
Pr(x|z,ϕ)=Norm f[z,ϕ],σ2I . (17.6)
x
The function f[z,ϕ] is described by a deep network with parameters ϕ. The latent vari-
able z is lower dimensional than the data x. The model f[z,ϕ] describes the important
aspects of the data, and the remaining unmodeled aspects are ascribed to the noise σ2I.
Notebook17.1
The data probability Pr(x|ϕ) is found by marginalizing over the latent variable z:
Latentvariable
models
Z
Pr(x|ϕ) = Pr(x,z|ϕ)dz
Z
= Pr(x|z,ϕ)·Pr(z)dz
Z h i
= Norm f[z,ϕ],σ2I ·Norm [0,I]dz. (17.7)
x z
This can be viewed as an infinite weighted sum (i.e., an infinite mixture) of spherical
Gaussians with different means, where the weights are Pr(z) and the means are the
network outputs f[z,ϕ] (figure 17.2).
17.2.1 Generation
A new example x∗ can be generated using ancestral sampling (figure 17.3). We draw z∗
from the prior Pr(z) and pass this through the network f[z∗,ϕ] to compute the mean of AppendixC.4.2
Ancestralsampling
the likelihood Pr(x|z∗,ϕ) (equation 17.6), from which we draw x∗. Both the prior and
likelihood are normal distributions, so this is straightforward.
Draft: please send errata to udlbookmail@gmail.com.

330 17 Variational autoencoders
Figure17.2Nonlinearlatentvariablemodel. Acomplex2DdensityPr(x)(right)
is created as the marginalization of the joint distribution Pr(x,z) (left) over the
latent variable z; to create Pr(x), we integrate the 3D volume over the dimen-
sion z. For each z, the distribution over x is a spherical Gaussian (two slices
shown) with a mean f[z,ϕ] that is a nonlinear function of z and depends on
parameters ϕ. The distribution Pr(x) is a weighted sum of these Gaussians.
Figure17.3Generationfromnonlinearlatentvariablemodel. a)Wedrawasam-
ple z∗ from the prior probability Pr(z) over the latent variable. b) A sample x∗
is then drawn from Pr(x|z∗,ϕ). This is a spherical Gaussian with a mean that
is a nonlinear function f[•,ϕ] of z∗ and a fixed variance σ2I. c) If we repeat this
process many times, we recover the density Pr(x|ϕ).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

17.3 Training 331
Figure 17.4Jensen’sinequality(discrete
case). The logarithm (black curve) is
a concave function; you can draw a
straight line between any two points on
thecurve,andthislinewillalwayslieun-
derneath it. It follows that any convex
combination (weighted sum with posi-
tive weights that sum to one) of the six
points on the log function must lie in
the gray region under the curve. Here,
wehaveweightedthepointsequally(i.e.,
takenthemean)toyieldthecyanpoint.
Since this point lies below the curve,
log[E[y]]>E[log[y]].
17.3 Training
To train the model, we maximize the log-likelihood over a training dataset {x }I with
i i=1
respect to the model parameters. For simplicity, we assume that the variance term σ2
in the likelihood expression is known and concentrate on learning ϕ:
" #
XI h i
ϕˆ = argmax log Pr(x |ϕ) , (17.8)
i
ϕ
i=1
where:
Z
Pr(x |ϕ) = Norm [f[z,ϕ],σ2I]·Norm [0,I]dz. (17.9)
i xi z
Unfortunately,thisisintractable. Thereisnoclosed-formexpressionfortheintegraland
no easy way to evaluate it for a particular value of x.
17.3.1 Evidence lower bound (ELBO)
Tomakeprogress,wedefinealowerboundonthelog-likelihood. Thisisafunctionthatis
alwayslessthanorequaltothelog-likelihoodforagivenvalueofϕandwillalsodepend
on some other parameters θ. Eventually, we will build a network to compute this lower
bound and optimize it. To define this lower bound, we need Jensen’s inequality.
17.3.2 Jensen’s inequality
Jensen’s inequality says that a concave function g[•] of the expectation of data y is
AppendixB.1.2
greater than or equal to the expectation of the function of the data: Concavefunctions
Draft: please send errata to udlbookmail@gmail.com.

332 17 Variational autoencoders
Figure 17.5 Jensen’s inequality (continuous case). For a concave function, com-
putingtheexpectationofadistributionPr(y)andpassingitthroughthefunction
givesaresultgreaterthanorequaltotransformingthevariableybythefunction
andthencomputingtheexpectationofthenewvariable. Inthecaseoftheloga-
rithm, we have log[E[y]]≥E[log[y]]. The left-hand side of the figure corresponds
to the left-hand side of this inequality and the right-hand side of the figure to
the right-hand side. One way of thinking about this is to consider that we are
takingaconvexcombinationofthepointsintheorangedistributiondefinedover
y∈[0,1]. Bythelogicoffigure17.4,thismustlieunderthecurve. Alternatively,
we can think about the concave function as compressing the high values of y
relativetothelowvalues,sotheexpectedvalueislowerwhenwepassy through
the function first.
(cid:2) (cid:3)
g[E[y]]≥E g[y] . (17.10)
In this case, the concave function is the logarithm, so we have:
Problems17.2–17.3
(cid:2) (cid:3) (cid:2) (cid:3)
log E[y] ≥E log[y] , (17.11)
or writing out the expression for the expectation in full, we have:
(cid:20)Z (cid:21) Z
log Pr(y)ydy ≥ Pr(y)log[y]dy. (17.12)
Thisisexploredinfigures17.4–17.5. Infact, theslightlymoregeneralstatementistrue:
(cid:20)Z (cid:21) Z
log Pr(y)h[y]dy ≥ Pr(y)log[h[y]]dy. (17.13)
where h[y] is a function of y. This follows because h[y] is another random variable with
a new distribution. Since we never specified Pr(y), the relation remains true.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

17.3 Training 333
Figure 17.6 Evidence lower bound (ELBO). The goal is to maximize the log-
likelihood log[Pr(x|ϕ)] (black curve) with respect to the parameters ϕ. The
ELBOisafunctionthatlieseverywherebelowthelog-likelihood. Itisafunction
of both ϕ and a second set of parameters θ. For fixed θ, we get a function
ofϕ(twocoloredcurvesfordifferentvaluesofθ). Consequently,wecanincrease
the log-likelihood by either improving the ELBO with respect to a) the new
parameters θ (moving from colored curve to colored curve) or b) the original
parameters ϕ (moving along the current colored curve).
17.3.3 Deriving the bound
We now use Jensen’s inequality to derive the lower bound for the log-likelihood. We
startbymultiplyinganddividingthelog-likelihoodbyanarbitraryprobabilitydistribu-
tion q(z) over the latent variables:
(cid:20)Z (cid:21)
log[Pr(x|ϕ)] = log Pr(x,z|ϕ)dz
(cid:20)Z (cid:21)
Pr(x,z|ϕ)
= log q(z) dz , (17.14)
q(z)
WethenuseJensen’sinequalityforthelogarithm(equation17.12)tofindalowerbound:
(cid:20)Z (cid:21) Z (cid:20) (cid:21)
Pr(x,z|ϕ) Pr(x,z|ϕ)
log q(z) dz ≥ q(z)log dz, (17.15)
q(z) q(z)
wheretheright-handsideistermedtheevidencelowerboundorELBO.Itgetsthisname
because Pr(x|ϕ) is called the evidence in the context of Bayes’ rule (equation 17.19).
In practice, the distribution q(z) has parameters θ, so the ELBO can be written as:
Z (cid:20) (cid:21)
Pr(x,z|ϕ)
ELBO[θ,ϕ]= q(z|θ)log dz. (17.16)
q(z|θ)
Draft: please send errata to udlbookmail@gmail.com.

334 17 Variational autoencoders
To learn the nonlinear latent variable model, we maximize this quantity as a function of
both ϕ and θ. The neural architecture that computes this quantity is the VAE.
17.4 ELBO properties
When first encountered, the ELBO is a somewhat mysterious object, so we now provide
some intuition about its properties. Consider that the original log-likelihood of the data
isafunctionoftheparametersϕandthatwewanttofinditsmaximum. Foranyfixedθ,
the ELBO is still a function of the parameters but one that must lie below the original
likelihood function. When we change θ, we modify this function, and depending on our
choice, the lower bound may move closer or further from the log-likelihood. When we
change ϕ, we move along the lower bound function (figure 17.6).
17.4.1 Tightness of bound
TheELBOistightwhen,forafixedvalueofϕ,theELBOandtheloglikelihoodfunction
coincide. To find the distribution q(z|θ) that makes the bound tight, we factor the
AppendixC.1.3
numerator of the log term in the ELBO using the definition of conditional probability:
Conditional
probability
Z (cid:20) (cid:21)
Pr(x,z|ϕ)
ELBO[θ,ϕ] = q(z|θ)log dz
q(z|θ)
Z (cid:20) (cid:21)
Pr(z|x,ϕ)Pr(x|ϕ)
= q(z|θ)log dz
q(z|θ)
Z Z (cid:20) (cid:21)
(cid:2) (cid:3) Pr(z|x,ϕ)
= q(z|θ)log Pr(x|ϕ) dz+ q(z|θ)log dz
q(z|θ)
Z (cid:20) (cid:21)
(cid:2) (cid:3) Pr(z|x,ϕ)
= log Pr(x|ϕ) + q(z|θ)log dz
q(z|θ)
(cid:2) (cid:3) h (cid:12) (cid:12) (cid:12) (cid:12) i
= log Pr(x|ϕ) −D q(z|θ)(cid:12)(cid:12)Pr(z|x,ϕ) . (17.17)
KL
Here, the first integral disappears between lines three and four since log[Pr(x|ϕ)] does
AppendixC.5.1 not depend on z, and the integral of the probability distribution q(z|θ) is one. In the
KLdivergence
last line, we have just used the definition of the Kullback-Leibler (KL) divergence.
This equation shows that the ELBO is the original log-likelihood minus the KL di-
vergence D [q(z|θ)||Pr(z|x,ϕ)]. The KL divergence measures the “distance” between
KL
distributions and can only take non-negative values. It follows the ELBO is a lower
bound on log[Pr(x|ϕ)]. The KL distance will be zero, and the bound will be tight
when q(z|θ) = Pr(z|x,ϕ). This is the posterior distribution over the latent variables z
given observed data x; it indicates which values of the latent variable could have been
responsible for the data point (figure 17.7).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

17.4 ELBO properties 335
Figure 17.7 Posterior distribution over latent variable. a) The posterior distri-
bution Pr(z|x∗,ϕ) is the distribution over the values of the latent variable z
that could be responsible for a data point x∗. We calculate this via Bayes’
rulePr(z|x∗,ϕ)∝Pr(x∗|z,ϕ)Pr(z). b)Wecomputethefirsttermontheright-
handside(thelikelihood)byassessingtheprobabilityofx∗againstthesymmetric
Gaussian associated with each value of z. Here, it was more likely to have been
createdfromz thanz . ThesecondtermisthepriorprobabilityPr(z)overthe
1 2
latent variable. Combiningthese twofactors and normalizing so the distribution
sums to one gives us the posterior Pr(z|x∗,ϕ).
17.4.2 ELBO as reconstruction loss minus KL distance to prior
Equations 17.16 and 17.17 are two different ways to express the ELBO. A third way is
to consider the bound as reconstruction error minus the distance to the prior:
Z (cid:20) (cid:21)
Pr(x,z|ϕ)
ELBO[θ,ϕ] = q(z|θ)log dz
q(z|θ)
Z (cid:20) (cid:21)
Pr(x|z,ϕ)Pr(z)
= q(z|θ)log dz
q(z|θ)
Z Z (cid:20) (cid:21)
Pr(z)
= q(z|θ)log[Pr(x|z,ϕ)]dz+ q(z|θ)log dz
q(z|θ)
Z (cid:2) (cid:3) h (cid:12) (cid:12) (cid:12) (cid:12) i
= q(z|θ)log Pr(x|z,ϕ) dz−D q(z|θ)(cid:12)(cid:12)Pr(z) , (17.18)
KL
where the joint distribution Pr(x,z|ϕ) has been factored into conditional probabil-
ity Pr(x|z,ϕ)Pr(z) between the first and second lines, and the definition of KL di- Problem17.4
vergence is used again in the last line.
Draft: please send errata to udlbookmail@gmail.com.

336 17 Variational autoencoders
In this formulation, the first term measures the average agreement Pr(x|z,ϕ) of the
latent variable and the data. This measures the reconstruction accuracy. The second
term measures the degree to which the auxiliary distribution q(z|θ) matches the prior.
This formulation is the one that is used in the variational autoencoder.
17.5 Variational approximation
Wesawinequation17.17thattheELBOistightwhenq(z|θ)istheposteriorPr(z|x,ϕ).
In principle, we can compute the posterior using Bayes’ rule:
Pr(x|z,ϕ)Pr(z)
Pr(z|x,ϕ)= , (17.19)
Pr(x|ϕ)
but in practice, this is intractable because we can’t evaluate the evidence term Pr(x|ϕ)
in the denominator (see section 17.3).
One solution is to make a variational approximation: we choose a simple parametric
form for q(z|θ) and use this to approximate the true posterior. Here, we choose a
AppendixC.3.2
multivariate normal distribution with mean µ and diagonal covariance Σ. This will not
Multivariate
normal always match the posterior well but will be better for some values of µ and Σ than
others. Duringtraining, wewillfindthenormaldistributionthatis“closest”tothetrue
posterior Pr(z|x) (figure 17.8). This corresponds to minimizing the KL divergence in
equation 17.17 and moving the colored curves in figure 17.6 upwards.
Since the optimal choice for q(z|θ) was the posterior Pr(z|x), and this depends on
the data example x, the variational approximation should do the same, so we choose:
h i
q(z|x,θ)=Norm g [x,θ],g [x,θ] , (17.20)
z µ Σ
where g[x,θ] is a second neural network with parameters θ that predicts the mean µ
and variance Σ of the normal variational approximation.
17.6 The variational autoencoder
Finally, we can describe the VAE. We build a network that computes the ELBO:
Z (cid:2) (cid:3) h (cid:12) (cid:12) (cid:12) (cid:12) i
ELBO[θ,ϕ]= q(z|x,θ)log Pr(x|z,ϕ) dz−D q(z|x,θ)(cid:12)(cid:12)Pr(z) , (17.21)
KL
where the distribution q(z|x,θ) is the approximation from equation 17.20.
AppendixC.2 Thefirsttermstillinvolvesanintractableintegral,butsinceitisanexpectationwith
Expectation respect to q(z|x,θ), we can approximate it by sampling. For any function a[•] we have:
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

17.6 The variational autoencoder 337
Figure 17.8 Variational approximation. The posterior Pr(z|x∗,ϕ) can’t be com-
putedinclosedform. Thevariationalapproximationchoosesafamilyofdistribu-
tionsq(z|x,θ)(hereGaussians)andtriestofindtheclosestmemberofthisfamily
tothetrueposterior. a)Sometimes,theapproximation(cyancurve)isgoodand
lies close to the true posterior (orange curve). b) However, if the posterior is
multi-modal (as in figure 17.7), then the Gaussian approximation will be poor.
Z
(cid:2) (cid:3) XN
1
E a[z] = a[z]q(z|x,θ)dz≈ a[z ∗ ], (17.22)
z N n
n=1
where z∗ is the nth sample from q(z|x,θ). This is known as a Monte Carlo estimate.
n
For a very approximate estimate, we can just use a single sample z∗ from q(z|x,θ):
(cid:2) (cid:3) h (cid:12) (cid:12) (cid:12) (cid:12) i
ELBO[θ,ϕ] ≈ log Pr(x|z ∗ ,ϕ) −D q(z|x,θ)(cid:12)(cid:12)Pr(z) . (17.23)
KL
The second term is the KL divergence between the variational distribution q(z|x,θ) =
AppendixC.5.4
Norm [µ,Σ]andthepriorPr(z)=Norm [0,I]. TheKLdivergencebetweentwonormal
z z KLdivergence
distributionscanbecalculatedinclosedform. Forthespecialcasewhereonedistribution betweennormal
has parameters µ,Σ and the other is a standard normal, it is given by: distributions
h (cid:12)(cid:12) i (cid:18) h i(cid:19)
(cid:12)(cid:12) 1
D q(z|x,θ)(cid:12)(cid:12)Pr(z) = Tr[Σ]+µTµ−D −log det[Σ] . (17.24)
KL 2 z
where D is the dimensionality of the latent space.
z
17.6.1 VAE algorithm
To summarize, we aim to build a model that computes the evidence lower bound for a
point x. Then we use an optimization algorithm to maximize this lower bound over the
Draft: please send errata to udlbookmail@gmail.com.

338 17 Variational autoencoders
Figure 17.9Variationalautoencoder. Theencoderg[x,θ]takesatrainingexam-
ple x and predicts the parameters µ,Σ of the variational distribution q(z|x,θ).
We sample from this distribution and then use the decoder f[z,ϕ] to predict the
datax. ThelossfunctionisthenegativeELBO,whichdependsonhowaccurate
this prediction is and how similar the variational distribution q(z|x,θ) is to the
prior Pr(z) (equation 17.21).
dataset and hence improve the log-likelihood. To compute the ELBO we:
• computethemeanµandvarianceΣofthevariationalposteriordistributionq(z|θ,x)
for this data point x using the network g[x,θ],
• draw a sample z∗ from this distribution, and
• compute the ELBO using equation 17.23.
The associated architecture is shown in figure 17.9. It should now be clear why this
is called a variational autoencoder. It is variational because it computes a Gaussian
approximation to the posterior distribution. It is an autoencoder because it starts with
adatapointx,computesalower-dimensionallatentvectorzfromthis,andthenusesthis
vector to recreate the data point x as closely as possible. In this context, the mapping
from the data to the latent variable by the network g[x,θ] is called the encoder, and the
mappingfromthelatentvariabletothedatabythenetworkf[z,ϕ]iscalledthedecoder.
The VAE computes the ELBO as a function of both ϕ and θ. To maximize this
bound,werunmini-batchesofsamplesthroughthenetworkandupdatetheseparameters
withanoptimizationalgorithmsuchasSGDorAdam. ThegradientsoftheELBOwith
respecttotheparametersarecomputedasusualusingautomaticdifferentiation. During
thisprocess,wearebothmovingbetweenthecoloredcurves(changingθ)andalongthem
(changingϕ)infigure17.10. Duringthisprocess,theparametersϕchangetoassignthe
data a higher likelihood in the nonlinear latent variable model.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

17.7 The reparameterization trick 339
Figure 17.10TheVAEupdatesbothfac-
tors that determine the lower bound at
eachiteration. Boththeparametersϕof
thedecoderandtheparametersθ ofthe
encoderaremanipulatedtoincreasethis
lower bound.
Figure 17.11 Reparameterization trick. With the original architecture (fig-
ure 17.9), we cannot easily backpropagate through the sampling step. The repa-
rameterization trick removes the sampling step from the main pipeline; we draw
fromastandardnormalandcombinethiswiththepredictedmeanandcovariance
to get a sample from the variational distribution.
17.7 The reparameterization trick
There is one more complication; the network involves a sampling step, and it is diﬀicult
to differentiate through this stochastic component. However, differentiating past this
step is necessary to update the parameters θ that precede it in the network.
Fortunately,thereisasimplesolution; wecanmovethestochasticpartintoabranch
of the network that draws a sample ϵ∗ from Norm [0,I] and then use the relation: Problem17.5
ϵ
z ∗ =µ+Σ1/2ϵ ∗ , (17.25)
to draw from the intended Gaussian. Now we can compute the derivatives as usual
Notebook17.2
becausethebackpropagationalgorithmdoesnotneedtopassdownthestochasticbranch.
Reparameterization
This is known as the reparameterization trick (figure 17.11). trick
Draft: please send errata to udlbookmail@gmail.com.

340 17 Variational autoencoders
17.8 Applications
Variational autoencoders have many uses, including denoising, anomaly detection, and
compression. This section reviews several applications for image data.
17.8.1 Approximating sample probability
In section 17.3, we argued that it is not possible to evaluate the probability of a sample
with the VAE, which describes this probability as:
Z
Pr(x) = Pr(x|z)Pr(z)dz
h i
= E Pr(x|z)
z
h i
= E Norm [f[z,ϕ],σ2I] . (17.26)
z x
In principle, we could approximate this probability using equation 17.22 by drawing
samples from Pr(z)=Norm [0,I] and computing:
z
XN
1
Pr(x)≈ Pr(x|z ). (17.27)
N n
n=1
However,thecurseofdimensionalitymeansthatalmostallvaluesofz thatwedraw
n
wouldhaveaverylowprobabilityPr(x|z );wewouldhavetodrawanenormousnumber
n
of samples to get a reliable estimate. A better approach is to use importance sampling.
Here, we sample z from an auxiliary distribution q(z), evaluate Pr(x|z ), and rescale
n
the resulting values by the probability q(z) under the new distribution:
Z
Pr(x) = Pr(x|z)Pr(z)dz
Z
Pr(x|z)Pr(z)
= q(z)dz
q(z)
(cid:20) (cid:21)
Pr(x|z)Pr(z)
= E
q(z) q(z)
1 XN Pr(x|z )Pr(z )
≈ n n , (17.28)
N q(z )
n
n=1
where now we draw the samples from q(z). If q(z) is close to the region of z where
Notebook17.3 the Pr(x|z) has high likelihood, then we will focus the sampling on the relevant area of
Importance
sampling space and estimate Pr(x) much more eﬀiciently.
The product Pr(x|z)Pr(z) that we are trying to integrate is proportional to the
posterior distribution Pr(z|x) (by Bayes’ rule). Hence, a sensible choice of auxiliary
distribution q(z) is the variational posterior q(z|x) computed by the encoder.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

17.8 Applications 341
Figure 17.12 Sampling from a standard VAE trained on CELEBA. In each col-
umn, a latent variable z∗ is drawn and passed through the model to predict the
mean f[z∗,ϕ] before adding independent Gaussian noise (see figure 17.3). a) A
setofsamplesthatarethesumofb)thepredictedmeansandc)sphericalGaus-
sian noise vectors. The images look too smooth before we add the noise and too
noisyafterward. Thisistypical,andusually,thenoise-freeversionisshownsince
the noise is considered to represent aspects of the image that are not modeled.
Adapted from Dorta et al. (2018). d) It is now possible to generate high-quality
images from VAEs using hierarchical priors, specialized architecture, and careful
regularization. Adapted from Vahdat & Kautz (2020).
In this way, we can approximate the probability of new samples. With suﬀicient
samples, this will provide a better estimate than the lower bound and could be used to
evaluate the quality of the model by evaluating the log-likelihood of test data. Alterna-
tively, it could be used as a criterion for determining whether new examples belong to
the distribution or are anomalous.
17.8.2 Generation
VAEs build a probabilistic model, and it’s easy to sample from this model by draw-
ing from the prior Pr(z) over the latent variable, passing this result through the de-
coder f[z,ϕ], and adding noise according to Pr(x|f[z,ϕ]). Unfortunately, samples from
Draft: please send errata to udlbookmail@gmail.com.

342 17 Variational autoencoders
vanilla VAEs are generally low-quality (figure 17.12a–c). This is partly because of the
naïve spherical Gaussian noise model and partly because of the Gaussian models used
for the prior and variational posterior. One trick tPo improve generation quality is to
sample from the aggregated posterior q(z|θ)=(1/I) q(z|x ,θ) rather than the prior;
i i
this is the average posterior over all samples and is a mixture of Gaussians that is more
representative of true distribution in latent space.
Modern VAEs can produce high-quality samples (figure 17.12d), but only by using
hierarchical priors and specialized network architecture and regularization techniques.
Diffusion models (chapter 18) can be viewed as VAEs with hierarchical priors. These
also create very high-quality samples.
17.8.3 Resynthesis
VAEs can also be used to modify real data. A data point x can be projected into the
latent space by either (i) taking the mean of the distribution predicted by the encoder
or (ii) by using an optimization procedure to find the latent variable z that maximizes
the posterior probability, which Bayes’ rule tells us is proportional to Pr(x|z)Pr(z).
In figure 17.13, multiple images labeled as “neutral” or “smiling” are projected into
latent space. The vector representing this change is estimated by taking the difference
in latent space between the means of these two groups. A second vector is estimated to
represent “mouth closed” versus “mouth open.”
Now the image of interest is projected into the latent space, and then the repre-
sentation is modified by adding or subtracting these vectors. To generate intermediate
images, spherical linear interpolation or Slerp is used rather than linear interpolation.
Problem17.6
In 3D, this would be the difference between interpolating along the surface of a sphere
versus digging a straight tunnel through its body.
Theprocessofencoding(andpossiblymodifying)inputdatabeforedecodingagainis
knownasresynthesis. ThiscanalsobedonewithGANsandnormalizingflows. However,
in GANs, there is no encoder, so a separate procedure must be used to find the latent
variable that corresponds to the observed data.
17.8.4 Disentanglement
Intheresynthesisexampleabove,thedirectionsinspacerepresentinginterpretableprop-
erties had to be estimated using labeled training data. Other work attempts to improve
thecharacteristicsofthelatentspacesothatitscoordinatedirectionscorrespondtoreal-
worldproperties. Wheneachdimensionrepresentsanindependentreal-worldfactor,the
latent space is described as disentangled. For example, when modeling face images, we
might hope to uncover head pose or hair color as independent factors.
Methods to encourage disentanglement typically add regularization terms to the loss
function based on either (i) the postPerior q(z|x,θ) over the latent variables z, or (ii) the
aggregated posterior q(z|θ)=(1/I) q(z|x ,θ):
i i
h (cid:2) (cid:3)i (cid:2) (cid:3)
Lnew =−ELBO[θ,ϕ]+λ
1
E
Pr(x)
r
1
q(z|x,θ) +λ
2
r
2
q(z|θ) . (17.29)
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

17.9 Summary 343
Figure 17.13Resynthesis. Theoriginalimageontheleftisprojectedintothela-
tentspaceusingtheencoder,andthemeanofthepredictedGaussianischosento
representtheimage. Thecenter-leftimageinthegridisthereconstructionofthe
input. The other images are reconstructions after manipulating the latent space
in directions representing smiling/neutral (horizontal) and mouth open/closed
(vertical). Adapted from White (2016).
Here the regularization term r [•] is a function of the posterior and is weighted by λ .
1 1
The term r [•] is a function of the aggregated posterior and is weighted by λ .
2 2
Forexample,thebetaVAEupweightsthesecondtermintheELBO(equation17.18):
(cid:2) (cid:3) h (cid:12) (cid:12) (cid:12) (cid:12) i
ELBO[θ,ϕ] ≈ log Pr(x|z ∗ ,ϕ) −β·D q(z|x,θ)(cid:12)(cid:12)Pr(z) , (17.30)
KL
where β >1 determines how much more the deviation from the prior Pr(z) is weighted
relativetothereconstructionerror. Sincethepriorisusuallyamultivariatenormalwith
a spherical covariance matrix, its dimensions are independent. Hence, up-weighting this
term encourages the posterior distributions to be less correlated. Another variant is the
total correlation VAE, which adds a term to decrease the total correlation between vari-
ables in the latent space (figure 17.14) and maximizes the mutual information between
a small subset of the latent variables and the observations.
17.9 Summary
The VAE is an architecture that helps to learn a nonlinear latent variable model over x.
Thismodelcangeneratenewexamplesbysamplingfromthelatentvariable,passingthe
result through a deep network, and then adding independent Gaussian noise.
Draft: please send errata to udlbookmail@gmail.com.

344 17 Variational autoencoders
Figure 17.14 Disentanglement in the total correlation VAE. The VAE model is
modified so that the loss function encourages the total correlation of the latent
variables to be minimized and hence encourages disentanglement. When trained
on a dataset of images of chairs, several of the latent dimensions have clear real-
world interpretations, including a) rotation, b) overall size, and c) legs (swivel
chair versus normal). In each case, the central column depicts samples from the
model, and as we move left to right, we are subtracting or adding a coordinate
vector in latent space. Adapted from Chen et al. (2018d).
It is not possible to compute the likelihood of a data point in closed form, and
this poses problems for training with maximum likelihood. However, we can define a
lower bound on the likelihood and maximize this bound. Unfortunately, for the bound
to be tight, we need to compute the posterior probability of the latent variable given
the observed data, which is also intractable. The solution is to make a variational
approximation. This is a simpler distribution (usually a Gaussian) that approximates
the posterior and whose parameters are computed by a second encoder network.
To create high-quality samples from the VAE, it seems to be necessary to model the
latent space with more sophisticated probability distributions than the Gaussian prior
and posterior. One option is to use hierarchical priors (in which one latent variable
generates another). The next chapter discusses diffusion models, which produce very
high-quality examples and can be viewed as hierarchical VAEs.
Notes
TheVAEwasoriginallyintroducedbyKingma&Welling(2014). Acomprehensiveintroduction
to variational autoencoders can be found in Kingma et al. (2019).
Applications: TheVAEandvariantsthereofhavebeenappliedtoimages(Kingma&Welling,
2014;Gregoretal.,2016;Gulrajanietal.,2016;Akuzawaetal.,2018),speech(Hsuetal.,2017b),
text(Bowmanetal.,2015;Huetal.,2017;Xuetal.,2020),molecules(Gómez-Bombarellietal.,
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 345
2018; Sultan et al., 2018), graphs (Kipf & Welling, 2016; Simonovsky & Komodakis, 2018),
robotics (Hernández et al., 2018; Inoue et al., 2018; Park et al., 2018), reinforcement learning
(Heessetal.,2015;VanHoofetal.,2016),3Dscenes(Eslamietal.,2016,2018;RezendeJimenez
et al., 2016), and handwriting (Chung et al., 2015).
Applications include resynthesis and interpolation (White, 2016; Bowman et al., 2015), collab-
orative filtering (Liang et al., 2018), and compression (Gregor et al., 2016). Gómez-Bombarelli
et al. (2018) use the VAE to construct a continuous representation of chemical structures that
canthenbeoptimizedfordesirableproperties. Ravanbakhshetal.(2017)simulateastronomical
observations for calibrating measurements.
Relation to other models: Theautoencoder(Rumelhartetal.,1985;Hinton&Salakhutdi-
nov, 2006)passes datathrough anencoder to abottlenecklayerand thenreconstructs it using
a decoder. The bottleneck is similar to latent variables in the VAE, but the motivation differs.
Here, the goal is not to learn a probability distribution but to create a low-dimensional repre-
sentation that captures the essence of the data. Autoencoders also have various applications,
including denoising (Vincent et al., 2008) and anomaly detection (Zong et al., 2018).
Iftheencoderanddecoderarelineartransformations,theautoencoderisjustprincipalcompo-
nent analysis (PCA). Hence, the nonlinear autoencoder is a generalization of PCA. There are
also probabilistic forms of PCA. Probabilistic PCA (Tipping & Bishop, 1999) adds spherical
Gaussian noise to the reconstruction to create a probability model, and factor analysis adds
diagonal Gaussian noise (see Rubin & Thayer, 1982). If we make the encoder and decoder of
these probabilistic variants nonlinear, we return to the variational autoencoder.
Architecturalvariations: TheconditionalVAE(Sohnetal.,2015)passesclassinformationc
intoboththeencoderanddecoder. Theresultisthatthelatentspacedoesnotneedtoencode
the class information. For example, when MNIST data are conditioned on the digit label, the
latentvariablesmightencodetheorientationandwidthofthedigitratherthanthedigitcategory
itself. Sønderby et al. (2016a) introduced ladder variational autoencoders, which recursively
correct the generative distribution with a data-dependent approximate likelihood term.
Modifyinglikelihood: OtherworkinvestigatesmoresophisticatedlikelihoodmodelsPr(x|z).
The PixelVAE (Gulrajani et al., 2016) used an autoregressive model over the output variables.
Dorta et al. (2018) modeled the covariance of the decoder output as well as the mean. Lamb
et al. (2016) improved the quality of reconstruction by adding extra regularization terms that
encourage the reconstruction to be similar to the original image in the space of activations
of a layer of an image classification model. This model encourages semantic information to
be retained and was used to generate the results in figure 17.13. Larsen et al. (2016) use an
adversarial loss for reconstruction, which also improves results.
Latent space, prior, and posterior: Manydifferentformsforthevariationalapproximation
to the posterior have been investigated, including normalizing flows (Rezende & Mohamed,
2015; Kingma et al., 2016), directed graphical models (Maaløe et al., 2016), undirected models
(Vahdat et al., 2020), and recursive models for temporal data (Gregor et al., 2016, 2019).
Otherauthorshaveinvestigatedusingadiscretelatentspace(VanDenOordetal.,2017;Razavi
et al., 2019b; Rolfe, 2017; Vahdat et al., 2018a,b). For example, Razavi et al. (2019b) use a
vectorquantizedlatentspaceandmodelthepriorwithanautoregressivemodel(equation12.15).
This is slow to sample from but can describe very complex distributions.
Draft: please send errata to udlbookmail@gmail.com.

346 17 Variational autoencoders
Jiang et al. (2016) use a mixture of Gaussians for the posterior, allowing clustering. This is a
hierarchical latent variable model that adds a discrete latent variable to improve the flexibility
of the posterior. Other authors (Salimans et al., 2015; Ranganath et al., 2016; Maaløe et al.,
2016; Vahdat & Kautz, 2020) have experimented with hierarchical models that use continuous
variables. These have a close connection with diffusion models (chapter 18).
Combination with other models: Gulrajani et al. (2016) combined VAEs with an autore-
gressive model to produce more realistic images. Chung et al. (2015) combine the VAE with
recurrent neural networks to model time-varying measurements.
As discussed above, adversarial losses have been used to inform the likelihood term directly.
However,othermodelshavecombinedideasfromgenerativeadversarialnetworks(GANs)with
VAEs in different ways. Makhzani et al. (2015) use an adversarial loss in the latent space;
the idea is that the discriminator will ensure that the aggregated posterior distribution q(z)
is indistinguishable from the prior distribution Pr(z). Tolstikhin et al. (2018) generalize this
to a broader family of distances between the prior and aggregated posterior. Dumoulin et al.
(2017) introduced adversarially learned inference which uses an adversarial loss to distinguish
two pairs of latent/observed data points. In one case, the latent variable is drawn from the
latent posterior distribution and, in the other, from the prior. Other hybrids of VAEs and
GANs were proposed by Larsen et al. (2016), Brock et al. (2016), and Hsu et al. (2017a).
Posterior collapse: One potential problem in training is posterior collapse, in which the
encoderalwayspredictsthepriordistribution. ThiswasidentifiedbyBowmanetal.(2015)and
canbemitigatedbygraduallyincreasingthetermthatencouragestheKLdistancebetweenthe
posteriorandthepriortobesmallduringtraining. Severalothermethodshavebeenproposed
to prevent posterior collapse (Razavi et al., 2019a; Lucas et al., 2019b,a), and this is also part
of the motivation for using a discrete latent space (Van Den Oord et al., 2017).
Blurry reconstructions: Zhaoetal.(2017c)provideevidencethattheblurryreconstructions
are partly due to Gaussian noise and also because of the sub-optimal posterior distributions
induced by the variational approximation. It is perhaps not coincidental that some of the
best synthesis results have come from using a discrete latent space modeled by a sophisticated
autoregressive model (Razavi et al., 2019b) or from using hierarchical latent spaces (Vahdat &
Kautz,2020;seefigure17.12d). Figure17.12a-cusedaVAEthatwastrainedontheCELEBA
database (Liu et al., 2015). Figure 17.12d uses a hierarchical VAE that was trained on the
CELEBA HQ dataset (Karras et al., 2018).
Otherproblems: Chenetal.(2017)notedthatwhenmorecomplexlikelihoodtermsareused,
such as the PixelCNN (Van den Oord et al., 2016c), the output can cease to depend on the
latentvariablesatall. Theytermthistheinformation preferenceproblem. Thiswasaddressed
byZhaoetal.(2017b)intheInfoVAE,whichaddedanextratermthatmaximizedthemutual
information between the latent and observed distributions.
Another problem with the VAE is that there can be “holes” in the latent space that do not
correspond to any realistic sample. Xu et al. (2020) introduce the constrained posterior VAE,
which helps prevent these vacant regions in latent space by adding a regularization term. This
allows for better interpolation from real samples.
Disentangling latent representation: Methods to “disentangle” the latent representation
includethebetaVAE(Higginsetal.,2017)andothers(e.g.,Kim&Mnih,2018;Kumaretal.,
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 347
Figure 17.15 Expectation maximization
(EM) algorithm. The EM algorithm al-
ternately adjusts the auxiliary parame-
ters θ (moves between colored curves)
and model parameters ϕ (moves along
colored curves) until the a maximum
is reached. These adjustments are
known as the E-step and the M-step,
respectively. Because the E-Step uses
the posterior distribution Pr(h|x,ϕ)
for q(h|x,θ), the bound is tight, and
the colored curve touches the black like-
lihood curve after each E-Step.
2018). Chen et al. (2018d) further decomposed the ELBO to show the existence of a term
measuring the total correlation between the latent variables (i.e., the distance between the
aggregate posterior and the product of its marginals). They use this to motivate the total
correlation VAE, which attempts to minimize this quantity. The Factor VAE (Kim & Mnih,
2018)usesadifferentapproachtominimizethetotalcorrelation. Mathieuetal.(2019)discuss
the factors that are important in disentangling representations.
Reparameterization trick: Considercomputinganexpectationofsomefunction,wherethe
probabilitydistributionwithwhichtheexpectationistakendependsonsomeparameters. The
reparameterization trick computes the derivative of this expectation with respect to these pa-
rameters. This chapter introduced this as a method to differentiate through the sampling
procedure approximating the expectation; there are alternative approaches (see problem 17.5),
but the reparameterization trick gives an estimator that (usually) has low variance. This issue
is discussed in Rezende et al. (2014), Kingma et al. (2015), and Roeder et al. (2017).
Lower bound and the EM algorithm: VAE training is based on optimizing the evidence
lower bound (sometimes also referred to as the ELBO, variational lower bound, or negative
variational free energy). Hoffman & Johnson (2016) and Lücke et al. (2020) re-express this
lower bound in several ways that elucidate its properties. Other work has aimed to make
this bound tighter (Burda et al., 2016; Li & Turner, 2016; Bornschein et al., 2016; Masrani
et al., 2019). For example, Burda et al. (2016) use a modified bound based on using multiple
importance-weighted samples from the approximate posterior to form the objective function.
The ELBO is tight when the distribution q(z|θ) matches the posterior Pr(z|x,ϕ). This is
the basis of the expectation maximization (EM) algorithm (Dempster et al., 1977). Here, we
alternately (i) choose θ so that q(z|θ) equals the posterior Pr(z|x,ϕ) and (ii) change ϕ to
Problem17.7
maximizethelowerbound(figure17.15). ThisisviableformodelslikethemixtureofGaussians,
where we can compute the posterior distribution in closed form. Unfortunately, this is not the
case for the nonlinear latent variable model, so this method cannot be used.
Problems
Problem17.1Howmanyparametersareneededtocreatea1DmixtureofGaussianswithn=5
Draft: please send errata to udlbookmail@gmail.com.

348 17 Variational autoencoders
components(equation17.4)? Statethepossiblerangeofvaluesthateachparametercouldtake.
Problem 17.2 A function is concave if its second derivative is less than or equal to zero every-
where. Show that this is true for the function g[x]=log[x].
Problem 17.3 For convex functions, Jensen’s inequality works the other way around.
(cid:2) (cid:3) (cid:2) (cid:3)
g E[y] ≤E g[y] . (17.31)
A function is convex if its second derivative is greater than or equal to zero everywhere. Show
thatthefunctiong[x]=x2nisconvexforarbitraryn∈[1,2,3,...]. UsethisresultwithJensen’s
inequality to show that the square of the mean E[x] of a distribution Pr(x) must be less than
or equal to its second moment E[x2].
Problem 17.4∗ Show that the ELBO, as expressed in equation 17.18, can alternatively be de-
rivedfromtheKLdivergencebetweenthevariationaldistributionq(z|x)andthetrueposterior
distribution Pr(z|x,ϕ):
h (cid:12)(cid:12) i Z (cid:20) (cid:21)
(cid:12)(cid:12) q(z|x)
D q(z|x)(cid:12)(cid:12)Pr(z|x,ϕ) = q(z|x)log dz. (17.32)
KL Pr(z|x,ϕ)
Start by using Bayes’ rule (equation 17.19).
Problem 17.5 The reparameterization trick computes the derivative of an expectation of a
function f[x]:
(cid:2) (cid:3)
∂
∂ϕ
E Pr(x|ϕ) f[x] , (17.33)
withrespecttotheparametersϕofthedistributionPr(x|ϕ)thattheexpectationisover. Show
that this derivative can also be computed as:
(cid:20) (cid:21)
(cid:2) (cid:3) (cid:2) (cid:3)
∂ ∂
∂ϕ
E Pr(x|ϕ) f[x] = E Pr(x|ϕ) f[x]
∂ϕ
log Pr(x|ϕ)
XI (cid:2) (cid:3)
1 ∂
≈ f[x ] log Pr(x |ϕ) . (17.34)
I i ∂ϕ i
i=1
This method is known as the REINFORCE algorithm or score function estimator.
Problem 17.6 Why is it better to use spherical linear interpolation rather than regular linear
interpolation when moving between points in the latent space? Hint: consider figure 8.13.
Problem 17.7∗ DerivetheEMalgorithmforfittingthe1DmixtureofGaussiansmodelwithN
components. Todothis,youneedto(i)findanexpressionfortheposteriordistributionPr(z|x)
over the latent variable z ∈ {1,2,...,N} for a data point x and (ii) find an expression that
updates the evidence lower bound given the posterior distributions for all of the data points.
YouwillneedtouseLagrangemultiplierstoensurethattheweightsλ ,...,λ oftheGaussians
1 N
sum to one.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.