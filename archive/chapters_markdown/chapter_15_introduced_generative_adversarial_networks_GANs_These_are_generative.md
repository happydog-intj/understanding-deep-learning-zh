# Chapter 15: introduced generative adversarial networks (GANs). These are generative

*Pages: 318-322*

---

Chapter 16
Normalizing flows
Chapter 15 introduced generative adversarial networks (GANs). These are generative
modelsthatpassalatentvariablethroughadeepnetworktocreateanewsample. GANs
are trained using the principle that the samples should be indistinguishable from real
data. However, they don’t define a distribution over data examples. Hence, assessing
the probability that a new example belongs to the same dataset isn’t straightforward.
In this chapter, we describe normalizing flows. These learn a probability model by
transforming a simple distribution into a more complicated one using a deep network.
Normalizing flows can both sample from this distribution and evaluate the probability
of new examples. However, they require specialized architecture: each layer must be
invertible. In other words, it must be able to transform data in both directions.
16.1 1D example
Normalizingflowsareprobabilisticgenerativemodels: theyfitaprobabilitydistribution
totrainingdata(figure14.2b). Considermodelinga1DdistributionPr(x). Normalizing
flows start with a simple tractable base distribution Pr(z) over a latent variable z and
apply a function x = f[z,ϕ], where the parameters ϕ are chosen so that Pr(x) has the
desireddistribution(figure16.1). Generatinganewexamplex∗ iseasy;wedrawz∗ from
the base density and pass this through the function so that x∗ =f[z∗,ϕ].
16.1.1 Measuring probability
Measuring the probability of a data point x is more challenging. Consider applying a
functionf[z,ϕ]torandomvariablez withknowndensityPr(z). Theprobabilitydensity
will decrease in areas that are stretched by the function and increase in areas that are
compressed so that the area under the new distribution remains one. The degree to
which a function f[z,ϕ] stretches or compresses its input depends on the magnitude of
its gradient. If a small change to the input causes a larger change in the output, it
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

16.1 1D example 305
Figure 16.1 Transforming probability distributions. a) The base density is a
standard normal defined on a latent variable z. b) This variable is transformed
byafunctionx=f[z,ϕ]toanewvariablex,whichc)hasanewdistribution. To
samplefromthismodel,wedrawvalueszfromthebasedensity(greenandbrown
arrowsinpanel(a)showtwoexamples). Wepassthesethroughthefunctionf[z,ϕ]
as shown by dotted arrows in panel (b) to generate the values of x, which are
indicated as arrows in panel (c).
Figure 16.2 Transforming distributions. The base density (cyan, bottom) passes
through a function (blue curve, top right) to create the model density (orange,
left). Considerdividingthebasedensityintoequalintervals(grayverticallines).
Theprobabilitymassbetweenadjacentlinesmustremainthesameaftertransfor-
mation. Thecyan-shadedregionpassesthroughapartofthefunctionwherethe
gradient is larger than one, so this region is stretched. Consequently, the height
oftheorange-shadedregionmustbelowersothatitretainsthesameareaasthe
cyan-shaded region. In other places (e.g., z =−2), the gradient is less than one,
and the model density increases relative to the base density.
Draft: please send errata to udlbookmail@gmail.com.

306 16 Normalizing flows
Figure16.3Inversemapping(normalizingdirection). Ifthefunctionisinvertible,
thenit’spossibletotransformthemodeldensitybacktotheoriginalbasedensity.
The probability of a point x under the model density depends partly on the
probability of the equivalent point z under the base density (see equation 16.1).
stretches the function. If a small change to the input causes a smaller change in the
output, it compresses the function (figure 16.2).
More precisely, the probability of data x under the transformed distribution is:
(cid:12) (cid:12)
Pr(x|ϕ) =
(cid:12)
(cid:12) (cid:12) ∂f[z,ϕ]
(cid:12)
(cid:12) (cid:12)
−1
·Pr(z), (16.1)
∂z
where z =f
−1[x,ϕ]
is the latent variable that created x. The term Pr(z) is the original
Notebook16.1
probability of this latent variable under the base density. This is moderated according
1Dnormalizing
flows to the magnitude of the derivative of the function. If this is greater than one, then the
probability decreases. If it is smaller, the probability increases.
16.1.2 Forward and inverse mappings
To draw samples from the distribution, we need the forward mapping x=f[z,ϕ], but to
measure the likelihood, we need to compute the inverse z = f
−1[x,ϕ].
Hence, we need
to choose f[z,ϕ] judiciously so that it is invertible.
Problems16.1–16.2
Theforwardmappingissometimestermedthegenerativedirection. Thebasedensity
is usually chosen to be a standard normal distribution. Hence, the inverse mapping is
termed the normalizing direction since this takes the complex distribution over x and
turns it into a normal distribution over z (figure 16.3).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

16.2 General case 307
16.1.3 Learning
To learn the distribution, we find parameters ϕ that maximize the likelihood of the
training data {x }I or equivalently minimize the negative log-likelihood:
i i=1
" #
YI
ϕˆ = argmax Pr(x |ϕ)
i
ϕ
"i=1 #
XI h i
= argmin −log Pr(x |ϕ)
i
ϕ
"i=1 "(cid:12) (cid:12)# #
XI (cid:12) (cid:12) (cid:2) (cid:3)
= argmin log (cid:12) (cid:12) ∂f[ ∂ z z i ,ϕ](cid:12) (cid:12) −log Pr(z i ) , (16.2)
ϕ i
i=1
where we have assumed that the data are independent and identically distributed in the
first line and used the likelihood definition from equation 16.1 in the third line.
16.2 General case
The previous section developed a simple 1D example that modeled a probability dis-
tribution Pr(x) by transforming a simpler base density Pr(z). We now extend this to
multivariate distributions Pr(x) and Pr(z) and add the complication that the transfor-
mation is defined by a deep neural network.
Consider applying a function x = f[z,ϕ] to a random variable z ∈ RD with base
density Pr(z), where f[z,ϕ] is a deep network. The resulting variable x ∈ RD has a
new distribution. A new sample x∗ can be drawn from this distribution by (i) drawing
a sample z∗ from the base density and (ii) passing this through the neural network so
that x∗ =f[z∗,ϕ].
By analogy with equation 16.1, the likelihood of a sample under this distribution is:
(cid:12) (cid:12)
Pr(x|ϕ)=
(cid:12)
(cid:12) (cid:12) ∂f[z,ϕ]
(cid:12)
(cid:12) (cid:12)
−1
·Pr(z), (16.3)
∂z
where z = f
−1[x,ϕ]
is the latent variable z that created x. The first term is the
inverse of the determinant of the D ×D Jacobian matrix ∂f[z,ϕ]/∂z, which contains
AppendixB.3.8
elements ∂f [z,ϕ]/∂z at position (i,j). Just as the absolute derivative measured the
j i Determinant
change of area at a point on a 1D function when the function was applied, the absolute
determinantmeasuresthechangeinvolumeatapointinthemultivariatefunction. The AppendixB.5
second term is the probability of the latent variable under the base density. Jacobian
Draft: please send errata to udlbookmail@gmail.com.

308 16 Normalizing flows
Figure 16.4 Forward and inverse mappings for a deep neural network. The base
density(left)isgraduallytransformedbythenetworklayersf [•,ϕ ],f [•,ϕ ],...
1 1 2 2
to create the model density. Each layer is invertible, and we can equivalently
think of the inverse of the layers as gradually transforming (or “flowing”) the
model density back to the base density.
16.2.1 Forward mapping with a deep neural network
Inpractice,theforwardmappingf[z,ϕ]isusuallydefinedbyaneuralnetwork,consisting
of a series of layers f [•,ϕ ] with parameters ϕ , which are composed together as:
k k k
(cid:20) h (cid:2) (cid:3) i (cid:21)
x=f[z,ϕ]=f K f K−1 ...f 2 f 1 [z,ϕ 1 ],ϕ 2 ,...ϕ K−1 ,ϕ K . (16.4)
Theinversemapping(normalizingdirection)isdefinedbythecompositionoftheinverse
of each layer f
−1[•,ϕ
] applied in the opposite order:
k k
(cid:20) h (cid:2) (cid:3) i (cid:21)
z=f
−1[x,ϕ]=f −1
f
−1
...f
−1
f
−1[x,ϕ
],ϕ ,...ϕ ,ϕ . (16.5)
1 2 K−1 K K K−1 2 1
The base density Pr(z) is usually defined as a multivariate standard normal (i.e., with
mean zero and identity covariance). Hence, the effect of each subsequent inverse layer is
tograduallymoveor“flow”thedatadensitytowardthisnormaldistribution(figure16.4).
This gives rise to the name “normalizing flows.”
The Jacobian of the forward mapping can be expressed as:
∂f[z,ϕ] = ∂f 1 [z,ϕ 1 ] · ∂f 2 [f 1 ,ϕ 2 ] ... ∂f K−1 [f K−2 ,ϕ K−1 ] ... ∂f K [f K−1 ,ϕ K ] , (16.6)
∂z ∂z ∂f 1 ∂f K−2 ∂f K−1
where we have overloaded the notation to make f the output of the function f [•,ϕ ].
k k k
The absolute determinant of this Jacobian can be computed by taking the product of
the individual absolute determinants:
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.