# Chapter 4: It also must be possible to evaluate the determinant of the Jacobian eﬀiciently for

*Pages: 323-340*

---

16.3 Invertible network layers 309
(cid:12) (cid:12) (cid:12) (cid:12) (cid:12) (cid:12) (cid:12) (cid:12) (cid:12) (cid:12)
(cid:12) (cid:12) (cid:12) (cid:12) (cid:12) (cid:12) (cid:12) (cid:12) (cid:12) (cid:12)
(cid:12) (cid:12) ∂f[z,ϕ](cid:12) (cid:12)= (cid:12) (cid:12) ∂f 1 [z,ϕ 1 ](cid:12) (cid:12) ·(cid:12) (cid:12) ∂f 2 [f 1 ,ϕ 2 ](cid:12) (cid:12)... (cid:12) (cid:12) ∂f K−1 [f K−2 ,ϕ K−1 ](cid:12) (cid:12) ·(cid:12) (cid:12) ∂f K [f K−1 ,ϕ K ](cid:12) (cid:12).
∂z ∂z ∂f 1 ∂f K−2 ∂f K−1
(16.7)
The absolute determinant of the Jacobian of the inverse mapping is found by applying
Problem16.3
the same rule to equation 16.5. It is the reciprocal of the absolute determinant in the
forward mapping.
We train normalizing flows with a dataset {x } of I training examples using the
i
negative log-likelihood criterion:
" (cid:12) (cid:12) #
ϕˆ = argmax
YI
Pr(z i )·
(cid:12)
(cid:12) (cid:12) ∂f[ ∂ z z i ,ϕ]
(cid:12)
(cid:12) (cid:12)
−1
ϕ i
"i=1 "(cid:12) (cid:12)# #
XI (cid:12) (cid:12) (cid:2) (cid:3)
= argmin log (cid:12) (cid:12) ∂f[ ∂ z z i ,ϕ](cid:12) (cid:12) −log Pr(z i ) , (16.8)
ϕ i
i=1
where z =f
−1[x
,ϕ], Pr(z ) is measured under the base distribution, and the absolute
i i i
determinant |∂f[z ,ϕ]/∂z | is given by equation 16.7.
i i
16.2.2 Desiderata for network layers
The theory of normalizing flows is straightforward. However, for this to be practical, we
need neural network layers f that have four properties.
k
1. Collectively, the set of network layers must be suﬀiciently expressive to map a
multivariate standard normal distribution to an arbitrary density.
2. The network layers must be invertible; each must define a unique one-to-one map-
ping from any input point to an output point (a bijection). If multiple inputs were
AppendixB.1
mapped to the same output, the inverse would be ambiguous.
Bijection
3. It must be possible to compute the inverse of each layer eﬀiciently. We need
to do this every time we evaluate the likelihood. This happens repeatedly during
training,sotheremustbeaclosed-formsolutionorafastalgorithmfortheinverse.
4. It also must be possible to evaluate the determinant of the Jacobian eﬀiciently for
either the forward or inverse mapping.
16.3 Invertible network layers
We now describe different invertible network layers or flows for use in these models.
We start with linear and elementwise flows. These are easy to invert, and it’s possible
to compute the determinant of their Jacobians, but neither is suﬀiciently expressive to
describe arbitrary transformations of the base density. However, they form the building
blocks of coupling, autoregressive, and residual flows, which are all more expressive.
Draft: please send errata to udlbookmail@gmail.com.

310 16 Normalizing flows
16.3.1 Linear flows
A linear flow has the form f[h] = β + Ωh. If the matrix Ω is invertible, the linear
transform is invertible. For Ω ∈ RD×D, the computation of the inverse is O[D3]. The
AppendixA
determinant of the Jacobian is just the determinant of Ω, which can also be computed
BigOnotation
in O[D3]. This means that linear flows become expensive as the dimension D increases.
If the matrix Ω takes a special form, then inversion and computation of the deter-
AppendixB.4
minant can become more eﬀicient, but the transformation becomes less general. For
Matrixtypes
example, diagonal matrices require only O[D] computation for the inversion and deter-
minant,buttheelementsofhdon’tinteract. Orthogonalmatricesarealsomoreeﬀicient
Problem16.4
to invert, and their determinant is fixed, but they do not allow scaling of the individual
dimensions. Triangular matrices are more practical; they are invertible using a process
known as back-substitution, which is O[D2], and the determinant is just the product of
the diagonal values.
One way to make a linear flow that is general, eﬀicient to invert, and for which the
Jacobian can be computed eﬀiciently is to parameterize it directly in terms of the LU
decomposition. In other words, we use:
Ω=PL(U+D), (16.9)
where P is a predetermined permutation matrix, L is a lower triangular matrix, U is
an upper triangular matrix with zeros on the diagonal, and D is a diagonal matrix that
supplies those missing diagonal elements. This can be inverted in O[D2], and the log
determinantisjustthesumofthelogoftheabsolutevaluesonthediagonalsofLandD.
Unfortunately, linear flows are not suﬀiciently expressive. When a linear func-
tion f[h] = β + Ωh is applied to normally distributed input Norm [µ,Σ], then the
h
result is also normally distributed with mean and variance, β+Ωµ and ΩΣΩT, respec-
Problems16.5–16.6
tively. Hence, it is not possible to map a normal distribution to an arbitrary density
using linear flows alone.
16.3.2 Elementwise flows
Since linear flows are not suﬀiciently expressive, we must turn to nonlinear flows. The
simplestoftheseareelementwiseflows,whichapplyapointwisenonlinearfunctionf[•,ϕ]
with parameters ϕ to each element of the input so that:
h i
T
f[h]= f[h ,ϕ],f[h ,ϕ],...f[h ,ϕ] . (16.10)
1 2 D
TheJacobian∂f[h]/∂hisdiagonalsincethedth inputtof[h]onlyaffectsthedth output.
Its determinant is the product of the entries on the diagonal, so:
(cid:12) (cid:12) (cid:12) (cid:12)
(cid:12) (cid:12) YD (cid:12) (cid:12)
(cid:12) (cid:12) ∂f[h](cid:12) (cid:12)= (cid:12) (cid:12) ∂f[h d ](cid:12) (cid:12). (16.11)
∂h ∂h
d
d=1
The function f[•,ϕ] could be a fixed invertible nonlinearity like the leaky ReLU
Problem16.7
(figure 3.13), in which case there are no parameters, or it may be any parameterized
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

16.3 Invertible network layers 311
Figure 16.5 Piecewise linear mapping. An invertible piecewise linear map-
ping h′ = f[h,ϕ] can be created by dividing the input domain h ∈ [0,1] into K
equally sized regions (here K =5). Each region has a slope with parameter, ϕ .
k
a) If these parameters are positive and sum to one, then b) the function will be
invertible and map to the output domain h′ ∈[0,1].
invertible one-to-one mapping. A simple example is a piecewise linear function with K
regions (figure 16.5) which maps [0,1] to [0,1] as:
!
Xb−1
f[h,ϕ]= ϕ +(hK−b+1)ϕ , (16.12)
k b
k=1
where the parameters ϕ ,ϕ ,...,ϕ are positive and sum to 1, and b=⌊Kh⌋+1 is the
1 2 K
index of the bin that contains h. The first term is the sum of all the preceding bins, and
thesecondtermrepresentstheproportionofthewaythroughthecurrentbinthathlies.
This function is easy to invert, and its gradient can be calculated almost everywhere.
Problems16.8–16.9
There are many similar schemes for creating smooth functions, often using splines with
parameters that ensure the function is monotonic and hence invertible.
Elementwiseflowsarenonlinearbutdon’tmixinputdimensions,sotheycan’tcreate
correlationsbetweenvariables. Whenalternatedwithlinearflows(whichdomixdimen-
sions),morecomplextransformationscanbemodeled. However,inpractice,elementwise
flows are used as components of more complex layers like coupling flows.
16.3.3 Coupling flows
Coupling flows divide the input h into two parts so that h = [hT,hT]T and define the
1 2
flow f[h,ϕ] as:
Draft: please send errata to udlbookmail@gmail.com.

312 16 Normalizing flows
Figure 16.6 Coupling flows. a) The input (orange vector) is divided into h
1
and h . The first part h′ of the output (cyan vector) is a copy of h . The
2 1 1
outputh′ iscreatedbyapplyinganinvertibletransformationg[•,ϕ]toh ,where
2 2
the parameters ϕ are themselves a (not necessarily invertible) function of h . b)
1
Intheinversemapping,h =h′. Thisallowsustocalculatetheparametersϕ[h ]
1 1 1
and then apply the inverse g−1[h′,ϕ] to retrieve h .
2 2
′
h = h
1 h1 i
′
h = g h ,ϕ[h ] . (16.13)
2 2 1
Hereg[•,ϕ]isanelementwiseflow(orotherinvertiblelayer)withparametersϕ[h ]that
1
are themselves a nonlinear function of the inputs h (figure 16.6). The function ϕ[•] is
1
usually a neural network of some kind and does not have to be invertible. The original
variables can be recovered as:
′
h = h
1 1 h i
h = g
−1
h
′
,ϕ[h ] . (16.14)
2 2 1
If the function g[•,ϕ] is an elementwise flow, the Jacobian will be lower triangular
with the identity matrix in the top-left quadrant and the derivatives of the elementwise
transformation in the bottom-right. Its determinant is the product of these diagonal
values.
The inverse and Jacobian can be computed eﬀiciently, but this approach only trans-
formsthesecondhalfoftheparametersinawaythatdependsonthefirsthalf. Tomake
a more general transformation, the elements of h are randomly shuffled using permuta-
AppendixB.4.4
tion matrices between layers, so every variable is ultimately transformed by every other.
Permutationmatrix
In practice, these permutation matrices are diﬀicult to learn. Hence, they are initialized
randomlyandthenfrozen. Forstructureddatalikeimages,thechannelsaredividedinto
two halves h and h and permuted between layers using 1×1 convolutions.
1 2
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

16.3 Invertible network layers 313
Figure 16.7 Autoregressive flows. The input h (orange column) and output h′
(cyancolumn)aresplitintotheirconstituentdimensions(herefourdimensions).
a) Output h′ is an invertible transformation of input h . Output h′ is an in-
1 1 2
vertible function of input h where the parameters depend on h . Output h′
2 1 3
is an invertible function of input h where the parameters depend on previous
3
inputs h and h , and so on. None of the outputs depend on one another, so
1 2
they can be computed in parallel. b) The inverse of the autoregressive flow is
computed using a similar method as for coupling flows. However, notice that to
compute h we must already know h , to compute h , we must already know h
2 1 3 1
and h , and so on. Consequently, the inverse cannot be computed in parallel.
2
16.3.4 Autoregressive flows
Autoregressiveflowsareageneralizationofcouplingflowsthattreateachinputdimension
as a separate “block” (figure 16.7). They compute the dth dimension of the output h′
based on the first d−1 dimensions of the input h:
h i
′
h d =g h d ,ϕ[h 1:d−1 ] . (16.15)
Thefunctiong[•,•]istermedthetransformer,1andtheparametersϕ,ϕ[h ],ϕ[h ,h ],...
1 1 2
are termed conditioners. As for coupling flows, the transformer g[•,ϕ] must be invert-
ible, buttheconditionersϕ[•]cantakeanyformandareusuallyneuralnetworks. Ifthe
transformer and conditioner are suﬀiciently flexible, autoregressive flows are universal
approximators in that they can represent any probability distribution.
It’s possible to compute all of the entriesof the output h′ in parallel using a network
with appropriate masks so that the parameters ϕ at position d only depend on previous
1Thisisnothingtodowiththetransformerlayersdiscussedinchapter12.
Draft: please send errata to udlbookmail@gmail.com.

314 16 Normalizing flows
positions. Thisisknownasamaskedautoregressiveflow. Theprincipleisverysimilarto
maskedself-attention(section12.7.2);connectionsthatrelateinputstopreviousoutputs
are pruned.
Inverting the transformation is less eﬀicient. Consider the forward mapping:
h i
′
h = g h ,ϕ
1 1
h i
′
h = g h ,ϕ[h ]
2 2 1
h i
′
h = g h ,ϕ[h ]
3 3 1:2
h i
′
h = g h ,ϕ[h ] . (16.16)
4 4 1:3
This must be inverted sequentially using a similar principle as for coupling flows:
h i
h = g
−1
h
′
,ϕ
1 1
h i
h = g
−1
h
′
,ϕ[h ]
2 2 1
h i
h = g
−1
h
′
,ϕ[h ]
3 3 1:2
h i
h = g
−1
h
′
,ϕ[h ] . (16.17)
4 4 1:3
This can’t be done in parallel as the computation for h d depends on h 1:d−1 (i.e., the
Notebook16.2
partial results so far). Hence, inversion is time-consuming when the input is large.
Autoregressiveflows
16.3.5 Inverse autoregressive flows
Masked autoregressive flows are defined in the normalizing (inverse) direction. This is
required to evaluate the likelihood eﬀiciently and hence to learn the model. However,
sampling requires the forward direction, in which each variable must be computed se-
quentially at each layer, which is slow. If we use an autoregressive flow for the forward
(generative)transformation,thensamplingiseﬀicient,butcomputingthelikelihood(and
training) is slow. This is known as an inverse autoregressive flow.
A trick that allows fast learning and also fast (but approximate) sampling is to
build a masked autoregressive flow to learn the distribution (the teacher) and then use
this to train an inverse autoregressive flow from which we can sample eﬀiciently (the
student). This requires a different formulation of normalizing flows that learns from
another function rather than a set of samples (see section 16.5.3).
16.3.6 Residual flows: iRevNet
Residual flows take their inspiration from residual networks. They divide the input into
two parts h=[hT,hT]T (as for coupling flows) and define the outputs as:
1 2
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

16.3 Invertible network layers 315
Figure16.8Residualflows. a)Aninvertiblefunctioniscomputedbysplittingthe
inputintoh andh andcreatingtworesiduallayers. Inthefirst,h isprocessed
1 2 2
and h is added. In the second, the result is processed, and h is added. b) In
1 2
thereversemechanismthefunctionsarecomputedintheoppositeorder,andthe
addition operation becomes subtraction.
′
h = h +f [h ,ϕ ]
1 1 1 2 1
′ ′
h = h +f [h ,ϕ ], (16.18)
2 2 2 1 2
wheref [•,ϕ ]andf [•,ϕ ]aretwofunctionsthatdonotnecessarilyhavetobeinvertible
1 1 2 2
(figure 16.8). The inverse can be computed by reversing the order of computation:
h = h ′ −f [h ′ ,ϕ ]
2 2 2 1 2
h = h ′ −f [h ,ϕ ]. (16.19)
1 1 1 2 1
As for coupling flows, the division into blocks restricts the family of transformations
that can be represented. Hence, the inputs are permuted between layers so that the
variables can mix in arbitrary ways.
Thisformulationcanbeinvertedeasily,butforgeneralfunctionsf [•,ϕ ]andf [•,ϕ ],
1 1 2 2
thereisnoeﬀicientwaytocomputetheJacobian. Thisformulationissometimesusedto
Problem16.10
save memory when training residual networks; because the network is invertible, storing
the activations at each layer in the forward pass is unnecessary.
16.3.7 Residual flows and contraction mappings: iResNet
A different approach to exploiting residual networks is to utilize the Banach fixed point
theorem or contraction mapping theorem, which states that every contraction mapping
has a fixed point. A contraction mapping f[•] has the property that:
Draft: please send errata to udlbookmail@gmail.com.

316 16 Normalizing flows
Figure 16.9 Contraction mappings. If a function has an absolute slope of less
thanoneeverywhere,iteratingthefunctionconvergestoafixedpointf[z]=z. a)
Startingatz ,weevaluatez =f[z ]. Wethenpassz backintothefunctionand
0 1 0 1
iterate. Eventually,theprocessconvergestothepointwheref[z]=z (i.e., where
the function crosses the dashed diagonal identity line). b) This can be used to
invertequations of the form y=z+f[z] fora value y∗ bynoticing that the fixed
point of y∗−f[z] (where the orange line crosses the dashed identity line) is at
the same position as where y∗ =z+f[z].
h i h i
dist f[z ′ ],f[z] <β·dist z ′ ,z ∀z,z ′ , (16.20)
wheredist[•,•]isadistancefunctionand0<β <1. Whenafunctionwiththisproperty
isiterated(i.e.,theoutputisrepeatedlypassedbackinasaninput),theresultconverges
Notebook16.3
to a fixed point where f[z] = z (figure 16.9). To understand this, consider applying the
Contractionmappings
function to both the fixed point and the current position; the fixed point remains static,
butthedistancebetweenthetwomustbecomesmaller, sothecurrentpositionmustget
closer to the fixed point.
This theorem can be exploited to invert an equation of the form:
y =z+f[z] (16.21)
if f[z] is a contraction mapping. In other words, it can be used to find the z∗ that maps
toagivenvalue,y∗. Thiscanbedonebystartingwithanypointz anditeratingz =
0 k+1
y∗−f[z ]. This has a fixed point at z+f[z]=y∗ (figure 16.9b).
k
AppendixB.1.1 The same principle can be used to invert residual network layers of the form h′ =
Lipschitzconstant
h+f[h,ϕ]ifweensurethatf[h,ϕ]isacontractionmapping. Inpractice,thismeansthat
AppendixB.3.7
the Lipschitz constant must be less than one. Assuming that the slope of the activation
Singularvalues
functionsisnotgreaterthanone,thisisequivalenttoensuringthelargestsingularvalue
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

16.4 Multi-scale flows 317
of each weight matrix Ω must be less than one. A crude way to do this is to ensure that
the absolute magnitudes of the weights Ω are small by clipping them.
The Jacobian determinant cannot be computed easily, but its logarithm can be ap-
proximated using a series of tricks.
"(cid:12) (cid:12)# " (cid:20) (cid:21)#
(cid:12) (cid:12)
(cid:12) ∂f[h,ϕ](cid:12) ∂f[h,ϕ]
log (cid:12)I+ (cid:12) = trace log I+
∂h ∂h
" #
X∞ (−1)k−1 ∂f[h,ϕ] k
= trace , (16.22)
k ∂h
k=1
where we have used the identity log[|A|] = trace[log[A]] in the first line and expanded
this into a power series in the second line.
Even when we truncate this series, it’s still computationally expensive to compute
AppendixB.3.8
the trace of the constituent terms. Hence, we approximate this using Hutchinson’s trace
Trace
estimator. Consider a normal random variable ϵ with mean 0 and variance I. The trace
of a matrix A can be estimated as:
(cid:2) (cid:2) (cid:3)(cid:3)
trace[A] = trace AE ϵϵT
(cid:2) (cid:2) (cid:3)(cid:3)
= trace E AϵϵT
(cid:2) (cid:2) (cid:3)(cid:3)
= E trace AϵϵT
(cid:2) (cid:2) (cid:3)(cid:3)
= E trace ϵTAϵ
(cid:2) (cid:3)
= E ϵTAϵ , (16.23)
wherethefirstlineistruebecauseE[ϵϵT]=I. Thesecondlinederivesfromtheproperties
oftheexpectationoperator. Thethirdlinecomesfromthelinearityofthetraceoperator.
Thefourthlineisduetotheinvarianceofthetracetocyclicpermutation. Thefinalline
is true because the argument in the fourth line is now a scalar. We estimate the trace
by drawing samples ϵ from Pr(ϵ):
i
(cid:2) (cid:3)
trace[A] = E ϵTAϵ
XI
1
≈ ϵTAϵ . (16.24)
I i i
i=1
In this way, we can approximate the trace of the powers of the Taylor expansion (equa-
tion 16.22) and evaluate the log probability.
16.4 Multi-scale flows
In normalizing flows, the latent space z must be the same size as the data space x, but
we know that natural datasets can often be described by fewer underlying variables. At
Draft: please send errata to udlbookmail@gmail.com.

318 16 Normalizing flows
Figure 16.10 Multiscale flows. The latent space z must be the same size as the
model density in normalizing flows. However, it can be partitioned into several
components, which can be gradually introduced at different layers. This makes
both density estimation and sampling faster. For the inverse process, the black
arrowsarereversed,andthelastpartofeachblockskipstheremainingprocessing.
For example, f−1[•,ϕ ] only operates on the first three blocks, and the fourth
3 3
block becomes z and is assessed against the base density.
4
some point, we have to introduce all of these variables, but it is ineﬀicient to pass them
through the entire network. This leads to the idea of multi-scale flows (figure 16.10).
In the generative direction, multi-scale flows partition the latent vector into z =
[z ,z ,...,z ]. The first partition z is processed by a series of reversible layers with
1 2 N 1
the same dimension as z until, at some point, z is appended and combined with the
1 2
first partition. This continues until the network is the same size as the data x. In the
normalizing direction, the network starts at the full dimension of x, but when it reaches
the point where z was added, this is assessed against the base distribution.
n
16.5 Applications
We now describe three applications of normalizing flows. First, we consider modeling
probability densities. Second, we consider the GLOW model for synthesizing images.
Finally, we discuss using normalizing flows to approximate other distributions.
16.5.1 Modeling densities
Ofthefourgenerativemodelsdiscussedinthisbook,normalizingflowsistheonlymodel
that can compute the exact log-likelihood of a new sample. Generative adversarial
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

16.5 Applications 319
Figure 16.11 Modeling densities. a) Toy 2D data samples. b) Modeled density
using iResNet. c–d) Second example. Adapted from Behrmann et al. (2019)
networks are not probabilistic, and both variational autoencoders and diffusion models
can only return a lower bound on the likelihood.2 Figure 16.11 depicts the estimated
probabilitydistributionsintwotoyproblemsusingi-ResNet. Oneapplicationofdensity
estimation is anomaly detection; the data distribution of a clean dataset is described
using a normalizing flow model. New examples with low probability are flagged as
outliers. However,cautionmustbeusedastheremayexistoutlierswithhighprobability
that don’t fall in the typical set (see figure 8.13).
16.5.2 Synthesis
Generative flows, or GLOW, is a normalizing flow model that can create high-fidelity
images (figure 16.12) and uses many of the ideas from this chapter. It is easiest under-
stood in the normalizing direction. GLOWstarts with a 256×256×3 tensor containing
an RGB image. It uses coupling layers, in which the channels are partitioned into two
halves. Thesecondhalfissubjecttoadifferentaﬀinetransformateachspatialposition,
where the parameters of the aﬀine transformation are computed by a 2D convolutional
neural network run on the other half of the channels. The coupling layers are alternated
with 1×1 convolutions, parameterized as LU decompositions which mix the channels.
Periodically, the resolution is halvedbycombiningeach 2×2 patchintoone position
withfourtimesasmanychannels. GLOWisamulti-scaleflow,andsomeofthechannels
are periodically removed to become part of the latent vector z. Images are discrete (due
tothequantizationofRGBvalues),sonoiseisaddedtotheinputstopreventthetraining
likelihood increasing without bound. This is known as dequantization.
To sample more realistic images, the GLOW model samples from the base density
raised to a positive power. This chooses examples that are closer to the center of the
density rather than from the tails. This is similar to the truncation trick in GANs
2Thelowerboundonthelikelihoodfordiffusionmodelscanactuallyexceedtheexactcomputation
innormalizingflows,butdatagenerationismuchslower(seechapter18).
Draft: please send errata to udlbookmail@gmail.com.

320 16 Normalizing flows
Figure 16.12 Samples from GLOW trained on the CelebA HQ dataset (Karras
etal.,2018). Thesamplesareofreasonablequality,althoughGANsanddiffusion
models produce superior results. Adapted from Kingma & Dhariwal (2018).
Figure16.13InterpolationusingGLOWmodel. Theleftandrightimagesarereal
people. Theintermediateimageswerecomputedbyprojectingtherealimagesto
the latent space, interpolating, and then projecting the interpolated points back
to image space. Adapted from Kingma & Dhariwal (2018).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

16.6 Summary 321
(figure 15.10). Notably, the samples are not as good as those from GANs or diffusion
models. It is unknown whether this is due to a fundamental restriction associated with
invertible layers or merely because less research effort has been invested in this goal.
Figure16.13showsanexampleofinterpolationusingGLOW.Twolatentvectorsare
computed by transforming two real images in the normalizing direction. Intermediate
points between these latent vectors are computed by linear interpolation, and these are
projected back to image space using the network in the generative direction. The result
is a set of images that interpolate realistically between the two real ones.
16.5.3 Approximating other density models
Normalizingflowscanalsolearntogeneratesamplesthatapproximateanexistingdensity
which is easy to evaluate but diﬀicult to sample from. In this context, we denote the
normalizing flow Pr(x|ϕ) as the student and the target density q(x) as the teacher.
To make progress, we generate samples x = f[z ,ϕ] from the student. Since we
i i
generated these samples ourselves, we know their corresponding latent variables z , and
i
we can calculate their likelihood in the student model without inversion. Thus, we can
use a model like a masked-autoregressive flow where inversion is slow. We define a loss
function based on the reverse KL divergence that encourages the student and teacher
likelihood to be identical and use this to train the student model (figure 16.14): Problem16.11
" " (cid:12)(cid:12) ##
XI (cid:2) (cid:3)(cid:12)(cid:12)
ϕˆ =argmin KL I 1 δ x−f[z i ,ϕ] (cid:12) (cid:12) (cid:12) (cid:12)q(x) . (16.25)
ϕ
i=1
This approach contrasts with the typical use of normalizing flows to build a proba-
bility model Pr(x ,ϕ) of data that came from an unknown distribution with samples x
i i
using maximum likelihood, which relies on the cross-entropy term from the forward KL
divergence (section 5.7):
" " (cid:12)(cid:12) ##
XI (cid:12)(cid:12)
ϕˆ =argmin KL I 1 δ[x−x i ] (cid:12) (cid:12) (cid:12) (cid:12)Pr(x i ,ϕ) . (16.26)
ϕ
i=1
Normalizing flows can model the posterior in VAEs using this trick (see chapter 17).
16.6 Summary
Normalizingflowstransformabasedistribution(usuallyanormaldistribution)tocreate
a new density. They have the advantage that they can both evaluate the likelihood
of samples exactly and generate new samples. However, they have the architectural
constraint that each layer must be invertible; we need the forward transformation to
generate samples and the backward transformation to evaluate the likelihoods.
It’s also important that the Jacobian can be estimated eﬀiciently to evaluate the
likelihood; this must be done repeatedly to learn the density. However, invertible layers
Draft: please send errata to udlbookmail@gmail.com.

322 16 Normalizing flows
Figure 16.14 Approximating density models. a) Training data. b) Usually, we
modifytheflowmodelparameterstominimizetheKLdivergencefromthetrain-
ing data to the flow model. This is equivalent to maximum likelihood fitting
(section5.7). c)Alternatively,wecanmodifytheflowparametersϕtominimize
the KL divergence from the flow samples x =f[z ,ϕ] to d) a target density.
i i
arestillusefulintheirownrightevenwhentheJacobiancannotbeestimatedeﬀiciently;
they reduce the memory requirements of training a K-layer network from O[K] to O[1].
This chapter reviewed invertible network layers or flows. We considered linear flows
andelementwiseflows,whicharesimplebutinsuﬀicientlyexpressive. Thenwedescribed
more complex flows, such as coupling, autoregressive, and residual flows. Finally, we
showed how normalizing flows can be used to estimate likelihoods, generate and inter-
polate between images, and approximate other distributions.
Notes
Normalizing flows were first introduced by Rezende & Mohamed (2015) but had intellectual
antecedents in the work of Tabak & Vanden-Eijnden (2010), Tabak & Turner (2013), and
Rippel & Adams (2013). Reviews of normalizing flows can be found in Kobyzev et al. (2020)
and Papamakarios et al. (2021). Kobyzev et al. (2020) presented a quantitative comparison of
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 323
many normalizing flow approaches. They concluded that the Flow++ model (a coupling flow
with a novel elementwise transformation and other innovations) performed best at the time.
Invertible network layers: Invertible layers decrease the memory requirements of the back-
propagation algorithm; the activations in the forward pass no longer need to be stored since
they can be recomputed in the backward pass. In addition to the regular network layers and
residual layers (Gomez et al., 2017; Jacobsen et al., 2018) discussed in this chapter, invertible
layers have been developed for graph neural networks (Li et al., 2021a), recurrent neural net-
works (MacKay et al., 2018), masked convolutions (Song et al., 2019), U-Nets (Brügger et al.,
2019; Etmann et al., 2020), and transformers (Mangalam et al., 2022).
Radial and planar flows: Theoriginalnormalizingflowspaper(Rezende&Mohamed,2015)
used planar flows (which contract or expand the distribution along certain dimensions) and
radial flows (which expand or contract around a certain point). Inverses for these flows can’t
becomputedeasily,buttheyareusefulforapproximatingdistributionswheresamplingisslow
or where the likelihood can only be evaluated up to an unknown scaling factor (figure 16.14).
Applications: Applications include image generation (Ho et al., 2019; Kingma & Dhariwal,
2018), noise modeling (Abdelhamed et al., 2019), video generation (Kumar et al., 2019b), au-
dio generation (Esling et al., 2019; Kim et al., 2018; Prenger et al., 2019), graph generation
(Madhawa et al., 2019), image classification (Kim et al., 2021; Mackowiak et al., 2021), im-
age steganography (Lu et al., 2021), super-resolution (Yu et al., 2020; Wolf et al., 2021; Liang
etal.,2021),styletransfer(Anetal.,2021),motionstyletransfer(Wenetal.,2021),3Dshape
modeling (Paschalidou et al., 2021), compression (Zhang et al., 2021b), sRGB to RAW image
conversion(Xingetal.,2021),denoising(Liuetal.,2021b),anomalydetection(Yuetal.,2021),
image-to-image translation (Ardizzone et al., 2020), synthesizing cell microscopy images under
different molecular interventions (Yang et al., 2021), and light transport simulation (Müller
etal.,2019b). Forapplicationsusingimagedata,noisemustbeaddedbeforelearningsincethe
inputs are quantized and hence discrete (see Theis et al., 2016).
Rezende & Mohamed (2015) used normalizing flows to model the posterior in VAEs. Abdal
etal.(2021)usednormalizingflowstomodelthedistributionofattributesinthelatentspaceof
StyleGANandthenusedthesedistributionstochangespecifiedattributesinrealimages. Wolf
etal.(2021)usenormalizingflowstolearntheconditionalimageofanoisyinputimagegivena
cleanoneandhencesimulatenoisydatathatcanbeusedtotraindenoisingorsuper-resolution
models.
Normalizing flows have also found diverse uses in physics (Kanwar et al., 2020; Köhler et al.,
2020;Noéetal.,2019;Wirnsbergeretal.,2020;Wongetal.,2020),naturallanguageprocessing
(Tranetal.,2019;Ziegler&Rush,2019;Zhouetal.,2019;Heetal.,2018;Jinetal.,2019),and
reinforcement learning (Schroecker et al., 2019; Haarnoja et al., 2018a; Mazoure et al., 2020;
Ward et al., 2019; Touati et al., 2020).
Linear flows: Diagonal linear flows can represent normalization transformations like Batch-
Norm(Dinhetal.,2016)andActNorm(Kingma&Dhariwal,2018). Tomczak&Welling(2016)
investigatedcombiningtriangularmatricesandusingorthogonaltransformationsparameterized
by the Householder transform. Kingma & Dhariwal (2018) proposed the LU parameterization
described in section 16.5.2. Hoogeboom et al. (2019b) proposed using the QR decomposition
instead, which does not require predetermined permutation matrices. Convolutions are lin-
ear transformations (figure 10.4) that are widely used in deep learning, but their inverse and
determinant are not straightforward to compute. Kingma & Dhariwal (2018) used 1×1 con-
volutions, which is effectively a full linear transformation applied separately at each position.
Zhengetal.(2017)introducedConvFlow,whichwasrestrictedto1Dconvolutions. Hoogeboom
et al. (2019b) provided more general solutions for modeling 2D convolutions either by stacking
together masked autoregressive convolutions or by operating in the Fourier domain.
Draft: please send errata to udlbookmail@gmail.com.

324 16 Normalizing flows
Elementwise flows and coupling functions: Elementwise flows transform each variable
independently using the same function (but with different parameters for each variable). The
same flows can be used to form the coupling functions in coupling and autoregressive flows, in
whichcasetheirparametersdependontheprecedingvariables. Tobeinvertible,thesefunctions
must be monotone.
An additive coupling function (Dinh et al., 2015) just adds an offset to the variable. Aﬀine
coupling functions scale the variable and add an offset and were used by Dinh et al. (2015),
Dinh et al. (2016), Kingma & Dhariwal (2018), Kingma et al. (2016), and Papamakarios et al.
(2017). Ziegler & Rush (2019) propose the nonlinear squared flow, which is an invertible ratio
of polynomials with five parameters. Continuous mixture CDFs (Ho et al., 2019) apply a
monotone transformation based on the cumulative density function (CDF) of a mixture of K
logistics, post-composed by an inverse logistic sigmoid, scaled, and offset.
The piecewise linear coupling function (figure 16.5) was developed by Müller et al. (2019b).
Sincethen,systemsbasedoncubicsplines(Durkanetal.,2019a)andrationalquadraticsplines
(Durkanetal.,2019b)havebeenproposed. Huangetal.(2018a)introducedneuralautoregres-
siveflows,inwhichthefunctionisrepresentedbyaneuralnetworkthatproducesamonotonic
function. A suﬀicient condition is that the weights are all positive and the activation functions
aremonotone. Itishardtotrainanetworkwiththeconstraintthattheweightsarepositive,so
this led to unconstrained monotone neural networks(Wehenkel & Louppe, 2019), whichmodel
strictly positive functions and then integrate them numerically to get a monotone function.
Jainietal.(2019)constructpositivefunctionsthatcanbeintegratedinclosedformbasedona
classicresultthatallpositivesingle-variablepolynomialsarethesumofsquaresofpolynomials.
Finally, Dinh et al. (2019) investigated piecewise monotonic coupling functions.
Coupling flows: Dinh et al. (2015) introduced coupling flows in which the dimensions were
splitinhalf(figure16.6). Dinhetal.(2016)introduced RealNVP,whichpartitionedtheimage
input by taking alternating pixels or blocks of channels. Das et al. (2019) proposed selecting
features for the propagated part based on the magnitude of the derivatives. Dinh et al. (2016)
interpretedmulti-scaleflows(inwhichdimensionsaregraduallyintroduced)ascouplingflowsin
which the parameters ϕ have no dependence on the other half of the data. Kruse et al. (2021)
introduce a hierarchical formulation of coupling flows in which each partition is recursively
dividedintotwo. GLOW(figures16.12–16.13)wasdesignedbyKingma&Dhariwal(2018)and
usescouplingflows,asdoNICE(Dinhetal.,2015),RealNVP(Dinhetal.,2016),FloWaveNet
(Kim et al., 2018), WaveGlOW (Prenger et al., 2019), and Flow++ (Ho et al., 2019).
Autoregressiveflows: Kingmaetal.(2016)usedautoregressivemodelsfornormalizingflows.
Germain et al. (2015) developed a general method for masking previous variables. This was
exploited by Papamakarios et al. (2017) to compute all of the outputs in the forward direction
simultaneously in masked autoregressive flows. Kingma et al. (2016) introduced the inverse
autoregressive flow. Parallel WaveNet (Van den Oord et al., 2018) distilled WaveNet (Van den
Oord et al., 2016a), which is a different type of generative model for audio, into an inverse
autoregressive flow so that sampling would be fast (see figure 16.14c–d).
Residual flows: Residual flows are based on residual networks (He et al., 2016a). RevNets
(Gomez et al., 2017) and iRevNets (Jacobsen et al., 2018) divide the input into two sections
(figure 16.8), each of which passes through a residual network. These networks are invertible,
but the determinant of the Jacobian cannot be computed easily. The residual connection can
beinterpretedasthediscretizationofanordinarydifferentialequation,andthisperspectiveled
todifferentinvertiblearchitectures(Changetal.,2018,2019a). However,theJacobianofthese
networkscouldstillnotbecomputedeﬀiciently. Behrmannetal.(2019)notedthatthenetwork
canbeinvertedusingfixedpointiterationsifitsLipschitzconstantislessthanone. Thisledto
iResNet,inwhichthelogdeterminantoftheJacobiancanbeestimatedusingHutchinson’strace
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 325
estimator (Hutchinson, 1989). Chen et al. (2019) removed the bias induced by the truncation
of the power series in equation 16.22 by using the Russian Roulette estimator.
Infinitesimal flows: If residual networks can be viewed as a discretization of an ordinary
differentialequation(ODE),thenthenextlogicalstepistorepresentthechangeinthevariables
directlybyanODE.TheneuralODEwasexploredbyChenetal.(2018e)andexploitsstandard
methods for forward and backward propagation in ODEs. The Jacobian is no longer required
to compute the likelihood; this is represented by a different ODE in which the change in log
probability is related to the trace of the derivative of the forward propagation. Grathwohl
et al. (2019) used the Hutchinson estimator to estimate the trace and simplified this further.
Finlay et al. (2020) added regularization terms to the loss function that make training easier,
and Dupont et al. (2019) augmented the representation to allow the neural ODE to represent
a broader class of diffeomorphisms. Tzen & Raginsky (2019) and Peluchetti & Favaro (2020)
replaced the ODEs with stochastic differential equations.
Universality: The universality property refers to the ability of a normalizing flow to model
anyprobabilitydistributionarbitrarilywell. Someflows(e.g.,planar,elementwise)donothave
this property. Autoregressive flows can be shown to have the universality property when the
coupling function is a neural monotone network (Huang et al., 2018a), based on monotone
polynomials (Jaini et al., 2020) or based on splines (Kobyzev et al., 2020). For dimension D,
a series of D coupling flows can form an autoregressive flow. To understand why, note that
the partitioning into two parts h and h means that at any given layer h depends only on
1 2 2
the previous variables (figure 16.6). Hence, if we increase the size of h by one at every layer,
1
we can reproduce an autoregressive flow, and the result is universal. It is not known whether
coupling flows can be universal with fewer than D layers. However, they work well in practice
(e.g., GLOW) without the need for this induced autoregressive structure.
Otherwork: Activeareasofresearchinnormalizingflowsincludetheinvestigationofdiscrete
flows(Hoogeboometal.,2019a;Tranetal.,2019),normalizingflowsonnon-Euclideanmanifolds
(Gemicietal.,2016;Wang&Wang,2019),andequivariant flows(Köhleretal.,2020;Rezende
et al., 2019) which aim to create densities that are invariant to families of transformations.
Problems
Problem 16.1 Consider transforming a uniform base density defined on z ∈ [0,1] using the
function x=f[z]=z2. Find an expression for the transformed distribution Pr(x).
Problem 16.2∗ Consider transforming a standard normal distribution:
(cid:20) (cid:21)
1 −z2
Pr(z)= √ exp , (16.27)
2π 2
with the function:
1
x=f[z]= . (16.28)
1+exp[−z]
Find an expression for the transformed distribution Pr(x).
Problem 16.3∗ WriteexpressionsfortheJacobianoftheinversemappingz=f−1[x,ϕ]andthe
absolute determinant of that Jacobian in forms similar to equations 16.6 and 16.7.
Draft: please send errata to udlbookmail@gmail.com.

326 16 Normalizing flows
Problem 16.4 Compute the inverse and the determinant of the following matrices by hand:
2 3 2 3
2 0 0 0 1 0 0 0
6 60 −5 0 0 7 7 6 62 4 0 0 7 7
Ω 1 =4 0 0 1 0 5 Ω 2 =4 1 −1 2 0 5. (16.29)
0 0 0 2 4 −2 −2 1
Problem16.5ConsiderarandomvariablezwithmeanµandcovarianceΣthatistransformed
as x=Az+b. Show that the expected value of x is Aµ+b and that the covariance of x is
AΣAT.
Problem 16.6∗ Prove that if x = f[z] = Az+b and Pr(z) = Norm [µ,Σ], then Pr(x) =
z
Norm [Aµ+b,AΣAT] using the relation:
x
(cid:12) (cid:12)
Pr(x)=Pr(z)·
(cid:12)
(cid:12) (cid:12) ∂f[z]
(cid:12)
(cid:12) (cid:12)
−1
. (16.30)
∂z
Problem 16.7 The Leaky ReLU is defined as:
(
0.1z z<0
LReLU[z]= . (16.31)
z z≥0
Write an expression for the inverse of the leaky ReLU. Write an expression for the inverse
absolute determinant of the Jacobian |∂f[z]/∂z|−1 for an elementwise transformation x = f[z]
of the multivariate variable z where:
h i
T
f[z]= LReLU[z ],LReLU[z ],...,LReLU[z ] . (16.32)
1 2 D
Problem 16.8 Consider applying the piecewise linear function f[h,ϕ] defined in equation 16.12
for the domain h′ ∈ [0,1] elementwise to an input h = [h ,h ,...,h ]T so that f[h] =
1 2 D
[f[h ,ϕ],f[h ,ϕ],...,f[h ,ϕ]]. What is the Jacobian ∂f[h]/∂h? What is the determinant of
1 2 D
the Jacobian?
Problem 16.9∗ Consider constructing an element-wise flow based on a conical combination of
square root functions in equally spaced bins:
p Xb−1p
h ′ =f[h,ϕ]= [Kh−b+1]ϕ + ϕ , (16.33)
b k
k=1
whereb=⌊Kh⌋+1isthebinthathfallsinto,andtheparametersϕ arepositive,andsumto
k
one. Consider the case where K =5 and ϕ =0.1,ϕ =0.2,ϕ =0.5,ϕ =0.1,ϕ =0.1. Draw
1 2 3 4 5
the function f[h,ϕ]. Draw the inverse function f−1[h′,ϕ].
Problem 16.10 DrawthestructureoftheJacobian(indicatingwhichelementsarezero)forthe
forwardmappingoftheresidualflowinfigure16.8forthecaseswheref [•,ϕ ]andf [•,ϕ ]are
1 1 2 2
(i) a fully connected neural network, (ii) an elementwise flow.
Problem 16.11∗ Write out the expression for the KL divergence in equation 16.25. Why does
it not matter if we can only evaluate the probability q(x) up to a scaling factor κ? Does the
network have to be invertible to minimize this loss function? Explain your reasoning.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.