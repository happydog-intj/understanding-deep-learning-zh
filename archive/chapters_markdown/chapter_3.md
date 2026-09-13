# Chapter 3

*Pages: 39-54*

---

Chapter 3
Shallow neural networks
Chapter2introducedsupervisedlearningusing1Dlinearregression. However,thismodel
canonlydescribetheinput/outputrelationshipasaline. Thischapterintroducesshallow
neural networks. These describe piecewise linear functions and are expressive enough
to approximate arbitrarily complex relationships between multi-dimensional inputs and
outputs.
3.1 Neural network example
Shallowneuralnetworksarefunctionsy=f[x,ϕ]withparametersϕthatmapmultivari-
ate inputs x to multivariate outputs y. We defer a full definition until section 3.4 and
introduce the main ideas using an example network f[x,ϕ] that maps a scalar input x to
a scalar output y and has ten parameters ϕ={ϕ ,ϕ ,ϕ ,ϕ ,θ ,θ ,θ ,θ ,θ ,θ }:
0 1 2 3 10 11 20 21 30 31
y = f[x,ϕ]
= ϕ +ϕ a[θ +θ x]+ϕ a[θ +θ x]+ϕ a[θ +θ x]. (3.1)
0 1 10 11 2 20 21 3 30 31
We can break down this calculation into three parts: first, we compute three linear
functions of the input data (θ +θ x, θ +θ x, and θ +θ x). Second, we pass the
10 11 20 21 30 31
three results through an activation function a[•]. Finally, we weight the three resulting
activations with ϕ ,ϕ , and ϕ , sum them, and add an offset ϕ .
1 2 3 0
To complete the description, we must define the activation function a[•]. There are
many possibilities, but the most common choice is the rectified linear unit or ReLU:
(
0 z <0
a[z]=ReLU[z]= . (3.2)
z z ≥0
This returns the input when it is positive and zero otherwise (figure 3.1).
It is probably not obvious which family of input/output relations is represented by
equation 3.1. Nonetheless, the ideas from the previous chapter are all applicable. Equa-
tion 3.1 represents a family of functions where the particular member of the family
Draft: please send errata to udlbookmail@gmail.com.

26 3 Shallow neural networks
Figure 3.1 Rectified linear unit (ReLU).
This activation function returns zero if
the input is less than zero and returns
theinputunchangedotherwise. Inother
words, it clips negative values to zero.
Note that there are many other possi-
ble choices for the activation function
(see figure 3.13), but the ReLU is the
most commonly used and the easiest to
understand.
Figure 3.2 Family of functions defined by equation 3.1. a–c) Functions for three
differentchoicesofthetenparametersϕ. Ineachcase,theinput/outputrelation
is piecewise linear. However, the positions of the joints, the slopes of the linear
regions between them, and the overall height vary.
depends on the ten parameters in ϕ. If we know these parameters, we can perform
inference (predict y) by evaluating the equation for a given input x. Given a training
dataset {x ,y }I , we can define a least squares loss function L[ϕ] and use this to mea-
i i i=1
sure how effectively the model describes this dataset for any given parameter values ϕ.
To train the model, we search for the values ϕˆ that minimize this loss.
3.1.1 Neural network intuition
In fact, equation 3.1 represents a family of continuous piecewise linear functions (fig-
ure 3.2) with up to four linear regions. We now break down equation 3.1 and show why
it describes this family. To make this easier to understand, we split the function into
two parts. First, we introduce the intermediate quantities:
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

3.1 Neural network example 27
h = a[θ +θ x]
1 10 11
h = a[θ +θ x]
2 20 21
h = a[θ +θ x], (3.3)
3 30 31
where we refer to h , h , and h as hidden units. Second, we compute the output by
1 2 3
combining these hidden units with a linear function:1
y =ϕ +ϕ h +ϕ h +ϕ h . (3.4)
0 1 1 2 2 3 3
Figure 3.3 shows the flow of computation that creates the function in figure 3.2a.
Each hidden unit contains a linear function θ•0 +θ•1 x of the input, and that line is
clipped by the ReLU function a[•] below zero. The positions where the three lines cross
zero become the three “joints” in the final output. The three clipped lines are then
weighted by ϕ , ϕ , and ϕ , respectively. Finally, the offset ϕ is added, which controls
1 2 3 0
the overall height of the final function.
Problems3.1–3.8
Each linear region in figure 3.3j corresponds to a different activation pattern in the
hidden units. When a unit is clipped, we refer to it as inactive, and when it is not
clipped, we refer to it as active. For example, the shaded region receives contributions
fromh andh (whichareactive)butnotfromh (whichisinactive). Theslopeofeach
1 3 2
linear region is determined by the original slopes θ•1 of the active inputs for this region
andtheweightsϕ• thatweresubsequentlyapplied. Forexample,theslopeintheshaded
region (see problem 3.3) is θ ϕ +θ ϕ , where the first term is the slope in panel (g)
11 1 31 3
and the second term is the slope in panel (i).
Each hidden unit contributes one “joint” to the function, so with three hidden units,
Notebook3.1
there can be four linear regions. However, only three of the slopes of these regions are
ShallownetworksI
independent; the fourth is either zero (if all the hidden units are inactive in this region)
or is a sum of slopes from the other regions. Problem3.9
3.1.2 Depicting neural networks
We have been discussing a neural network with one input, one output, and three hidden
units. Wevisualizethisnetworkinfigure3.4a. Theinputisontheleft,thehiddenunits
are in the middle, and the output is on the right. Each connection represents one of the
ten parameters. To simplify this representation, we do not typically draw the intercept
parameters, so this network is usually depicted as in figure 3.4b.
∑
1Forthepurposesofthisbook,alinearfunctionhastheformz′=ϕ0+
i
ϕizi. Anyothertypeof
functionisnonlinear. Forinstance, theReLUfunction(equation3.2)andtheexampleneuralnetwork
thatcontainsit(equation3.1)arebothnonlinear. Seenotesatendofchapterforfurtherclarification.
Draft: please send errata to udlbookmail@gmail.com.

28 3 Shallow neural networks
Figure 3.3 Computation for function in figure 3.2a. a–c) The input x is passed
throughthreelinearfunctions,eachwithadifferenty-interceptθ•0 andslopeθ•1 .
d–f) Each line is passed through the ReLU activation function, which clips neg-
ative values to zero. g–i) The three clipped lines are then weighted (scaled) by
ϕ ,ϕ , and ϕ , respectively. j) Finally, the clipped and weighted functions are
1 2 3
summed, and an offset ϕ that controls the height is added. Each of the four
0
linear regions corresponds to a different activation pattern in the hidden units.
In the shaded region, h is inactive (clipped), but h and h are both active.
2 1 3
(Interactive figure)
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

3.2 Universal approximation theorem 29
Figure 3.4 Depicting neural networks. a) The input x is on the left, the hidden
units h ,h , and h in the center, and the output y on the right. Computation
1 2 3
flowsfromlefttoright. Theinputisusedtocomputethehiddenunits,whichare
combined to create the output. Each of the ten arrows represents a parameter
(intercepts in orange and slopes in black). Each parameter multiplies its source
and adds the result to its target. For example, we multiply the parameter ϕ
1
by source h and add it to y. We introduce additional nodes containing ones
1
(orange circles) to incorporate the offsets into this scheme, so we multiply ϕ by
0
one (with no effect) and add it to y. ReLU functions are applied at the hidden
units. b) More typically, the intercepts, ReLU functions, and parameter names
are omitted; this simpler depiction represents the same network.
3.2 Universal approximation theorem
In the previous section, we introduced an example neural network with one input, one
output, ReLU activation functions, and three hidden units. Let’s now generalize this
slightly and consider the case with D hidden units where the dth hidden unit is:
h =a[θ +θ x], (3.5)
d d0 d1
and these are combined linearly to create the output:
XD
y =ϕ + ϕ h . (3.6)
0 d d
d=1
The number of hidden units in a shallow network is a measure of the network capacity.
With ReLU activation functions, the output of a network with D hidden units has at
Problem3.10
mostD jointsandsoisapiecewiselinearfunctionwithatmostD+1linearregions. As
we add more hidden units, the model can approximate more complex functions.
Indeed, with enough capacity (hidden units), a shallow network can describe any
continuous1Dfunctiondefinedonacompactsubsetofthereallinetoarbitraryprecision.
Toseethis,considerthateverytimeweaddahiddenunit,weaddanotherlinearregionto
the function. As these regions become more numerous, they represent smaller sections
of the function, which are increasingly well approximated by a line (figure 3.5). The
universal approximation theorem proves that for any continuous function, there exists a
shallow network that can approximate this function to any specified precision.
Draft: please send errata to udlbookmail@gmail.com.

30 3 Shallow neural networks
Figure 3.5 Approximation of a 1D function (dashed line) by a piecewise linear
model. a–c) As the number of regions increases, the model becomes closer and
closer to the continuous function. A neural network with a scalar input creates
one extra linear region per hidden unit. This idea generalizes to functions in
D dimensions. The universal approximation theorem proves that, with enough
i
hidden units, there exists a shallow neural network that can describe any given
continuous function defined on a compact subset of RDi to arbitrary precision.
3.3 Multivariate inputs and outputs
Intheaboveexample,thenetworkhasasinglescalarinputxandasinglescalaroutputy.
However, the universal approximation theorem also holds for the more general case
wherethenetworkmapsmultivariateinputsx=[x ,x ,...,x ]T tomultivariateoutput
1 2 Di
predictions y = [y ,y ,...,y ]T. We first explore how to extend the model to predict
1 2 Do
multivariate outputs. Then we consider multivariate inputs. Finally, in section 3.4, we
present a general definition of a shallow neural network.
3.3.1 Visualizing multivariate outputs
Toextendthenetworktomultivariateoutputsy,wesimplyuseadifferentlinearfunction
of the hidden units for each output. So, a network with a scalar input x, four hidden
units h ,h ,h , and h , and a 2D multivariate output y=[y ,y ]T would be defined as:
1 2 3 4 1 2
h = a[θ +θ x]
1 10 11
h = a[θ +θ x]
2 20 21
h = a[θ +θ x]
3 30 31
h = a[θ +θ x], (3.7)
4 40 41
and
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

3.3 Multivariate inputs and outputs 31
Figure 3.6 Network with one input, four hidden units, and two outputs. a)
Visualizationofnetworkstructure. b)Thisnetworkproducestwopiecewiselinear
functions,y [x]andy [x]. Thefour“joints”ofthesefunctions(atverticaldotted
1 2
lines) are constrained to be in the same places since they share the same hidden
units, but the slopes and overall height may differ.
Figure 3.7 Visualization of neural net-
work with 2D multivariate input x =
[x ,x ]T and scalar output y.
1 2
y = ϕ +ϕ h +ϕ h +ϕ h +ϕ h
1 10 11 1 12 2 13 3 14 4
y = ϕ +ϕ h +ϕ h +ϕ h +ϕ h . (3.8)
2 20 21 1 22 2 23 3 24 4
The two outputs are two different linear functions of the hidden units.
As we saw in figure 3.3, the “joints” in the piecewise functions depend on where the
initial linear functions θ•0 +θ•1 x are clipped by the ReLU functions a[•] at the hidden
units. Sincebothoutputsy andy aredifferentlinearfunctionsofthesamefourhidden
1 2
Problem3.11
units, the four “joints” in each must be in the same places. However, the slopes of the
linear regions and the overall vertical offset can differ (figure 3.6).
3.3.2 Visualizing multivariate inputs
To cope with multivariate inputs x, we extend the linear relations between the input
and the hidden units. So a network with two inputs x=[x ,x ]T and a scalar output y
1 2
(figure 3.7) might have three hidden units defined by:
Draft: please send errata to udlbookmail@gmail.com.

32 3 Shallow neural networks
Figure 3.8 Processing in network with two inputs x = [x ,x ]T, three hidden
1 2
units h ,h ,h , and one output y. a–c) The input to each hidden unit is a
1 2 3
linearfunctionofthetwoinputs,whichcorrespondstoanorientedplane. Bright-
ness indicates function output. For example, in panel (a), the brightness repre-
sents θ +θ x +θ x . Thin lines are contours. d–f) Each plane is clipped by
10 11 1 12 2
theReLUactivationfunction(cyanlinesareequivalentto“joints”infigures3.3d–
f). g-i) The clipped planes are then weighted, and j) summed together with an
offsetthatdeterminestheoverallheightofthesurface. Theresultisacontinuous
surfacemadeupofconvexpiecewiselinearpolygonalregions. (Interactivefigure)
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

3.4 Shallow neural networks: general case 33
h = a[θ +θ x +θ x ]
1 10 11 1 12 2
h = a[θ +θ x +θ x ]
2 20 21 1 22 2
h = a[θ +θ x +θ x ], (3.9)
3 30 31 1 32 2
where there is now one slope parameter for each input. The hidden units are combined
to form the output in the usual way:
y =ϕ +ϕ h +ϕ h +ϕ h . (3.10)
0 1 1 2 2 3 3
Figure3.8illustratestheprocessingofthisnetwork. Eachhiddenunitreceivesalinear
Problems3.12–3.13
combination of the two inputs, which forms an oriented plane in the 3D input/output
space. The activation function clips the negative values of these planes to zero. The
Notebook3.2
clipped planes are then recombined in a second linear function (equation 3.10) to create ShallownetworksII
acontinuouspiecewiselinearsurfaceconsistingofconvexpolygonalregions(figure3.8j).
Each region corresponds to a different activation pattern. For example, in the central AppendixB.1.2
Convexregion
triangular region, the first and third hidden units are active, and the second is inactive.
When there are more than two inputs to the model, it becomes diﬀicult to visualize.
However, the interpretation is similar. The output will be a continuous piecewise linear
function of the input, where the linear regions are now convex polytopes in the multi-
dimensional input space.
Notethatastheinputdimensionsgrow,thenumberoflinearregionsincreasesrapidly
(figure 3.9). To get a feeling for how rapidly, consider that each hidden unit defines a
hyperplane that delineates the part of space where this unit is active from the part
Notebook3.3
where it is not (cyan lines in 3.8d–f). If we had the same number of hidden units as
Shallownetwork
input dimensions D i , we could align each hyperplane with one of the coordinate axes regions
(figure3.10). Fortwoinputdimensions,thiswoulddividethespaceintofourquadrants.
Forthreedimensions, thiswouldcreateeightoctants, andforD dimensions, thiswould
i
create2Di orthants. Shallowneuralnetworksusuallyhavemorehiddenunitsthaninput
dimensions, so they typically create more than 2Di linear regions.
3.4 Shallow neural networks: general case
Wehavedescribedseveralexampleshallownetworkstohelpdevelopintuitionabouthow
they work. We now define a general equation for a shallow neural network y = f[x,ϕ]
that maps a multi-dimensional input x ∈ RDi to a multi-dimensional output y ∈ RDo
using h∈RD hidden units. Each hidden unit is computed as:
" #
XDi
h =a θ + θ x , (3.11)
d d0 di i
i=1
and these are combined linearly to create the output:
Draft: please send errata to udlbookmail@gmail.com.

34 3 Shallow neural networks
Figure 3.9 Linear regions vs. hidden units. a) Maximum possible regions as a
function of the number of hidden units for five different input dimensions D =
i
{1,5,10,50,100}. The number of regions increases rapidly in high dimensions;
with D = 500 units and input size D = 100, there can be greater than 10107
i
regions(solidcircle). b)Thesamedataareplottedasafunctionofthenumberof
parameters. Thesolidcirclerepresentsthesamemodelasinpanel(a)withD=
500 hidden units. This network has 51,001 parameters and would be considered
very small by modern standards.
Figure3.10Numberoflinearregionsvs.inputdimensions. a)Withasingleinput
dimension,amodelwithonehiddenunitcreatesonejoint,whichdividestheaxis
into two linear regions. b) With two input dimensions, a model with two hidden
unitscandividetheinputspaceusingtwolines(herealignedwithaxes)tocreate
four regions. c) With three input dimensions, a model with three hidden units
candividetheinputspaceusingthreeplanes(againalignedwithaxes)tocreate
eight regions. Continuing this argument, it follows that a model with D input
i
dimensions and D hidden units can divide the input space with D hyperplanes
i i
to create 2Di linear regions.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

3.5 Terminology 35
Figure 3.11 Visualization of neural net-
work with three inputs and two out-
puts. This network has twenty param-
eters. Therearefifteenslopes(indicated
by arrows) and five offsets (not shown).
XD
y =ϕ + ϕ h , (3.12)
j j0 jd d
d=1
where a[•] is a nonlinear activation function. The model has parameters ϕ={θ••,ϕ•• }.
Figure 3.11 shows an example with three inputs, three hidden units, and two outputs.
Problems3.14–3.17
The activation function permits the model to describe nonlinear relations between
input and the output, and as such, it must be nonlinear itself; with no activation func-
tion, or a linear activation function, the overall mapping from input to output would
be restricted to be linear. Many different activation functions have been tried (see fig-
ure 3.13), but the most common choice is the ReLU (figure 3.1), which has the merit
Notebook3.4
of being easily interpretable. With ReLU activations, the network divides the input
Activation
space into convex polytopes defined by the intersections of hyperplanes computed by functions
the “joints” in the ReLU functions. Each convex polytope contains a different linear
function. The polytopes are the same for each output, but the linear functions they
contain can differ.
3.5 Terminology
Weconcludethischapterbyintroducingsometerminology. Regrettably,neuralnetworks
have a lot of associated jargon. They are often referred to in terms of layers. The left of
figure3.12istheinput layer,thecenteristhehidden layer,andtotherightistheoutput
layer. We would say that the network in figure 3.12 has one hidden layer containing
four hidden units. The hidden units themselves are sometimes referred to as neurons.
When we pass data through the network, the values of the inputs to the hidden layer
(i.e., before the ReLU functions are applied) are termed pre-activations. The values at
the hidden layer (i.e., after the ReLU functions) are termed activations.
Forhistoricalreasons,anyneuralnetworkwithatleastonehiddenlayerisalsocalled
amulti-layerperceptron,orMLPforshort. Networkswithonehiddenlayer(asdescribed
in this chapter) are sometimes referred to as shallow neural networks. Networks with
multiple hidden layers (as described in the next chapter) are referred to as deep neural
networks. Neural networks in which the connections form an acyclic graph (i.e., a graph
with no loops, as in all the examples in this chapter) are referred to as feed-forward
networks. If every element in one layer connects to every element in the next (as in
all the examples in this chapter), the network is fully connected. These connections
Draft: please send errata to udlbookmail@gmail.com.

36 3 Shallow neural networks
Figure 3.12 Terminology. A shallow network consists of an input layer, a hidden
layer, and an output layer. Each layer is connected to the next by forward con-
nections (arrows). For this reason, these models are referred to as feed-forward
networks. When every variable in one layer connects to every variable in the
next, we call this a fully connected network. Each connection represents a slope
parameter in the underlying equation, and these parameters are termed weights.
Thevariablesinthehiddenlayeraretermedneuronsorhiddenunits. Thevalues
feeding into the hidden units are termed pre-activations, and the values at the
hidden units (i.e., after the ReLU function is applied) are termed activations.
represent slope parameters in the underlying equations and are referred to as network
weights. The offset parameters (not shown in figure 3.12) are called biases.
3.6 Summary
Shallowneuralnetworkshaveonehiddenlayer. They(i)computeseverallinearfunctions
of the input, (ii) pass each result through an activation function, and then (iii) take a
linear combination of these activations to form the outputs. Shallow neural networks
make predictions y based on inputs x by dividing the input space into a continuous
surface of piecewise linear regions. With enough hidden units (neurons), shallow neural
networks can approximate any continuous function to arbitrary precision.
Chapter4discussesdeepneuralnetworks,whichextendthemodelsfromthischapter
by adding more hidden layers. Chapters 5–7 describe how to train these models.
Notes
“Neural” networks: If the models in this chapter are just functions, why are they called
“neural networks”? The connection is, unfortunately, tenuous. Visualizations like figure 3.12
consistofnodes(inputs,hiddenunits,andoutputs)thataredenselyconnectedtooneanother.
This bears a superficial similarity to neurons in the mammalian brain, which also have dense
connections. However, there is scant evidence that brain computation works in the same way
as neural networks, and it is unhelpful to think about biology going forward.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 37
Figure 3.13 Activation functions. a) Logistic sigmoid and tanh functions. b)
Leaky ReLU and parametric ReLU with parameter 0.25. c) SoftPlus, Gaussian
errorlinearunit,andsigmoidlinearunit. d)Exponentiallinearunitwithparam-
eters0.5and1.0,e)Scaledexponentiallinearunit. f)Swishwithparameters0.4,
1.0, and 1.4.
History of neural networks: McCulloch & Pitts (1943) first came up with the notion of an
artificialneuronthatcombinedinputstoproduceanoutput,butthismodeldidnothaveaprac-
tical learning algorithm. Rosenblatt (1958) developed the perceptron, which linearly combined
inputs and then thresholded them to make a yes/no decision. He also provided an algorithm
to learn the weights from data. Minsky & Papert (1969) argued that the linear function was
inadequate for general classification problems but that adding hidden layers with nonlinear
activation functions (hence the term multi-layer perceptron) could allow the learning of more
generalinput/outputrelations. However,theyconcludedthatRosenblatt’salgorithmcouldnot
learn the parameters of such models. It was not until the 1980s that a practical algorithm
(backpropagation, see chapter 7) was developed, and significant work on neural networks re-
sumed. ThehistoryofneuralnetworksischronicledbyKurenkov(2020),Sejnowski(2018),and
Schmidhuber (2022).
Activation functions: The ReLU function has been used as far back as Fukushima (1969).
However,intheearlydaysofneuralnetworks,itwasmorecommontousethelogisticsigmoidor
tanhactivationfunctions(figure3.13a). TheReLUwasre-popularizedbyJarrettetal.(2009),
Nair&Hinton(2010),andGlorotetal.(2011)andisanimportantpartofthesuccessstoryof
modernneuralnetworks. Ithasthenicepropertythatthederivativeoftheoutputwithrespect
to the input is always one for inputs greater than zero. This contributes to the stability and
eﬀiciency of training (see chapter 7) and contrasts with the derivatives of sigmoid activation
Draft: please send errata to udlbookmail@gmail.com.

38 3 Shallow neural networks
functions, which saturate (become close to zero) for large positive and large negative inputs.
However,theReLUfunctionhasthedisadvantagethatitsderivativeiszerofornegativeinputs.
If all the training examples produce negative inputs to a given ReLU function, then we cannot
improve the parameters feeding into this ReLU during training. The gradient with respect to
the incoming weights is locally flat, so we cannot “walk downhill.” This is known as the dying
ReLU problem. Many variations on the ReLU have been proposed to resolve this problem
(figure3.13b), including(i)theleaky ReLU(Maasetal.,2013), whichalsohasa linearoutput
fornegativevalueswithasmallerslopeof0.1,(ii)theparametricReLU(Heetal.,2015),which
treats the slope of the negative portion as an unknown parameter, and (iii) the concatenated
ReLU(Shangetal.,2016),whichproducestwooutputs,oneofwhichclipsbelowzero(i.e.,like
a typical ReLU) and one of which clips above zero.
A variety of smooth functions have also been investigated (figure 3.13c–d), including the soft-
plus function (Glorot et al., 2011), Gaussian error linear unit (Hendrycks & Gimpel, 2016),
sigmoid linear unit (Hendrycks & Gimpel, 2016), and exponential linear unit (Clevert et al.,
2015). MostoftheseareattemptstoavoidthedyingReLUproblemwhilelimitingthegradient
for negative values. Klambauer et al. (2017) introduced the scaled exponential linear unit (fig-
ure 3.13e), which is particularly interesting as it helps stabilize the variance of the activations
when the input variance has a limited range (see section 7.5). Ramachandran et al. (2017)
adopted an empirical approach to choosing an activation function. They searched the space
of possible functions to find the one that performed best over a variety of supervised learning
tasks. The optimal function was found to be a[x] = x/(1+exp[−βx]), where β is a learned
parameter(figure3.13f). TheytermedthisfunctionSwish. Interestingly,thiswasarediscovery
of activation functions previously proposed by Hendrycks & Gimpel (2016) and Elfwing et al.
(2018). Howardetal.(2019)approximatedSwishbytheHardSwishfunction,whichhasavery
similar shape but is faster to compute:
8
><0 z<−3
HardSwish[z]= z(z+3)/6 −3≤z≤3. (3.13)
>:
z z>3
There is no definitive answer as to which of these activations functions is empirically superior.
However, the leaky ReLU, parameterized ReLU, and many of the continuous functions can be
shown to provide minor performance gains over the ReLU in particular situations. We restrict
attentiontoneuralnetworkswiththebasicReLUfunctionfortherestofthisbookbecauseit’s
easy to characterize the functions they create in terms of the number of linear regions.
Universal approximation theorem: The width version of this theorem states that there
exists a network with one hidden layer containing a finite number of hidden units that can
approximateanyspecifiedcontinuousfunctiononacompactsubsetofRn toarbitraryaccuracy.
This was proved by Cybenko (1989) for a class of sigmoid activations and was later shown to
be true for a larger class of nonlinear activation functions (Hornik, 1991).
Number of linear regions: Consider a shallow network with D ≥ 2-dimensional inputs
i
and D hidden units. The number of linear regions is determined by the intersections of the D
hyperplanes created by the “joints” in the ReLU functions (e.g., figure 3.8d–f). Each region is
AppendixB.2 created by a different combination of the ReLU functions clipping or not clipping the input.
Binomial The number of regions created by D hypePrplane(cid:0)s i(cid:1)n the D
i
≤ D-dimensional input space was
coeﬀicient shown by Zaslavsky (1975) to be at most Di D (i.e., a sum of binomial coeﬀicients). As a
j=0 j
rule of thumb, shallow neural networks almost always have a larger number D of hidden units
Problem3.18 than input dimensions D
i
and create between 2Di and 2D linear regions.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 39
Linear, aﬀine, and nonlinear functions: Technically, a linear transformation f[•] is any
functionthatobeystheprincipleofsuperposition,sof[a+b]=f[a]+f[b]. Thisdefinitionimplies
that f[2a] = 2f[a].The weighted sum f[h ,h ,h ] = ϕ h +ϕ h +ϕ h is linear, but once the
1 2 3 1 1 2 2 3 3
offset (bias) is added so f[h ,h ,h ]=ϕ +ϕ h +ϕ h +ϕ h , this is no longer true. To see
1 2 3 0 1 1 2 2 3 3
this,considerthattheoutputisdoubledwhenwedoubletheargumentsoftheformerfunction.
This is not the case for the latter function, which is more properly termed an aﬀine function.
However, it is common in machine learning to conflate these terms. We follow this convention
in this book and refer to both as linear. All other functions we will encounter are nonlinear.
Problems
Problem 3.1 What kind of mapping from input to output would be created if the activation
function in equation 3.1 was linear so that a[z]=ψ +ψ z? What kind of mapping would be
0 1
created if the activation function was removed, so a[z]=z?
Problem 3.2 For each of the four linear regions in figure 3.3j, indicate which hidden units are
inactive and which are active (i.e., which do and do not clip their inputs).
Problem 3.3∗ Derive expressions for the positions of the “joints” in function in figure 3.3j in
terms of the ten parameters ϕ and the input x. Derive expressions for the slopes of the four
linear regions.
Problem 3.4 Draw a version of figure 3.3 where the y-intercept and slope of the third hidden
unit have changed as in figure 3.14c. Assume that the remaining parameters remain the same.
Figure 3.14 Processing in network with one input, three hidden units, and one
outputforproblem3.4. a–c)Theinputtoeachhiddenunitisalinearfunctionof
the inputs. The first two are the same as in figure 3.3, but the last one differs.
Problem 3.5 Prove that the following property holds for α∈R+:
ReLU[α·z]=α·ReLU[z]. (3.14)
This is known as the non-negative homogeneity property of the ReLU function.
Draft: please send errata to udlbookmail@gmail.com.

40 3 Shallow neural networks
Problem 3.6 Following on from problem 3.5, what happens to the shallow network defined in
equations 3.3 and 3.4 when we multiply the parameters θ and θ by a positive constant α
10 11
and divide the slope ϕ by the same parameter α? What happens if α is negative?
1
Problem3.7Considerfittingthemodelinequation3.1usingaleastsquareslossfunction. Does
this loss function have a unique minimum? i.e., is there a single “best” set of parameters?
Problem 3.8 ConsiderreplacingtheReLUactivationfunctionwith(i)theHeavisidestepfunc-
tion heaviside[z], (ii) the hyperbolic tangent function tanh[z], and (iii) the rectangular func-
tion rect[z], where:
8
( ><0 z<0
0 z<0
heaviside[z]= rect[z]= 1 0≤z≤1. (3.15)
1 z≥0 >:
0 z>1
Redraw a version of figure 3.3 for each of these functions. The original parameters were: ϕ=
{ϕ ,ϕ ,ϕ ,ϕ ,θ ,θ ,θ ,θ ,θ ,θ }={−0.23,−1.3,1.3,0.66,−0.2,0.4,−0.9,0.9,1.1,−0.7}.
0 1 2 3 10 11 20 21 30 31
Provideaninformaldescriptionofthefamilyoffunctionsthatcanbecreatedbyneuralnetworks
with one input, three hidden units, and one output for each activation function.
Problem 3.9∗ Show that the third linear region in figure 3.3 has a slope that is the sum of the
slopes of the first and fourth linear regions.
Problem 3.10 Consider a neural network with one input, one output, and three hidden units.
The construction in figure 3.3 shows how this creates four linear regions. Under what circum-
stances could this network produce a function with fewer than four linear regions?
Problem 3.11∗ How many parameters does the model in figure 3.6 have?
Problem 3.12 How many parameters does the model in figure 3.7 have?
Problem3.13Whatistheactivationpatternforeachofthesevenregionsinfigure3.8? Inother
words, which hidden units are active (pass the input) and which are inactive (clip the input)
for each region?
Problem 3.14 Write out the equations that define the network in figure 3.11. There should
be three equations to compute the three hidden units from the inputs and two equations to
compute the outputs from the hidden units.
Problem3.15∗ Whatisthemaximumpossiblenumberof3Dlinearregionsthatcanbecreated
by the network in figure 3.11?
Problem 3.16 Write out the equations for a network with two inputs, four hidden units, and
three outputs. Draw this model in the style of figure 3.11.
Problem 3.17∗ Equations 3.11 and 3.12 define a general neural network with D inputs, one
i
hiddenlayercontainingD hiddenunits,andD outputs. Findanexpressionforthenumberof
o
parameters in the model in terms of D , D, and D .
i o
Problem 3.18∗ Show that the maximum number of regions created by a shallow network
withD =2-dimensionalinput,D =1-dimensionaloutput,andD=3hiddenunitsisseven,as
i o
in figure 3.8j. Use the result of Zaslavsky (1975) that the maxPimum(cid:0)n(cid:1)umber of regions created
bypartitioningaD -dimensionalspacewithDhyperplanesis Di D . Whatisthemaximum
i j=0 j
number of regions if we add two more hidden units to this model, so D=5?
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.