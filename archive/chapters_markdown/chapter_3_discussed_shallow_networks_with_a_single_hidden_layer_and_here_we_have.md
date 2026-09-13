# Chapter 3: discussed shallow networks (with a single hidden layer), and here we have

*Pages: 63-69*

---

4.5 Shallow vs. deep neural networks 49
or even more compactly in matrix notation as:
h = a[θ +θx]
0
′
h = a[ψ +Ψh]
0
′ ′ ′ ′
y = ϕ +ϕh, (4.14)
0
where, in each case, the function a[•] applies the activation function separately to every
element of its vector input.
4.4.1 General formulation
This notation becomes cumbersome for networks with many layers. Hence, from now
on, we will describe the vector of hidden units at layer k as h , the vector of biases
k
(intercepts) that contribute to hidden layer k+1 as β , and the weights (slopes) that
k
are applied to the kth layer and contribute to the (k+1)th layer as Ω . A general deep
k
network y=f[x,ϕ] with K layers can now be written as:
h = a[β +Ω x]
1 0 0
h = a[β +Ω h ]
2 1 1 1
h = a[β +Ω h ]
3 2 2 2
.
.
.
h K = a[β K−1 +Ω K−1 h K−1 ]
y = β +Ω h . (4.15)
K K K
The parameters ϕ of this model comprise all of these weight matrices and bias vectors
ϕ={β ,Ω }K .
k k k=0
If the kth layer has D hidden units, then the bias vector β will be of size D .
k k−1 k
The last bias vector β has the size D of the output. The first weight matrix Ω has
size D ×D where D K is the size of the o input. The last weight matrix Ω is D × 0 D , Notebook4.3
1 i i K o K Deepnetworks
and the remaining matrices Ω are D ×D (figure 4.6).
k k+1 k
We can equivalently write the network as a single function:
Problems4.3–4.6
(cid:2) (cid:3)
y = β K +Ω K a β K−1 +Ω K−1 a[...β 2 +Ω 2 a[β 1 +Ω 1 a[β 0 +Ω 0 x]]...] .
(4.16)
4.5 Shallow vs. deep neural networks
Chapter 3 discussed shallow networks (with a single hidden layer), and here we have
described deep networks (with multiple hidden layers). We now compare these models.
Draft: please send errata to udlbookmail@gmail.com.

50 4 Deep neural networks
4.5.1 Ability to approximate different functions
In section 3.2, we argued that shallow neural networks with enough capacity (hidden
units) could model any continuous function arbitrarily closely. In this chapter, we saw
that a deep network with two hidden layers could represent the composition of two
shallow networks. If the second of these networks computes the identity function, then
this deep network replicates a single shallow network. Hence, it can also approximate
any continuous function arbitrarily closely given suﬀicient capacity.
Problem4.7
4.5.2 Number of linear regions per parameter
A shallow network with one input, one output, and D > 2 hidden units can create up
to D+1 linear regions and is defined by 3D+1 parameters. A deep network with one
Problems4.8–4.11
input, one output, and K layers of D >2 hidden units can create a function with up to
(D+1)K linear regions using 3D+1+(K−1)D(D+1) parameters.
Figure4.7ashowshowthemaximumnumberoflinearregionsincreasesasafunction
of the number of parameters for networks mapping scalar input x to scalar output y.
Deepneuralnetworkscreatemuchmorecomplexfunctionsforafixedparameterbudget.
This effect is magnified as the number of input dimensions D increases (figure 4.7b),
i
although computing the maximum number of regions is less straightforward.
Thisseemsattractive,buttheflexibilityofthefunctionsisstilllimitedbythenumber
of parameters. Deep networks can create extremely large numbers of linear regions, but
these contain complex dependencies and symmetries. We saw some of these when we
considereddeepnetworksas“folding”theinputspace(figure4.3). So,it’snotclearthat
the greater number of regions is an advantage unless (i) there are similar symmetries in
the real-world functions that we wish to approximate or (ii) we have reason to believe
that the mapping from input to output really does involve a composition of simpler
functions.
4.5.3 Depth eﬀiciency
Both deep and shallow networks can model arbitrary functions, but some functions
can be approximated much more eﬀiciently with deep networks. Functions have been
identifiedthatrequireashallownetworkwithexponentiallymorehiddenunitstoachieve
an equivalent approximation to that of a deep network. This phenomenon is referred to
as the depth eﬀiciency of neural networks. This property is also attractive, but it’s not
clear that the real-world functions that we want to approximate fall into this category.
4.5.4 Large, structured inputs
Wehavediscussedfullyconnectednetworkswhereeveryelementofeachlayercontributes
to every element of the subsequent one. However, these are not practical for large,
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

4.5 Shallow vs. deep neural networks 51
Figure 4.7 The maximum number of linear regions for neural networks increases
rapidly with the network depth. a) Network with D =1 input. Each curve rep-
i
resentsafixednumberofhiddenlayersK,aswevarythenumberofhiddenunits
D perlayer. Forafixedparameterbudget(horizontalposition),deepernetworks
produce more linear regions than shallower ones. A network with K = 5 layers
and D = 10 hidden units per layer has 471 parameters (highlighted point) and
can produce 161,051 regions. b) Network with D =10 inputs. Each subsequent
i
pointalongacurverepresentstenhiddenunits. Here,amodelwithK =5layers
andD=50hiddenunitsperlayerhas10,801parameters(highlightedpoint)and
can create more than 1040 linear regions.
structuredinputslikeimages,wheretheinputmightcomprise∼106 pixels. Thenumber
of parameters would be prohibitive, and moreover, we want different parts of the image
to be processed similarly; there is no point in independently learning to recognize the
same object at every possible position in the image.
Thesolutionistoprocesslocalimageregionsinparallelandthengraduallyintegrate
information from increasingly large regions. This kind of local-to-global processing is
diﬀicult to specify without using multiple layers (see chapter 10).
4.5.5 Training and generalization
A further possible advantage of deep networks over shallow networks is their ease of fit-
ting;itisusuallyeasiertotrainmoderatelydeepnetworksthantotrainshallowones(see
figure 20.2). It may be that over-parameterized deep models (i.e., those with more pa-
rametersthantrainingexamples)havealargefamilyofroughlyequivalentsolutionsthat
areeasytofind. However,asweaddmorehiddenlayers,trainingbecomesmorediﬀicult
again. Many methods have been developed to mitigate this problem (see chapter 11).
Deep neural networks also seem to generalize to new data better than shallow ones.
In practice, the best results for most tasks have been achieved using networks with tens
or hundreds of layers. Neither of these phenomena are well understood, and we return
to them in chapter 20.
Draft: please send errata to udlbookmail@gmail.com.

52 4 Deep neural networks
4.6 Summary
Inthischapter,wefirstconsideredwhathappenswhenwecomposetwoshallownetworks.
We argued that the first network “folds” the input space, and the second network then
applies a piecewise linear function. The effects of the second network are duplicated
where the input space is folded onto itself.
We then showed that this composition of shallow networks is a special case of a deep
network with two layers. We interpreted the ReLU functions in each layer as clipping
theinputfunctionsinmultipleplacesandcreatingmore“joints”intheoutputfunction.
We introduced the idea of hyperparameters, which for the networks we’ve seen so far,
comprise the number of hidden layers and the number of hidden units in each.
Finally, we compared shallow and deep networks. We saw that (i) both networks
can approximate any function given enough capacity, (ii) deep networks produce many
more linear regions per parameter, (iii) some functions can be approximated much more
eﬀiciently by deep networks, (iv) large, structured inputs like images are best processed
in multiple stages, and (v) in practice, the best results for most tasks are achieved using
deep networks with many layers.
Now that we understand deep and shallow network models, we turn our attention to
training them. In the next chapter, we discuss loss functions. For any given parameter
values ϕ, the loss function returns a single number that indicates the mismatch between
themodeloutputsandthegroundtruthpredictionsforatrainingdataset. Inchapters6
and 7, we deal with the training process itself, in which we seek the parameter values
that minimize this loss.
Notes
Deeplearning: Ithaslongbeenunderstoodthatitispossibletobuildmorecomplexfunctions
bycomposingshallowneuralnetworksordevelopingnetworkswithmorethanonehiddenlayer.
Indeed,theterm“deeplearning”wasfirstusedbyDechter(1986). However,interestwaslimited
due to practical concerns; it was not possible to train such networks well. The modern era of
deep learning was kick-started by startling improvements in image classification reported by
Krizhevsky et al. (2012). This sudden progress was arguably due to the confluence of four
factors: larger training datasets, improved processing power for training, the use of the ReLU
activation function, and the use of stochastic gradient descent (see chapter 6). LeCun et al.
(2015) present an overview of early advances in the modern era of deep learning.
Number of linear regions: For deep networks using a total of D hidden units with ReLU
activations, the upper bound on the number of regions is 2D (Montúfar et al., 2014). The
same authors show that a deep ReLU n(cid:16)etwork with D
i
-dime(cid:17)nsional input and K layers, each
containingD≥D
i
hiddenunits,hasO (D/D
i
)(K−1)DiDDi linearregions. Montúfar(2017),
Arora et al. (2016) and Serra et al. (2018) all provide tighter upper bounds that consider the
possibility that each layer has different numbers of hidden units. Serra et al. (2018) provide
an algorithm that counts the number of linear regions in a neural network, although it is only
practical for very small networks.
If the number of hidden units D in each of the K layers is the same, and D is an integer
multipleoftheinputdimensionalityD ,thenthemaximumnumberoflinearregionsN canbe
i r
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 53
computed exactly and is:
(cid:18) (cid:19) !
D
Di(K−1) XDi
D
N = +1 · . (4.17)
r D j
i
j=0
ThefirstterminthisexpressioncorrespondstothefirstK−1layersofthenetwork,whichcan
be thought of as repeatedly folding the input space. However, we now need to devote D/D
i
hidden units to each input dimension to create these folds. The last term in this equation (a
sum of binomial coeﬀicients) is the number of regions that a shallow network can create and is
AppendixB.2
attributable to the last layer. For further information, consult Montúfar et al. (2014), Pascanu
Binomialcoeﬀicient
et al. (2013), and Montúfar (2017).
Universal approximation theorem: We argued in section 4.5.1 that if the layers of a deep
network have enough hidden units, then the width version of the universal approximation the-
orem applies: there exists a network that can approximate any given continuous function on
a compact subset of RDi to arbitrary accuracy. Lu et al. (2017) proved that there exists a
networkwithReLUactivationfunctionsandatleastD +4hiddenunitsineachlayerthatcan
i
approximate any specified D -dimensional Lebesgue integrable function to arbitrary accuracy
i
givenenoughlayers. Thisisknownasthedepthversionoftheuniversalapproximationtheorem.
Depth eﬀiciency: Severalresultsshowthattherearefunctionsthatcanberealizedbydeep
networks but not by any shallow network whose capacity is bounded above exponentially. In
other words, it would take an exponentially larger number of units in a shallow network to
describe these functions accurately. This is known as the depth eﬀiciency of neural networks.
Telgarsky(2016)showsthatforanyintegerk,itispossibletoconstructnetworkswithoneinput,
one output, and O[k3] layers of constant width, which cannot be realized with O[k] layers and
less than 2k width. Perhaps surprisingly, Eldan & Shamir (2016) showed that when there are
multivariate inputs, there is a three-layer network that cannot be realized by any two-layer
network if the capacity is sub-exponential in the input dimension. Cohen et al. (2016), Safran
& Shamir (2017), and Poggio et al. (2017) also demonstrate functions that deep networks can
approximateeﬀiciently,butshallowonescannot. Liang&Srikant(2016)showthatforabroad
class of functions, including univariate functions, shallow networks require exponentially more
hidden units than deep networks for a given upper bound on the approximation error.
Width eﬀiciency: Luetal.(2017)investigatewhethertherearewideshallownetworks(i.e.,
shallow networks with lots of hidden units) that cannot be realized by narrow networks whose
depth is not substantially larger. They show that there exist classes of wide, shallow networks
that can only be expressed by narrow networks with polynomial depth. This is known as the
width eﬀiciency of neural networks. This polynomial lower bound on width is less restrictive
than the exponential lower bound on depth, suggesting that depth is more important. Vardi
et al. (2022) subsequently showed that the price for making the width small is only a linear
increase in the network depth for networks with ReLU activations.
Problems
Problem 4.1∗ Consider composing the two neural networks in figure 4.8. Draw a plot of the
relationship between the input x and output y′ for x∈[−1,1].
Problem 4.2 Identify the four hyperparameters in figure 4.6.
Problem 4.3 Using the non-negative homogeneity property of the ReLU function (see prob-
lem 3.5), show that:
Draft: please send errata to udlbookmail@gmail.com.

54 4 Deep neural networks
Figure 4.8Composition oftwonetworksforproblem4.1. a)Theoutput y ofthe
first network becomes the input to the second. b) The first network computes
this function with output values y ∈ [−1,1]. c) The second network computes
this function on the input range y∈[−1,1].
h i (cid:20) (cid:20) (cid:21)(cid:21)
1 1
ReLU β +λ ·Ω ReLU[β +λ ·Ω x] =λ λ ·ReLU β +Ω ReLU β +Ω x ,
1 1 1 0 0 0 0 1 λ λ 1 1 λ 0 0
0 1 0
(4.18)
where λ and λ are non-negative scalars. From this, we see that the weight matrices can be
0 1
rescaled by any magnitude as long as the biases are also adjusted, and the scale factors can be
re-applied at the end of the network.
Problem4.4WriteouttheequationsforadeepneuralnetworkthattakesD =5inputs,D =4
i o
outputs and has three hidden layers of sizes D = 20, D = 10, and D = 7, respectively, in
1 2 3
both the forms of equations 4.15 and 4.16. What are the sizes of each weight matrix Ω• and
bias vector β ?
•
Problem 4.5 Consider a deep neural network with D =5 inputs, D =1 output, and K =20
i o
hidden layers containing D=30 hidden units each. What is the depth of this network? What
is the width?
Problem 4.6 Consider a network with D =1 input, D =1 output, and K =10 layers, with
i o
D = 10 hidden units in each. Would the number of weights increase more if we increased the
depth by one or the width by one? Provide your reasoning.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 55
Problem 4.7 Choose values for the parameters ϕ={ϕ ,ϕ ,ϕ ,ϕ ,θ ,θ ,θ ,θ ,θ ,θ } for
0 1 2 3 10 11 20 21 30 31
theshallowneuralnetworkinequation3.1(withReLUactivationfunctions)thatwilldefinean
identity function over a finite range x∈[a,b].
Problem 4.8∗ Figure 4.9 shows the activations in the three hidden units of a shallow network
(as in figure 3.3). The slopes in the hidden units are 1.0, 1.0, and -1.0, respectively, and the
“joints”inthehiddenunitsareatpositions1/6,2/6,and4/6. Findvaluesofϕ ,ϕ ,ϕ ,andϕ
0 1 2 3
that will combine the hidden unit activations as ϕ +ϕ h +ϕ h +ϕ h to create a function
0 1 1 2 2 3 3
with four linear regions that oscillate between output values of zero and one. The slope of the
leftmost region should be positive, the next one negative, and so on. How many linear regions
will we create if we compose this network with itself? How many will we create if we compose
it with itself K times?
Problem4.9∗ Followingproblem4.8,isitpossibletocreateafunctionwiththreelinearregions
that oscillates back and forth between output values of zero and one using a shallow network
withtwohiddenunits? Isitpossibletocreateafunctionwithfivelinearregionsthatoscillates
in the same way using a shallow network with four hidden units?
Figure 4.9 Hidden unit activations for problem 4.8. a) First hidden unit has a
jointatpositionx=1/6andaslopeofoneintheactiveregion. b)Secondhidden
unit has a joint at position x = 2/6 and a slope of one in the active region. c)
Thirdhiddenunithasajointatpositionx=4/6andaslopeofminusoneinthe
active region.
Problem 4.10 Consider a deep neural network with a single input, a single output, and K
hiddenlayers,eachofwhichcontainsD hiddenunits. Showthatthisnetworkwillhaveatotal
of 3D+1+(K−1)D(D+1) parameters.
Problem 4.11∗ Consider two neural networks that map a scalar input x to a scalar output y.
The first network is shallow and has D=95 hidden units. The second is deep and has K =10
layers, each containing D = 5 hidden units. How many parameters does each network have?
Howmanylinearregionscaneachnetworkmake(seeequation4.17)? Whichwouldrunfaster?
Draft: please send errata to udlbookmail@gmail.com.