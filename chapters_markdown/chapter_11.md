# Chapter 11

*Pages: 200-220*

---

Chapter 11
Residual networks
The previous chapter described how image classification performance improved as the
depth of convolutional networks was extended from eight layers (AlexNet) to nineteen
layers (VGG). This led to experimentation with even deeper networks. However, per-
formance decreased again when many more layers were added.
This chapter introduces residual blocks. Here, each network layer computes an addi-
tive change to the current representation instead of transforming it directly. This allows
deeper networks to be trained but causes an exponential increase in the activation mag-
nitudes at initialization. Residual blocks employ batch normalization to compensate for
this, which re-centers and rescales the activations at each layer.
Residual blocks with batch normalization allow much deeper networks to be trained,
and these networks improve performance across a variety of tasks. Architectures that
combine residual blocks to tackle image classification, medical image segmentation, and
human pose estimation are described.
11.1 Sequential processing
Every network we have seen so far processes the data sequentially; each layer receives
the previous layer’s output and passes the result to the next (figure 11.1). For example,
a three-layer network is defined by:
h = f [x,ϕ ]
1 1 1
h = f [h ,ϕ ]
2 2 1 2
h = f [h ,ϕ ]
3 3 2 3
y = f [h ,ϕ ], (11.1)
4 3 4
where h , h , and h denote the intermediate hidden layers, x is the network input, y
1 2 3
is the output, and the functions f [•,ϕ ] perform the processing.
k k
In a standard neural network, each layer consists of a linear transformation followed
byanactivationfunction, andtheparametersϕ comprisetheweightsandbiasesofthe
k
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

11.1 Sequential processing 187
Figure 11.1 Sequential processing. Standard neural networks pass the output of
each layer directly into the next layer.
lineartransformation. Inaconvolutionalnetwork,eachlayerconsistsofasetofconvolu-
tions followed by an activation function, and the parameters comprise the convolutional
kernels and biases.
Since the processing is sequential, we can equivalently think of this network as a
series of nested functions:
(cid:20) h (cid:2) (cid:3) i (cid:21)
y=f f f f [x,ϕ ],ϕ ,ϕ ,ϕ . (11.2)
4 3 2 1 1 2 3 4
11.1.1 Limitations of sequential processing
Inprinciple, wecan add asmanylayersas wewant, andin the previous chapter, wesaw
thataddingmorelayerstoaconvolutionalnetworkdoesimproveperformance;theVGG
network (figure 10.17), which has nineteen layers, outperforms AlexNet (figure 10.16),
which has eight layers. However, image classification performance decreases again as
further layers are added (figure 11.2). This is surprising since models generally perform
betterasmorecapacityisadded(figure8.10). Indeed,thedecreaseispresentforboththe
training set and the test set, which implies that the problem is training deeper networks
rather than the inability of deeper networks to generalize.
This phenomenon is not completely understood. One conjecture is that right after
initialization, the loss gradients change unpredictably when we modify parameters in
earlynetworklayers. Withappropriateinitializationoftheweights(seesection7.5), the
gradientofthelosswithrespecttotheseparameterswillbereasonable(i.e.,noexploding
or vanishing gradients). However, the derivative assumes an infinitesimal change in the
parameter,whereasoptimizationalgorithmsuseafinitestepsize. Anyreasonablechoice
Notebook11.1
of step size may move to a place with a completely different and unrelated gradient; the
Shattered
loss surface looks like an enormous range of tiny mountains rather than a single smooth gradients
structure that is easy to descend. Consequently, the algorithm doesn’t make progress in
the way that it does when the loss function gradient changes more slowly.
This conjecture is supported by empirical observations of gradients in networks with
a single input and output. For a shallow network, the gradient of the output with re-
spect to the input changes slowly as we change the input (figure 11.3a). However, for a
AppendixB.2.1
deep network, a tiny change in the input results in a completely different gradient (fig-
Autocorrelation
ure11.3b). Thisiscapturedbytheautocorrelationfunctionofthegradient(figure11.3c). function
Nearby gradients are correlated for shallow networks, but this correlation quickly drops
to zero for deep networks. This is termed the shattered gradients phenomenon.
Draft: please send errata to udlbookmail@gmail.com.

188 11 Residual networks
Figure11.2Decreaseinperformancewhenaddingmoreconvolutionallayers. a)A
20-layer convolutional network outperforms a 56-layer neural network for image
classification on the test set of the CIFAR-10 dataset (Krizhevsky & Hinton,
2009). b) This is also true for the training set, which suggests that the problem
relatestotrainingtheoriginalnetworkratherthanafailuretogeneralizetonew
data. Adapted from He et al. (2016a).
Figure 11.3 Shattered gradients. a) Consider a shallownetwork with 200 hidden
units and Glorot initialization (He initialization without the factor of two) for
both the weights and biases. The gradient ∂y/∂x of the scalar network output y
with respect to the scalar input x changes relatively slowly as we change the in-
put x. b) For a deep network with 24 layers and 200 hidden units per layer, this
gradientchangesveryquicklyandunpredictably. c)Theautocorrelationfunction
ofthegradientshowsthatnearbygradientsbecomeunrelated(haveautocorrela-
tion close to zero) for deep networks. This shattered gradients phenomenon may
explain why it is hard to train deep networks. Gradient descent algorithms rely
on the loss surface being relatively smooth, so the gradients should be related
before and after each update step. Adapted from Balduzzi et al. (2017).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

11.2 Residual connections and residual blocks 189
Shatteredgradientspresumablyarisebecausechangesinearlynetworklayersmodify
theoutputinanincreasinglycomplexwayasthenetworkbecomesdeeper. Thederivative
of the output y with respect to the first layer f of the network in equation 11.1 is:
1
AppendixB.5
Matrixcalculus
∂y ∂f ∂f ∂f
= 2 3 4. (11.3)
∂f ∂f ∂f ∂f
1 1 2 3
Whenwechangetheparametersthatdeterminef ,allofthederivativesinthissequence
1
are evaluated at slightly different locations since layers f ,f , and f are themselves
2 3 4
computed from f . Consequently, the updated gradient at each training example may
1
be completely different, and the loss function becomes badly behaved.1
11.2 Residual connections and residual blocks
Residual or skip connections are branches in the computational path, whereby the input
to each network layer f[•] is added back to the output (figure 11.4a). By analogy to
equation 11.1, the residual network is defined as:
h = x+f [x,ϕ ]
1 1 1
h = h +f [h ,ϕ ]
2 1 2 1 2
h = h +f [h ,ϕ ]
3 2 3 2 3
y = h +f [h ,ϕ ], (11.4)
3 4 3 4
where the first term on the right-hand side of each line is the residual connection. Each
function f learns an additive change to the current representation. It follows that their
k
outputs must be the same size as their inputs. Each additive combination of the input
and the processed output is known as a residual block or residual layer.
Once more, we can write this as a single function by substituting in the expressions
Problem11.1
for the intermediate quantities h :
k
y=x + f [x] (11.5)
1(cid:2) (cid:3)
+ f x+f [x]
2h 1 (cid:2) (cid:3)i
+ f x+f [x]+f x+f [x]
3 1 2 1
(cid:20) (cid:2) (cid:3) h (cid:2) (cid:3)i(cid:21)
+ f x+f [x]+f x+f [x] +f x+f [x]+f x+f [x] ,
4 1 2 1 3 1 2 1
where we have omitted the parameters ϕ for clarity. We can think of this equation as
•
“unraveling” the network (figure 11.4b). We see that the final network output is a sum
of the input and four smaller networks, corresponding to each line of the equation; one
1Inequations11.3and11.6,weoverloadnotationtodefinef astheoutputofthefunctionf [•].
k k
Draft: please send errata to udlbookmail@gmail.com.

190 11 Residual networks
Figure 11.4 Residual connections. a) The output of each function f [x,ϕ ] is
k k
addedbacktoitsinput,whichispassedviaaparallelcomputationalpathcalled
a residual or skip connection. Hence, the function computes an additive change
totherepresentation. b)Uponexpanding(unraveling)thenetworkequations,we
findthattheoutputisthesumoftheinputplusfoursmallernetworks(depicted
in white, orange, gray, and cyan, respectively, and corresponding to terms in
equation 11.5); we can think of this as an ensemble of networks. Moreover,
the output from the cyan network is itself a transformation f [•,ϕ ] of another
4 4
ensemble,andsoon. Alternatively,wecanconsiderthenetworkasacombination
of16differentpathsthroughthecomputationalgraph. Oneexampleisthedashed
path from input x to output y, which is the same in panels (a) and (b).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

11.2 Residual connections and residual blocks 191
Figure 11.5Orderofoperationsinresid-
ual blocks. a) The usual order of linear
transformation or convolution followed
byaReLUnonlinearitymeansthateach
residualblockcanonlyaddnon-negative
quantities. b) With the reverse order,
bothpositiveandnegativequantitiescan
beadded. However,wemustaddalinear
transformation at the start of the net-
workincasetheinputisallnegative. c)
In practice, it’s common for a residual
block to contain several network layers.
interpretation is that residual connections turn the original network into an ensemble of
these smaller networks whose outputs are summed to compute the result.
Acomplementarywayofthinkingaboutthisresidualnetworkisthatitcreatessixteen
pathswithdifferingnumbersoftransformationsbetweeninputandoutput. Forexample,
Problem11.2
thefirstfunctionf [x]occursineightofthesesixteenpaths,includingasadirectadditive
1
term (i.e., a path length of one), and the analogous derivative to equation 11.3 is:
Problem11.3
(cid:18) (cid:19) (cid:18) (cid:19)
∂y ∂f ∂f ∂f ∂f ∂f ∂f ∂f ∂f ∂f ∂f ∂f ∂f
=I+ 2 + 3 + 2 3 + 4 + 2 4 + 3 4 + 2 3 4 , (11.6)
∂f ∂f ∂f ∂f ∂f ∂f ∂f ∂f ∂f ∂f ∂f ∂f ∂f
1 1 1 1 2 1 1 2 1 3 1 2 3
where there is one term for each of the eight paths. The identity term on the right-
hand side shows that changes in the parameters ϕ in the first layer f [x,ϕ ] contribute
1 1 1
directly to changes in the network output y. They also contribute indirectly through
the other chains of derivatives of varying lengths. In general, gradients through shorter
Notebook11.2
paths will be better behaved. Since both the identity term and various short chains of
Residual
derivatives will contribute to the derivative for each layer, networks with residual links networks
suffer less from shattered gradients.
11.2.1 Order of operations in residual blocks
Until now, we have implied that the additive functions f[x] could be any valid network
layer (e.g., fully connected or convolutional). This is technically true, but the order of
operations in these functions is important. They must contain a nonlinear activation
functionlikeaReLU,ortheentirenetworkwillbelinear. However, inatypicalnetwork
layer (figure 11.5a), the ReLU function is at the end, so the output is non-negative. If
we adopt this convention, then each residual block can only increase the input values.
Hence,itistypicaltochangetheorderofoperationssothattheactivationfunctionis
applied first, followed by the linear transformation (figure 11.5b). Sometimes there may
be several layers of processing within the residual block (figure 11.5c), but these usually
terminatewithalineartransformation. Finally,wenotethatwhenwestarttheseblocks
withaReLUoperation,theywilldonothingiftheinitialnetworkinputisnegativesince
theReLU will clipthe entiresignalto zero. Hence, it’s typicaltostart the networkwith
a linear transformation rather than a residual block, as in figure 11.5b.
Draft: please send errata to udlbookmail@gmail.com.

192 11 Residual networks
11.2.2 Deeper networks with residual connections
Adding residual connections roughly doubles the depth of a network that can be practi-
callytrainedbeforeperformancedegrades. However,wewouldliketoincreasethedepth
further. To understand why residual connections do not allow us to increase the depth
arbitrarily, we must consider how the variance of the activations changes during the
forward pass and how the gradient magnitudes change during the backward pass.
11.3 Exploding gradients in residual networks
In section 7.5, we saw that initializing the network parameters is critical. Without
careful initialization, the magnitudes of the intermediate values during the forward pass
ofbackpropagationcanincreaseordecreaseexponentially. Similarly,thegradientsduring
the backward pass can explode or vanish as we move backward through the network.
Hence, we initialize the network parameters so that the expected variance of the
activations(intheforwardpass)andgradients(inthebackwardpass)remainsthesame
between layers. He initialization (section 7.5) achieves this for ReLU activations by
initializing the biases β to zero and choosing normally distributed weights Ω with mean
zero and variance 2/D where D is the number of hidden units in the previous layer.
h h
Now consider a residual network. We do not have to worry about the intermediate
values or gradients vanishing with network depth since there exists a path whereby
each layer directly contributes to the network output (equation 11.5 and figure 11.4b).
However, even if we use He initialization within the residual block, the values in the
forward pass increase exponentially as we move through the network.
Toseewhy,considerthatweaddtheresultoftheprocessingintheresidualblockback
totheinput. Eachbranchhassome(uncorrelated)variability. Hence,theoverallvariance
Problem11.4
increases when we recombine them. With ReLU activations and He initialization, the
expected variance is unchanged by the processing in each block. Consequently, when
we recombine with the input, the variance doubles (figure 11.6a), growing exponentially
withthenumberofresidualblocks. Thislimitsthepossiblenetworkdepthbeforefloating
point precision is exceeded in the forward pass. A similar argument applies to the
gradients in the backward pass of the backpropagation algorithm.
Hence,residualnetworksstillsufferfromunstableforwardpropagationandexploding
gradientsevenwithHeinitialization. Oneapproachthatwouldstabilizetheforwardand
backwardpasseswouldbeto√useHeinitializationandthenmultiplythecombinedoutput
of each residual block by 1/ 2 to compensate for the doubling (figure 11.6b). However,
it is more usual to use batch normalization.
11.4 Batch normalization
Batch normalizationorBatchNormshiftsandrescaleseachactivationhsothatitsmean
and variance across the batch B become values that are learned during training. First,
the empirical mean m and standard deviation s are computed:
h h
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

11.4 Batch normalization 193
Figure 11.6 Variance in residual networks. a) He initialization ensures that the
expectedvarianceremainsunchangedafteralinearplusReLUlayerf . Unfortu-
k
nately,inresidualnetworks,theinputofeachblockisaddedbacktotheoutput,
sothevariancedoublesateachlayer(graynumbersindicatevariance√)andgrows
exponentially. b) One approach would be to rescale the signal by 1/ 2 between
each residual block. c) A second method uses batch normalization (BN) as the
first step in the residual block and initializes the associated offset δ to zero and
scaleγ toone. Thistransformstheinputtoeachlayertohaveunitvariance,and
with He initialization, the output variance will also be one. Now the variance
increases linearly with the number of residual blocks. A side-effect is that, at
initialization, later network layers are dominated by the residual connection and
are hence close to computing the identity.
X
1
m = h
h |B| i
s
i∈B
X
1
s = (h −m )2, (11.7)
h |B| i h
i∈B
where all quantities are scalars. Then we use these statistics to standardize the batch
AppendixC.2.4
activations to have mean zero and unit variance:
Standardization
h −m
h ← i h ∀i∈B, (11.8)
i s +ϵ
h
where ϵ is a small number that prevents division by zero if h is the same for every
i
member of the batch and s =0.
h
Finally, the normalized variable is scaled by γ and shifted by δ:
h ←γh +δ ∀i∈B. (11.9)
i i
Draft: please send errata to udlbookmail@gmail.com.

194 11 Residual networks
After this operation, the activations have mean δ and standard deviation γ across all
Problem11.5
members of the batch. Both of these quantities are learned during training.
Batch normalization is applied independently to each hidden unit. In a standard
neural network with K layers, each containing D hidden units, there would be KD
Problem11.6
learned offsets δ and KD learned scales γ. In a convolutional network, the normalizing
statistics are computed over both the batch and the spatial position. If there were K
Notebook11.3
layers, each containing C channels, there would be KC offsets and KC scales. At test
BatchNorm
time, we do not have a batch from which we can gather statistics. To resolve this, the
statistics m and s are calculated across the whole training dataset (rather than just a
h h
batch) and frozen in the final network.
11.4.1 Costs and benefits of batch normalization
Batch normalization makes the network invariant to rescaling the weights and biases
thatcontributetoeachactivation; ifthesearedoubled, thentheactivationsalsodouble,
the estimated standard deviation s doubles, and the normalization in equation 11.8
h
compensates for these changes.2 This happens separately for each hidden unit. Con-
sequently, there will be a large family of weights and biases that all produce the same
effect. Batch normalization also adds two parameters, γ and δ, at every hidden unit,
which makes the model somewhat larger. Hence, it both creates redundancy in the
weights and biases and adds extra parameters to compensate for that redundancy. This
is obviously ineﬀicient, but batch normalization also provides several benefits.
Stableforwardpropagation: Ifweinitializetheoffsetsδtozeroandthescalesγ toone,
then each output activation will have unit variance. In a regular network, this ensures
thevarianceisstableduringforwardpropagationatinitialization. Inaresidualnetwork,
the variance must still increase as we add a new source of variation to the input at each
layer. However, it will increase linearly with each residual block; the kth layer adds one
unit of variance to the existing variance of k (figure 11.6c).
At initialization, this has the side-effect that later layers make a smaller change to
theoverallvariationthanearlierones. Thenetworkiseffectivelylessdeepatthestartof
training since later layers are close to computing the identity. As training proceeds, the
network can increase the scales γ in later layers and can control its own effective depth.
Higher learning rates: Empirical studies and theory both show that batch normaliza-
tion makes the loss surface and its gradient change more smoothly (i.e., reduces shat-
tered gradients). This means we can use higher learning rates as the surface is more
predictable. We saw in section 9.2 that higher learning rates improve test performance.
Regularization: We saw in chapter 9 that noise in the training process can improve
generalization. BatchNorminjectsnoisebecausethenormalizationdependsonthebatch
statistics. Theactivationsforagiventrainingexamplearenormalizedbyanamountthat
depends on the other members of the batch and is different at each training iteration.
2Technically,thisisonlytrueiftheBatchNormoperationisapplieddirectlyafterthenetworklayer.
Thesituationissomewhatmorecomplexifresidualpathsconvergeandre-dividebetweenthenetwork
layerandthenormalizationasinfigure11.6c. However,thespiritoftheargumentremainsunchanged.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

11.5 Common residual architectures 195
11.5 Common residual architectures
Residual connections are now a standard part of deep learning pipelines. This section
reviews some well-known architectures that incorporate them.
11.5.1 ResNet
Residual blocks were first used in convolutional networks for image classification. The
resultingnetworksareknownasresidualnetworks,orResNetsforshort. InResNets,each
residualblockcontainsabatchnormalizationoperation,aReLUactivationfunction,and
a convolutional layer. This is followed by the same sequence again before being added
Problem11.7
backtotheinput(figure11.7a). Trialanderrorhaveshownthatthisorderofoperations
works well for image classification.
For very deep networks, the number of parameters may become undesirably large.
Bottleneckresidualblocksmakemoreeﬀicientuseofparametersusingthreeconvolutions.
The first has a 1×1 kernel and reduces the number of channels. The second is a regular
3×3kernel, andthethirdisanother1×1kerneltoincreasethenumberofchannelsback
to the original amount (figure 11.7b). In this way, we can integrate information over a
3×3 pixel area using fewer parameters.
Problem11.8
The ResNet-200 model (figure 11.8) contains 200 layers and was used for image clas-
sification on the ImageNet database (figure 10.15). The architecture resembles AlexNet
and VGG but uses bottleneck residual blocks instead of vanilla convolutional layers. As
with AlexNet and VGG, these are periodically interspersed with decreases in spatial
resolution and simultaneous increases in the number of channels. The resolution is de-
creased between adjacent ResNet blocks using convolutions with stride two. Channels
aresimilarlyaddedbyeitherappendingzerostotherepresentationorapplyinganextra
1×1 convolution. At the start of the network is a 7×7 convolutional layer, followed by a
downsampling operation. At the end, a fully connected layer maps the block to a vector
of length 1000. This is passed through a softmax layer to generate class probabilities.
The ResNet-200 model achieved a remarkable 4.8% error rate for the correct class
beinginthetopfiveand20.1%foridentifyingthecorrectclasscorrectly. Thiscompared
favorably with AlexNet (16.4%, 38.1%) and VGG (6.8%, 23.7%) and was one of the
first networks to exceed human performance (5.1% for being in the top five guesses).
However, this model was conceived in 2016 and is far from state-of-the-art. At the time
of writing, the best-performing model on this task has a 9.0% error for identifying the
class correctly (see figure 10.21). This and all the other current top-performing models
for image classification are now based on transformers (see chapter 12).
11.5.2 DenseNet
Residual blocks receive the output from the previous layer, modify it by passing it
through some network layers, and add it back to the original input. An alternative is
to concatenate the modified and original signals. This increases the representation size
Draft: please send errata to udlbookmail@gmail.com.

196 11 Residual networks
Figure 11.7 ResNet blocks. a) A standard block in the ResNet architecture con-
tains a batch normalization operation, followed by an activation function, and
a 3×3 convolutional layer. Then, this sequence is repeated. b). A bottleneck
ResNet block still integrates information over a 3×3 region but uses fewer pa-
rameters. It contains three convolutions. The first 1×1 convolution reduces the
number of channels. The second 3×3 convolution is applied to the smaller rep-
resentation. A final 1×1 convolution increases the number of channels again so
that it can be added back to the input.
Figure 11.8 ResNet-200 model. A standard 7×7 convolutional layer with stride
two is applied, followed by a MaxPool operation. A series of bottleneck residual
blocks follow (number in brackets is channels after first 1×1 convolution), with
periodic downsampling and accompanying increases in the number of channels.
The network concludes with average pooling across all spatial positions and a
fully connected layer that maps to pre-softmax activations.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

11.5 Common residual architectures 197
Figure11.9DenseNet. Thisarchitectureusesresidualconnectionstoconcatenate
theoutputsofearlierlayerstolaterones. Here,thethree-channelinputimageis
processed to form a 32-channel representation. The input image is concatenated
to this to give a total of 35 channels. This combined representation is processed
tocreateanother32-channelrepresentation, andbothearlierrepresentationsare
concatenated to this to create a total of 67 channels and so on.
(in terms of channels for a convolutional network), but an optional subsequent linear
transformation can map back to the original size (a 1×1 convolution for a convolutional
network). This allows the model to add the representations together, take a weighted
sum, or combine them in a more complex way.
The DenseNet architecture uses concatenation so that the input to a layer comprises
the concatenated outputs from all previous layers (figure 11.9). These are processed to
create a new representation that is itself concatenated with the previous representation
and passed to the next layer. This concatenation means there is a direct contribution
from earlier layers to the output, so the loss surface behaves reasonably.
Inpractice,thiscanonlybesustainedforafewlayersbecausethenumberofchannels
(and hence the number of parameters required to process them) becomes increasingly
large. This problem can be alleviated by applying a 1×1 convolution to reduce the
number of channels before the next 3×3 convolution is applied. In a convolutional
network,theinputisperiodicallydownsampled. Concatenationacrossthedownsampling
makes no sense since the representations have different spatial sizes. Consequently, the
chain of concatenation is broken at this point, and a smaller representation starts a
new chain. In addition, another bottleneck 1×1 convolution can be applied when the
downsampling occurs to control the representation size further.
ThisnetworkperformscompetitivelywithResNetmodelsonimageclassification(see
figure 10.21); indeed, it can perform better for a comparable parameter count. This is
presumably because it can reuse processing from earlier layers more flexibly.
11.5.3 U-Nets and hourglass networks
Section10.5.3describedasemanticsegmentationnetworkthathadanencoder-decoderor
hourglass structure. The encoder repeatedly downsamples the image until the receptive
fields are large and information is integrated from across the image. Then the decoder
Draft: please send errata to udlbookmail@gmail.com.

198 11 Residual networks
Figure11.10U-NetforsegmentingHeLacells. TheU-Nethasanencoder-decoder
structure, in which the representation is downsampled (orange blocks) and then
re-upsampled (blue blocks). The encoder uses regular convolutions, and the de-
coder uses transposed convolutions. Residual connections append the last repre-
sentationateachscaleintheencodertothefirstrepresentationatthesamescale
inthedecoder(orangearrows). TheoriginalU-Netused“valid”convolutions,so
the size decreased slightly with each layer, even without downsampling. Hence,
the representations from the encoder were cropped (dashed squares) before ap-
pending to the decoder. Adapted from Ronneberger et al. (2015).
upsamples it back to the size of the original image. The final output is a probability
over possible object classes at each pixel. One drawback of this architecture is that
the low-resolution representation in the middle of the network must “remember” the
high-resolution details to make the final result accurate. This is unnecessary if residual
connectionstransfertherepresentationsfromtheencodertotheirpartnerinthedecoder.
The U-Net (figure 11.10) is an encoder-decoder architecture where the earlier repre-
sentations are concatenated to the later ones. The original implementation used “valid”
convolutions, so the spatial size decreases by two pixels each time a 3×3 convolutional
layer is applied. This means that the upsampled version is smaller than its counterpart
in the encoder, which must be cropped before concatenation. Subsequent implementa-
tions have used zero-padding, where this cropping is unnecessary. Note that the U-Net
is completely convolutional, so after training, it can be run on an image of any size.
Problem11.9
The U-Net was intended for segmenting medical images (figure 11.11) but has found
many other uses in computer graphics and vision. Hourglass networks are similar but
applyfurtherconvolutionallayersintheskipconnectionsandaddtheresultbacktothe
decoder rather than concatenating it. A series of these models form a stacked hourglass
network that alternates between considering the image at local and global levels. Such
networksareusedforposeestimation(figure11.12). Thesystemistrainedtopredictone
“heatmap” for each joint, and the estimated position is the maximum of each heatmap.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

11.6 Why do nets with residual connections perform so well? 199
Figure 11.11 Segmentation using U-Net in 3D. a) Three slices through a 3D
volume of mouse cortex taken by scanning electron microscope. b) A single U-
Net is used to classify voxels as being inside or outside neurites. Connected
regions are identified with different colors. c) For a better result, an ensemble of
five U-Nets is trained, and a voxel is only classified as belonging to the cell if all
five networks agree. Adapted from Falk et al. (2019).
11.6 Why do nets with residual connections perform so well?
Residual networks allow much deeper networks to be trained; it’s possible to extend the
ResNet architecture to 1000 layers and still train effectively. The improvement in image
classification performance was initially attributed to the additional network depth, but
two pieces of evidence contradict this viewpoint.
First,shallower,widerresidualnetworkssometimesoutperformdeeper,narrowerones
with a comparable parameter count. In other words, better performance can sometimes
beachievedwithanetworkwithfewerlayersbutmorechannelsperlayer. Second,there
is evidence that the gradients during training do not propagate effectively through very
long paths in the unraveled network (figure 11.4b). In effect, a very deep network may
act more like a combination of shallower networks.
The current view is that residual connections add some value of their own, as well
as allowing deeper networks to be trained. This perspective is supported by the fact
that the loss surfaces of residual networks around a minimum tend to be smoother and
morepredictablethanthoseforthesamenetworkwhentheskipconnectionsareremoved
(figure 11.13). This may make it easier to learn a good solution that generalizes well.
11.7 Summary
Increasingnetworkdepthindefinitelycausesbothtrainingandtestperformanceforimage
classification to decrease. This may be because the gradient of the loss with respect to
Draft: please send errata to udlbookmail@gmail.com.

200 11 Residual networks
Figure 11.12 Stacked hourglass networks for pose estimation. a) The network
inputisanimagecontainingaperson,andtheoutputisasetofheatmaps,with
oneheatmapforeachjoint. Thisisformulatedasaregressionproblemwherethe
targets are heatmap images with small, highlighted regions at the ground-truth
jointpositions. Thepeakoftheestimatedheatmapisusedtoestablisheachfinal
joint position. b) The architecture consists of initial convolutional and residual
layers followed by a series of hourglass blocks. c) Each hourglass block consists
ofanencoder-decodernetworksimilartotheU-Netexceptthattheconvolutions
usezero-padding,somefurtherprocessingisdoneintheresiduallinks,andthese
links add this processed representation rather than concatenate it. Each blue
cuboid is itself a bottleneck residual block (figure 11.7b). Adapted from Newell
et al. (2016).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 201
Figure 11.13 Visualizing neural network loss surfaces. Each plot shows the loss
surfaceintworandomdirectionsinparameterspacearoundtheminimumfound
by SGD for an image classification task on the CIFAR-10 dataset. These direc-
tionsarenormalizedtofacilitateside-by-sidecomparison. a)Residualnetwith56
layers. b) Results from the same network without skip connections. The surface
is smoother with the skip connections. This facilitates learning and makes the
final network performance more robust to minor errors in the parameters, so it
will likely generalize better. Adapted from Li et al. (2018b).
parametersearlyinthenetworkchangesquicklyandunpredictablyrelativetotheupdate
stepsize. Residualconnectionsaddtheprocessedrepresentationbacktotheirowninput.
Now each layer contributes directly to the output as well as indirectly, so propagating
gradients through many layers is not mandatory, and the loss surface is smoother.
Residualnetworksdon’tsufferfromvanishinggradientsbutintroduceanexponential
increaseinthevarianceoftheactivationsduringforwardpropagationandcorresponding
problems with exploding gradients. This is usually handled by adding batch normaliza-
tion, which compensates for the empirical mean and variance of the batch and then
shifts and rescales using learned parameters. If these parameters are initialized judi-
ciously, very deep networks can be trained. There is evidence that both residual links
and batch normalization make the loss surface smoother, which permits larger learning
rates. Moreover, the variability in the batch statistics adds a source of regularization.
Residual blocks have been incorporated into convolutional networks. They allow
deeper networks to be trained with commensurate increases in image classification per-
formance. Variations of residual networks include the DenseNet architecture, which
concatenatesoutputsofallprior layerstofeedintothecurrentlayer, andU-Nets, which
incorporate residual connections into encoder-decoder models.
Notes
Residual connections: ResidualconnectionswereintroducedbyHeetal.(2016a),whobuilt
anetworkwith152layers,whichwaseighttimeslargerthanVGG(figure10.17),andachieved
state-of-the-artperformanceontheImageNetclassificationtask. Eachresidualblockconsisted
Draft: please send errata to udlbookmail@gmail.com.

202 11 Residual networks
of a convolutional layer followed by batch normalization, a ReLU activation, a second convolu-
tional layer, and second batch normalization. A second ReLU function was applied after this
block was added back to the main representation. This architecture was termed ResNet v1.
He et al. (2016b) investigated different variations of residual architectures, in which either (i)
processing could also be applied along the skip connection or (ii) after the two branches had
recombined. They concluded neither was necessary, leading to the architecture in figure 11.7,
which is sometimes termed a pre-activation residual block and is the backbone of ResNet v2.
They trained a network with 200 layers that improved further on the ImageNet classification
task (see figure 11.8). Since this time, new methods for regularization, optimization, and data
augmentationhavebeendeveloped,andWightmanetal.(2021)exploitthesetopresentamore
modern training pipeline for the ResNet architecture.
Why residual connections help: Residual networks certainly allow deeper networks to be
trained. Presumably, this is related to reducing shattered gradients (Balduzzi et al., 2017) at
the start of training and the smoother loss surface near the minima as depicted in figure 11.13
(Li et al., 2018b). Residual connections alone (i.e., without batch normalization) increase the
trainabledepthofanetworkbyroughlyafactoroftwo(Sankararamanetal.,2020). Withbatch
normalization, very deep networks can be trained, but it is unclear that depth is critical for
performance. Zagoruyko&Komodakis(2016)showedthatwideresidualnetworkswithonly16
layers outperformed all residual networks of the time for image classification. Orhan & Pitkow
(2017) propose a different explanation for why residual connections improve learning in terms
of eliminating singularities (places on the loss surface where the Hessian is degenerate).
Related architectures: Residualconnectionsareaspecialcaseofhighway networks(Srivas-
tavaetal.,2015)whichalsosplitthecomputationintotwobranchesandadditivelyrecombine.
Highway networks use a gating function that weights the inputs to the two branches in a way
thatdependsonthedataitself,whereasresidualnetworkssendthedatadownbothbranchesin
astraightforwardmanner. Xieetal.(2017)introducedtheResNeXtarchitecture,whichplaces
a residual connection around multiple parallel convolutional branches.
Residual networks as ensembles: Veit et al. (2016) characterized residual networks as en-
semblesofshorternetworksanddepictedthe“unravelednetwork”interpretation(figure11.4b).
They provide evidence that this interpretation is valid by showing that deleting layers in a
trained network (and hence a subset of paths) only has a modest effect on performance. Con-
versely, removing a layer in a purely sequential network like VGG is catastrophic. They also
lookedatthegradientmagnitudesalongpathsofdifferentlengthsandshowedthatthegradient
vanishesinlongerpaths. Inaresidualnetworkconsistingof54blocks,almostallofthegradient
updates during training were from paths of length 5 to 17 blocks long, even though these only
constitute 0.45% of the total paths. It seems that adding more blocks effectively adds more
parallel shorter paths rather than creating a network that is truly deeper.
Regularization for residual networks: L2regularizationoftheweightshasafundamentally
differenteffectinvanillanetworksandresidualnetworkswithoutBatchNorm. Intheformer,it
encourages the output of the layer to be a constant function determined by the biases. In the
latter, it encourages the residual block to compute the identity plus a constant determined by
the biases.
Several regularization methods have been developed that are targeted specifically at residual
architectures. ResDrop (Yamada et al., 2016), stochastic depth (Huang et al., 2016), and
RandomDrop (Yamada et al., 2019) all regularize residual networks by randomly dropping
residualblocksduringthetrainingprocess. Inthelattercase,thepropensityfordroppingablock
isdeterminedbyaBernoullivariable,whoseparameterislinearlydecreasedduringtraining. At
testtime,theresidualblocksareaddedbackinwiththeirexpectedprobability. Thesemethods
are effectively versions of dropout, in which all the hidden units in a block are simultaneously
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 203
droppedinconcert. Inthemultiplepathsviewofresidualnetworks(figure11.4b),theysimply
removesomeofthepathsateachtrainingstep. Wuetal.(2018b)developedBlockDrop,which
analyzesanexistingnetworkanddecideswhichresidualblockstouseatruntimewiththegoal
of improving the eﬀiciency of inference.
Other regularization methods have been developed for networks with multiple paths inside
the residual block. Shake-shake (Gastaldi, 2017a,b) randomly re-weights the paths during the
forward and backward passes. In the forward pass, this can be viewed as synthesizing random
data, and in the backward pass, as injecting another form of noise into the training method.
ShakeDrop (Yamada et al., 2019) draws a Bernoulli variable that decides whether each block
will be subject to Shake-Shake or behave like a standard residual unit on this training step.
Batchnormalization: BatchnormalizationwasintroducedbyIoffe&Szegedy(2015)outside
of the context of residual networks. They showed empirically that it allowed higher learning
rates,increasedconvergencespeed,andmadesigmoidactivationfunctionsmorepractical(since
the distribution of outputs is controlled, so examples are less likely to fall in the saturated
extremes of the sigmoid). Balduzzi et al. (2017) investigated the activation of hidden units in
laterlayersofdeepnetworkswithReLUfunctionsatinitialization. Theyshowedthatmanysuch
hiddenunitswerealwaysactiveoralwaysinactiveregardlessoftheinputbutthatBatchNorm
reduced this tendency.
Although batch normalization helps stabilize the forward propagation of signals through a
network,Yangetal.(2019)showedthatitcausesgradientexplosioninReLUnetpworkswithout
skip connections, with each layer increasing the magnitude of the gradients by π/(π−1) ≈
1.21. This argument is summarized by Luther (2020). Since a residual network can be seen
as a combination of paths of different lengths (figure 11.4), this effect must also be present in
residualnetworks. Presumably,however,thebenefitofremovingthe2K increasesinmagnitude
in the forward pass of a network with K layers outweighs the harm done by increasing the
gradients by 1.21K in the backward pass, so overall BatchNorm makes training more stable.
Variations of batch normalization: Several variants of BatchNorm have been proposed
(figure 11.14). BatchNorm normalizes each channel separately based on statistics gathered
across the batch. Ghost batch normalization or GhostNorm (Hoffer et al., 2017) uses only part
of the batch to compute the normalization statistics, which makes them noisier and increases
the amount of regularization when the batch size is very large (figure 11.14b).
Whenthebatchsizeisverysmallorthefluctuationswithinabatchareverylarge(asisoftenthe
caseinnaturallanguageprocessing),thestatisticsinBatchNormmaybecomeunreliable. Ioffe
(2017) proposed batch renormalization, which keeps a running average of the batch statistics
and modifies the normalization of any batch to ensure that it is more representative. Another
problemisthatbatchnormalizationisunsuitableforuseinrecurrentneuralnetworks(networks
forprocessing sequences, in whichthe previous outputis fedbackas anadditional input aswe
movethroughthesequence,seefigure12.19). Here,thestatisticsmustbestoredateachstepin
thesequence,andit’sunclearwhattodoifatestsequenceislongerthanthetrainingsequences.
A third problem is that batch normalization needs access to the whole batch. However, this
may not be easily available when training is distributed across several machines.
LayernormalizationorLayerNorm(Baetal.,2016)avoidsusingbatchstatisticsbynormalizing
eachdataexampleseparately,usingstatisticsgatheredacrossthechannelsandspatialposition
(figure 11.14c). However, there is still a separate learned scale γ and offset δ per channel.
Group normalization or GroupNorm (Wu & He, 2018) is similar to LayerNorm but divides the
channels into groups and computes the statistics for each group separately across the within-
groupchannelsandthespatialpositions(figure11.14d). Again,therearestillseparatescaleand
offset parameters per channel. Instance normalization or InstanceNorm (Ulyanov et al., 2016)
takes this to the extreme where the number of groups is the same as the number of channels,
soeachchannelisnormalizedseparately(figure11.14e),usingstatisticsgatheredacrossspatial
Draft: please send errata to udlbookmail@gmail.com.

204 11 Residual networks
Figure 11.14 Normalization schemes. BatchNorm modifies each channel sepa-
ratelybutadjustseachbatchmemberinthesamewaybasedonstatisticsgathered
acrossthebatchandspatialposition. Ghostbatchnormalizationcomputesthese
statistics from only part of the batch to make them more variable. LayerNorm
computes statistics for each batch member separately, based on statistics gath-
eredacrossthechannelsandspatialposition. Itretainsaseparatelearnedscaling
factor for each channel. GroupNorm normalizes within each group of channels
and also retains a separate scale and offset parameter for each channel. Instan-
ceNormnormalizeswithineachchannelseparately,computingthestatisticsonly
across spatial position. Adapted from Wu & He (2018).
positionalone. Salimans&Kingma(2016)investigatednormalizingthenetworkweightsrather
thantheactivations,butthishasbeenlessempiricallysuccessful. Teyeetal.(2018)introduced
MonteCarlobatchnormalization,whichcanprovidemeaningfulestimatesofuncertaintyinthe
predictionsofneuralnetworks. Arecentcomparisonofthepropertiesofdifferentnormalization
schemes can be found in Lubana et al. (2021).
Why BatchNorm helps: BatchNormhelpscontroltheinitialgradientsinaresidualnetwork
(figure 11.6c). However, the mechanism by which BatchNorm improves performance is not
well understood. The stated goal of Ioffe & Szegedy (2015) was to reduce problems caused
by internal covariate shift, which is the change in the distribution of inputs to a layer caused
by updating preceding layers during the backpropagation update. However, Santurkar et al.
(2018) provided evidence against this view by artificially inducing covariate shift and showing
that networks with and without BatchNorm performed equally well.
Motivated by this, they searched for another explanation for why BatchNorm should improve
performance. They showed empirically for the VGG network that adding batch normalization
decreases the variation in both the loss and its gradient as we move in the gradient direction.
Inotherwords,thelosssurfaceisbothsmootherandchangesmoreslowly,whichiswhylarger
learning rates are possible. They also provide theoretical proofs for both these phenomena
and show that for any parameter initialization, the distance to the nearest optimum is less for
networks with batch normalization. Bjorck et al. (2018) also argue that BatchNorm improves
the properties of the loss landscape and allows larger learning rates.
OtherexplanationsofwhyBatchNormimprovesperformanceincludedecreasingtheimportance
of tuning the learning rate (Ioffe & Szegedy, 2015; Arora et al., 2018). Indeed Li & Arora
(2019)showthatusinganexponentiallyincreasinglearningratescheduleispossiblewithbatch
normalization. Ultimately, this is because batch normalization makes the network invariant to
the scales of the weight matrices (see Huszár, 2019, for an intuitive visualization).
Hoffer et al. (2017) identified that BatchNorm has a regularizing effect due to statistical fluc-
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 205
tuations from the random composition of the batch. They proposed using a ghost batch size,
in which the mean and standard deviation statistics are computed from a subset of the batch.
Largebatchescannowbeusedwithoutlosingtheregularizingeffectoftheextranoiseinsmaller
batch sizes. Luo et al. (2018) investigate the regularization effects of batch normalization.
Alternativestobatchnormalization: AlthoughBatchNormiswidelyused,itisnotstrictly
necessarytotraindeepresidualnets;thereareotheprwaysofmakingthelosssurfacetractable.
Balduzzi et al. (2017) proposed the rescaling by 1/2 in figure 11.6b; they argued that it
prevents gradient explosion but does not resolve the problem of shattered gradients.
Otherworkhasinvestigatedrescalingthefunction’soutputintheresidualblockbeforeadding
it back to the input. For example, De & Smith (2020) introduce SkipInit, in which a learnable
scalar multiplier is placepd at the end of each residual branch. This helps if this multiplier is
initialized to less than 1/K, where K is the number of residual blocks. In practice, they
suggest initializing this to zero. Similarly, Hayou et al. (2021) introduce Stable ResNet, which
rescalestheoutputofthefunctioninthekthresidualblock(beforeadditiontothemainbranch)
byaconstantλ . Theyprovethatinthelimitofinfinitewidth,theexpectedgradientnormof
k
the weights in the first layer is lower bopunded by the sum of squares of the scalings λ
k
. They
investigate setting these to a constant 1/K, where K is the number of residual blocks and
show that it is possible to train networks with up to 1000 blocks.
Zhangetal.(2019a)introduceFixUp,inwhicheverylayerisinitializedusingHenormalization,
but the last linear/convolutional layer of every residual block is set to zero. Now the initial
forwardpassisstable(sinceeachresidualblockcontributesnothing),andthegradientsdonot
explodeinthebackwardpass(forthesamereason). Theyalsorescalethebranchessothatthe
magnitudeofthetotalexpectedchangeintheparametersisconstantregardlessofthenumber
of residual blocks. These methods allow training of deep residual networks but don’t usually
achieve the same test performance as when using BatchNorm. This is probably because they
donotbenefitfromtheregularizationinducedbythenoisybatchstatistics. De&Smith(2020)
modify their method to induce regularization via dropout, which helps close this gap.
DenseNet and U-Net: DenseNet was first introduced by Huang et al. (2017b), U-Net was
developedbyRonnebergeretal.(2015),andstackedhourglassnetworksbyNewelletal.(2016).
Of these architectures, U-Net has been the most extensively adapted. Çiçek et al. (2016) in-
troduced 3D U-Net, and Milletari et al. (2016) introduced V-Net, both of which extend U-Net
to process 3D data. Zhou et al. (2018) combine the ideas of DenseNet and U-Net in an archi-
tecture that downsamples and re-upsamples the image but also repeatedly uses intermediate
representations. U-Netsarecommonlyusedinmedicalimagesegmentation(seeSiddiqueetal.,
2021,forareview). However,theyhavebeenappliedtootherareas,includingdepthestimation
(Garg et al., 2016), semantic segmentation (Iglovikov & Shvets, 2018), inpainting (Zeng et al.,
2019), pansharpening (Yao et al., 2018), and image-to-image translation (Isola et al., 2017).
U-Nets are also a key component in diffusion models (chapter 18).
Problems
Problem 11.1 Derive equation 11.5 from the network definition in equation 11.4.
Problem 11.2 Unraveling the four-block network in figure 11.4a produces one path of length
zero,fourpathsoflengthone,sixpathsoflengthtwo,fourpathsoflengththree,andonepath
of length four. How many paths of each length would there be with (i) three residual blocks
and (ii) five residual blocks? Deduce the rule for K residual blocks.
Problem 11.3 Showthatthederivativeofthenetworkinequation11.5withrespecttothefirst
layer f [x] is given by equation 11.6.
1
Draft: please send errata to udlbookmail@gmail.com.

206 11 Residual networks
Figure 11.15 Computational graph for batch normalization (see problem 11.5).
Problem11.4∗ Explainwhythevaluesinthetwobranchesoftheresidualblocksinfigure11.6a
are uncorrelated. Show that the variance of the sum of uncorrelated variables is the sum of
their individual variances.
Problem 11.5∗ Theforwardpassforbatchnormalizationgivenabatchofscalarvalues{z }I
i i=1
consists of the following operations (figure 11.15):
p
f 1 =E[z i ] f 5 = f 4 +ϵ
f 2i =z i −f 1 f 6 =1/f 5
(11.10)
f =f2 f =f ×f
3i 2i 7i 2i 6
f =E[f ] z ′ =f ×γ+δ,
4 3i i 7i
P
where E[z ] = 1 z . Write Python code to implement the forward pass. Now derive the
i I i i
algorithmforthebackwardpass. Workbackwardthroughthecomputationalgraphcomputing
the derivatives to generate a set of operations that computes ∂z′/∂z for every element in the
i i
batch. Write Python code to implement the backward pass.
Problem 11.6 Consider a fully connected neural network with one input, one output, and ten
hidden layers, each of which contains twenty hidden units. How many parameters does this
network have? How many parameters will it have if we place a batch normalization operation
between each linear transformation and ReLU?
Problem 11.7∗ Consider applying an L2 regularization penalty to the weights in the convolu-
tional layers in figure 11.7a, but not to the scaling parameters of the subsequent BatchNorm
layers. What do you expect will happen as training proceeds?
Problem11.8Consideraconvolutionalresidualblockthatcontainsabatchnormalizationoper-
ation,followedbyaReLUactivationfunction,andthena3×3convolutionallayer. Iftheinput
andoutputbothhave512channels,howmanyparametersareneededtodefinethisblock? Now
considerabottleneckresidualblockthatcontainsthreebatchnormalization/ReLU/convolution
sequences. Thefirstusesa1×1convolutiontoreducethenumberofchannelsfrom512to128.
The second uses a 3×3 convolution with the same number of input and output channels. The
third uses a 1×1 convolution to increase the number of channels from 128 to 512 (see fig-
ure 11.7b). How many parameters are needed to define this block?
Problem11.9TheU-Netiscompletelyconvolutionalandcanberunwithanysizedimageafter
training. Why do we not train with a collection of arbitrarily-sized images?
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.