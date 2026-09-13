# Chapter 10

*Pages: 175-180*

---

Chapter 10
Convolutional networks
Chapters2–9introducedthesupervisedlearningpipelinefordeepneuralnetworks. How-
ever, these chapters only considered fully connected networks with a single path from
input to output. Chapters 10–13 introduce more specialized network components with
sparser connections, shared weights, and parallel processing paths. This chapter de-
scribes convolutional layers, which are mainly used for processing image data.
Images have three properties that suggest the need for specialized model architec-
ture. First, they are high-dimensional. A typical image for a classification task contains
224×224 RGB values (i.e., 150,528 input dimensions). Hidden layers in fully connected
networks are generally larger than the input size, so even for a shallow network, the
number of weights would exceed 150,5282, or 22 billion. This poses obvious practical
problems in terms of the required training data, memory, and computation.
Second, nearby image pixels are statistically related. However, fully connected net-
workshavenonotionof“nearby”andtreattherelationshipbetweeneveryinputequally.
If the pixels of the training and test images were randomly permuted in the same way,
the network could still be trained with no practical difference. Third, the interpretation
of an image is stable under geometric transformations. An image of a tree is still an
image of a tree if we shift it leftwards by a few pixels. However, this shift changes every
input to the network. Hence, a fully connected model must learn the patterns of pixels
that signify a tree separately at every position, which is clearly ineﬀicient.
Convolutionallayersprocesseachlocalimageregionindependently,usingparameters
shared across the whole image. They use fewer parameters than fully connected layers,
exploit the spatial relationships between nearby pixels, and don’t have to re-learn the
interpretation of the pixels at every position. A network predominantly consisting of
convolutional layers is known as a convolutional neural network or CNN.
10.1 Invariance and equivariance
We argued above that some properties of images (e.g., tree texture) are stable under
transformations. In this section, we make this idea more mathematically precise. A
Draft: please send errata to udlbookmail@gmail.com.

162 10 Convolutional networks
Figure 10.1 Invariance and equivariance for translation. a–b) In image classi-
fication, the goal is to categorize both images as “mountain” regardless of the
horizontal shift that has occurred. In other words, we require the network pre-
diction to be invariant to translation. c,e) The goal of semantic segmentation is
to associate a label with each pixel. d,f) When the input image is translated, we
want the output (colored overlay) to translate in the same way. In other words,
we require the output to be equivariant with respect to translation. Panels c–f)
adapted from Bousselham et al. (2021).
function f[x] of an image x is invariant to a transformation t[x] if:
(cid:2) (cid:3)
f t[x] =f[x]. (10.1)
In other words, the output of the function f[x] is the same regardless of the transfor-
mation t[x]. Networks for image classification should be invariant to geometric trans-
formations of the image (figure 10.1a–b). The network f[x] should identify an image as
containing the same object, even if it has been translated, rotated, flipped, or warped.
A function f[x] of an image x is equivariant or covariant to a transformation t[x] if:
(cid:2) (cid:3) (cid:2) (cid:3)
f t[x] =t f[x] . (10.2)
In other words, f[x] is equivariant to the transformation t[x] if its output changes in
the same way under the transformation as the input. Networks for per-pixel image
segmentation should be equivariant to transformations (figure 10.1c–f); if the image is
translated, rotated, or flipped, the network f[x] should return a segmentation that has
been transformed in the same way.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

10.2 Convolutional networks for 1D inputs 163
Figure 10.2 1D convolution with kernel size three. Each output z is a weighted
i
sum of the nearest three inputs x i−1 , x i , and x i+1 , where the weights are ω =
[ω ,ω ,ω ]. a)Outputz iscomputedasz =ω x +ω x +ω x . b)Outputz
1 2 3 2 2 1 1 2 2 3 3 3
is computed as z = ω x +ω x +ω x . c) At position z , the kernel extends
3 1 2 2 3 3 4 1
beyond the first input x . This can be handled by zero-padding, in which we
1
assume values outside the input are zero. The final output is treated similarly.
d)Alternatively,wecouldonlycomputeoutputswherethekernelfitswithinthe
inputrange(“valid”convolution);now,theoutputwillbesmallerthantheinput.
10.2 Convolutional networks for 1D inputs
Convolutionalnetworksconsistofaseriesofconvolutionallayers,eachofwhichisequiv-
arianttotranslation. Theyalsotypicallyincludepoolingmechanismsthatinducepartial
invariance to translation. For clarity of exposition, we first consider convolutional net-
works for 1D data, which are easier to visualize. In section 10.3, we progress to 2D
convolution, which can be applied to image data.
10.2.1 1D convolution operation
Convolutional layers are network layers based on the convolution operation. In 1D, a
convolution transforms an input vector x into an output vector z so that each output z
i
is a weighted sum of nearby inputs. The same weights are used at every position and
are collectively called the convolution kernel or filter. The size of the region over which
inputs are combined is termed the kernel size. For a kernel size of three, we have:
z i =ω 1 x i−1 +ω 2 x i +ω 3 x i+1 , (10.3)
where ω = [ω ,ω ,ω ]T is the kernel (figure 10.2).1 Notice that the convolution oper-
1 2 3
Problem10.1
ation is equivariant with respect to translation. If we translate the input x, then the
corresponding output z is translated in the same way.
1Strictlyspeaking, thisisacross-correlationandnotaconvolution, inwhichtheweightswouldbe
flippedrelativetotheinput(sowewouldswitchxi−1withxi+1). Regardless,this(incorrect)definition
istheusualconventioninmachinelearning.
Draft: please send errata to udlbookmail@gmail.com.

164 10 Convolutional networks
Figure 10.3Stride,kernelsize,anddilation. a)Withastrideoftwo,weevaluate
the kernel at every other position, so the first output z is computed from a
1
weighted sum centered at x , and b) the second output z is computed from a
1 2
weighted sum centered at x and so on. c) The kernel size can also be changed.
3
With a kernel size of five, we take a weighted sum of the nearest five inputs. d)
In dilated or atrous convolution (from the French “à trous” – with holes), we
intersperse zeros in the weight vector to allow us to combine information over a
large area using fewer weights.
10.2.2 Padding
Equation 10.3 shows that each output is computed by taking a weighted sum of the
previous, current, and subsequent positions in the input. This begs the question of how
to deal with the first output (where there is no previous input) and the final output
(where there is no subsequent input).
There are two common approaches. The first is to pad the edges of the inputs with
new values and proceed as usual. Zero-padding assumes the input is zero outside its
valid range (figure 10.2c). Other possibilities include treating the input as circular or
reflecting it at the boundaries. The second approach is to discard the output positions
wherethe kernelexceeds the range of input positions. These valid convolutions havethe
advantage of introducing no extra information at the edges of the input. However, they
have the disadvantage that the representation decreases in size.
10.2.3 Stride, kernel size, and dilation
In the example above, each output was a sum of the nearest three inputs. However,
this is just one of a larger family of convolution operations, the members of which are
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

10.2 Convolutional networks for 1D inputs 165
distinguishedbytheirstride,kernelsize,anddilationrate. Whenweevaluatetheoutput
at every position, we term this a stride of one. However, it is also possible to shift the
kernel by a stride greater than one. If we have a stride of two, we create roughly half
the number of outputs (figure 10.3a–b).
The kernel size can be increased to integrate over a larger area (figure 10.3c). How-
ever, it typically remains an odd number so that it can be centered around the current
position. Increasingthekernelsizehasthedisadvantageofrequiringmoreweights. This
leads to the idea of dilated or atrous convolutions, in which the kernel values are inter-
spersedwithzeros. Forexample, wecanturnakernelofsizefiveintoadilatedkernelof
size three by setting the second and fourth elements to zero. We still integrate informa-
Problems10.2–10.4
tion from a larger input region but only require three weights to do this (figure 10.3d).
The number of zeros we intersperse between the weights determines the dilation rate.
10.2.4 Convolutional layers
Aconvolutionallayercomputesitsoutputbyconvolvingtheinput, addingabiasβ, and
passing each result through an activation function a[•]. With kernel size three, stride
one, and dilation rate one, the ith hidden unit h would be computed as:
i
h i = a[ 2 β+ω 1 x i−1 +ω 2 x i3 +ω 3 x i+1 ]
X3
4 5
= a β+ ω j x i+j−2 , (10.4)
j=1
where the bias β and kernel weights ω ,ω ,ω are trainable parameters, and (with zero-
1 2 3
padding) we treat the input x as zero when it is out of the valid range. This is a special
case of a fully connected layer that computes the ith hidden unit as:
2 3
XD
4 5
h = a β + ω x . (10.5)
i i ij j
j=1
IfthereareD inputsx• andD hiddenunitsh•,thisfullyconnectedlayerwouldhaveD2
weights ω•• and D biases β•. The convolutional layer only uses three weights and one
bias. A fully connected layer can reproduce this exactly if most weights are set to zero
Problem10.5
and others are constrained to be identical (figure 10.4).
10.2.5 Channels
If we only apply a single convolution, information will likely be lost; we are averaging
nearby inputs, and the ReLU activation function clips results that are less than zero.
Hence,itisusualtocomputeseveralconvolutionsinparallel. Eachconvolutionproduces
a new set of hidden variables, termed a feature map or channel.
Draft: please send errata to udlbookmail@gmail.com.

166 10 Convolutional networks
Figure 10.4 Fully connected vs. convolutional layers. a) A fully connected layer
has a weight connecting each input x to each hidden unit h (colored arrows)
and a bias for each hidden unit (not shown). b) Hence, the associated weight
matrixΩcontains36weightsrelatingthesixinputstothesixhiddenunits. c)A
convolutionallayerwithkernelsizethreecomputeseachhiddenunitasthesame
weighted sum of the three neighboring inputs (arrows) plus a bias (not shown).
d)The weightmatrix isa specialcase ofthe fullyconnected matrixwhere many
weightsarezeroandothersarerepeated(samecolorsindicatesamevalue,white
indicates zero weight). e) A convolutional layer with kernel size three and stride
two computes a weighted sum at every other position. f) This is also a special
case of a fully connected network with a different sparse weight structure.
Figure10.5Channels. Typically,multipleconvolutionsareappliedtotheinputx
and stored in channels. a) A convolution is applied to create hidden units h
1
toh ,whichformthefirstchannel. b)Asecondconvolutionoperationisapplied
6
to create hidden units h to h , which form the second channel. The channels
7 12
arestoredina2DarrayH thatcontainsallthehiddenunitsinthefirsthidden
1
layer. c) If we add a further convolutional layer, there are now two channels at
each input position. Here, the 1D convolution defines a weighted sum over both
input channels at the three closest positions to create each new output channel.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.