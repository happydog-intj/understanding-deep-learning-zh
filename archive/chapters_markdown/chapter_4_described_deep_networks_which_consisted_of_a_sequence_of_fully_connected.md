# Chapter 4: described deep networks, which consisted of a sequence of fully connected

*Pages: 181-199*

---

10.2 Convolutional networks for 1D inputs 167
Figure 10.5a–b illustrates this with two convolution kernels of size three and with
zero-padding. Thefirstkernelcomputesaweightedsumofthenearestthreepixels,adds
abias,andpassestheresultsthroughtheactivationfunctiontoproducehiddenunitsh
1
toh . Thesecomprisethefirstchannel. Thesecondkernelcomputesadifferentweighted
6
sum of the nearest three pixels, adds a different bias, and passes the results through the
activationfunctiontocreatehiddenunitsh toh . Thesecomprisethesecondchannel.
7 12
Ingeneral,theinputandthehiddenlayersallhavemultiplechannels(figure10.5c). If
theincominglayerhasC channelsandweselectakernelsizeK perchannel,thehidden
i
Problems10.6–10.8
unitsineachoutputchannelarecomputedasaweightedsumoverallC channelsandK
i
kernel entries using a weight matrix Ω ∈ RCi ×K and one bias. Hence, if there are C
o Notebook10.1
channels in the next layer, then we need Ω∈RCi ×Co ×K weights and β ∈RCo biases. 1Dconvolution
10.2.6 Convolutional networks and receptive fields
Chapter 4 described deep networks, which consisted of a sequence of fully connected
layers. Similarly, convolutional networks comprise a sequence of convolutional layers.
Thereceptive fieldofahiddenunitinthenetworkistheregionoftheoriginalinputthat
feedsintoit. Consideraconvolutionalnetworkwhereeachconvolutionallayerhaskernel
size three. The hidden units in the first layer take a weighted sum of the three closest
inputs,sohavereceptivefieldsofsizethree. Theunitsinthesecondlayertakeaweighted
sum of the three closest positions in the first layer, which are themselves weighted sums
of three inputs. Hence, the hidden units in the second layer have a receptive field of size
five. Inthisway,thereceptivefieldofunitsinsuccessivelayersincreases,andinformation
from across the input is gradually integrated (figure 10.6). Problems10.9–10.11
10.2.7 Example: MNIST-1D
We now apply a convolutional network to the MNIST-1D data (see figure 8.1). The
input x is a 40D vector, and the output f is a 10D vector that is passed through a
softmax layer to produce class probabilities. We use a network with three hidden layers
(figure 10.7). The fifteen channels of the first hidden layer H are each computed using
1
a kernel size of three and a stride of two with “valid” padding, giving nineteen spatial
positions. The second hidden layer H is also computed using a kernel size of three, a
2
strideoftwo,and“valid”padding. Thethirdhiddenlayeriscomputedsimilarly. Atthis
stage, the representation has four spatial positions and fifteen channels. These values
are reshaped into a vector of size sixty, which is mapped by a fully connected layer to
the ten output activations.
Thisnetworkwastrainedfor100,000stepsusingSGDwithoutmomentum,alearning
rate of 0.01, and a batch size of 100 on a dataset of 4,000 examples. We compare this to
Problem10.12
a fully connected network with the same number of layers and hidden units (i.e., three
hidden layers with 285, 135, and 60 hidden units, respectively). The convolutional net-
work has 2,050 parameters, and the fully connected network has 59,065 parameters. By
the logic of figure 10.4, the convolutional network is a special case of the fully connected
Draft: please send errata to udlbookmail@gmail.com.

168 10 Convolutional networks
Figure 10.6 Receptive fields for network with kernel width of three. a) An input
with eleven dimensions feeds into a hidden layer with three channels and convo-
lution kernel of size three. The pre-activations of the three highlighted hidden
unitsinthefirsthiddenlayerH aredifferentweightedsumsofthenearestthree
1
inputs, so the receptive field in H has size three. b) The pre-activations of the
1
four highlighted hidden units in layer H each take a weighted sum of the three
2
channels in layer H at each of the three nearest positions. Each hidden unit in
1
layer H weights the nearest three input positions. Hence, hidden units in H
1 2
have a receptive field size of five. c) The hidden units in the third layer (kernel
size three, stride two) increases the receptive field size to seven. d) By the time
we add a fourth layer, the receptive field of the hidden units at position three
have a receptive field that covers the entire input.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

10.2 Convolutional networks for 1D inputs 169
Figure10.7ConvolutionalnetworkforclassifyingMNIST-1Ddata(seefigure8.1).
The MNIST-1D input has dimension D =40. The first convolutional layer has
i
fifteenchannels,kernelsizethree,stridetwo,andonlyretains“valid”positionsto
make a hidden layer with nineteen positions and fifteen channels. The following
two convolutional layers have the same settings, gradually reducing the repre-
sentation size at each subsequent hidden layer. Finally, a fully connected layer
takesallsixtyhiddenunitsfromthethirdhiddenlayer. Itoutputstenactivations
that are subsequently passed through a softmax layer to produce the ten class
probabilities.
Figure 10.8 MNIST-1D results. a) The convolutional network from figure 10.7
eventually fits the training data perfectly and has ∼17% test error. b) A fully
connected network with the same number of hidden layers and the number of
hiddenunitsineachlearnsthetrainingdatafasterbutfailstogeneralizewellwith
∼40% test error. The latter model can reproduce the convolutional model but
failstodoso. Theconvolutionalstructurerestrictsthepossiblemappingstothose
thatprocesseverypositionsimilarly,andthisrestrictionimprovesperformance.
Draft: please send errata to udlbookmail@gmail.com.

170 10 Convolutional networks
one. The latter has enough flexibility to replicate the former exactly. Figure 10.8 shows
Notebook10.2
bothmodelsfitthetrainingdataperfectly. However, thetesterrorfortheconvolutional
Convolution
forMNIST-1D network is much less than for the fully connected network.
This discrepancy is probably not due to the difference in the number of parameters;
we know overparameterization usually improves performance (section 8.4.1). The likely
explanation is that the convolutional architecture has a superior inductive bias (i.e.,
interpolates between the training data better) because we have embodied some prior
knowledge in the architecture; we have forced the network to process each position in
the input in the same way. We know that the data were created by starting with a
template that is (among other operations) randomly translated, so this is sensible.
Thefullyconnectednetworkhastolearnwhateachdigittemplatelookslikeatevery
position. In contrast, the convolutional network shares information across positions and
hence learns to identify each category more accurately. Another way of thinking about
thisisthatwhenwetraintheconvolutionalnetwork,wesearchthroughasmallerfamily
of input/output mappings, all of which are plausible. Alternatively, the convolutional
structure can be considered a regularizer that applies an infinite penalty to most of the
solutions that a fully connected network can describe.
10.3 Convolutional networks for 2D inputs
The previous section described convolutional networks for processing 1D data. Such
networkscanbeappliedtofinancialtimeseries,audio,andtext. However,convolutional
networks are more usually applied to 2D image data. The convolutional kernel is now
a 2D object. A 3×3 kernel Ω ∈ R3×3 applied to a 2D input comprising of elements x
ij
computes a single layer of hidden units h as:
ij
" #
X3 X3
h ij = a β+ ω mn x i+m−2,j+n−2 , (10.6)
m=1n=1
where ω are the entries of the convolutional kernel. This is simply a weighted sum
mn
overasquare3×3inputregion. Thekernelistranslatedbothhorizontallyandvertically
Problem10.13
across the 2D input (figure 10.9) to create an output at each position.
OftentheinputisanRGBimage,whichistreatedasa2Dsignalwiththreechannels
(figure 10.10). Here, a 3×3 kernel would have 3×3×3 weights and be applied to the
Notebook10.3 threeinputchannelsateachofthe3×3positionstocreatea2Doutputthatisthesame
2Dconvolution
height and width as the input image (assuming zero-padding). To generate multiple
Problem10.14 output channels, we repeat this process with different kernel weights and append the
resultstoforma3Dtensor. IfthekernelissizeK×K, andthereareC inputchannels,
i
AppendixB.3 eachoutputchannelisaweightedsumofC ×K×K quantitiesplusonebias. Itfollows
i
Tensors that to compute C output channels, we need C ×C ×K×K weights and C biases.
o i o o
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

10.4 Downsampling and upsampling 171
Figure10.92Dconvolutionallayer. Eachoutputh computesaweightedsumof
ij
the 3×3 nearest inputs, adds a bias, and passes the result through an activation
function. a) Here, the output h (shaded output) is a weighted sum of the nine
23
positions from x to x (shaded inputs). b) Different outputs are computed
12 34
by translating the kernel across the image grid in two dimensions. c–d) With
zero-padding, positions beyond the image’s edge are considered to be zero.
10.4 Downsampling and upsampling
The network in figure 10.7 increased receptive field size by scaling down the representa-
tion at each layer using stride two convolutions. We now consider methods for scaling
down or downsampling 2D input representations. We also describe methods for scaling
them back up (upsampling), which is useful when the output is also an image. Finally,
we consider methods to change the number of channels between layers. This is helpful
when recombining representations from two branches of a network (chapter 11).
10.4.1 Downsampling
Therearethreemainapproachestoscalingdowna2Drepresentation. Here,weconsider
the most common case of scaling down both dimensions by a factor of two. First, we
Draft: please send errata to udlbookmail@gmail.com.

172 10 Convolutional networks
Figure 10.10 2D convolution applied to an image. The image is treated as a 2D
inputwiththreechannelscorrespondingtothered,green,andbluecomponents.
With a 3×3 kernel, each pre-activation in the first hidden layer is computed by
pointwisemultiplyingthe3×3×3kernelweightswiththe3×3RGBimagepatch
centered at the same position, summing, and adding the bias. To calculate all
the pre-activations in the hidden layer, we “slide” the kernel over the image in
bothhorizontalandverticaldirections. Theoutputisa2Dlayerofhiddenunits.
To create multiple output channels, we would repeat this process with multiple
kernels, resulting in a 3D tensor of hidden units at hidden layer H .
1
can sample every other position. When we use a stride of two, we effectively apply this
Problem10.15
method simultaneously with the convolution operation (figure 10.11a).
Second, max pooling retains the maximum of the 2×2 input values (figure 10.11b).
This induces some invariance to translation; if the input is shifted by one pixel, many
of these maximum values remain the same. Finally, mean pooling or average pooling
averages the inputs. For all approaches, we apply downsampling separately to each
channel, so the output has half the width and height but the same number of channels.
10.4.2 Upsampling
The simplest way to scale up a network layer to double the resolution is to duplicate
all the channels at each spatial position four times (figure 10.12a). A second method
is max unpooling; this is used where we have previously used a max pooling operation
for downsampling, and we distribute the values to the positions they originated from
(figure 10.12b). A third approach uses bilinear interpolation to fill in the missing values
between the points where we have samples. (figure 10.12c).
A fourth approach is roughly analogous to downsampling using a stride of two. In
Notebook10.4
that method, there were half as many outputs as inputs, and for kernel size three, each
Downsampling
&upsampling output was a weighted sum of the three closest inputs (figure 10.13a). In transposed
convolution, this picture is reversed (figure 10.13c). There are twice as many outputs
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

10.4 Downsampling and upsampling 173
Figure 10.11 Methods for scaling down representation size (downsampling). a)
Sub-sampling. Theoriginal4×4representation(left)isreducedtosize2×2(right)
byretainingeveryotherinput. Colorsontheleftindicatewhichinputscontribute
totheoutputsontheright. Thisiseffectivelywhathappenswithakernelofstride
two, except that the intermediate values are never computed. b) Max pooling.
Each output comprises the maximum value of the corresponding 2×2 block. c)
Mean pooling. Each output is the mean of the values in the 2×2 block.
Figure 10.12 Methods for scaling up representation size (upsampling). a) The
simplest way to double the size of a 2D layer is to duplicate each input four
times. b) In networks where we have previously used a max pooling operation
(figure10.11b),wecanredistributethevaluestothesamepositionstheyoriginally
camefrom(i.e.,wherethemaximawere). Thisisknownasmaxunpooling. c)A
third option is bilinear interpolation between the input values.
Figure 10.13 Transposed convolution in 1D. a) Downsampling with kernel size
three, stride two, and zero-padding. Each output is a weighted sum of three
inputs (arrows indicate weights). b) This can be expressed by a weight matrix
(same color indicates shared weight). c) In transposed convolution, each input
contributesthreevaluestotheoutputlayer,whichhastwiceasmanyoutputsas
inputs. d) The associated weight matrix is the transpose of that in panel (b).
Draft: please send errata to udlbookmail@gmail.com.

174 10 Convolutional networks
as inputs, and each input contributes to three of the outputs. When we consider the
associated weight matrix of this upsampling mechanism (figure 10.13d), we see that it is
the transpose of the matrix for the downsampling mechanism (figure 10.13b).
10.4.3 Changing the number of channels
Sometimes we want to change the number of channels between one hidden layer and the
nextwithoutfurtherspatialpooling. Thisisusuallysowecancombinetherepresentation
with another parallel computation (see chapter 11). To accomplish this, we apply a
convolution with kernel size one. Each element of the output layer is computed by
taking a weighted sum of all the channels at the same position (figure 10.14). We can
repeatthismultipletimeswithdifferentweightstogenerateasmanyoutputchannelsas
we need. The associated convolution weights have size 1×1×C ×C . Hence, this is
i o
knownas1×1convolution. Combinedwithabiasandactivationfunction,itisequivalent
to running the same fully connected network on the input channels at every position.
10.5 Applications
We conclude by describing three computer vision applications. We describe convolu-
tional networks for image classification where the goal is to assign the image to one of a
predetermined set of categories. Then we consider object detection, where the goal is to
identify multiple objects in an image and find the bounding box around each. Finally,
wedescribeanearlysystemforsemanticsegmentationwherethegoalistoassignalabel
to each pixel according to which object is present.
10.5.1 Image classification
Much of the pioneering work on deep learning in computer vision focused on image
classificationusingtheImageNetdataset(figure10.15). Thiscontains1,281,167training
images, 50,000validationimages, and100,000testimages, andeveryimageislabeledas
belonging to one of 1000 possible categories.
Most methods reshape the input images to a standard size; in a typical system,
the input x to the network is a 224×224 RGB image, and the output is a probability
distribution over the 1000 classes. The task is challenging; there are a large number
of classes, and they exhibit considerable variation (figure 10.15). In 2011, before deep
networkswereapplied,thestate-of-the-artmethodclassifiedthetestimageswith∼25%
errors for the correct class being in the top five suggestions. Five years later, the best
deep learning models eclipsed human performance.
In 2012, AlexNet was the first convolutional network to perform well on this task.
It consists of eight hidden layers with ReLU activation functions, of which the first
five are convolutional and the rest fully connected (figure 10.16). The network starts by
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

10.5 Applications 175
Figure10.141×1convolution. Tochangethenumberofchannelswithoutspatial
pooling, we apply a 1×1 kernel. Each output channel is computed by taking
a weighted sum of all of the channels at the same position, adding a bias, and
passing through an activation function. Multiple output channels are created by
repeating this operation with different weights and biases.
Figure 10.15ExampleImageNetclassificationimages. Themodelaimstoassign
an input image to one of 1000 classes. This task is challenging because the
images vary widely along different attributes (columns). These include rigidity
(monkey<canoe), number of instances in image (lizard<strawberry), clutter
(compass<steeldrum),size(candle<spiderweb),texture(screwdriver<leopard),
distinctiveness of color (mug<red wine), and distinctiveness of shape (headland
<bell). Adapted from Russakovsky et al. (2015).
Draft: please send errata to udlbookmail@gmail.com.

176 10 Convolutional networks
Figure 10.16AlexNet(Krizhevskyetal.,
2012). The network maps a 224×224
color image to a 1000-dimensional vec-
tor representing class probabilities. The
network first convolves with 11×11 ker-
nels and stride 4 to create 96 channels.
It decreases the resolution again using a
max pool operation and applies a 5×5
convolutional layer. Another max pool-
ing layer follows, and three 3×3 convo-
lutional layers are applied. After a fi-
nal max pooling operation, the result
is vectorized and passed through three
fully connected (FC) layers and finally
the softmax layer.
downsamplingtheinputusingan11×11kernelwithastrideoffourtocreate96channels.
It then downsamples again using a max pooling layer before applying a 5×5 kernel to
create 256 channels. There are three more convolutional layers with kernel size 3×3,
Problems10.16–10.17 eventually resulting in a 13×13 representation with 256 channels. A final max-pooling
layer yields a 6×6 representation with 256 channels which is resized into a vector of
length9,216andpassedthroughthreefullyconnectedlayerscontaining4096, 4096, and
1000hiddenunits,respectively. Thelastlayerispassedthroughthesoftmaxfunctionto
output a probability distribution over the 1000 classes. The complete network contains
∼60 million parameters, most of which are in the fully connected layers.
The dataset size was augmented by a factor of 2048 using (i) spatial transformations
Notebook10.5
and (ii) modifications of the input intensities. At test time, five different cropped and
Convolution
forMNIST mirrored versions of the image were run through the network, and their predictions
averaged. The system was learned using SGD with a momentum coeﬀicient of 0.9 and a
batch size of 128. Dropout was applied in the fully connected layers, and an L2 (weight
decay) regularizer was used. This system achieved a 16.4% top-5 error rate and a 38.1%
top-1errorrate. Atthetime,thiswasanenormousleapforwardinperformanceatatask
considered far beyond the capabilities of contemporary methods. This result revealed
the potential of deep learning and kick-started the modern era of AI research.
The VGG network was also targeted at classification in the ImageNet task and
achieved a considerably better performance of 6.8% top-5 error rate and a 23.7% top-1
error rate. This network is similarly composed of a series of interspersed convolutional
and max pooling layers, where the spatial size of the representation gradually decreases,
but the number of channels increase. These are followed by three fully connected layers
(figure 10.17). The VGG network was also trained using data augmentation, weight
decay, and dropout.
Althoughtherewerevariousminordifferencesinthetrainingregime,themostimpor-
tant change between AlexNet and VGG was the depth of the network. The latter used
Problem10.18
19 hidden layers and 144 million parameters. The networks in figures 10.16 and 10.17
are depicted at the same scale for comparison. There was a general trend for several
years for performance on this task to improve as the depth of the networks increased,
and this is evidence that depth is important in neural networks.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

10.5 Applications 177
Figure 10.17 VGGnetwork(Simonyan& Zisserman,2014)depicted atthe same
scale as AlexNet (see figure 10.16). This network consists of a series of convolu-
tional layers and max pooling operations, in which the spatial scale of the rep-
resentation gradually decreases, but the number of channels gradually increases.
The hidden layer after the last convolutional operation is resized to a 1D vector
and three fully connected layers follow. The network outputs 1000 activations
corresponding to the class labels that are passed through a softmax function to
create class probabilities.
10.5.2 Object detection
In object detection, thegoal is to identifyand localize multipleobjects within the image.
An early method based on convolutional networks was You Only Look Once, or YOLO
for short. The input to the YOLO network is a 448×448 RGB image. This is passed
through 24 convolutional layers that gradually decrease the representation size using
max pooling operations while concurrently increasing the number of channels, similarly
totheVGGnetwork. Thefinalconvolutionallayerisofsize7×7andhas1024channels.
This is reshaped to a vector, and a fully connected layer maps it to 4096 values. One
further fully connected layer maps this representation to the output.
The output values encode which class is present at each of a 7×7 grid of locations
(figure 10.18a–b). For each location, the output values also encode a fixed number of
bounding boxes. Five parameters define each box: the x- and y-positions of the center,
the height and width of the box, and the confidence of the prediction (figure 10.18c).
The confidence estimates the overlap between the predicted and ground truth bound-
ing boxes. The system is trained using momentum, weight decay, dropout, and data
augmentation. Transfer learning is employed; the network is initially trained on the
ImageNet classification task and is then fine-tuned for object detection.
After the network is run, a heuristic process is used to remove rectangles with low
confidenceandtosuppresspredictedboundingboxesthatcorrespondtothesameobject
so only the most confident one is retained.
Draft: please send errata to udlbookmail@gmail.com.

178 10 Convolutional networks
Figure10.18YOLOobjectdetection. a)Theinputimageisreshapedto448×448
anddividedintoaregular7×7grid. b)Thesystempredictsthemostlikelyclass
ateachgridcell. c)Italsopredictstwoboundingboxespercell,andaconfidence
value (represented by thickness of line). d) During inference, the most likely
boundingboxesareretained,andboxeswithlowerconfidencevaluesthatbelong
to the same object are suppressed. Adapted from Redmon et al. (2016).
10.5.3 Semantic segmentation
Thegoalofsemanticsegmentationistoassignalabeltoeachpixelaccordingtotheobject
thatitbelongstoornolabelifthatpixeldoesnotcorrespondtoanythinginthetraining
database. An early network for semantic segmentation is depicted in figure 10.19. The
input is a 224×224 RGB image, and the output is a 224×224×21 array that contains
the probability of each of 21 possible classes at each position.
ThefirstpartofthenetworkisasmallerversionofVGG(figure10.17)thatcontains
thirteenratherthansixteenconvolutionallayersanddownsizestherepresentationtosize
14×14. There is then one more max pooling operation, followed by two fully connected
layers that map to two 1D representations of size 4096. These layers do not represent
spatial position but instead, combine information from across the whole image.
Here, the architecture diverges from VGG. Another fully connected layer reconsti-
tutes the representation into 7×7 spatial positions and 512 channels. This is followed
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

10.6 Summary 179
Figure10.19SemanticsegmentationnetworkofNohetal.(2015). Theinputisa
224×224image,whichispassedthroughaversionoftheVGGnetworkandeven-
tuallytransformedintoarepresentationofsize4096usingafullyconnectedlayer.
This contains information about the entire image. This is then reformed into a
representation of size 7×7 using another fully connected layer, and the image is
upsampled and deconvolved (transposed convolutions without upsampling) in a
mirror image of the VGG network. The output is a 224×224×21 representation
that gives the output probabilities for the 21 classes at each position.
by a series of max unpooling layers (see figure 10.12b) and deconvolution layers. These
are transposed convolutions (see figure 10.13) but in 2D and without the upsampling.
Finally,thereisa1×1convolutiontocreate21channelsrepresentingthepossibleclasses
and a softmax operation at each spatial position to map the activations to class proba-
bilities. The downsampling side of the network is sometimes referred to as an encoder,
and the upsampling side as a decoder, so networks of this type are sometimes called
encoder-decoder networks or hourglass networks due to their shape.
The final segmentation is generated using a heuristic method that greedily searches
for the class that is most represented and infers its region, taking into account the
probabilities but also encouraging connectedness. Then the next most-represented class
isaddedwhereitdominatesattheremainingunlabeledpixels. Thiscontinuesuntilthere
is insuﬀicient evidence to add more (figure 10.20).
10.6 Summary
In convolutional layers, each hidden unit is computed by taking a weighted sum of the
nearby inputs, adding a bias, and applying an activation function. The weights and
the bias are the same at every spatial position, so there are far fewer parameters than
in a fully connected network, and the number of parameters doesn’t increase with the
input image size. To ensure that information is not lost, this operation is repeated with
Draft: please send errata to udlbookmail@gmail.com.

180 10 Convolutional networks
Figure 10.20 Semantic segmentation results. The final result is created from the
21 probability maps by greedily selecting the best class and using a heuristic
methodtofindasensiblebinarymapbasedontheprobabilitiesandtheirspatial
proximity. If there is enough evidence, subsequent classes are added, and their
segmentation maps are combined. Adapted from Noh et al. (2015).
different weights and biases to create multiple channels at each spatial position.
Typicalconvolutionalnetworksconsistofconvolutionallayersinterspersedwithlayers
thatdownsamplebyafactoroftwo. Asadataexamplepassesthroughthenetwork, the
spatialdimensionsusuallydecreasebyfactorsoftwo,andthechannelsincreasebyfactors
of two. At the end of the network, there are typically one or more fully connected layers
that integrate information from across the entire input and create the desired output. If
the output is an image, a mirrored “decoder” upsamples back to the original size.
Thetranslationalequivarianceofconvolutionallayersimposesausefulinductivebias
that increases performance for image-based tasks relative to fully connected networks.
Wedescribedimageclassification,objectdetection,andsemanticsegmentationnetworks.
Image classification performance was shown to improve as the network became deeper.
However, subsequent experiments showed that increasing the network depth indefinitely
doesn’t continue to help; after a certain depth, the system becomes diﬀicult to train.
This is the motivation for residual connections, which are the topic of the next chapter.
Notes
Dumoulin&Visin(2016)presentanoverviewofthemathematicsofconvolutionsthatexpands
on the brief treatment in this chapter.
Convolutional networks: Early convolutional networks were developed by Fukushima &
Miyake (1982), LeCun et al. (1989a), and LeCun et al. (1989b). Initial applications included
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 181
handwritingrecognition(LeCunetal.,1989a;Martin,1993),facerecognition(Lawrenceetal.,
1997),phonemerecognition(Waibeletal.,1989),spokenwordrecognition(Bottouetal.,1990),
and signature verification (Bromley et al., 1993). However, convolutional networks were popu-
larizedbyLeCunetal.(1998),whobuiltasystemcalledLeNetforclassifying28×28grayscale
images of handwritten digits. This is immediately recognizable as a precursor of modern net-
works;itusesaseriesofconvolutionallayers,followedbyfullyconnectedlayers,sigmoidactiva-
tions rather than ReLUs, and average pooling rather than max pooling. AlexNet (Krizhevsky
et al., 2012) is widely considered the starting point for modern deep convolutional networks.
ImageNet Challenge: Dengetal.(2009)collatedtheImageNetdatabaseandtheassociated
classificationchallengedroveprogressindeeplearningforseveralyearsafterAlexNet. Notable
subsequent winners of this challenge include the network-in-network architecture (Lin et al.,
2014), which alternated convolutions with fully connected layers that operated independently
on all of the channels at each position (i.e., 1×1 convolutions). Zeiler & Fergus (2014) and
Simonyan&Zisserman(2014)trainedlargeranddeeperarchitecturesthatwerefundamentally
similar to AlexNet. Szegedy et al. (2017) developed an architecture called GoogLeNet, which
introduced inception blocks. These use several parallel paths with different filter sizes, which
are then recombined. This effectively allowed the system to learn the filter size.
Thetrendwasforperformancetoimprovewithincreasingdepth. However,itultimatelybecame
diﬀicult to train deeper networks without modifications; these include residual connections
and normalization layers, both of which are described in the next chapter. Progress in the
ImageNet challenges is summarized in Russakovsky et al. (2015). A more general survey of
image classification using convolutional networks can be found in Rawat & Wang (2017). The
improvement of image classification networks over time is visualized in figure 10.21.
Types of convolutional layers: Atrous or dilated convolutions were introduced by Chen
etal.(2018c)andYu&Koltun(2015). TransposedconvolutionswereintroducedbyLongetal.
(2015). Odenaetal.(2016)pointedoutthattheycanleadtocheckerboardartifactsandshould
be used with caution. Lin et al. (2014) is an early example of convolution with 1×1 filters.
Many variants of the standard convolutional layer aim to reduce the number of parameters.
Theseincludedepthwiseorchannel-separateconvolution(Howardetal.,2017;Tranetal.,2018),
inwhichadifferentfilterconvolveseachchannelseparatelytocreateanewsetofchannels. For
akernelsizeofK×K withC inputchannelsandC outputchannels,thisrequiresK×K×C
parameters rather than the K ×K ×C ×C parameters in a regular convolutional layer. A
related approach is grouped convolutions (Xie et al., 2017), where each convolution kernel is
only applied to a subset of the channels with a commensurate reduction in the parameters. In
fact, groupedconvolutionswereusedinAlexNetforcomputationalreasons; thewholenetwork
could not run on a single GPU, so some channels were processed on one GPU and some on
another, with limited interaction points. Separable convolutions treat each kernel as an outer
product of 1D vectors; they use C +K +K parameters for each of the C channels. Partial
convolutions (Liu et al., 2018a) are used when inpainting missing pixels and account for the
partial masking of the input. Gated convolutions learn the mask from the previous layer (Yu
et al., 2019; Chang et al., 2019b). Hu et al. (2018b) propose squeeze-and-excitation networks
which re-weight the channels using information pooled across all spatial positions.
Downsamplingandupsampling: AveragepoolingdatesbacktoatleastLeCunetal.(1989a)
and max pooling to Zhou & Chellappa (1988). Scherer et al. (2010) compared these methods
and concluded that max pooling was superior. The max unpooling method was introduced by
Zeiler et al. (2011) and Zeiler & Fergus (2014). Max pooling can be thought of as applying
Draft: please send errata to udlbookmail@gmail.com.

182 10 Convolutional networks
Figure 10.21ImageNetperformance. Eachcirclerepresentsadifferentpublished
model. Blue circles represent models that were state-of-the-art. Models dis-
cussed in this book are also highlighted. The AlexNet and VGG networks were
remarkable for their time but are now far from state of the art. ResNet-200 and
DenseNet are discussed in chapter 11. ImageGPT, ViT, SWIN, and DaViT are
discussedinchapter12. Adaptedfromhttps://paperswithcode.com/sota/image-
classification-on-imagenet.
an L∞ norm to the hidden units that are to be pooled. This led to applying other L
k
norms
AppendixB.3.2
(Springenberg et al., 2015; Sainath et al., 2013), although these require more computation and
Vectornorms
are not widely used. Zhang (2019) introduced max-blur-pooling, in which a low-pass filter is
appliedbeforedownsamplingtopreventaliasing,andshowedthatthisimprovesgeneralization
over translation of the inputs and protects against adversarial attacks (see section 20.4.6).
Shi et al. (2016) introduced PixelShuffle, which used convolutional filters with a stride of 1/s
to scale up 1D signals by a factor of s. Only the weights that lie exactly on positions are
used to create the outputs, and the ones that fall between positions are discarded. This can
be implemented by multiplying the number of channels in the kernel by a factor of s, where
the sth output position is computed from just the sth subset of channels. This can be trivially
extended to 2D convolution, which requires s2 channels.
Convolution in 1D and 3D: Convolutionalnetworksareusuallyappliedtoimagesbuthave
also been applied to 1D data in applications that include speech recognition (Abdel-Hamid
etal.,2012),sentenceclassification(Zhangetal.,2015;Conneauetal.,2017),electrocardiogram
classification (Kiranyaz et al., 2015), and bearing fault diagnosis (Eren et al., 2019). A survey
of 1D convolutional networks can be found in Kiranyaz et al. (2021). Convolutional networks
havealsobeenappliedto3Ddata,includingvideo(Jietal.,2012;Sahaetal.,2016;Tranetal.,
2015) and volumetric measurements (Wu et al., 2015b; Maturana & Scherer, 2015).
Invariance and equivariance: Part of the motivation for convolutional layers is that they
are approximately equivariant with respect to translation, and part of the motivation for max
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 183
pooling is to induce invariance to small translations. Zhang (2019) considers the degree to
which convolutional networks really have these properties and proposes the max-blur-pooling
modification that demonstrably improves them. There is considerable interest in making net-
works equivariant or invariant to other types of transformations, such as reflections, rotations,
and scaling. Sifre & Mallat (2013) constructed a system based on wavelets that induced both
translational and rotational invariance in image patches and applied this to texture classifica-
tion. Kanazawa et al. (2014) developed locally scale-invariant convolutional neural networks.
Cohen&Welling(2016)exploitedgrouptheorytoconstructgroupCNNs,whichareequivariant
to larger families of transformations, including reflections and rotations. Esteves et al. (2018)
introduced polar transformer networks, which are invariant to translations and equivariant to
rotation and scale. Worrall et al. (2017) developed harmonic networks, the first example of a
group CNN that was equivariant to continuous rotations.
Initialization and regularization: Convolutional networks are typically initialized using
Xavierinitialization(Glorot&Bengio,2010)orHeinitialization(Heetal.,2015),asdescribed
in section 7.5. However, the ConvolutionOrthogonal initializer (Xiao et al., 2018a) is special-
Problem10.19
ized for convolutional networks. Networks of up to 10,000 layers can be trained using this
initialization without the need for residual connections.
Dropout is effective for fully connected networks but less so for convolutional layers (Park &
Kwak,2016). Thismaybebecauseneighboringimagepixelsarehighlycorrelated,soifahidden
unitdropsout,thesameinformationispassedonviaadjacentpositions. Thisisthemotivation
for spatial dropout and cutout. In spatial dropout (Tompson et al., 2015), entire feature maps
are discarded instead of individual pixels. This circumvents the problem of neighboring pixels
carryingthesameinformation. Similarly, DeVries&Taylor(2017b)propose cutout, inwhicha
square patch of each input image is masked at training time. Wu & Gu (2015) modified max
poolingfordropoutlayersusingamethodthatinvolvessamplingfromaprobabilitydistribution
over the constituent elements rather than always taking the maximum.
Adaptive Kernels: The inception block (Szegedy et al., 2017) applies convolutional filters of
different sizes in parallel and, as such, provides a crude mechanism by which the network can
learn the appropriate filter size. Other work has investigated learning the scale of convolutions
as part of the training process (e.g., Pintea et al., 2021; Romero et al., 2021) or the stride of
downsampling layers (Riad et al., 2022).
Insomesystems,thekernelsizeischangedadaptivelybasedonthedata. Thisissometimesin
thecontextofguidedconvolution,whereoneinputisusedtohelpguidethecomputationfrom
another input. For example, an RGB image might be used to help upsample a low-resolution
depth map. Jia et al. (2016) directly predicted the filter weights themselves using a different
network branch. Xiong et al. (2020b) change the kernel size adaptively. Su et al. (2019a)
moderate weights of fixed kernels by a function learned from another modality. Dai et al.
(2017) learn offsets of weights so that they do not have to be applied in a regular grid.
Object detection and semantic segmentation: Objectdetectionmethodscanbedivided
into proposal-based and proposal-free schemes. In the former case, processing occurs in two
stages. A convolutional network ingests the whole image and proposes regions that might
contain objects. These proposal regions are then resized, and a second network analyzes them
toestablishwhetherthereisanobjectthereandwhatitis. Anearlyexampleofthisapproach
wasR-CNN(Girshicketal.,2014). Thiswassubsequentlyextendedtoallowend-to-endtraining
(Girshick, 2015) and to reduce the cost of the region proposals (Ren et al., 2015). Subsequent
workonfeaturepyramidnetworksimprovedbothperformanceandspeedbycombiningfeatures
Draft: please send errata to udlbookmail@gmail.com.

184 10 Convolutional networks
across multiple scales (Lin et al., 2017b). In contrast, proposal-free schemes perform all the
processinginasinglepass. YOLO(Redmonetal.,2016),whichwasdescribedinsection10.5.2,
is the most celebrated example of a proposal-free scheme. The most recent iteration of this
framework at the time of writing is YOLOv7 (Wang et al., 2022a). A recent review of object
detection can be found in Zou et al. (2023).
The semantic segmentation network described in section 10.5.3 was developed by Noh et al.
(2015). ManysubsequentapproacheshavebeenvariationsofU-Net(Ronnebergeretal.,2015),
which is described in section 11.5.3. Recent surveys of semantic segmentation can be found in
Minaee et al. (2021) and Ulku & Akagündüz (2022).
Visualizing Convolutional Networks: The dramatic success of convolutional networks led
toaseriesofeffortstovisualizetheinformationtheyextractfromtheimage(seeQinetal.,2018,
for a review). Erhan et al. (2009) visualized the optimal stimulus that activated a hidden unit
by starting with an image containing noise and then optimizing the input to make the hidden
unitmostactiveusinggradientascent. Zeiler&Fergus(2014)trainedanetworktoreconstruct
the input and then set all the hidden units to zero except the one they were interested in;
the reconstruction then provides information about what drives the hidden unit. Mahendran
& Vedaldi (2015) visualized an entire layer of a network. Their network inversion technique
aimedtofindanimagethatresultedintheactivationsatthatlayerbutalsoincorporatesprior
knowledge that encourages this image to have similar statistics to natural images.
Finally, Bau et al. (2017) introduced network dissection. Here, a series of images with known
pixel labels capturing color, texture, and object type are passed through the network, and the
correlation of a hidden unit with each property is measured. This method has the advantage
that it only uses the forward pass of the network and does not require optimization. These
methodsdidprovidesomepartialinsightintohowthenetworkprocessesimages. Forexample,
Bau et al. (2017) showed that earlier layers correlate more with texture and color and later
layers with the object type. However, it is fair to say that fully understanding the processing
of networks containing millions of parameters is currently not possible.
Problems
Problem 10.1∗ Showthattheoperationinequation10.3isequivariantwithrespecttotransla-
tion.
Problem 10.2 Equation 10.3 defines 1D convolution with a kernel size of three, stride of one,
and dilation one. Write out the equivalent equation for the 1D convolution with a kernel size
of three and a stride of two as pictured in figure 10.3a–b.
Problem 10.3 Writeouttheequationforthe1Ddilatedconvolutionwithakernelsizeofthree
and a dilation rate of two, as pictured in figure 10.3d.
Problem 10.4 Write out the equation for a 1D convolution with kernel size of seven, a dilation
rate of three, and a stride of three.
Problem 10.5 Draw weight matrices in the style of figure 10.4d for (i) the strided convolution
in figure 10.3a–b, (ii) the convolution with kernel size 5 in figure 10.3c, and (iii) the dilated
convolution in figure 10.3d.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 185
Problem10.6∗ Drawa12×6weightmatrixinthestyleoffigure10.4drelatinginputsx ,...,x
1 6
to outputs h ,...,h in the multi-channel convolution as depicted in figures 10.5a–b.
1 12
Problem10.7∗ Drawa6×12weightmatrixinthestyleoffigure10.4drelatinginputsh ,...,h
1 12
to outputs h′,...,h′ in the multi-channel convolution in figure 10.5c.
1 6
Problem 10.8 Consider a 1D convolutional network where the input has three channels. The
first hidden layer is computed using a kernel size of three and has four channels. The second
hiddenlayeriscomputedusingakernelsizeoffiveandhastenchannels. Howmanybiasesand
how many weights are needed for each of these two convolutional layers?
Problem10.9Anetworkconsistsofthree1Dconvolutionallayers. Ateachlayer,azero-padded
convolution with kernel size three, stride one, and dilation one is applied. What size is the
receptive field of the hidden units in the third layer?
Problem 10.10 A network consists of three 1D convolutional layers. At each layer, a zero-
paddedconvolutionwithkernelsizeseven,strideone,anddilationoneisapplied. Whatsizeis
the receptive field of hidden units in the third layer?
Problem10.11Consideraconvolutionalnetworkwith1Dinputx. ThefirsthiddenlayerH is
1
computed using a convolution with kernel size five, stride two, and a dilation rate of one. The
second hidden layer H is computed using a convolution with kernelsize three, stride one, and
2
a dilation rate of one. The third hidden layer H is computed using a convolution with kernel
3
sizefive,strideone,andadilationrateoftwo. Whatarethereceptivefieldsizesateachhidden
layer?
Problem10.12The1Dconvolutionalnetworkinfigure10.7wastrainedusingstochasticgradient
descentwithalearningrateof0.01andabatchsizeof100onatrainingdatasetof4,000examples
for 100,000 steps. How many epochs was the network trained for?
Problem 10.13 Draw a weight matrix in the style of figure 10.4d that shows the relationship
between the 24 inputs and the 24 outputs in figure 10.9.
Problem 10.14 Consider a 2D convolutional layer with kernel size 5×5 that takes 3 input
channels and returns 10 output channels. How many convolutional weights are there? How
many biases?
Problem 10.15 Draw a weight matrix in the style of figure 10.4d that samples every other
variable in a 1D input (i.e., the 1D analog of figure 10.11a). Show that the weight matrix for
1D convolution with kernel size three and stride two is equivalent to composing the matrices
for 1D convolution with kernel size three and stride one and this sampling matrix.
Problem 10.16∗ Consider the AlexNet network (figure 10.16). How many parameters are used
in each convolutional and fully connected layer? What is the total number of parameters?
Problem 10.17 What is the receptive field size at each of the first three layers of AlexNet (i.e.,
the first three orange blocks in figure 10.16)?
Problem 10.18 How many weights and biases are there at each convolutional layer and fully
connected layer in the VGG architecture (figure 10.17)?
Problem 10.19∗ Consider two hidden layers of size 224×224 with C and C channels, respec-
1 2
tively, connected by a 3×3 convolutional layer. Describe how to initialize the weights using He
initialization.
Draft: please send errata to udlbookmail@gmail.com.