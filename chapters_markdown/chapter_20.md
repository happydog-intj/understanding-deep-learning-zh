# Chapter 20

*Pages: 416-431*

---

Chapter 20
Why does deep learning work?
Thischapterdiffersfromthosethatprecedeit. Insteadofpresentingestablishedresults,
it poses questions about how and why deep learning works so well. These questions are
rarely discussed in textbooks. However, it’s important to realize that (despite the title
of this book) understanding of deep learning is still limited.
Wearguethatitissurprisingthatdeepnetworksareeasytotrainandalsosurprising
that they generalize. Then we consider each of these topics in turn. We enumerate the
factorsthatinfluencetrainingsuccessanddiscusswhatisknownaboutlossfunctionsfor
deep networks. Then we consider the factors that influence generalization. We conclude
with a discussion of whether networks need to be overparameterized and deep.
20.1 The case against deep learning
The MNIST-1D dataset (figure 8.1) has just forty input dimensions and ten output
dimensions. With enough hidden units per layer, a two-layer fully connected network
classifies 10000 MNIST-1D training data points perfectly and generalizes reasonably to
unseen examples (figure 8.10a). Indeed, we now take it for granted that with suﬀicient
hidden units, deep networks will classify almost any training set near-perfectly. We also
take for granted that the fitted model will generalize to new data. However, it’s not at
all obvious either that the training process should succeed or that the resulting model
should generalize. This section argues that both these phenomena are surprising.
20.1.1 Training
Performanceofatwo-layerfullyconnectednetworkon10000MNIST-1Dtrainingexam-
plesisperfectoncethereare43hiddenunitsperlayeror∼4000parameters(figure8.10).
However, finding the global minimum of an arbitrary non-convex function is NP-hard
(Murty & Kabadi, 1987), and this is also true for certain neural network loss functions
(Blum&Rivest,1992). It’sremarkablethatthefittingalgorithmdoesn’tgettrappedin
local minima or stuck near saddle points and that it can eﬀiciently recruit spare model
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

20.2 Factors that influence fitting performance 403
capacity to fit unexplained training data wherever they lie.
Perhapsthissuccessislesssurprisingwhentherearefarmoreparametersthantrain-
ing data. However, it’s debatable whether this is generally the case. AlexNet had ∼60
millionparametersandwastrainedwith∼1milliondatapoints. However,tocomplicate
matters, each training example was augmented with 2048 transformations. GPT-3 had
175 billion parameters and was trained with 300 billion tokens. There is not a clear-cut
case that either model was overparameterized, and yet they were successfully trained.
In short, it’s surprising that we can fit deep networks reliably and eﬀiciently. Either
the data, the models, the training algorithms, or some combination of all three must
have some special properties that make this possible.
20.1.2 Generalization
If the eﬀicient fitting of neural networks is startling, their generalization to new data
is dumbfounding. First, it’s not obvious a priori that typical datasets are suﬀicient to
characterize the input/output mapping. The curse of dimensionality implies that the
training dataset is tiny compared to the possible inputs; if each of the 40 inputs of the
MNIST-1D data were quantized into 10 possible values, there would be 1040 possible
inputs, which is a factor of 1036 more than the number of training examples.
Problem20.1
Second,deepnetworksdescribeverycomplicatedfunctions. Forexample,afullycon-
nectednetworkforMNIST-1Dwithtwohiddenlayersofwidth400cancreatemappings
withupto1042 linearregions. That’sroughly1038 regionspertrainingexample,sovery
few of these regions contain data at any stage during training; regardless, those regions
that do encounter data points constrain the remaining regions to behave reasonably.
Third,generalizationgetsbetterwithmoreparameters(figure8.10). Themodelinthe
previous paragraph has 177,201 parameters. Assuming it can fit one training example
per parameter, it has 167,201 spare degrees of freedom. This surfeit gives the model
latitude to do almost anything between the training data, and yet it behaves sensibly.
20.1.3 The unreasonable effectiveness of deep learning
To summarize, it’s neither obvious that we should be able to fit deep networks nor that
they should generalize. A priori, deep learning shouldn’t work. And yet it does. This
chapter investigates why. Sections 20.2–20.3 describe what we know about fitting deep
networks and their loss functions. Sections 20.4–20.6 examine generalization.
20.2 Factors that influence fitting performance
Figure 6.4 showed that loss functions for nonlinear models can have both local minima
and saddle points. However, we can reliably fit deep networks to complex training sets.
Forexample,figure8.10showsperfecttrainingperformanceonMNIST-1D,MNIST,and
CIFAR-100. This section considers factors that might resolve this contradiction.
Draft: please send errata to udlbookmail@gmail.com.

404 20 Why does deep learning work?
Figure 20.1Fittingrandomdata. Losses
for AlexNet architecture trained on
CIFAR-10 dataset with SGD. When
the pixels are drawn from a Gaus-
sian random distribution with the same
meanandvarianceastheoriginalimage
dataset,themodelcanstillbefit(albeit
more slowly). When the labels are ran-
domized,themodelcanstillbefit(albeit
evenmoreslowly). AdaptedfromZhang
et al. (2017a).
20.2.1 Dataset
It’simportanttorealizethatwecan’tlearnanyfunction. Consideracompletelyrandom
mapping from every possible 28×28 binary image to one of ten categories. Since there
is no structure to this function, the only recourse is to memorize the 2784 assignments.
However, it’s easy to train a model on the MNIST dataset (figures 8.10 and 15.15),
whichcontains60,000examplesof28×28imageslabeledwithoneoftencategories. One
explanation for this contradiction could be that it is easy to find global minima because
the real-world functions that we approximate are relatively simple.1
This hypothesis was investigated by Zhang et al. (2017a), who trained AlexNet on
Notebook20.1 the CIFAR-10 image classification dataset (which has 50,000 examples of 32×32×3
Randomdata
images labeled with one of 10 classes) when (i) each image was replaced with Gaussian
noise and (ii) the labels of the ten classes were randomly permuted (figure 20.1). These
changes slowed down learning, but the network could still fit this finite dataset well.
Problem20.2
This suggests that the properties of the dataset aren’t critical.
20.2.2 Regularization
Another possible explanation for the ease with which models are trained is that some
regularizationmethodslikeL2regularization(weightdecay)makethelosssurfaceflatter
andmoreconvex. However,Zhangetal.(2017a)foundthatneitherL2regularizationnor
Dropoutwasrequiredtofitrandomdata. Thisdoesnoteliminateimplicitregularization
due to the finite step size of the fitting algorithms (section 9.2). However, this effect
increases with the learning rate (equation 9.9), and model-fitting does not get easier
with larger learning rates.
20.2.3 Stochastic training algorithms
Chapter6arguedthattheSGDalgorithmpotentiallyallowstheoptimizationtrajectory
to move between “valleys” during training. However, Keskar et al. (2017) show that
1Inthischapter,weusetheterm“globalminimum”looselytomeananysolutionwherealldataare
classifiedcorrectly. Wehavenowayofknowingiftherearesolutionswithalowerlosselsewhere.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

20.2 Factors that influence fitting performance 405
Figure 20.2 MNIST-1D training. Four
fullyconnectednetworkswerefitto4000
MNIST-1Dexampleswithrandomlabels
usingfullbatchgradientdescent,Heini-
tialization, no momentum or regulariza-
tion, and learning rate 0.0025. Mod-
els with 1,2,3,4 layers had 298, 100, 75,
and63hiddenunitsperlayerand15208,
15210, 15235, and 15139 parameters, re-
spectively. All models train successfully,
but deeper models require fewer epochs.
severalmodels(includingfullyconnectedandconvolutionalnetworks)canbefittomany
datasets (including CIFAR-100 and MNIST) almost perfectly with very large batches of
5000-6000 images. This eliminates most of the randomness but training still succeeds.
Figure 20.2 shows training results for four fully connected models fitted to 4000
Notebook20.2
MNIST-1D examples with randomized labels using full-batch (i.e., non-stochastic) gra-
Fullbatch
dient descent. There was no explicit regularization, and the learning rate was set to a gradientdescent
small constant value of 0.0025 to minimize implicit regularization. Here, the true map-
ping from data to labels has no structure, the training is deterministic, and there is no Problem20.3
regularization,andyetthetrainingerrorstilldecreasestozero. Thissuggeststhatthese
loss functions may genuinely have no local minima.
20.2.4 Overparameterization
Overparameterization almost certainly is an important factor that contributes to ease
of training. It implies that there is a large family of degenerate solutions, so there may
always be a direction in which the parameters can be modified to decrease the loss.
Sejnowski (2020) suggests that “... the degeneracy of solutions changes the nature of
the problem from finding a needle in a haystack to a haystack of needles.”
In practice, networks are frequently overparameterized by one or two orders of mag-
nitude (figure 20.3). However, data augmentation makes it diﬀicult to make precise
statements. Augmentation may increase the data by several orders of magnitude, but
these are manipulations of existing examples rather than independent new data points.
Moreover, figure 8.10 shows that neural networks can sometimes fit the training data
well when there are the same number or fewer parameters than data points. This is
presumably due to redundancy in training examples from the same underlying function.
Several theoretical convergence results show that, under certain circumstances, SGD
converges to a global minimum when the network is suﬀiciently overparameterized. For
example, Du et al. (2019b) show that randomly initialized SGD converges to a global
minimum for shallow fully connected ReLU networks with a least squares loss with
enough hidden units. Similarly, Du et al. (2019a) consider deep, residual, and convolu-
tional networks when the activation function is smooth and Lipschitz. Zou et al. (2020)
analyzed the convergence of gradient descent on deep, fully connected networks using a
hinge loss. Allen-Zhu et al. (2019) considered deep networks with ReLU functions.
Draft: please send errata to udlbookmail@gmail.com.

406 20 Why does deep learning work?
Figure 20.3 Overparameterization. Im-
ageNet performance for convolutional
nets as a function of overparameteriza-
tion (in multiples of dataset size). Most
models have 10–100 times more param-
eters than there were training exam-
ples. Models compared are ResNet (He
etal.,2016a,b),DenseNet(Huangetal.,
2017b), Xception (Chollet, 2017), Eﬀi-
cientNet (Tan & Le, 2019), Inception
(Szegedy et al., 2017), ResNeXt (Xie
et al., 2017), and AmoebaNet (Cubuk
et al., 2019).
If a neural network is suﬀiciently overparameterized so that it can memorize any
dataset of a fixed size, then all stationary points become global minima (Livni et al.,
2014; Nguyen & Hein, 2017, 2018). Other results show that if the network is wide
enough, local minima where the loss is higher than the global minimum are rare (see
Choromanska et al., 2015; Pascanu et al., 2014; Pennington & Bahri, 2017). Kawaguchi
et al. (2019) prove that as a network becomes deeper, wider, or both, the loss at local
minima becomes closer to that at the global minimum for squared loss functions.
These theoretical results are intriguing but usually make unrealistic assumptions
aboutthenetworkstructure. Forexample,Duetal.(2019a)showthatresidualnetworks
convergetozerotraininglosswhenthewidthofthenetworkD(i.e.,thenumberofhidden
units) is Ω[I4K2] where I is the amount of training data, and K is the depth of the
network. Similarly,Nguyen&Hein(2017)assumethatthenetwork’swidthislargerthan
the dataset size, which is unrealistic in most practical scenarios. Overparameterization
seems to be important, but theory cannot yet explain empirical fitting performance.
20.2.5 Activation functions
The activation function is also known to affect training diﬀiculty. Networks where the
activationonlychangesoverasmallpartoftheinputrangearehardertofitthanReLUs
(which vary over half the input range) or Leaky ReLUs (which vary over the full range);
For example, sigmoid and tanh nonlinearities (figure 3.13a) have shallow gradients in
their tails; where the activation function is near-constant, the training gradient is near-
zero, so the mechanism to improve the model is extremely weak.
20.2.6 Initialization
Another potential explanation is that Xavier/He initialization sets the parameters to
values that are easy to optimize. Of course, for deeper networks, such initialization is
necessary to avoid exploding and vanishing gradients, so in a trivial sense, initialization
iscriticaltotrainingsuccess. However,forshallowernetworks,theinitialvarianceofthe
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

20.3 Properties of loss functions 407
Figure 20.4 Initialization and fitting. A
three-layerfullyconnectednetworkwith
200 hidden units per layer was trained
on 1000 MNIST examples with AdamW
using one-hot targets and mean-squared
errorloss. Ittakeslongertofitnetworks
whenlargermultiplesofHeinitialization
areused,butthisdoesn’tchangetheout-
come. This may simply reflect the ex-
tradistancethattheweightsmustmove.
Adapted from Liu et al. (2023c).
weightsislessimportant. Liuetal.(2023c)traineda3-layerfullyconnectednetworkwith
200hiddenunitsperlayeron1000MNISTdatapoints. Theyfoundthatmoreiterations
wererequiredtofitthetrainingdataasthevarianceincreasedfromthatproposedbyHe
(figure20.4),butthisdidnotultimatelyimpedefitting. Hence,initializationdoesn’tshed
muchlightonwhyfittingneuralnetworksiseasy,althoughexploding/vanishinggradients
do reveal initializations that make training diﬀicult with finite precision arithmetic.
20.2.7 Network depth
Neural networks are harder to fit when the depth becomes very large due to exploding
and vanishing gradients (figure 7.7) and shattered gradients (figure 11.3). However,
these are (arguably) practical numerical issues. There is no definitive evidence that
the underlying loss function is fundamentally more or less convex as the network depth
increases. Figure 20.2 does show that for MNIST data with randomized labels and He
initialization, deeper networks train in fewer iterations. However, this might be because
either (i) the gradients in deeper networks are steeper or (ii) He initialization just starts
wider, shallower networks further away from the optimal parameters.
Frankle & Carbin (2019) show that for small networks like VGG, you can get the
same or better performance if you (i) train the network, (ii) prune the weights with
the smallest magnitudes and (iii) retrain from the same initial weights. This does not
work if the weights are randomly re-initialized. They concluded that the original over-
parameterized network contains small trainable sub-networks, which are suﬀicient to
Notebook20.3
provide the performance. They term this the lottery ticket hypothesis and denote the
Lotterytickets
sub-networksaswinningtickets. Thissuggeststhattheeffectivenumberofsub-networks
may have a key role to play in fitting. This (perhaps) varies with the network depth for
a fixed parameter count, but a precise characterization of this idea is lacking.
20.3 Properties of loss functions
Theprevioussectiondiscussedfactorsthatcontributetotheeasewithwhichneuralnet-
works can be trained. The number of parameters (degree of overparameterization) and
Draft: please send errata to udlbookmail@gmail.com.

408 20 Why does deep learning work?
the choice of activation function are both important. Surprisingly, the choice of dataset,
therandomnessofthefittingalgorithm, andtheuseofregularizationdon’tseemimpor-
tant. There is no definitive evidence that (for a fixed parameter count) the depth of the
network matters (other than numerical problems due to exploding/vanishing/shattered
gradients). This section tackles the same topic from a different angle by considering the
empirical properties of loss functions. Most of this evidence comes from fully connected
networks and CNNs; loss functions of transformer networks are less well understood.
20.3.1 Multiple global minima
We expect loss functions for deep networks to have a large family of equivalent global
minima. Infullyconnectednetworks,thehiddenunitsateachlayerandtheirassociated
weights can be permuted without changing the output. In convolutional networks, per-
muting the channels and convolution kernels appropriately doesn’t change the output.
We can multiply the weight before any ReLU function and divide the weight after by a
positive number without changing the output. Using BatchNorm induces another set of
redundancies because the mean and variance of each hidden unit or channel are reset.
The above modifications all produce the same output for every input. However, the
global minimum only depends on the output at the training data points. In overparam-
eterized networks, there will also be families of solutions that behave identically at the
data points but differently between them. All of these are also global minima.
20.3.2 Route to the minimum
Goodfellow et al. (2015b) considered a straight line between the initial parameters and
the final values. They show that the loss function along this line usually decreases
monotonically (except for a small bump near the start sometimes). This phenomenon is
observed for several different types of networks and activation functions (figure 20.5a).
Of course, real optimization trajectories do not proceed in a straight line. However,
Li et al. (2018b) find that they do lie in low-dimensional subspaces. They attribute this
to the existence of large, nearly convex regions in the loss landscape that capture the
trajectory early on and funnel it in a few important directions. Surprisingly, Li et al.
(2018a) showed that networks still train well if optimization is constrained to lie in a
random low-dimensional subspace (figure 20.6).
Li & Liang (2018) show that the relative change in the parameters during training
decreases as network width increases; for larger widths, the parameters start at smaller
values, change by a smaller proportion of those values, and converge in fewer steps.
20.3.3 Connections between minima
Goodfellow et al. (2015b) examined the loss function along a straight line between two
minima that were found independently. They saw a pronounced increase in the loss be-
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

20.3 Properties of loss functions 409
Figure 20.5 Linear slices through loss function. a) A two-layer fully connected
ReLUnetworkistrainedonMNIST.Thelossalongastraightlinestartingatthe
initial parameters (δ=0) and finishing at the trained parameters (δ=1) descends
monotonically. b)However,inthistwo-layerfullyconnectedMaxOutnetworkon
MNIST,thereisanincreaseinthelossalongastraightlinebetweenonesolution
(δ=0) and another (δ=1). Adapted from Goodfellow et al. (2015b).
Figure 20.6 Subspace training. A fully
connected network with two hidden lay-
ers, each with 200 units was trained on
MNIST. Parameters were initialized us-
ing a standard method but then con-
strained to lie within a random sub-
space. Performance reaches 90% of the
unconstrainedlevelwhenthissubspaceis
750D (termed the intrinsic dimension),
whichis0.4%oftheoriginalparameters.
Adapted from Li et al. (2018a).
tween them (figure 20.5b); good minima are not generally linearly connected. However,
Frankle et al. (2020) showed that this increase vanishes if the networks are identically
trained initially and later allowed to diverge by using different SGD noise and augmen-
tation. This suggests that the solution is constrained early in training and that some
families of minima are linearly connected.
Draxler et al. (2018) found minima with good (but different) performance on the
CIFAR-10 dataset. They then showed that it is possible to construct paths from one to
theother,wherethelossfunctionremainslowalongthispath. Theyconcludethatthere
is a single connected manifold of low loss (figure 20.7). This seems to be increasingly
true as the width and depth of the network increase. Garipov et al. (2018) and Fort &
Jastrzębski (2019) present other schemes for connecting minima.
Draft: please send errata to udlbookmail@gmail.com.

410 20 Why does deep learning work?
Figure 20.7 Connections between min-
ima. A slice through the loss function
of DenseNet on CIFAR-10. Parameters
ϕ andϕ aretwoindependentlydiscov-
1 2
ered minima. Linear interpolation be-
tween these parameters reveals an en-
ergy barrier (dashed line). However, for
suﬀiciently deep and wide networks, it
is possible to find a curved path of low
energy between two minima (cyan line).
Adapted from Draxler et al. (2018).
Figure20.8Criticalpointsvs.loss. a)InrandomGaussianfunctions,thenumber
ofdirectionsinwhichthefunctioncurvesdownatpointswithzerogradient(i.e.,
saddle points) decreases with the height of the function, so minima all appear
at lower function values. b) Dauphin et al. (2014) found critical points on a
neural network loss surface (i.e., points with zero gradient). They showed that
theproportionofnegativeeigenvalues(directionsthatpointdown)decreaseswith
the loss. The implication is that all minima (points with zero gradient where no
directions point down) have low losses. Adapted from Dauphin et al. (2014) and
Bahri et al. (2020).
Figure 20.9 Goldilocks zone. The pro-
portionofeigenvaluesoftheHessianthat
are greater than zero (a measure of pos-
itive curvature/convexity) within a ran-
domsubspaceofdimensionD inatwo-
s
layerfullyconnectednetworkwithReLU
functions applied to MNIST as a func-
tion of the squared radius r2 of the pa-
rametersrelativetoXavierinitialization.
There is a pronounced region of positive
curvature known as the Goldilocks zone.
Adapted from Fort & Scherlis (2019).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

20.4 Factors that determine generalization 411
Figure 20.10 Batch size to learning rate
ratio. Generalization of two models
on the CIFAR-10 database depends on
the ratio of batch size to the learning
rate. As the batch size increases, gen-
eralization decreases. As the learning
rate increases, generalization increases.
Adapted from He et al. (2019).
20.3.4 Curvature of loss surface
Random Gaussian functions (in which points are jointly distributed with covariance
given by a kernel function of their distance) have an interesting property: for points
where the gradient is zero, the fraction of directions where the function curves down
becomes smaller when these points occur at lower loss values (see Bahri et al., 2020).
Dauphin et al. (2014) searched for saddle points in a neural network loss function and
similarly found a correlation between the loss and the number of negative eigenvalues
(figure20.8). Baldi&Hornik(1989)analyzedtheerrorsurfaceofashallownetworkand
foundthattherewerenolocalminimabutonlysaddlepoints. Theseresultssuggestthat
there are few or no bad local minima.
Fort&Scherlis(2019)measuredthecurvatureatrandompointsonaneuralnetwork
loss surface; they showed that the curvature of the surface is unusually positive when
the ℓ norm of the weights lies within a certain range (figure 20.9), which they term the
2
Goldilocks zone. He and Xavier initialization fall within this range.
20.4 Factors that determine generalization
The last two sections considered factors that determine whether the network trains suc-
cessfully and what is known about neural network loss functions. This section considers
factors that determine how well the network generalizes. This complements the discus-
sion of regularization (chapter 9), which explicitly aims to encourage generalization.
20.4.1 Training algorithms
Since deep networks are usually overparameterized, the details of the training process
determine which of the degenerate family of minima the algorithm converges to. Some
of these details reliably improve generalization.
LeCun et al. (2012) show that SGD generalizes better than full-batch gradient de-
scent. It has been argued that SGD generalizes better than Adam (e.g., Wilson et al.,
2017; Keskar & Socher, 2017), but more recent studies suggest that there is little dif-
Draft: please send errata to udlbookmail@gmail.com.

412 20 Why does deep learning work?
Figure 20.11 Flat vs. sharp minima.
Flat minima are expected to generalize
better. Smallerrorsinestimatingthepa-
rametersorinthealignmentofthetrain
and test loss functions are less problem-
aticinflatregions. AdaptedfromKeskar
et al. (2017).
ference when the hyperparameter search is done carefully (Choi et al., 2019). Keskar
etal.(2017)showthatdeepnetsgeneralizebetterwithsmallerbatch-sizewhennoother
form of regularization is used. It is also well-known that larger learning rates tend to
generalize better (e.g., figure 9.5). Jastrzębski et al. (2018), Goyal et al. (2018), and He
et al. (2019) argue that the batch size/learning rate ratio is important. He et al. (2019)
show a significant correlation between this ratio and the degree of generalization and
prove a generalization bound for neural networks, which has a positive correlation with
this ratio (figure 20.10).
These observations are aligned with the discovery that SGD implicitly adds regu-
larization terms to the loss function (section 9.2), and their magnitude depends on the
learning rate. The trajectory of the parameters is changed by this regularization, and
they converge to a part of the loss function that generalizes well.
20.4.2 Flatness of minimum
There has been speculation dating at least to Hochreiter & Schmidhuber (1997a) that
flat minima in the loss function generalize better than sharp minima (figure 20.11).
Informally, if the minimum is flatter, then small errors in the estimated parameters are
less important. This can also be motivated from various theoretical viewpoints. For
example, minimum description length theory suggests models specified by fewer bits
generalize better (Rissanen, 1983). For wide minima, the precision needed to store the
weights is lower, so they should generalize better.
Flatnesscanbemeasuredby(i)thesizeoftheconnectedregionaroundtheminimum
for which training loss is similar (Hochreiter & Schmidhuber, 1997a), (ii) the second-
order curvature around the minimum (Chaudhari et al., 2019), or (iii) the maximum
loss within a neighborhood of the minimum (Keskar et al., 2017). However, caution is
required;estimatedflatnesscanbeaffectedbytrivialreparameterizationsofthenetwork
due to the non-negative homogeneity property of the ReLU function (Dinh et al., 2017).
Nonetheless, Keskar et al. (2017) varied the batch size and learning rate and showed
that flatness correlates with generalization. Izmailov et al. (2018) average together
weights from multiple points in a learning trajectory. This both results in flatter test
andtrainingsurfacesattheminimumandimprovesgeneralization. Otherregularization
techniques can also be viewed through this lens. For example, averaging model outputs
(ensembling) may also make the test loss surface flatter. Kleinberg et al. (2018) showed
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

20.4 Factors that determine generalization 413
thatlargegradientvarianceduringtraininghelpsavoidsharpregions. Thismayexplain
why reducing the batch size and adding noise helps generalization.
The above studies consider flatness for a single model and training set. However,
sharpness alone is not a good criterion to predict generalization between datasets; when
the labels in the CIFAR dataset are randomized (making generalization impossible),
there is no commensurate sharpening of the minimum (Neyshabur et al., 2017).
20.4.3 Architecture
The inductive bias of a network is determined by its architecture, and judicious choices
of model can drastically improve generalization. Chapter 10 introduced convolutional
networks, which are designed to process data on regular grids; they implicitly assume
that the input statistics are the same across the input, so they share parameters across
position. Similarly,transformersaresuitedformodelingdatathatisinvarianttopermu-
tations, and graph neural networks are suited to data represented on irregular graphs.
Matching the architecture to the properties of the data improves generalization over
generic, fully connected architectures (see figure 10.8).
20.4.4 Norm of weights
Section20.3.4reviewedthefindingofFort&Scherlis(2019)thatthecurvatureoftheloss
surfaceisunusuallypositivewhentheℓ normoftheweightslieswithinacertainrange.
2
The same authors provided evidence that generalization is also good when the ℓ weight
2
norm falls within this Goldilocks zone (figure 20.12). This is perhaps unsurprising. The
norm of the weights is (indirectly) related to the Lipschitz constant of the model. If this
norm is too small, then the model will not be able to change fast enough to capture the
variation in the underlying function. If the norm is too large, then the model will be
unnecessarily variable between training points and will not interpolate smoothly.
This finding was used by Liu et al. (2023c) to explain the phenomenon of grokking
(Power et al., 2022), in which a sudden improvement in generalization can occur many
epochsafterthetrainingerrorisalreadyzero(figure20.13). Itisproposedthatgrokking
occurs when the norm of the weights is initially too large; the training data fits well,
but the variation of the model between the data points is large. Over time, implicit or
explicit regularization decreases the norm of the weights until they reach the Goldilocks
zone, and generalization suddenly improves.
20.4.5 Overparameterization
Figure 8.10 showed that generalization performance tends to improve with the degree
of overparameterization. When combined with the bias/variance trade-off curve, this
results in double descent. The putative explanation for this improvement is that the
network has more latitude to become smoother between the training data points when
Draft: please send errata to udlbookmail@gmail.com.

414 20 Why does deep learning work?
Figure 20.12 Generalization on hyper-
spheres. Afullyconnectednetworkwith
two hidden layers, each with 200 units
(199,210parameters)wastrainedonthe
MNIST database. The parameters are
initialized to a given ℓ norm and then
2
constrained to maintain this norm and
to lie in a subspace (vertical direction).
The network generalizes well in a small
range around the radius r defined by
Xavier initialization (cyan dotted line).
Adapted from Fort & Scherlis (2019).
Figure 20.13 Grokking. When the pa-
rameters are initialized so that their
ℓ norm (radius) is considerably larger
2
than is specified by He initialization,
trainingtakeslonger(dashed lines), and
generalization takes much longer (solid
lines). The lag in generalization is at-
tributed to the time taken for the norm
of the weights to decrease back to the
Goldilockszone. AdaptedfromLiuetal.
(2023c).
the model is overparameterized.
It follows that the norm of the weights can also be used to explain double descent.
The norm of the weights increases when the number of parameters is similar to the
number of data points (as the model contorts itself to fit these points exactly), causing
generalization to reduce. As the network becomes wider and the number of weights
increases, the overall norm of these weights decreases; the weights are initialized with a
variancethatisinverselyproportionaltothewidth(i.e.,withHeorGlorotinitialization),
and the weights need not change as drastically to fit the data well.
20.4.6 Leaving the data manifold
Untilthispoint,wehavediscussedhowmodelsgeneralizetonewdatathatisdrawnfrom
the same distribution as the training data. This is a reasonable assumption for experi-
mentation. However,systemsdeployedintherealworldmayencounterunexpecteddata
due to noise, changes in the data statistics over time, or deliberate attacks. Of course,
it is harder to make definite statements about this scenario, but D’Amour et al. (2020)
show that the variability of identical models trained with different seeds on corrupted
data can be enormous and unpredictable.
Goodfellowetal.(2015a)showedthatdeeplearningmodelsaresusceptibletoadver-
Notebook20.4 sarial attacks. Consider perturbing an image that is correctly classified by the network
Adversarialattacks
as “dog” so that the probability of the correct class decreases as fast as possible un-
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

20.5 Do we need so many parameters? 415
Figure 20.14 Adversarial examples. In
eachcase,theleftimageiscorrectlyclas-
sified by AlexNet. By considering the
gradientsofthenetworkoutputwithre-
spect to the input, it’s possible to find
a small perturbation (center, magnified
by 10 for visibility) that, when added
to the original image (right), causes the
network to misclassify it as an ostrich.
This is despite the fact that the original
and perturbed images are almost indis-
tinguishable to humans. Adapted from
Szegedy et al. (2014).
til the class flips. If this image is now classified as an airplane, you might expect the
perturbed image to look like a cross between a dog and an airplane. However, in prac-
tice, the perturbed image looks almost indistinguishable from the original dog image
(figure 20.14).
The conclusion is that there are positions that are close to but not on the data man-
ifold that are misclassified. These are known as adversarial examples. Their existence
is surprising; how can such a small change to the network input make such a drastic
change to the output? The best current explanation is that adversarial examples aren’t
due to a lack of robustness to data from outside the training data manifold. Instead,
they are exploiting a source of information that is in the training distribution but which
has a small norm and is imperceptible to humans (Ilyas et al., 2019).
20.5 Do we need so many parameters?
Section 20.4 argued that models generalize better when over-parameterized. Indeed,
there are almost no examples of state-of-the-art test performance on complex datasets
wherethemodelhassignificantlyfewerparametersthanthereweretrainingdatapoints.
However, section 20.2 reviewed evidence that training becomes easier as the number
of parameters increases. Hence, it’s not clear if some fundamental property of smaller
models prevents them from performing as well or whether the training algorithms can’t
findgoodsolutionsforsmallmodels. Pruninganddistillingaretwomethodsforreducing
the size of trained models. This section examines whether these methods can produce
underparameterized models which retain the performance of overparameterized ones.
20.5.1 Pruning
Pruningtrainedmodelsreducestheirsizeandhencestoragerequirements(figure20.15).
The simplest approach is to remove individual weights. This can be done based on the
second derivatives of the loss function (LeCun et al., 1990; Hassibi & Stork, 1993) or
Draft: please send errata to udlbookmail@gmail.com.

416 20 Why does deep learning work?
Figure 20.15 Pruning neural networks. The goal is to remove as many weights
aspossiblewithoutdecreasingperformance. Thisisoftendonejustbasedonthe
magnitude of the weights. Typically, the network is fine-tuned after pruning. a)
Example fully connected network. b) After pruning.
(more practically) based on the absolute value of the weight (Han et al., 2016, 2015).
Otherworkpruneshiddenunits(Zhouetal.,2016a;Alvarez&Salzmann,2016),channels
in convolutional networks (Li et al., 2017a; Luo et al., 2017b; He et al., 2017; Liu et al.,
2019a), or entire layers in residual nets (Huang & Wang, 2018). Often, the network is
fine-tuned after pruning, and sometimes this process is repeated.
For example, Han et al. (2016) maintained good performance for the VGG network
on ImageNet classification when 8% of the weights were retained. This significantly
decreases the model size but isn’t enough to show that overparameterization is not re-
quired; the VGG network has ∼100 times as many parameters as there are ImageNet
training data (disregarding augmentation).
Pruning is a form of architecture search. In their work on lottery tickets (see sec-
tion20.2.7), Frankle&Carbin(2019)(i)trainedanetwork, (ii)prunedtheweightswith
the smallest magnitudes, and (iii) retrained the remaining network from the same ini-
tial weights. By iterating this procedure, they reduced the size of the VGG-19 network
(originally 138 million parameters) by 98.5% on the CIFAR-10 database (60,000 exam-
ples) while maintaining good performance. For ResNet-50 (25.6 million parameters),
they reduced the parameters by 80% without reducing the performance on ImageNet
(1.28 million examples). These demonstrations are impressive but (disregarding data
augmentation) these networks are still over-parameterized after pruning.
20.5.2 Knowledge distillation
The parameters can also be reduced by training a smaller network (the student) to
replicate the performance of a larger one (the teacher). This is known as knowledge
distillation and dates back to at least Buciluǎ et al. (2006). Hinton et al. (2015) showed
that the pattern of information across the output classes is important and trained a
smaller network to approximate the pre-softmax logits of the larger one (figure 20.16).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

20.5 Do we need so many parameters? 417
Figure20.16Knowledgedistillation. a)Ateachernetworkforimageclassification
istrainedasusual,usingamulticlasscross-entropyclassificationloss. b)Asmaller
student network is trained with the same loss, plus also a distillation loss that
encourages the pre-softmax activations to be the same as for the teacher.
Zagoruyko & Komodakis (2017) further encouraged the spatial maps of the activa-
tionsofthestudentnetworktobesimilartotheteachernetworkatvariouspoints. They
usethisattention transfermethodtoapproximatetheperformanceofa34-layerresidual
network(∼63millionparameters)withan18-layerresidualnetwork(∼11millionparam-
eters) on the ImageNet classification task. However, this is still larger than the number
oftrainingexamples(∼1millionimages). Modernmethods(e.g.Chenetal.,2021a)can
improve on this result, but distillation has not yet provided convincing evidence that
under-parameterized models can perform well.
20.5.3 Discussion
Current evidence suggests that overparameterization is needed for generalization — at
least for the size and complexity of datasets that are currently used. There are no
demonstrationsofstate-of-the-artperformanceoncomplexdatasetswheretherearesig-
nificantly fewer parameters than training examples. Attempts to reduce model size by
pruning or distilling trained networks have not changed this picture.
Moreover,recenttheoryshowsthatthereisatrade-offbetweenthemodel’sLipschitz
constantandoverparameterization;Bubeck&Sellke(2021)provedthatinDdimensions,
Draft: please send errata to udlbookmail@gmail.com.