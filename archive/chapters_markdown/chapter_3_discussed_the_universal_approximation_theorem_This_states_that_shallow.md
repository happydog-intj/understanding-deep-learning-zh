# Chapter 3: discussed the universal approximation theorem. This states that shallow

*Pages: 432-434*

---

418 20 Why does deep learning work?
smooth interpolation requires D times more parameters than mere interpolation. They
argue that current models for large datasets (e.g., ImageNet) aren’t overparameterized
enough; increasing model capacity further may be key to improving performance.
20.6 Do networks have to be deep?
Chapter 3 discussed the universal approximation theorem. This states that shallow
neuralnetworkscanapproximateanyfunctiontoarbitraryaccuracygivenenoughhidden
units. This raises the obvious question of whether networks need to be deep.
First, let’s consider the evidence that depth is required. Historically, there has been
adefinitecorrelationbetweenperformanceanddepth. Forexample, performanceonthe
ImageNet benchmark initially improved as a function of network depth until training
became diﬀicult. Subsequently, residual connections and batch normalization (chap-
ter 11) allowed training of deeper networks with commensurate gains in performance.
Atthetimeofwriting,almostallstate-of-the-artapplications,includingimageclassifica-
tion (e.g., the vision transformer), text generation (e.g., GPT3), and text-guided image
synthesis (e.g., DALL·E-2), are based on deep networks with tens or hundreds of layers.
Despite this trend, there have been efforts to use shallower networks. Zagoruyko &
Komodakis(2016)constructedshallowerbutwiderresidualneuralnetworksandachieved
similarperformancetoResNet. Morerecently,Goyaletal.(2021)constructedanetwork
thatusedparallelconvolutionalchannelsandachievedperformancesimilartodeepernet-
workswithonly12layers. Furthermore,Veitetal.(2016)showedthatitispredominantly
shorter paths of 5–17 layers that drive performance in residual networks.
Nonetheless,thebalanceofevidencesuggeststhatdepthiscritical; eventheshallow-
est networks with good image classification performance require >10 layers. However,
there is no definitive explanation for why. Three possible explanations are that (i) deep
networks can represent more complex functions than shallow ones, (ii) deep networks
are easier to train, and (iii) deep networks impose better inductive biases.
20.6.1 Complexity of modeled function
Chapter4showedthatdeepnetworksmakefunctionswithmanymorelinearregionsthan
shallow ones for the same parameter count. We also saw that “pathological” functions
havebeenidentifiedthatrequireexponentiallymorehiddenunitstomodelwithashallow
networkthanadeepone(e.g.,Eldan&Shamir,2016;Telgarsky,2016). IndeedLiang&
Srikant(2016)foundquitegeneralfamiliesoffunctionsthataremoreeﬀicientlymodeled
by deep networks. However, Nye & Saxe (2018) found that some of these functions
cannot easily be fit by deep networks in practice. Moreover, there is little evidence that
the real-world functions that we are approximating have these pathological properties.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

20.7 Summary 419
20.6.2 Tractability of training
An alternative explanation is that shallow networks with a practical number of hidden
units could support state-of-the-art performance, but it is just diﬀicult to find a good
solution that both fits the training data well and interpolates sensibly.
Onewaytoshowthisistodistillsuccessfuldeepnetworksintoshallower(butwider)
student models and see if performance can be maintained. Urban et al. (2017) dis-
tilledanensembleof16convolutionalnetworksforimageclassificationontheCIFAR-10
dataset into student models of varying depths. They found that shallow networks could
not replicate the performance of the deeper teacher and that the student performance
increased as a function of depth for a constant parameter budget.
20.6.3 Inductive bias
Mostcurrentmodelsrelyonconvolutionalblocksortransformers. Thesenetworksshare
parameters for local regions of the input data, and often they gradually integrate this
information across the whole input. These constraints mean that the functions that
these networks can represent are not general. One explanation for the supremacy of
deep networks, then, is that these constraints have a good inductive bias and that it is
diﬀicult to induce shallow networks to obey these constraints.
Multi-layer convolutional architectures seem to be inherently helpful, even without
training. Ulyanov et al. (2018) demonstrated that the structure of an untrained CNN
can be used as a prior in low-level tasks such as denoising and super-resolution. Frankle
etal.(2021)achievedgoodperformanceinimageclassificationbyinitializingthekernels
randomly,fixingtheirvalues,andjusttrainingthebatchnormalizationoffsetandscaling
factors. Zhang et al. (2017a) show that features from randomly initialized convolutional
filters can support subsequent image classification using a kernel model.
Additionalevidencethatconvolutionalnetworksprovideausefulinductivebiascomes
from Urban et al. (2017), who attempted to distill convolutional networks into shal-
lower networks. They found that distilling into convolutional architectures systemat-
ically worked better than distilling into fully connected networks. This suggests that
the convolutional architecture has some inherent advantages. Since the sequential local
processing of convolutional networks cannot easily be replicated by shallower networks,
this argues that depth is indeed important.
20.7 Summary
This chapter has made the case that the success of deep learning is surprising. We
discussed the challenges of optimizing high-dimensional loss functions and argued that
overparameterization and the choice of activation function are the two most important
factors that make this tractable in deep networks. We saw that, during training, the
parameters move through a low-dimensional subspace to one of a family of connected
Draft: please send errata to udlbookmail@gmail.com.

420 20 Why does deep learning work?
global minima and that local minima are not apparent.
Generalizationofneuralnetworksalsoimproveswithoverparameterization,although
otherfactors,suchastheflatnessoftheminimumandtheinductivebiasofthearchitec-
ture,arealsoimportant. Itappearsthatbothalargenumberofparametersandmultiple
network layers are required for good generalization, although we do not yet know why.
Manyquestionsremainunanswered. Wedonotcurrentlyhaveanyprescriptivetheory
that will allow us to predict the circumstances in which training and generalization will
succeed or fail. We do not know the limits of learning in deep networks or whether
much more eﬀicient models are possible. We do not know if there are parameters that
wouldgeneralizebetterwithinthesamemodel. Thestudyofdeeplearningisstilldriven
by empirical demonstrations. These are undeniably impressive, but they are not yet
matched by our understanding of deep learning mechanisms.
Problems
Problem20.1ConsidertheImageNetimageclassificationtaskinwhichtheinputimagescontain
224×224×3RGBvalues. ConsidercoarselyquantizingtheseinputsintotenbinsperRGBvalue
and training with ∼ 107 training examples. How many possible inputs are there per training
data point?
Problem 20.2 Consider figure 20.1. Why do you think that the algorithm fits the data faster
when the pixels are randomized relative to when the labels are randomized?
Problem 20.3 Figure 20.2 shows a non-stochastic fitting process with a fixed learning rate
successfully fitting random data. Does this imply that the loss function has no local minima?
Doesthisimplythatthefunctionisconvex? Justifyyouranswerandgiveacounter-exampleif
you think either statement is false.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.