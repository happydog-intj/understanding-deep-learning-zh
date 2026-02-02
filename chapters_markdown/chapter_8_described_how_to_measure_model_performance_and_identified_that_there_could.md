# Chapter 8: described how to measure model performance and identified that there could

*Pages: 152-174*

---

Chapter 9
Regularization
Chapter 8 described how to measure model performance and identified that there could
beasignificantperformancegapbetweenthetrainingandtestdata. Possiblereasonsfor
this discrepancy include: (i) the model describes statistical peculiarities of the training
data that are not representative of the true mapping from input to output (overfitting),
and (ii) the model is unconstrained in areas with no training examples, leading to sub-
optimal predictions.
This chapter discusses regularization techniques. These are a family of methods that
reduce the generalization gap between training and test performance. Strictly speaking,
regularization involves adding explicit terms to the loss function that favor certain pa-
rameter choices. However, in machine learning, this term is commonly used to refer to
any strategy that improves generalization.
We start by considering regularization in its strictest sense. Then we show how
the stochastic gradient descent algorithm itself favors certain solutions. This is known
as implicit regularization. Following this, we consider a set of heuristic methods that
improve test performance. These include early stopping, ensembling, dropout, label
smoothing, and transfer learning.
9.1 Explicit regularization
Consider fitting a model f[x,ϕ] with parameters ϕ using a training set {x ,y } of in-
i i
put/output pairs. We seek the parameters ϕˆ that minimize the loss function L[ϕ]:
(cid:2) (cid:3)
ϕˆ = argmin L[ϕ]
ϕ
" #
XI
= argmin ℓ [x ,y ] , (9.1)
i i i
ϕ
i=1
where the individual terms ℓ [x ,y ] measure the mismatch between the network pre-
i i i
dictions f[x ,ϕ] and output targets y for each training pair. To bias this minimization
i i
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

9.1 Explicit regularization 139
Figure 9.1 Explicit regularization. a) Loss function for Gabor model (see sec-
tion6.1.2). Cyancirclesrepresentlocalminima. Graycirclerepresentstheglobal
minimum. b)Theregularizationtermfavorsparametersclosetothecenterofthe
plot by adding an increasing penalty as we move away from this point. c) The
final loss function is the sum of the original loss function plus the regularization
term. This surface has fewer local minima, and the global minimum has moved
to a different position (arrow shows change).
toward certain solutions, we include an additional term:
" #
XI
ϕˆ =argmin ℓ [x ,y ]+λ·g[ϕ] , (9.2)
i i i
ϕ
i=1
where g[ϕ] is a function that returns a scalar which takes larger values when the pa-
rameters are less preferred. The term λ is a positive scalar that controls the relative
contribution of the original loss function and the regularization term. The minima of
the regularized loss function usually differ from those in the original, so the training
procedure converges to different parameter values (figure 9.1).
9.1.1 Probabilistic interpretation
Regularization can be viewed from a probabilistic perspective. Section 5.1 shows how
loss functions are constructed from the maximum likelihood criterion:
" #
YI
ϕˆ =argmax Pr(y |x ,ϕ) . (9.3)
i i
ϕ
i=1
The regularization term can be considered as a prior Pr(ϕ) that represents knowledge
about the parameters before we observe the data and we now have the maximum a
posteriori or MAP criterion:
Draft: please send errata to udlbookmail@gmail.com.

140 9 Regularization
" #
YI
ϕˆ =argmax Pr(y |x ,ϕ)Pr(ϕ) . (9.4)
i i
ϕ
i=1
Movingbacktothenegativelog-likelihoodlossfunctionbytakingthelogandmultiplying
by minus one, we see that λ·g[ϕ]=−log[Pr(ϕ)].
9.1.2 L2 regularization
This discussion has sidestepped the question of which solutions the regularization term
should penalize (or equivalently that the prior should favor). Since neural networks are
used in an extremely broad range of applications, these can only be very generic pref-
erences. The most commonly used regularization term is the L2 norm, which penalizes
the sum of the squares of the parameter values:
2 3
XI X
ϕˆ =argmin 4 ℓ [x ,y ]+λ ϕ25 , (9.5)
i i i j
ϕ
i=1 j
where j indexes the parameters. This is also referred to as Tikhonov regularization or
Problems9.1–9.2
ridge regression, or (when applied to matrices) Frobenius norm regularization.
For neural networks, L2 regularization is usually applied to the weights but not
the biases and is hence referred to as a weight decay term. The effect is to encourage
smaller weights, so the output function is smoother. To see this, consider that the
output prediction is a weighted sum of the activations at the last hidden layer. If the
Notebook9.1
weights have a smaller magnitude, the output will vary less. The same logic applies to
L2regularization
the computation of the pre-activations at the last hidden layer and so on, progressing
backward through the network. In the limit, if we forced all the weights to be zero, the
network would produce a constant output determined by the final bias parameter.
Figure9.2showstheeffectoffittingthesimplifiednetworkfromfigure8.4withweight
decay and different values of the regularization coeﬀicient λ. When λ is small, it has
little effect. However, as λ increases, the fit to the data becomes less accurate, and the
function becomes smoother. This might improve the test performance for two reasons:
• If the network is overfitting, then adding the regularization term means that the
network must trade off slavish adherence to the data against the desire to be
smooth. Onewaytothinkaboutthisisthattheerrorduetovariancereduces(the
model no longer needs to pass through every data point) at the cost of increased
bias (the model can only describe smooth functions).
• When the network is over-parameterized, some of the extra model capacity de-
scribes areas with no training data. Here, the regularization term will favor func-
tions that smoothly interpolate between the nearby points. This is reasonable
behavior in the absence of knowledge about the true function.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

9.2 Implicit regularization 141
Figure 9.2 L2 regularization in simplified network with 14 hidden units (see fig-
ure8.4). a–f)Fittedfunctionsasweincreasetheregularizationcoeﬀicientλ. The
black curve is the true function, the orange circles are the noisy training data,
andthecyancurveisthefittedmodel. Forsmallλ(panelsa–b),thefittedfunc-
tionpassesexactlythroughthedatapoints. Forintermediateλ(panelsc–d),the
function is smoother and more similar to the ground truth. For large λ (panels
e–f),theregularizationtermoverpowersthelikelihoodterm,sothefittedfunction
is too smooth and the overall fit is worse.
9.2 Implicit regularization
An intriguing recent finding is that neither gradient descent nor stochastic gradient
descent moves neutrally to the minimum of the loss function; each exhibits a preference
for some solutions over others. This is known as implicit regularization.
9.2.1 Implicit regularization in gradient descent
Consider a continuous version of gradient descent where the step size is infinitesimal.
The change in parameters ϕ will be governed by the differential equation:
dϕ ∂L
=− . (9.6)
dt ∂ϕ
Draft: please send errata to udlbookmail@gmail.com.

142 9 Regularization
Figure9.3Implicitregularizationingradientdescent. a)Lossfunctionwithfamily
ofglobalminimaonhorizontallineϕ =0.61. Dashedbluelineshowscontinuous
1
gradient descent path starting in bottom-left. Cyan trajectory shows discrete
gradient descent with step size 0.1 (first few steps shown explicitly as arrows).
Thefinitestepsizecausesthepathstodivergeandreachadifferentfinalposition.
b) This disparity can be approximated by adding a regularization term to the
continuous gradient descent loss function that penalizes the squared gradient
magnitude. c) After adding this term, the continuous gradient descent path
converges to the same place that the discrete one did on the original function.
Gradient descent approximates this process with a series of discrete steps of size α:
∂L[ϕ ]
ϕ =ϕ −α t , (9.7)
t+1 t ∂ϕ
The discretization causes a deviation from the continuous path (figure 9.3).
ThisdeviationcanbeunderstoodbyderivingamodifiedlosstermL˜ forthecontinu-
ouscasethatarrivesatthesameplaceasthediscretizedversionontheoriginallossL. It
can be shown (see notes “Implicit regularization in gradient descent” at end of chapter)
that this modified loss is:
(cid:13) (cid:13)
L˜ GD [ϕ]=L[ϕ]+ α 4 (cid:13) (cid:13) (cid:13) ∂ ∂ ϕ L (cid:13) (cid:13) (cid:13) 2 . (9.8)
In other words, the discrete trajectory is repelled from places where the gradient norm
is large (the surface is steep). This doesn’t change the position of the minima where the
gradients are zero anyway. However, it changes the effective loss function elsewhere and
modifiestheoptimizationtrajectory,whichpotentiallyconvergestoadifferentminimum.
Implicit regularization due to gradient descent may be responsible for the observation
that full batch gradient descent generalizes better with larger step sizes (figure 9.5a).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

9.3 Heuristics to improve performance 143
9.2.2 Implicit regularization in stochastic gradient descent
Asimilaranalysiscanbeappliedtostochasticgradientdescent. Nowweseekamodified
loss function such that the continuous version reaches the same place as the average of
the possible random SGD updates. This can be shown to be:
(cid:13) (cid:13)
L˜ SGD [ϕ] = L˜ GD [ϕ]+ 4 α B XB (cid:13) (cid:13) (cid:13) ∂ ∂ L ϕ b − ∂ ∂ ϕ L (cid:13) (cid:13) (cid:13) 2
b=1
(cid:13) (cid:13) (cid:13) (cid:13)
= L[ϕ]+ α (cid:13) (cid:13) (cid:13) ∂L (cid:13) (cid:13) (cid:13) 2 + α XB (cid:13) (cid:13) (cid:13) ∂L b − ∂L (cid:13) (cid:13) (cid:13) 2 . (9.9)
4 ∂ϕ 4B ∂ϕ ∂ϕ
b=1
Here, L is the loss for the bth of the B batches in an epoch, and both L and L now
b b
represent the means of the I individual losses in the full dataset and the |B| individual
losses in the batch, respectively:
XI X
1 1
L= ℓ [x ,y ] and L = ℓ [x ,y ]. (9.10)
I i i i b |B| i i i
i=1 i∈B b
Equation 9.9 reveals an extra regularization term, which corresponds to the variance
of the gradients of the batch losses L . In other words, SGD implicitly favors places
b
wherethegradientsarestable(whereallthebatchesagreeontheslope). Oncemore,this
modifies the trajectory of the optimization process (figure 9.4) but does not necessarily
change the position of the global minimum; if the model is over-parameterized, then it
may fit all the training data exactly, so each of these gradient terms will be zero at the
global minimum.
SGD generalizes better than gradient descent, and smaller batch sizes generally per-
form better than larger ones (figure 9.5b). One possible explanation is that the inherent
randomness allows the algorithm to reach different parts of the loss function. However,
Notebook9.2
it’s also possible that some or all of this performance increase is due to implicit regular-
Implicit
ization; this encourages solutions where all the data fits well (so the batch variance is regularization
small)ratherthansolutionswheresomeofthedatafitextremelywellandotherdataless
well (perhaps with the same overall loss, but with larger batch variance). The former
solutions are likely to generalize better.
9.3 Heuristics to improve performance
We’ve seen that explicit regularization encourages the training algorithm to find a good
solutionbyaddingextratermstothelossfunction. Thisalsooccursimplicitlyasanun-
intended (but seemingly helpful) byproduct of stochastic gradient descent. This section
describes other heuristic methods used to improve generalization.
Draft: please send errata to udlbookmail@gmail.com.

144 9 Regularization
Figure9.4Implicitregularizationforstochasticgradientdescent. a)Originalloss
functionforGabormodel(section6.1.2). Bluepointrepresentsglobalminimum.
b) Implicit regularization term from gradient descent penalizes the squared gra-
dient magnitude. c) Additional implicit regularization from stochastic gradient
descent penalizes the variance of the batch gradients. d) Modified loss function
(sum of original loss plus two implicit regularization components). Blue point
representsglobalminimumwhichmaynowbeinadifferentplacefrompanel(a).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

9.3 Heuristics to improve performance 145
Figure 9.5 Effect of learning rate (LR) and batch size for 4000 training and
4000 test examples from MNIST-1D (see figure 8.1) for a neural network with
two hidden layers. a) Performance is better for large learning rates than for
intermediateorsmallones. Ineachcase,thenumberofiterationsis6000/LR,so
each solution has the opportunity to move the same distance. b) Performance is
superiorforsmallerbatchsizes. Ineachcase,thenumberofiterationswaschosen
so that the training data were memorized at roughly the same model capacity.
9.3.1 Early stopping
Early stopping refers to stopping the training procedure before it has fully converged.
This can reduce overfitting if the model has already captured the coarse shape of the
underlying function but has not yet had time to overfit to the noise (figure 9.6). One
way of thinking about this is that since the weights are initialized to small values (see
section7.5),theysimplydon’thavetimetobecomelarge,soearlystoppinghasasimilar
effect to explicit L2 regularization. A different view is that early stopping reduces the
effectivemodelcomplexity. Hence,wemovebackdownthebias/variancetrade-offcurve
from the critical region, and performance improves (see figures 8.9 and 8.10).
Earlystoppinghasasinglehyperparameter,thenumberofstepsafterwhichlearning
is terminated. As usual, this is chosen empirically using a validation set (section 8.5).
However, for early stopping, the hyperparameter can be selected without the need to
train multiple models. The model is trained once, the performance on the validation set
is monitored every T iterations, and the associated parameters are stored. The stored
parameters where the validation performance was best are selected.
9.3.2 Ensembling
Another approach to reducing the generalization gap between training and test data is
to build several models and average their predictions. A group of such models is known
Draft: please send errata to udlbookmail@gmail.com.

146 9 Regularization
Figure 9.6 Early stopping. a) Simplified shallow network model with 14 linear
regions (figure 8.4) is initialized randomly (cyan curve) and trained with SGD
using a batch size of five and a learning rate of 0.05. b–d) As training proceeds,
thefunctionfirstcapturesthecoarsestructureofthetruefunction(blackcurve)
before e–f) overfitting to the noisy training data (orange points). Although the
traininglosscontinuestodecreasethroughoutthisprocess,thelearnedmodelsin
panels(c)and(d)areclosesttothetrueunderlyingfunction. Theywillgeneralize
better on average to test data than those in panels (e) or (f).
asanensemble. Thistechniquereliablyimprovestestperformanceatthecostoftraining
and storing multiple models and performing inference multiple times.
The models can be combined by taking the mean of the outputs (for regression
problems) or the mean of the pre-softmax activations (for classification problems). The
assumption is that model errors are independent and will cancel out. Alternatively,
we can take the median of the outputs (for regression problems) or the most frequent
predicted class (for classification problems) to make the predictions more robust.
Onewaytotraindifferentmodelsisjusttousedifferentrandominitializations. This
may help in regions of input space far from the training data. Here, the fitted function
Notebook9.3
is relatively unconstrained, and different models may produce different predictions, so
Ensembling
the average of several models may generalize better than any single model.
A second approach is to generate several different datasets by re-sampling the train-
ing data with replacement and training a different model from each. This is known as
bootstrap aggregating or bagging for short (figure 9.7). It has the effect of smoothing
out the data; if a data point is not present in one training set, the model will interpo-
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

9.3 Heuristics to improve performance 147
Figure 9.7 Ensemble methods. a) Fitting a single model (gray curve) to the
entiredataset(orangepoints). b–e)Fourmodelscreatedbyre-samplingthedata
with replacement (bagging) four times (size of orange point indicates number of
timesthedatapointwasre-sampled). f)Whenweaveragethepredictionsofthis
ensemble, the result (cyan curve) is smoother than the result from panel (a) for
the full dataset (gray curve) and will probably generalize better.
late from nearby points; hence, if that point was an outlier, the fitted function will be
more moderate in this region. Other approaches include training models with different
hyperparameters or training completely different families of models.
9.3.3 Dropout
Dropoutclampsarandomsubset(typically50%)ofhiddenunitstozeroateachiteration
ofSGD(figure9.8). Thismakesthenetworklessdependentonanygivenhiddenunitand
encourages the weights to have smaller magnitudes so that the change in the function
due to the presence or absence of any specific hidden unit is reduced.
This technique has the positive benefit that it can eliminate undesirable “kinks” in
the function that are far from the training data and don’t affect the loss. For example,
consider three hidden units that become active sequentially as we move along the curve
(figure9.9a). Thefirsthiddenunitcausesalargeincreaseintheslope. Asecondhidden
Draft: please send errata to udlbookmail@gmail.com.

148 9 Regularization
Figure 9.8 Dropout. a) Original network. b–d) At each training iteration, a
random subset of hidden units is clamped to zero (gray nodes). The result is
thattheincomingandoutgoingweightsfromtheseunitshavenoeffect,soweare
training with a slightly different network each time.
unit decreases the slope, so the function goes back down. Finally, the third unit cancels
out this decrease and returns the curve to its original trajectory. These three units
conspire to make an undesirable local change in the function. This will not change the
training loss but is unlikely to generalize well.
Whenseveralunitsconspireinthisway,eliminatingone(aswouldhappenindropout)
causes a considerable change to the output function in the half-space where that unit
wasactive(figure9.9b). Asubsequentgradientdescentstepwillattempttocompensate
for the change that this induces, and such dependencies will be eliminated over time.
The overall effect is that large unnecessary changes between training data points are
gradually removed even though they contribute nothing to the loss (figure 9.9).
At test time, we can run the network as usual with all the hidden units active;
however, the network now has more hidden units than it was trained with at any given
iteration,sowemultiplytheweightsbyoneminusthedropoutprobabilitytocompensate.
This is known as the weight scaling inference rule. A different approach to inference is
to use Monte Carlo dropout, in which we run the network multiple times with different
random subsets of units clamped to zero (as in training) and combine the results. This
iscloselyrelatedtoensemblinginthateveryrandomversionofthenetworkisadifferent
model; however, we do not have to train or store multiple networks here.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

9.3 Heuristics to improve performance 149
Figure 9.9 Dropout mechanism. a) An undesirable kink in the curve is caused
by a sequential increase in the slope, decrease in the slope (at circled joint), and
then another increase to return the curve to its original trajectory. Here we are
using full-batch gradient descent, and the model (from figure 8.4) fits the data
as well as possible, so further training won’t remove the kink. b) Consider what
happens if we remove the eighth hidden unit that produced the circled joint in
panel (a), as might happen using dropout. Without the decrease in the slope,
theright-handsideofthefunctiontakesanupwardstrajectory,andasubsequent
gradientdescentstepwillaimtocompensateforthischange. c)Curveafter2000
iterations of (i) randomly removing one of the three hidden units that cause the
kinkand(ii)performingagradientdescentstep. Thekinkdoesnotaffecttheloss
but is nonetheless removed by this approximation of the dropout mechanism.
9.3.4 Applying noise
Dropout can be interpreted as applying multiplicative Bernoulli noise to the network
activations. Thisleadstotheideaofapplyingnoisetootherpartsofthenetworkduring
training to make the final model more robust.
One option is to add noise to the input data; this smooths out the learned function
Problem9.3
(figure 9.10). For regression problems, it can be shown to be equivalent to adding a
regularizing term that penalizes the derivatives of the network’s output with respect to
itsinput. Anextremevariantisadversarialtraining,inwhichtheoptimizationalgorithm
actively searches for small perturbations of the input that cause large changes to the
output. These can be thought of as worst-case additive noise vectors.
A second possibility is to add noise to the weights. This encourages the network to
make sensible predictions even for small perturbations of the weights. The result is that
thetrainingconvergestolocalminimainthemiddleofwide,flatregions,wherechanging
the individual weights does not matter much.
Finally, we can perturb the labels. The maximum-likelihood criterion for multiclass
classification aims to predict the correct class with absolute certainty (equation 5.24).
To this end, the final network activations (i.e., before the softmax function) are pushed
to very large values for the correct class and very small values for the wrong classes.
We could discourage this overconfident behavior by assuming that a proportion ρ of
Draft: please send errata to udlbookmail@gmail.com.

150 9 Regularization
Figure 9.10 Adding noise to inputs. At each step of SGD, random noise with
variance σ2 is added to the batch data. a–c) Fitted model with different noise
x
levels (small dots represent ten samples). Adding more noise smooths out the
fitted function (cyan line).
the training labels are incorrect and belong with equal probability to the other classes.
Thiscouldbedonebyrandomlychangingthelabelsateachtrainingiteration. However,
the same end can be achieved by changing the loss function to minimize the cross-
entropy between the predicted distribution and a distribution where the true label has
Problem9.4 probability 1−ρ, and the other classes have equal probability. This is known as label
smoothing and improves generalization in diverse scenarios.
9.3.5 Bayesian inference
The maximum likelihood approach is generally overconfident; it selects the most likely
parameters during training and uses these to make predictions. However, many param-
eter values may be broadly compatible with the data and only slightly less likely. The
Bayesian approach treats the parameters as unknown variables and computes a distri-
AppendixC.1.4 butionPr(ϕ|{x ,y })overtheseparametersϕconditionedonthetrainingdata{x ,y }
Bayes’rule i i i i
using Bayes’ rule:
Q
I Pr(y |x ,ϕ)Pr(ϕ)
Pr(ϕ|{x ,y })= R Qi=1 i i , (9.11)
i i I Pr(y |x ,ϕ)Pr(ϕ)dϕ
i=1 i i
where Pr(ϕ) is the prior probability of the parameters, and the denominator is a nor-
malizing term. Hence, every parameter choice is assigned a probability (figure 9.11).
The prediction y for new input x is an infinite weighted sum (i.e., an integral) of the
predictions for each parameter set, where the weights are the associated probabilities:
Z
Pr(y|x,{x ,y })= Pr(y|x,ϕ)Pr(ϕ|{x ,y })dϕ. (9.12)
i i i i
This is effectively an infinite weighted ensemble, where the weight depends on (i) the
prior probability of the parameters and (ii) their agreement with the data.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

9.3 Heuristics to improve performance 151
Figure9.11Bayesianapproachforsimplifiednetworkmodel(seefigure8.4). The
parametersaretreatedasuncertain. TheposteriorprobabilityPr(ϕ|{x ,y })for
i i
a set of parameters is determined by their compatibility with the data {x ,y }
i i
and a prior distribution Pr(ϕ). a–c) Two sets of parameters (cyan and gray
curves) sampled from the posterior using normally distributed priors with mean
zero and three variances. When the prior variance σ2 is small, the parameters
ϕ
also tend to be small, and the functions smoother. d–f) Inference proceeds by
taking a weighted sum over all possible parameter values where the weights are
the posterior probabilities. This produces both a prediction of the mean (cyan
curves)andtheassociateduncertainty(grayregionistwostandarddeviations).
The Bayesian approach is elegant and can provide more robust predictions than
those that derive from maximum likelihood. Unfortunately, for complex models like
neural networks, there is no practical way to represent the full probability distribution
Notebook9.4
over the parameters or to integrate over it during the inference phase. Consequently, all
Bayesian
currentmethodsofthistypemakeapproximationsofsomekind,andtypicallytheseadd approach
considerable complexity to learning and inference.
9.3.6 Transfer learning and multi-task learning
Whentrainingdataarelimited,otherdatasetscanbeexploitedtoimproveperformance.
In transfer learning (figure 9.12a), the network is pre-trained to perform a related sec-
Draft: please send errata to udlbookmail@gmail.com.

152 9 Regularization
ondary task for which data are more plentiful. The resulting model is then adapted to
the original task. This is typically done by removing the last layer and adding one or
more layers that produce a suitable output. The main model may be fixed, and the new
layers trained for the original task, or we may fine-tune the entire model.
The principle is that the network will build a good internal representation of the
data from the secondary task, which can subsequently be exploited for the original task.
Equivalently, transfer learning can be viewed as initializing most of the parameters of
thefinalnetworkinasensiblepartofthespacethatislikelytoproduceagoodsolution.
Multi-tasklearning(figure9.12b)isarelatedtechniqueinwhichthenetworkistrained
to solve several problems concurrently. For example, the network might take an image
andsimultaneouslylearntosegmentthescene,estimatethepixel-wisedepth,andpredict
a caption describing the image. All of these tasks require some understanding of the
image and, when learned simultaneously, the model performance for each may improve.
9.3.7 Self-supervised learning
Theabovediscussionassumesthatwehaveplentifuldataforasecondarytaskordatafor
multiple tasks to be learned concurrently. If not, we can create large amounts of “free”
labeled data using self-supervised learning and use this for transfer learning. There are
two families of methods for self-supervised learning: generative and contrastive.
In generative self-supervised learning, part of each data example is masked, and the
secondary task is to predict the missing part (figure 9.12c). For example, we might use
a corpus of unlabeled images and a secondary task that aims to inpaint (fill in) missing
partsoftheimage(figure9.12c). Similarly,wemightusealargecorpusoftextandmask
somewords. Wetrainthenetworktopredictthemissingwordsandthenfine-tuneitfor
the actual language task we are interested in (see chapter 12).
Incontrastiveself-supervisedlearning,pairsofexampleswithcommonalitiesarecom-
pared to unrelated pairs. For images, the secondary task might be to identify whether a
pairofimagesaretransformedversionsofoneanotherorareunconnected. Fortext, the
secondarytaskmightbetodeterminewhethertwosentencesfollowedoneanotherinthe
original document. Sometimes, the precise relationship between a connected pair must
be identified (e.g., finding the relative position of two patches from the same image).
9.3.8 Augmentation
Transfer learning improves performance by exploiting a different dataset. Multi-task
learning improves performance using additional labels. A third option is to expand the
dataset. We can often transform each input data example in such a way that the label
stays the same. For example, we might aim to determine if there is a bird in an image
(figure 9.13). Here, we could rotate, flip, blur, or manipulate the color balance of the
image, and the label “bird” remains valid. Similarly, for tasks where the input is text,
Notebook9.5
we can substitute synonyms or translate to another language and back again. For tasks
Augmentation
where the input is audio, we can amplify or attenuate different frequency bands.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

9.3 Heuristics to improve performance 153
Figure 9.12 Transfer, multi-task, and self-supervised learning. a) Transfer learn-
ing is used when we have limited labeled data for the primary task (here depth
estimation)butplentifuldataforasecondarytask(heresegmentation). Wetrain
a model for the secondary task, remove the final layers, and replace them with
new layers appropriate to the primary task. We then train only the new layers
or fine-tune the entire network for the primary task. The network learns a good
internalrepresentationfromthesecondarytaskthatisthenexploitedforthepri-
marytask. b)Inmulti-tasklearning,wetrainamodeltoperformmultipletasks
simultaneously, hoping that performance on each will improve. c) In generative
self-supervised learning, we remove part of the data and train the network to
complete the missing information. Here, the task is to fill in (inpaint) a masked
portionoftheimage. Thispermitstransferlearningwhennolabelsareavailable.
Images from Cordts et al. (2016).
Draft: please send errata to udlbookmail@gmail.com.

154 9 Regularization
Figure 9.13 Data augmentation. For some problems, each data example can be
transformed to augment the dataset. a) Original image. b–h) Various geometric
andphotometrictransformationsofthisimage. Forimageclassification,allthese
images still have the same label, “bird.” Adapted from Wu et al. (2015a).
Generating extra training data in this way is known as data augmentation. The aim
is to teach the model to be indifferent to these irrelevant data transformations.
9.4 Summary
Explicit regularization involves adding an extra term to the loss function that changes
the position of the minimum. The term can be interpreted as a prior probability over
the parameters. Stochastic gradient descent with a finite step size does not neutrally
descend to the minimum of the loss function. This bias can be interpreted as adding
additional terms to the loss function, and this is known as implicit regularization.
Therearealsomanyheuristicsforimprovinggeneralization,includingearlystopping,
dropout, ensembling, the Bayesian approach, adding noise, transfer learning, multi-task
learning, and data augmentation. There are four main principles behind these methods
(figure9.14). Wecan(i)encouragethefunctiontobesmoother(e.g.,L2regularization),
(ii) increase the amount of data (e.g., data augmentation), (iii) combine models (e.g.,
ensembling), or (iv) search for wider minima (e.g., applying noise to network weights).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 155
Figure9.14Regularizationmethods. Theregularizationmethodsdiscussedinthis
chapteraimtoimprovegeneralizationbyoneoffourmechanisms. Somemethods
aimtomakethemodeledfunctionsmoother. Othermethodsincreasetheeffective
amountofdata. Thethirdgroupofmethodscombinemultiplemodelsandhence
mitigate against uncertainty in the fitting process. Finally, the fourth group of
methods encourages the training process to converge to a wide minimum where
smallerrorsintheestimatedparametersarelessimportant(seealsofigure20.11).
Anotherwaytoimprovegeneralizationistochoosethemodelarchitecturetosuitthe
task. For example, in image segmentation, we can share parameters within the model,
so we don’t need to independently learn what a tree looks like at every image location.
Chapters 10–13 consider architectural variations designed for different tasks.
Notes
An overview and taxonomy of regularization techniques in deep learning can be found in
Kukačka et al. (2017). Notably missing from the discussion in this chapter is BatchNorm
(Szegedy et al., 2016) and its variants, which are described in chapter 11.
Regularization: L2 regularization penalizes the sum of squares of the network weights. This
encourages the output function to change slowly (i.e., become smoother) and is the most used
regularizationterm. ItissometimesreferredtoasFrobeniusnormregularizationasitpenalizes
the Frobenius norms of the weight matrices. It is often also mistakenly referred to as “weight
decay,” although this is a separate technique devised by Hanson & Pratt (1988) in which the
parameters ϕ are updated as:
ϕ←−(1−λ ′ )ϕ−α ∂L , (9.13)
∂ϕ
Draft: please send errata to udlbookmail@gmail.com.

156 9 Regularization
where, as usual, α is the learning rate, and L is the loss. This is identical to gradient descent,
exceptthattheweightsarereducedbyafactorof1−λ′beforethegradientupdate. Forstandard
SGD,weightdecayisequivalenttoL2regularization(equation9.5)withcoeﬀicientλ=λ′/2α.
Problem9.5
However, for Adam, the learning rate α is different for each parameter, so L2 regularization
andweightdecaydiffer. Loshchilov&Hutter(2019)presentAdamW,whichmodifiesAdamto
implement weight decay correctly and show that this improves performance.
Other choices of vector norm encourage sparsity in the weights. The L0 regularization term
AppendixB.3.2
applies a fixed penalty for every non-zero weight. The effect is to “prune” the network. L0
Vectornorms
regularizationcanalsobeusedtoencouragegroupsparsity; thismightapplyafixedpenaltyif
anyoftheweightscontributingtoagivenhiddenunitarenon-zero. Iftheyareallzero,wecan
remove the unit, decreasing the model size and making inference faster.
Unfortunately,L0regularizationischallengingtoimplementsincethederivativeoftheregular-
ization term is not smooth, and more sophisticated fitting methods are required (see Louizos
et al., 2018). Somewhere between L2 and L0 regularization is L1 regularization or LASSO
(leastabsoluteshrinkageandselectionoperator),whichimposesapenaltyontheabsoluteval-
ues of the weights. L2 regularization somewhat discourages sparsity in that the derivative of
the squared penalty decreases as the weight becomes smaller, lowering the pressure to make it
smallerstill. L1regularizationdoesnothavethisdisadvantage,asthederivativeofthepenalty
is constant. This can produce sparser solutions than L2 regularization but is much easier to
Problem9.6
optimize than L0 regularization. Sometimes both L1 and L2 regularization terms are used,
which is termed an elastic net penalty (Zou & Hastie, 2005).
A different approach to regularization is to modify the gradients of the learning algorithm
withouteverexplicitlyformulatinganewlossfunction(e.g.,equation9.13). Thisapproachhas
been used to promote sparsity during backpropagation (Schwarz et al., 2021).
Theevidenceontheeffectivenessofexplicitregularizationismixed. Zhangetal.(2017a)showed
thatL2regularizationcontributeslittletogeneralization. IthasbeenproventhattheLipschitz
constant of the network (how fast the function can change as we modify the input) bounds
AppendixB.1.1
the generalization error (Bartlett et al., 2017; Neyshabur et al., 2018). However, the Lipschitz
Lipschitzconstant
constant depends on the product of the spectral norms of the weight matrices Ω , which are
k
only indirectly dependent on the magnitudes of the individual weights. Bartlett et al. (2017),
AppendixB.3.7 Neyshabur et al. (2018), and Yoshida & Miyato (2017) all add terms that indirectly encourage
Spectralnorm the spectral norms to be smaller. Gouk et al. (2021) take a different approach and develop an
algorithmthatconstrainstheLipschitzconstantofthenetworktobebelowaparticularvalue.
Implicit regularization in gradient descent: The gradient descent step is:
ϕ =ϕ +α·g[ϕ ], (9.14)
1 0 0
whereg[ϕ ]isthenegativeofthegradientofthelossfunction,andαisthestepsize. Asα→0,
0
the gradient descent process can be described by a differential equation:
dϕ
=g[ϕ]. (9.15)
dt
Fortypicalstepsizesα,thediscreteandcontinuousversionsconvergetodifferentsolutions. We
can use backward error analysis to find a correction g [ϕ] to the continuous version:
1
dϕ
≈g[ϕ]+αg [ϕ]+..., (9.16)
dt 1
so that it gives the same result as the discrete version.
ConsiderthefirsttwotermsofaTaylorexpansionofthemodifiedcontinuoussolutionϕaround
initial position ϕ :
0
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 157
(cid:12)
ϕ[α] ≈ ϕ+α dϕ + α2d2ϕ (cid:12) (cid:12) (cid:12)
dt 2 dt2
ϕ=ϕ0 (cid:18) (cid:19)(cid:12)
≈ ϕ+α(g[ϕ]+αg [ϕ])+ α2 ∂g[ϕ]dϕ +α ∂g 1 [ϕ]dϕ (cid:12) (cid:12) (cid:12)
1 2 ∂ϕ dt ∂ϕ dt
(cid:18) (cid:19)ϕ(cid:12)=ϕ0
= ϕ+α(g[ϕ]+αg [ϕ])+ α2 ∂g[ϕ] g[ϕ]+α ∂g 1 [ϕ] g[ϕ] (cid:12) (cid:12) (cid:12)
1 2 ∂ϕ ∂ϕ
(cid:18) (cid:19)(cid:12) ϕ=ϕ0
(cid:12)
≈ ϕ+αg[ϕ]+α2 g [ϕ]+ 1∂g[ϕ] g[ϕ] (cid:12) (cid:12) , (9.17)
1 2 ∂ϕ
ϕ=ϕ0
where in the second line, we have introduced the correction term (equation 9.16), and in the
final line, we have removed terms of greater order than α2.
Note that the first two terms on the right-hand side ϕ +αg[ϕ ] are the same as the discrete
0 0
update(equation9.14). Hence,tomakethecontinuousanddiscreteversionsarriveatthesame
place, the third term on the right-hand side must equal zero, allowing us to solve for g [ϕ]:
1
1∂g[ϕ]
g [ϕ]=− g[ϕ]. (9.18)
1 2 ∂ϕ
During training, the evolution function g[ϕ] is the negative of the gradient of the loss:
dϕ
≈ g[ϕ]+αg [ϕ]
dt 1
(cid:18) (cid:19)
∂L α ∂2L ∂L
= − − . (9.19)
∂ϕ 2 ∂ϕ2 ∂ϕ
This is equivalent to performing continuous gradient descent on the loss function:
(cid:13) (cid:13)
α (cid:13) (cid:13)∂L (cid:13) (cid:13) 2
L GD [ϕ]=L[ϕ]+ 4 (cid:13) ∂ϕ (cid:13) , (9.20)
because the right-hand side of equation 9.19 is the derivative of that in equation 9.20.
This formulation of implicit regularization was developed by Barrett & Dherin (2021) and
extended to stochastic gradient descent by Smith et al. (2021). Smith et al. (2020) and others
haveshownthatstochasticgradientdescentwithsmallormoderatebatchsizesoutperformsfull
batch gradient descent on the test set, and this may in part be due to implicit regularization.
Relatedly,Jastrzębskietal.(2021)andCohenetal.(2021)bothshowthatusingalargelearn-
ingratereducesthetendencyoftypicaloptimizationtrajectoriestomoveto“sharper”partsof
the loss function (i.e., where at least one direction has high curvature). This implicit regular-
ization effect of large learning rates can be approximated by penalizing the trace of the Fisher
Information Matrix, which is closely related to penalizing the gradient norm in equation 9.20
(Jastrzębski et al., 2021).
Early stopping: Bishop(1995)andSjöberg&Ljung(1995)arguedthatearlystoppinglimits
the effective solution space that the training procedure can explore; given that the weights are
initialized to small values, this leads to the idea that early stopping helps prevent the weights
from getting too large. Goodfellow et al. (2016) show that under a quadratic approximation
of the loss function with parameters initialized to zero, early stopping is equivalent to L2 reg-
ularization in gradient descent. The effective regularization weight λ is approximately 1/(τα)
where α is the learning rate, and τ is the early stopping time.
Draft: please send errata to udlbookmail@gmail.com.

158 9 Regularization
Ensembling: Ensembles can be trained using different random seeds (Lakshminarayanan
et al., 2017), hyperparameters (Wenzel et al., 2020b), or even entirely different families of
models. Themodelscanbecombinedbyaveragingtheirpredictions,weightingthepredictions,
or stacking (Wolpert, 1992), in which the results are combined using another machine learning
model. Lakshminarayanan et al. (2017) showed that averaging the output of independently
trained networks can improve accuracy, calibration, and robustness. Conversely, Frankle et al.
(2020) showed that if we average together the weights to make one model, the network fails.
Fort et al. (2019) compared ensembling solutions that resulted from different initializations
with ensembling solutions that were generated from the same original model. For example, in
the latter case, they consider exploring around the solution in a limited subspace to find other
AppendixB.3.6
goodnearbypoints. Theyfoundthatbothtechniquesprovidecomplementarybenefitsbutthat
Subspaces
genuine ensembling from different random starting points provides a bigger improvement.
Aneﬀicientwayofensemblingistocombinemodelsfromintermediatestagesoftraining. Tothis
end, Izmailov et al. (2018) introduce stochastic weight averaging, in which the model weights
are sampled at different time steps and averaged together. As the name suggests, snapshot
ensembles (Huang et al., 2017a) also store the models from different time steps and average
their predictions. The diversity of these models can be improved by cyclically increasing and
decreasing the learning rate. Garipov et al. (2018) observed that different minima of the loss
functionareoftenconnectedbyalow-energypath(i.e.,apathwithalowlosseverywherealong
it). Motivated by this observation, they developed a method that explores low-energy regions
around an initial solution to provide diverse models without retraining. This is known as fast
geometric ensembling. A review of ensembling methods can be found in Ganaie et al. (2022).
Dropout: DropoutwasfirstintroducedbyHintonetal.(2012b)andSrivastavaetal.(2014).
Dropout is applied at the level of hidden units. Dropping a hidden unit has the same effect
as temporarily setting all the incoming and outgoing weights and the bias to zero. Wan et al.
(2013)generalizeddropoutbyrandomlysettingindividualweightstozero. Gal&Ghahramani
(2016)andKendall&Gal(2017)proposedMonteCarlodropout,inwhichinferenceiscomputed
withseveraldropoutpatterns,andtheresultsareaveragedtogether. Gal&Ghahramani(2016)
argued that this could be interpreted as approximating Bayesian inference.
Dropout is equivalent to applying multiplicative Bernoulli noise to the hidden units. Similar
benefits derive from using other distributions, including the normal (Srivastava et al., 2014;
Shen et al., 2017), uniform (Shen et al., 2017), and beta distributions (Liu et al., 2019b).
Adding noise: Bishop (1995) and An (1996) added Gaussian noise to the network inputs to
improveperformance. Bishop(1995)showedthatthisisequivalenttoweightdecay. An(1996)
also investigated adding noise to the weights. DeVries & Taylor (2017a) added Gaussian noise
tothehiddenunits. Therandomized ReLU(Xuetal.,2015)appliesnoiseinadifferentwayby
making the activation functions stochastic.
Label smoothing: LabelsmoothingwasintroducedbySzegedyetal.(2016)forimageclassi-
ficationbuthassincebeenshowntobehelpfulinspeechrecognition(Chorowski&Jaitly,2017),
machine translation (Vaswani et al., 2017), and language modeling (Pereyra et al., 2017). The
precise mechanism by which label smoothing improves test performance isn’t well understood,
although Müller et al. (2019a) show that it improves the calibration of the predicted output
probabilities. A closely related technique is DisturbLabel (Xie et al., 2016), in which a certain
percentage of the labels in each batch are randomly switched at each training iteration.
Finding wider minima: Itisthoughtthatwiderminimageneralizebetter(seefigure20.11).
Here, the exact values of the weights are less important, so performance should be robust to
errorsintheirestimates. Oneofthereasonsthatapplyingnoisetopartsofthenetworkduring
training is effective is that it encourages the network to be indifferent to their exact values.
Chaudhari et al. (2019) developed a variant of SGD that biases the optimization toward flat
minima,whichtheycallentropy SGD.Theideaistoincorporatelocalentropyasaterminthe
loss function. In practice, this takes the form of one SGD-like update within another. Keskar
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 159
et al. (2017) showed that SGD finds wider minima as the batch size is reduced. This may be
because of the batch variance term that results from implicit regularization by SGD.
Ishida et al. (2020) use a technique named flooding, in which they intentionally prevent the
traininglossfrombecomingzero. Thisencouragesthesolutiontoperformarandomwalkover
the loss landscape and drift into a flatter area with better generalization.
Bayesian approaches: For some models, including the simplified neural network model in
figure 9.11, the Bayesian predictive distribution can be computed in closed form (see Bishop,
2006; Prince, 2012). For neural networks, the posterior distribution over the parameters can-
not be represented in closed form and must be approximated. The two main approaches are
variationalBayes(Hinton&vanCamp,1993;MacKay,1995;Barber&Bishop,1997;Blundell
et al., 2015), in which the posterior is approximated by a simpler tractable distribution, and
MarkovChainMonteCarlo(MCMC)methods,whichapproximatethedistributionbydrawing
a set of samples (Neal, 1995; Welling & Teh, 2011; Chen et al., 2014; Ma et al., 2015; Li et al.,
2016a). The generation of samples can be integrated into SGD, and this is known as stochas-
tic gradient MCMC (see Ma et al., 2015). It has recently been discovered that “cooling” the
posteriordistributionovertheparameters(makingitsharper)improvespredictionsfromthese
models(Wenzeletal.,2020a),butthisisnotcurrentlyfullyunderstood(seeNocietal.,2021).
Transfer learning: Transfer learning for visual tasks works extremely well (Sharif Razavian
etal.,2014)andhassupportedrapidprogressincomputervision,includingtheoriginalAlexNet
results(Krizhevskyetal.,2012). Transferlearninghasalsoimpactednaturallanguageprocess-
ing(NLP),wheremanymodelsarebasedonpre-trainedfeaturesfromtheBERTmodel(Devlin
et al., 2019). More information can be found in Zhuang et al. (2020) and Yang et al. (2020b).
Self-supervised learning: Self-supervised learning techniques for images have included in-
paintingmaskedimageregions(Pathaketal.,2016),predictingtherelativepositionofpatches
in an image (Doersch et al., 2015), re-arranging permuted image tiles back into their original
configuration (Noroozi & Favaro, 2016), colorizing grayscale images (Zhang et al., 2016b), and
transforming rotated images back to their original orientation (Gidaris et al., 2018). In Sim-
CLR (Chen et al., 2020c), a network is learned that maps versions of the same image that
have been photometrically and geometrically transformed to the same representation while re-
pelling versions of different images, with the goal of becoming indifferent to irrelevant image
transformations. Jing & Tian (2020) present a survey of self-supervised learning in images.
Self-supervisedlearninginNLPcanbebasedonpredictingmaskedwords(Devlinetal.,2019),
predicting the next word in a sentence (Radford et al., 2019; Brown et al., 2020), or predicting
whethertwosentencesfollowoneanother(Devlinetal.,2019). Inautomaticspeechrecognition,
the Wav2Vec model (Schneider et al., 2019) aims to distinguish an original audio sample from
one where 10ms of audio has been swapped out from elsewhere in the clip. Self-supervision
has also been applied to graph neural networks (chapter 13). Tasks include recovering masked
features(Youetal.,2020)andrecoveringtheadjacencystructureofthegraph(Kipf&Welling,
2016). Liu et al. (2023a) review self-supervised learning for graph models.
Data augmentation: Data augmentation for images dates back to at least LeCun et al.
(1998)andcontributedtothesuccessofAlexNet(Krizhevskyetal.,2012),inwhichthedataset
was increased by a factor of 2048. Image augmentation approaches include geometric transfor-
mations,changingormanipulatingthecolorspace,noiseinjection,andapplyingspatialfilters.
Moreelaboratetechniquesincluderandomlymixingimages(Inoue,2018;Summers&Dinneen,
2019), randomly erasing parts of the image (Zhong et al., 2020), style transfer (Jackson et al.,
2019), and randomly swapping image patches (Kang et al., 2017). In addition, many studies
haveusedgenerativeadversarialnetworksorGANs(seechapter15)toproducenovelbutplau-
sible data examples (e.g., Calimeri et al., 2017). In other cases, the data have been augmented
with adversarial examples (Goodfellow et al., 2015a), which are minor perturbations of the
training data that cause the example to be misclassified. A review of data augmentation for
images can be found in Shorten & Khoshgoftaar (2019).
Draft: please send errata to udlbookmail@gmail.com.

160 9 Regularization
Augmentationmethodsforacousticdataincludepitchshifting,timestretching,dynamicrange
compression, and adding random noise (e.g., Abeßer et al., 2017; Salamon & Bello, 2017; Xu
etal.,2015;Lasseck,2018),aswellasmixingdatapairs(Zhangetal.,2017c;Yunetal.,2019),
maskingfeatures(Parketal.,2019),andusingGANstogeneratenewdata(Munetal.,2017).
Augmentationforspeechdataincludesvocaltractlengthperturbation(Jaitly&Hinton,2013;
Kandaetal.,2013),styletransfer(Gales,1998;Ye&Young,2004),addingnoise(Hannunetal.,
2014), and synthesizing speech (Gales et al., 2009).
Augmentationmethodsfortextincludeaddingnoiseatacharacterlevelbyswitching,deleting,
and inserting letters (Belinkov & Bisk, 2018; Feng et al., 2020), or by generating adversarial
examples(Ebrahimietal.,2018),usingcommonspellingmistakes(Coulombe,2018),randomly
swapping or deleting words (Wei & Zou, 2019), using synonyms (Kolomiyets et al., 2011),
altering adjectives (Li et al., 2017c), passivization (Min et al., 2020), using generative models
tocreatenewdata(Qiuetal.,2020), and round-triptranslationtoanotherlanguageandback
(Aiken & Park, 2010). Augmentation methods for text are reviewed by Bayer et al. (2022).
Problems
Problem 9.1 Consider a model where the prior distribution over the parameters is a normal
distribution with mean zero and variance σ2 so that
ϕ
YJ
Pr(ϕ)= Norm [0,σ2], (9.21)
ϕj ϕ
j=1
Q
wherej indexesthemodelparameters. Wenowmaximize I Pr(y |x ,ϕ)Pr(ϕ). Showthat
i=1 i i
the associated loss function of this model is equivalent to L2 regularization.
Problem 9.2 How do the gradients of the loss function change when L2 regularization (equa-
tion 9.5) is added?
Problem 9.3∗ Consider a linear regression model y = ϕ +ϕ x with input x, output y, and
0 1
parameters ϕ and ϕ . Assume we have I training examples {x ,y } and use a least squares
0 1 i i
loss. Consider adding Gaussian noise with mean zero and variance σ2 to the inputs x at each
x i
training iteration. Derive an expression for the expected loss.
Problem 9.4∗ Derive the loss function for multiclass classification when we use label smooth-
ing so that the target probability distribution has 0.9 at the correct class and the remaining
probability mass of 0.1 is divided between the remaining D −1 classes.
o
Problem 9.5 Show that the weight decay parameter update with decay rate λ:
∂L
ϕ←−(1−λ)ϕ−α , (9.22)
∂ϕ
ontheoriginallossfunctionL[ϕ]isequivalenttoastandardgradientupdateusingL2regular-
ization so that the modified loss function L˜[ϕ] is:
X
λ
L˜[ϕ]=L[ϕ]+ ϕ2, (9.23)
2α k
k
where ϕ are the parameters, and α is the learning rate.
Problem 9.6 Consider a model with parameters ϕ = [ϕ ,ϕ ]T. Draw the L0, L1, and L1
0 1 P2
regularizationtermsinasimilarformtofigure9.1b. TheLP regularizationtermis D |ϕ |P.
d=1 d
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.