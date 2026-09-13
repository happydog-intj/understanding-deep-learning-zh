# Chapter 8: discusses how to measure model performance. Chapter 9 considers regular-

*Pages: 36-38*

---

22 2 Supervised learning
2.2.3 Training
Theprocessoffindingparametersthatminimizethelossistermedmodelfitting,training,
or learning. The basic method is to choose the initial parameters randomly and then
improvethemby“walkingdown”thelossfunctionuntilwereachthebottom(figure2.4).
One way to do this is to measure the gradient of the surface at the current position and
take a step in the direction that is most steeply downhill. Then we repeat this process
until the gradient is flat and we can improve no further.2
2.2.4 Testing
Having trained the model, we want to know how it will perform in the real world. We
do this by computing the loss on a separate set of test data. The degree to which the
prediction accuracy generalizes to the test data depends in part on how representative
andcompletethetrainingdatais. However,italsodependsonhowexpressivethemodel
is. Asimplemodellikealinemightnotbeabletocapturethetruerelationshipbetween
input and output. This is known as underfitting. Conversely, a very expressive model
may describe statistical peculiarities of the training data that are atypical and lead to
unusual predictions. This is known as overfitting.
2.3 Summary
A supervised learning model is a function y=f[x,ϕ] that relates inputs x to outputs y.
The particular relationship is determined by parameters ϕ. To train the model, we
definealossfunctionL[ϕ]overatrainingdataset{x ,y }. Thisquantifiesthemismatch
i i
between the model predictions f[x ,ϕ] and observed outputs y as a function of the
i i
parameters ϕ. Then we search for the parameters that minimize the loss. We evaluate
the model on a different set of test data to see how well it generalizes to new inputs.
Chapters 3–9 expand on these ideas. First, we tackle the model itself; 1D linear
regressionhastheobviousdrawbackthatitcanonlydescribetherelationshipbetweenthe
inputandoutputasastraightline. Shallowneuralnetworks(chapter3)areonlyslightly
more complex than linear regression but describe a much larger family of input/output
relationships. Deep neural networks (chapter 4) are just as expressive but can describe
complex functions with fewer parameters and work better in practice.
Chapter 5 investigates loss functions for different tasks and reveals the theoretical
underpinnings of the least-squares loss. Chapters 6 and 7 discuss the training process.
Chapter 8 discusses how to measure model performance. Chapter 9 considers regular-
ization techniques, which aim to improve that performance.
2Thisiterativeapproachisnotactuallynecessaryforthelinearregressionmodel. Here,it’spossible
to find closed-form expressions for the parameters. However, this gradient descent approach works for
morecomplexmodelswherethereisnoclosed-formsolutionandwheretherearetoomanyparameters
toevaluatethelossforeverycombinationofvalues.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 23
Figure2.4Linearregressiontraining. Thegoalistofindthey-interceptandslope
parameters that correspond to the smallest loss. a) Iterative training algorithms
initializetheparametersrandomlyandthenimprovethemby“walkingdownhill”
untilnofurtherimprovementcanbemade. Here,westartatposition0andmove
a certain distance downhill (perpendicular to the contours) to position 1. Then
we re-calculate the downhill direction and move to position 2. Eventually, we
reachtheminimumofthefunction(position4). b)Eachposition0–4frompanel
(a) corresponds to a different y-intercept and slope and so represents a different
line. Asthelossdecreases,thelinesfitthedatamoreclosely. (Interactivefigure)
Notes
Loss functions vs. cost functions: Inmuchofmachinelearningandinthisbook,theterms
lossfunctionandcostfunctionareusedinterchangeably. However,moreproperly,alossfunction
istheindividualtermassociatedwithadatapoint(i.e.,eachofthesquaredtermsontheright-
handsideofequation 2.5), andthecostfunction istheoverallquantitythat isminimized(i.e.,
the entire right-hand side of equation 2.5). A cost function can contain additional terms that
are not associated with individual data points (see section 9.1). More generally, an objective
function is any function that is to be maximized or minimized.
Generative vs. discriminative models: Themodelsy=f[x,ϕ]inthischapterarediscrim-
inative models. Thesemakeanoutputpredictionyfromreal-worldmeasurementsx. Another
Problem2.3
approach is to build a generative model x = g[y,ϕ], in which the real-world measurements x
are computed as a function of the output y.
The generative approach has the disadvantage that it doesn’t directly predict y. To perform
inference, we must invert the generative equation as y = g−1[x,ϕ], and this may be diﬀicult.
However,generativemodelshavetheadvantagethatwecanbuildinpriorknowledgeabouthow
the data were created. For example, if we wanted to predict the 3D position and orientation y
Draft: please send errata to udlbookmail@gmail.com.

24 2 Supervised learning
ofacarinanimagex,thenwecouldbuildknowledgeaboutcarshape,3Dgeometry,andlight
transport into the function x=g[y,ϕ].
This seems like a good idea, but in fact, discriminative models dominate modern machine
learning; the advantage gained from exploiting prior knowledge in generative models is usually
trumped by learning very flexible discriminative models with large amounts of training data.
Problems
Problem2.1Towalk“downhill”onthelossfunction(equation2.5),wemeasureitsgradientwith
respecttotheparametersϕ andϕ . Calculateexpressionsfortheslopes∂L/∂ϕ and∂L/∂ϕ .
0 1 0 1
Problem 2.2 Showthatwecanfindtheminimumofthelossfunctioninclosedformbysetting
theexpressionforthederivativesfromproblem2.1tozeroandsolvingforϕ andϕ . Notethat
0 1
this works for linear regression but not for more complex models; this is why we use iterative
model fitting methods like gradient descent (figure 2.4).
Problem 2.3∗ Consider reformulating linear regression as a generative model, so we have x =
g[y,ϕ] = ϕ +ϕ y. What is the new loss function? Find an expression for the inverse func-
0 1
tion y = g−1[x,ϕ] that we would use to perform inference. Will this model make the same
predictions as the discriminative version for a given training dataset {x ,y }? One way to es-
i i
tablish this is to write code that fits a line to three data points using both methods and see if
the result is the same.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.