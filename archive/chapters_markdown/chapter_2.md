# Chapter 2

*Pages: 31-35*

---

Chapter 2
Supervised learning
A supervised learning model defines a mapping from one or more inputs to one or more
outputs. For example, the input might be the age and mileage of a second-hand Toyota
Prius, and the output might be the estimated value of the car in dollars.
The model is just a mathematical function (i.e., an equation); when the inputs are
passed through this function, it computes the output, and this is termed inference. The
model equation also contains parameters. Different parameter values change the out-
comeofthecomputation;themodelequationdescribesafamilyofpossiblerelationships
between inputs and outputs, and the parameters specify the particular relationship.
Whenwetrainorlearnamodel,wefindparametersthatdescribethetruerelationship
between inputs and outputs. A learning algorithm takes a training set of input/output
pairs and manipulates the parameters until the inputs predict their corresponding out-
putsascloselyaspossible. Ifthemodelworkswellforthesetrainingpairs,thenwehope
it will make good predictions for new inputs where the true output is unknown.
Thegoalofthischapteristoexpandontheseideas. First,wedescribethisframework
more formally and introduce some notation. Then we work through a simple example
in which we use a straight line to describe the relationship between input and output.
This linear model is both familiar and easy to visualize, but nevertheless illustrates all
the main ideas of supervised learning.
2.1 Supervised learning overview
In supervised learning, we aim to build a model that takes an input x and outputs a
prediction y. For simplicity, we assume that both the input x and output y are vectors
ofapredeterminedandfixedsizeandthattheelementsofeachvectorarealwaysordered
in the same way; in the Prius example above, the input x would always contain the age
ofthecarandthenthemileage, inthatorder. Thisistermedstructuredortabulardata.
To make the prediction, we need a model f[•] that takes input x and returns y, so:
y=f[x]. (2.1)
Draft: please send errata to udlbookmail@gmail.com.

18 2 Supervised learning
When we compute the prediction y from the input x, we call this inference.
The model is just a mathematical equation with a fixed form. It represents a family
ofdifferentrelationsbetweentheinputandtheoutput. Themodelalsocontainsparam-
eters ϕ. The choice of parameters determines the particular relation between input and
output, so we should really write:
y=f[x,ϕ]. (2.2)
When we talk about learning or training a model, we mean that we attempt to find
parameters ϕ that make sensible output predictions from the input. We learn these
parametersusingatrainingdatasetofI pairsofinputandoutputexamples{x ,y }. We
i i
aimtoselectparametersthatmapeachtraininginputtoitsassociatedoutputasclosely
as possible. We quantify the degree of mismatch in this mapping with the loss L. This
is a scalar value that summarizes how poorly the model predicts the training outputs
from their corresponding inputs for parameters ϕ.
We can treat the loss as a function L[ϕ] of these parameters. When we train the
model, we are seeking parameters ϕˆ that minimize this loss function:1
AppendixA
h i
Argminfunction
ϕˆ =argmin L[ϕ] . (2.3)
ϕ
If the loss is small after this minimization, we have found model parameters that accu-
rately predict the training outputs y from the training inputs x .
i i
After training a model, we must now assess its performance; we run the model on
separatetest datatoseehowwellitgeneralizestoexamplesthatitdidn’tobserveduring
training. If the performance is adequate, then we are ready to deploy the model.
2.2 Linear regression example
Let’s now make these ideas concrete with a simple example. We consider a model y =
f[x,ϕ] that predicts a single output y from a single input x. Then we develop a loss
function, and finally, we discuss model training.
2.2.1 1D linear regression model
A 1D linear regression model describes the relationship between input x and output y
as a straight line:
y = f[x,ϕ]
= ϕ +ϕ x. (2.4)
0 1
1More properly, the loss function also depends on the training data {xi,yi }, so we should
writeL[{xi,yi },ϕ],butthisisrathercumbersome.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

2.2 Linear regression example 19
Figure 2.1 Linear regression model. For
agivenchoiceofparametersϕ=[ϕ ,ϕ ],
0 1
themodelmakesapredictionfortheout-
put(y-axis)basedontheinput(x-axis).
Different choices for the y-intercept ϕ
0
andtheslopeϕ changethesepredictions
1
(cyan, orange, and gray lines). The lin-
ear regression model (equation 2.4) de-
fines a family of input/output relations
(lines)andtheparametersdeterminethe
member of the family (the particular
line). (Interactive figure)
Thismodelhastwoparametersϕ=[ϕ ,ϕ ],whereϕ isthey-interceptofthelineandϕ
0 1 0 1
is the slope. Different choices for the y-intercept and slope result in different relations
between input and output (figure 2.1). Hence, equation 2.4 defines a family of possible
input-output relations (all possible lines), and the choice of parameters determines the
member of this family (the particular line).
2.2.2 Loss
Forthismodel,thetrainingdataset(figure2.2a)consistsofI input/outputpairs{x ,y }.
i i
Figures 2.2b–d show three lines defined by three sets of parameters. The green line
in figure 2.2d describes the data more accurately than the other two since it is much
closer to the data points. However, we need a principled approach for deciding which
parameters ϕ are better than others. To this end, we assign a numerical value to each
choice of parameters that quantifies the degree of mismatch between the model and the
data. We term this value the loss; a lower loss means a better fit.
The mismatch is captured by the deviation between the model predictions f[x ,ϕ]
i
(heightofthelineatx )andthegroundtruthoutputsy . Thesedeviationsaredepicted
i i
asorangedashedlinesinfigures2.2b–d. Wequantifythetotalmismatch,training error,
or loss as the sum of the squares of these deviations for all I training pairs:
XI
L[ϕ] = (f[x ,ϕ]−y )2
i i
i=1
XI
= (ϕ +ϕ x −y )2. (2.5)
0 1 i i
i=1
Sincethebestparametersminimizethisexpression,wecallthisaleast-squaresloss. The
squaring operation means that the direction of the deviation (i.e., whether the line is
Draft: please send errata to udlbookmail@gmail.com.

20 2 Supervised learning
Figure 2.2Linearregressiontrainingdata,model,andloss. a)Thetrainingdata
(orange points) consist of I = 12 input/output pairs {x ,y }. b–d) Each panel
i i
shows the linear regression model with different parameters. Depending on the
choiceofy-interceptandslopeparametersϕ=[ϕ ,ϕ ],themodelerrors(orange
0 1
dashed lines) may be larger or smaller. The loss L is the sum of the squares
of these errors. The parameters that define the lines in panels (b) and (c) have
large losses L=7.07 and L=10.28, respectively because the models fit badly.
The loss L=0.20 in panel (d) is smaller because the model fits well; in fact, this
has the smallest loss of all possible lines, so these are the optimal parameters.
(Interactive figure)
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

2.2 Linear regression example 21
Figure2.3Lossfunctionforlinearregressionmodelwiththedatasetinfigure2.2a.
a)Eachcombinationofparametersϕ=[ϕ ,ϕ ]hasanassociatedloss. Theresult-
0 1
ing loss function L[ϕ] can be visualized as a surface. The three circles represent
the lines from figure 2.2b–d. b) The loss can also be visualized as a heatmap,
where brighter regions represent larger losses; here we are looking straight down
atthesurfacein(a)fromaboveandgrayellipsesrepresentisocontours. Thebest
fittingline(figure2.2d)hastheparameterswiththesmallestloss(greencircle).
above or below the data) is unimportant. There are also theoretical reasons for this
choice which we return to in chapter 5.
The loss L is a function of the parameters ϕ; it will be larger when the model fit is
Notebook2.1
poor (figure 2.2b,c) and smaller when it is good (figure 2.2d). Considered in this light,
Supervised
we term L[ϕ] the loss function or cost function. The goal is to find the parameters ϕˆ learning
that minimize this quantity:
h i
ϕˆ = argmin L[ϕ]
ϕ
" #
XI
= argmin (f[x ,ϕ]−y )2
i i
ϕ
"i=1 #
XI
= argmin (ϕ +ϕ x −y )2 . (2.6)
0 1 i i
ϕ
i=1
There are only two parameters (the y-intercept ϕ and slope ϕ ), so we can calculate
0 1 Problems2.1–2.2
the loss for every combination of values and visualize the loss function as a surface
(figure 2.3). The “best” parameters are at the minimum of this surface.
Draft: please send errata to udlbookmail@gmail.com.