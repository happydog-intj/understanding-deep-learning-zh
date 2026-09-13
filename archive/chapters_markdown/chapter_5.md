# Chapter 5

*Pages: 70-73*

---

Chapter 5
Loss functions
The last three chapters described linear regression, shallow neural networks, and deep
neural networks. Each represents a family of functions that map input to output, where
the particular member of the family is determined by the model parameters ϕ. When
we train these models, we seek the parameters that produce the best possible mapping
frominputtooutputforthetaskweareconsidering. Thischapterdefineswhatismeant
by the “best possible” mapping.
That definition requires a training dataset {x ,y } of input/output pairs. A loss
i i
function or cost function L[ϕ] returns a single number that describes the mismatch
betweenthemodelpredictionsf[x ,ϕ]andtheircorrespondingground-truthoutputsy .
i i
During training, we seek parameter values ϕ that minimize the loss and hence map the
training inputs to the outputs as closely as possible. We saw one example of a loss
function in chapter 2; the least squares loss function is suitable for univariate regression
problemsforwhichthetargetisarealnumbery ∈R. Itcomputesthesumofthesquares
AppendixA
of the deviations between the model predictions f[x ,ϕ] and the true values y .
Sets i i
This chapter provides a framework that both justifies the choice of the least squares
criterionforreal-valuedoutputsandallowsustobuildlossfunctionsforotherprediction
types. We consider binary classification, where the prediction y ∈ {0,1} is one of two
categories, multiclass classification, where the prediction y ∈ {1,2,...,K} is one of K
categories, and more complex cases. In the following two chapters, we address model
training,wherethegoalistofindtheparametervaluesthatminimizetheselossfunctions.
5.1 Maximum likelihood
In this section, we develop a recipe for constructing loss functions. Consider a model
f[x,ϕ] with parameters ϕ that computes an output from input x. Until now, we have
AppendixC.1.3
implied that the model directly computes a prediction y. We now shift perspective and
Conditional
probability consider the model as computing a conditional probability distribution Pr(y|x) over
possible outputs y given input x. The loss encourages each training output y to have
i
a high probability under the distribution Pr(y |x ) computed from the corresponding
i i
input x (figure 5.1).
i
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

5.1 Maximum likelihood 57
Figure 5.1 Predicting distributions over outputs. a) Regression task, where the
goalistopredictareal-valuedoutputy fromtheinputxbasedontrainingdata
{x ,y }(orangepoints). Foreachinputvaluex,themachinelearningmodelpre-
i i
dictsadistributionPr(y|x)overtheoutputy∈R(cyancurvesshowdistributions
for x=2.0 and x=7.0). Minimizing the loss function corresponds to maximizing
the probability of the training outputs y under the distribution predicted from
i
the corresponding inputs x . b) To predict discrete classes y ∈ {1,2,3,4} in a
i
classification task, we use a discrete probability distribution, so the model pre-
dicts a different histogram over the four possible values of y for each value of
i
x . c) To predict counts y ∈ {0,1,2,...} and d) direction y ∈ (−π,π], we use
i
distributions defined over positive integers and circular domains, respectively.
Draft: please send errata to udlbookmail@gmail.com.

58 5 Loss functions
5.1.1 Computing a distribution over outputs
Thisshiftinperspectiveraisesthequestionofexactlyhowamodelf[x,ϕ]canbeadapted
to compute a probability distribution. The solution is simple. First, we choose a para-
metric distribution Pr(y|θ) defined on the output domain y. Then we use the network
to compute one or more of the parameters θ of this distribution.
For example, suppose the prediction domain is the set of real numbers, so y ∈ R.
Here, we might choose the univariate normal distribution, which is defined on R. This
distribution is defined by the mean µ and variance σ2, so θ = {µ,σ2}. The machine
learning model might predict the mean µ, and the variance σ2 could be treated as an
unknown constant.
5.1.2 Maximum likelihood criterion
Themodelnowcomputesdifferentdistributionparametersθ =f[x ,ϕ]foreachtraining
i i
input x . Each observed training output y should have high probability under its
i i
correspondingdistributionPr(y |θ ). Hence, wechoosethemodelparametersϕsothat
i i
they maximize the combined probability across all I training examples:
" #
YI
ϕˆ = argmax Pr(y |x )
i i
ϕ
"i=1 #
YI
= argmax Pr(y |θ )
i i
ϕ
"i=1 #
YI
= argmax Pr(y |f[x ,ϕ]) . (5.1)
i i
ϕ
i=1
Thecombinedprobabilitytermisthelikelihoodoftheparameters,andhenceequation5.1
is known as the maximum likelihood criterion.1
Here we are implicitly making two assumptions. First, we assume that the data
are identically distributed (the form of the probability distribution over the outputs y
i
is the same for each data point). Second, we assume that the conditional distribu-
AppendixC.1.5
Independence tions Pr(y i |x i ) of the output given the input are independent, so the total likelihood of
the training data decomposes as:
YI
Pr(y ,y ,...,y |x ,x ,...,x )= Pr(y |x ). (5.2)
1 2 I 1 2 I i i
i=1
In other words, we assume the data are independent and identically distributed (i.i.d.).
1A conditional probability Pr(z|ψ) can be considered in two ways. As a function of z, it is a
probability distribution that sums to one. As a function of ψ, it is known as a likelihood and does not
generallysumtoone.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

5.1 Maximum likelihood 59
Figure 5.2 The log transform. a) The log function is monotonically increasing.
If z>z′, then log[z]>log[z′]. It follows that the maximum of any function g[z]
will be at the same position as the maximum of log[g[z]]. b) A function g[z]. c)
Thelogarithmofthisfunctionlog[g[z]]. Allpositionsong[z]withapositiveslope
retain a positive slope after the log transform, and those with a negative slope
retain a negative slope. The position of the maximum remains the same.
5.1.3 Maximizing log-likelihood
The maximum likelihood criterion (equation 5.1) is not very practical. Each term
Pr(y |f[x ,ϕ]) can be small, so the product of many of these terms can be tiny. It
i i
may be diﬀicult to represent this quantity with finite precision arithmetic. Fortunately,
we can equivalently maximize the logarithm of the likelihood:
" #
YI
ϕˆ = argmax Pr(y |f[x ,ϕ])
i i
ϕ
"i=1" ##
YI
= argmax log Pr(y |f[x ,ϕ])
i i
ϕ
" i=1 #
XI h i
= argmax log Pr(y |f[x ,ϕ]) . (5.3)
i i
ϕ
i=1
This log-likelihood criterion is equivalent because the logarithm is a monotonically in-
creasing function: if z>z′, then log[z]>log[z′] and vice versa (figure 5.2). It follows
thatwhenwechangethemodelparametersϕtoimprovethelog-likelihoodcriterion,we
also improve the original maximum likelihood criterion. It also follows that the overall
maxima of the two criteria must be in the same place, so the best model parameters ϕˆ
are the same in both cases. However, the log-likelihood criterion has the practical ad-
vantage of using a sum of terms, not a product, so representing it with finite precision
isn’t problematic.
Draft: please send errata to udlbookmail@gmail.com.