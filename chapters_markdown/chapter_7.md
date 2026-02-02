# Chapter 7

*Pages: 110-131*

---

Chapter 7
Gradients and initialization
Chapter6introducediterativeoptimizationalgorithms. Thesearegeneral-purposemeth-
ods for finding the minimum of a function. In the context of neural networks, they find
parameters that minimize the loss so that the model accurately predicts the training
outputs from the inputs. The basic approach is to choose initial parameters randomly
andthenmakeaseriesofsmallchangesthatdecreasethelossonaverage. Eachchangeis
based on the gradient of the loss with respect to the parameters at the current position.
This chapter discusses two issues that are specific to neural networks. First, we
consider how to calculate the gradients eﬀiciently. This is a serious challenge since the
largest models at the time of writing have ∼1012 parameters, and the gradient needs to
be computed for every parameter at every iteration of the training algorithm. Second,
we consider how to initialize the parameters. If this is not done carefully, the initial
losses and their gradients can be extremely large or small. In either case, this impedes
the training process.
7.1 Problem definitions
Consider a network f[x,ϕ] with multivariate input x, parameters ϕ, and three hidden
layers h ,h , and h :
1 2 3
h = a[β +Ω x]
1 0 0
h = a[β +Ω h ]
2 1 1 1
h = a[β +Ω h ]
3 2 2 2
f[x,ϕ] = β +Ω h , (7.1)
3 3 3
wherethefunctiona[•]appliestheactivationfunctionseparatelytoeveryelementofthe
input. The model parameters ϕ = {β ,Ω ,β ,Ω ,β ,Ω ,β ,Ω } consist of the bias
0 0 1 1 2 2 3 3
vectors β and weight matrices Ω between every layer (figure 7.1).
k k
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

7.2 Computing derivatives 97
We also have individual loss terms ℓ , which return the negative log-likelihood of
i
the ground truth label y given the model prediction f[x ,ϕ] for training input x . For
i i i
example, this might be the least squares loss ℓ = (f[x ,ϕ]−y )2. The total loss is the
i i i
sum of these terms over the training data:
XI
L[ϕ]= ℓ . (7.2)
i
i=1
The most commonly used optimization algorithm for training neural networks is
stochastic gradient descent (SGD), which updates the parameters as:
X
∂ℓ [ϕ ]
ϕ ←−ϕ −α i t , (7.3)
t+1 t ∂ϕ
i∈B
t
whereαisthelearningrate,andB containsthebatchindicesatiterationt. Tocompute
t
this update, we need to calculate the derivatives:
∂ℓ ∂ℓ
i and i , (7.4)
∂β ∂Ω
k k
for the parameters {β ,Ω } at every layer k ∈ {0,1,...,K} and for each index i in
k k Problem7.1
the batch. The first part of this chapter describes the backpropagation algorithm, which
computes these derivatives eﬀiciently.
Inthesecondpartofthechapter,weconsiderhowtoinitializethenetworkparameters
beforewecommencetraining. WedescribemethodstochoosetheinitialweightsΩ and
k
biases β so that training is stable.
k
7.2 Computing derivatives
The derivatives of the loss tell us how the loss changes when we make a small change
to the parameters. Optimization algorithms exploit this information to manipulate the
parameters so that the loss becomes smaller. The backpropagation algorithm computes
thesederivatives. Themathematicaldetailsaresomewhatinvolved,sowefirstmaketwo
observations that provide some intuition.
Observation1: Eachweight(elementofΩ )multipliestheactivationatasourcehidden
k
unitandaddstheresulttoadestinationhiddenunitinthenextlayer. Itfollowsthatthe
effect of any small change to the weight is amplified or attenuated by the activation at
the source hidden unit. Hence, we run the network for each data example in the batch
and store the activations of all the hidden units. This is known as the forward pass
(figure 7.1). The stored activations will subsequently be used to compute the gradients.
Observation 2: A small change in a bias or weight causes a ripple effect of changes
throughthesubsequentnetwork. Thechangemodifiesthevalueofitsdestinationhidden
Draft: please send errata to udlbookmail@gmail.com.

98 7 Gradients and initialization
Figure7.1Backpropagationforwardpass. Thegoalistocomputethederivatives
ofthelossℓwithrespecttoeachoftheweights(arrows)andbiases(notshown).
Inotherwords,wewanttoknowhowasmallchangetoeachparameterwillaffect
theloss. Eachweightmultipliesthehiddenunitatitssourceandcontributesthe
resulttothehiddenunitatitsdestination. Consequently,theeffectsofanysmall
change to the weight will be scaled by the activation of the source hidden unit.
For example, the blue weight is applied to the second hidden unit at layer 1; if
the activation of this unit doubles, then the effect of a small change to the blue
weightwilldoubletoo. Hence,tocomputethederivativesoftheweights,weneed
to calculate and store the activations at the hidden layers. This is known as the
forward pass since it involves running the network equations sequentially.
unit. This,inturn,changesthevaluesofthehiddenunitsinthesubsequentlayer,which
will change the hidden units in the layer after that, and so on, until a change is made to
the model output and, finally, the loss.
Hence, to know how changing a parameter modifies the loss, we also need to know
howchangestoeverysubsequenthiddenlayerwill,inturn,modifytheirsuccessor. These
same quantities are required when considering other parameters in the same or earlier
layers. Itfollowsthatwecancalculatethemonceandreusethem. Forexample,consider
computing the effect of a small change in weights that feed into hidden layers h , h ,
3 2
and h , respectively:
1
• To calculate how a small change in a weight or bias feeding into hidden layer h
3
modifies the loss, we need to know (i) how a change in layer h changes the model
3
output f, and (ii) how a change in this output changes the loss ℓ (figure 7.2a).
• To calculate how a small change in a weight or bias feeding into hidden layer h
2
modifiestheloss,weneedtoknow(i)howachangeinlayerh affectsh ,(ii)howh
2 3 3
changes the model output, and (iii) how this output changes the loss (figure 7.2b).
• To calculate how a small change in a weight or bias feeding into hidden layer h
1
modifies the loss, we need to know (i) how a change in layer h affects layer h ,
1 2
(ii) how a change in layer h affects layer h , (iii) how layer h changes the model
2 3 3
output, and (iv) how the model output changes the loss (figure 7.2c).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

7.2 Computing derivatives 99
Figure 7.2 Backpropagation backward pass. a) To compute how a change to
a weight feeding into layer h (blue arrow) changes the loss, we need to know
3
how the hidden unit in h changes the model output f and how f changes the
3
loss (orange arrows). b) To compute how a small change to a weight feeding
into h (blue arrow) changes the loss, we need to know (i) how the hidden unit
2
in h changesh , (ii) howh changesf, and (iii) howf changesthe loss (orange
2 3 3
arrows). c)Similarly,tocomputehowasmallchangetoaweightfeedingintoh
1
(blue arrow) changes the loss, we need to know how h changes h and how
1 2
thesechangespropagatethroughtotheloss(orangearrows). Thebackwardpass
firstcomputesderivativesattheendofthenetworkandthenworksbackwardto
exploit the inherent redundancy of these computations.
Draft: please send errata to udlbookmail@gmail.com.

100 7 Gradients and initialization
As we move backward through the network, we see that most of the terms we need
were already calculated in the previous step, so we do not need to re-compute them.
Proceeding backward through the network in this way to compute the derivatives is
known as the backward pass.
The ideas behind backpropagation are relatively easy to understand. However, the
derivation requires matrix calculus because the bias and weight terms are vectors and
matrices, respectively. To help grasp the underlying mechanics, the following section
derives backpropagation for a simpler toy model with scalar parameters. We then apply
the same approach to a deep neural network in section 7.4.
7.3 Toy example
Consider a model f[x,ϕ] with eight scalar parameters ϕ={β ,ω ,β ,ω ,β ,ω ,β ,ω }
0 0 1 1 2 2 3 3
that consists of a composition of the functions sin[•],exp[•], and cos[•]:
h (cid:2) (cid:3)i
f[x,ϕ]=β +ω ·cos β +ω ·exp β +ω ·sin[β +ω ·x] , (7.5)
3 3 2 2 1 1 0 0
P
and a least squares loss function L[ϕ]= ℓ with individual terms:
i i
ℓ =(f[x ,ϕ]−y )2, (7.6)
i i i
where, as usual, x is the ith training input, and y is the ith training output. You can
i i
think of this as a simple neural network with one input, one output, one hidden unit at
eachlayer,anddifferentactivationfunctionssin[•],exp[•],andcos[•]betweeneachlayer.
We aim to compute the derivatives:
∂ℓ ∂ℓ ∂ℓ ∂ℓ ∂ℓ ∂ℓ ∂ℓ ∂ℓ
i , i , i , i , i , i , i , and i . (7.7)
∂β ∂ω ∂β ∂ω ∂β ∂ω ∂β ∂ω
0 0 1 1 2 2 3 3
Of course, we could find expressions for these derivatives by hand and compute them
directly. However, some of these expressions are quite complex. For example:
(cid:16) h (cid:2) (cid:3)i (cid:17)
∂ℓ
i = −2 β +ω ·cos β +ω ·exp β +ω ·sin[β +ω ·x ] −y
∂ω 3 3 2 2 1 1 0 0 i i
0 h i
·ω ω ω ·x ·cos[β +ω ·x ]·exp β +ω ·sin[β +ω ·x ]
1 2 3 i 0 0 i 1 1 0 0 i
(cid:20) h i(cid:21)
·sin β +ω ·exp β +ω ·sin[β +ω ·x ] . (7.8)
2 2 1 1 0 0 i
Such expressions are awkward to derive and code without mistakes and do not exploit
the inherent redundancy; notice that the three exponential terms are the same.
The backpropagation algorithm is an eﬀicient method for computing all of these
derivatives at once. It consists of (i) a forward pass, in which we compute and store a
series of intermediate values and the network output, and (ii) a backward pass, in which
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

7.3 Toy example 101
Figure 7.3 Backpropagation forward pass. We compute and store each of the
intermediate variables in turn until we finally calculate the loss.
we calculate the derivatives of each parameter, starting at the end of the network, and
reusing previous calculations as we move toward the start.
Forward pass: We treat the computation of the loss as a series of calculations:
f = β +ω ·x
0 0 0 i
h = sin[f ]
1 0
f = β +ω ·h
1 1 1 1
h = exp[f ]
2 1
f = β +ω ·h
2 2 2 2
h = cos[f ]
3 2
f = β +ω ·h
3 3 3 3
ℓ = (f −y )2. (7.9)
i 3 i
We compute and store the values of the intermediate variables f and h (figure 7.3).
k k
Backward pass #1: We now compute the derivatives of ℓ with respect to these inter-
i
mediate variables, but in reverse order:
∂ℓ ∂ℓ ∂ℓ ∂ℓ ∂ℓ ∂ℓ ∂ℓ
i, i , i, i , i, i , and i. (7.10)
∂f ∂h ∂f ∂h ∂f ∂h ∂f
3 3 2 2 1 1 0
The first of these derivatives is straightforward:
∂ℓ
i =2(f −y ). (7.11)
∂f 3 i
3
The next derivative can be calculated using the chain rule:
∂ℓ ∂f ∂ℓ
i = 3 i. (7.12)
∂h ∂h ∂f
3 3 3
Theleft-handsideaskshowℓ changeswhenh changes. Theright-handsidesayswecan
i 3
decomposethisinto(i)howf changeswhenh changesand(ii)howℓ changeswhenf
3 3 i 3
changes. In the original equations, h changes f , which changes ℓ , and the derivatives
3 3 i
Draft: please send errata to udlbookmail@gmail.com.

102 7 Gradients and initialization
Figure7.4Backpropagationbackwardpass#1. Weworkbackwardfromtheend
of the function computing the derivatives ∂ℓ /∂f and ∂ℓ /∂h of the loss with
i k i k
respect to the intermediate quantities. Each derivative is computed from the
previous one by multiplying by terms of the form ∂f k /∂h k or ∂h k /∂f k−1 .
represent the effects of this chain. Notice that we already computed the second of these
derivatives,andtheotheristhederivativeofβ +ω ·h withrespecttoh , whichisω .
3 3 3 3 3
We continue in this way, computing the derivatives of the output with respect to
these intermediate quantities (figure 7.4):
(cid:18) (cid:19)
∂ℓ ∂h ∂f ∂ℓ
i = 3 3 i
∂f ∂f ∂h ∂f
2 2 (cid:18) 3 3 (cid:19)
∂ℓ ∂f ∂h ∂f ∂ℓ
i = 2 3 3 i
∂h ∂h ∂f ∂h ∂f
2 2 (cid:18) 2 3 3 (cid:19)
∂ℓ ∂h ∂f ∂h ∂f ∂ℓ
i = 2 2 3 3 i
∂f ∂f ∂h ∂f ∂h ∂f
1 1 (cid:18) 2 2 3 3 (cid:19)
∂ℓ ∂f ∂h ∂f ∂h ∂f ∂ℓ
i = 1 2 2 3 3 i
∂h ∂h ∂f ∂h ∂f ∂h ∂f
1 1 (cid:18) 1 2 2 3 3 (cid:19)
∂ℓ ∂h ∂f ∂h ∂f ∂h ∂f ∂ℓ
i = 1 1 2 2 3 3 i . (7.13)
∂f ∂f ∂h ∂f ∂h ∂f ∂h ∂f
0 0 1 1 2 2 3 3
In each case, we have already computed the quantities in the brackets in the previous
Problem7.2
step, and the last term has a simple expression. These equations embody Observation 2
from the previous section (figure 7.2); we can reuse the previously computed derivatives
if we calculate them in reverse order.
Backward pass #2: Finally, we consider how the loss ℓ changes when we change the
i
parameters {β } and {ω }. Once more, we apply the chain rule (figure 7.5):
k k
∂ℓ ∂f ∂ℓ
i = k i
∂β ∂β ∂f
k k k
∂ℓ ∂f ∂ℓ
i = k i. (7.14)
∂ω ∂ω ∂f
k k k
In each case, the second term on the right-hand side was computed in equation 7.13.
When k >0, we have f =β +ω ·h , so:
k k k k
∂f ∂f
k =1 and k = h . (7.15)
∂β ∂ω k
k k
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

7.4 Backpropagation algorithm 103
Figure 7.5 Backpropagation backward pass #2. Finally, we compute the deriva-
tives ∂ℓ /∂β and ∂ℓ /∂ω . Each derivative is computed by multiplying the
i k i k
term ∂ℓ /∂f by ∂f /∂β or ∂f /∂ω as appropriate.
i k k k k k
This is consistent with Observation 1 from the previous section; the effect of a change
in the weight ω is proportional to the value of the source variable h (which was stored
k k
in the forward pass). The final derivatives from the term f =β +ω ·x are:
0 0 0 i
Notebook7.1
Backpropagation
∂f ∂f intoymodel
0 =1 and 0 = x . (7.16)
∂β ∂ω i
0 0
Backpropagation is both simpler and more eﬀicient than computing the derivatives in-
dividually, as in equation 7.8.1
7.4 Backpropagation algorithm
Nowwerepeatthisprocessforathree-layernetwork(figure7.1). Theintuitionandmuch
of the algebra are identical. The main differences are that intermediate variables f ,h
k k
are vectors, the biases β are vectors, the weights Ω are matrices, and we are using
k k
ReLU functions rather than simple algebraic functions like cos[•].
Forward pass: We write the network as a series of sequential calculations:
f = β +Ω x
0 0 0 i
h = a[f ]
1 0
f = β +Ω h
1 1 1 1
h = a[f ]
2 1
f = β +Ω h
2 2 2 2
h = a[f ]
3 2
f = β +Ω h
3 3 3 3
ℓ = l[f ,y ], (7.17)
i 3 i
1Notethatwedidnotactuallyneedthederivatives∂li/∂h
k
ofthelosswithrespecttotheactivations.
Inthefinalbackpropagationalgorithm,wewillnotcomputetheseexplicitly.
Draft: please send errata to udlbookmail@gmail.com.

104 7 Gradients and initialization
Figure 7.6 Derivative of rectified linear
unit. The rectified linear unit (orange
curve) returns zero when the input is
lessthanzeroandreturnstheinputoth-
erwise. Its derivative (cyan curve) re-
turns zero when the input is less than
zero (since the slope here is zero) and
one when the input is greater than zero
(since the slope here is one).
where f k−1 represents the pre-activations at the kth hidden layer (i.e., the values before
theReLUfunctiona[•])andh containstheactivationsatthekthhiddenlayer(i.e.,after
k
the ReLU function). The term l[f ,y ] represents the loss function (e.g., least squares or
3 i
binary cross-entropy loss). In the forward pass, we work through these calculations and
store all the intermediate quantities.
Backward pass #1: Now let’s consider how the loss changes when the pre-activations
f ,f ,f change. Applying the chain rule, the expression for the derivative of the loss ℓ
0 1 2 i
AppendixB.5
with respect to f is:
Matrixcalculus 2
∂ℓ ∂h ∂f ∂ℓ
i = 3 3 i. (7.18)
∂f ∂f ∂h ∂f
2 2 3 3
The three terms on the right-hand side have sizes D × D ,D × D , and D × 1,
3 3 3 f f
respectively, where D is the number of hidden units in the third layer, and D is the
3 f
dimensionality of the model output f .
3
Similarly, we can compute how the loss changes when we change f and f :
1 0
(cid:18) (cid:19)
∂ℓ ∂h ∂f ∂h ∂f ∂ℓ
i = 2 2 3 3 i (7.19)
∂f ∂f ∂h ∂f ∂h ∂f
1 1 2 (cid:18) 2 3 3 (cid:19)
∂ℓ ∂h ∂f ∂h ∂f ∂h ∂f ∂ℓ
i = 1 1 2 2 3 3 i . (7.20)
∂f ∂f ∂h ∂f ∂h ∂f ∂h ∂f
0 0 1 1 2 2 3 3
Note that in each case, the term in brackets was computed in the previous step. By
Problem7.3
working backward through the network, we can reuse the previous computations.
Moreover, the terms themselves are simple. Working backward through the right-
Problems7.4–7.5
hand side of equation 7.18, we have:
• The derivative ∂ℓ /∂f of the loss ℓ with respect to the network output f will
i 3 i 3
depend on the loss function but usually has a simple form.
• The derivative ∂f /∂h of the network output with respect to hidden layer h is:
3 3 3
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

7.4 Backpropagation algorithm 105
∂f ∂
3 = (β +Ω h )=ΩT. (7.21)
∂h ∂h 3 3 3 3
3 3
Ifyouareunfamiliarwithmatrixcalculus, thisresultisnotobvious. Itisexplored
Problem7.6
in problem 7.6.
• The derivative ∂h /∂f of the output h of the activation function with respect to
3 2 3
its input f will depend on the activation function. It will be a diagonal matrix
2
since each activation only depends on the corresponding pre-activation. For ReLU
functions, the diagonal terms are zero everywhere f is less than zero and one
2
Problems7.7–7.8
otherwise(figure7.6). Ratherthanmultiplybythismatrix,weextractthediagonal
terms as a vector I[f >0] and pointwise multiply, which is more eﬀicient.
2
The terms on the right-hand side of equations 7.19 and 7.20 have similar forms. As
we progress back through the network, we alternately (i) multiply by the transpose of
the weight matrices ΩT k and (ii) threshold based on the inputs f k−1 to the hidden layer.
These inputs were stored during the forward pass.
Backward pass #2: Now that we know how to compute ∂ℓ /∂f , we can focus on
i k
calculatingthederivativesofthelosswithrespecttotheweightsandbiases. Tocalculate
the derivatives of the loss with respect to the biases β , we again use the chain rule:
k
∂ℓ ∂f ∂ℓ
i = k i
∂β ∂β ∂f
k k k
∂ ∂ℓ
= (β +Ω h ) i
∂β k k k ∂f
k k
∂ℓ
= i, (7.22)
∂f
k
which we already calculated in equations 7.18 and 7.19.
Similarly, the derivative for the weights matrix Ω , is given by:
k
∂ℓ ∂f ∂ℓ
i = k i
∂Ω ∂Ω ∂f
k k k
∂ ∂ℓ
= (β +Ω h ) i
∂Ω k k k ∂f
k k
∂ℓ
= ihT. (7.23)
∂f k
k
Again, the progression from line two to line three is not obvious and is explored in
Problem7.9
problem7.9. However, theresultmakessense. Thefinallineisamatrixofthesamesize
asΩ . Itdependslinearlyonh , whichwasmultipliedbyΩ intheoriginalexpression.
k k k
This is also consistent with the initial intuition that the derivative of the weights in Ω
k
will be proportional to the values of the hidden units h that they multiply. Recall that
k
we already computed these during the forward pass.
Draft: please send errata to udlbookmail@gmail.com.

106 7 Gradients and initialization
7.4.1 Backpropagation algorithm summary
We now briefly summarize the final backpropagation algorithm. Consider a deep neural
network f[x ,ϕ] that takes input x , has K hidden layers with ReLU activations, and
i i
individual loss term ℓ = l[f[x ,ϕ],y ]. The goal of backpropagation is to compute the
i i i
derivatives ∂ℓ /∂β and ∂ℓ /∂Ω with respect to the biases β and weights Ω .
i k i k k k
Forward pass: We compute and store the following quantities:
f = β +Ω x
0 0 0 i
h k = a[f k−1 ] k ∈{1,2,...,K}
f = β +Ω h . k ∈{1,2,...,K} (7.24)
k k k k
Backwardpass: Westartwiththederivative∂ℓ /∂f ofthelossfunctionℓ withrespect
i K i
to the network output f and work backward through the network:
K
∂ℓ ∂ℓ
i = i k ∈{K,K−1,...,1}
∂β ∂f
k k
∂ℓ ∂ℓ
i = ihT k ∈{K,K−1,...,1}
∂Ω ∂f k
k k (cid:18) (cid:19)
∂ℓ ∂ℓ
∂f k− i 1 = I[f k−1 >0]⊙ ΩT k ∂f k i , k ∈{K,K−1,...,1} (7.25)
where ⊙ denotes pointwise multiplication, and I[f k−1 > 0] is a vector containing ones
wheref k−1 isgreaterthanzeroandzeroselsewhere. Finally, wecomputethederivatives
with respect to the first set of biases and weights:
∂ℓ ∂ℓ
i = i
∂β ∂f
0 0
∂ℓ ∂ℓ
i = ixT. (7.26)
∂Ω ∂f i
0 0
We calculate these derivatives for every training example in the batch and sum them
Problem7.10
together to retrieve the gradient for the SGD update.
Note that the backpropagation algorithm is extremely eﬀicient; the most demanding
Notebook7.2
Backpropagation computationalstepinboththeforwardandbackwardpassismatrixmultiplication(byΩ
and ΩT, respectively) which only requires additions and multiplications. However, it is
notmemoryeﬀicient;theintermediatevaluesintheforwardpassmustallbestored,and
this can limit the size of the model we can train.
7.4.2 Algorithmic differentiation
Althoughit’simportanttounderstandthebackpropagationalgorithm, it’sunlikelythat
you will need to code it in practice. Modern deep learning frameworks such as PyTorch
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

7.5 Parameter initialization 107
and TensorFlow calculate the derivatives automatically, given the model specification.
This is known as algorithmic differentiation.
Each functional component (linear transform, ReLU activation, loss function) in the
framework knows how to compute its own derivative. For example, the PyTorch ReLU
function z = relu[z ] knows how to compute the derivative of its output z with
out in out
respect to its input z . Similarly, a linear function z = β +Ωz knows how to
in out in
compute the derivatives of the output z with respect to the input z and with re-
out in
spect to the parameters β and Ω. The algorithmic differentiation framework also knows
the sequence of operations in the network and thus has all the information required to
perform the forward and backward passes.
Theseframeworksexploitthemassiveparallelismofmoderngraphicsprocessingunits
(GPUs). Computationssuchasmatrixmultiplication(whichfeaturesinboththeforward
and backward pass) are naturally amenable to parallelization. Moreover, it’s possible to
Problem7.11
perform the forward and backward passes for the entire batch in parallel if the model
and intermediate results in the forward pass do not exceed the available memory.
Since the training algorithm now processes the entire batch in parallel, the input
becomes a multi-dimensional tensor. In this context, a tensor can be considered the
generalization of a matrix to arbitrary dimensions. Hence, a vector is a 1D tensor, a
matrix is a 2D tensor, and a 3D tensor is a 3D grid of numbers. Until now, the training
data have been 1D, so the input for backpropagation would be a 2D tensor where the
first dimension indexes the batch element and the second indexes the data dimension.
In subsequent chapters, we will encounter more complex structured input data. For
example, in models where the input is an RGB image, the original data examples are
3D (height × width × channel). Here, the input to the learning framework would be a
4D tensor, where the extra dimension indexes the batch element.
7.4.3 Extension to arbitrary computational graphs
Wehavedescribedbackpropagationinadeepneuralnetworkthatisnaturallysequential;
we calculate the intermediate quantities f ,h ,f ,h ...,f in turn. However, models
0 1 1 2 k
need not be restricted to sequential computation. Later in this book, we will meet
models with branching structures. For example, we might take the values in a hidden
layer and process them through two different sub-networks before recombining.
Problems7.12–7.13
Fortunately, the ideas of backpropagation still hold if the computational graph is
acyclic. ModernalgorithmicdifferentiationframeworkssuchasPyTorchandTensorFlow
can handle arbitrary acyclic computational graphs.
7.5 Parameter initialization
The backpropagation algorithm computes the derivatives that are used by stochastic
gradient descent and Adam to train the model. We now address how to initialize the
parameters before we start training. To see why this is crucial, consider that during the
forward pass, each set of pre-activations f is computed as:
k
Draft: please send errata to udlbookmail@gmail.com.

108 7 Gradients and initialization
f = β +Ω h
k k k k
= β k +Ω k a[f k−1 ], (7.27)
wherea[•]appliestheReLUfunctionsandΩ andβ aretheweightsandbiases,respec-
k k
tively. Imagine that we initialize all the biases to zero and the elements of Ω according
k
to a normal distribution with mean zero and variance σ2. Consider two scenarios:
• Ifthevarianceσ2 isverysmall(e.g.,10−5),theneachelementofβ +Ω h willbe
k k k
a weighted sum of h where the weights are very small; the result will likely have
k
a smaller magnitude than the input. In addition, the ReLU function clips values
less than zero, so the range of h k will be half that of f k−1 . Consequently, the
magnitudes of the pre-activations at the hidden layers will get smaller and smaller
as we progress through the network.
• If the variance σ2 is very large (e.g., 105), then each element of β +Ω h will be
k k k
a weighted sum of h where the weights are very large; the result is likely to have
k
a much larger magnitude than the input. The ReLU function halves the range of
theinputs,butifσ2 islargeenough,themagnitudesofthepre-activationswillstill
get larger as we progress through the network.
Inthesetwosituations, the valuesat thepre-activationscan becomesosmall orso large
that they cannot be represented with finite precision floating point arithmetic.
Even if the forward pass is tractable, the same logic applies to the backward pass.
Each gradient update (equation 7.25) consists of multiplying by ΩT. If the values of Ω
are not initialized sensibly, then the gradient magnitudes may decrease or increase un-
controllably during the backward pass. These cases are known as the vanishing gradient
problem and the exploding gradient problem, respectively. In the former case, updates to
the model become vanishingly small. In the latter case, they become unstable.
7.5.1 Initialization for forward pass
Wenowpresentamathematicalversionofthesameargument. Considerthecomputation
between adjacent pre-activations f and f′ with dimensions D
h
and D h′, respectively:
h = a[f],
′
f = β+Ωh (7.28)
where h represents the activations, Ω and β represent the weights and biases, and a[•]
is the activation function.
Assume the pre-activations f in the input layer f have variance σ2. Consider ini-
j f
tializing the biases β to zero and the weights Ω as normally distributed with mean
i ij
zero and variance σ2. Now we derive expressions for the mean and variance of the
Ω
pre-activations f′ in the subsequent layer.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

7.5 Parameter initialization 109
The expectation (mean) E[f′] of the intermediate values f′ is: AppendixC.2
i i
Expectation
2 3
XDh
E[f ′ ] = E4 β + Ω h 5
i i ij j
j=1
XDh
= E[β ]+ E[Ω h ]
i ij j
j=1
XDh
= E[β ]+ E[Ω ]E[h ]
i ij j
j=1
XDh
= 0+ 0·E[h ]=0, (7.29)
j
j=1
whereD isthedimensionalityoftheinputlayerh. Wehaveusedtherulesformanipu-
h
AppendixC.2.1
latingexpectations,andwehaveassumedthatthedistributionsoverthehiddenunitsh
j Expectationrules
and the network weights Ω are independent between the second and third lines.
ij
Using this result, we see that the variance σ2 of the pre-activations f′ is:
f′ i
σ2 = E[f ′2]−E[f ′ ]2
f′ 2i 0 i 1 3
2
6
XDh
7
= E4@ β + Ω h A 5−0
i ij j
j=1
20 1 3
2
6
XDh
7
= E4@ Ω h A 5
ij j
j=1
XDh (cid:2) (cid:3) (cid:2) (cid:3)
= E Ω2 E h2
ij j
j=1
XDh (cid:2) (cid:3) XDh (cid:2) (cid:3)
= σ2E h2 =σ2 E h2 , (7.30)
Ω j Ω j
j=1 j=1
where we have used the variance identity σ2 = E[(z−E[z])2] = E[z2]−E[z]2. We have
AppendixC.2.3
assumedoncemorethatthedistributionsoftheweightsΩ andthehiddenunitsh are
ij j Varianceidentity
independent between lines three and four.
Assumingthatthedistributionofpre-activationsf atthepreviouslayerissymmetric
j
about zero, half of these pre-activations will be clipped by the ReLU function, and the
second moment E[h2] will be half the variance σ2 of f (see problem 7.14):
j f j Problem7.14
XDh σ2 1
σ2 =σ2 f = D σ2σ2. (7.31)
f i ′ Ω 2 2 h Ω f
j=1
Draft: please send errata to udlbookmail@gmail.com.

110 7 Gradients and initialization
Figure 7.7 Weight initialization. Consider a deep network with 50 hidden layers
andD =100hiddenunitsperlayer. Thenetworkhasa100-dimensionalinputx
h
initialized from a standard normal distribution, a single fixed target y = 0, and
a least squares loss function. The bias vectors β are initialized to zero, and the
k
weightmatricesΩ areinitializedwithanormaldistributionwithmeanzeroand
k
five different variances σ2 ∈ {0.001,0.01,0.02,0.1,1.0}. a) Variance of hidden
Ω
unitactivationscomputedinforwardpassasafunctionofthenetworklayer. For
He initialization (σ2 =2/D =0.02), the variance is stable. However, for larger
Ω h
values, it increases rapidly, and for smaller values, it decreases rapidly (note
log scale). b) The variance of the gradients in the backward pass (solid lines)
continuesthistrend; ifweinitializewithavaluelargerthan0.02,themagnitude
of the gradients increases rapidly as we pass back through the network. If we
initialize with a value smaller, then the magnitude decreases. These are known
as the exploding gradient and vanishing gradient problems, respectively.
This,inturn,impliesthatifwewantthevarianceσ2 ofthesubsequentpre-activationsf′
f′
to be the same as the variance σ2 of the original pre-activations f during the forward
f
pass, we should set:
2
σ2 = , (7.32)
Ω D
h
where D is the dimension of the original layer to which the weights were applied. This
h
is known as He initialization.
7.5.2 Initialization for backward pass
A similar argument establishes how the variance of the gradients ∂l/∂f changes during
k
the backward pass. During the backward pass, we multiply by the transpose ΩT of the
weight matrix (equation 7.25), so the equivalent expression becomes:
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

7.6 Example training code 111
2
σ2 = , (7.33)
Ω D h′
where D h′ is the dimension of the layer that the weights feed into.
7.5.3 Initialization for both forward and backward pass
If the weight matrix Ω is not square (i.e., there are different numbers of hidden units
in the two adjacent layers, so D h and D h′ differ), then it is not possible to choose the
variancetosatisfybothequations7.32and7.33simultaneously. Onepossiblecompromise
is to use the mean (D
h
+D h′)/2 as a proxy for the number of terms, which gives:
4
σ2 = . (7.34)
Ω D h +D h′
Figure 7.7 shows empirically that both the variance of the hidden units in the forward
Problem7.15
pass and the variance of the gradients in the backward pass remain stable when the
parameters are initialized appropriately.
Notebook7.3
Initialization
7.6 Example training code
The primary focus of this book is scientific; this is not a guide for implementing deep
learning models. Nonetheless, in figure 7.8, we present PyTorch code that implements
the ideas explored in this book so far. The code defines a neural network and initializes
Problems7.16–7.17
theweights. Itcreatesrandominputandoutputdatasetsanddefinesaleastsquaresloss
function. The model is trained from the data using SGD with momentum in batches of
size 10 over 100 epochs. The learning rate starts at 0.01 and halves every 10 epochs.
The takeaway is that although the underlying ideas in deep learning are quite com-
plex, implementation is relatively simple. For example, all of the details of the back-
propagation are hidden in the single line of code: loss.backward().
7.7 Summary
The previous chapter introduced stochastic gradient descent (SGD), an iterative opti-
mizationalgorithmthataimstofindtheminimumofafunction. Inthecontextofneural
networks, this algorithm finds the parameters that minimize the loss function. SGD re-
lies on the gradient of the loss function with respect to the parameters, which must be
initialized before optimization. This chapter has addressed these two problems for deep
neural networks.
The gradients must be evaluated for a very large number of parameters, for each
memberofthebatch,andateachSGDiteration. Itishenceimperativethatthegradient
Draft: please send errata to udlbookmail@gmail.com.

112 7 Gradients and initialization
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR
# define input size, hidden layer size, output size
D_i, D_k, D_o = 10, 40, 5
# create model with two hidden layers
model = nn.Sequential(
nn.Linear(D_i, D_k),
nn.ReLU(),
nn.Linear(D_k, D_k),
nn.ReLU(),
nn.Linear(D_k, D_o))
# He initialization of weights
def weights_init(layer_in):
if isinstance(layer_in, nn.Linear):
nn.init.kaiming_normal_(layer_in.weight)
layer_in.bias.data.fill_(0.0)
model.apply(weights_init)
# choose least squares loss function
criterion = nn.MSELoss()
# construct SGD optimizer and initialize learning rate and momentum
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9)
# object that decreases learning rate by half every 10 epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
# create 100 random data points and store in data loader class
x = torch.randn(100, D_i)
y = torch.randn(100, D_o)
data_loader = DataLoader(TensorDataset(x,y), batch_size=10, shuffle=True)
# loop over the dataset 100 times
for epoch in range(100):
epoch_loss = 0.0
# loop over batches
for i, data in enumerate(data_loader):
# retrieve inputs and labels for this batch
x_batch, y_batch = data
# zero the parameter gradients
optimizer.zero_grad()
# forward pass
pred = model(x_batch)
loss = criterion(pred, y_batch)
# backward pass
loss.backward()
# SGD update
optimizer.step()
# update statistics
epoch_loss += loss.item()
# print error
print(f'Epoch {epoch:5d}, loss {epoch_loss:.3f}')
# tell scheduler to consider updating learning rate
scheduler.step()
Figure 7.8 Sample code for training two-layer network on random data.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 113
computationiseﬀicient,andtothisend,thebackpropagationalgorithmwasintroduced.
Careful parameter initialization is also critical. The magnitudes of the hidden unit
activations can either decrease or increase exponentially in the forward pass. The same
istrueofthegradientmagnitudesinthebackwardpass,wherethesebehaviorsareknown
as the vanishing gradient and exploding gradient problems. Both impede training but
can be avoided with appropriate initialization.
We’ve now defined the model and the loss function, and we can train a model for a
given task. The next chapter discusses how to measure the model performance.
Notes
Backpropagation: Eﬀicientreuseofpartialcomputationswhilecalculatinggradientsincom-
putational graphs has been repeatedly discovered, including by Werbos (1974), Bryson et al.
(1979), LeCun (1985), and Parker (1985). However, the most celebrated description of this
idea was by Rumelhart et al. (1985) and Rumelhart et al. (1986), who also coined the term
“backpropagation.” Thislatterworkkick-startedanewphaseofneuralnetworkresearchinthe
eighties and nineties; for the first time, it was practical to train networks with hidden layers.
However, progress stalled due (in retrospect) to a lack of training data, limited computational
power,andtheuseofsigmoidactivations. Areassuchasnaturallanguageprocessingandcom-
puter vision did not rely on neural network models until the remarkable image classification
results of Krizhevsky et al. (2012) ushered in the modern era of deep learning.
The implementation of backpropagation in modern deep learning frameworks such as PyTorch
andTensorFlowisanexampleofreverse-modealgorithmicdifferentiation. Thisisdistinguished
from forward-mode algorithmic differentiation in which the derivatives from the chain rule
are accumulated while moving forward through the computational graph (see problem 7.13).
Further information about algorithmic differentiation can be found in Griewank & Walther
(2008) and Baydin et al. (2018).
Initialization: He initialization was first introduced by He et al. (2015). It follows closely
from Glorot or Xavier initialization (Glorot & Bengio, 2010), which is very similar but does
not consider the effect of the ReLU layer and so differs by a factor of two. Essentially the
same method was proposed much earlier by LeCun et al. (2012) but with a slightly different
motivation;inthiscase,sigmoidalactivationfunctionswereused,whichnaturallynormalizethe
rangeofoutputsateachlayer,andhencehelppreventanexponentialincreaseinthemagnitudes
of the hidden units. However, if the pre-activations are too large, they fall into the flat regions
of the sigmoid function and result in very small gradients. Hence, it is still important to
initialize the weights sensibly. Klambauer et al. (2017) introduce the scaled exponential linear
unit (SeLU) and show that, within a certain range of inputs, this activation function tends to
make the activations in network layers automatically converge to mean zero and unit variance.
Acompletelydifferentapproachistopassdatathroughthenetworkandthennormalizebythe
empirically observed variance. Layer-sequential unit variance initialization (Mishkin & Matas,
2016) is an example of this kind of method, in which the weight matrices are initialized as
orthonormal. GradInit (Zhu et al., 2021) randomizes the initial weights and temporarily fixes
them while it learns non-negative scaling factors for each weight matrix. These factors are
selected to maximize the decrease in the loss for a fixed learning rate subject to a constraint
onthemaximumgradientnorm. ActivationnormalizationorActNormaddsalearnablescaling
and offset parameter after each network layer at each hidden unit. They run an initial batch
throughthenetworkandthenchoosetheoffsetandscalesothatthemeanoftheactivationsis
zero and the variance one. After this, these extra parameters are learned as part of the model.
Draft: please send errata to udlbookmail@gmail.com.

114 7 Gradients and initialization
Closely related to these methods are schemes such as BatchNorm (Ioffe & Szegedy, 2015), in
which the network normalizes the variance of each batch as part of its processing at every
step. BatchNormanditsvariantsarediscussedinchapter11. Otherinitializationschemeshave
been proposed for specific architectures, including the ConvolutionOrthogonal initializer (Xiao
etal.,2018a)forconvolutionalnetworks,Fixup(Zhangetal.,2019a)forresidualnetworks,and
TFixup (Huang et al., 2020a) and DTFixup (Xu et al., 2021b) for transformers.
Reducing memory requirements: Trainingneuralnetworksismemoryintensive. Wemust
storeboththe modelparameters andthe pre-activationsat thehidden unitsfor everymember
of the batch during the forward pass. Two methods that decrease memory requirements are
gradientcheckpointing(Chenetal.,2016a)andmicro-batching(Huangetal.,2019). Ingradient
checkpointing, the activations are only stored every N layers during the forward pass. During
thebackwardpass,theintermediatemissingactivationsarerecalculatedfromthenearestcheck-
point. Inthismanner,wecandrasticallyreducethememoryrequirementsatthecomputational
cost of performing the forward pass twice (problem 7.11). In micro-batching, the batch is sub-
dividedintosmallerparts,andthegradientupdatesareaggregatedfromeachsub-batchbefore
being applied to the network. A completely different approach is to build a reversible network
(e.g.,Gomezetal.,2017),inwhichtheactivationsatthepreviouslayercanbecomputedfrom
theactivationsatthecurrentone,sothereisnoneedtocacheanythingduringtheforwardpass
(see chapter 16). Sohoni et al. (2019) review approaches to reducing memory requirements.
Distributed training: For suﬀiciently large models, the memory requirements or total re-
quired time may be too much for a single processor. In this case, we must use distributed
training, in which training takes place in parallel across multiple processors. There are several
approaches to parallelism. In data parallelism, each processor or node contains a full copy of
themodelbutrunsasubsetofthebatch(seeXingetal.,2015;Lietal.,2020b). Thegradients
from each node are aggregated centrally and then redistributed back to each node to ensure
thatthemodelsremainconsistent. Thisisknownassynchronoustraining. Thesynchronization
required to aggregate and redistribute the gradients can be a performance bottleneck, and this
leads to the idea of asynchronous training. For example, in the Hogwild! algorithm (Recht
et al., 2011), the gradient from a node is used to update a central model whenever it is ready.
The updated model is then redistributed to the node. This means that each node may have a
slightly different version of the model at any given time, so the gradient updates may be stale;
however, it works well in practice. Other decentralized schemes have also been developed. For
example, in Zhang et al. (2016a), the individual nodes update one another in a ring structure.
Data parallelism methods still assume that the entire model can be held in the memory of a
single node. Pipeline model parallelism stores different layers of the network on different nodes
and hence does not have this requirement. In a naïve implementation, the first node runs the
forwardpass for the batchon the first few layersand passes the result to the next node, which
runstheforwardpassonthenextfewlayersandsoon. Inthebackwardpass,thegradientsare
updatedintheoppositeorder. Theobviousdisadvantageofthisapproachisthateachmachine
lies idle for most of the cycle. Various schemes revolving around each node processing micro-
batches sequentially have been proposed to reduce this ineﬀiciency (e.g., Huang et al., 2019;
Narayananetal.,2021a). Finally,intensor model parallelism,computationatasinglenetwork
layer is distributed across nodes (e.g., Shoeybi et al., 2019). A good overview of distributed
training methods can be found in Narayananet al. (2021b), who combinetensor, pipeline, and
data parallelism to train a language model with one trillion parameters on 3072 GPUs.
Problems
Problem 7.1 A two-layer network with two hidden units in each layer can be defined as:
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 115
h i
y = ϕ +ϕ a ψ +ψ a[θ +θ x]+ψ a[θ +θ x]
0 1 01 11 01 11 21 02 12
h i
+ϕ a ψ +ψ a[θ +θ x]+ψ a[θ +θ x] , (7.35)
2 02 12 01 11 22 02 12
where the functions a[•] are ReLU functions. Compute the derivatives of the output y with
respecttoeachofthe13parametersϕ
•
,θ••,andψ••directly(i.e.,notusingthebackpropagation
algorithm). The derivative of the ReLU function with respect to its input ∂a[z]/∂z is the
indicator function I[z > 0], which returns one if the argument is greater than zero and zero
otherwise (figure 7.6).
Problem 7.2 Find an expression for the final term in each of the five chains of derivatives in
equation 7.13.
Problem 7.3 What size are each of the terms in equation 7.20?
Problem 7.4 Calculate the derivative ∂ℓ /∂f[x ,ϕ] for the least squares loss function:
i i
ℓ =(y −f[x ,ϕ])2. (7.36)
i i i
Problem 7.5 Calculate the derivative ∂ℓ /∂f[x ,ϕ] for the binary classification loss function:
i i
h (cid:2) (cid:3)i h (cid:2) (cid:3)i
ℓ =−(1−y )log 1−sig f[x ,ϕ] −y log sig f[x ,ϕ] , (7.37)
i i i i i
where the function sig[•] is the logistic sigmoid and is defined as:
1
sig[z]= . (7.38)
1+exp[−z]
Problem 7.6∗ Show that for z=β+Ωh:
∂z
=ΩT, (7.39)
∂h
where∂z/∂hisamatrixcontainingtheterm∂z /∂h initsith columnandjth row. Todothis,
i j
first find an expression for the constituent elements ∂z /∂h , and then consider the form that
i j
the matrix ∂z/∂h must take.
Problem 7.7 Consider the case where we use the logistic sigmoid (see equation 7.38) as an
activation function, so h = sig[f]. Compute the derivative ∂h/∂f for this activation function.
What happens to the derivative when the input takes (i) a large positive value and (ii) a large
negative value?
Problem 7.8 Consider using (i) the Heaviside function and (ii) the rectangular function as
activation functions:
(
0 z<0
Heaviside[z]= , (7.40)
1 z≥0
Draft: please send errata to udlbookmail@gmail.com.

116 7 Gradients and initialization
Figure 7.9 Computational graph for problem 7.12 and problem 7.13. Adapted
from Domke (2010).
and
8
><0 z<0
rect[z]= 1 0≤z≤1. (7.41)
>:
0 z>1
Discuss why these functions are problematic for neural network training with gradient-based
optimization methods.
Problem 7.9∗ Consider a loss function ℓ[f], where f =β+Ωh. We want to find how the loss ℓ
changes when we change Ω, which we’ll express with a matrix that contains the derivative
∂ℓ/∂Ω at the ith row and jth column. Find an expression for ∂f /∂Ω and, using the chain
ij i ij
rule, show that:
∂ℓ ∂ℓ
= hT. (7.42)
∂Ω ∂f
Problem 7.10∗ Derive the equations for the backward pass of the backpropagation algorithm
for a network that uses leaky ReLU activations, which are defined as:
(
α·z z<0
a[z]=ReLU[z]= , (7.43)
z z≥0
where α is a small positive constant (typically 0.1).
Problem 7.11 Consider training a network with fifty layers using gradient checkpointing. As-
sume that we store the pre-activations at every tenth hidden layer during the forward pass.
Explain how to compute the derivatives in this situation.
Problem 7.12∗ This problem explores computing derivatives on general acyclic computational
graphs. Consider the function:
(cid:2) (cid:3)
y=exp exp[x]+exp[x]2 +sin[exp[x]+exp[x]2]. (7.44)
We can break this down into a series of intermediate computations so that:
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 117
f = exp[x]
1
f = f2
2 1
f = f +f
3 1 2
f = exp[f ]
4 3
f = sin[f ]
5 3
y = f +f . (7.45)
4 5
The associated computational graph is depicted in figure 7.9. Compute the derivative ∂y/∂x
by reverse-mode differentiation. In other words, compute in order:
∂y ∂y ∂y ∂y ∂y ∂y
, , , , and , (7.46)
∂f ∂f ∂f ∂f ∂f ∂x
5 4 3 2 1
using the chain rule in each case to make use of the derivatives already computed.
Problem 7.13∗ For the same function as in problem 7.12, compute the derivative ∂y/∂x by
forward-mode differentiation. In other words, compute in order:
∂f ∂f ∂f ∂f ∂f ∂y
1, 2, 3, 4, 5, and , (7.47)
∂x ∂x ∂x ∂x ∂x ∂x
using the chain rule in each case to make use of the derivatives already computed. Why do
we not use forward-mode differentiation when we calculate the parameter gradients for deep
networks?
Problem 7.14 Consider a random variable a with variance Var[a] = σ2 and a symmetrical
distribution around the mean E[a]= 0. Prove that if we pass this variable through the ReLU
function:
(
0 a<0
b=ReLU[a]= , (7.48)
a a≥0
then the second moment of the transformed variable is E[b2]=σ2/2.
Problem 7.15 What would you expect to happen if we initialized all of the weights and biases
in the network to zero?
Problem 7.16 Implement the code in figure 7.8 in PyTorch and plot the training loss as a
function of the number of epochs.
Problem 7.17 Change the code in figure 7.8 to tackle a binary classification problem. You will
need to (i) change the targets y so they are binary, (ii) change the network to predict numbers
between zero and one (iii) change the loss function appropriately.
Draft: please send errata to udlbookmail@gmail.com.