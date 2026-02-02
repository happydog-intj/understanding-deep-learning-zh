# Chapter 4: The final output is a linear combination of these hidden units (equation 4.9).

*Pages: 60-62*

---

46 4 Deep neural networks
Considering these equations leads to another way to think about how the network con-
Notebook4.2
structs an increasingly complicated function (figure 4.5):
Clipping
functions
1. The three hidden units h ,h , and h in the first layer are computed as usual by
1 2 3
forming linear functions of the input and passing these through ReLU activation
functions (equation 4.7).
2. The pre-activations at the second layer are computed by taking three new linear
functions of these hidden units (arguments of the activation functions in equa-
tion 4.8). At this point, we effectively have a shallow network with three outputs;
wehavecomputedthreepiecewiselinearfunctionswiththe“joints”betweenlinear
regions in the same places (see figure 3.6).
3. Atthesecondhiddenlayer,anotherReLUfunctiona[•]isappliedtoeachfunction
(equation 4.8), which clips them and adds new “joints” to each.
4. The final output is a linear combination of these hidden units (equation 4.9).
Inconclusion,wecaneitherthinkofeachlayeras“folding”theinputspaceorascre-
atingnewfunctions,whichareclipped(creatingnewregions)andthenrecombined. The
former view emphasizes the dependencies in the output function but not how clipping
creates new joints, and the latter has the opposite emphasis. Ultimately, both descrip-
tions provide only partial insight into how deep neural networks operate. Regardless,
it’s important not to lose sight of the fact that this is still merely an equation relating
input x to output y′. Indeed, we can combine equations 4.7–4.9 to get one expression:
′ ′ ′
y = ϕ +ϕ a[ψ +ψ a[θ +θ x]+ψ a[θ +θ x]+ψ a[θ +θ x]]
0 1 10 11 10 11 12 20 21 13 30 31
′
+ϕ a[ψ +ψ a[θ +θ x]+ψ a[θ +θ x]+ψ a[θ +θ x]]
2 20 21 10 11 22 20 21 23 30 31
′
+ϕ a[ψ +ψ a[θ +θ x]+ψ a[θ +θ x]+ψ a[θ +θ x]],
3 30 31 10 11 32 20 21 33 30 31
(4.10)
although this is admittedly rather diﬀicult to understand.
4.3.1 Hyperparameters
We can extend the deep network construction to more than two hidden layers; modern
networksmighthavemorethanahundredlayerswiththousandsofhiddenunitsateach
layer. Thenumberofhiddenunitsineachlayerisreferredtoasthewidthofthenetwork,
and the number of hidden layers as the depth. The total number of hidden units is a
measure of the network’s capacity.
We denote the number of layers as K and the number of hidden units in each layer
as D ,D ,...,D . These are examples of hyperparameters. They are quantities chosen
1 2 K
Problem4.2
before we learn the model parameters (i.e., the slope and intercept terms). For fixed
hyperparameters (e.g., K = 2 layers with D = 3 hidden units in each), the model
k
describes a family of functions, and the parameters determine the particular function.
Hence, when we also consider the hyperparameters, we can think of neural networks as
representing a family of families of functions relating input to output.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

4.3 Deep neural networks 47
Figure 4.5 Computation for the deep network in figure 4.4. a–c) The inputs
to the second hidden layer (i.e., the pre-activations) are three piecewise linear
functions where the “joints” between the linear regions are at the same places
(see figure 3.6). d–f) Each piecewise linear function is clipped to zero by the
ReLU activation function. g–i) These clipped functions are then weighted with
parameters ϕ′,ϕ′, and ϕ′, respectively. j) Finally, the clipped and weighted
1 2 3
functions are summed and an offset ϕ′ that controls the overall height is added.
0
(Interactive figure)
Draft: please send errata to udlbookmail@gmail.com.

48 4 Deep neural networks
Figure4.6MatrixnotationfornetworkwithD =3-dimensionalinputx,D =2-
i o
dimensional output y, and K = 3 hidden layers h ,h , and h of dimensions
1 2 3
D =4,D =2,andD =3respectively. TheweightsarestoredinmatricesΩ
1 2 3 k
thatmultiplytheactivationsfromtheprecedinglayertocreatethepre-activations
at the subsequent layer. For example, the weight matrix Ω that computes the
1
pre-activationsath fromtheactivationsath hasdimension2×4. Itisapplied
2 1
to the four hidden units in layer one and creates the inputs to the two hidden
units at layer two. The biases are stored in vectors β and have the dimension
k
ofthelayerintowhichtheyfeed. Forexample,thebiasvectorβ islengththree
2
because layer h contains three hidden units.
3
4.4 Matrix notation
We have seen that a deep neural network consists of linear transformations alternating
AppendixB.3
with activation functions. We could equivalently describe equations 4.7–4.9 in matrix
Matrices
notation as:
2 3 22 3 2 3 3
h θ θ
1 10 11
4 5 44 5 4 5 5
h =a θ + θ x , (4.11)
2 20 21
h θ θ
3 30 31
2 3 22 3 2 32 33
h′ ψ ψ ψ ψ h
1 10 11 12 13 1
4 h′5
=a
44
ψ
5
+
4
ψ ψ ψ
54
h
55
, (4.12)
2 20 21 22 23 2
h′ ψ ψ ψ ψ h
3 30 31 32 33 3
and
2 3
(cid:2) (cid:3)
h′
1
y ′ =ϕ ′ + ϕ′ ϕ′ ϕ′ 4 h′5 , (4.13)
0 1 2 3 2
h′
3
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.