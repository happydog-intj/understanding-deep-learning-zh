# Chapter 4

*Pages: 55-59*

---

Chapter 4
Deep neural networks
The last chapter described shallow neural networks, which have a single hidden layer.
This chapter introduces deep neural networks, which have more than one hidden layer.
With ReLU activation functions, both shallow and deep networks describe piecewise
linear mappings from input to output.
As the number of hidden units increases, shallow neural networks improve their
descriptive power. Indeed, with enough hidden units, shallow networks can describe
arbitrarily complex functions in high dimensions. However, it turns out that for some
functions,therequirednumberofhiddenunitsisimpracticallylarge. Deepnetworkscan
produce many more linear regions than shallow networks for a given number of parame-
ters. Hence, from a practical standpoint, they can be used to describe a broader family
of functions.
4.1 Composing neural networks
To gain insight into the behavior of deep neural networks, we first consider composing
twoshallownetworkssotheoutputofthefirstbecomestheinputofthesecond. Consider
twoshallownetworkswiththreehiddenunitseach(figure4.1a). Thefirstnetworktakes
an input x and returns output y and is defined by:
h = a[θ +θ x]
1 10 11
h = a[θ +θ x]
2 20 21
h = a[θ +θ x], (4.1)
3 30 31
and
y =ϕ +ϕ h +ϕ h +ϕ h . (4.2)
0 1 1 2 2 3 3
The second network takes y as input and returns y′ and is defined by:
Draft: please send errata to udlbookmail@gmail.com.

42 4 Deep neural networks
Figure 4.1Composingtwosingle-layernetworkswiththreehiddenunitseach. a)
Theoutputy ofthefirstnetworkconstitutestheinputtothesecondnetwork. b)
Thefirstnetworkmapsinputsx∈[−1,1]tooutputsy∈[−1,1]usingafunction
comprising three linear regions that are chosen so that they alternate the sign
of their slope (fourth linear region is outside range of graph). Multiple inputs x
(graycircles)nowmaptothesameoutputy(cyancircle). c)Thesecondnetwork
definesafunctioncomprisingthreelinearregionsthattakesyandreturnsy′ (i.e.,
the cyan circle is mapped to the brown circle). d) The combined effect of these
two functions when composed is that (i) three different inputs x are mapped to
anygivenvalueofybythefirstnetworkand(ii)areprocessedinthesamewayby
thesecondnetwork;theresultisthatthefunctiondefinedbythesecondnetwork
inpanel(c)isduplicatedthreetimes,variouslyflippedandrescaledaccordingto
the slope of the regions of panel (b). (Interactive figure)
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

4.2 From composing networks to deep networks 43
′ ′ ′
h = a[θ +θ y]
1 10 11
′ ′ ′
h = a[θ +θ y]
2 20 21
′ ′ ′
h = a[θ +θ y], (4.3)
3 30 31
and
′ ′ ′ ′ ′ ′ ′ ′
y =ϕ +ϕ h +ϕ h +ϕ h . (4.4)
0 1 1 2 2 3 3
With ReLU activations, this model also describes a family of piecewise linear functions.
However, the number of linear regions is potentially greater than for a shallow network
with six hidden units. To see this, consider choosing the first network to produce three
Problem4.1
alternating regions of positive and negative slope (figure 4.1b). This means that three
differentrangesofxaremappedtothesameoutputrangey ∈[−1,1],andthesubsequent
mapping from this range of y to y′ is applied three times. The overall effect is that the
Notebook4.1
function defined by the second network is duplicated three times to create nine linear
Composing
regions. The same principle applies in higher dimensions (figure 4.2). networks
A different way to think about composing networks is that the first network “folds”
the input space x back onto itself so that multiple inputs generate the same output.
Then the second network applies a function, which is replicated at all points that were
folded on top of one another (figure 4.3).
4.2 From composing networks to deep networks
The previous section showed that we could create complex functions by passing the
output of one shallow neural network into a second network. We now show that this is
a special case of a deep network with two hidden layers.
The output of the first network (y = ϕ +ϕ h +ϕ h +ϕ h ) is a linear combina-
0 1 1 2 2 3 3
tion of the activations at the hidden units. The first operations of the second network
(equation 4.3 in which we calculate θ′ +θ′ y, θ′ +θ′ y, and θ′ +θ′ y) are linear in
10 11 20 21 30 31
the output of the first network. Applying one linear function to another yields another
linear function. Substituting the expression for y into equation 4.3 gives:
h ′ = a[θ′ +θ′ y] = a[θ ′ +θ ′ ϕ +θ ′ ϕ h +θ ′ ϕ h +θ ′ ϕ h ]
1 10 11 10 11 0 11 1 1 11 2 2 11 3 3
h ′ = a[θ′ +θ′ y] = a[θ ′ +θ ′ ϕ +θ ′ ϕ h +θ ′ ϕ h +θ ′ ϕ h ]
2 20 21 20 21 0 21 1 1 21 2 2 21 3 3
h ′ = a[θ′ +θ′ y] = a[θ ′ +θ ′ ϕ +θ ′ ϕ h +θ ′ ϕ h +θ ′ ϕ h ], (4.5)
3 30 31 30 31 0 31 1 1 31 2 2 31 3 3
which we can rewrite as:
′
h = a[ψ +ψ h +ψ h +ψ h ]
1 10 11 1 12 2 13 3
′
h = a[ψ +ψ h +ψ h +ψ h ]
2 20 21 1 22 2 23 3
′
h = a[ψ +ψ h +ψ h +ψ h ], (4.6)
3 30 31 1 32 2 33 3
Draft: please send errata to udlbookmail@gmail.com.

44 4 Deep neural networks
Figure 4.2 Composing neural networks with a 2D input. a) The first network
(fromfigure3.8)hasthreehiddenunitsandtakestwoinputsx andx andreturns
1 2
ascalaroutputy. Thisispassedintoasecondnetworkwithtwohiddenunitsto
produce y′. b) The first network produces a function consisting of seven linear
regions,oneofwhichisflat. c)Thesecondnetworkdefinesafunctioncomprising
two linear regions in y ∈[−1,1]. d) When these networks are composed, each of
the six non-flat regions from the first networkis divided intotwo new regions by
the second network to create a total of 13 linear regions.
Figure 4.3 Deep networks as folding input space. a) One way to think about
the first network from figure 4.1 is that it “folds” the input space back on top
of itself. b) The second network applies its function to the folded space. c) The
final output is revealed by “unfolding” again.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

4.3 Deep neural networks 45
Figure 4.4 Neural network with one input, one output, and two hidden layers,
each containing three hidden units.
where ψ = θ′ +θ′ ϕ ,ψ = θ′ ϕ ,ψ = θ′ ϕ and so on. The result is a network
10 10 11 0 11 11 1 12 11 2
with two hidden layers (figure 4.4).
Itfollowsthatanetworkwithtwolayerscanrepresentthefamilyoffunctionscreated
by passing the output of one single-layer network into another. In fact, it represents a
broader family because in equation 4.6, the nine slope parameters ψ ,ψ ,...,ψ can
11 21 33
take arbitrary values, whereas, in equation 4.5, these parameters are constrained to be
the outer product [θ′ ,θ′ ,θ′ ]T[ϕ ,ϕ ,ϕ ].
11 21 31 1 2 3
4.3 Deep neural networks
Intheprevioussection, weshowedthatcomposingtwoshallownetworksyieldsaspecial
case of a deep network with two hidden layers. Now we consider the general case of a
deep network with two hidden layers, each containing three hidden units (figure 4.4).
The first layer is defined by:
h = a[θ +θ x]
1 10 11
h = a[θ +θ x]
2 20 21
h = a[θ +θ x], (4.7)
3 30 31
the second layer by:
′
h = a[ψ +ψ h +ψ h +ψ h ]
1 10 11 1 12 2 13 3
′
h = a[ψ +ψ h +ψ h +ψ h ]
2 20 21 1 22 2 23 3
′
h = a[ψ +ψ h +ψ h +ψ h ], (4.8)
3 30 31 1 32 2 33 3
and the output by:
′ ′ ′ ′ ′ ′ ′ ′
y =ϕ +ϕ h +ϕ h +ϕ h . (4.9)
0 1 1 2 2 3 3
Draft: please send errata to udlbookmail@gmail.com.