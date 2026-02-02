# Chapter 10: argued applying fully connected networks to images isn’t sensible because

*Pages: 263-282*

---

13.4 Graph convolutional networks 249
13.4.1 Equivariance and invariance
We noted before that the indexing of the nodes in the graph is arbitrary, and any
permutation of the node indices does not change the graph. It is hence imperative that
any model respects this property. It follows that each layer must be equivariant (see
section 10.1) with respect to permutations of the node indices. In other words, if we
permute the node indices, the node embeddings at each stage will be permuted in the
same way. In mathematical terms, if P is a permutation matrix, then we must have:
H P=F[H P,PTAP,ϕ ]. (13.6)
k+1 k k
For node classification and edge prediction tasks, the output should also be equiv-
ariant with respect to permutations of the node indices. However, for graph-level tasks,
the final layer aggregates information from across the graph, so the output is invariant
Problem13.6
to the node order. In fact, the output layer from equation 13.2 achieves this because:
y =sig[β +ω H 1/N]=sig[β +ω H P1/N], (13.7)
K K K K K K
for any permutation matrix P (see problem 13.6).
This mirrors the case for images, where segmentation should be equivariant to geo-
metric transformations, and image classification should be invariant (figure 10.1). Here,
convolutional and pooling layers partially achieve this with respect to translations, but
there is no known way to guarantee these properties exactly for more general transfor-
mations. However, for graphs, it is possible to define networks that ensure equivariance
or invariance to permutations.
13.4.2 Parameter sharing
Chapter 10 argued applying fully connected networks to images isn’t sensible because
this requires the network to learn how to recognize an object separately at every image
position. Instead,weusedconvolutionallayersthatprocessedeverypositionintheimage
identically. This reduced the number of parameters and introduced an inductive bias
that forced the model to treat every part of the image in the same way.
The same argument can be made about nodes in a graph. We could learn a model
with separate parameters associated with each node. However, now the network must
independently learn the meaning of the connections in the graph at each position, and
training would require many graphs with the same topology. Instead, we build a model
that uses the same parameters at every node, reducing the number of parameters and
sharing what the network learns at each node across the entire graph.
Recall that a convolution (equation 10.3) updates a variable by taking a weighted
sum of information from its neighbors. One way to think of this is that each neighbor
sends a message to the variable of interest, which aggregates these messages to form the
update. When we considered images, the neighbors were pixels from a fixed-size square
region around the current position, so the spatial relationships at each position are the
same. However, in a graph, each node may have a different number of neighbors, and
there are no consistent relationships; there is no sense that we can weight information
Draft: please send errata to udlbookmail@gmail.com.

250 13 Graph neural networks
Figure 13.7 Simple Graph CNN layer. a) Input graph consists of structure (em-
bodied in graph adjacency matrix A, not shown) and node embeddings (stored
in columns of X). b) Each node in the first hidden layer is updated by (i) ag-
gregating the neighboring nodes to form a single vector, (ii) applying a linear
transformationΩ totheaggregatedvector,(iii)applyingthesamelineartrans-
0
formation Ω to the original node, (iv) adding these together with a bias β ,
0 0
andfinally(v)applyinganonlinearactivationfunctiona[•]likeaReLU.c)This
process is repeated at subsequent layers (but with different parameters for each
layer) until we produce the final embeddings at the end of the network.
from a node that is “above” the node of interest differently to information from a node
that is “below” it.
13.4.3 Example GCN layer
TheseconsiderationsleadtoasimpleGCNlayer(figure13.7). Ateachnodeninlayerk,
weaggregateinformationfromneighboringnodesbysummingtheirnodeembeddingsh•:
X
agg[n,k]= h(m), (13.8)
k
m∈ne[n]
where ne[n] returns the set of indices of the neighbors of node n. Then we apply a
linear transformation Ω to the embedding h(n) at the current node and to this ag-
k k
gregated value, add a bias term β , and pass the result through a nonlinear activation
k
function a[•], which is applied independently to every member of its vector argument:
h i
h(n) =a β +Ω ·h(n)+Ω ·agg[n,k] . (13.9)
k+1 k k k k
We can write this more succinctly by noting that post-multiplication of a matrix
by a vector returns a weighted sum of its columns. The nth column of the adjacency
matrix A contains ones at the positions of the neighbors. Hence, if we collect the node
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

13.5 Example: graph classification 251
embeddings into the D×N matrix H and post-multiply by the adjacency matrix A,
k
the nth column of the result is agg[n,k]. The update for the nodes is now:
(cid:2) (cid:3)
H = a β 1T +Ω H +Ω H A
k+1 (cid:2) k k k k (cid:3)k
= a β 1T +Ω H (A+I) , (13.10)
k k k
where 1 is an N×1 vector containing ones. Here, the nonlinear activation function a[•]
is applied independently to every member of its matrix argument.
This layer satisfies the design considerations: it is equivariant to permutations of the
Problem13.7
node indices, can cope with any number of neighbors, exploits the graph structure to
provide a relational inductive bias, and shares parameters throughout the graph.
13.5 Example: graph classification
We now combine these ideas to describe a network that classifies molecules as toxic or
harmless. The network inputs are the adjacency matrix and node embedding matrix X.
The adjacency matrix A ∈ RN×N derives from the molecular structure. The columns Notebook13.2
Graphclassification
of the node embedding matrix X ∈ R118×N are one-hot vectors indicating which of the
118elementsoftheperiodictablearepresent. Inotherwords, theyarevectorsoflength
118 where every position is zero except for the position corresponding to the relevant
element, which is set to one. The node embeddings can be transformed to an arbitrary
size D by the first weight matrix Ω ∈RD×118.
0
The network equations are:
(cid:2) (cid:3)
H = a β 1T +Ω X(A+I)
1 (cid:2) 0 0 (cid:3)
H = a β 1T +Ω H (A+I)
2 1 1 1
. .
. .
. = .
(cid:2) (cid:3)
H K = a β K−1 1T +Ω K−1 H k−1 (A+I)
f[X,A,Φ] = sig[β +ω H 1/N], (13.11)
K K K
where the network output f[X,A,Φ] is a single value that determines the probability
that the molecule is toxic (see equation 13.2).
13.5.1 Training with batches
Given I training graphs {X ,A } and their labels y , the parameters Φ = {β ,Ω }K
i i i k k k=0
can be learned using SGD and the binary cross-entropy loss (equation 5.19). Fully
connectednetworks,convolutionalnetworks,andtransformersallexploittheparallelism
ofmodernhardwaretoprocessanentirebatchoftrainingexamplesconcurrently. Tothis
end,thebatchelementsareconcatenatedintoahigher-dimensionaltensor(section7.4.2).
Draft: please send errata to udlbookmail@gmail.com.

252 13 Graph neural networks
Figure13.8Inductivevs.transductiveproblems. a)Nodeclassificationtaskinthe
inductive setting. We are given a set of I training graphs, where the node labels
(orange and cyan colors) are known. After training, we are given a test graph
and must assign labels to each node. b) Node classification in the transductive
setting. There is one large graph in which some nodes have labels (orange and
cyan colors), and others are unknown. We train the model to predict the known
labels correctly and then examine the predictions at the unknown nodes.
However, each graph may have a different number of nodes. Hence, the matrices X
i
and A have different sizes, and there is no way to concatenate them into 3D tensors.
i
Luckily, a simple trick allows us to process the whole batch in parallel. The graphs
inthebatcharetreatedasdisjointcomponentsofasinglelargegraph. Thenetworkcan
then be run as a single instance of the network equations. The mean pooling is carried
out only over the individual graphs to make a single representation per graph that can
be fed into the loss function.
13.6 Inductive vs. transductive models
Untilthispoint,allofthemodelsinthisbookhavebeeninductive: weexploitatraining
set of labeled data to learn the relation between the inputs and outputs. Then we apply
thistonewtestdata. Onewaytothinkofthisisthatwearelearningtherulethatmaps
inputs to outputs and then applying it elsewhere.
By contrast, a transductive model considers both the labeled and unlabeled data
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

13.7 Example: node classification 253
at the same time. It does not produce a rule but merely a labeling for the unknown
outputs. This is sometimes termed semi-supervised learning. It has the advantage that
it can use patterns in the unlabeled data to help make its decisions. However, it has the
disadvantagethatthemodelneedstoberetrainedwhenextraunlabeleddataareadded.
Both problem types are commonly encountered for graphs (figure 13.8). Sometimes,
we have many labeled graphs and learn a mapping between the graph and the labels.
For example, we might have many molecules, each labeled according to whether it is
toxictohumans. Welearntherulethatmapsthegraphtothetoxic/non-toxiclabeland
then apply this rule to new molecules. However, sometimes there is a single monolithic
graph. Inthegraphofscientificpapercitations,wemighthavelabelsindicatingthefield
(physics, biology, etc.) for some nodes and wish to label the remaining nodes. Here, the
training and test data are irrevocably connected.
Graph-leveltasksonlyoccurintheinductivesettingwheretherearetrainingandtest
graphs. However, node-level tasks and edge prediction tasks can occur in either setting.
In the transductive case, the loss function minimizes the mismatch between the model
output and the ground truth where this is known. New predictions are computed by
running the forward pass and retrieving the results where the ground truth is unknown.
13.7 Example: node classification
Asasecondexample,considerabinarynodeclassificationtaskinatransductivesetting.
Westartwithacommercial-sizedgraphwithmillionsofnodes. Somenodeshaveground
truth binary labels, and the goal is to label the remaining unlabeled nodes. The body
of the network will be the same as in the previous example (equation 13.11) but with a
different final layer that produces an output vector of size 1×N:
(cid:2) (cid:3)
f[X,A,Φ]=sig β 1T +ω H , (13.12)
K K K
where the function sig[•] applies the sigmoid function independently to every element
of the row vector input. As usual, we use the binary cross-entropy loss, but now only
at nodes where we know the ground truth label y. Note that equation 13.12 is just a
vectorized version of the node classification loss from equation 13.3.
Training this network raises two problems. First, it is logistically diﬀicult to train a
graph neural network of this size. Consider that we must store the node embeddings at
every network layer in the forward pass. This will involve both storing and processing
a structure several times the size of the entire graph, and this may not be practical.
Second, we have only a single graph, so it’s not obvious how to perform stochastic
gradient descent. How can we form a batch if there is only a single object?
13.7.1 Choosing batches
One way to form a batch is to choose a random subset of labeled nodes at each training
step. Each node depends on its neighbors in the previous layer. These, in turn, depend
Draft: please send errata to udlbookmail@gmail.com.

254 13 Graph neural networks
Figure 13.9Receptivefieldsingraphneuralnetworks. Considertheorangenode
in hidden layer two (right). This receives input from the nodes in the 1-hop
neighborhood in hidden layer one (shaded region in center). These nodes in
hiddenlayeronereceiveinputsfromtheirneighborsinturn,andtheorangenode
in layer two receives inputs from all the input nodes in the 2-hop neighborhood
(shaded area on left). The region of the graph that contributes to a given node
is equivalent to the notion of a receptive field in convolutional neural networks.
on their neighbors in the layer before, so (similarly to convolutional networks) each
node has a receptive field (figure 13.9). The receptive field region is termed the k-hop
neighborhood. We can hence perform a gradient descent step using the graph that forms
the union of the k-hop neighborhoods of the batch nodes; the remaining inputs do not
contribute.
Unfortunately, if there are many layers and the graph is densely connected, every
input node may be in the receptive field of every output, and this may not reduce the
graph size at all. This is known as the graph expansion problem. Two approaches that
tackle this problem are neighborhood sampling and graph partitioning.
Neighborhood sampling: Thefullgraphthatfeedsintothebatchofnodesissampled,
thereby reducing the connections at each network layer (figure 13.10). For example, we
mightstartwiththebatchnodesandrandomlysampleafixednumberoftheirneighbors
Notebook13.3
in the previous layer. Then, we randomly sample a fixed number of their neighbors in
Neighborhood
sampling the layer before, and so on. The graph still increases in size with each layer but in
a much more controlled way. This is done anew for each batch, so the contributing
neighbors differ even if the same batch is drawn twice. This is also reminiscent of
dropout (section 9.3.3) and adds some regularization.
Graph partitioning: A second approach is to cluster the original graph into disjoint
subsets of nodes (i.e., smaller graphs that are not connected to one another) before
processing (figure 13.11). There are standard algorithms to choose these subsets to
maximize the number of internal links. These smaller graphs can each be treated as
batches, or a random subset of them can be combined to form a batch (reinstating any
edges between them from the original graph).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

13.7 Example: node classification 255
Figure 13.10 Neighborhood sampling. a) One way of forming batches on large
graphs is to choose a subset of labeled nodes in the output layer (here, just one
nodeinlayertwo,right)andthenworkingbacktofindallofthenodesintheK-
hop neighborhood (receptive field). Only this sub-graph is needed to train this
batch. Unfortunately, if the graph is densely connected, this may retain a large
proportion of the graph. b) One solution is neighborhood sampling. As we work
back from the final layer, we select a subset of neighbors (here, three) in the
layer before and a subset of the neighbors of these in the layer before that. This
restrictsthesizeofthegraphfortrainingthebatch. Inallpanels,thebrightness
represents the distance from the original node.
Draft: please send errata to udlbookmail@gmail.com.

256 13 Graph neural networks
Figure 13.11 Graph partitioning. a) Input graph. b) The input graph is parti-
tionedintosmallersubgraphsusingaprincipledmethodthatremovesthefewest
edges. c-d)Wecannowusethesesubgraphsasbatchestotraininatransductive
setting, so here, there are four possible batches. e) Alternatively, we can use
combinations of the subgraphs as batches, reinstating the edges between them.
If we use pairs of subgraphs, there would be six possible batches here.
Given one of the above methods to form batches, we can now train the network
parameters in the same way as for the inductive setting, dividing the labeled nodes into
train, test, and validation sets as desired; we have effectively converted a transductive
problem to an inductive one. To perform inference, we compute predictions for the
unknownnodesbasedontheirk-hopneighborhood. Unliketraining,thisdoesnotrequire
storing the intermediate representations, so it is much more memory eﬀicient.
13.8 Layers for graph convolutional networks
Inthepreviousexamples,wecombinedmessagesfromadjacentnodesbysummingthem
togetherwiththetransformedcurrentnode. Thiswasaccomplishedbypost-multiplying
thenodeembeddingmatrixHbytheadjacencymatrixplustheidentityA+I. Wenow
considerdifferentapproachestoboth(i)thecombinationofthecurrentembeddingwith
the aggregated neighbors and (ii) the aggregation process itself.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

13.8 Layers for graph convolutional networks 257
13.8.1 Combining current node and aggregated neighbors
In the example GCN layer above, we combined the aggregated neighbors HA with the
current nodes H by just summing them:
h i
H =a β 1T +Ω H (A+I) . (13.13)
k+1 k k k
In another variation, the current node is multiplied by a factor of (1+ϵ ) before con-
k
tributing to the sum, where ϵ is a learned scalar that is different for each layer:
k
h i
H =a β 1T +Ω H (A+(1+ϵ )I) . (13.14)
k+1 k k k k
This is known as diagonal enhancement. A related variation applies a different linear
transform Ψ to the current node:
k
(cid:2) (cid:3)
H = a β 1T +Ω H A+Ψ H
k+1 (cid:20) k k k (cid:20) k k(cid:21)(cid:21)
(cid:2) (cid:3)
H A
= a β 1T + Ω Ψ k
k k k H
(cid:20) (cid:20) (cid:21)(cid:21) k
= a β 1T +Ω ′ H k A , (13.15)
k k H
k
(cid:2) (cid:3)
′
where we have defined Ω = Ω Ψ in the third line.
k k k
13.8.2 Residual connections
With residual connections, the aggregated representation from the neighbors is trans-
formed and passed through the activation function before summation or concatenation
with the current node. For the latter case, the associated network equations are:
(cid:20) (cid:2) (cid:3)(cid:21)
a β 1T +Ω H A
H = k k k . (13.16)
k+1 H
k
13.8.3 Mean aggregation
Theabovemethodsaggregatetheneighborsbysummingthenodeembeddings. However,
it’s possible to combine the embeddings in differentways. Sometimes it’s better to take
the average of the neighbors rather than the sum; this can be superior if the embedding
informationismoreimportantandthestructuralinformationlesssosincethemagnitude
of the neighborhood contributions will not depend on the number of neighbors:
X
1
agg[n]= h , (13.17)
|ne[n]| m
m∈ne[n]
Draft: please send errata to udlbookmail@gmail.com.

258 13 Graph neural networks
where as before, ne[n] denotes a set containing the indices of the neighbors of the nth
node. Equation 13.17 can be computed neatly in matrix form by introducing the diago-
nal N×N degree matrix D. Each non-zero element of this matrix contains the number
Problem13.8 ofneighborsfortheassociatednode. Itfollowsthateachdiagonalelementintheinverse
matrix D−1 contains the denominator that we need to compute the average. The new
GCN layer can be written as:
h i
H =a β 1T +Ω H (AD −1+I) . (13.18)
k+1 k k k
13.8.4 Kipf normalization
There are many variations of graph neural networks based on mean aggregation. Some-
times the current node is included with its neighbors in the mean computation rather
than treated separately. In Kipf normalization, the sum of the node representations is
Problem13.9
normalized as:
X
h
agg[n]= p m , (13.19)
|ne[n]||ne[m]|
m∈ne[n]
withthelogicthatinformationcomingfromnodeswithaverylargenumberofneighbors
should be down-weighted since there are many connections and they provide less unique
information. This can also be expressed in matrix form using the degree matrix:
h i
H =a β 1T +Ω H (D −1/2AD −1/2+I) . (13.20)
k+1 k k k
13.8.5 Max pooling aggregation
Analternativeoperationthatisalsoinvarianttopermutationiscomputingthemaximum
of a set of objects. The max pooling aggregation operator is:
(cid:2) (cid:3)
agg[n]= max h , (13.21)
m∈ne[n] m
where the operator max[•] returns the element-wise maximum of the vectors h that
m
are neighbors to the current node n.
13.8.6 Aggregation by attention
Theaggregationmethodsdiscussedsofareitherweightthecontributionoftheneighbors
equally or in a way that depends on the graph topology. Conversely, in graph attention
layers, the weights depend on the data at the nodes. A linear transform is applied to
the current node embeddings so that:
H ′ =β 1T +Ω H . (13.22)
k k k k
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

13.8 Layers for graph convolutional networks 259
Figure13.12Comparisonofgraphconvolutionalnetwork,dotproductattention,
and graph attention network. In each case, the mechanism maps N embeddings
of size D stored in a D×N matrix X to an output of the same size. a) The
graph convolutional network applies a linear transformation X′ = ΩX to the
data matrix. It then computes a weighted sum of the transformed data, where
theweightingisbasedontheadjacencymatrix. Abiasβisadded,andtheresult
ispassedthroughanactivationfunction. b)Theoutputsofthedot-productself-
attentionmechanisminthetransformerarealsoweightedsumsofthetransformed
inputs, but this time the weights depend on the data itself via the attention
matrix. c)Thegraphattentionnetworkcombinesbothofthesemechanisms;the
weights are both computed from the data and based on the adjacency matrix.
Draft: please send errata to udlbookmail@gmail.com.

260 13 Graph neural networks
Then the similarity s of each transformed node embedding h′ to the transformed
mn m
node embedding h′ is computed by concatenating the pairs, taking a dot product with
n
a column vector ϕ of learned parameters, and applying an activation function:
k
(cid:20) (cid:20) (cid:21)(cid:21)
h′
s =a ϕT m . (13.23)
mn k h′
n
These variables are stored in an N×N matrix S, where each element represents the
similarity of every node to every other. As in dot-product self-attention, the attention
weights contributing to each output embedding are normalized to be positive and sum
to one using the softmax operation. However, only those values corresponding to the
current node and its neighbors should contribute. The attention weights are applied to
the transformed embeddings:
h i
H =a H ′ ·Softmask[S,A+I] , (13.24)
k+1 k
where a[•] is a second activation function. The function Softmask[•,•] computes the
attention values by applying softmax operation separately to each column of its first
argument S, but only after setting values where the second argument A+I is zero to
negative infinity, so they do not contribute. This ensures that the attention to non-
neighboring nodes is zero.
This is very similar to the dot-product self-attention computation in transformers
Notebook13.4
(seefigure13.12), exceptthat(i)Thekeys, queries, andvaluesareallthesame, (ii)The
Graph
attention measure of similarity is different, and (iii) The attentions are masked so that each node
onlyattendstoitselfanditsneighbors. Asintransformers, thissystemcanbeextended
Problem13.10 to use multiple heads that are run in parallel and recombined.
13.9 Edge graphs
Until now, we have focused on processing node embeddings. These evolve as they are
passed through the network so that by the end of the network, they represent both the
node and its context in the graph. We now consider the case where the information is
associated with the edges of the graph.
It is easy to adapt the machinery for node embeddings to process edge embeddings
using the edge graph (also known as the adjoint graph or line graph). This is a com-
plementary graph, in which each edge in the original graph becomes a node, and every
two edges with a common node in the original graph create an edge in the new graph
(figure 13.13). In general, a graph can be recovered from its edge graph, so it’s possible
to swap between these two representations.
Problems13.11–13.13
To process edge embeddings, the graph is translated to its edge graph. Then we
use exactly the same techniques, aggregating information at each new node from its
neighbors and combining this with the current representation. When both node and
edge embeddings are present, we can translate back and forth between the two graphs.
Now there are four possible updates (nodes update nodes, nodes update edges, edges
update nodes, and edges update edges), and these can be alternated as desired, or with
Problem13.14
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

13.10 Summary 261
Figure 13.13Edgegraph. a)Graphwithsixnodes. b)Tocreatetheedgegraph,
we assign one node for each original edge (cyan circles), and c) connect the new
nodesiftheedgestheyrepresentconnecttothesamenodeintheoriginalgraph.
minor modifications, nodes can be updated simultaneously from both nodes and edges.
13.10 Summary
Graphsconsistofasetofnodes,wherepairsofthesenodesareconnectedbyedges. Both
nodes and edges can have data attached, and these are referred to as node embeddings
andedgeembeddings,respectively. Manyreal-worldproblemscanbeframedintermsof
graphs, where the goal is to establish a property of the entire graph, properties of each
node or edge, or the presence of additional edges in the graph.
Graphneuralnetworksaredeeplearningmodelsthatareappliedtographs. Sincethe
nodeorderingraphsisarbitrary,thelayersofgraphneuralnetworksmustbeequivariant
to permutations of the node indices. Spatial-based convolutional networks are a family
of graph neural networks that aggregate information from the neighbors of a node and
then use this to update the node embeddings.
Onechallengeofprocessinggraphsisthattheyoftenoccurinthetransductivesetting,
where there is only one partially labeled graph rather than sets of training and test
graphs. This graph can be extremely large, which adds further challenges in terms of
training and has led to sampling and partitioning algorithms. The edge graph has a
node for every edge in the original graph. By converting to this representation, graph
neural networks can be used to update the edge embeddings.
Draft: please send errata to udlbookmail@gmail.com.

262 13 Graph neural networks
Notes
Sanchez-Lengeling et al. (2021) and Daigavane et al. (2021) present good introductory articles
ongraphprocessingusingneuralnetworks. Recentsurveysofresearchingraphneuralnetworks
can be found in articles by Zhou et al. (2020a), Wu et al. (2020c), and Veličković (2023), and
thebooksofHamilton(2020)and Ma&Tang(2021). GraphEDM(Chamietal.,2020)unifies
manyexistinggraphalgorithmsintoasingleframework. Inthischapter,wehaverelatedgraphs
to convolutional networks following Bruna et al. (2013), but there are also strong connections
withbeliefpropagation(Daietal.,2016)andgraphisomorphismtests(Hamiltonetal.,2017a).
Zhang et al. (2019c) provide a review focusing specifically on graph convolutional networks.
Bronsteinetal.(2021)provideageneraloverviewofgeometricdeeplearning,includinglearning
on graphs. Loukas (2020) discusses what types of functions graph neural networks can learn.
Applications: Applicationsincludegraphclassification(e.g.,Zhangetal.,2018b),nodeclas-
sification (e.g., Kipf & Welling, 2017), edge prediction (e.g., Zhang & Chen, 2018), graph clus-
tering (e.g., Tsitsulin et al., 2020), and recommender systems (e.g., Wu et al., 2023). Methods
for node classification are reviewed by Xiao et al. (2022a), methods for graph classification by
Errica et al. (2019), and methods for edge prediction by Mutlu et al. (2020) and Kumar et al.
(2020a).
Graph neural networks: Graph neural networks were introduced by Gori et al. (2005) and
Scarselli et al. (2008), who formulated them as a generalization of recursive neural networks.
The latter model used the iterative update:
(cid:2) (cid:3)
h n ←f x n ,x m∈ne[n] ,e e∈nee[n] ,h m∈ne[n] ,ϕ , (13.25)
in which each node embedding h is updated from the initial embedding x , initial embed-
n n
dings x m∈ne[n] at the adjacent nodes, initial embeddings e e∈nee[n] at the adjacent edges, and
adjacent node embeddings h m∈ne[n] . For convergence, the function f[•,•,•,•,ϕ] must be a
contraction mapping (see figure 16.9). If we unroll this equation in time for K steps and allow
differentparametersϕ ateachtimeK,thenequation13.25becomessimilartothegraphcon-
k
volutional network. Subsequent work extended graph neural networks to use gated recurrent
units (Li et al., 2016b) and long short-term memory networks (Selsam et al., 2019).
Spectral methods: Bruna et al. (2013) applied the convolution operation in the Fourier
domain. TheFourierbasisvectorscanbefoundbytakingtheeigendecompositionofthegraph
Laplacian matrix, L = D−A where D is the degree matrix and A is the adjacency matrix.
This has disadvantages: the filters are not localized, and the decomposition is prohibitively
expensiveforlargegraphs. Henaffetal.(2015)tackledthefirstproblembyforcingtheFourier
representation to be smooth (and hence the spatial domain to be localized). Defferrard et al.
(2016) introduced ChebNet, which approximates the filters eﬀiciently by using the recursive
propertiesofChebyshevpolynomials. Thisbothprovidesspatiallylocalizedfiltersandreduces
thecomputation. Kipf&Welling(2017)simplifiedthisfurthertoconstructfiltersthatuseonly
a 1-hop neighborhood, resulting in a formulation similar to the spatial methods described in
this chapter and providing a bridge between spectral and spatial methods.
Spatial methods: Spectral methods are ultimately based on the Graph Laplacian, so if the
graph changes, the model must be retrained. This problem spurred the development of spatial
methods. Duvenaud et al. (2015) defined convolutions in the spatial domain, using a different
weight matrix to combine the adjacent embeddings for each node degree. This has the disad-
vantage that it becomes impractical if some nodes have a very large number of connections.
Diffusion convolutional neural networks (Atwood & Towsley, 2016) use powers of the normal-
izedadjacencymatrixtoblendfeaturesacrossdifferentscales,sumthese,pointwisemultiplyby
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 263
weights, and pass through an activation function to create the node embeddings. Gilmer et al.
(2017) introduced message-passing neural networks, which defined convolutions on the graph
as propagating messages from spatial neighbors. The “aggregate and combine” formulation of
GraphSAGE (Hamilton et al., 2017a) fits into this framework.
Aggregate and combine: Graph convolutional networks (Kipf & Welling, 2017) take a
weighted average of the neighbors and current node and then apply a linear mapping and
ReLU. GraphSAGE (Hamilton et al., 2017a) applies a neural network layer to each neighbor,
takingtheelementwisemaximumtoaggregate. Chiangetal.(2019)proposediagonal enhance-
ment in which the previous embedding is weighted more than the neighbors. Kipf & Welling
(2017)introducedKipfnormalization,whichnormalizesthesumoftheneighboringembeddings
based on the degrees of the current node and its neighbors (see equation 13.19).
ThemixturemodelnetworkorMoNet(Montietal.,2017)takesthisonestepfurtherbylearning
aweightingbasedonthedegreesofthecurrentnodeandtheneighbor. Theyassociateapseudo-
coordinate system with each node, where the positions of the neighbors depend on these two
quantities. TheythenlearnacontinuousfunctionbasedonamixtureofGaussiansandsample
this at the pseudo-coordinates of the neighbors to get the weights. In this way, they can learn
the weightings for nodes and neighbors with arbitrary degrees. Pham et al. (2017) use a linear
interpolation of the node embedding and neighbors with a different weighted combination for
each dimension. The weight of this gating mechanism is generated as a function of the data.
Higher-order convolutional layers: Zhou & Li (2017) used higher-order convolutions by
replacing the adjacency matrix A with A˜ = Min[AL+I,1] where L is the maximum walk-
length, 1isamatrixcontainingonlyones, and Min[•]takesthepointwiseminimumofitstwo
matrix arguments; the updates now sum together contributions from any nodes where there is
at least one walk of length L. Abu-El-Haija et al. (2019) proposed MixHop, which computes
nodeupdatesfromtheneighbors(usingtheadjacencymatrixA),theneighborsoftheneighbors
(usingA2),andsoon. Theyconcatenatetheseupdatesateachlayer. Leeetal.(2018)combined
informationfromnodesbeyondtheimmediateneighborsusinggeometricmotifs,whicharesmall
local geometric patterns in the graph (e.g., a fully connected clique of five nodes).
Residual connections: Kipf & Welling (2017) proposed a residual connection in which the
original embeddings are added to the updated ones. Hamilton et al. (2017b) concatenate the
previous embedding to the output of the next layer (see equation 13.16). Rossi et al. (2020)
present an inception-style network where the node embedding is concatenated to not only the
aggregation of its neighbors but also the aggregation of all neighbors within a walk of two (via
computing powers of the adjacency matrix). Xu et al. (2018) introduced jump knowledge con-
nections in which the final output at each node consists of the concatenated node embeddings
throughout the network. Zhang & Meng (2019) present a general formulation of residual em-
beddings called GResNet and investigate several variations in which the embeddings from the
previous layer are added, the input embeddings are added, or versions of these that aggregate
information from their neighbors (without further transformation) are added.
Attention in graph neural networks: Veličkovićetal.(2019)developedthegraphattention
network (figure 13.12c). Their formulation uses multiple heads whose outputs are combined
symmetrically. GatedAttentionNetworks(Zhangetal.,2018a)weighttheoutputofthedifferent
heads in a way that depends on the data itself. Graph-BERT (Zhang et al., 2020) performs
nodeclassificationusingself-attentionalone;thegraph’sstructureiscapturedbyaddingposition
embeddings to the data, similarly to how the absolute or relative position of words is captured
inthetransformer(chapter12). Forexample,theyaddpositionalinformationthatdependson
the number of hops between nodes in the graph.
Draft: please send errata to udlbookmail@gmail.com.

264 13 Graph neural networks
Permutation invariance: InDeepSets,Zaheeretal.(2017)presentedageneralpermutation
invariantoperatorforprocessingsets. Janossypooling(Murphyetal.,2018)acceptsthatmany
functions are not permutation equivariant and instead uses a permutation-sensitive function
and averages the results across many permutations.
Edge graphs: The notation of the edge graph, line graph, or adjoint graph dates to Whitney
(1932). The idea of “weaving” layers that update node embeddings from node embeddings,
node embeddings from edge embeddings, edge embeddings from edge embeddings, and edge
embeddings from node embeddings was proposed by Kearnes et al. (2016). However, here the
node-nodeandedge-edgeupdatesdonotinvolvetheneighbors. Montietal.(2018)introduced
thedual-primalgraphCNN,amodernformulationinaCNNframeworkthatalternatesbetween
updates in the original and edge graphs.
Power of graph neural networks: Xu et al. (2019) argue that a neural network should
be able to distinguish different graph structures; it is undesirable to map two graphs to the
same output if they have the same initial node embeddings but different adjacency matrices.
They identified graph structures that could not be distinguished by previous approaches such
as GCNs (Kipf & Welling, 2017) and GraphSAGE (Hamilton et al., 2017a). They developed a
morepowerfularchitecturewiththesamediscriminativepowerastheWeisfeiler-Lehmangraph
isomorphism test (Weisfeiler & Leman, 1968), which is known to discriminate a broad class of
graphs. This resulting graph isomorphism network was based on the aggregation operation:
2 3
X
h(n) =mlp 4 (1+ϵ )h(n)+ h(m)5 . (13.26)
k+1 k k k
m∈ne[n]
Batches: Theoriginalpaperongraphconvolutionalnetworks(Kipf&Welling,2017)usedfull-
batch gradient descent. This has memory requirements proportional to the number of nodes,
embedding size, and number of layers during training. Since then, three types of methods
have been proposed to reduce the memory requirements and create batches for SGD in the
transductive setting: node sampling, layer sampling, and sub-graph sampling.
Node sampling methods start by randomly selecting a subset of target nodes and then work
back through the network, adding a subset of the nodes in the receptive field at each stage.
GraphSAGE (Hamilton et al., 2017a) proposed a fixed number of neighborhood samples as
in figure 13.10b. Chen et al. (2018b) introduce a variance reduction technique, but this uses
historical activations of nodes and so still has a high memory requirement. PinSAGE (Ying
etal.,2018a)usesrandomwalksfromthetargetnodesandchoosestheKnodeswiththehighest
visit count. This prioritizes ancestors that are more closely connected.
Node sampling still requires increasing numbers of nodes as we pass back through the graph.
Layer sampling methods address this by directly sampling the receptive field in each layer
independently. Examples of layer sampling include FastGCN (Chen et al., 2018a), adaptive
sampling (Huang et al., 2018b), and layer-dependent importance sampling (Zou et al., 2019).
Subgraph sampling methods randomly draw subgraphs or divide the original graph into sub-
graphs. These are then trained as independent data examples. Examples of these approaches
include GraphSAINT (Zeng et al., 2020), which samples sub-graphs during training using ran-
dom walks and then runs a full GCN on the subgraph while also correcting for the bias and
varianceoftheminibatch. Cluster GCN(Chiangetal.,2019)partitionsthegraphintoclusters
(bymaximizingtheembeddingutilizationornumberofwithin-batchedges)inapre-processing
stageandrandomlyselectsclusterstoformminibatches. Tocreatemorerandomness,theytrain
random subsets of these clusters plus the edges between them (see figure 13.11).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 265
Wolfe et al. (2021) proposed a distributed training method that both partitions the graph and
trains narrower GCNs in parallel by partitioning the feature space at different layers. More
information about sampling graphs can be found in Rozemberczki et al. (2020).
Regularization and normalization: Rongetal.(2020)proposedDropEdge,whichrandomly
dropsedgesfromthegraphduringeachtrainingiterationbymaskingtheadjacencymatrix. This
canbedoneforthewholeneuralnetworkordifferentlyineachlayer(layer-wiseDropEdge). Ina
sense,thisissimilartodropoutinthatitbreaksconnectionsintheflowofdata,butitcanalsobe
consideredanaugmentationmethodsincechangingthegraphissimilartoperturbingthedata.
Schlichtkrulletal.(2018),Teruetal.(2020),andVeličkovićetal.(2019)alsoproposedrandomly
dropping edges from the graph as a form of regularization similar to dropout. Node sampling
methods(Hamiltonetal.,2017a;Huangetal.,2018b;Chenetal.,2018a)canalsobeconsidered
regularizers. Hasanzadeh et al. (2020) present a general framework called DropConnect that
unifies many of the above approaches.
There are also many proposed normalization schemes for graph neural networks, including
PairNorm (Zhao & Akoglu, 2020), weight normalization (Oono & Suzuki, 2019), differentiable
group normalization (Zhou et al., 2020b), and GraphNorm (Cai et al., 2021).
Multi-relational graphs: Schlichtkrull et al. (2018) proposed a variation of graph convolu-
tional networks for multi-relational graphs (i.e., graphs with more than one edge type). Their
scheme separately aggregates information from each edge type using different parameters. If
there are many edge types, the number of parameters may become large, and to combat this,
they propose that each edge type uses a different weighting of a basis set of parameters.
Hierarchical representations and pooling: CNNs for image classification gradually de-
crease the representation size but increase the number of channels as the network progresses.
However, the GCNs for graph classification in this chapter maintain the entire graph until the
last layer and then combine all the nodes to compute the final prediction. Ying et al. (2018b)
proposed DiffPool, which clusters graph nodes to make a graph that gets progressively smaller
as the depth increases in a way that is differentiable, and so can be learned. This can be done
basedonthegraphstructurealoneoradaptivelybasedonthegraphstructureandtheembed-
dings. Other pooling methods include SortPool (Zhang et al., 2018b) and self-attention graph
pooling (Lee et al., 2019). A comparison of pooling layers for graph neural networks can be
found in Grattarola et al. (2022). Gao & Ji (2019) propose an encoder-decoder structure for
graphs based on the U-Net (see figure 11.10).
Geometric graphs: TheMoNetmodel(Montietal.,2017)canexploitgeometricinformation
because neighboring nodes have well-defined spatial positions. They learn a mixture of Gaus-
sians function and sample from this based on the relative coordinates of the neighbor. In this
way,theycanweightneighboringnodesbasedontheirrelativepositionsasinstandardconvolu-
tionalneuralnetworks,eventhoughthesepositionsarenotconstant. ThegeodesicCNN(Masci
et al., 2015) and anisotropic CNN (Boscaini et al., 2016) both adapt convolution to manifolds
(i.e., surfaces) as represented by triangular meshes. They locally approximate the surface as a
plane and define a coordinate system on this plane around the current node.
Oversmoothingandsuspendedanimation: Unlikeotherdeeplearningmodels,graphneu-
ralnetworksdidnot,untilrecently,benefitsignificantlyfromincreasingdepth. Indeed,theorig-
inal GCN paper (Kipf & Welling, 2017) and GraphSAGE (Hamilton et al., 2017a) both only
usetwolayers,andChiangetal.(2019)trainedafive-layerCluster-GCNtogetstate-of-the-art
performanceonthePPIdataset. Onepossibleexplanationisover-smoothing(Lietal.,2018c);
at each layer, the network incorporates information from a larger neighborhood, and it may
be that this ultimately results in the dissolution of (important) local information. Indeed (Xu
et al., 2018) prove that the influence of one node on another is proportional to the probability
Draft: please send errata to udlbookmail@gmail.com.

266 13 Graph neural networks
ofreachingthatnodeinaK-steprandomwalk. Thisapproachesthestationarydistributionof
walks over the graph with increasing K, causing the local neighborhood to be washed out.
Alon&Yahav(2021)proposedanotherexplanationforwhyperformancedoesn’timprovewith
networkdepth. Theyarguethataddingdepthallowsinformationtobeaggregatedfromlonger
paths. However,inpractice,theexponentialgrowthinthenumberofneighborsmeansthereis
abottleneckwherebytoomuchinformationis“squashed”intothefixed-sizenodeembeddings.
Ying et al. (2018a) also note that when the depth of the network exceeds a certain limit, the
gradientsnolongerpropagateback,andlearningfailsforboththetrainingandtestdata. They
termthiseffectsuspended animation. Thisissimilartowhenmanylayersarenaïvelyaddedto
convolutionalneuralnetworks(figure11.2). Theyproposeafamilyofresidualconnectionsthat
allowdeepernetworkstobetrained. Vanishinggradients(section7.5)havealsobeenidentified
as a limitation by Li et al. (2021b).
It has recently become possible to train deeper graph neural networks using various forms of
residual connection (Xu et al., 2018; Li et al., 2020a; Gong et al., 2020; Chen et al., 2020b; Xu
etal.,2021a). Lietal.(2021a)trainastate-of-the-artmodelwithmorethan1000layersusing
an invertible network to reduce the memory requirements of training (see chapter 16).
Problems
Problem 13.1 Write out the adjacency matrices for the two graphs in figure 13.14.
Problem 13.2∗ Draw graphs that correspond to the following adjacency matrices:
2 3 2 3
0 1 1 0 0 0 0 0 0 1 1 0 0 1
6 7 6 7
61 0 0 1 1 1 07 60 0 1 1 1 0 07
6 7 6 7
61 0 0 0 0 1 17 61 1 0 0 0 0 07
6 7 6 7
A
1
=60 1 0 0 0 1 17 and A
2
=61 1 0 0 1 1 17.
6 7 6 7
60 1 0 0 0 0 17 60 1 0 1 0 0 17
4 5 4 5
0 1 1 1 0 0 0 0 0 0 1 0 0 1
0 0 1 1 1 0 0 1 0 0 1 1 1 0
Problem13.3∗ Considerthetwographsinfigure13.14. Howmanywaysaretheretowalkfrom
node one to node two in (i) three steps and (ii) seven steps?
Problem 13.4 The diagonal of A2 in figure 13.4c contains the number of edges that connect to
each corresponding node. Explain this phenomenon.
Problem 13.5 What permutation matrix is responsible for the transformation between the
AppendixB.4.4
graphs in figures 13.5a–c and figure 13.5d–f?
Permutation
matrix
Problem 13.6 Prove that:
sig[β +ω H 1]=sig[β +ω H P1], (13.27)
K K K K K K
wherePisanN×N permutationmatrix(amatrixthatisallzerosexceptforexactlyoneentry
in each row and each column, which is one), and 1 is an N ×1 vector of ones.
Problem 13.7∗ Consider the simple GNN layer:
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 267
Figure 13.14 Graphs for problems 13.1, 13.3, and 13.8.
Figure 13.15 Graphs for problems 13.11–13.13.
H = GraphLayer[H ,A]
k+1 (cid:20) (cid:20)k (cid:21)(cid:21)
H
= a β 1T +Ω k , (13.28)
k k H A
k
whereHisaD×N matrixcontainingtheN nodeembeddingsinitscolumns,AistheN×N
adjacency matrix, β is the bias vector, and Ω is the weight matrix. Show that this layer is
equivariant to permutations of the node order so that:
GraphLayer[H ,A]P=GraphLayer[H P,PTAP], (13.29)
k k
where P is an N ×N permutation matrix.
Problem 13.8 What is the degree matrix D for each graph in figure 13.14?
Problem 13.9 The authors of GraphSAGE (Hamilton et al., 2017a) propose a pooling method
in which the node embedding is averaged together with its neighbors so that:
0 1
X
1 @ A
agg[n]= h + h . (13.30)
1+|ne[n]| n m
m∈ne[n]
ShowhowthisoperationcanbecomputedsimultaneouslyforallnodeembeddingsintheD×N
embedding matrix H using linear algebra. You will need to use both the adjacency matrix A
and the degree matrix D.
Problem 13.10∗ Devise a graph attention mechanism based on dot-product self-attention and
draw its mechanism in the style of figure 13.12.
Problem 13.11∗ Draw the edge graph associated with the graph in figure 13.15a.
Draft: please send errata to udlbookmail@gmail.com.

268 13 Graph neural networks
Problem 13.12∗ Draw the node graph corresponding to the edge graph in figure 13.15b.
Problem 13.13 For a general undirected graph, describe how the adjacency matrix of the node
graph relates to the adjacency matrix of the corresponding edge graph.
Problem 13.14∗ Design a layer that updates a node embedding h based on its neighboring
n
node embeddings {h m } m∈ne[n] and neighboring edge embeddings {e m } m∈nee[n] . You should
considerthepossibilitythattheedgeembeddingsarenotthesamesizeasthenodeembeddings.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.