# Chapter 10: described convolutional networks, which specialize in processing regular ar-

*Pages: 254-262*

---

Chapter 13
Graph neural networks
Chapter 10 described convolutional networks, which specialize in processing regular ar-
rays of data (e.g., images). Chapter 12 described transformers, which specialize in pro-
cessing sequences of variable length (e.g., text). This chapter describes graph neural
networks. As the name suggests, these are neural architectures that process graphs (i.e.,
sets of nodes connected by edges).
Therearethreenovelchallengesassociatedwithprocessinggraphs. First,theirtopol-
ogyisvariable,anditishardtodesignnetworksthatarebothsuﬀicientlyexpressiveand
can cope with this variation. Second, graphs may be enormous; a graph representing
connections between users of a social network might have a billion nodes. Third, there
may only be a single monolithic graph available, so the usual protocol of training with
many data examples and testing with new data is not always appropriate.
This chapter starts by presenting real-world examples of graphs. It then describes
how to encode these graphs and how to formulate supervised learning problems for
graphs. Thealgorithmicrequirementsforprocessinggraphsarediscussed,andtheselead
naturally to graph convolutional networks, a particular type of graph neural network.
13.1 What is a graph?
Agraphis averygeneralstructure andconsists ofa setofnodesor vertices, where pairs
ofnodesareconnectedbyedgesorlinks. Graphsaretypicallysparse;onlyasmallsubset
of the possible edges are present.
Some objects in the real world naturally take the form of graphs. For example,
road networks can be considered graphs where the nodes are physical locations, and the
edgesrepresentroadsbetweenthem(figure13.1a). Chemicalmoleculesaresmallgraphs
wherethenodesrepresentatoms,andtheedgesrepresentchemicalbonds(figure13.1b).
Electrical circuits are graphs where the nodes represent components and junctions, and
the edges are electrical connections (figure 13.1c).
Furthermore, many datasets can also be represented by graphs, even if this is not
their obvious surface form. For example:
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

13.1 What is a graph? 241
Figure 13.1 Real-world graphs. Some objects, such as a) road networks, b)
molecules, and c) electrical circuits, are naturally structured as graphs.
• Socialnetworksaregraphswherenodesarepeople,andtheedgesrepresentfriend-
ships between them.
• The scientific literature can be viewed as a graph where the nodes are papers, and
the edges represent citations.
• Wikipedia can be considered a graph where the nodes are articles, and the edges
represent hyperlinks between articles.
• Computer programs can be represented as graphs where the nodes are syntax
tokens (variables at different points in the program flow), and the edges represent
computations involving these variables.
• Geometric point clouds can be represented as graphs. Here, each point is a node
with edges connecting to other nearby points.
• Protein interactions in a cell can be expressed as graphs, where the nodes are the
proteins, and there is an edge between two proteins if they interact.
Inaddition,aset(anunorderedlist)canbetreatedasagraphinwhicheverymember
is a node and connects to every other. An image can be treated as a graph with regular
topology, in which each pixel is a node with edges to the adjacent pixels.
13.1.1 Types of graphs
Graphs can be categorized in various ways. The social network in figure 13.2a contains
undirected edges;eachpairofindividualswithaconnectionbetweenthemhavemutually
agreedtobefriends, sothereisnosensethattherelationshipisdirectional. Incontrast,
the citation network in figure 13.2b contains directed edges. Each paper cites other
papers, and this relationship is inherently one-way.
Figure 13.2c depicts a knowledge graph that encodes a set of facts about objects by
definingrelationsbetweenthem. Technically,thisisadirected heterogeneous multigraph.
Itisheterogeneousbecausethenodescanrepresentdifferenttypesofentities(e.g.,people,
countries,companies). Itisamultigraphbecausetherecanbemultipleedgesofdifferent
types between any two nodes.
Draft: please send errata to udlbookmail@gmail.com.

242 13 Graph neural networks
Figure 13.2 Types of graphs. a) A social network is an undirected graph; the
connections between people are symmetric. b) A citation network is a directed
graph; one publication cites another, so the relationship is asymmetric. c) A
knowledge graph is a directed heterogeneous multigraph. The nodes are hetero-
geneous in that they represent different object types (people, places, companies)
and multiple edges may represent different relations between each node. d) A
point set can be converted to a graph by forming edges between nearby points.
Eachnodehasanassociatedpositionin3Dspace,andthisistermedageometric
graph(adaptedfromHuetal.,2022). e)Thesceneontheleftcanberepresented
byahierarchicalgraph. Thetopologyoftheroom,table,andlightareallrepre-
sentedbygraphs. Thesegraphsformnodesinalargergraphrepresentingobject
adjacency (adapted from Fernández-Madrigal & González, 2002).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

13.2 Graph representation 243
Figure 13.3 Graph representation. a) Example graph with six nodes and seven
edges. Each node has an associated embedding of length five (brown vectors).
Eachedgehasanassociatedembeddingoflengthfour(bluevectors). Thisgraph
canberepresentedbythreematrices. b)Theadjacencymatrixisabinarymatrix
where element (m,n) is set to one if node m connects to node n. c) The node
data matrix X contains the concatenated node embeddings. d) The edge data
matrix E contains the edge embeddings.
The point set representing the airplane in figure 13.2d can be converted into a graph
by connecting each point to its K nearest neighbors. The result is a geometric graph
where each point is associated with a position in 3D space. Figure 13.2e represents a
hierarchical graph. Thetable, light, androomareeachdescribedbygraphsrepresenting
the adjacency of their respective components. These three graphs are themselves nodes
in another graph that represents the topology of the objects in a larger model.
All types of graphs can be processed using deep learning. However, this chapter
focuses on undirected graphs like the social network in figure 13.2a.
13.2 Graph representation
In addition to the graph structure itself, information is typically associated with each
node. Forexample,inasocialnetwork,eachindividualmightbecharacterizedbyafixed-
length vector representing their interests. Sometimes, the edges also have information
attached. For example, in the road network example, each edge might be characterized
by its length, number of lanes, frequency of accidents, and speed limit. The information
at a node is stored in a node embedding, and the information at an edge is stored in an
edge embedding.
Moreformally,agraphconsistsofasetofN nodesconnectedbyasetofE edges. The
graph can be encoded by three matrices A, X, and E, representing the graph structure,
node embeddings, and edge embeddings, respectively (figure 13.3).
Draft: please send errata to udlbookmail@gmail.com.

244 13 Graph neural networks
Figure 13.4 Properties of the adjacency matrix. a) Example graph. b) Posi-
tion(m,n)oftheadjacencymatrixAcontainsthenumberofwalksoflengthone
from node m to node n. c) Position (m,n) of the squared adjacency matrix A2
contains the number of walks of length two from node m to node n. d) One hot
vector representing node six, which was highlighted in panel (a). e) When we
pre-multiply this vector by A, the result contains the number of walks of length
one from node six to each node; we can reach nodes five, seven, and eight in one
move. f) When we pre-multiply this vector by A2, the resulting vector contains
thenumberofwalksoflengthtwofromnodesixtoeachnode;wecanreachnodes
two, three, four, five, and eight in two moves, and we can return to the original
node in three different ways (via nodes five, seven, and eight).
The graph structure is represented by the adjacency matrix, A. This is an N ×N
matrix where entry (m,n) is set to one if there is an edge between nodes m and n and
Problems13.1–13.2
zerootherwise. Forundirectedgraphs,thismatrixisalwayssymmetric. Forlargesparse
graphs, it can be stored as a list of connections (m,n) to save memory.
Thenth nodehasanassociatednodeembeddingx(n) oflengthD. Theseembeddings
areconcatenatedandstoredintheD×N nodedatamatrixX. Similarly,theethedgehas
an associated edge embedding e(e) of length D . These edge embeddings are collected
E
into the D ×E matrix E. For simplicity, we initially consider graphs that only have
E
node embeddings and return to edge embeddings in section 13.9.
13.2.1 Properties of the adjacency matrix
The adjacency matrix can be used to find the neighbors of a node using linear algebra.
Considerencodingthenth node’spositionasaone-hotcolumnvector(avectorwithonly
onenon-zero entryatposition n, whichis set toone). When wepre-multiplythis vector
bytheadjacencymatrix, itextractsthenth columnoftheadjacencymatrixandreturns
a vector with ones at the positions of the neighbors (i.e., all the places we can reach in
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

13.3 Graph neural networks, tasks, and loss functions 245
a walk of length one from the nth node). If we repeat this procedure (i.e., pre-multiply
byAagain),theresultingvectorcontainsthenumberofwalksoflengthtwofromnoden
Problems13.3–13.4
to every node (figures 13.4d–f).
In general, if we raise the adjacency matrix to the power of L, the entry at posi-
tion (m,n) of AL contains the number of unique walks of length L from node m to
Notebook13.1
node n (figures 13.4a–c). This is not the same as the number of unique paths since the
Encoding
walks include routes that visit the same node more than once. Nonetheless, AL still graphs
contains valuable information about the graph connectivity; a non-zero entry at posi-
tion (m,n) indicates that the distance from m to n must be less than or equal to L.
13.2.2 Permutation of node indices
Node indexing in graphs is arbitrary; permuting the node indices results in a permu-
tation of the columns of the node data matrix X and a permutation of both the rows
and columns of the adjacency matrix A. However, the underlying graph is unchanged
(figure13.5). Thisisincontrasttoimages,wherepermutingthepixelscreatesadifferent
image, and to text, where permuting the words creates a different sentence.
The operation of exchanging node indices can be expressed mathematically by a
permutation matrix, P. This is a matrix where exactly one entry in each row and
column take the value one, and the remaining values are zero. When position (m,n) of
the permutation matrix is set to one, it indicates that node m will become node n after
Problem13.5
the permutation. To map from one indexing to another, we use the operations:
′
X = XP
A ′ = PTAP, (13.1)
wherepost-multiplyingbyPpermutesthecolumnsandpre-multiplyingbyPT permutes
the rows. It follows that any processing applied to the graph should also be indifferent
to these permutations. Otherwise, the result will depend on the choice of node indices.
13.3 Graph neural networks, tasks, and loss functions
AgraphneuralnetworkisamodelthattakesthenodeembeddingsXandtheadjacency
matrix A as inputs and passes them through a series of K layers. The node embeddings
are updated at each layer to create intermediate “hidden” representations H before
k
finally computing output embeddings H .
K
At the start of this network, each column of the input node embeddings X just con-
tainsinformationaboutthenodeitself. Attheend,eachcolumnofthemodeloutputH
K
includes information about the node and its context within the graph. This is similar to
word embeddings passing through a transformer network. These represent words at the
start, but represent the word meanings in the context of the sentence at the end.
Draft: please send errata to udlbookmail@gmail.com.

246 13 Graph neural networks
Figure 13.5 Permutation of node indices. a) Example graph, b) associated adja-
cencymatrixandc)nodeembeddings. d)Thesamegraphwherethe(arbitrary)
order of the indices has been changed. e) The adjacency matrix and f) node
matrix are now different. Consequently, any network layer that operates on the
graph should be indifferent to the ordering of the nodes.
13.3.1 Tasks and loss functions
We defer discussion of graph neural network models until section 13.4 and first describe
the types of problems these networks tackle and their associated loss functions. Super-
vised graph problems usually fall into one of three categories (figure 13.6).
Graph-level tasks: The network assigns a label or estimates one or more values from
the entire graph, exploiting both the structure and node embeddings. For example, we
might want to predict the temperature at which a molecule becomes liquid (a regression
task) or whether a molecule is poisonous to human beings or not (a classification task).
Forgraph-leveltasks,theoutputnodeembeddingsarecombined(e.g.,byaveraging),
and the resulting vector is mapped via a linear transformation or neural network to a
fixed-size vector. For regression, the mismatch between the result and the ground truth
values is computed using the least squares loss. For binary classification, the output
is passed through a sigmoid function, and the mismatch is calculated using the binary
cross-entropy loss. Here, the probability that the graph belongs to class one might be
given by:
Pr(y =1|X,A)=sig[β +ω H 1/N], (13.2)
K K K
where the scalar β and 1×D vector ω are learned parameters. Post-multiplying the
K K
output embedding matrix H by the column vector 1 that contains ones has the effect
K
of summing together all the embeddings and subsequently dividing by the number of
nodes N computes the average. This is known as mean pooling (see figure 10.11).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

13.3 Graph neural networks, tasks, and loss functions 247
Figure 13.6 Common tasks for graphs. In each case, the input is a graph repre-
sentedbyitsadjacencymatrixandnodeembeddings. Thegraphneuralnetwork
processes the node embeddings by passing them through a series of layers. The
node embeddings at the last layer contain information about both the node and
itscontextinthegraph. a)Graphclassification. Thenodeembeddingsarecom-
bined (e.g., by averaging) and then mapped to a fixed-size vector that is passed
throughasoftmaxfunctiontoproduceclassprobabilities. b)Nodeclassification.
Each node embedding is used individually as the basis for classification (cyan
and orange colors represent assigned node classes). c) Edge prediction. Node
embeddings adjacent to the edge are combined (e.g., by taking the dot product)
to compute a single number that is mapped via a sigmoid function to produce a
probability that a missing edge should be present.
Draft: please send errata to udlbookmail@gmail.com.

248 13 Graph neural networks
Node-level tasks: The network assigns a label (classification) or one or more values
(regression) to each node of the graph, using both the graph structure and node em-
beddings. For example, given a graph constructed from a 3D point cloud similar to
figure 13.2d, the goal might be to classify the nodes according to whether they belong
to the wings or fuselage. Loss functions are defined in the same way as for graph-level
tasks, except that now this is done independently at each node n:
h i
Pr(y(n) =1|X,A)=sig β +ω h(n) . (13.3)
K K K
Edge prediction tasks: The network predicts whether or not there should be an edge
between nodes n and m. For example, in the social network setting, the network might
predict whether two people know and like each other and suggest that they connect if
thatisthecase. Thisisabinaryclassificationtaskwherethetwonodeembeddingsmust
bemappedtoasinglenumberrepresentingtheprobabilitythattheedgeispresent. One
possibilityistotakethedotproductofthenodeembeddingsandpasstheresultthrough
a sigmoid function to create the probability:
h i
Pr(y(mn) =1|X,A)=sig h(m)Th(n) . (13.4)
13.4 Graph convolutional networks
There are many types of graph neural networks, but here we focus on spatial-based
convolutional graph neural networks,orGCNsforshort. Thesemodelsareconvolutional
in that they update each node by aggregating information from nearby nodes. As such,
they induce a relational inductive bias (i.e., a bias toward prioritizing information from
neighbors). They are spatial-based because they use the original graph structure. This
contrasts with spectral-based methods, which apply convolutions in the Fourier domain.
Each layer of the GCN is a function F[•] with parameters Φ that takes the node
embeddings and adjacency matrix and outputs new node embeddings. The network can
hence be written as:
H = F[X,A,ϕ ]
1 0
H = F[H ,A,ϕ ]
2 1 1
H = F[H ,A,ϕ ]
3 2 2
. .
. .
. = .
H K = F[H K−1 ,A,ϕ K−1 ], (13.5)
where X is the input, A is the adjacency matrix, H contains the modified node em-
k
beddings at the kth layer, and ϕ denotes the parameters that map from layer k to
k
layer k+1.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.