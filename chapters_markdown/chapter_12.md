# Chapter 12

*Pages: 221-238*

---

Chapter 12
Transformers
Chapter10introducedconvolutionalnetworks, whicharespecializedforprocessingdata
that lie on a regular grid. They are particularly suited to processing images, which have
a very large number of input variables, precluding the use of fully connected networks.
Each layer of a convolutional network employs parameter sharing so that local image
patches are processed similarly at every position in the image.
This chapter introduces transformers. These were initially targeted at natural lan-
guageprocessing(NLP)problems,wherethenetworkinputisaseriesofhigh-dimensional
embeddingsrepresentingwordsorwordfragments. Languagedatasetssharesomeofthe
characteristics of image data. The number of input variables can be very large, and the
statistics are similar at every position; it’s not sensible to re-learn the meaning of the
word dog at every possible position in a body of text. However, language datasets have
the complication that text sequences vary in length, and unlike images, there is no easy
way to resize them.
12.1 Processing text data
To motivate the transformer, consider the following passage:
The restaurant refused to serve me a ham sandwich because it only cooks vegetarian
food. Intheend,theyjustgavemetwoslicesofbread. Theirambiancewasjustasgood
as the food and service.
The goal is to design a network to process this text into a representation suitable for
downstream tasks. For example, it might be used to classify the review as positive or
negative or to answer questions such as “Does the restaurant serve steak?”.
Wecanmakethreeimmediateobservations. First, theencodedinputcanbesurpris-
ingly large. In this case, each of the 37 words might be represented by an embedding
vector of length 1024, so the encoded input would be of length 37×1024=37888 even
for this small passage. A more realistically sized body of text might have hundreds or
even thousands of words, so fully connected neural networks are impractical.
Draft: please send errata to udlbookmail@gmail.com.

208 12 Transformers
Second,oneofthedefiningcharacteristicsofNLPproblemsisthateachinput(oneor
more sentences) is of a different length; hence, it’s not even obvious how to apply a fully
connectednetwork. Theseobservationssuggestthatthenetworkshouldshareparameters
across words at different input positions, similarly to how convolutional networks share
parameters across different image positions.
Third,languageisambiguous;itisunclearfromthesyntaxalonethatthepronounit
refers to the restaurant and not to the ham sandwich. To understand the text, the word
itshouldsomehowbeconnectedtothewordrestaurant. Intheparlanceoftransformers,
the former word should pay attention to the latter. This implies that there must be
connections between the words and that the strength of these connections will depend
on the words themselves. Moreover, these connections need to extend across large text
spans. For example, the word their in the last sentence also refers to the restaurant.
12.2 Dot-product self-attention
The previous section argued that a model for processing text will (i) use parameter
sharingtocopewithlonginputpassagesofdifferinglengthsand(ii)containconnections
between word representations that depend on the words themselves. The transformer
acquires both properties by using dot-product self-attention.
A standard neural network layer f[x], takes a D ×1 input x and applies a linear
transformation followed by an activation function like a ReLU, so:
f[x]=ReLU[β+Ωx], (12.1)
where β contains the biases, and Ω contains the weights.
A self-attention block sa[•] takes N inputs x ,...,x , each of dimension D×1, and
1 N
returnsN outputs, eachofwhichisalsoofsizeD×1. InthecontextofNLP,eachinput
represents a word or word fragment. First, a set of values are computed for each input:
v =β +Ω x , (12.2)
m v v m
where β ∈RD×1 and Ω ∈RD×D represent biases and weights, respectively.
v v
Then the nth output sa [x ,...,x ] is a weighted sum of all the values v ,...,v :
n 1 N 1 N
XN
sa [x ,...,x ]= a[x ,x ]v . (12.3)
n 1 N m n m
m=1
Thescalarweighta[x ,x ]istheattentionthatthenthoutputpaystoinputx . TheN
m n m
weights a[•,x ] are non-negative and sum to one. Hence, self-attention can be thought
n
of as routing the values in different proportions to create each output (figure 12.1).
The following sections examine dot-product self-attention in more detail. First, we
consider the computation of the values and their subsequent weighting (equation 12.3).
Then we describe how to compute the attention weights a[x ,x ] themselves.
m n
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

12.2 Dot-product self-attention 209
Figure 12.1 Self-attention as routing. The self-attention mechanism takes N
inputs x ,...,x ∈ RD (here N =3 and D=4) and processes each separately
1 N
to compute N value vectors. The nth output sa
n
[x
1
,...x
N
] (written as sa
n
[x•]
forshort)isthencomputedasaweightedsumoftheN valuevectors,wherethe
weightsarepositiveandsumtoone. a)Outputsa
1
[x•]iscomputedasa[x
1
,x
1
]=
0.1 times the first value vector, a[x ,x ] = 0.3 times the second value vector,
2 1
and a[x
3
,x
1
]=0.6 times the third value vector. b) Output sa
2
[x•] is computed
inthesameway,butthistimewithweightsof0.5,0.2,and0.3. c)Theweighting
for output sa
3
[x•] is different again. Each output can hence be thought of as a
different routing of the N values.
12.2.1 Computing and weighting values
Equation12.2showsthatthesameweightsΩ ∈RD×D andbiasesβ ∈RD areapplied
v v
to each input x• ∈ RD. This computation scales linearly with the sequence length N,
so it needs fewer parameters than a fully connected network relating all DN inputs to
allDN values. Infact,thevaluecomputationcanbeviewedasasparsematrixoperation
with shared parameters that relates these DN quantities (figure 12.2b).
The attention weights a[x ,x ] combine the values from different inputs. They
m n
are also sparse since there is only one weight for each ordered pair of inputs (x ,x ),
m n
regardlessofthesizeoftheseinputs(figure12.2c). Itfollowsthatthenumberofattention
weightshasaquadraticdependenceonthesequencelengthN,butisindependentofthe
length D of each input.
12.2.2 Computing attention weights
Intheprevioussection, wesawthattheoutputsresultfromtwochainedlineartransfor-
mations; the value vectors β +Ω x are computed independently for each input x ,
v v m m
and these vectors are combined linearly by the attention weights a[x ,x ]. However,
m n
the overall self-attention computation is nonlinear. As we’ll see shortly, the attention
weights are themselves nonlinear functions of the input. This is an example of a hyper-
network, where one network branch computes the weights of another. To compute the
Draft: please send errata to udlbookmail@gmail.com.

210 12 Transformers
Figure12.2Self-attentionforN=3inputsx ,eachwithdimensionD=4. a)Each
n
inputx isoperatedonindependentlybythesameweightsΩ (samecolorequals
m v
same weight) and biases β (not shown) to form the values β +Ω x . Each
v v v m
output is a linear combination of the values, with the attention weight a[x ,x ]
m n
defining the contribution of the mth value to the nth output. b) Matrix showing
block sparsity of linear transformation Ω between inputs and values. c) Matrix
v
showing sparsity of attention weights relating values and outputs.
attention, we apply two more linear transformations to the inputs:
q = β +Ω x
n q q n
k = β +Ω x , (12.4)
m k k m
where {q } and {k } are termed queries and keys, respectively. Then we compute dot
n m
AppendixB.3.4
products between the queries and keys and pass the results through a softmax function:
Dotproduct
(cid:2) (cid:3)
a[x ,x ] = softmax kTq
m n m(cid:2) • (cid:3)n
exp kTq
= P m(cid:2) n (cid:3), (12.5)
N exp kT q
m′=1 m′ n
so for each x , they are positive and sum to one (figure 12.3). For obvious reasons, this
n
is known as dot-product self-attention.
Thenames“queries”and“keys”wereinheritedfromthefieldofinformationretrieval
and have the following interpretation: the dot product operation returns a measure of
similarity between its inputs, so the weights a[x•,x
n
] depend on the relative similarities
between the nth query and all of the keys. The softmax function means that the key
vectors “compete” with one another to contribute to the final result. The queries and
keys must have the same dimensions. However, these can differ from the dimension of
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

12.2 Dot-product self-attention 211
Figure 12.3 Computing attention weights. a) Query vectors q = β +Ω x
n q q n
and key vectors k = β +Ω x are computed for each input x . b) The dot
n k k n n
products between each query and the three keys are passed through a softmax
function to form non-negative attentions that sum to one. c) These route the
value vectors (figure 12.1) via the sparse matrix from figure 12.2c.
the values, which is usually the same size as the input, so the representation doesn’t Problems12.1–12.2
change size.
12.2.3 Self-attention summary
The nth output is a weighted sum of the same linear transformation v• = β
v
+Ω
v
x•
applied to all of the inputs, where these attention weights are positive and sum to one.
The weights depend on a measure of similarity between input x and the other inputs.
n
There is no activation function, but the mechanism is nonlinear due to the dot-product
and a softmax operation used to compute the attention weights.
Note that this mechanism fulfills the initial requirements. First, there is a single
shared set of parameters ϕ = {β ,Ω ,β ,Ω ,β ,Ω }. This is independent of the
v v q q k k
Draft: please send errata to udlbookmail@gmail.com.

212 12 Transformers
Figure 12.4 Self-attention in matrix form. Self-attention can be implemented
eﬀicientlyifwestoretheN inputvectorsx inthecolumnsoftheD×N matrixX.
n
TheinputXisoperatedonseparatelybythequerymatrixQ,keymatrixK,and
valuematrixV. Thedotproductsarethencomputedusingmatrixmultiplication,
andasoftmaxoperationisappliedindependentlytoeachcolumnoftheresulting
matrix to calculate the attentions. Finally, the values are post-multiplied by the
attentions to create an output of the same size as the input.
numberofinputsN,sothenetworkcanbeappliedtodifferentsequencelengths. Second,
there are connections between the inputs (words), and the strength of these connections
depends on the inputs themselves via the attention weights.
12.2.4 Matrix form
The above computation can be written in a compact form if the N inputs x form the
n
columns of the D×N matrix X. The values, queries, and keys can be computed as:
V[X] = β 1T+Ω X
v v
Q[X] = β 1T+Ω X
q q
K[X] = β 1T+Ω X, (12.6)
k k
where 1 is an N ×1 vector containing ones. The self-attention computation is then:
h i
Sa[X]=V[X]·Softmax K[X]TQ[X] , (12.7)
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

12.3 Extensions to dot-product self-attention 213
Figure 12.5 Positional encodings. The
self-attention architecture is equivariant
to permutations of the inputs. To en-
surethatinputsatdifferentpositionsare
treateddifferently,apositionalencoding
matrix Π can be added to the data ma-
trix. Eachcolumnisdifferent,sothepo-
sitions can be distinguished. Here, the
position encodings use a predefined pro-
ceduralsinusoidalpattern(whichcanbe
extended to larger values of N if neces-
sary). However, in other cases, they are
learned.
where the function Softmax[•] takes a matrix and performs the softmax operation
independentlyoneachofitscolumns(figure12.4). Inthisformulation,wehaveexplicitly
Notebook12.1
included the dependence of the values, queries, and keys on the input X to emphasize
Self-attention
thatself-attentioncomputesakindoftripleproductbasedontheinputs. However,from
now on, we will drop this dependence and just write:
h i
Sa[X]=V·Softmax KTQ . (12.8)
12.3 Extensions to dot-product self-attention
In the previous section, we described self-attention. Here, we introduce three extensions
that are almost always used in practice.
12.3.1 Positional encoding
Observant readers will have noticed that the self-attention mechanism overlooks impor-
Problem12.3
tantinformation: thecomputationdoesnottakeintoaccounttheorderoftheinputsx .
n
More precisely, it is equivariant with respect to input permutations. However, order is
important when the inputs correspond to the words in a sentence. The sentence The
womanatetheraccoonhasadifferentmeaningthanTheraccoonatethewoman. There
are two main approaches to incorporating position information.
Absolute positional encodings: A matrix Π is added to the input X that encodes
positional information (figure 12.5). Each column of Π is unique and hence contains
information about the absolute position in the input sequence. This matrix can be
chosen by hand or learned. It may be added to the network inputs or at every network
layer. Sometimes it is added to X in the computation of the queries and keys but not
to the values.
Draft: please send errata to udlbookmail@gmail.com.

214 12 Transformers
Relative positional encodings: The input to a self-attention mechanism may be an
entire sentence, many sentences, or just a fragment of a sentence, and the absolute
position of a word is much less important than the relative position between two words.
Of course, this can be recovered if the system knows the absolute position of both,
but relative positional encodings encode this information directly. Each element of the
attention matrix corresponds to a particular offset between key position a and query
position b. Relative positional encodings learn a parameter π for each offset and use
a,b
this to modify the attention matrix by adding these values, multiplying by them, or
using them to alter the attention matrix in some other way.
12.3.2 Scaled dot-product self-attention
The dot products in the attention computation can have large magnitudes and move
the arguments to the softmax function into a region where the largest value completely
dominates. Smallchangestotheinputstothesoftmaxfunctionnowhavelittleeffecton
Problem12.4
the output (i.e., the gradients are very small), making the model diﬀicult to train. To
prevent this, the dot products are scaled by the square root of the dimension D of the
q
queries and keys (i.e., the number of rows in Ω and Ω , which must be the same):
q k
" #
KTQ
Sa[X]=V·Softmax √ . (12.9)
D
q
This is known as scaled dot-product self-attention.
12.3.3 Multiple heads
Multiple self-attention mechanisms are usually applied in parallel, and this is known as
multi-headself-attention. NowH differentsetsofvalues,keys,andqueriesarecomputed:
V = β 1T+Ω X
h vh vh
Q = β 1T+Ω X
h qh qh
K = β 1T+Ω X. (12.10)
h kh kh
The hth self-attention mechanism or head can be written as:
" #
KTQ
Sa [X]=V ·Softmax √h h , (12.11)
h h
D
q
where we have different parameters {β ,Ω }, {β ,Ω }, and {β ,Ω } for each
vh vh qh qh kh kh
head. Typically,ifthedimensionoftheinputsx isDandthereareH heads,thevalues,
m
queries, and keys will all be of size D/H, as this allows for an eﬀicient implementation.
Problem12.5
Theoutputsoftheseself-attentionmechanismsareverticallyconcatenated, andanother
linear transform Ω is applied to combine them (figure 12.6):
c
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

12.4 Transformer layers 215
Figure 12.6 Multi-head self-attention. Self-attention occurs in parallel across
multiple“heads.” Eachhasitsownqueries,keys,andvalues. Heretwoheadsare
depicted, in the cyan and orange boxes, respectively. The outputs are vertically
concatenated,andanotherlineartransformationΩ isusedtorecombinethem.
c
h i
T
MhSa[X]=Ω Sa [X]T,Sa [X]T,...,Sa [X]T . (12.12)
c 1 2 H
Notebook12.2
Multiple heads seem to be necessary to make self-attention work well. It has been Multi-head
speculated that they make the self-attention network more robust to bad initializations. self-attention
12.4 Transformer layers
Self-attention is just one part of a larger transformer layer. This consists of a multi-
head self-attention unit (which allows the word representations to interact with each
Draft: please send errata to udlbookmail@gmail.com.

216 12 Transformers
Figure 12.7Transformerlayer. TheinputconsistsofaD×N matrixcontaining
theD-dimensionalwordembeddingsforeachoftheN inputtokens. Theoutputis
amatrixofthesamesize. Thetransformerlayerconsistsofaseriesofoperations.
First, there is a multi-head attention block, allowing the word embeddings to
interact with one another. This forms the processing of a residual block, so the
inputs are added back to the output. Second, a LayerNorm operation is applied
separately to each embedding. Third, there is a second residual layer where the
samefully connected neuralnetworkis appliedseparately to eachof the N word
representations (columns). Finally, LayerNorm is applied again.
other) followed by a fully connected network mlp[x•] (that operates separately on each
word). Both units are residual networks (i.e., their output is added back to the original
input). In addition, it is typical to add a LayerNorm operation after both the self-
attention and fully connected networks. This is similar to BatchNorm but normalizes
each embedding in each batch element separately using statistics calculated across its
D embedding dimensions (section 11.4 and figure 11.14). The complete layer can be
described by the following series of operations (figure 12.7):
X ← X+MhSa[X]
X ← LayerNorm[X]
x ← x +mlp[x ] ∀n∈{1,...,N}
n n n
X ← LayerNorm[X], (12.13)
wherethecolumnvectorsx areseparatelytakenfromthefulldatamatrixX. Inareal
n
network, the data passes through a series of these transformer layers.
12.5 Transformers for natural language processing
The previous section described the transformer layer. This section describes how it is
used in natural language processing (NLP) tasks. A typical NLP pipeline starts with a
tokenizer that splits the text into words or word fragments. Then each of these tokens
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

12.5 Transformers for natural language processing 217
Figure 12.8 Sub-word tokenization. a) A passage of text from a nursery rhyme.
The tokens are initially just the characters and whitespace (represented by an
underscore),andtheirfrequenciesaredisplayedinthetable. b)Ateachiteration,
the sub-word tokenizer looks for the most commonly occurring adjacent pair of
tokens(inthiscase,se)andmergesthem. Thiscreatesanewtokenanddecreases
thecountsfortheoriginaltokenssande. Notethatthelastcharacterofthefirst
token to be merged cannot be whitespace, which prevents merging across words.
c)Attheseconditeration,thealgorithmmergeseandthewhitespacecharacter_.
d)After22iterations,thetokensconsistofamixofletters,wordfragments,and
commonlyoccurringwords. e)Ifwecontinuethisprocessindefinitely,thetokens
eventuallyrepresentthefullwords. f)Overtime,thenumberoftokensincreases
as we add word fragments to the letters and then decreases again as we merge
thesefragments. Inarealsituation,therewouldbeaverylargenumberofwords,
andthealgorithmwouldterminatewhenthevocabularysize(numberoftokens)
reached a predetermined value. Punctuation and capital letters would also be
treated as separate input characters.
Draft: please send errata to udlbookmail@gmail.com.

218 12 Transformers
is mapped to a learned embedding. These embeddings are passed through a series of
transformer layers. We now consider each of these stages in turn.
12.5.1 Tokenization
A text processing pipeline begins with a tokenizer. This splits the text into smaller
constituent units (tokens) from a vocabulary of possible tokens. In the discussion above,
we have implied that these tokens represent words, but there are several diﬀiculties.
• Inevitably, some words (e.g., names) will not be in the vocabulary.
• It’s unclear how to handle punctuation, but this is important. If a sentence ends
in a question mark, we must encode this information.
• The vocabulary would need different tokens for versions of the same word with
differentsuﬀixes(e.g.,walk,walks,walked,walking),andthereisnowaytoclarify
that these variations are related.
Oneapproachwouldbetouselettersandpunctuationmarksasthevocabulary,butthis
wouldmeansplittingtextintoverysmallpartsandrequiringthesubsequentnetworkto
re-learn the relations between them.
In practice, a compromise between letters and full words is used, and the final vo-
Notebook12.3
cabulary includes both common words and word fragments from which larger and less
Tokenization
frequent words can be composed. The vocabulary is computed using a sub-word tok-
enizer such as byte pair encoding (figure 12.8) that greedily merges commonly occurring
sub-strings based on their frequency.
12.5.2 Embeddings
Each token in the vocabulary V is mapped to a unique word embedding, and the embed-
dings for the whole vocabulary are stored in a matrix Ω ∈RD×|V|. To accomplish this,
e
the N input tokens are first encoded in the matrix T ∈ R|V|×N, where the nth column
corresponds to the nth token and is a |V|×1 one-hot vector (i.e., a vector where every
entry is zero except for the entry corresponding to the token, which is set to one). The
input embeddings are computed as X=Ω T, and Ω is learned like any other network
e e
parameter (figure 12.9). A typical embedding size D is 1024, and a typical total vocab-
ulary size |V| is 30,000, so even before the main network, there are many parameters
in Ω to learn.
e
12.5.3 Transformer model
Finally, the embedding matrix X representing the text is passed through a series of K
transformer layers, called a transformer model. There are three types of transformer
models. An encoder transforms the text embeddings into a representation that can
support a variety of tasks. A decoder predicts the next token to continue the input
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

12.6 Encoder model example: BERT 219
Figure 12.9 The input embedding matrix X∈RD×N contains N embeddings of
length D and is created by multiplying a matrix Ω containing the embeddings
e
fortheentirevocabularywithamatrixcontainingone-hotvectorsinitscolumns
that correspond to the word or sub-word indices. The vocabulary matrix Ω is
e
considered a parameter of the model and is learned along with the other param-
eters. Note that the two embeddings for the word an in X are the same.
text. Encoder-decoders are used in sequence-to-sequence tasks, where one text string is
converted into another (e.g., machine translation). These variations are described in
sections 12.6–12.8, respectively.
12.6 Encoder model example: BERT
BERT is an encoder model that uses a vocabulary of 30,000 tokens. Input tokens are
converted to 1024-dimensional word embeddings and passed through 24 transformer
layers. Each contains a self-attention mechanism with 16 heads. The queries, keys, and
valuesforeachheadareofdimension64(i.e.,thematricesΩ ,Ω ,Ω are64×1024).
vh qh kh
The dimension of the single hidden layer in the fully connected networks is 4096. The
total number of parameters is ∼ 340 million. When BERT was introduced, this was
considered large, but it is now much smaller than state-of-the-art models.
Encoder models like BERT exploit transfer learning (section 9.3.6). During pre-
training,theparametersofthetransformerarchitecturearelearnedusingself-supervision
from a large corpus of text. The goal here is for the model to learn general information
aboutthestatisticsoflanguage. Inthefine-tuningstage,theresultingnetworkisadapted
to solve a particular task using a smaller body of labelled training data.
Draft: please send errata to udlbookmail@gmail.com.

220 12 Transformers
Figure 12.10 Pre-training for BERT-like encoder. The input tokens (and a spe-
cial <cls> token denoting the start of the sequence) are converted to word em-
beddings. Here, these are represented as rows rather than columns, so the box
labeled “word embeddings” is XT. These embeddings are passed through a se-
ries of transformer layers (orange connections indicate that every token attends
to every other token in these layers) to create a set of output embeddings. A
smallfractionoftheinputtokensarerandomlyreplacedwithageneric<mask>
token. Inpre-training,thegoalistopredictthemissingwordfromtheassociated
outputembedding. Tothisend,theoutputscorrespondingtothemaskedtokens
are passed through softmax functions, and a multiclass classification loss (sec-
tion 5.24) is applied to each. This task has the advantage that it uses both the
left and right context to predict the missing word but has the disadvantage that
it does not makeeﬀicient use of data; here, seventokens need to be processed to
add two terms to the loss function.
12.6.1 Pre-training
In the pre-training stage, the network is trained using self-supervision. This allows the
useofenormousamountsofdatawithouttheneedformanuallabels. ForBERT,theself-
supervisiontaskconsistsofpredictingmissingwordsfromsentencesfromalargeinternet
Problem12.6
corpus (figure 12.10).1 During training, the maximum input length is 512 tokens, and
thebatchsizeis256. Thesystemistrainedforamillionsteps,correspondingtoroughly
50 epochs of the 3.3-billion word corpus.
Predictingmissingwordsforcesthetransformernetworktounderstandsomesyntax.
For example, it might learn that the adjective red is often found before nouns like house
or car but never before a verb like shout. It also allows the model to learn superficial
common sense about the world. For example, after training, the model will assign a
higher probability to the missing word train in the sentence The <mask> pulled into
the station than it would to the word peanut. However, the degree of “understanding”
this type of model can ever have is limited.
1BERT also uses a secondary task that predicts whether two sentences were originally adjacent in
thetextornot,butthisonlymarginallyimprovesperformance.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

12.6 Encoder model example: BERT 221
Figure12.11Afterpre-training,theencoderisfine-tunedusingmanuallylabeled
data to solve a particular task. Usually, a linear transformation or a multi-layer
perceptron (MLP) is appended to the encoder to produce whatever output is
required. a) Example text classification task. In this sentiment classification
task, the <cls> token embedding is used to predict the probability that the
review is positive. b) Example word classification task. In this named entity
recognitionproblem,theembeddingforeachwordisusedtopredictwhetherthe
word corresponds to a person, place, or organization, or is not an entity.
12.6.2 Fine-tuning
In the fine-tuning stage, the model parameters are adjusted to specialize the network to
a particular task. An extra layer is appended onto the transformer network to convert
the output vectors to the desired output format. Examples include:
Text classification: In BERT, a special token known as the classification or <cls>
token is placed at the start of each string during pre-training. For text classification
tasks like sentiment analysis (in which the passage is labeled as having a positive or
negative emotional tone), the vector associated with the <cls> token is mapped to a
singlenumberandpassedthroughalogisticsigmoid(figure 12.11a). Thiscontributesto
a standard binary cross-entropy loss (section 5.4).
Draft: please send errata to udlbookmail@gmail.com.

222 12 Transformers
Word classification: The goal of named entity recognition is to classify each word as
an entity type (e.g., person, place, organization, or no-entity). To this end, each input
embedding x is mapped to an E ×1 vector where the E entries correspond to the E
n
entity types. This is passed through a softmax function to create probabilities for each
class, which contribute to a multiclass cross-entropy loss (figure 12.11b).
Text span prediction: In the SQuAD 1.1 question answering task, the question and a
passage from Wikipedia containing the answer are concatenated and tokenized. BERT
is then used to predict the text span in the passage that contains the answer. Each
token maps to two numbers indicating how likely it is that the text span begins and
ends at this location. The resulting two sets of numbers are put through two softmax
functions. Thelikelihoodofanytextspanbeingtheanswercanbederivedbycombining
the probability of starting and ending at the appropriate places.
12.7 Decoder model example: GPT3
This section presents a high-level description of GPT3, an example of a decoder model.
The basic architecture is extremely similar to the encoder model and comprises a series
of transformer layers that operate on learned word embeddings. However, the goal is
different. The encoder aimed to build a representation of the text that could be fine-
tuned to solve a variety of more specific NLP tasks. Conversely, the decoder has one
purpose: to generate the next token in a sequence. It can generate a coherent text
passage by feeding the extended sequence back into the model.
12.7.1 Language modeling
GPT3isanautoregressivelanguagemodel. Thisiseasiesttounderstandwithaconcrete
example. Consider the sentence It takes great courage to let yourself appear weak. For
simplicity, let’s assume that the tokens are the full words. The probability of the full
sentence can be factored as:
Pr(It takes great courage to let yourself appear weak) =
Pr(It)×Pr(takes|It)×Pr(great|It takes)×Pr(courage|It takes great)×
Pr(to|It takes great courage)×Pr(let|It takes great courage to)×
Pr(yourself|It takes great courage to let)×
Pr(appear|It takes great courage to let yourself)×
Pr(weak|It takes great courage to let yourself appear). (12.14)
An autoregressive model predicts the conditional distributions Pr(t n |t 1 ,...,t n−1 ) of
each token given all the prior tokens, and hence indirectly computes the joint proba-
bility Pr(t ,t ,...,t ) of all N tokens:
1 2 N
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

12.7 Decoder model example: GPT3 223
YN
Pr(t 1 ,t 2 ,...,t N )=Pr(t 1 ) Pr(t n |t 1 ,...,t n−1 ). (12.15)
n=2
The autoregressive formulation demonstrates the connection between maximizing the
joint probability of the tokens and the next token prediction task.
12.7.2 Masked self-attention
To train a decoder, we seek parameters that maximize the log probability of the input
text under the autoregressive model (i.e., that maximize the sum of the log conditional
probability terms). Ideally, we would pass in the whole sentence and compute all the
log probabilities and gradients in the same forward pass rather than doing a forward
pass for each token in the sentence. However, if we pass in the full sentence, the term
computing log[Pr(great|It takes)] would have access to both the answer great and the
right context courage to let yourself appear weak. Hence, the system can cheat rather
than learn to predict the following words and won’t train properly.
Fortunately, the tokens only interact in the self-attention layers in a transformer
network. Hence, the problem can be resolved by ensuring that the attention to the
answer and the right context is zero. This can be achieved by setting the corresponding
dotproductsintheself-attentioncomputation(equation12.5)tonegativeinfinitybefore
theyarepassedthroughthesoftmax[•]function. Thisisknownasmaskedself-attention.
The effect is to make the weight of all the upward-angled arrows in figure 12.1 zero.
The entire decoder network operates as follows. The input text is tokenized, and the
tokens are converted to embeddings. The embeddings are passed into the transformer
network, but now the transformer layers use masked self-attention so that they can
only attend to the current and previous tokens. Each of the output embeddings can be
thoughtofasrepresentingapartialsentence,andforeach,thegoalistopredictthenext
token in the sequence. Consequently, after the transformer layers, a single linear layer
maps each output embedding to the size of the vocabulary, followed by a softmax[•]
functionthatconvertsthesevaluestoprobabilities. Duringtraining,weaimtomaximize
the sum of the log probabilities of the next token in the ground truth sequence at every
position using a standard multiclass cross-entropy loss (figure 12.12).
12.7.3 Generating text from a decoder
The autoregressive language model is the first example of a generative model discussed
in this book. Since it defines a probability model over text sequences, it can be used
to sample new examples of plausible text. To generate from the model, we start with
an input sequence of text (which might be just the special <start> token indicating
the beginning of the sequence) and feed this into the network, which then outputs the
probabilities over possible subsequent tokens. We can then either pick the most likely
token or sample from this probability distribution. The new extended sequence can be
fed back into the decoder network to yield the probability distribution over the next
Draft: please send errata to udlbookmail@gmail.com.

224 12 Transformers
Figure 12.12 Training GPT3-type decoder network. The tokens are mapped to
wordembeddingswithaspecial<start>tokenatthebeginningofthesequence.
Theembeddingsarepassedthroughaseriesoftransformerlayersthatusemasked
self-attention. Here, each position in the sentence can only attend to its own
embeddingandthoseoftokensearlierinthesequence(orangeconnections). The
goalateachpositionistomaximizetheprobabilityofthefollowinggroundtruth
tokeninthesequence. Inotherwords,atpositionone, wewanttomaximizethe
probabilityofthetokenIt;atpositiontwo,wewanttomaximizetheprobability
of the token takes; and so on. Masked self-attention ensures the system cannot
cheatbylookingatsubsequentinputs. Theautoregressivetaskhastheadvantage
ofmakingeﬀicientuseofthedatasinceeverywordcontributesatermtotheloss
function. However, it only exploits the left context of each word.
token. Byrepeatingthisprocess,wecangeneratelargebodiesoftext. Thecomputation
can be made quite eﬀicient as prior embeddings do not depend on subsequent ones due
to the masked self-attention. Hence, much of the earlier computation can be recycled as
Problem12.7
we generate subsequent tokens.
In practice, many strategies can make the output text more coherent. For example,
Notebook12.4
beamsearchkeepstrackofmultiplepossiblesentencecompletionstofindtheoverallmost
Decoding
strategies likely sequence of words (which is not necessarily found by greedily choosing the most
likely word at each step). Top-k sampling randomly draws the next word from only the
top-Kmostlikelypossibilitiestopreventthesystemfromaccidentallychoosingfromthe
long tail of low-probability tokens and leading to an unnecessary linguistic dead end.
12.7.4 GPT3 and few-shot learning
Large language models like GPT3 apply these ideas on a massive scale. In GPT3, the
sequence lengths are 2048 tokens long, and the total batch size is 3.2 million tokens.
Thereare96transformerlayers(someofwhichimplementasparseversionofattention),
eachprocessingawordembeddingofsize12288. Thereare96headsintheself-attention
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.