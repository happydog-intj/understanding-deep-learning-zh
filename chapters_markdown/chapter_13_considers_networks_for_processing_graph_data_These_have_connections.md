# Chapter 13: considers networks for processing graph data. These have connections

*Pages: 246-253*

---

232 12 Transformers
increase the number of channels (embedding dimension).
A representative example of a multi-scale transformer is the shifted-window or SWin
transformer. This is an encoder transformer that divides the image into patches and
groups these patches into a grid of windows within which self-attention is applied in-
dependently (figure 12.18). These windows are shifted in adjacent transformers, so the
effective receptive field at a given patch can expand beyond the window border.
Thescaleisreducedperiodicallybyconcatenatingfeaturesfromnon-overlapping2×2
patches and applying a linear transformation that maps these concatenated features to
twice the original number of channels. This architecture does not have a <cls> token
but instead averages the output features at the last layer. These are then mapped via a
linear layer to the desired number of classes and passed through a softmax function to
output class probabilities. At the time of writing, the most sophisticated version of this
architecture achieves a 9.89% top-1 error rate on the ImageNet database.
A related idea is periodically to integrate information from across the whole image.
Dual attention vision transformers (DaViT) alternate two types of transformers. In the
first, image patches attend to one another, and the self-attention computation uses all
the channels. In the second, the channels attend to one another, and the self-attention
computation uses all the image patches. This architecture reaches a 9.60% top-1 error
Problem12.9
rate on ImageNet and is close to the state-of-the-art at the time of writing.
12.11 Summary
This chapter introduced self-attention and the transformer architecture. Encoder, de-
coder, and encoder-decoder models were then described. The transformer operates on
sets of high-dimensional embeddings. It has a low computational complexity per layer,
and much of the computation can be performed in parallel using the matrix form. Since
every input embedding interacts with every other, it can describe long-range dependen-
cies in text. Ultimately, the computation scales quadratically with the sequence length;
one approach to reducing the complexity is sparsifying the interaction matrix.
The training of transformers with very large unlabeled datasets is the first example
of unsupervised learning (learning without labels) in this book. Encoders learn a repre-
sentation that can be used for other tasks by predicting missing tokens. Decoders build
an autoregressive model over the inputs and are the first example of a generative model
in this book. The generative decoders can be used to create new data examples.
Chapter 13 considers networks for processing graph data. These have connections
with transformers in that the nodes of the graph attend to one another in each network
layer. Chapters 14–18 return to unsupervised learning and generative models.
Notes
Naturallanguageprocessing: Transformersweredevelopedfornaturallanguageprocessing
(NLP)tasks. Thisisanenormousareathatdealswithtextanalysis,categorization,generation,
andmanipulation. Exampletasksincludepartofspeechtagging,translation,textclassification,
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 233
Figure 12.19 Recurrent neural networks (RNNs). The word embeddings are
passed sequentially through a series of identical neural networks. Each network
has two outputs; one is the output embedding, and the other (orange arrows)
feeds back into the next neural network, along with the next word embedding.
Each output embedding contains information about the word itself and its con-
text in the preceding sentence fragment. In principle, the final output contains
informationabouttheentiresentenceandcouldbeusedtosupportclassification
tasks similarly to the <cls> token in a transformer encoder model. However,
RNNssometimesgradually“forget”abouttokensthatarefurtherbackintime.
entity recognition (people, places, companies, etc.), text summarization, question answering,
word sense disambiguation, and document clustering. NLP was initially tackled by rule-based
methodsthatexploitedthestructureandstatisticsofgrammar. SeeManning&Schutze(1999)
and Jurafsky & Martin (2000) for early approaches.
Recurrent neural networks: Before the introduction of transformers, many state-of-the-art
NLP applications used recurrent neural networks, or RNNs for short (figure 12.19). The term
“recurrent” was introduced by Rumelhart et al. (1985), but the main idea dates to at least
Minsky & Papert (1969). RNNs ingest a sequence of inputs (words in NLP) one at a time.
At each step, the network receives both the new input and a hidden representation computed
from the previous time step (the recurrent connection). The final output contains information
about the whole input. This representation can then support NLP tasks like classification or
translation. They have also been used in a decoding context in which generated tokens are
fed back into the model to form the next input to the sequence. For example, the PixelRNN
(Van den Oord et al., 2016c) used RNNs to build an autoregressive model of images.
From RNNs to transformers: One of the problems with RNNs is that they can forget in-
formationthatisfurtherbackinthesequence. Moresophisticatedversionsofthisarchitecture,
such as long short-term memory networks or LSTMs (Hochreiter & Schmidhuber, 1997b) and
gated recurrent units or GRUs (Cho et al., 2014; Chung et al., 2014) partially addressed this
problem. However, in machine translation, the idea emerged that all of the intermediate rep-
resentations in the RNN could be exploited to produce the output sentence. Moreover, certain
outputwordsshouldattendmoretocertaininputwordsaccordingtotheirrelation(Bahdanau
etal.,2015). Thisultimatelyledtodispensingwiththerecurrentstructureandreplacingitwith
theencoder-decodertransformer(Vaswanietal.,2017). Hereinputtokensattendtooneanother
(self-attention), output tokens attend to those earlier in the sequence (masked self-attention),
and output tokens also attend to the input tokens (cross-attention). A formal algorithmic de-
scriptionofthetransformercanbefoundinPhuong&Hutter(2022),andasurveyofworkcan
be found in Lin et al. (2022). The literature should be approached with caution, as many en-
Draft: please send errata to udlbookmail@gmail.com.

234 12 Transformers
hancementstotransformersdonotmakemeaningfulperformanceimprovementswhencarefully
assessed in controlled experiments (Narang et al., 2021).
Applications: Models based on self-attention and/or the transformer architecture have been
applied to text sequences (Vaswani et al., 2017), image patches (Dosovitskiy et al., 2021),
protein sequences (Rives et al., 2021), graphs (Veličković et al., 2019), database schema (Xu
et al., 2021b), speech (Wang et al., 2020c), mathematical integration when formulated as a
translation problem (Lample & Charton, 2020), and time series (Wu et al., 2020b). However,
theirmostcelebratedsuccesseshavebeeninbuildinglanguagemodelsand,morerecently,asa
replacement for convolutional networks in computer vision.
Large language models: Vaswani et al. (2017) targeted translation tasks, but transformers
arenowmoreusuallyusedtobuildeitherpureencoderorpuredecodermodels,themostfamous
of which are BERT (Devlin et al., 2019) and GPT2/GPT3 (Radford et al., 2019; Brown et al.,
2020), respectively. These models are usually tested against benchmarks like GLUE (Wang
et al., 2019b), which includes the SQuAD question-answering task (Rajpurkar et al., 2016)
describedinsection12.6.2,SuperGLUE(Wangetal.,2019a)andBIG-bench(Srivastavaetal.,
2022), which combine many NLP tasks to create an aggregate score for measuring language
ability. Decodermodelsaregenerallynotfine-tunedforthesetasksbutcanperformwellanyway
when given a few examples of questions and answers and asked to complete the text from the
next question. This is referred to as few-shot learning (Brown et al., 2020).
Since GPT3, many decoder language models have been released with steady improvement in
few-shot results. These include GLaM (Du et al., 2022), Gopher (Rae et al., 2021), Chinchilla
(Hoffmann et al., 2023), Megatron-Turing NLG (Smith et al., 2022), and LaMDa (Thoppilan
et al., 2022). Most of the performance improvement is attributable to increased model size,
using sparsely activated modules, and exploiting larger datasets. At the time of writing, the
most recent model is PaLM (Chowdhery et al., 2022), which has 540 billion parameters and
was trained on 780 billion tokens across 6144 processors. Interestingly, since text is highly
compressible,thismodelhasmorethanenoughcapacitytomemorizetheentiretrainingdataset.
Thisistrueformanylanguagemodels. Manyboldstatementshavebeenmadeabouthowlarge
language models exceed human performance. This is probably true for some tasks, but such
statementsshouldbetreatedwithcaution(seeRibeiroetal.,2021;McCoyetal.,2019;Bowman
& Dahl, 2021; and Dehghani et al., 2021).
These models have considerable world knowledge. For example, in section 12.7.4, the model
knows key facts about deep learning, including that it is a type of machine learning with
associated algorithms and applications. Indeed, one such model has been mistakenly identified
as being sentient (Clark, 2022). However, there are persuasive arguments that the degree of
“understanding” this type of model can ever have is limited (Bender & Koller, 2020).
Tokenizers: Schuster & Nakajima (2012) and Sennrich et al. (2015) introduced WordPiece
andbyte pair encoding(BPE),respectively. Bothmethodsgreedilymergepairsoftokensbased
on their frequency of adjacency (figure 12.8), with the main difference being how the initial
tokens are chosen. For example, in BPE, the initial tokens are characters or punctuation with
a special token to denote whitespace. The merges cannot occur over the whitespace. As the
algorithm proceeds, new tokens are formed by combining characters recursively so that sub-
word and word tokens emerge. The unigram language model (Kudo, 2018) generates several
possiblecandidatemergesandchoosesthebestonebasedonthelikelihoodinalanguagemodel.
Provilkov et al. (2020) develop BPE dropout, which generates the candidates more eﬀiciently
byintroducingrandomnessintotheprocessofcountingfrequencies. Versionsofbothbytepair
encoding and the unigram language model are included in the SentencePiece library (Kudo &
Richardson,2018),whichworksdirectlyonUnicodecharactersandcanworkwithanylanguage.
Heetal.(2020)introduceamethodthattreatsthesub-wordsegmentationasalatentvariable
that should be marginalized out for learning and inference.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 235
Decoding algorithms: Transformer decoder models take a body of text and return a prob-
ability over the next token. This is then added to the preceding text, and the model is run
again. Theprocessofchoosingtokensfromtheseprobabilitydistributionsisknownasdecoding.
Näivewaystodothiswouldbetoeither(i)greedilychoosethemostlikelytokenor(ii)choose
a token randomly according to the distribution. However, neither of these methods works well
in practice. In the former case, the results may be very generic, and the latter case may lead
to degraded quality outputs (Holtzman et al., 2020). This is partly because, during training,
themodelwasonlyexposedtosequencesofgroundtruthtokens(knownasteacherforcing)but
sees its own output when deployed.
It is not computationally feasible to try every combination of tokens in the output sequence,
but it is possible to maintain a fixed number of parallel hypotheses and choose the most likely
overall sequence. This is known as beam search. Beam search tends to produce many similar
hypotheses and has been modified to investigate more diverse sequences (Vijayakumar et al.,
2016;Kulikovetal.,2018). Onepossibleproblemwithrandomsamplingisthatthereisavery
long tail of unlikely following words that collectively have a significant probability. This has
led to the development of top-K sampling, in which tokens are sampled from only the K most
likelyhypotheses(Fanetal.,2018). Top-Ksamplingstillsometimesallowsunreasonabletoken
choices when there are only a few high-probability choices. To resolve this problem, Holtzman
etal.(2020)proposednucleussampling,inwhichtokensaresampledfromafixedproportionof
thetotalprobabilitymass. ElAsri&Prince(2020)discussdecodingalgorithmsinmoredepth.
Types of attention: Scaled dot-product attention (Vaswani et al., 2017) is just one of a
familyofattentionmechanismsthatincludesadditiveattention(Bahdanauetal.,2015),multi-
plicativeattention(Luongetal.,2015),key-valueattention(Daniluketal.,2017),andmemory-
compressed attention (Liu et al., 2019c). Zhai et al. (2021) constructed “attention-free” trans-
formers,inwhichthetokensinteractinawaythatdoesnothavequadraticcomplexity. Multi-
headattentionwasalsointroducedbyVaswanietal.(2017). Interestingly,itappearsthatmost
of the heads can be pruned after training without critically affecting the performance (Voita
etal.,2019);ithasbeensuggestedthattheirroleistoguardagainstbadinitializations. Huetal.
(2018b)proposesqueeze-and-excitationnetworks,attention-likemechanismsthatre-weightthe
channels in a convolutional layer based on globally computed features.
Relationship of self-attention to other models: Theself-attentioncomputationhasclose
connectionstoothermodels. First,itisanexampleofahypernetwork(Haetal.,2017)inthat
itusesonepartofthenetworktochoosetheweightsofanotherpart: theattentionmatrixforms
the weights of a sparse network layer that maps the values to the outputs (figure 12.3). The
synthesizer(Tayetal.,2021)simplifiesthisideabyusinganeuralnetworktocreateeachrowof
theattentionmatrixfromthecorrespondinginputembedding. Eventhoughtheinputtokensno
longerinteractwitheachothertocreatetheattentionweights,thisworkssurprisinglywell. Wu
et al. (2019) present a similar system that produces an attention matrix with a convolutional
structuresothetokensattendtotheirneighbors. Thegatedmulti-layerperceptron(Wuetal.,
2019)computesamatrixthatpointwisemultipliesthevaluesandhencemodifiesthemwithout
mixing them. Transformers are also closely related to fast weight memory systems, which were
the intellectual forerunners of hypernetworks (Schlag et al., 2021).
Self-attentioncanalsobethoughtofasaroutingmechanism(figure12.1), andfromthisview-
point,thereisaconnectiontocapsulenetworks(Sabouretal.,2017). Thesecapturehierarchical
relations in images; lower network levels might detect facial parts (noses, mouths), which are
then combined (routed) in higher-level capsules that represent a face. However, capsule net-
works use routing by agreement. In self-attention, the inputs compete with each other for how
much they contribute to a given output (via the softmax operation). In capsule networks, the
outputs of the layer compete with each other for inputs from earlier layers. Once we consider
self-attentionasaroutingnetwork,wecanquestionwhethermakingthisroutingdynamic(i.e.,
dependentonthedata)isnecessary. Therandomsynthesizer(Tayetal.,2021)removedthede-
Draft: please send errata to udlbookmail@gmail.com.

236 12 Transformers
pendenceoftheattentionmatrixontheinputsentirelyandeitherusedpredeterminedrandom
values or learned values. This performed surprisingly well across a variety of tasks.
Multi-head self-attention also has close connections to graph neural networks (see chapter 13),
convolution (Cordonnier et al., 2020), recurrent neural networks (Choromanski et al., 2020),
and memory retrieval in Hopfield networks (Ramsauer et al., 2021). For more information on
the relationships between transformers and other models, consult Prince (2021a).
Positional encoding: The original transformer paper (Vaswani et al., 2017) experimented
with predefining the positional encoding matrix Π, and learning the positional encoding Π.
It might seem odd to add the positional encodings to the D×N data matrix X rather than
concatenate them. However, the data dimension D is usually greater than the number of
tokensN,sothepositionalencodingliesinasubspace. ThewordembeddingsinXarelearned,
so the system can theoretically keep the two components in orthogonal subspaces and retrieve
the positional encodings as required. The predefined embeddings chosen by Vaswani et al.
(2017) were a family of sinusoidal components with two attractive properties: (i) the relative
positionoftwoembeddingsiseasytorecoverusingalinearoperationand(ii)theirdotproduct
generally decreased as the distance between positions increased (see Prince, 2021a, for more
details). Many systems, such as GPT3 and BERT, learn positional encodings. Wang et al.
(2020a)examinedthecosinesimilaritiesofthepositionalencodingsinthesemodelsandshowed
thattheygenerallydeclinewithrelativedistance,althoughtheyalsohaveaperiodiccomponent.
Muchsubsequentworkhasmodifiedjusttheattentionmatrixsothatinthescaleddot-product
self-attention equation:
" #
KTQ
Sa[X]=V·Softmax √ , (12.16)
D
q
only the queries and keys contain position information:
V = β 1T+Ω X
v v
Q = β 1T+Ω (X+Π)
q q
K = β 1T+Ω (X+Π). (12.17)
k k
Thishasledtotheideaofmultiplyingoutthequadraticcomponentinthenumeratorofequa-
tion12.16andretainingonlysomeoftheterms. Forexample,Keetal.(2021)decoupleoruntie
thecontentandpositioninformationbyretainingonlythecontent-contentandposition-position
terms and using different projection matrices Ω• for each.
Anothermodificationistoinjectinformationdirectlyabouttherelativeposition. Thisismore
important than absolute position since a batch of text can start at an arbitrary place in a
document. Shaw et al. (2018), Raffel et al. (2020), and Huang et al. (2020b) all developed
systems where a single term was learned for each relative position offset, and the attention
matrixwasmodifiedinvariouswaysusingtheserelative positional encodings. Weietal.(2019)
investigatedrelativepositionalencodingsbasedonpredefinedsinusoidalembeddingsratherthan
learned values. DeBERTa (He et al., 2021) combines these ideas; they retain only a subset of
termsfromthequadraticexpansion,applydifferentprojectionmatricestothem,anduserelative
positionalencodings. Otherworkhasexploredsinusoidalembeddingsthatencodeabsoluteand
relative position information in more complex ways (Su et al., 2021).
Wang et al. (2020a) compare the performance of transformers in BERT with different posi-
tional encodings. They found that relative positional encodings perform better than absolute
positional encodings, but there was little difference between using sinusoidal and learned em-
beddings. A survey of positional encodings can be found in Dufter et al. (2021).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 237
Extending transformers to longer sequences: Thecomplexityoftheself-attentionmech-
anism increases quadratically with the sequence length. Some tasks like summarization or
question answering may require long inputs, so this quadratic dependence limits performance.
Threelinesofworkhaveattemptedtoaddressthisproblem. Thefirstdecreasesthesizeofthe
attention matrix, the second makes the attention sparse, and the third modifies the attention
mechanism to make it more eﬀicient.
To decrease the size of the attention matrix, Liu et al. (2018b) introduced memory-compressed
attention. This applies strided convolution to the keys and values, which reduces the number
of positions in a very similar way to downsampling in a convolutional network. Attention is
now applied between weighted combinations of neighboring positions, where the weights are
learned. Along similar lines, Wang et al. (2020b) observed that the quantities in the attention
mechanismareoftenlowrankinpracticeanddevelopedtheLinFormer,whichprojectsthekeys
and values onto a smaller subspace before computing the attention matrix.
To make attention sparse, Liu et al. (2018b) proposed local attention, in which neighboring
blocksoftokensonlyattendtooneanother. Thiscreatesablockdiagonalinteractionmatrix(see
figure12.15). Informationcannotpassfromblocktoblock,sosuchlayersaretypicallyalternated
with full attention. Along the same lines, GPT3 (Brown et al., 2020) uses a convolutional
interactionmatrixandalternatesthiswithfullattention. Childetal.(2019)andBeltagyetal.
(2020) experimented with various interaction matrices, including convolutional structures with
different dilation rates but allowing some queries to interact with every other key. Ainslie
et al. (2020) introduced the extended transformer construction (figure 12.15h), which uses a
set of global embeddings that interact with every other token. This can only be done in the
encoder version, or these implicitly allow the system to “look ahead.” When combined with
relative position encoding, this scheme requires special encodings for mapping to, from, and
between these global embeddings. BigBird (Ainslie et al., 2020) combined global embeddings
and a convolutional structure with a random sampling of possible connections. Other work
has investigated learning the sparsity pattern of the attention matrix (Roy et al., 2021; Kitaev
et al., 2020; Tay et al., 2020).
Finally,ithasbeennotedthatthetermsinthenumeratoranddenominatorofthesoftmaxoper-
ationthatcomputesattentionhavetheformexp[kTq]. Thiscanbetreatedasakernelfunction
and,assuch,canbeexpressedasthedotproductg[k]Tg[q]whereg[•]isanonlineartransforma-
Problem12.10
tion. Thisformulationdecouplesthequeriesandkeys,makingtheattentioncomputationmore
eﬀicient. Unfortunately,toreplicatetheformoftheexponentialterms, thetransformationg[•]
mustmaptheinputstotheinfinitespace. Thelineartransformer(Katharopoulosetal.,2020)
recognizesthisandreplacestheexponentialtermwithadifferentsimilaritymeasure. ThePer-
former(Choromanskietal.,2020)approximatesthisinfinitemappingwithafinite-dimensional
one. MoredetailsaboutextendingtransformerstolongersequencescanbefoundinTayetal.
(2023) and Prince (2021a).
Training transformers: Trainingtransformersischallengingandrequiresbothlearningrate
warm-up(Goyaletal.,2018)andAdam(Kingma&Ba,2015). IndeedXiongetal.(2020a)and
Huangetal.(2020a)showthatthegradientsvanish,andtheAdamupdatesdecreaseinmagni-
tude without learning rate warm-up. Several interacting factors cause this problem. Residual
connections cause the exploding gradients (figure 11.6), but normalization layers prevent this.
Vaswanietal.(2017)usedLayerNormratherthanBatchNormbecauseNLPstatisticsarehighly
variablebetweenbatches, although subsequentworkhas modified BatchNormfor transformers
(Shen et al., 2020a). The positioning of the LayerNorm outside of the residual block causes
gradients to shrink as they pass back through the network (Xiong et al., 2020a). In addition,
the relative weight of the residual connections and main self-attention mechanism varies as we
move through the network upon initialization (see figure 11.6c). There is the additional com-
plication that the gradients for the query and key parameters are smaller than for the value
parameters (Liu et al., 2020), which necessitates the use of Adam. These factors interact in a
complex way, making training unstable and necessitating learning rate warm-up.
Draft: please send errata to udlbookmail@gmail.com.

238 12 Transformers
Therehavebeenvariousattemptstostabilizetraining,including(i)avariationofFixUpcalled
TFixup(Huangetal.,2020a)thatallowstheLayerNormcomponentstoberemoved,(ii)chang-
ing the position of the LayerNorm components in the network (Liu et al., 2020), and (iii)
re-weightingthe twopaths in the residual branches(Liu et al., 2020; Bachlechneret al., 2021).
Xu et al. (2021b) introduced an initialization scheme called DTFixup that allows transformers
to be trained with smaller datasets. A detailed discussion can be found in Prince (2021b).
Applications in vision: ImageGPT (Chen et al., 2020a) and the Vision Transformer (Doso-
vitskiy et al., 2021) were both early transformer architectures applied to images. Transformers
have been used for image classification (Dosovitskiy et al., 2021; Touvron et al., 2021), object
detection (Carion et al., 2020; Zhu et al., 2020b; Fanget al., 2021), semantic segmentation(Ye
et al., 2019; Xie et al., 2021; Gu et al., 2022), super-resolution (Yang et al., 2020a), action
recognition(Sunetal.,2019;Girdharetal.,2019),imagegeneration(Chenetal.,2021b;Nash
etal.,2021),visualquestionanswering(Suetal.,2019b;Tan&Bansal,2019),inpainting(Wan
et al., 2021; Zheng et al., 2021; Zhao et al., 2020b; Li et al., 2022), colorization (Kumar et al.,
2021), and many other vision tasks (Khan et al., 2022; Liu et al., 2023b).
Transformers and convolutional networks: Transformers have been combined with con-
volutional neural networks for many tasks, including image classification (Wu et al., 2020a),
object detection (Hu et al., 2018a; Carion et al., 2020), video processing (Wang et al., 2018c;
Sunetal.,2019),unsupervisedobjectdiscovery(Locatelloetal.,2020)andvarioustext/vision
tasks(Chenetal.,2020d;Luetal.,2019;Lietal.,2019). Transformerscanoutperformconvolu-
tional networks for vision tasks but usually require large quantities of data to achieve superior
performance. Often, they are pre-trained on enormous datasets like JRT (Sun et al., 2017)
and LAION (Schuhmann et al., 2021). The transformer doesn’t have the inductive bias of
convolutionalnetworks,butbyusinghugeamountsofdata,itcansurmountthisdisadvantage.
From pixels to video: Non-local networks (Wang et al., 2018c) were an early application of
self-attentiontoimagedata. Transformerswereinitiallyappliedtopixelsinlocalneighborhoods
(Parmaretal.,2018;Huetal.,2019;Parmaretal.,2019;Zhaoetal.,2020a). ImageGPT(Chen
et al., 2020a) scaled this to model all pixels in a small image. The Vision Transformer (ViT)
(Dosovitskiy et al., 2021) used non-overlapping patches to analyze bigger images.
Since then, many multi-scale systems have been developed, including the SWin transformer
(Liu et al., 2021c), SWinV2 (Liu et al., 2022), multi-scale transformers (MViT) (Fan et al.,
2021), and pyramid vision transformers (Wang et al., 2021). The Crossformer (Wang et al.,
2022b)modelsinteractionsbetweenspatialscales. Alietal.(2021)introducedcross-covariance
image transformers, in which the channels rather than spatial positions attend to one another,
hence making the size of the attention matrix indifferent to the image size. The dual attention
vision transformer (DaViT) was developed by Ding et al. (2022) and alternates between local
spatialattentionwithinsub-windowsandspatiallyglobalattentionbetweenchannels. Chuetal.
(2021) similarly alternate between local attention within sub-windows and global attention by
subsampling the spatial domain. Dong et al. (2022) adapt the ideas of figure 12.15, in which
the interactions between elements are sparsified to the 2D image domain.
Transformersweresubsequentlyadaptedtovideoprocessing(Arnabetal.,2021;Bertasiusetal.,
2021; Liu et al., 2021c; Neimark et al., 2021; Patrick et al., 2021). A survey of transformers
applied to video can be found in Selva et al. (2022).
Combining images and text: CLIP(Radfordetal.,2021)learnsajointencoderforimages
and their captions using a contrastive pre-training task. The system ingests N images and
their captions and produces a matrix of compatibility between images and captions. The loss
functionencouragesthecorrectpairstohaveahighscoreandtheincorrectpairstohavealow
score. Ramesh et al. (2021) and Ramesh et al. (2022) train a diffusion decoder to invert the
CLIP image encoder for text-conditional image generation (see chapter 18).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Notes 239
Problems
Problem 12.1 Consider a self-attention mechanism that processes N inputs of length D to
produce N outputs of the same size. How many weights and biases are used to compute the
queries, keys, and values? Assume that all three quantities are also of length D. How many
attention weights a[•,•] will there be? How many weights and biases would there be in a fully
connected shallow network relating all DN inputs to all DN outputs?
Problem 12.2 Why might we want to ensure that the input to the self-attention mechanism is
the same size as the output?
Problem 12.3∗ Show that the self-attention mechanism (equation 12.8) is equivariant to a
permutation XP of the data X, where P is a permutation matrix. In other words, show that:
AppendixB.4.4
Permutation
Sa[XP]=Sa[X]P. (12.18) matrix
Problem 12.4 Consider the softmax operation:
exp[z ]
y =softmax [z]= P i , (12.19)
i i 5 exp[z ]
j=1 j
inthecasewheretherearefiveinputswithvalues:z =−3,z =1,z =100,z =5,z =−1.
1 2 3 4 5
Compute the 25 derivatives, ∂y /∂z for all i,j ∈{1,2,3,4,5}. What do you conclude?
i j
Problem 12.5 Why is implementation more eﬀicient if the values, queries, and keys in each of
the H heads each have dimension D/H where D is the original dimension of the data?
Problem12.6BERTwaspre-trainedusingtwotasks. Thefirsttaskrequiresthesystemtopre-
dictmissing(masked)words. Thesecondtaskrequiresthesystemtoclassifypairsofsentences
as being adjacent or not in the original text. Identify whether each of these tasks is generative
or contrastive (see section 9.3.7). Why do you think they used two tasks? Propose two novel
contrastive tasks that could be used to pre-train a language model.
Problem12.7Consideraddinganewtokentoaprecomputedmaskedself-attentionmechanism
with N tokens. Describe the extra computation that must be done to incorporate this new
token.
Problem 12.8 Computation in vision transformers expands quadratically with the number of
patches. Devisetwomethodstoreducethecomputationusingtheprinciplesfromfigure12.15.
Problem 12.9 Consider representing an image with a grid of 16×16 patches, each represented
by a patch embedding of length 512. Compare the amount of computation required in the
DaViTtransformertoperformattention(i)betweenthepatches,usingallofthechannels,and
(ii) between the channels, using all of the patches.
Problem 12.10∗ Attention weights are usually computed as:
(cid:2) (cid:3)
h i
exp kTq
a[x
m
,x
n
]=softmax
m
kT •q
n
= P
N exp
m(cid:2)
k
n
T q
(cid:3). (12.20)
m′=1 m′ n
(cid:2) (cid:3)
Consider replacing exp kTq with the dot product g[k ]Tg[q ] where g[•] is a nonlinear
m n m n
transformation. Show how this makes the computation of the attention weights more eﬀicient.
Draft: please send errata to udlbookmail@gmail.com.