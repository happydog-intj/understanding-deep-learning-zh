# Chapter 4: Applications of deep learning

*Pages: 239-245*

---

12.7 Decoder model example: GPT3 225
layers, and the value, query, and key dimension is 128. It is trained with 300 billion
tokens and contains 175 billion parameters.
Here’sanexampleofcompletingtextfromtheGPT3modelwherethetextprovided
to the model is in orange, and the generated text is in cyan:
Understanding Deep Learning is a new textbook from MIT Press by Simon Prince that’s de-
signed to offer an accessible, broad introduction to the field. Deep learning is a branch of
machine learning that is concerned with algorithms that learn from data that is unstructured
or unlabeled. The book is divided into four sections:
1. Introduction to deep learning
2. Deep learning architecture
3. Deep learning algorithms
4. Applications of deep learning
The first section offers an introduction to deep learning, including its history and origins.
The second section covers deep learning architecture, discussing various types of neural net-
works and their applications. The third section dives into deep learning algorithms, including
supervised and unsupervised learning, reinforcement learning, and more. The fourth section
appliesdeeplearningtovariousdomains,suchascomputervision,naturallanguageprocessing,
and robotics.
This continuation is plausible, albeit only partially accurate.
Onesurprisingpropertyoflearningmodelsonthisscaleisthattheycanperformmany
taskswithoutfine-tuning. Ifweprovideseveralexamplesofcorrectquestion/answerpairs
and then another question, they often answer the final question correctly by completing
the sequence. One example is correcting English grammar:
Poor English input: I eated the purple berries.
Good English output: I ate the purple berries.
Poor English input: Thank you for picking me as your designer. I’d appreciate it.
Good English output: Thank you for choosing me as your designer. I appreciate it.
Poor English input: The mentioned changes have done. or I did the alteration that you
requested. or I changed things you wanted and did the modifications.
Good English output: Therequestedchangeshavebeenmade. orImadethealterationthat
you requested. or I changed things you wanted and made the modifications.
Poor English input: I’d be more than happy to work with you in another project.
Good English output: I’d be more than happy to work with you on another project.
(result from Brown et al., 2020)
Here, the text containing the paired examples in orange was provided as context for
GPT3, and the system then generated the correct answer in cyan. This phenomenon
extendstomanysituations,includinggeneratingcodesnippetsbasedonnaturallanguage
descriptions, arithmetic, translating between languages, and answering questions about
text passages. Consequently, it is argued that enormous language models are few-shot
learners; they can learn to do novel tasks based on just a few examples. However,
performance is erratic in practice, and the extent to which it is extrapolating from
learned examples rather than merely interpolating or copying verbatim is unclear.
Draft: please send errata to udlbookmail@gmail.com.

226 12 Transformers
Figure 12.13 Encoder-decoder architecture. Two sentences are passed to the
systemwiththegoaloftranslatingthefirstintothesecond. a)Thefirstsentence
ispassedthroughastandardencoder. b)Thesecondsentenceispassedthrougha
decoderthatusesmaskedself-attentionbutalsoattendstotheoutputembeddings
of the encoder using cross-attention (orange rectangle). The loss function is the
same as for the decoder model; we want to maximize the probability of the next
word in the output sequence.
12.8 Encoder-decoder model example: machine translation
Translation between languages is an example of a sequence-to-sequence task. One com-
mon approach uses both an encoder (to compute a good representation of the source
sentence) and a decoder (to generate the sentence in the target language). This is aptly
called an encoder-decoder model.
Consider translating from English to French. The encoder receives the sentence
in English and processes it through a series of transformer layers to create an output
representation for each token. During training, the decoder receives the ground truth
translationinFrenchandpassesitthroughaseriesoftransformerlayersthatusemasked
self-attention and predict the following word at each position. However, the decoder
layersalsoattendtotheoutputoftheencoder. Consequently,eachFrenchoutputwordis
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

12.9 Transformers for long sequences 227
Figure12.14Cross-attention. Theflowofcomputationisthesameasinstandard
self-attention, but the queries are calculated from the decoder embeddings X ,
dec
and the keys and values from the encoder embeddings X . For translation
enc
tasks,theencodercontainsinformationaboutthesourcelanguagestatistics,and
the decoder contains information about the target language statistics.
conditionedonthepreviousoutputwordsandthesourceEnglishsentence(figure12.13).
This is achieved by modifying the transformer layers in the decoder. Originally,
these consisted of a masked self-attention layer followed by a neural network applied
individually to each embedding (figure 12.12). A new self-attention layer is added be-
tween these two components, in which the decoder embeddings attend to the encoder
embeddings. This uses a version of self-attention known as encoder-decoder attention or
cross-attention, where the queries are computed from the decoder embeddings and the
keys and values from the encoder embeddings (figure 12.14).
12.9 Transformers for long sequences
Since each token in a transformer encoder model interacts with every other token, the
computational complexity scales quadratically with the length of the sequence. For a
decoder model, each token only interacts with previous tokens, so there are roughly
half the number of interactions, but the complexity still scales quadratically. These
relationships can be visualized as interaction matrices (figure 12.15a–b).
Thisquadraticincreaseintheamountofcomputationultimatelylimitsthelengthof
sequences that can be used. Many methods have been developed to extend the trans-
Draft: please send errata to udlbookmail@gmail.com.

228 12 Transformers
Figure12.15Interactionmatricesforself-attention. a)Inanencoder,everytoken
interactswitheveryothertoken,andcomputationexpandsquadraticallywiththe
number of tokens. b) In a decoder, each token only interacts with the previous
tokens, but complexity is still quadratic. c) Complexity can be reduced by using
a convolutional structure (encoder case). d) Convolutional structure for decoder
case. e–f) Convolutional structure with dilation rate of two and three (decoder
case). g) Another strategy is to allow selected tokens to interact with all the
other tokens (encoder case) or all the previous tokens (decoder case pictured).
h) Alternatively, global tokens can be introduced (left two columns and top two
rows). These interact with all of the tokens as well as with each other.
former to cope with longer sequences. One approach is to prune the self-attention in-
teractions or, equivalently, to sparsify the interaction matrix (figures 12.15c-h). For
example, this can be restricted to a convolutional structure so that each token only in-
teracts with a few neighboring tokens. Across multiple layers, tokens still interact at
larger distances as the receptive field expands. As for convolution in images, the kernel
can vary in size and dilation rate.
A pure convolutional approach requires many layers to integrate information over
large distances. One way to speed up this process is to allow select tokens (perhaps at
the start of every sentence) to attend to all other tokens (encoder model) or all previous
tokens (decoder model). A similar idea is to have a small number of global tokens that
connect to all the other tokens and themselves. Like the <cls> token, these do not
represent any word but serve to provide long-distance connections.
12.10 Transformers for images
Transformers were initially developed for text data. Their enormous success in this area
led to experimentation on images. This was not obviously a promising idea for two
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

12.10 Transformers for images 229
reasons. First, there are many more pixels in an image than words in a sentence, so the
quadraticcomplexityofself-attentionposesapracticalbottleneck. Second,convolutional
nets have a good inductive bias because each layer is equivariant to spatial translation,
and takes into account the 2D structure of the image. However, this must be learned in
a transformer network.
Regardless of these apparent disadvantages, transformer networks for images have
noweclipsedtheperformanceofconvolutionalnetworksforimageclassificationandother
tasks. This is partly because of the enormous scale at which they can be constructed
and the large amounts of data that can be used to pre-train the networks. This section
describes transformer models for images.
12.10.1 ImageGPT
ImageGPT is a transformer decoder; it builds an autoregressive model of image pixels
that ingests a partial image and predicts the subsequent pixel value. The quadratic
complexity of the transformer network means that the largest model (which contained
6.8billionparameters)couldstillonlyoperateon64×64images. Moreover,tomakethis
tractable, the original 24-bit RGB color space had to be quantized into a nine-bit color
space, so the system ingests (and predicts) one of 512 possible tokens at each position.
Images are naturally 2D objects, but ImageGPT simply learns a different positional
encoding at each pixel. Hence it must learn that each pixel has a close relationship with
itsprecedingneighborsandalsowithnearbypixelsintherowabove. Figure12.16shows
example generation results.
The internal representation of this decoder was used as a basis for image classifi-
cation. Each pixel’s final embedding is averaged, and a linear layer maps these values
to activations which are passed through a softmax layer to predict class probabilities.
The system is pre-trained on a large corpus of web images and then fine-tuned on the
ImageNet database resized to 48×48 pixels using a loss function that contains both a
cross-entropy term for image classification and a generative loss term for predicting the
pixels. Despite using a large amount of external training data, the system achieved only
a 27.4% top-1 error rate on ImageNet (figure 10.15). This was worse than convolutional
architectures of the time (see figure 10.21) but is still impressive given the small input
image size; unsurprisingly, it fails where the target object is small or thin.
12.10.2 Vision Transformer (ViT)
The Vision Transformer tackled the problem of image resolution by dividing the image
into 16×16 patches (figure 12.17). Each patch is mapped to an input embedding via
Problem12.8
a learned linear transformation, and these representations are fed into the transformer
network. Once again, standard 1D positional encodings are learned.
This is an encoder model with a <cls> token (see figures 12.10–12.11). However,
unlike BERT, it uses supervised pre-training on a large database of 303 million labeled
images from 18,000 classes. The <cls> token is mapped via a final network layer to
create activations that are fed into a softmax function to generate class probabilities.
After pre-training, the system is applied to the final classification task by replacing this
Draft: please send errata to udlbookmail@gmail.com.

230 12 Transformers
Figure12.16ImageGPT.a)ImagesgeneratedfromtheautoregressiveImageGPT
model. The top-left pixel is drawn from the estimated empirical distribution at
thisposition. Subsequentpixelsaregeneratedinturn,conditionedontheprevious
ones, working along the rows until the bottom-right of the image is reached. For
each pixel, the transformer decoder generates a conditional distribution as in
equation 12.15, and a sample is drawn. The extended sequence is then fed back
into the network to generate the next pixel, and so on. b) Image completion.
In each case, the lower half of the image is removed (top row), and ImageGPT
completes the remaining part pixel by pixel (three different completions shown).
Adapted from https://openai.com/blog/image-gpt/.
final layer with one that maps to the desired number of classes and is fine-tuned.
FortheImageNetbenchmark,thissystemachievedan11.45%top-1errorrate. How-
ever,itdidnotperformaswellasthebestcontemporaryconvolutionalnetworkswithout
supervised pre-training. The strong inductive bias of convolutional networks can only
be superseded by employing extremely large amounts of training data.
12.10.3 Multi-scale vision transformers
The Vision Transformer differs from convolutional architectures in that it operates on a
single scale and has a receptive field that covers the whole image. Several transformer
models that process the image at multiple scales have been proposed. Similarly to
convolutional networks, these generally start with small high resolution patches and few
channels and gradually enlarge the receptive field, decrease the spatial resolution and
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

12.10 Transformers for images 231
Figure12.17Visiontransformer. TheVisionTransformer(ViT)breakstheimage
into a grid of patches (16×16 in the original implementation). Each of these
is projected via a learned linear transformation to become a patch embedding.
These patch embeddings are fed into a transformer encoder network, and the
<cls> token is used to predict the class probabilities.
Figure 12.18 Shiftedwindow(SWin)transformer(Liuetal.,2021c). a)Original
image. b) The SWin transformer breaks the image into a grid of windows and
eachofthesewindowsintoasub-gridofpatches. Thetransformernetworkapplies
self-attentiontothepatcheswithineachwindowindependently(i.e.,patchesonly
attend to other patches in the same window). c) Each alternate layer shifts the
windows so that the subsets of patches that interact with one another change,
and information can propagate across the whole image. d) After several layers,
the2×2blocksofpatchrepresentationsareconcatenatedtoincreasetheeffective
patch(andwindow)size. e)Alternatelayersuseshiftedwindowsatthisnewlower
resolution. f)Eventually,theresolutionissuchthatthereisjustasinglewindow,
and the patches span the entire image.
Draft: please send errata to udlbookmail@gmail.com.