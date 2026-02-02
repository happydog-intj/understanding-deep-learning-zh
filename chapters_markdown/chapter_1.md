# Chapter 1

*Pages: 15-30*

---

Chapter 1
Introduction
Artificial intelligence, orAI,isconcernedwithbuildingsystemsthatsimulateintelligent
behavior. It encompasses a wide range of approaches, including those based on logic,
search, and probabilistic reasoning. Machine learning is a subset of AI that learns to
make decisions by fitting mathematical models to observed data. This area has seen
explosive growth and is now (incorrectly) almost synonymous with the term AI.
A deep neural network (or deep network for short) is a type of machine learning
model, and the process of fitting these models to data is referred to as deep learning. At
thetimeofwriting, deepnetworksarethemostpowerfulandpracticalmachinelearning
modelsandareoftenencounteredinday-to-daylife. Itiscommonplacetotranslatetext
to another language using a natural language processing algorithm, to search for images
of a given object using a computer vision system, or to converse with a digital assistant
viaaspeechrecognitioninterface. Alloftheseapplicationsarepoweredbydeeplearning.
As the title suggests, this book aims to help a reader new to this field understand
the principles behind deep learning. The book is neither terribly theoretical (there are
no proofs) nor extremely practical (there is almost no code). The goal is to explain the
underlying ideas; after consuming this volume, the reader will be able to apply deep
learning to novel situations where there is no existing recipe for success.
Machinelearningmethodscancoarselybedividedintothreeareas: supervised,unsu-
pervised, and reinforcement learning. At the time of writing, the cutting-edge methods
in all three areas rely on deep learning (figure 1.1). This introductory chapter describes
thesethreeareasatahighlevel,andthistaxonomyisalsolooselyreflectedinthebook’s
organization. Whether we like it or not, deep learning is poised to change our world,
and this change will not all be positive. Hence, this chapter also contains a brief primer
on AI ethics. We conclude with advice on how to make the most of this book.
1.1 Supervised learning
Supervised learning models define a mapping from input data to an output prediction.
In the following sections, we discuss the inputs, the outputs, the model itself, and what
is meant by “training” a model.
Draft: please send errata to udlbookmail@gmail.com.

2 1 Introduction
Figure 1.1 Machine learning is an area
of artificial intelligence that fits math-
ematical models to observed data. It
can coarsely be divided into supervised
learning, unsupervised learning, and re-
inforcement learning. Deep neural net-
works contribute to each of these areas.
1.1.1 Regression and classification problems
Figure 1.2 depicts several regression and classification problems. In each case, there is a
meaningfulreal-worldinput(asentence,asoundfile,animage,etc.),andthisisencoded
asavectorofnumbers. Thisvectorformsthemodelinput. Themodelmapstheinputto
an output vector which is then “translated” back to a meaningful real-world prediction.
For now, we focus on the inputs and outputs and treat the model as a black box that
ingests a vector of numbers and returns another vector of numbers.
The model in figure 1.2a predicts the price of a house based on input characteristics
such as the square footage and the number of bedrooms. This is a regression problem
because the model returns a continuous number (rather than a category assignment).
In contrast, the model in figure 1.2b takes the chemical structure of a molecule as an
inputandpredictsboththefreezingandboilingpoints. Thisisamultivariate regression
problem since it predicts more than one number.
Themodelinfigure1.2creceivesatextstringcontainingarestaurantreviewasinput
and predicts whether the review is positive or negative. This is a binary classification
problem because the model attempts to assign the input to one of two categories. The
output vector contains the probabilities that the input belongs to each category. Fig-
ures 1.2d and 1.2e depict multiclass classification problems. Here, the model assigns the
input to one of N >2 categories. In the first case, the input is an audio file, and the
model predicts which genre of music it contains. In the second case, the input is an
image, and the model predicts which object it contains. In each case, the model returns
a vector of size N that contains the probabilities of the N categories.
1.1.2 Inputs
The input data in figure 1.2 varies widely. In the house pricing example, the input is a
fixed-length vector containing values that characterize the property. This is an example
of tabular data because it has no internal structure; if we change the order of the inputs
and build a new model, then we expect the model prediction to remain the same.
Conversely, the input in the restaurant review example is a body of text. This may
be of variable length depending on the number of words in the review, and here input
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

1.1 Supervised learning 3
Figure1.2Regressionandclassificationproblems. a)Thisregressionmodeltakes
a vector of numbers that characterize a property and predicts its price. b) This
multivariate regression model takes the structure of a chemical molecule and
predictsitsfreezingandboilingpoints. c)Thisbinaryclassificationmodeltakesa
restaurantreviewandclassifiesitaseitherpositiveornegative. d)Thismulticlass
classificationproblemassignsasnippetofaudiotooneofN genres. e)Asecond
multiclassclassificationprobleminwhichthemodelclassifiesanimageaccording
to which of N possible objects it might contain.
Draft: please send errata to udlbookmail@gmail.com.

4 1 Introduction
Figure1.3Machinelearningmodel. Themodelrepresentsafamilyofrelationships
thatrelatetheinput(ageofchild)totheoutput(heightofchild). Theparticular
relationship is chosen using training data, which consists of input/output pairs
(orange points). When we train the model, we search through the possible re-
lationships for one that describes the data well. Here, the trained model is the
cyan curve and can be used to compute the height for any age.
order is important; my wife ate the chicken is not the same as the chicken ate my wife.
The text must be encoded into numerical form before passing it to the model. Here, we
use a fixed vocabulary of size 10,000 and simply concatenate the word indices.
Forthemusicclassificationexample, theinputvectormightbeoffixedsize(perhaps
a10-secondclip)butisveryhigh-dimensional(i.e.,containsmanyentries). Digitalaudio
is usually sampled at 44.1 kHz and represented by 16-bit integers, so a ten-second clip
consists of 441,000 integers. Clearly, supervised learning models will have to be able to
process sizeable inputs. The input in the image classification example (which consists
of the concatenated RGB values at every pixel) is also enormous. Moreover, it contains
spatial structure; two pixels above and below one another are closely related, even if
they are not adjacent in the input vector.
Finally,considertheinputforthemodelthatpredictsthefreezingandboilingpoints
ofthemolecule. Amoleculemaycontainvaryingnumbersofatomsthatcanbeconnected
in different ways. In this case, the model must ingest both the geometric structure of
the molecule and the constituent atoms to the model.
1.1.3 Machine learning models
Untilnow,wehavetreatedthemachinelearningmodelasablackboxthattakesaninput
vector and returns an output vector. But what exactly is in this black box? Consider a
model to predict the height of a child from their age (figure 1.3). The machine learning
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

1.1 Supervised learning 5
model is a mathematical equation that describes how the average height varies as a
function of age (cyan curve in figure 1.3). When we run the age through this equation,
itreturnstheheight. Forexample, iftheageis10years, thenwepredictthattheheight
will be 139 cm.
More precisely, the model represents a family of equations mapping the input to
the output (i.e., a family of different cyan curves). The particular equation (curve) is
chosen using training data (examples of input/output pairs). In figure 1.3, these pairs
arerepresentedbytheorangepoints,andwecanseethatthemodel(cyanline)describes
thesedatareasonably. Whenwetalkabouttrainingorfittingamodel, wemeanthatwe
search through the family of possible equations (possible cyan curves) relating input to
output to find the one that describes the training data most accurately.
Itfollowsthatthemodelsinfigure1.2requirelabeledinput/outputpairsfortraining.
For example, the music classification model would require a large number of audio clips
where a human expert had identified the genre of each. These input/output pairs take
theroleofateacherorsupervisorforthetrainingprocess,andthisgivesrisetotheterm
supervised learning.
1.1.4 Deep neural networks
Thisbookconcernsdeepneuralnetworks,whichareaparticularlyusefultypeofmachine
learning model. They are equations that can represent an extremely broad family of
relationships between input and output, and where it is particularly easy to search
through this family to find the relationship that describes the training data.
Deep neural networks can process inputs that are very large, of variable length,
and contain various kinds of internal structures. They can output single real numbers
(regression),multiplenumbers(multivariateregression),orprobabilitiesovertwoormore
classes (binary and multiclass classification, respectively). As we shall see in the next
section, their outputs may also be very large, of variable length, and contain internal
structure. Itisprobablyhardtoimagineequationswiththeseproperties,andthereader
should endeavor to suspend disbelief for now.
1.1.5 Structured outputs
Figure1.4adepictsamultivariatebinaryclassificationmodelforsemanticsegmentation.
Here, every pixel of an input image is assigned a binary label that indicates whether it
belongs to a cow or the background. Figure 1.4b shows a multivariate regression model
where the input is an image of a street scene and the output is the depth at each pixel.
In both cases, the output is high-dimensional and structured. However, this structure is
closely tied to the input, and this can be exploited; if a pixel is labeled as “cow,” then a
neighbor with a similar RGB value probably has the same label.
Figures 1.4c–e depict three models where the output has a complex structure that is
not so closely tied to the input. Figure 1.4c shows a model where the input is an audio
file and the output is the transcribed words from that file. Figure 1.4d is a translation
Draft: please send errata to udlbookmail@gmail.com.

6 1 Introduction
Figure 1.4 Supervised learning tasks with structured outputs. a) This semantic
segmentation model maps an RGB image to a binary image indicating whether
each pixel belongs to the background or a cow (adapted from Noh et al., 2015).
b) This monocular depth estimation model maps an RGB image to an output
image where each pixel represents the depth (adapted from Cordts et al., 2016).
c) This audio transcription model maps an audio sample to a transcription of
the spoken words in the audio. d) This translation model maps an English text
stringtoitsFrenchtranslation. e)Thisimagesynthesismodelmapsacaptionto
animage(examplefromhttps://openai.com/dall-e-2/). Ineachcase,theoutput
has a complex internal structure or grammar. In some cases, many outputs are
compatible with the input.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

1.2 Unsupervised learning 7
modelinwhichtheinputisabodyoftextinEnglish,andtheoutputcontainstheFrench
translation. Figure 1.4e depicts a very challenging task in which the input is descriptive
text, and the model must produce an image that matches this description.
Inprinciple, thelatterthreetaskscanbetackledinthestandardsupervisedlearning
framework, but they are more diﬀicult for two reasons. First, the output may genuinely
beambiguous;therearemultiplevalidtranslationsfromanEnglishsentencetoaFrench
one and multiple images that are compatible with any caption. Second, the output
contains considerable structure; not all strings of words make valid English and French
sentences, and not all collections of RGB values make plausible images. In addition to
learning the mapping, we also have to respect the “grammar” of the output.
Fortunately, this “grammar” can be learned without the need for output labels. For
example, wecanlearnhowtoformvalidEnglishsentencesbylearningthestatisticsofa
large corpus of text data. This provides a connection with the next section of the book,
which considers unsupervised learning models.
1.2 Unsupervised learning
Constructing a model from input data without corresponding output labels is termed
unsupervised learning; theabsenceofoutputlabelsmeanstherecanbeno“supervision.”
Rather than learning a mapping from input to output, the goal is to describe or under-
stand the structure of the data. As was the case for supervised learning, the data may
have very different characteristics; it may be discrete or continuous, low-dimensional or
high-dimensional, and of constant or variable length.
1.2.1 Generative models
This book focuses on generative unsupervised models, which learn to synthesize new
data examples that are statistically indistinguishable from the training data. Some
generativemodelsexplicitlydescribetheprobabilitydistributionovertheinputdataand
herenewexamplesaregeneratedbysamplingfromthisdistribution. Othersmerelylearn
a mechanism to generate new examples without explicitly describing their distribution.
State-of-the-art generative models can synthesize examples that are extremely plau-
sible but distinct from the training examples. They have been particularly successful
at generating images (figure 1.5) and text (figure 1.6). They can also synthesize data
under the constraint that some outputs are predetermined (termed conditional genera-
tion). Examples include image inpainting (figure 1.7) and text completion (figure 1.8).
Indeed, modern generative models for text are so powerful that they can appear intel-
ligent. Given a body of text followed by a question, the model can often “fill in” the
missing answer by generating the most likely completion of the document. However, in
reality, the model only knows about the statistics of language and does not understand
the significance of its answers.
Draft: please send errata to udlbookmail@gmail.com.

8 1 Introduction
Figure 1.5 Generative models for images. Left: two images were generated from
a model trained on pictures of cats. These are not real cats, but samples from a
probabilitymodel. Right: twoimagesgeneratedfromamodeltrainedonimages
of buildings. Adapted from Karras et al. (2020b).
ThemoonhadrisenbythetimeIreachedtheedgeoftheforest,andthelightthatfilteredthroughthe
treeswassilverandcold. Ishivered, thoughIwasnotcold, andquickenedmypace. Ihadneverbeen
so far from the village before, and I was not sure what to expect. I had been walking for hours, and I
was tired and hungry. I had left in such a hurry that I had not thought to pack any food, and I had
notthoughttobringaweapon. Iwasunarmedandaloneinastrangeplace,andIdidnotknowwhat
Iwasdoing.
IhadbeenwalkingforsolongthatIhadlostallsenseoftime,andIhadnoideahowfarIhadcome.
I only knew that I had to keep going. I had to find her. I was getting close. I could feel it. She was
nearby,andshewasintrouble. Ihadtofindherandhelpher,beforeitwastoolate.
Figure 1.6 Short story synthesized from a generative model of text data. The
model describes a probability distribution that assigns a probability to every
output string. Sampling from the model creates strings that follow the statistics
of the training data (here, short stories) but have never been seen before.
Figure 1.7 Inpainting. In the original image (left), the boy is obscured by metal
cables. Theseundesirableregions(center)areremovedandthegenerativemodel
synthesizes a new image (right) under the constraint that the remaining pixels
must stay the same. Adapted from Saharia et al. (2022a).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

1.2 Unsupervised learning 9
I was a little nervous before my first lecture at the University of Bath. It seemed like there were
hundredsofstudentsandtheylookedintimidating. Isteppeduptothelecternandwasabouttospeak
whensomethingbizarrehappened.
Suddenly, the room was filled with a deafening noise, like a giant roar. It was so loud that I
couldn’t hear anything else and I had to cover my ears. I could see the students looking around, con-
fusedandfrightened. Then,asquicklyasithadstarted,thenoisestoppedandtheroomwassilentagain.
Istoodthereforafewmoments,tryingtomakesenseofwhathadjusthappened. ThenIrealizedthat
thestudentswereallstaringatme,waitingformetosaysomething. Itriedtothinkofsomethingwitty
orclevertosay,butmymindwasblank. SoIjustsaid,“Well,thatwasstrange,’andthenIstartedmy
lecture.
Figure 1.8 Conditional text synthesis. Given an initial body of text (in black),
generative models of text can continue the string plausibly by synthesizing the
“missing”remainingpartofthestring. GeneratedbyGPT3(Brownetal.,2020).
Figure 1.9 Variation of the human face. The human face contains roughly 42
muscles, so it’s possible to describe most of the variation in images of the same
personinthesamelightingwithjust42numbers. Ingeneral,datasetsofimages,
music, and text can be described by a relatively small number of underlying
variables although it is typically more diﬀicult to tie these to particular physical
mechanisms. Images from Dynamic FACES database (Holland et al., 2019).
1.2.2 Latent variables
Some(butnotall)generativemodelsexploitthefactthatdatacanbelowerdimensional
than the raw number of observed variables suggests. For example, the number of valid
andmeaningfulEnglishsentencesismuchsmallerthanthenumberofstringscreatedby
drawing words at random. Similarly, real-world images are a tiny subset of the images
that can be created by drawing random red, green, and blue (RGB) values for every
pixel. This is because images are generated by physical processes (see figure 1.9).
Thisleadstotheideathatwecandescribeeachdataexampleusingasmallernumber
of underlying latent variables. Here, the role of deep learning is to describe the mapping
betweentheselatentvariablesandthedata. Thelatentvariablestypicallyhaveasimple
Draft: please send errata to udlbookmail@gmail.com.

10 1 Introduction
Figure 1.10Latentvariables. Manygenerativemodelsuseadeeplearningmodel
to describe the relationship between a low-dimensional “latent” variable and the
observed high-dimensional data. The latent variables have a simple probability
distributionbydesign. Hence,newexamplescanbegeneratedbysamplingfrom
thesimpledistributionoverthelatentvariablesandthenusingthedeeplearning
model to map the sample to the observed data space.
Figure 1.11 Image interpolation. In each row the left and right images are real
and the three images in between represent a sequence of interpolations created
byagenerativemodel. Thegenerativemodelsthatunderpintheseinterpolations
havelearnedthatallimagescanbecreatedbyasetofunderlyinglatentvariables.
Byfindingthesevariablesforthetworealimages,interpolatingtheirvalues,and
then using these intermediate variables to create new images, we can generate
intermediate results that are both visually plausible and mix the characteristics
of the two original images. Top row adapted from Sauer et al. (2022). Bottom
row adapted from Ramesh et al. (2022).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

1.3 Reinforcement learning 11
Figure 1.12 Multiple images generated from the caption “A teddy bear on a
skateboard in Times Square.” Generated by DALL·E-2 (Ramesh et al., 2022).
probability distribution by design. By sampling from this distribution and passing the
result through the deep learning model, we can create new samples (figure 1.10).
Thesemodelsleadtonewmethodsformanipulatingrealdata. Forexample,consider
findingthelatentvariablesthatunderpintworealexamples. Wecaninterpolatebetween
these examples by interpolating between their latent representations and mapping the
intermediate positions back into the data space (figure 1.11).
1.2.3 Connecting supervised and unsupervised learning
Generative models with latent variables can also benefit supervised learning models
where the outputs have structure (figure 1.4). For example, consider learning to predict
the images corresponding to a caption. Rather than directly map the text input to an
image, we can learn a relation between latent variables that explain the text and the
latent variables that explain the image.
This has three advantages. First, we may need fewer text/image pairs to learn this
mapping now that the inputs and outputs are lower dimensional. Second, we are more
likely to generate a plausible-looking image; any sensible values of the latent variables
should produce something that looks like a plausible example. Third, if we introduce
randomnesstoeitherthemappingbetweenthetwosetsoflatentvariablesorthemapping
fromthelatentvariablestotheimage,thenwecangeneratemultipleimagesthatareall
described well by the caption (figure 1.12).
1.3 Reinforcement learning
The final area of machine learning is reinforcement learning. This paradigm introduces
the idea of an agent which lives in a world and can perform certain actions at each time
step. The actions change the state of the system but not necessarily in a deterministic
way. Taking an action can also produce rewards, and the goal of reinforcement learning
Draft: please send errata to udlbookmail@gmail.com.

12 1 Introduction
is for the agent to learn to choose actions that lead to high rewards on average.
One complication is that the reward may occur some time after the action is taken,
so associating a reward with an action is not straightforward. This is known as the
temporal credit assignment problem. As the agent learns, it must trade off exploration
andexploitationofwhatitalreadyknows; perhapstheagenthasalreadylearnedhowto
receive modest rewards; should it follow this strategy (exploit what it knows), or should
it try different actions to see if it can improve (explore other opportunities)?
1.3.1 Two examples
Consider teaching a humanoid robot to locomote. The robot can perform a limited
number of actions at a given time (moving various joints), and these change the state of
the world (its pose). We might reward the robot for reaching checkpoints in an obstacle
course. To reach each checkpoint, it must perform many actions, and it’s unclear which
ones contributed to the reward when it is received and which were irrelevant. This is an
example of the temporal credit assignment problem.
Asecondexampleislearningtoplaychess. Again,theagenthasasetofvalidactions
(chess moves) at any given time. However, these actions change the state of the system
in a non-deterministic way; for any choice of action, the opposing player might respond
withmanydifferentmoves. Here,wemightsetuparewardstructurebasedoncapturing
piecesorjusthaveasinglerewardattheendofthegameforwinning. Inthelattercase,
the temporal credit assignment problem is extreme; the system must learn which of the
many moves it made were instrumental to success or failure.
The exploration-exploitation trade-off is also apparent in these two examples. The
robot may have discovered that it can make progress by lying on its side and pushing
withoneleg. Thisstrategywillmovetherobotandyieldsrewards,butmuchmoreslowly
than the optimal solution: to balance on its legs and walk. So, it faces a choice between
exploitingwhatitalreadyknows(howtoslidealongthefloorawkwardly)andexploring
the space of actions (which might result in much faster locomotion). Similarly, in the
chess example, the agent may learn a reasonable sequence of opening moves. Should it
exploit this knowledge or explore different opening sequences?
Itisperhapsnotobvioushowdeeplearningfitsintothereinforcementlearningframe-
work. There are several possible approaches, but one technique is to use deep networks
to build a mapping from the observed world state to an action. This is known as a
policy network. In the robot example, the policy network would learn a mapping from
its sensor measurements to joint movements. In the chess example, the network would
learnamappingfromthecurrentstateoftheboardtothechoiceofmove(figure1.13).
1.4 Ethics
It would be irresponsible to write this book without discussing the ethical implications
of artificial intelligence. This potent technology will change the world to at least the
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

1.4 Ethics 13
Figure 1.13 Policy networks for reinforcement learning. One way to incorporate
deepneuralnetworksintoreinforcementlearningistousethemtodefineamap-
pingfromthestate(herepositiononchessboard)totheactions(possiblemoves).
This mapping is known as a policy.
sameextentaselectricity,theinternalcombustionengine,thetransistor,ortheinternet.
The potential benefits in healthcare, design, entertainment, transport, education, and
almosteveryareaofcommerceareenormous. However,scientistsandengineersareoften
unrealistically optimistic about the outcomes of their work, and the potential for harm
is just as great. The following paragraphs highlight five concerns.
Bias and fairness: If we train a system to predict salary levels for individuals based
on historical data, then this system will reproduce historical biases; for example, it will
probably predict that women should be paid less than men. Several such cases have
already become international news stories: an AI system for super-resolving face images
made non-white people look more white; a system for generating images produced only
pictures of men when asked to synthesize pictures of lawyers. Careless application of
algorithmicdecision-makingusingAIhasthepotentialtoentrenchoraggravateexisting
biases. See Binns (2018) for further discussion.
Explainability: Deep learning systems make decisions, but we do not usually know
exactly how or based on what information. They may be enormous, and there is no way
we can understand how they work based on examination. This has led to the sub-field
of explainable AI. One moderately successful area is producing local explanations; we
cannotexplaintheentiresystem,butwecanproduceaninterpretabledescriptionofwhy
a particular decision was made. However, it remains unknown whether it is possible to
build complex decision-making systems that are fully transparent to their users or even
their creators. See Grennan et al. (2022) for further information.
Weaponizing AI: All significant technologies have been applied directly or indirectly
toward war. Sadly, violent conflict seems to be an inevitable feature of human behavior.
AI is arguably the most powerful technology ever built and will doubtless be deployed
extensively in a military context. Indeed, this is already happening (Heikkilä, 2022).
Draft: please send errata to udlbookmail@gmail.com.

14 1 Introduction
Concentrating power: It is not from a benevolent interest in improving the lot of the
human race that the world’s most powerful companies are investing heavily in artifi-
cial intelligence. They know that these technologies will allow them to reap enormous
profits. Like any advanced technology, deep learning is likely to concentrate power in
the hands of the few organizations that control it. Automating jobs that are currently
donebyhumanswillchangetheeconomicenvironmentanddisproportionatelyaffectthe
livelihoods of lower-paid workers with fewer skills. Optimists argue similar disruptions
happened during the industrial revolution and resulted in shorter working hours. The
truthisthatwesimplydonotknowwhateffectsthelarge-scaleadoptionofAIwillhave
on society (see David, 2015).
Existential risk: The major existential risks to the human race all result from tech-
nology. Climate change has been driven by industrialization. Nuclear weapons derive
from the study of physics. Pandemics are more probable and spread faster because in-
novations in transport, agriculture, and construction have allowed a larger, denser, and
more interconnected population. Artificial intelligence brings new existential risks. We
should be very cautious about building systems that are more capable and extensible
than human beings. In the most optimistic case, it will put vast power in the hands
of the owners. In the most pessimistic case, we will be unable to control it or even
understand its motives (see Tegmark, 2018).
This list is far from exhaustive. AI could also enable surveillance, disinformation,
violations of privacy, fraud, and manipulation of financial markets, and the energy re-
quired to train AI systems contributes to climate change. Moreover, these concerns are
not speculative; there are already many examples of ethically dubious applications of
AI (consult Dao, 2021, for a partial list). In addition, the recent history of the inter-
net has shown how new technology can cause harm in unexpected ways. The online
community of the eighties and early nineties could hardly have predicted the prolifera-
tion of fake news, spam, online harassment, fraud, cyberbullying, incel culture, political
manipulation, doxxing, online radicalization, and revenge porn.
Everyone studying or researching (or writing books about) AI should contemplate
to what degree scientists are accountable for the uses of their technology. We should
consider that capitalism primarily drives the development of AI and that legal advances
and deployment for social good are likely to lag significantly behind. We should reflect
on whether it’s possible, as scientists and engineers, to control progress in this field and
to reduce the potential for harm. We should consider what kind of organizations we
are prepared to work for. How serious are they in their commitment to reducing the
potential harms of AI? Are they simply “ethics-washing” to reduce reputational risk, or
do they actually implement mechanisms to halt ethically suspect projects?
All readers are encouraged to investigate these issues further. The online course
at https://ethics-of-ai.mooc.fi/ is a useful introductory resource. If you are a professor
teaching from this book, you are encouraged to raise these issues with your students. If
you are a student taking a course where this is not done, then lobby your professor to
make this happen. If you are deploying or researching AI in a corporate environment,
you are encouraged to scrutinize your employer’s values and to help change them (or
leave) if they are wanting.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

1.5 Structure of book 15
1.5 Structure of book
The structure of the book follows the structure of this introduction. Chapters 2–9 walk
throughthesupervisedlearningpipeline. Wedescribeshallowanddeepneuralnetworks
and discuss how to train them and measure and improve their performance. Chap-
ters 10–13 describe common architectural variations of deep neural networks, including
convolutional networks, residual connections, and transformers. These architectures are
used across supervised, unsupervised, and reinforcement learning.
Chapters 14–18 tackle unsupervised learning using deep neural networks. We devote
a chapter each to four modern deep generative models: generative adversarial networks,
variational autoencoders, normalizing flows, and diffusion models. Chapter 19 is a brief
introduction to deep reinforcement learning. This is a topic that easily justifies its own
book, so the treatment is necessarily superficial. However, this treatment is intended to
be a good starting point for readers unfamiliar with this area.
Despite the title of this book, some aspects of deep learning remain poorly under-
stood. Chapter20posessomefundamentalquestions. Whyaredeepnetworkssoeasyto
train? Why do they generalize so well? Why do they need to be so large? Do they need
to be deep? Along the way, we explore unexpected phenomena such as the structure
of the loss function, double descent, grokking, and lottery tickets. The book concludes
with chapter 21, which discusses ethics and deep learning.
1.6 Other books
This book is self-contained but is limited to coverage of deep learning. It is intended to
bethespiritualsuccessortoDeep Learning(Goodfellowetal.,2016)whichisafantastic
resourcebutdoesnotcoverrecentadvances. Forabroaderlookatmachinelearning,the
most up-to-date and encyclopedic resource is Probabilistic Machine Learning (Murphy,
2022, 2023). However, Pattern Recognition and Machine Learning (Bishop, 2006) is still
an excellent and relevant book.
Ifyouenjoythisbook,thenmypreviousvolume,ComputerVision: Models,Learning,
and Inference (Prince, 2012), is still worth reading. Some parts have dated badly, but it
contains a thorough introduction to probability, including Bayesian methods, and good
introductorycoverageoflatentvariablemodels,geometryforcomputervision,Gaussian
processes,andgraphicalmodels. Itusesidenticalnotationtothisbookandcanbefound
online. AdetailedtreatmentofgraphicalmodelscanbefoundinProbabilistic Graphical
Models: Principles and Techniques (Koller & Friedman, 2009), and Gaussian processes
arecoveredbyGaussianProcessesforMachineLearning(Williams&Rasmussen,2006).
For background mathematics, consult Mathematics for Machine Learning (Deisen-
rothetal.,2020). Foramorecoding-orientedapproach,consultDiveintoDeepLearning
(Zhangetal.,2023). ThebestoverviewsforcomputervisionareComputerVision: Algo-
rithms and Applications(Szeliski,2022),andFoundations of Computer Vision(Torralba
et al., 2024). A good starting point to learn about graph neural networks is Graph Rep-
resentation Learning (Hamilton, 2020). The definitive work on reinforcement learning
Draft: please send errata to udlbookmail@gmail.com.

16 1 Introduction
is Reinforcement Learning: An Introduction (Sutton & Barto, 2018). A good initial
resource is Foundations of Deep Reinforcement Learning (Graesser & Keng, 2019).
1.7 How to read this book
Most remaining chapters in this book contain a main body of text, a notes section, and
asetofproblems. Themainbodyofthetextisintendedtobeself-containedandcanbe
readwithoutrecoursetotheotherpartsofthechapter. Asmuchaspossible,background
mathematics is incorporated into the main body of the text. However, for larger topics
thatwouldbeadistractiontothemainthreadoftheargument,thebackgroundmaterial
isappendicized, andareferenceisprovidedinthemargin. Mostnotationinthisbookis
AppendixA
standard. However, some conventions are less widely used, and the reader is encouraged
Notation
to consult appendix A before proceeding.
The main body of text includes many novel illustrations and visualizations of deep
learning models and results. I’ve worked hard to provide new explanations of existing
ideas rather than merely curate the work of others. Deep learning is a new field, and
sometimes phenomena are poorly understood. I try to make it clear where this is the
case and when my explanations should be treated with caution.
References are included in the main body of the chapter only where results are de-
picted. Instead, they can be found in the notes section at the end of the chapter. I do
not generally respect historical precedent in the main text; if an ancestor of a current
techniqueisnolongeruseful,thenIwillnotmentionit. However,thehistoricaldevelop-
mentofthefieldisdescribedinthenotessection,andhopefully,creditisfairlyassigned.
The notes are organized into paragraphs and provide pointers for further reading. They
should help the reader orient themselves within the sub-area and understand how it re-
lates to other parts of machinelearning. The notes are less self-contained than the main
text. Dependingonyourlevelofbackgroundknowledgeandinterest,youmayfindthese
sections more or less useful.
Eachchapterhasanumberofassociatedproblems. Theyarereferencedinthemargin
of the main text at the point that they should be attempted. As George Pólya noted,
“Mathematics,yousee,isnotaspectatorsport.” Hewascorrect,andIhighlyrecommend
that you attempt the problems as you go. In some cases, they provide insights that will
helpyouunderstandthemaintext. Problemsforwhichtheanswersareprovidedonthe
associated website (http://udlbook.com) are indicated with an asterisk. Additionally,
Pythonnotebooksthatwillhelpyouunderstandtheideasinthisbookarealsoavailable
via the website, and these are also referenced in the margins of the text. Indeed, if
Notebook1.1
you are feeling rusty, it might be worth working through the notebook on background
Background
mathematics mathematics right now.
Unfortunately, the pace of research in AI makes it inevitable that this book will be a
constant work in progress. If there are parts you find hard to understand, notable omis-
sions, or sections that seem extraneous, please get in touch via the associated website.
Together, we can make the next edition better.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.