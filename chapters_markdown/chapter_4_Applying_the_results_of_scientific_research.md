# Chapter 4: Applying the results of scientific research.

*Pages: 446-447*

---

432 21 Deep learning and ethics
persecutionofLGBTQ+individualsincountrieswherequeernessiscriminalized. Fourth,
with regard to transparency, explainability, and value alignment more generally, the
“gaydar”modelappearstopickuponspuriouscorrelationsduetopatternsingrooming,
presentation,andlifestyleratherthanfacialstructure, astheauthorsclaimed(Agüeray
Arcasetal.,2018). Fifth,withregardtodataprivacy,questionsariseregardingtheethics
of scraping “public” photos and sexual orientation labels from a dating website. Finally,
withregardtoscientificcommunication, theresearcherscommunicatedtheirresultsina
waythatwassuretogenerateheadlines: eventhetitleofthepaperisanoverstatementof
the model’s abilities: Deep Neural Networks Can Detect Sexual Orientation from Faces.
(They cannot.)
It should also be apparent that a facial-analysis model for determining sexual orien-
tation does nothing whatsoever to benefit the LGBTQ+ community. If it is to benefit
society, the most important question is whether a particular study, experiment, model,
application, or technology serves the interests of the community to which it pertains.
21.5 The value-free ideal of science
This chapter has enumerated a number of ways that the objectives of AI systems can
unintentionally, or through misuse, diverge from the values of humanity. We now argue
that scientists are not neutral actors; their values inevitably impinge on their work.
Perhaps this is surprising. There is a broad belief that science is—or ought to be—
objective. This is codified by the value-free ideal of science. Many would argue that
machine learning is objective because algorithms are just mathematics. However, analo-
gous to algorithmic bias (section 21.1.1), there are four stages at which the values of AI
practitioners can affect their work (Reiss & Sprenger, 2017):
1. The choice of research problem.
2. Gathering evidence related to a research problem.
3. Accepting a scientific hypothesis as an answer to a problem.
4. Applying the results of scientific research.
It is perhaps uncontroversial that values play a significant role in the first and last of
these stages. The initial selection of research problems and the choice of subsequent ap-
plicationsareinfluencedbytheinterestsofscientists, institutions, andfundingagencies.
However, the value-free ideal of science prescribes minimizing the influence of moral,
personal, social, political, and cultural values on the intervening scientific process. This
idea presupposes the value-neutrality thesis, which suggests that scientists can (at least
in principle) attend to stages (2) and (3) without making these value judgments.
However, whether intentional or not, values are embedded in machine learning re-
search. Most of these values would be classed as epistemic (e.g., performance, gener-
alization, building on past work, eﬀiciency, novelty). But deciding the set of values is
itself a value-laden decision; few papers explicitly discuss societal need, and fewer still
discuss potential negative impacts (Birhane et al., 2022b). Philosophers of science have
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

21.6 Responsible AI research as a collective action problem 433
questioned whether the value-free ideal of science is attainable or desirable. For exam-
ple, Longino (1990, 1996) argues that these epistemic values are not purely epistemic.
Kitcher (2011a,b) argues that scientists don’t typically care about truth itself; instead,
they pursue truths relevant to their goals and interests.
Machinelearningdependsoninductiveinferenceandishencepronetoinductiverisk.
Models are only constrained at the training data points, and the curse of dimensionality
meansthisisatinyproportionoftheinputspace; outputscanalwaysbewrong, regard-
less of how much data we use to train the model. It follows that choosing to accept or
reject a model prediction requires a value judgment: that the risks if we are wrong in
acceptance are lower than the risks if we are wrong in rejection.
Hence,theuseofinductiveinferenceimpliesthatmachinelearningmodelsaredeeply
value-laden (Johnson, 2022). In fact, if they were not, they would have no application:
it is precisely because they are value-laden that they are useful. Thus, accepting that
algorithms are used for ranking, sorting, filtering, recommending, categorizing, label-
ing, predicting, etc., in the real world implies that these processes will have real-world
effects. As machine learning systems become increasingly commercialized and applied,
they become more entrenched in the things we care about.
Theseinsightshaveimplicationsforresearcherswhobelievethatalgorithmsaresome-
howmoreobjectivethanhumandecision-makers(and,therefore,oughttoreplacehuman
decision-makers in areas where we think objectivity matters).
21.6 Responsible AI research as a collective action problem
Itiseasytodeferresponsibility. Studentsandprofessionalswhoreadthischaptermight
thinktheirworkissofarremovedfromtherealworldorasmallpartofalargermachine
that their actions could not make a difference. However, this is a mistake. Researchers
often have a choice about the projects to which they devote their time, the companies
or institutions for which they work, the knowledge they seek, the social and intellectual
circles in which they interact, and the way they communicate.
Doing the right thing, whatever that may comprise, often takes the form of a social
dilemma;thebestoutcomesdependuponcooperation,althoughitisn’tnecessarilyinany
Problem21.12
individual’s interest to cooperate: responsible AI research is a collective action problem.
21.6.1 Scientific communication
One positive step is to communicate responsibly. Misinformation spreads faster and
persists more readily than the truth in many types of social networks (LaCroix et al.,
2021; Ceylan et al., 2023). As such, it is important not to overstate machine learning
systems’ abilities (see case study above) and to avoid misleading anthropomorphism. It
is also important to be aware of the potential for the misapplication of machine learning
techniques. For example, pseudoscientific practices like phrenology and physiognomy
have found a surprising resurgence in AI (Stark & Hutson, 2022).
Draft: please send errata to udlbookmail@gmail.com.