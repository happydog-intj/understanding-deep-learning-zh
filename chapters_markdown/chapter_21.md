# Chapter 21

*Pages: 435-438*

---

Chapter 21
Deep learning and ethics
This chapter was written by Travis LaCroix and Simon J.D. Prince.
AIispoisedtochangesocietyforbetterorworse. Thesetechnologieshaveenormous
potential for social good (Taddeo & Floridi, 2018; Tomašev et al., 2020), including im-
portant roles in healthcare (Rajpurkar et al., 2022) and the fight against climate change
(Rolnick et al., 2023). However, they also have the potential for misuse and unintended
harm. This has led to the emergence of the field of AI ethics.
Themoderneraofdeeplearningstartedin2012withAlexNet,butsustainedinterest
in AI ethics did not follow immediately. Indeed, a workshop on fairness in machine
learning was rejected from NeurIPS 2013 for want of material. It wasn’t until 2016 that
AIEthicshadits“AlexNet”moment,withProPublica’sexposéonbiasintheCOMPAS
recidivism-prediction model (Angwin et al., 2016) and Cathy O’Neil’s book Weapons
of Math Destruction (O’Neil, 2016). Interest has swelled ever since; submissions to the
ConferenceonFairness,Accountability,andTransparency(FAccT)haveincreasednearly
ten-fold in the five years since its inception in 2018.
Inparallel,manyorganizationshaveproposedpolicyrecommendationsforresponsible
AI. Jobin et al. (2019) found 84 documents containing AI ethics principles, with 88%
releasedsince2016. Thisproliferationofnon-legislativepolicyagreements,whichdepend
on voluntary, non-binding cooperation, calls into question their eﬀicacy (McNamara
et al., 2018; Hagendorff, 2020; LaCroix & Mohseni, 2022). In short, AI Ethics is in its
infancy, and ethical considerations are often reactive rather than proactive.
ThischapterconsiderspotentialharmsarisingfromthedesignanduseofAIsystems.
Theseincludealgorithmicbias,lackofexplainability,dataprivacyviolations,militariza-
tion,fraud,andenvironmentalconcerns. Theaimisnottoprovideadviceonbeingmore
ethical. Instead, the goal is to express ideas and start conversations in key areas that
have received attention in philosophy, political science, and the broader social sciences.
21.1 Value alignment
WhenwedesignAIsystems,wewishtoensurethattheir“values”(objectives)arealigned
with those of humanity. This is sometimes called the value alignment problem (Russell, Problem21.1
Draft: please send errata to udlbookmail@gmail.com.

422 21 Deep learning and ethics
2019; Christian, 2020; Gabriel, 2020). This is challenging for three reasons. First, it’s
diﬀiculttodefineourvaluescompletelyandcorrectly. Second, itishardtoencodethese
valuesasobjectivesofanAImodel, andthird, itishardtoensurethatthemodellearns
to carry out these objectives.
In a machine learning model, the loss function is a proxy for our true objectives,
Problem21.2
and a misalignment between the two is termed the outer alignment problem (Hubinger
et al., 2019). To the extent that this proxy is inadequate, there will be “loopholes” that
the system can exploit to minimize its loss function while failing to satisfy the intended
objective. For example, consider training an RL agent to play chess. If the agent is
rewarded for capturing pieces, this may result in many drawn games rather than the
desired behavior (to win the game). In contrast, the inner alignment problem is to
ensure that the behavior of an AI system does not diverge from the intended objectives
even when the loss function is well specified. If the learning algorithm fails to find
the global minimum or the training data are unrepresentative, training can converge to
a solution that is misaligned with the true objective resulting in undesirable behavior
(Goldberg, 1987; Mitchell et al., 1992; Lehman & Stanley, 2008).
Gabriel (2020) divides the value alignment problem into technical and normative
components. The technical component concerns how we encode values into the models
so that they reliably do what they should. Some concrete problems, such as avoiding
rewardhackingandsafeexploration,mayhavepurelytechnicalsolutions(Amodeietal.,
2016). Incontrast,thenormativecomponentconcernswhatthecorrectvaluesareinthe
first place. There may be no single answer to this question, given the range of things
that different cultures and societies value. It’s important that the encoded values are
representative of everyone and not just culturally dominant subsets of society.
Another way to think about value alignment is as a structural problem that arises
when a human principal delegates tasks to an artificial agent (LaCroix, 2022). This is
similar to the principal-agent problem in economics (Laffont & Martimort, 2002), which
allows that there are competing incentives inherent in any relationship where one party
isexpectedtoactinanother’sbestinterests. IntheAIcontext,suchconflictsofinterest
can arise when either (i) the objectives are misspecified or (ii) there is an informational
asymmetry between the principal and the agent (figure 21.1).
Many topics in AI ethics can be understood in terms of this structural view of value
alignment. The following sections discuss problems of bias and fairness and artificial
moral agency (both pertaining to specifying objectives) and transparency and explain-
ability (both related to informational asymmetry).
21.1.1 Bias and fairness
From a purely scientific perspective, bias refers to statistical deviation from some norm.
InAI,itcanbeperniciouswhenthisdeviationdependsonillegitimatefactorsthatimpact
an output. For example, gender is irrelevant to job performance, so it is illegitimate to
use gender as a basis for hiring a candidate. Similarly, race is irrelevant to criminality,
so it is illegitimate to use race as a feature for recidivism prediction.
Bias in AI models can be introduced in various ways (Fazelpour & Danks, 2021):
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

21.1 Value alignment 423
Figure 21.1 Structural description of the value alignment problem. Problems
arise from a) misaligned objectives (e.g., bias) or b) informational asymmetries
betweena(human)principalandan(artificial)agent(e.g.,lackofexplainability).
Adapted from LaCroix (2025).
• Problem specification: Choosing a model’s goals requires a value judgment
about what is important to us, which allows for the creation of biases (Fazelpour
& Danks, 2021). Further biases may emerge if we fail to operationalize these
choices successfully and the problem specification fails to capture our intended
goals (Mitchell et al., 2021).
• Data: Algorithmic bias can result when the dataset is unrepresentative or incom-
plete (Danks & London, 2017). For example, the PULSE face super-resolution
algorithm (Menon et al., 2020) was trained on a database of photos of predom-
inantly white celebrities. When applied to a low-resolution portrait of Barack
Obama, it generated a photo of a white man (Vincent, 2020).
If the society in which training data are generated is structurally biased against
marginalized communities, even complete and representative datasets will elicit
biases(Mayson,2018). Forexample,BlackindividualsintheUShavebeenpoliced
and jailed more frequently than white individuals. Hence, historical data used to
train recidivism prediction models are already biased against Black communities.
• Modelingandvalidation: Choosingamathematicaldefinitiontomeasuremodel
fairness requires a value judgment. There exist distinct but equally-intuitive defi-
nitions that are logically inconsistent (Kleinberg et al., 2017; Chouldechova, 2017;
Berk et al., 2021). This suggests the need to move from a purely mathemati-
cal conceptualization of fairness toward a more substantive evaluation of whether
algorithms promote justice in practice (Green, 2022).
• Deployment: Deployed algorithms may interact with other algorithms, struc-
tures, or institutions in society to create complex feedback loops that entrench ex-
tant biases (O’Neil, 2016). For example, large language models like GPT3 (Brown
etal.,2020)aretrainedonwebdata. However, whenGPT3outputsarepublished Problem21.3
Draft: please send errata to udlbookmail@gmail.com.

424 21 Deep learning and ethics
Datacollection Pre-processing Training Post-processing
•Identifylackof •Modifylabels •Adversarialtraining •Changethresholds
•examplesor •Modifyinputdata •Regularizeforfairness •Trade-offaccuracy
•variatesandcollect •Modifyinput/ •Constraintobefair •forfairness
•outputpairs
Figure21.2Biasmitigation. Methodshavebeenproposedtocompensateforbias
at all stages of the training pipeline, from data collection to post-processing of
already trained models. See Barocas et al. (2023) and Mehrabi et al. (2022).
online,thetrainingdataforfuturemodelsisdegraded. Thismayexacerbatebiases
and generate novel societal harm (Falbo & LaCroix, 2022).
Unfairnesscanbeexacerbatedbyconsiderationsofintersectionality;socialcategories
can combine to create overlapping and interdependent systems of oppression. For ex-
ample, the discrimination experienced by a queer woman of color is not merely the
sum of the discrimination she might experience as queer, as gendered, or as racialized
(Crenshaw, 1991). Within AI, Buolamwini & Gebru (2018) showed that face analysis
algorithms trained primarily on lighter-skinned faces underperform for darker-skinned
faces. However, they perform even worse on combinations of features such as skin color
and gender than might be expected by considering those features independently.
Of course, steps can be taken to ensure that data are diverse, representative, and
complete. Butifthesocietyinwhichthetrainingdataaregeneratedisstructurallybiased
against marginalized communities, even completely accurate datasets will elicit biases.
In light of the potential for algorithmic bias and the lack of representation in training
datasetsdescribedabove,itisalsonecessarytoconsiderhowfailureratesfortheoutputs
of these systems are likely to exacerbate discrimination against already-marginalized
communities (Buolamwini & Gebru, 2018; Raji & Buolamwini, 2019; Raji et al., 2022).
Theresultingmodelsmaycodifyandentrenchsystemsofpowerandoppression,including
capitalism and classism; sexism, misogyny, and patriarchy; colonialism and imperialism;
racism and white supremacy; ableism; and cis- and heteronormativity. A perspective
on bias that maintains sensitivity to power dynamics requires accounting for historical
inequities and labor conditions encoded in data (Micelli et al., 2022).
To prevent this, we must actively ensure that our algorithms are fair. A naïve ap-
proach is fairness through unawareness which simply removes the protected attributes
(e.g.,race,gender)fromtheinputfeatures. Unfortunately,thisisineffective;theremain-
ing features can still carry information about the protected attributes. More practical
approachesfirstdefineamathematicalcriterionforfairness. Forexample,theseparation
measure in binary classification requires that the prediction yˆ is conditionally indepen-
dent of the protected variable a (e.g., race) given the true label y. Then they intervene
in various ways to minimize the deviation from this measure (figure 21.2).
Notebook21.1
Afurthercomplicatingfactoristhatwecannottellifanalgorithmisunfairtoacom-
Biasmitigation
munityortakestepstoavoidthisunlesswecanestablishcommunitymembership. Most
research on algorithmic bias and fairness has focused on ostensibly observable features
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.