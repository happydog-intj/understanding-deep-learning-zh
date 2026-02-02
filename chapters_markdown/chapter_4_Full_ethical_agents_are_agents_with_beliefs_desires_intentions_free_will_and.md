# Chapter 4: Full ethical agents are agents with beliefs, desires, intentions, free will, and

*Pages: 439-445*

---

21.1 Value alignment 425
that might be present in training data (e.g., gender). However, features of marginalized
communitiesmaybeunobservable,makingbiasmitigationevenmorediﬀicult. Examples
includequeerness(Tomasevetal.,2021),disabilitystatus,neurotype,class,andreligion.
A similar problem occurs when observable features have been excised from the training
data to prevent models from exploiting them.
21.1.2 Artificial moral agency
Many decision spaces do not include actions that carry moral weight. For example,
choosing the next chess move has no obvious moral consequence. However, elsewhere
actions can carry moral weight. Examples include decision-making in autonomous vehi-
cles (Awad et al., 2018; Evans et al., 2020), lethal autonomous weapons systems (Arkin,
2008a,b),andprofessionalservicerobotsforchildcare,elderlycare,andhealthcare(An-
derson & Anderson, 2008; Sharkey & Sharkey, 2012). As these systems become more
autonomous, they may need to make moral decisions independent of human input.
This leads to the notion of artificial moral agency. An artificial moral agent is
an autonomous AI system capable of making moral judgments. Moral agency can be
categorized in terms of increasing complexity (Moor, 2006):
1. Ethical impact agents are agents that affect a situation for the better or worse,
but are not designed with ethics in mind. Hence, almost any technology deployed
in society might count as an ethical impact agent.
2. Implicit ethical agents are ethical impact agents that include some in-built
safety features.
3. Explicit ethical agents can contextually follow general moral principles or rules
of ethical conduct.
4. Full ethical agents are agents with beliefs, desires, intentions, free will, and
consciousness of their actions.
Thefieldofmachineethicsseeksapproachestocreatingartificialmoralagents. These
approachescanbecategorizedastop-down,bottom-up,orhybrid(Allenetal.,2005). Top-
down (theory-driven) methods directly implement and hierarchically arrange concrete
rules based on some moral theory to guide ethical behavior. Asimov’s “Three Laws of
Robotics” are a trivial example of this approach.
In bottom-up (learning-driven) approaches, a model learns moral regularities from
data without explicit programming (Wallach et al., 2008). For example, Noothigattu
et al. (2018) designed a voting-based system for ethical decision-making that uses data
collected from humanpreferences in moral dilemmas to learn social preferences; the sys-
tem then summarizes and aggregates the results to render an “ethical” decision. Hybrid
approaches combine top-down and bottom-up approaches.
Someresearchershavequestionedtheveryideaofartificialmoralagencyandargued
thatmoralagencyisunnecessaryforensuringsafety(vanWynsberghe&Robbins,2019).
See Cervantes et al. (2019) for a recent survey of artificial moral agency and Tolmeijer
et al. (2020) for a recent survey on technical approaches to artificial moral agency.
Draft: please send errata to udlbookmail@gmail.com.

426 21 Deep learning and ethics
21.1.3 Transparency and opacity
A complex computational system is transparent if all of the details of its operation are
known. Asystemisexplainableifhumanscanunderstandhowitmakesdecisions. Inthe
absenceoftransparencyorexplainability, thereisanasymmetryofinformationbetween
the user and the AI system, which makes it hard to ensure value alignment.
Creel (2020) characterizes transparency at several levels of granularity. Functional
transparency refers to knowledge of the algorithmic functioning of the system (i.e., the
logical rules that map inputs to outputs). The methods in this book are described at
this level of detail. Structural transparency entails knowing how a program executes the
algorithm. Thiscanbeobscuredwhencommandswritteninhigh-levelprogramminglan-
guages are executed by machine code. Finally, run transparency requires understanding
how a program was executed in a particular instance. For deep networks, this includes
Problem21.4
knowledgeaboutthehardware,inputdata,trainingdata,andinteractionsthereof. None
of these can be ascertained by scrutinizing code.
Forexample,GPT3isfunctionallytransparent;itsarchitectureisdescribedinBrown
et al. (2020). However, it does not exhibit structural transparency as we do not have
access to the code, and it does not exhibit run transparency as we have no access to the
learned parameters, hardware, or training data. The subsequent version GPT4 is not
transparent at all. The details of how this commercial product works are unknown.
21.1.4 Explainability and interpretability
Even if a system is transparent, this does not imply that we can understand how a
decision is made or what information this decision is based on. Deep networks may
contain billions of parameters, so there is no way we can understand how they work
basedonexaminationalone. However,insomejurisdictions,thepublicmayhavearight
toanexplanation. Article22oftheEUGeneralDataProtectionRegulationsuggestsall
data subjects should have the right to “obtain an explanation of the decision reached”
in cases where a decision is based solely on automated processes.1
Thesediﬀicultieshaveledtothesub-fieldofexplainableAI.Onemoderatelysuccess-
ful area is producing local explanations. Although we can’t explain the entire system,
Notebook21.2
we can sometimes describe how a particular input was classified. For example, Local
Explainability
interpretable model-agnostic explanations or LIME (Ribeiro et al., 2016) samples the
model output at nearby inputs and uses these samples to construct a simpler model
(figure 21.3). This provides insight into the classification decision, even if the original
model is neither transparent nor explainable.
Itremainstobeseenwhetheritispossibletobuildcomplexdecision-makingsystems
that are fully understandable to their users or even their creators. There is also an
ongoing debate about what it means for a system to be explainable, understandable, or
interpretable (Erasmus et al., 2021); there is currently no concrete definition of these
concepts. See Molnar (2022) for more information.
1WhetherArticle22actuallymandatessucharightisdebatable(seeWachteretal.,2017).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

21.2 Intentional misuse 427
Figure 21.3 LIME. Output functions of deep networks are complex; in high di-
mensions, it’s hard to know why a decision was made or how to modify the
inputs to change it without access to the model. a) Consider trying to under-
stand why Pr(y = 1|x) is low at the white cross. LIME probes the network at
nearby points to see if it identifies these as Pr(y = 1|x) < 0.5 (cyan points) or
Pr(y = 1|x) ≥ 0.5 (gray points). It weights these points by proximity to the
point of interest (weight indicated by circle size). b) The weighted points are
usedtotrainasimplermodel(here,logisticregression—alinearfunctionpassed
throughasigmoid). c)Nearthewhitecross,thisapproximationisclosetod)the
original function. Even though we did not have access to the original model, we
can deduce from the parameters of this approximate model, that if we increase
x or decrease x , Pr(y = 1|x) will increase, and the output class will change.
1 2
Adapted from Prince (2022).
21.2 Intentional misuse
The problems in the previous section arise from poorly specified objectives and infor-
mational asymmetries. However, even when a system functions correctly, it can entail
unethical behavior or be intentionally misused. This section highlights some specific
Problem21.5
ethical concerns arising from the misuse of AI systems.
21.2.1 Face recognition and analysis
Face recognition technologies have an especially high risk for misuse. Authoritarian
states can use them to identify and silence protesters, thus risking democratic ideals
of free speech and the right to protest. Smith & Miller (2022) argue that there is a
mismatch between the values of liberal democracy (e.g., security, privacy, autonomy,
and accountability) and the potential use cases for these technologies (e.g., border se-
curity, criminal investigation and policing, national security, and the commercialization
Draft: please send errata to udlbookmail@gmail.com.

428 21 Deep learning and ethics
of personal data). Thus, some researchers, activists, and policymakers have questioned
whether this technology should exist (Barrett, 2020).
Moreover,thesetechnologiesoftendonotdowhattheypurportto(Rajietal.,2022).
Forexample,theNewYorkMetropolitanTransportationAuthoritymovedforwardwith
and expanded its use of facial recognition despite a proof-of-concept trial reporting a
100%failureratetodetectfaceswithinacceptableparameters(Berger,2019). Similarly,
facialanalysistoolsoftenoverselltheirabilities(Raji&Fried,2020), dubiouslyclaiming
tobeabletoinferindividuals’sexualorientation(Leuner,2019),emotions(Stark&Hoey,
2021), hireability (Fetscherin et al., 2020), or criminality (Wu & Zhang, 2016). Stark
& Hutson (2022) highlight that computer vision systems have created a resurgence in
the“scientificallybaseless,racist,anddiscreditedpseudoscientificfields”ofphysiognomy
and phrenology.
21.2.2 Militarization and political interference
Governments have a vested interest in funding AI research in the name of national
securityandstatebuilding. Thisrisksanarmsracebetweennation-states,whichcarries
with it “high rates of investment, a lack of transparency, mutual suspicion and fear, and
a perceived intent to deploy first” (Sisson et al., 2020).
Lethal autonomous weapons systems receive significant attention because they are
Problem21.6
easytoimagine,andindeedmanysuchsystemsareunderdevelopment(Heikkilä,2022).
However,AIalsofacilitatescyber-attacksanddisinformationcampaigns(i.e.,inaccurate
ormisleadinginformationthatissharedwiththeintenttodeceive). AIsystemsallowthe
creation of highly realistic fake content and facilitate the dissemination of information,
oftentotargetedaudiences(Akersetal.,2018)andatscale(Bontridder&Poullet,2021).
Kosinski et al. (2013) suggest that sensitive variables, including sexual orientation,
ethnicity, religious and political views, personality traits, intelligence, happiness, use of
addictive substances, parental separation, age, and gender, can be predicted by “likes”
on social media alone. From this information, personality traits like “openness” can be
used for manipulative purposes (e.g., to change voting behavior).
21.2.3 Fraud
Unfortunately,AIisausefultoolforautomatingfraudulentactivities(e.g.,sendingmass
emails or text messages that trick people into revealing sensitive information or sending
money). Generative AI can be used to deceive people into thinking they are interacting
with a legitimate entity or generate fake documents that mislead or deceive people.
Additionally,AIcouldincreasethesophisticationofcyber-attacks,suchasbygenerating
more convincing phishing emails or adapting to the defenses of targeted organizations.
This highlights the downside of calls for transparency in machine learning systems:
the more open and transparent these systems are, the more vulnerable they may be to
security risks or use by bad-faith actors. For example, generative language models, like
Problem21.7
ChatGPT,havebeenusedtowritesoftwareandemailsthatcouldbeusedforespionage,
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

21.3 Other social, ethical, and professional issues 429
ransomware, and other malware (Goodin, 2023).
The tendency to anthropomorphize computer behaviors and particularly the projec-
tion of meaning onto strings of symbols is termed the ELIZA effect (Hofstadter, 1995).
Thisleadstoafalsesenseofsecuritywheninteractingwithsophisticatedchatbots,mak-
ingpeoplemoresusceptibletotext-basedfraudsuchasromancescamsorbusinessemail
compromise schemes (Abrahams, 2023). Véliz (2023) highlights how emoji use in some
chatbots is inherently manipulative, exploiting instinctual responses to emotive images.
21.2.4 Data privacy
Modern deep learning methods rely on huge crowd-sourced datasets, which may contain
sensitive or private information. Even when sensitive information is removed, auxiliary
Problem21.8
knowledge and redundant encodings can be used to de-anonymize datasets (Narayanan
&Shmatikov,2008). Indeed, thisfamouslyhappenedtotheGovernorofMassachusetts,
William Weld, in 1997. After an insurance group released health records that had been
stripped of obvious personal information like patient name and address, an aspiring
graduate student was able to “de-anonymize” which records belonged to Governor Weld
by cross-referencing with public voter rolls.
Hence, privacy-first design is important for ensuring the security of individuals’ in-
formation, especially when applying deep learning techniques to high-risk areas such
as healthcare and finance. Differential privacy and semantic security (homomorphic en-
cryptionorsecuremulti-partycomputation)methodscanbeusedtoensuredatasecurity
during model training (see Mireshghallah et al., 2020; Boulemtafes et al., 2020).
21.3 Other social, ethical, and professional issues
TheprevioussectionidentifiedareaswhereAIcanbedeliberatelymisused. Thissection
describes other potential side effects of the widespread adoption of AI.
21.3.1 Intellectual property
Intellectualproperty(IP)canbecharacterizedasnon-physicalpropertythatistheprod-
uctoforiginalthought(Moore&Himma,2022). Inpractice,manyAImodelsaretrained
on copyrighted material. Consequently, these models’ deployment can pose legal and
ethical risks and run afoul of intellectual property rights (Henderson et al., 2023).
Sometimes, these issues are explicit. When language models are prompted with
excerpts of copyrighted material, their outputs may include copyrighted text verbatim,
andsimilarissuesapplyinthecontextofimagegenerationindiffusionmodels(Henderson
et al., 2023; Carlini et al., 2022, 2023). Even if the training falls under “fair use,” this
may violate the moral rights of content creators in some cases (Weidinger et al., 2022).
Moresubtly,generativemodels(chapters12,14–18)raisenovelquestionsregardingAI
Draft: please send errata to udlbookmail@gmail.com.

430 21 Deep learning and ethics
and intellectual property. Can the output of a machine learning model (e.g., art, music,
code, text) be copyrighted or patented? Is it morally acceptable or legal to fine-tune a
model on a particular artist’s work to reproduce that artist’s style? IP law is one area
Problem21.9
that highlights how existing legislation was not created with machine learning models
in mind. Although governments and courts may set precedents in the near future, these
questions are still open at the time of writing.
21.3.2 Automation bias and moral deskilling
As society relies more on AI systems, there is an increased risk of automation bias (i.e.,
expectations that the model outputs are correct because they are “objective”). This
leads to the view that quantitative methods are better than qualitative ones. However,
as we shall see in section 21.5, purportedly objective endeavors are rarely value-free.
The sociological concept of deskilling refers to the redundancy and devaluation of
skillsin lightof automation (Braverman, 1974). Forexample, off-loading cognitiveskills
like memory onto technology may cause a decrease in our capacity to remember things.
Analogously, the automation of AI in morally-loaded decision-making may lead to a
decrease in our moral abilities (Vallor, 2015). For example, in the context of war, the
automationofweaponssystemsmayleadtothedehumanizationofvictimsofwar(Asaro,
2012; Heyns, 2017). Similarly, care robots in elderly-, child-, or healthcare settings may
reduce our ability to care for one another (Vallor, 2011).
21.3.3 Environmental impact
Training deep networks requires significant computational power and hence consumes a
largeamountofenergy. Strubelletal.(2019,2020)estimatethattrainingatransformer
model with 213 million parameters emitted around 284 tonnes of CO .2 Luccioni et al.
2
(2022) have provided similar estimates for the emissions produced from training the
BLOOMlanguagemodel. Unfortunately,theincreasingprevalenceofclosed,proprietary
modelsmeansthatweknownothingabouttheirenvironmentalimpacts(Luccioni,2023).
21.3.4 Employment and society
The history of technological innovation is a history of job displacement. In 2018, the
McKinsey Global Institute estimated that AI may increase economic output by approx-
imately US $13 trillion by 2030, primarily from the substitution of labor by automation
(Bughin et al., 2018). Another study from the McKinsey Global Institute suggests that
upto30%oftheglobalworkforce(10-800millionpeople)couldhavetheirjobsdisplaced
due to AI between 2016 and 2030 (Manyika et al., 2017; Manyika & Sneader, 2018).
2As a baseline, it is estimated that the average human is responsible for around 5 tonnes of CO2
per year, with individuals from major oil-producing countries responsible for three times this amount.
Seehttps://ourworldindata.org/co2-emissions.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

21.4 Case study 431
However,forecastingisinherentlydiﬀicult,andalthoughautomationbyAImaylead
Problem21.10
toshort-termjoblosses,theconceptoftechnologicalunemploymenthasbeendescribedas
a “temporary phase of maladjustment” (Keynes, 1931). This is because gains in wealth
can offset gains in productivity by creating increased demand for products and services.
In addition, new technologies can create new types of jobs.
Evenifautomationdoesn’tleadtoanetlossofoverallemploymentinthelongterm,
newsocialprogramsmayberequiredintheshortterm. Therefore,regardlessofwhether
one is optimistic (Brynjolfsson & McAfee, 2016; Danaher, 2019), neutral (Metcalf et al.,
2016;Calo,2018;Frey,2019),orpessimistic(Frey&Osborne,2017)aboutthepossibility
of unemployment in light of AI, it is clear that society will be changed significantly.
21.3.5 Concentration of power
Asdeepnetworksincreaseinsize,thereisacorrespondingincreaseintheamountofdata
and computing power required to train these models. In this regard, smaller companies
and start-ups may not be able to compete with large, established tech companies. This
may give rise to a feedback loop whereby the power and wealth become increasingly
concentrated in the hands of a small number of corporations. A recent study finds
an increasing discrepancy between publications at major AI venues by large tech firms
and “elite” universities versus mid- or lower-tier universities (Ahmed & Wahed, 2016).
In many views, such a concentration of wealth and power is incompatible with just
distributions in society (Rawls, 1971).
This has led to calls to democratize AI by making it possible for everyone to create
Problem21.11
such systems (Li, 2018; Knight, 2018; Kratsios, 2019; Riedl, 2020). Such a process
requires making deep learning technologies more widely available and easier to use via
open source and open science so that more people can benefit from them. This reduces
barriers to entry and increases access to AI while cutting down costs, ensuring model
accuracy, and increasing participation and inclusion (Ahmed et al., 2020).
21.4 Case study
We now describe a case study that speaks to many of the issues that we have discussed
in this chapter. In 2018, the popular media reported on a controversial facial analysis
model—dubbed“gaydarAI”(Wang&Kosinski,2018)—withsensationalistheadlineslike
AI Can Tell If You’re Gay: Artificial Intelligence Predicts Sexuality From One Photo
with Startling Accuracy (Ahmed, 2017); A Frightening AI Can Determine Whether a
Person Is Gay With 91 Percent Accuracy (Matsakis, 2017); and Artificial Intelligence
System Can Tell If You’re Gay (Fernandez, 2017).
Thereareanumberofproblemswiththiswork. First,thetrainingdatasetwashighly
biased and unrepresentative, being comprised mostly of Caucasian images. Second,
modelingandvalidationarealsoquestionable, giventhefluidityofgenderandsexuality.
Third, the most obvious use case for such a model is the targeted discrimination and
Draft: please send errata to udlbookmail@gmail.com.