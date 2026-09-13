# Chapter 12: (transformers) and chapter 17 (variational autoencoders) were first published

*Pages: 14-14*

---

xii Contents
Tom Jacobs, Lei Fang, Fabian Henning, Umesh Rajashekar, Jay Park, Kai Liu, Pablo Renard
Guiral, Federico Barbero, Rongjiang Pan, Betin Bilkan Karaman, Leonidas Varveropoulos,
William Locke IV, Filip Jasionek, Yuanhang Wang, Stefan Bach, Ivan Yevtushenko, David
Gwyer, Bohan Cui, Ali Darijani, Rouhollah Farhang, Li Tang, Aleksandrs Koselevs, Mason
Wang, Pablo Fernandez, Angelo Coluccia, Vladyslav Moroshan, Rami Luisto, Peter Zaki, Lu-
case Curtin, Victor Liu, Giacomo Cirò, Louis Neltner, Ahmet Çeşmeci, Yanzhe Bekkemoen,
Judith Katzy, and Jannes Bruns.
I’mparticularlygratefultoDaniyarTurmukhambetov,AmedeoBuonanno,AndreaPanizza,
Mark Hudson, Bernhard Pfahringer, Alexander Nordin, and Nicholas Lord who provided de-
tailedcommentsonmultiplechaptersofthebook. I’dliketoespeciallythankAndrewFitzgib-
bon, Konstantinos Derpanis, Toshiaki Kurokawa, and Tyler Mills, who read the whole book
and whose enthusiasm helped me complete this project. I’d also like to thank Neill Campbell
andÖzgürŞimşek,whohostedmeattheUniversityofBath,whereItaughtacoursebasedon
this material for the first time. Finally, I’m extremely grateful to my editor Elizabeth Swayze
for her frank advice throughout this process.
Chapter 12 (transformers) and chapter 17 (variational autoencoders) were first published
as blogs for Borealis AI, and adapted versions are reproduced with permission of Royal Bank
ofCanadaalongwithBorealisAI.Iamgratefulfortheirsupportinthisendeavor. Chapter16
(normalizing flows) is loosely based on the review article by Kobyzev et al. (2020), on which
I was a co-author. I was very fortunate to be able to collaborate on Chapter 21 with Travis
LaCroixfromDalhousieUniversity, who wasbotheasyand funtoworkwith, and whodidthe
lion’s share of the work.
Attribution
• Chessboardimageinfigure1.13adaptedfromhttp://tinyurl.com/yc2d54d4.
• Cogsimageinfigures1.2,1.4,1.10adaptedfromhttp://tinyurl.com/2c7tttr8.
• Penguinimageinfigures19.1–19.5and19.6–19.9adaptedfromhttp://tinyurl.com/ycx9je56.
• Fishimageinfigures19.2–19.5,19.7,19.10–19.12adaptedfromhttp://tinyurl.com/4ueyhtsu.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.