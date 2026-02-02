# Chapter 4: Socialandethicalproblemsdon’tnecessarilyrequiretechnicalsolutions.

*Pages: 449-476*

---

21.8 Summary 435
ting. However, this luxury is dwindling due to the vast economic incentives to
commercialize AI and the degree to which academic work is funded by industry
(see Abdalla & Abdalla, 2021); even theoretical studies may have social impacts,
so researchers must engage with the social and ethical dimensions of their work.
2. Even purely technical decisions can be value-laden. There is still a widely-
held view that AI is fundamentally just mathematics and, therefore, it is “objec-
tive,” and ethics are irrelevant. This assumption is not true when we consider the
creation of AI systems or their deployment.
3. We should question the structures within which AI work takes place.
Much research on AI ethics focuses on specific situations rather than questioning
the larger social structures within which AI will be deployed. For example, there
is considerable interest in ensuring algorithmic fairness, but it may not always be
possibletoinstantiateconceptionsoffairness,justice,orequitywithinextantsocial
and political structures. Therefore, technology is inherently political.
4. Socialandethicalproblemsdon’tnecessarilyrequiretechnicalsolutions.
Many potential ethical problems surrounding AI technologies are primarily social
and structural, so technical innovation alone cannot solve these problems; if scien-
tists are to effect positive change with new technology, they must take a political
Problem21.13
and moral position.
Where does this leave the average scientist? Perhaps with the following imperative:
itisnecessarytoreflectuponthemoralandsocialdimensionsofone’swork. Thismight
require actively engaging those communities that are likely to be most affected by new
technologies,thuscultivatingrelationshipsbetweenresearchersandcommunitiesandem-
powering those communities. Likewise, it might involve engagement with the literature
beyond one’s own discipline. For philosophical questions, the Stanford Encyclopedia of
Philosophy is an invaluable resource. Interdisciplinary conferences are also useful in this
regard. Leading work is published at both the Conference on Fairness, Accountability,
and Transparency (FAccT) and the Conference on AI, Ethics, and Society (AIES).
21.8 Summary
This chapter considered the ethical implications of deep learning and AI. The value
alignment problem is the task of ensuring that the objectives of AI systems are aligned
withhumanobjectives. Bias,explainability,artificialmoralagency,andothertopicscan
be viewed through this lens. AI can be intentionally misused, and this chapter detailed
some ways this can happen. Progress in AI has further implications in areas as diverse
as IP law and climate change.
Ethical AI is a collective action problem, and the chapter concludes with an appeal
to scientists to consider the moral and ethical implications of their work. Every ethical
issueisnotwithinthecontrolofeveryindividualcomputerscientist. However, thisdoes
not imply that researchers have no responsibility whatsoever to consider—and mitigate
where they can—the potential for misuse of the systems they create.
Draft: please send errata to udlbookmail@gmail.com.

436 21 Deep learning and ethics
Problems
Problem 21.1 It was suggested that the most common specification of the value alignment
problem for AI is “the problem of ensuring that the values of AI systems are aligned with the
valuesofhumanity.” Discussthewaysinwhichthisstatementoftheproblemisunderspecified.
Discussion Resource: LaCroix (2025).
Problem 21.2 Goodhart’s law states that “when a measure becomes a target, it ceases to be a
good measure.” Consider how this law might be reformulated to apply to value alignment for
artificial intelligence, given that the loss function is a mere proxy for our true objectives.
Problem21.3 Supposeauniversityusesdatafrompaststudentstobuildmodelsforpredicting
“student success,” where those models can support informed changes in policies and practices.
Considerhowbiasesmightaffecteachofthefourstagesofthedevelopmentanddeploymentof
this model.
Discussion Resource: Fazelpour & Danks (2021).
Problem 21.4 We might think of functional transparency, structural transparency, and run
transparencyasorthogonal. Provideanexampleofhowanincreaseinoneformoftransparency
may not lead to a concomitant increase in another form of transparency.
Discussion Resource: Creel (2020).
Problem 21.5 IfacomputerscientistwritesaresearchpaperonAIorpushescodetoapublic
repository, do you consider them responsible for future misuse of their work?
Problem 21.6 To what extent do you think the militarization of AI is inevitable?
Problem21.7 InlightofthepossiblemisuseofAIhighlightedinsection21.2,makearguments
both for and against the open-source culture of research in deep learning.
Problem21.8 Somehavesuggestedthatpersonaldataisasourceofpowerforthosewhoownit.
Discussthewayspersonaldataisvaluabletocompaniesthatutilizedeeplearningandconsider
the claim that losses to privacy are experienced collectively rather than individually.
Discussion Resource: Véliz (2020).
Problem 21.9 What are the implications of generative AI for the creative industries? How do
you think IP laws should be modified to cope with this new development?
Problem 21.10 A good forecast must (i) be specific enough to know when it is wrong, (ii)
account for possible cognitive biases, and (iii) allow for rationally updating beliefs. Consider
any claim in the recent media about future AI and discuss whether it satisfies these criteria.
Discussion Resource: Tetlock & Gardner (2016).
Problem21.11 SomecriticshavearguedthatcallstodemocratizeAIhavefocusedtooheavilyon
theparticipatoryaspectsofdemocracy,whichcanincreaserisksoferrorsincollectiveperception,
reasoning,andagency,leadingtomorally-badoutcomes. Reflectoneachofthefollowing: What
aspects of AI should be democratized? Why should AI be democratized? How should AI be
democratized?
Discussion Resource: Himmelreich (2022).
Problem 21.12 InMarch2023,theFutureofLifeInstitutepublishedaletter,“PauseGiantAI
Experiments,” in which they called on all AI labs to immediately pause for at least six months
the training of AI systems more powerful than GPT-4. Discuss the motivations of the authors
in writing this letter, the public reaction, and the implications of such a pause. Relate this
episodetotheviewthatAIethicscanbeconsideredacollectiveactionproblem(section21.6).
Discussion Resource: Gebru et al. (2023).
Problem 21.13 Discussthemeritsofthefourpointsinsection21.7. Doyouagreewiththem?
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Appendix A
Notation
This appendix details the notation used in this book. This mostly adheres to standard
conventionsincomputerscience, butdeeplearningisapplicabletomanydifferentareas,
so it is explained in full. In addition, there are several notational conventions that
are unique to this book, including notation for functions and the systematic distinction
between parameters and variables.
Scalars, vectors, matrices, and tensors
Scalarsaredenotedbyeithersmallorcapitallettersa,A,α. Columnvectors(i.e.,1Dar-
raysofnumbers)aredenotedbysmallboldlettersa,ϕ,androwvectorsasthetranspose
of column vectors aT,ϕT. Matrices and tensors (i.e., 2D and ND arrays of numbers,
respectively) are both represented by bold capital letters B,Φ.
Variables and parameters
Variables (usually the inputs and outputs of functions or intermediate calculations) are
always denoted by Roman letters a,b,C. Parameters (which are internal to functions
or probability distributions) are always denoted by Greek letters α,β,Γ. Generic, un-
specifiedparametersaredenotedbyϕ. Thisdistinctionisretainedthroughoutthebook
except for the policy in reinforcement learning, which is denoted by π according to the
usual convention.
Sets
Sets are denoted by curly brackets, so {0,1,2} denotes the numbers 0, 1, and 2. The
notation {0,1,2,...} denotes the set of non-negative integers. Sometimes, we want to
specify a set of variables and {x }I denotes the I variables x ,...x . When it’s not
i i=1 1 I
necessary to specify how many items are in the set, this is shortened to {x }. The
i
notation {x ,y }I denotes the set of I pairs x ,y . The convention for naming sets is
i i i=1 i i
to use calligraphic letters. Notably, B is used to denote the set of indices in a batch at
t
iteration t during training. The number of elements in a set S is denoted by |S|.
ThesetRdenotesthesetofrealnumbers. ThesetR+ denotesthesetofnon-negative
realnumbers. ThenotationRD denotesthesetofD-dimensionalvectorscontainingreal
Draft: please send errata to udlbookmail@gmail.com.

438 A Notation
numbers. The notation RD1 ×D2 denotes the set of matrices of dimension D
1
×D
2
. The
notation RD1 ×D2 ×D3 denotes the set of tensors of size D
1
×D
2
×D
3
and so on.
Thenotation[a,b]denotestherealnumbersfromatob,includingaandbthemselves.
When the square brackets are replaced by round brackets, this means that the adjacent
value is not included in the set. For example, the set (−π,π] denotes the real numbers
from −π to π, but excluding −π.
Membershipofsetsisdenotedbythesymbol∈,sox∈R+ meansthatthevariablex
is a non-negative real number, and the notation Σ∈RD×D denotes that Σ is a matrix
ofsizeD×D. Sometimes,wewanttoworkthrougheachelementofasetsystematically,
and the notation ∀ {1,...,K} means “for all” the integers from 1 to K.
Functions
Functions are expressed as a name, followed by square brackets that contain the argu-
mentsofthefunction. Forexample,log[x]returnsthelogarithmofthevariablex. When
the function returns a vector, it is written in bold and starts with a small letter. For
example, the function y = mlp[x,ϕ] returns a vector y and has vector arguments x
and ϕ. When a function returns a matrix or tensor, it is written in bold and starts with
a capital letter. For example, the function Y = Sa[X,ϕ] returns a matrix Y and has
arguments X and ϕ. When we want to leave the arguments of a function deliberately
ambiguous, we use the bullet symbol (e.g., mlp[•,ϕ]).
Minimizing and maximizing
Some special functions are used repeatedly throughout the text:
• The function min [f[x]] returns the minimum value of the function f[x] over all
x
possible values of the variable x. This notation is often used without specifying
the details of how this minimum might be found.
• The function argmin [f[x]] returns the value of x that minimizes f[x], so if y =
x
argmin [f[x]], then min [f[x]]=f[y].
x x
• The functions max [f[x]] and argmax [f[x]] perform the equivalent operations for
x x
maximizing functions.
Probability distributions
Probability distributions should be written as Pr(x = a), denoting that the random
variable x takes the value of a. However, this notation is cumbersome. Hence, we
usually simplify this and just write Pr(x), where x denotes either the random variable
or the value it takes according to the sense of the equation. The conditional probability
ofy givenxiswrittenasPr(y|x). Thejointprobabilityofy andxiswrittenasPr(y,x).
Thesetwoformscanbecombined,soPr(y|x,ϕ)denotestheprobabilityofthevariabley,
giventhatweknowxandϕ. Similarly,Pr(y,x|ϕ)denotestheprobabilityofvariablesy
and x given that we know ϕ. When we need two probability distributions over the
same variable, we write Pr(x) for the first distribution and q(x) for the second. More
information about probability distributions can be found in appendix C.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

439
Asymptotic notation
Asymptoticnotationisusedtocomparetheamountofworkdonebydifferentalgorithms
asthesizeDoftheinputincreases. Thiscanbedoneinvariousways,butthisbookonly
uses big-O notation, which represents an upper bound on the growth of computation in
an algorithm. A function f[n] is O[g[n]] if there exists a constant c > 0 and integer n
0
such that f[n]<c·g[n] for all n>n .
0
This notation provides a bound on the worst-case running time of an algorithm.
For example, when we say that inversion of a D×D matrix is O[D3], we mean that the
computationwillincreasenofasterthansomeconstanttimesD3 onceDislargeenough.
Thisgivesusanideaofhowfeasibleitistoinvertmatricesofdifferentsizes. IfD =103,
then it may take of the order of 109 operations to invert it.
Miscellaneous
A small dot in a mathematical equation is intended to improve ease of reading and
has no real meaning (or just implies multiplication). For example, α·f[x] is the same
as αf[x] but is easier to read. To avoid ambiguity, dot products are written as aTb (see
appendix B.3.4). A left arrow symbol ← denotes assignment, so x ← x+2 means that
we are adding two to the current value of x.
Draft: please send errata to udlbookmail@gmail.com.

Appendix B
Mathematics
This appendix reviews mathematical concepts that are used in the main text.
B.1 Functions
A function defines a mapping from a set X (e.g., the set of real numbers) to another
set Y. An injection is a one-to-one function where every element in the first set maps to
a unique position in the second set (but there may be elements of the second set that
are not mapped to). A surjection is a function where every element in the second set
receivesamappingfromthefirst(buttheremaybemultipleelementsofthefirstsetthat
are mapped to the same element of the second set). A bijection or bijective mapping is
a function that is both injective and surjective. It provides a one-to-one correspondence
between all members of the two sets. A diffeomorphism is a special case of a bijection
where both the forward and reverse mapping are differentiable.
B.1.1 Lipschitz constant
A function f[z] is Lipschitz continuous if for all z ,z :
1 2
||f[z ]−f[z ]||≤β||z −z ||, (B.1)
1 2 1 2
where β is known as the Lipschitz constant and determines the maximum gradient of
the function (i.e., how fast the function can change) with respect to the distance metric.
If the Lipschitz constant is less than one, the function is a contraction mapping, and we
can use Banach’s theorem to find the inverse for any point (see figure 16.9).
Composingtwofunctionswith Lipschitzconstantsβ and β createsa newLipschitz
1 2
continuous function with a constant that is less than or equal to β β . Adding two
1 2
functions with Lipschitz constants β and β creates a new Lipschitz continuous func-
1 2
tion with a constant that is less than or equal to β +β . The Lipschitz constant of a
1 2
linear transformation f[z]=Az+b with respect to a Euclidean distance measure is the
maximum eigenvalue of A.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

B.1 Functions 441
B.1.2 Convexity
A function is convex if we can draw a straight line between any two points on the
function, and this line always lies above the function. Similarly, a function is concave
if a straight line between any two points always lies below the function. By definition,
convex (concave) functions have at most one minimum (maximum).
AregionofRD isconvexifwecandrawastraightlinebetweenanytwopointsonthe
boundary of the region without intersecting the boundary in another place. Gradient
descent guarantees to find the global minimum of any function that is both convex and
defined on a convex region.
B.1.3 Special functions
The following functions are used in the main text:
• The exponential function y =exp[x] (figure B.1a) maps a real variable x∈R to a
non-negative number y ∈R+ as y =ex.
• The logarithm x = log[y] (figure B.1b) is the inverse of the exponential function
and maps a non-negative number y ∈ R+ to a real variable x ∈ R. Note that all
logarithms in this book are natural (i.e., in base e).
• The gamma function Γ[x] (figure B.1c) is defined as:
Z
∞
Γ[x]=
tx−1e −tdt.
(B.2)
0
This extends the factorial function to continuous values so that Γ[x]=(x−1)! for
x∈{1,2,...}.
• TheDiracdeltafunctionδ[z]hasatotalareaofone,allofwhichisatpositionz=0.
AdatasetwithN elementscanbethoughtofasaprobabilitydistributionconsisting
of a sum of N delta functions centered at each data point x and scaled by 1/N.
i
The delta function is usually drawn as an arrow (e.g., figure 5.12). The delta
function has the key property that:
Z
f[x]δ[x−x ]dx=f[x ]. (B.3)
0 0
B.1.4 Stirling’s formula
Stirling’sformula(figureB.2)approximatesthefactorialfunction(andhencetheGamma
function) using the formula:
√ (cid:16) (cid:17)
x x
x!≈ 2πx . (B.4)
e
Draft: please send errata to udlbookmail@gmail.com.

442 B Mathematics
Figure B.1 Exponential, logarithm, and gamma functions. a) The exponential
function maps a real number to a positive number. It is a convex function. b)
Thelogarithmistheinverseoftheexponentialandmapsapositivenumbertoa
real number. It is a concave function. c) The Gamma function is a continuous
extension of the factorial function so that Γ[x]=(x−1)! for x∈{1,2,...}.
Figure B.2Stirling’sformula. Thefacto-
rial function x! can be approximated by
Stirling’sformulaStir[x]whichisdefined
for every real value.
B.2 Binomial coeﬀicients
(cid:0) (cid:1)
Binomial coeﬀicients are written as n and pronounced as “n choose k.” They are
k
positiveintegersthatrepresentthenumberofwaysofchoosinganunorderedsubsetofk
items from a set of n items without replacement. Binomial coeﬀicients can be computed
using the simple formula:
(cid:18) (cid:19)
n n!
= . (B.5)
k k!(n−k)!
B.2.1 Autocorrelation
The autocorrelation r[τ] of a continuous function f[z] is defined as:
Z
∞
r[τ]= f[t+τ]f[t]dt, (B.6)
−∞
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

B.3 Vector, matrices, and tensors 443
whereτ isthetimelag. Sometimes,thisisnormalizedbyr[0]sothattheautocorrelation
attimelagzeroisone. Theautocorrelationfunctionisameasureofthecorrelationofthe
function with itself as a function of an offset (i.e., the time lag). If a function changes
slowly and predictably, then the autocorrelation function will decrease slowly as the
time lag increases from zero. If the function changes fast and unpredictably, then it will
decrease quickly to zero.
B.3 Vector, matrices, and tensors
In machine learning, a vector x ∈ RD is a one-dimensional array of D numbers, which
we will assume are organized in a column. Similarly, a matrix Y ∈ RD1 ×D2 is a two-
dimensionalarrayofnumberswithD
1
rowsandD
2
columns. Atensorz∈RD1 ×D2...×DN
isanN-dimensionalarrayofnumbers. Confusingly,allthreeofthesequantitiesarestored
in objects known as “tensors” in deep learning APIs such as PyTorch and TensorFlow.
B.3.1 Transpose
The transpose AT ∈RD2 ×D1 of a matrix A∈RD1 ×D2 is formed by reflecting it around
the principal diagonal so that the kth column becomes the kth row and vice-versa. If we
take the transpose of a matrix product AB, then we take the transpose of the original
matrices but reverse the order so that
(AB)T =BTAT. (B.7)
The transpose of a column vector a is a row vector aT and vice-versa.
B.3.2 Vector and matrix norms
For a vector z, the ℓ norm is defined as:
p
!
XD 1/p
||z|| = |z |p , (B.8)
p d
d=1
for real-valued p > 1. When p = 2, this returns the length of the vector, and this
is known as the Euclidean norm. It is this case that is most commonly used in deep
learning, and often the exponent p is omitted, and the Euclidean norm is just written
as ||z||. When p=∞, the operator returns the maximum absolute value in the vector.
Norms can be computed in a similar way for matrices. For example, the ℓ norm of
2
a matrix Z (known as the Frobenius norm) is calculated as:
Draft: please send errata to udlbookmail@gmail.com.

444 B Mathematics
0 1
1/2
XI XJ
||Z|| = @ |z |2A . (B.9)
F ij
i=1j=1
B.3.3 Product of matrices
The product C = AB of two matrices A ∈ RD1 ×D2 and B ∈ RD2 ×D3 is a third ma-
trix C∈RD1 ×D3 where:
XD2
C = A B . (B.10)
ij id dj
d=1
B.3.4 Dot product of vectors
The dot product aTb of two vectors a∈RD and b∈RD is a scalar and is defined as:
XD
aTb=bTa= a b . (B.11)
d d
d=1
It can be shown that the dot product is proportional to the Euclidean norm of the first
vector times the Euclidean norm of the second vector times the angle θ between them:
aTb=||a||||b|| cos[θ]. (B.12)
B.3.5 Inverse
A square matrix A may or maynot havean inverseA−1 suchthat A−1A=AA−1 =I.
If a matrix does not have an inverse, it is called singular. If we take the inverse of a
matrix product AB where A and B are square and invertible, then we can equivalently
take the inverse of each matrix individually and reverse the order of multiplication.
(AB)
−1
=B
−1A −1.
(B.13)
In general, it takes O[D3] operations to invert a D×D matrix. However, inversion is
moreeﬀicientforspecialtypesofmatrices,includingdiagonal,orthogonal,andtriangular
matrices (see section B.4).
B.3.6 Subspaces
ConsideramatrixA∈RD1 ×D2. IfthenumberofcolumnsD
2
ofthematrixisfewerthan
the number of rows D (i.e., the matrix is “portrait”), the product Ax cannot reach all
1
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

B.3 Vector, matrices, and tensors 445
Figure B.3 Singular values. When the
points {x } on the unit circle are trans-
i
formed to points {x′} by a linear trans-
i
formation x′ = Ax , they are mapped
i i
toanellipse. Forexample,thelightblue
point on the unit circle is mapped to
the light blue point on the ellipse. The
length of the major (longest) axis of the
ellipse (long gray arrow) is the magni-
tudeofthefirstsingularvalueofthema-
trix, and the length of the minor (short-
est)axisoftheellipse(shortgrayarrow)
is the magnitude of the second singular
value.
possible positions in the D -dimensional output space. This product consists of the D
1 2
columns of A weighted by the D elements of x and can only reach the linear subspace
2
that is spanned by these columns. This is known as the column space of the matrix.
Conversely,foralandscapematrixA,thepartoftheinputspacethatmapstozero(i.e.,
those x where Ax=0) is termed the nullspace of the matrix.
B.3.7 Eigenspectrum
If we multiply the set of 2D points on a unit circle by a 2×2 matrix A, they map to
an ellipse (figure B.3). The radii of the major and minor axes of this ellipse (i.e., the
longest and shortest directions) correspond to the magnitude of the singular values λ
1
and λ of the matrix. The same idea applies in higher dimensions. A D−dimensional
2
spheroid is mapped by a D×D matrix A to a D-dimensional ellipsoid. The radii of
theD principalaxesofthisellipsoiddeterminethemagnitudeofthesingularvalues. For
symmetric square matrices, the same information is captured by the eigenvalues which
are here the same as the singular values.
The spectral norm of a square matrix is the largest absolute eigenvalue. It captures
the largest possible change in magnitude when the matrix is applied to a vector of unit
length. Assuch,ittellsusabouttheLipschitzconstantofthetransformation. Thesetof
eigenvaluesissometimescalledtheeigenspectrumandtellsusaboutthemagnitudeofthe
scaling applied by the matrix across all directions. This information can be summarized
using the determinant and trace of the matrix.
B.3.8 Determinant and trace
EverysquarematrixAhasascalarassociatedwithitcalledthedeterminantanddenoted
by |A| or det[A], which is the product of the eigenvalues. It is hence related to the
average scaling applied by the matrix for different inputs. Matrices with small absolute
determinants tend to decrease the norm of vectors upon multiplication. Matrices with
Draft: please send errata to udlbookmail@gmail.com.

446 B Mathematics
large absolute determinants tend to increase the norm. If a matrix is singular, the
determinantwillbezero,andtherewillbeatleastonedirectioninspacethatismapped
to the origin when the matrix is applied. Determinants of matrix expressions obey the
following rules:
|AT| = |A|
|AB| = |A||B|
|A −1| = 1/|A|. (B.14)
Thetraceofasquarematrixisthesumofthediagonalvalues(thematrixitselfneed
not be diagonal) or the sum of the eigenvalues. Traces obey these rules:
trace[AT] = trace[A]
trace[AB] = trace[BA]
trace[A+B] = trace[A]+trace[B]
trace[ABC] = trace[BCA]=trace[CAB], (B.15)
where in the last relation, the trace is invariant for cyclic permutations only, so in
general, trace[ABC]̸=trace[BAC].
B.4 Special types of matrix
CalculatingtheinverseofasquarematrixA∈RD×D hasacomplexityofO[D3],asdoes
thecomputationofthedeterminant. However,forsomematriceswithspecialproperties,
these computations can be more eﬀicient.
B.4.1 Diagonal matrices
A diagonal matrix has zeros everywhere except on the principal diagonal. If these diag-
onal entries are all non-zero, the inverse is also a diagonal matrix, with each diagonal
entry d replaced by 1/d . The determinant is the product of the values on the di-
ii ii
agonal. A special case of this is the identity matrix, which has ones on the diagonal.
Consequently, its inverse is also the identity matrix, and its determinant is one.
B.4.2 Triangular matrices
A lower triangular matrix has all of its non-zero values on the principal diagonal and/or
the positions below this. An upper triangular matrix has all of its non-zero values on
the principal diagonal and/or the positions above this. In both cases, the matrix can
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

B.4 Special types of matrix 447
be inverted in O[D2] (see problem 16.4), and the determinant is just the product of the
values on the diagonal.
B.4.3 Orthogonal matrices
Orthogonalmatricesrepresentrotationsandreflectionsaroundtheorigin,soinfigureB.3,
the circle would be mapped to another circle of unit radius but rotated and possibly
reflected. Accordingly,theeigenvaluesmustallhavemagnitudeone,andthedeterminant
must be either one or minus one. The inverse of an orthogonal matrix is its transpose,
so A−1 =AT.
B.4.4 Permutation matrices
A permutation matrix has exactly one non-zero entry in each row and column, and all
of these entries take the value one. It is a special case of an orthogonal matrix, so its
inverse is its own transpose, and its determinant is always ±1. As the name suggests, it
has the effect of permuting the entries of a vector. For example:
2 32 3 2 3
0 1 0 a b
4 54 5 4 5
0 0 1 b = c . (B.16)
1 0 0 c a
B.4.5 Linear algebra
Linear algebra is the mathematics of linear functions, which have the form:
f[z ,z ,...z ]=ϕ z +ϕ z +...ϕ z , (B.17)
1 2 D 1 1 2 2 D D
where ϕ ,...,ϕ are parameters that define the function. We often add a constant
1 D
term ϕ to the right-hand side. This is technically an aﬀine function but is commonly
0
referred to as linear in machine learning. We adopt this convention throughout.
B.4.6 Linear equations in matrix form
Consider a collection of linear functions:
y = ϕ +ϕ z +ϕ z +ϕ z
1 10 11 1 12 2 13 3
y = ϕ +ϕ z +ϕ z +ϕ z
2 20 21 1 22 2 23 3
y = ϕ +ϕ z +ϕ z +ϕ z . (B.18)
3 30 31 1 32 2 33 3
These can be written in matrix form as:
Draft: please send errata to udlbookmail@gmail.com.

448 B Mathematics
2 3 2 3 2 32 3
y ϕ ϕ ϕ ϕ z
1 10 11 12 13 1
4 5 4 5 4 54 5
y = ϕ + ϕ ϕ ϕ z , (B.19)
2 20 21 22 23 2
y ϕ ϕ ϕ ϕ z
3 30 31 32 33 3
P
or as y=ϕ +Φz for short, where y =ϕ + 3 ϕ z .
0 i i0 j=1 ij j
B.5 Matrix calculus
Most readers of this book will be accustomed to the idea that if we have a function
y =f[x], we can compute the derivative ∂y/∂x, and this represents how y changes when
we make a small change in x. This idea extends to functions y =f[x] mapping a vector
x to a scalar y, functions y=f[x] mapping a vector x to a vector y, functions y=f[X]
mapping a matrix X to a vector y, and so on. The rules of matrix calculus help us
compute derivatives of these quantities. The derivatives take the following forms:
• For a function y = f[x] where y ∈ R and x ∈ RD, the derivative ∂y/∂x is also a
D-dimensional vector, where the ith element is computed as ∂y/∂x .
i
• For a function y = f[x] where y ∈ RDy and x ∈ RDx, the derivative ∂y/∂x is
a D ×D matrix where element (i,j) contains the derivative ∂y /∂x . This is
x y j i
known as a Jacobian and is sometimes written as ∇ y in other documents.
x
• For a function y = f[X] where y ∈ RDy and X ∈ RD1 ×D2, the derivative ∂y/∂X
is a 3D tensor containing the derivatives ∂y /∂x .
i jk
Oftenthesematrixandvectorderivativeshavesuperficiallysimilarformstothescalar
case. For example, we have:
∂y
y =ax −→ =a, (B.20)
∂x
and
∂y
y=Ax −→ =AT. (B.21)
∂x
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

Appendix C
Probability
Probability is critical to deep learning. In supervised learning, deep networks implic-
itly rely on a probabilistic formulation of the loss function. In unsupervised learning,
generative models aim to produce samples that are drawn from the same probability
distributionasthetrainingdata. ReinforcementlearningoccurswithinMarkovdecision
processes, and these are defined in terms of probability distributions. This appendix
provides a primer for probability as used in machine learning.
C.1 Random variables and probability distributions
A random variable x denotes a quantity that is uncertain. It may be discrete (take only
certain values, for example integers) or continuous (take any value on a continuum, for
example real numbers). If we observe several instances of a random variable x, it will
take different values, and the relative propensity to take different values is described by
a probability distribution Pr(x).
Foradiscretevariable,thisdistributionassociatesaprobabilityPr(x=k)∈[0,1]with
each potential outcome k, and the sum of these probabilities is one. For a continuous
variable, there is a non-negative probability density Pr(x=a) ≥ 0 associated with each
value a in the domain of x, and the integral of this probability density function (PDF)
over this domain must be one. This density can be greater than one for any point a.
Fromhereon,weassumethattherandomvariablesarecontinuous. Theideasareexactly
the same for discrete distributions but with sums replacing integrals.
C.1.1 Joint probability
Consider the case where we have two random variables x and y. The joint distribu-
tion Pr(x,y) tells us about the propensity that x and y take particular combinations of
values (figure C.1a). Now there is a non-negative probability density Pr(x=a,y=b)
associated with each pair of values x=a and y =b and this must satisfy:
Draft: please send errata to udlbookmail@gmail.com.

450 C Probability
Figure C.1 Joint and marginal distribu-
tions. a) The joint distribution Pr(x,y)
captures the propensity of variables x
and y to take different combinations of
values. Here, the probability density is
representedbythecolormap,sobrighter
positions are more probable. For exam-
ple, the combination x=6,y=6 is much
less likely to be observed than the com-
bination x=5,y=0. b) The marginal
distribution Pr(x) of variable x can be
recovered by integrating over y. c) The
marginaldistributionPr(y)ofvariabley
can be recovered by integrating over x.
ZZ
Pr(x,y)·dxdy =1. (C.1)
Thisideaextendstomorethantwovariables,sothejointdensityofx,y,andz iswritten
asPr(x,y,z). Sometimes,westoremultiplerandomvariablesinavectorx,andwewrite
their joint density as Pr(x). Extending this, we can write the joint density of all of the
variables in two vectors x and y as Pr(x,y).
C.1.2 Marginalization
IfweknowthejointdistributionPr(x,y)overtwovariables,wecanrecoverthemarginal
distributions Pr(x) and Pr(y) by integrating over the other variable (figure C.1b-c):
Z
Pr(x,y)·dx = Pr(y)
Z
Pr(x,y)·dy = Pr(x). (C.2)
This process is called marginalization and has the interpretation that we are comput-
ing the distribution of one variable regardless of the value the other one took. The
idea of marginalization extends to higher dimensions, so if we have a joint distribu-
tion Pr(x,y,z), we can recover the joint distribution Pr(x,z) by integrating over y.
C.1.3 Conditional probability and likelihood
TheconditionalprobabilityPr(x|y)istheprobabilityofvariablextakingacertainvalue,
assuming we know the value of y. The vertical line is read as the English word “given,”
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

C.1 Random variables and probability distributions 451
FigureC.2Conditionaldistributions. a)JointdistributionPr(x,y)ofvariablesx
and y. b) The conditional probability Pr(x|y = 3.0) of variable x, given that y
takes the value 3.0, is found by taking the horizontal “slice” Pr(x,y=3.0) of
the joint probability (top cyan line in panel a), and dividing this by the total
area Pr(y = 3.0) in that slice so that it forms a valid probability distribution
thatintegratestoone. c)ThejointprobabilityPr(x,y=−1.0)isfoundsimilarly
using the slice at y=−1.0.
so Pr(x|y) is the probability of x given y. The conditional probability Pr(x|y) can be
found by taking a slice through the joint distribution Pr(x,y) for a fixed y. This slice is
then divided by the probability of that value y occurring (the total area under the slice)
so that the conditional distribution sums to one (figure C.2):
Pr(x,y)
Pr(x|y)= . (C.3)
Pr(y)
Similarly,
Pr(x,y)
Pr(y|x)= . (C.4)
Pr(x)
WhenweconsidertheconditionalprobabilityPr(x|y)asafunctionofx,itmustsum
to one. When we consider the same quantity Pr(x|y) as a function of y, it is termed the
likelihood of x given y and does not have to sum to one.
C.1.4 Bayes’ rule
From equations C.3 and C.4, we get two expressions for the joint probability Pr(x,y):
Pr(x,y)=Pr(x|y)Pr(y)=Pr(y|x)Pr(x), (C.5)
which we can rearrange to get:
Draft: please send errata to udlbookmail@gmail.com.

452 C Probability
Figure C.3 Independence. a) When two variables x and y are independent, the
jointdistributionfactorsintotheproductofmarginaldistributions,soPr(x,y)=
Pr(x)Pr(y). Independence implies that knowing the value of one variable tells
us nothing about the other. b–c) Accordingly, all of the conditional distribu-
tionsPr(x|y=•)arethesameandareequaltothemarginaldistributionPr(x).
Pr(y|x)Pr(x)
Pr(x|y)= . (C.6)
Pr(y)
ThisexpressionrelatestheconditionalprobabilityPr(x|y)ofxgivenytotheconditional
probability Pr(y|x) of y given x and is known as Bayes’ rule.
Each term in this Bayes’ rule has a name. The term Pr(y|x) is the likelihood of y
given x, and the term Pr(x) is the prior probability of x. The denominator Pr(y) is
knownastheevidence, andtheleft-handsidePr(x|y)istermedtheposterior probability
of x given y. The equation maps from the prior Pr(x) (what we know about x before
observing y) to the posterior Pr(x|y) (what we know about x after observing y).
C.1.5 Independence
If the value of the random variable y tells us nothing about x and vice-versa, we say
that x and y are independent, and we can write Pr(x|y)=Pr(x) and Pr(y|x)=Pr(y).
It follows that all of the conditional distributions Pr(y|x=•) are identical, as are the
conditional distributions Pr(x|y=•).
Starting from the first expression for the joint probability in equation C.5, we see
that the joint distribution becomes the product of the marginal distributions:
Pr(x,y) = Pr(x|y)Pr(y)=Pr(x)Pr(y) (C.7)
when the variables are independent (figure C.3).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

C.2 Expectation 453
C.2 Expectation
Consider a function f[x] and a probability distribution Pr(x) defined over x. The ex-
pected value of a function f[•] of a random variable x with respect to the probability
distribution Pr(x) is defined as:
Z
(cid:2) (cid:3)
E f[x] = f[x]Pr(x)dx. (C.8)
x
Asthenamesuggests,thisistheexpectedoraveragevalueoff[x]aftertakingintoaccount
the probabilityof seeing different valuesof x. This idea generalizes to functions f[•,•] of
more than one random variable:
ZZ
(cid:2) (cid:3)
E f[x,y] = f[x,y]Pr(x,y)dxdy. (C.9)
x,y
Anexpectationisalwaystakenwithrespecttoadistributionoveroneormorevariables.
However, we don’t usually make this explicit when the choice of distribution is obvious
and write E[f[x]] instead of E [f[x]].
x
If we drew a large number I of samples {x }I from Pr(x), calculated f[x ] for
i i=1 i
each sample and took the average of these values, the result would approximate the
expectation E[f[x]] of the function:
(cid:2) (cid:3) XI
1
E f[x] ≈ f[x ]. (C.10)
x I i
i=1
C.2.1 Rules for manipulating expectations
There are four rules for manipulating expectations:
(cid:2) (cid:3)
E k = k
(cid:2) (cid:3) (cid:2) (cid:3)
E k·f[x] = k·E f[x]
(cid:2) (cid:3) (cid:2) (cid:3) (cid:2) (cid:3)
E f[x]+g[x] = E f[x] +E g[x]
(cid:2) (cid:3) (cid:2) (cid:3) (cid:2) (cid:3)
E f[x]·g[y] = E f[x] ·E g[y] if x,y independent, (C.11)
x,y x y
where k is an arbitrary constant. These are proven below for the continuous case.
Rule 1: The expectation E[k] of a constant value k is just k.
Z
(cid:2) (cid:3)
E k = k·Pr(x)dx
Z
= k· Pr(x)dx
= k. (C.12)
Draft: please send errata to udlbookmail@gmail.com.

454 C Probability
Rule 2: The expectation E[k·f[x]] of a constant k times a function of the variable x
is k times the expectation E[f[x]] of the function:
Z
(cid:2) (cid:3)
E k·f[x] = k·f[x]Pr(x)dx
Z
= k· f[x]Pr(x)dx
(cid:2) (cid:3)
= k·E f[x] . (C.13)
Rule 3: The expectation of a sum E[f[x]+g[x]] of terms is the sum E[f[x]]+E[g[x]] of
the expectations:
Z
(cid:2) (cid:3)
E f[x]+g[x] = (f[x]+g[x])·Pr(x)dx
Z
(cid:0) (cid:1)
= f[x]·Pr(x)+g[x]·Pr(x) dx
Z Z
= f[x]·Pr(x)dx+ g[x]·Pr(x)dx
(cid:2) (cid:3) (cid:2) (cid:3)
= E f[x] +E g[x] . (C.14)
Rule 4: TheexpectationofaproductE[f[x]·g[y]]oftermsistheproductE[f[x]]·E[g[y]]
if x and y are independent.
ZZ
(cid:2) (cid:3)
E f[x]·g[y] = f[x]·g[y]Pr(x,y)dxdy
ZZ
= f[x]·g[y]Pr(x)Pr(y)dxdy
Z Z
= f[x]·Pr(x)dx g[y]·Pr(y)dy
(cid:2) (cid:3) (cid:2) (cid:3)
= E f[x] E g[y] , (C.15)
where we used the definition of independence (equation C.7) between the first two lines.
The four rules generalize to the multivariate case:
(cid:2) (cid:3)
E A = A
(cid:2) (cid:3) (cid:2) (cid:3)
E A·f[x] = AE f[x]
(cid:2) (cid:3) (cid:2) (cid:3) (cid:2) (cid:3)
E f[x]+g[x] = E f[x] +E g[x]
(cid:2) (cid:3) (cid:2) (cid:3) (cid:2) (cid:3)
E f[x]Tg[y] = E f[x] TE g[y] if x,y independent, (C.16)
x,y x y
where now A is a constant matrix and f[x] is a function of the vector x that returns a
vector, and g[y] is a function of the vector y that also returns a vector.
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

C.2 Expectation 455
C.2.2 Mean, variance, and covariance
For some choices of function f[•], the expectation is given a special name. These quan-
tities are often used to summarize the properties of complex distributions. For example,
when f[x] = x, the resulting expectation E[x] is termed the mean, µ. It is a mea-
sure of the center of a distribution. Similarly, the expected squared deviation from the
mean E[(x−µ)2] is termed the variance, σ2. This is a measure of the spread of the
distribution. The standard deviation σ is the positive square root of the variance. It
also measures the spread of the distribution but has the merit that it is expressed in the
same units as the variable x.
As the name suggests, the covariance E[(x−µ )(y−µ )] of two variables x and y
x y
measures the degree to which they co-vary. Here µ and µ represent the mean of the
x y
variables x and y, respectively. The covariance will be large when the variance of both
variablesislargeandwhenthevalueofxtendstoincreasewhenthevalueofy increases.
Iftwovariablesareindependent,thentheircovarianceiszero. However,acovariance
ofzerodoesnotimplyindependence. Forexample,consideradistributionPr(x,y)where
the probability is uniformly distributed on a circle of radius one centered on the origin
of the x,y plane. There is no tendency on average for x to increase when y increases or
vice-versa. However, knowing the value of x = 0 tells us that y has an equal chance of
taking the values ±1, so the variables cannot be independent.
Thecovariancesofmultiplerandomvariablesstoredinacolumnvectorx∈RD canbe
represented by the D×D covariance matrix E[(x−µ )(x−µ )T], where the vector µ
x x x
contains the means E[x]. The element at position (i,j) of this matrix represents the
covariance between variables x and x .
i j
C.2.3 Variance identity
The rules of expectation (appendix C.2.1) can be used to prove the following identity
that allows us to write the variance in a different form:
(cid:2) (cid:3) (cid:2) (cid:3) (cid:2) (cid:3)
E (x−µ)2 =E x2 −E x 2 . (C.17)
Proof:
(cid:2) (cid:3) (cid:2) (cid:3)
E (x−µ)2 = E x2−2µx+µ2
(cid:2) (cid:3) (cid:2) (cid:3) (cid:2) (cid:3)
= E x2 −E 2µx +E µ2
(cid:2) (cid:3) (cid:2) (cid:3)
= E x2 −2µ·E x +µ2
(cid:2) (cid:3)
= E x2 −2µ2+µ2
(cid:2) (cid:3)
= E x2 −µ2
(cid:2) (cid:3) (cid:2) (cid:3)
= E x2 −E x 2 , (C.18)
where we have used rule 3 between lines 1 and 2, rules 1 and 2 between lines 2 and 3,
and the definition µ=E[x] in the remaining two lines.
Draft: please send errata to udlbookmail@gmail.com.

456 C Probability
C.2.4 Standardization
Setting the mean of a random variable to zero and the variance to one is known as
standardization. This is achieved using the transformation:
x−µ
z = , (C.19)
σ
where µ is the mean of x and σ is the standard deviation.
Proof: The mean of the new distribution over z is given by:
(cid:20) (cid:21)
x−µ
E[z] = E
σ
(cid:2) (cid:3)
1
= E x−µ
σ
(cid:0) (cid:2) (cid:3) (cid:2) (cid:3)(cid:1)
1
= E x −E µ
σ
1
= (µ−µ)=0, (C.20)
σ
whereagain, wehaveusedthefourrulesformanipulatingexpectations. Thevarianceof
the new distribution is given by:
(cid:2) (cid:3) (cid:2) (cid:3)
E (z−µ )2 = E (z−E[z])2
z (cid:2) (cid:3)
= E z2
"(cid:18) (cid:19) #
x−µ 2
= E
σ
1
= ·E[(x−µ)2]
σ2
1
= ·σ2 =1. (C.21)
σ2
By a similar argument, we can take a standardized variable z with mean zero and unit
variance and convert it to a variable x with mean µ and variance σ2 using:
x=µ+σz. (C.22)
Inthemultivariatecase,wecanstandardizeavariablexwithmeanµandcovariance
matrix Σ using:
z=Σ
−1/2(x−µ).
(C.23)
The result will have a mean E[z]=0 and an identity covariance matrix E[(z−E[z])(z−
E[z])T]=I. To reverse this process, we use:
x=µ+Σ1/2z. (C.24)
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

C.3 Normal probability distribution 457
C.3 Normal probability distribution
ProbabilitydistributionsusedinthisbookincludetheBernoullidistribution(figure5.6),
categorical distribution (figure 5.9), Poisson distribution (figure 5.15), von Mises distri-
bution (figure 5.13), and mixture of Gaussians (figures 5.14 and 17.1). However, the
most common distribution in machine learning is the normal or Gaussian distribution.
C.3.1 Univariate normal distribution
A univariate normal distribution (figure 5.3) over scalar variable x has two parameters,
the mean µ and the variance σ2, and is defined as:
(cid:20) (cid:21)
1 (x−µ)2
Pr(x)=Norm [µ,σ2]= √ exp − . (C.25)
x 2πσ2 2σ2
Unsurprisingly, the mean E[x] of a normally distributed variable is given by the mean
parameter µ and the variance E[(x−E[x])2] by the variance parameter σ2. When the
mean is zero and the variance is one, we refer to this as a standard normal distribution.
The shape of the normal distribution can be inferred from the following argument.
Theterm−(x−µ)2/2σ2isaquadraticfunctionthatfallsawayfromzerowhenx=µata
ratethatincreaseswhenσ becomessmaller. Whenwepassthisthroughtheexponential
function (figure B.1), we get a bell-shaped curve, whi√ch has a value of one at x = µ
and falls away to either side. Dividing by the constant 2πσ2 ensures that the function
integrates to one and is a valid distribution. It follows from this argument that the
mean µ control the position of the center of the bell curve, and the square root σ of the
variance (the standard deviation) controls the width of the bell curve.
C.3.2 Multivariate normal distribution
The multivariate normal distribution generalizes the normal distribution to describe the
probability over a vector quantity x of length D. It is defined by a D×1 mean vector µ
and a symmetric positive definite D×D covariance matrix Σ:
(cid:20) (cid:21)
1 (x−µ)TΣ −1(x−µ)
Norm [µ,Σ]= exp − . (C.26)
x (2π)D/2|Σ|1/2 2
Theinterpretationissimilartotheunivariatecase. Thequadraticterm−(x−µ)TΣ −1(x−
µ)/2 returns a scalar that decreases as x grows further from the mean µ, at a rate that
dependsonthematrixΣ. Thisisturnedintoabell-curveshapebytheexponential,and
dividing by (2π)D/2|Σ|1/2 ensures that the distribution integrates to one.
The covariance matrix can take spherical, diagonal, and full forms:
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
σ2 0 σ2 0 σ2 σ2
Σ = Σ = 1 Σ = 11 12 . (C.27)
spher 0 σ2 diag 0 σ2 full σ2 σ2
2 21 22
Draft: please send errata to udlbookmail@gmail.com.

458 C Probability
Figure C.4 Bivariate normal distribution. a–b) When the covariance matrix is a
multipleoftheidentitymatrix,theisocontoursarecircles,andwerefertothisas
spherical covariance. c–d) When the covariance is an arbitrary diagonal matrix,
theisocontoursareaxis-alignedellipses,andwerefertothisasdiagonalcovariance
e–f) When the covariance is an arbitrary symmetric positive definite matrix, the
iso-contours are general ellipses, and we refer to this as full covariance.
In two dimensions (figure C.4), spherical covariances produce circular iso-density
contours,anddiagonalcovariancesproduceellipsoidaliso-contoursthatarealignedwith
the coordinate axes. Full covariances produce general ellipsoidal iso-density contours.
When the covariance is spherical or diagonal, the individual variables are independent:
(cid:20) (cid:18) (cid:19)(cid:21)
(cid:0) (cid:1)
Pr(x ,x ) = p 1 exp −0.5 x x Σ −1 x 1
1 2 2π |Σ| 1 2 x 2
(cid:20) (cid:18) (cid:19)(cid:18) (cid:19)(cid:21)
1 (cid:0) (cid:1) σ −2 0 x
= exp −0.5 x x 1 1
2πσ 1 σ 2 (cid:20) (cid:21) 1 2 0 (cid:20) σ 2 −2 (cid:21) x 2
1 x2 1 x2
= p exp − 1 · p exp − 2
2πσ2 2σ2 2πσ2 2σ2
1 1 2 2
= Pr(x )·Pr(x ). (C.28)
1 2
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

C.3 Normal probability distribution 459
Figure C.5 Change of variables. a) The conditional distribution Pr(x|y) is a
normaldistributionwithconstantvarianceandameanthatdependslinearlyony.
Cyan distribution shows one example for y = −0.2. b) This is proportional to
theconditionalprobabilityPr(y|x),whichisanormaldistributionwithconstant
variance and a mean that depends linearly on x. Cyan distribution shows one
example for x=−3.
C.3.3 Product of two normal distributions
The product of two normal distributions is proportional to a third normal distribution
according to the relation:
h i
Norm [a,A]Norm [b,B]∝Norm (A −1+B −1) −1(A −1a+B −1b),(A −1+B −1) −1 .
x x x
(C.29)
Thisiseasilyprovedbymultiplyingouttheexponentialtermsandcompletingthesquare
(see problem 18.5).
C.3.4 Change of variable
When the mean of a multivariate normal in x is a linear function Ay+b of a second
variable y, this is proportional to another normal distribution in y, where the mean is a
linear function of x:
Norm [Ay+b,Σ]∝Norm [(ATΣ −1A) −1ATΣ −1(x−b),(ATΣ −1A) −1]. (C.30)
x y
At first sight, this relation is rather opaque, but figure C.5 shows the case for scalar x
and y, which is easy to understand. As for the previous relation, this can be proved by
expanding the quadratic product in the exponential term and completing the square to
make this a distribution in y. (see problem 18.4).
Draft: please send errata to udlbookmail@gmail.com.

460 C Probability
C.4 Sampling
TosamplefromaunivariatedistributionPr(x),wefirstcomputethecumulativedistribu-
tionF[x](theintegralofPr(x)). Thenwedrawasamplez∗ fromauniformdistribution
over the range [0,1] and evaluate this against the inverse of the cumulative distribution,
so the sample x∗ is created as:
x
∗
=F
−1[z ∗
]. (C.31)
C.4.1 Sampling from normal distributions
Themethodabovecanbeusedtogenerateasamplex∗fromaunivariatestandardnormal
distribution. Asamplefromanormaldistributionwithmeanµandvarianceσ2 canthen
becreatedusingequationC.22. Similarly,asamplex∗fromaD-dimensionalmultivariate
standard distribution can be created by independently sampling D univariate standard
normal variables. A sample from a multivariate normal distribution with mean µ and
covariance Σ can be then created using equation C.24.
C.4.2 Ancestral sampling
When the joint distribution can be factored into a series of conditional probabilities, we
can generate samples using ancestral sampling. The basic idea is to generate a sample
from the root variable(s) and then sample from the subsequent conditional distributions
based on this instantiation. This process is known as ancestral sampling and is easiest
to understand with an example. Consider a joint distribution Pr(x,y,z) over three
variables, x,y, and z, which (in this particular case) factors as:
Pr(x,y,z)=Pr(x)Pr(y|x)Pr(z|y). (C.32)
To sample from this joint distribution, we first draw a sample x∗ from Pr(x). Then we
draw a sample y∗ from Pr(y|x∗). Finally, we draw a sample z∗ from Pr(z|y∗).
C.5 Distances between probability distributions
Supervised learning can be framed in terms of minimizing the distance between the
probability distribution implied by the model and the discrete probability distribution
implied by the samples (section 5.7). Unsupervised learning can often be framed in
terms of minimizing the distance between the probability distribution of real examples
andthedistributionofdatafromthemodel. Inbothcases,weneedameasureofdistance
between two probability distributions. This section considers the properties of several
different measures of distance between distributions (see also figure 15.8 for a discussion
of the Wasserstein or earth mover’s distance).
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.

C.5 Distances between probability distributions 461
Figure C.6Lowerboundonnegativelog-
arithm. The function 1 − y is always
less than the function −log[y]. This re-
lationisusedtoshowthattheKullback-
Leiblerdivergenceisalwaysgreaterthan
or equal to zero.
C.5.1 Kullback-Leibler divergence
The most common measure of distance between probability distributions p(x) and q(x)
is the Kullback-Leibler or KL divergence and is defined as:
Z (cid:20) (cid:21)
(cid:2) (cid:3)
p(x)
D p(x)||q(x) = p(x)log dx. (C.33)
KL q(x)
This distance is always greater than or equal to zero, which is easily demonstrated by
noting that −log[y]≥1−y (figure C.6) so:
h (cid:12)(cid:12) i Z (cid:20) (cid:21)
(cid:12)(cid:12) p(x)
D p(x) q(x) = p(x)log dx
KL q(x)
Z (cid:20) (cid:21)
q(x)
= − p(x)log dx
p(x)
Z (cid:18) (cid:19)
q(x)
≥ p(x) 1− dx
p(x)
Z
= p(x)−q(x)dx
= 1−1=0. (C.34)
The KL divergence is infinite if there are places where q(x) is zero but p(x) is non-zero.
This can lead to problems when we are minimizing a function based on this distance.
C.5.2 Jensen-Shannon divergence
The KL divergence is not symmetric (i.e., D [p(x)||q(x)]̸= D [q(x)||p(x)]). The
KL KL
Jensen-Shannon divergence is a measure of distance that is symmetric by construction:
D h p(x) (cid:12) (cid:12) (cid:12) (cid:12) q(x) i = 1 D (cid:20) p(x) (cid:12) (cid:12) (cid:12) (cid:12) (cid:12) (cid:12) p(x)+q(x) (cid:21) + 1 D (cid:20) q(x) (cid:12) (cid:12) (cid:12) (cid:12) (cid:12) (cid:12) p(x)+q(x) (cid:21) . (C.35)
JS 2 KL 2 2 KL 2
It is the mean divergence of p(x) and q(x) to the average of the two distributions.
Draft: please send errata to udlbookmail@gmail.com.

462 C Probability
C.5.3 Fréchet distance
The Fréchet distance D between two distributions p(x) and q(x) is given by:
Fr
s
h (cid:12)(cid:12) i (cid:20)ZZ (cid:21)
D p(x) (cid:12)(cid:12) q(y) = min π(x,y)|x−y|2dxdy , (C.36)
Fr
π(x,y)
where π(x,y) represents the set of joint distributions that are compatible with the
marginal distributions p(x) and q(y). The Fréchet distance can also be formulated as a
measure of the maximum distance between the cumulative probability curves.
C.5.4 Distances between normal distributions
Often we want to compute the distance between two multivariate normal distributions
with means µ and µ and covariances Σ and Σ . In this case, various measures of
1 2 1 2
distance can be written in closed form.
The KL divergence can be computed as:
h (cid:12)(cid:12) i
(cid:12)(cid:12)
D Norm[µ ,Σ ](cid:12)(cid:12)Norm[µ ,Σ ] = (C.37)
KL 1 1 2 2
(cid:18) (cid:20) (cid:21) (cid:19)
1 |Σ | (cid:2) (cid:3)
log 2 −D+tr Σ −1Σ +(µ −µ )TΣ −1(µ −µ ) ,
2 |Σ | 2 1 2 1 2 2 1
1
where tr[•] is the trace of the matrix argument. The Fréchet/2-Wasserstein distance is
given by:
h (cid:12)(cid:12) i h i
(cid:12)(cid:12)
D2 Norm[µ ,Σ ](cid:12)(cid:12)Norm[µ ,Σ ] =|µ −µ |2+tr Σ +Σ −2(Σ Σ )1/2 .
Fr/W2 1 1 2 2 1 2 1 2 1 2
(C.38)
This work is subject to a Creative Commons CC-BY-NC-ND license. (C) MIT Press.