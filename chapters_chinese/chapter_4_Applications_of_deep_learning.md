# 第4章：深度学习的应用

*第239–245页*

---

12.7 解码器模型示例：GPT-3  
该模型包含 225 层，其值（value）、查询（query）和键（key）向量的维度均为 128。它使用 3000 亿个词元（tokens）进行训练，总参数量达 1750 亿。

以下是 GPT-3 模型完成文本的一个示例：输入给模型的提示文本以橙色显示，模型生成的续写内容以青色显示：

> Understanding Deep Learning is a new textbook from MIT Press by Simon Prince that’s de-  
> signed to offer an accessible, broad introduction to the field. Deep learning is a branch of  
> machine learning that is concerned with algorithms that learn from data that is unstructured  
> or unlabeled. The book is divided into four sections:  
> 1. Introduction to deep learning  
> 2. Deep learning architecture  
> 3. Deep learning algorithms  
> 4. Applications of deep learning  
> The first section offers an introduction to deep learning, including its history and origins.  
> The second section covers deep learning architecture, discussing various types of neural net-  
> works and their applications. The third section dives into deep learning algorithms, including  
> supervised and unsupervised learning, reinforcement learning, and more. The fourth section  
> applies deep learning to various domains, such as computer vision, natural language processing,  
> and robotics.

这一续写内容虽看似合理，但仅部分准确。

此类超大规模学习模型一个令人惊讶的特性在于：它们无需微调（fine-tuning），即可执行多种任务。若向模型提供若干组正确的“问题/答案”示例，再给出一个新问题，模型往往能通过序列补全的方式正确回答该最终问题。一个典型示例是英语语法修正任务：

> Poor English input: I eated the purple berries.  
> Good English output: I ate the purple berries.  
>   
> Poor English input: Thank you for picking me as your designer. I’d appreciate it.  
> Good English output: Thank you for choosing me as your designer. I appreciate it.  
>   
> Poor English input: The mentioned changes have done. or I did the alteration that you  
> requested. or I changed things you wanted and did the modifications.  
> Good English output: The requested changes have been made. or I made the alteration that  
> you requested. or I changed things you wanted and made the modifications.  
>   
> Poor English input: I’d be more than happy to work with you in another project.  
> Good English output: I’d be more than happy to work with you on another project.  
> （引自 Brown 等，2020）

此处，橙色标注的成对示例文本作为上下文提供给 GPT-3，系统随后生成了青色标注的正确答案。这一现象广泛存在于多种场景中，包括：依据自然语言描述生成代码片段、算术运算、跨语言翻译，以及针对文本段落回答问题等。因此，有观点认为，超大规模语言模型属于“少样本学习者”（few-shot learners）——仅凭少量示例即可学会执行全新任务。然而在实际应用中，其性能表现并不稳定；此外，目前尚不明确模型究竟是基于已学示例进行了真正的外推（extrapolation），还是仅仅在已有样本间插值（interpolation），甚至只是逐字复制（verbatim copying）。

勘误请发送至：udlbookmail@gmail.com。

226 12 变换器（Transformers）  
图 12.13 编码器–解码器架构。向系统输入两个句子，目标是将第一个句子翻译为第二个句子。  
a) 第一个句子经由标准编码器处理；  
b) 第二个句子则送入解码器，该解码器采用掩码自注意力机制，同时还通过交叉注意力（图中橙色矩形所示）关注编码器输出的嵌入表示。损失函数与纯解码器模型相同：即最大化输出序列中下一个词的概率。

12.8 编码器–解码器模型示例：机器翻译  
语言之间的翻译属于典型的序列到序列（sequence-to-sequence）任务。一种常用方法同时使用编码器（用于计算源语句的高质量表征）和解码器（用于生成目标语言的语句），这种结构恰当地被称为**编码器–解码器模型**。  

以英译法为例：编码器接收英文句子，并通过若干层变换器结构对其进行处理，从而为每个词元（token）生成对应的输出表征。在训练过程中，解码器接收法语的标准参考译文（ground truth translation），并将其送入一系列采用掩码自注意力机制的变换器层，逐位置预测后续词元。然而，解码器各层还会通过交叉注意力机制关注编码器的输出。因此，每个法语输出词元的生成不仅依赖于此前已生成的法语词元，还受到原始英文句子整体语义信息的引导。  

本作品受知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议约束。（©）麻省理工学院出版社。

12.9 面向长序列的 Transformer　227  
图 12.14　交叉注意力（Cross-Attention）。其计算流程与标准自注意力相同，但查询（queries）由解码器嵌入 $ \mathbf{X}_{\text{dec}} $ 计算得出，而键（keys）和值（values）则由编码器嵌入 $ \mathbf{X}_{\text{enc}} $ 计算得出。在机器翻译任务中，编码器包含源语言的统计信息，解码器则包含目标语言的统计信息。  
该模型以先前生成的词以及源端英文句子为条件（见图 12.13）。其实现方式是对解码器中的 Transformer 层进行修改：原始结构由一个带掩码的自注意力层后接一个逐嵌入（per-embedding）应用的神经网络构成（见图 12.12）；现于这两者之间新增一层自注意力机制，使解码器嵌入能够关注编码器嵌入。该机制采用一种称为编码器–解码器注意力（encoder-decoder attention）或交叉注意力（cross-attention）的自注意力变体，其中查询由解码器嵌入计算，而键与值则由编码器嵌入计算（见图 12.14）。  

12.9 面向长序列的 Transformer  
由于 Transformer 编码器模型中每个 token 均需与其余所有 token 相互作用，其计算复杂度随序列长度呈二次方增长。对于解码器模型，每个 token 仅与此前的 token 相互作用，因此交互总数约为编码器的一半，但其复杂度仍为二次方级。这些关系可借助交互矩阵加以可视化（见图 12.15a–b）。  
计算量的这种二次方增长最终限制了可处理序列的最大长度。为此，研究者已提出多种方法以扩展 Transformer 的上下文长度能力。  
草稿：勘误请发送至 udlbookmail@gmail.com。

228 12 变压器（Transformer）

图12.15　自注意力机制中的交互矩阵。  
a) 在编码器中，每个词元（token）均与其他所有词元相互作用，计算复杂度随词元数量呈二次方增长。  
b) 在解码器中，每个词元仅与此前的所有词元相互作用，但复杂度仍为二次方。  
c) 可通过采用卷积结构（编码器情形）来降低复杂度。  
d) 解码器情形下的卷积结构。  
e–f) 解码器情形下、膨胀率（dilation rate）分别为2和3的卷积结构。  
g) 另一种策略是：允许部分选定的词元与所有其他词元（编码器情形）或所有此前词元（图中所示的解码器情形）相互作用。  
h) 或者，可引入全局词元（global token）（对应左侧两列及顶部两行）。这些全局词元不仅与所有其他词元相互作用，彼此之间也相互作用。

前者旨在应对更长的序列。一种可行方法是对自注意力交互进行剪枝，等价于对交互矩阵进行稀疏化处理（见图12.15c–h）。例如，可将交互限制为卷积结构，使得每个词元仅与若干邻近词元发生交互；在多层堆叠下，随着感受野（receptive field）不断扩展，词元之间仍可在更大距离上实现交互。与图像中的卷积类似，该卷积核的尺寸及膨胀率均可变化。

纯卷积方法需堆叠大量层才能实现远距离信息整合。一种加速该过程的方式是：允许部分选定的词元（例如每句开头的词元）关注全部其他词元（编码器模型）或全部此前词元（解码器模型）。类似地，也可引入少量全局词元，使其既连接所有其他词元，又彼此相连。这类全局词元与 `<cls>` 词元类似，并不表征任何具体词汇，而专用于提供长程连接能力。

12.10 图像领域的Transformer  
Transformer最初是为文本数据设计的。其在该领域取得的巨大成功，促使研究者开始将其应用于图像任务。然而，这一思路从直觉上看并非显然可行，原因有二：  

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（©）麻省理工学院出版社。

12.10 用于图像的 Transformer　229  
原因有二。首先，一幅图像中的像素数量远多于一个句子中的单词数量，因此自注意力机制的二次时间复杂度构成了实际应用中的性能瓶颈；其次，卷积网络具有良好的归纳偏置（inductive bias），因为每一层均对空间平移保持等变性（equivariant），并天然地建模了图像的二维结构；而在 Transformer 网络中，这种二维结构关系必须通过学习获得。  

尽管存在上述明显劣势，面向图像的 Transformer 网络如今已在图像分类及其他任务上全面超越卷积网络的性能表现。这在一定程度上得益于其可构建的超大规模，以及可用于预训练的大规模数据集。本节将介绍面向图像的 Transformer 模型。  

12.10.1 ImageGPT  
ImageGPT 是一种 Transformer 解码器，它构建了一个基于图像像素的自回归模型：以部分图像为输入，预测下一个像素值。由于 Transformer 网络具有二次时间复杂度，其最大规模模型（含 68 亿参数）仍仅能处理 $64 \times 64$ 分辨率的图像。此外，为使计算可行，原始的 24 位 RGB 颜色空间被量化为 9 位颜色空间，因此系统在每个位置接收（并预测）512 个可能 token 中的一个。  

图像本质上是二维对象，但 ImageGPT 仅为每个像素单独学习一个位置编码。因此，模型必须自行学习如下关系：每个像素不仅与其前序邻近像素密切相关，也与上方行中邻近像素存在强关联。图 12.16 展示了若干生成结果示例。  

该解码器的内部表征被用作图像分类任务的基础：对每个像素的最终嵌入向量取平均，再经由一个线性层映射为激活值，最后送入 Softmax 层以输出类别概率。该系统首先在海量网络图像语料库上进行预训练，随后在调整为 $48 \times 48$ 像素尺寸的 ImageNet 数据集上进行微调；微调所用损失函数同时包含图像分类任务的交叉熵项和像素预测任务的生成损失项。尽管使用了大量外部训练数据，该系统在 ImageNet 上仍仅取得 27.4% 的 Top-1 错误率（见图 10.15）。这一结果虽逊于当时主流的卷积架构（参见图 10.21），但考虑到其极小的输入图像尺寸（$48 \times 48$），该性能依然十分可观；不出所料，当目标物体尺寸过小或形态细长时，模型性能显著下降。  

12.10.2 视觉 Transformer（ViT）  
视觉 Transformer（Vision Transformer, ViT）通过将图像划分为 $16 \times 16$ 的图像块（patch）来应对图像分辨率问题（见图 12.17）。每个图像块经由一个可学习的线性变换映射为输入嵌入向量，这些嵌入向量随后被馈入 Transformer 网络。同样地，模型采用标准的一维位置编码（positional encoding），并对其进行端到端学习。  

ViT 是一种编码器模型，并引入了 `<cls>` 特殊标记（参见图 12.10–12.11）。然而，与 BERT 不同，ViT 在一个包含 3.03 亿张带标签图像、涵盖 18,000 个类别的大型数据库上进行监督式预训练。`<cls>` 标记经由网络最后一层映射为激活向量，再输入 Softmax 函数以生成类别概率分布。预训练完成后，系统通过替换该 `<cls>` 头部（即最后的分类层）即可直接应用于下游图像分类任务。  

问题 12.8  

草稿：勘误请发送至 udlbookmail@gmail.com。

230 12 变压器（Transformer）

图12.16 ImageGPT。  
a）由自回归式ImageGPT模型生成的图像。左上角像素从该位置处估计的经验分布中采样得到；随后各像素依次生成，每一步均以此前已生成的像素为条件，按行扫描顺序逐像素进行，直至抵达图像右下角。对每个像素，Transformer解码器均生成一个如公式（12.15）所示的条件分布，并从中采样；所得扩展序列随即被反馈至网络，用于生成下一个像素，依此类推。  
b）图像补全任务。在每种情况下，原始图像的下半部分均被移除（见顶行），ImageGPT则逐像素地补全剩余区域（图中展示了三种不同的补全结果）。  
资料来源：https://openai.com/blog/image-gpt/  

将最终层替换为映射至目标类别数的新层，并对其进行微调。在ImageNet基准测试中，该系统实现了11.45%的top-1错误率。然而，其性能仍不及当时未经监督预训练的最佳卷积神经网络。卷积网络所具有的强归纳偏置，唯有依靠海量训练数据才可能被超越。

12.10.3 多尺度视觉Transformer（Multi-scale Vision Transformers）  
视觉Transformer（ViT）与卷积架构的关键区别在于：它仅在单一尺度上操作，且其感受野覆盖整幅图像。目前已提出若干种可在多个尺度上处理图像的Transformer模型。与卷积网络类似，这类模型通常起始于高分辨率、小尺寸的图像块（patch），且通道数较少；随后逐步扩大感受野、降低空间分辨率，并  

本作品遵循知识共享署名-非商业性使用-禁止演绎（CC-BY-NC-ND）许可协议。（C）麻省理工学院出版社。

12.10 用于图像的 Transformer　231  
图 12.17　视觉 Transformer（Vision Transformer，ViT）。视觉 Transformer（ViT）将输入图像划分为若干图像块（patch）组成的网格（原始实现中为 $16 \times 16$ 像素）。每个图像块经由一个可学习的线性变换投影为一个块嵌入（patch embedding）。这些块嵌入被送入 Transformer 编码器网络，而 `<cls>` 标记（token）则用于预测各类别的概率。  

图 12.18　移位窗口（Shifted Window，SWin）Transformer（Liu 等，2021c）。  
a) 原始图像；  
b) SWin Transformer 首先将图像划分为若干窗口（window）构成的网格，再将每个窗口进一步划分为子网格形式的图像块。Transformer 网络在每个窗口内部独立地对图像块执行自注意力机制（即：每个图像块仅关注同一窗口内的其他图像块）；  
c) 每隔一层，窗口的位置发生平移，从而改变彼此交互的图像块子集，使信息得以在整个图像范围内传播；  
d) 经过若干层后，将 $2 \times 2$ 的图像块表征块进行拼接，以增大有效图像块（及窗口）尺寸；  
e) 在该新降低的分辨率下，交替层继续采用移位窗口机制；  
f) 最终，分辨率降低至整幅图像仅对应单个窗口，此时图像块覆盖整个图像。  

勘误请发送至：udlbookmail@gmail.com。