Hoeffding's Inequality

假设有一个已知的模型h，它在总体数据上的错误率为$\mu$，现在我们有一个从总体中采样出的N个样本，利用这个模型h，去计算这N个样本的标签和样本自身的标签的错误率，表示为$\nu$。根据大数定律，$\nu$可以近似估计$\mu$，当N很大时，$\mu$和$\nu$满足Hoeffding不等式
$$
\mathbb{P}[|\nu-\mu|>\epsilon] \leq 2 \exp \left(-2 \epsilon^{2} N\right)
$$
即$\nu=\mu$是PAC的（probably approximately correct）

$\epsilon$是用户的容忍度。

它想表达的意思是，当我们想考察h在样本上的错误率$\nu$和总体上的错误率$\mu$之间的差别时，我们设置一个范围$\epsilon$，它们的差距要大于$\epsilon$的概率存在上界，由样本数N和容忍度$\epsilon$所决定。当样本数越大，上界越低。容忍度越大，上界越低（但是这是自欺欺人的表现）。这个上界不需要知道总体的参数$\mu$，因此不需要对总体有所假设。

需要注意的是，Hoeffding保证的是，对于一个模型h，可以保证它的错误率$\nu$与f的错误率$\mu$接近。但是当我**可以对模型h进行选择**时，这个选出的h在样本N上的错误率$\nu$则无法保证它在总体数据集上的错误率还很低。

|       | $D_1$   | $D_2$   | ...  | $D_{1126}$ | ...  | $D_{5678}$ | Hoeffding                             |
| ----- | ------- | ------- | ---- | ---------- | ---- | ---------- | ------------------------------------- |
| $h_1$ | BAD     |         |      |            |      | BAD        | $\mathbb{P}_D[\text{BAD D for } h_1]$ |
| $h_2$ |         | BAD     |      |            |      |            | $\mathbb{P}_D[\text{BAD D for } h_2]$ |
| $h_3$ | BAD     | BAD     |      |            |      | BAD        | $\mathbb{P}_D[\text{BAD D for } h_3]$ |
| ...   |         |         |      |            |      |            |                                       |
| $h_M$ | BAD     |         |      |            |      | BAD        | $\mathbb{P}_D[\text{BAD D for } h_M]$ |
| all   | **BAD** | **BAD** |      |            |      | **BAD**    | X                                     |

每个$h$表示一个模型，这个模型在**任意**给定的大小为N的数据集$D_1 \dots D_{5678}$上，如果它的错误率$\nu$与在总体上的错误率$\mu$之间的差距很大，我们称为BAD，出现BAD的概率有Hoeffding不等式保证了其概率的上界。

由于算法是一个训练模型的过程，算法可以在$h_1 \ldots h_{M}$上不停做选择，因此如果给定一个样本集$D_1$，那么算法很有可能最终选择的模型$h$虽然在$D_1$上表现好，但是在总体上表现不好。因此只有当给定的样本集是$D_{1126}$这样优质的数据，算法才可以自由的选择模型，得到的模型在样本上的错误率可以表示总体的错误率。

这里假设了模型集合是有限数量的，为M个。如果一个样本集，在M个模型中的任意一个是有问题的，那我们就认为这个样本集是不好的。

我们可以计算算法任意选择出的模型h，它在样本和总体上的错误率差别很大的概率的上界为
$$
\begin{aligned}
\mathbb{P}_{\mathcal{D}}[\text { BAD } \mathcal{D}]
&=\mathbb{P}_{\mathcal{D}}\left[\mathrm{BAD } \mathcal{D} \text { for } h_{1} \text { or } \mathrm{BAD } \mathcal{D} \text { for } h_{2} \text { or } \ldots \text { or } \mathrm{BAD} \mathcal{D} \text { for } h_{M}\right] \\
&\leq \mathbb{P}_{\mathcal{D}}\left[\mathrm{BAD } \mathcal{D} \text { for } h_{1}\right]+\mathbb{P}_{\mathcal{D}}\left[\mathrm{BAD} \mathcal{D} \text { for } h_{2}\right]+\ldots+\mathbb{P}_{\mathcal{D}}\left[\mathrm{BAD} \mathcal{D} \text { for } h_{M}\right]
\\
&\text{(union bound)} \\
&\leq 2 \exp \left(-2 \epsilon^{2} N\right)+2 \exp \left(-2 \epsilon^{2} N\right)+\ldots+2 \exp \left(-2 \epsilon^{2} N\right)
\\ 
&=2 M \exp \left(-2 \epsilon^{2} N\right)
\end{aligned}
$$
所以当算法可以选择的模型数量有限，为M时，Hoeffding的上界扩大M倍，同时当样本集的数量N足够大时，算法还是可以选择到一个不错的模型h，使得样本内的错误率$\nu$与总体的错误率$\mu$接近。

