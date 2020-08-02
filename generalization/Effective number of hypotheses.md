Effective number of hypotheses

回顾上篇，我们说一个Hypotheses set（模型集合）的大小，决定了能否机器学习，即能否找到一个模型，使得它在样本外的数据中的错误率$E_{out}(h)$很低。在上篇中我们提到了模型集合中，可选择的模型个数M的大小时算法能否做到样本外错误率$E_{out}(h)$很低的关键，具体而言
$$
P\left[\left|E_{i n}(g)-E_{o u t}(g)\right|>\epsilon\right] \leq 2 \cdot M \cdot \exp \left(-2 \epsilon^{2} N\right) \tag{1}
$$
我们发现，对于PLA这种模型集合中，对于线性可分的数据集而言，拥有无数条分割线，即$M=\infin$，则表明没有理论保证$E_{in}\approx E_{out}$，不等式右边无限大。事实上，通过概率union的方式是扩大了逃逸概率的上界。

<img src="https://i.loli.net/2020/07/28/tmXYOSgfkrueqL6.png" alt="image-20200728213926382" style="zoom: 50%;" />

例如图中所示，每个圈表示一个hypothesis在Hoeffding不等式保证下，样本内外错误率的差别的上界。我们发现，大部分的hypothesis的上界是有重叠区域的，如果我们采用union的方式，则表示这三个圈的面积相加，会导致过分估计了整个hypotheses set的样本内外错误率差别的上界。

这个消息提升了我们对$M=\infin$的情况下，同样能够保证$E_{in}\approx E_{out}$的信心，即是一个有限大的上界。那么该如何去计算这个上界？我们可以对整个hypotheses set中的无限个hypothesis进行分类，该分类方法取决于数据集的分布，下面给出具体例子。

为了简单起见，我们考虑二维的情况，在一个平面上，利用直线将数据集中的点进行类别划分。当数据集中只有1个点时，hypotheses set中有将该点分为正例的hypothesis（如图中的$h_1$和$h_2$，图中箭头的方向表示正例，即$h_1$认为$x_1$为正），也有分为负例的，也就是有两种情况

<img src="https://i.loli.net/2020/07/28/Mvj1AoD68n7WThV.png" alt="image-20200728220419809" style="zoom:50%;" />

当平面上有两个点时，我们一共可以得到以下几种情况

<img src="https://i.loli.net/2020/07/28/yNgwLE1bQFGJjUO.png" alt="image-20200728220419809" style="zoom:50%;" />

如果平面上有三个点，则一条直线可以将它们分为以下8种情况

![image-20200729220056189](https://i.loli.net/2020/07/29/WpsKlH9VAPtGNMu.png)

不过，当平面上的三个点排放成一条直线时，则可以被划分的情况就少了两种，不足8种了。少的两种如图所示，排成直线的中间的点与两边的点不同的情况是无法生成的。

![](https://i.loli.net/2020/07/30/LVbDqoasnX79HJf.png)

总结起来，当平面有三个点时，如果用直线来分类，可以得到小于等于8种类别分布。

我们进一步观察，当平面上有四个点时，理论上有$2^4=16$种组合，但是如果用直线去分类，我们最多能够得到14种分类，如图所示（图中画了前8种组合，将图中⭕️和✖️反转，即可得到后8种组合），我们发现在16种组合中，只能形成$14<2^4$种分类，我们称之为有效分类数$effective(N)\leq 2^N$，或者叫有效直线的个数，这是关于平面上的样本点$N$的函数。

![](https://i.loli.net/2020/07/30/tMrZoHxE7OpTiGD.png)

回顾上面的式$(1)$中的$M$，表示hypotheses set中可选择的hypothesis的个数，如果我们可以利用$effective(N)$代替式$(1)$中的$M$，如果能够保证$effective(N)\ll 2^N$，则即使$M$等于无穷，有效的直线也是有限的，那么就可以保证不等式右边是有限的，$E_{in}$的逃逸概率有限。
$$
P\left[\left|E_{i n}(g)-E_{o u t}(g)\right|>\epsilon\right] \leq 2 \cdot effective(N) \cdot \exp \left(-2 \epsilon^{2} N\right) \tag{2}
$$
二分类Dichotomy

我们定义，利用直线将平面上的点分割而成的某种类别分布为一个二分类，根据上文讨论我们发现二分类的个数上限是$2^N$（根据上文的讨论，当用直线去分类时，当点数大于等于4时，则二分类的个数无法达到16），由于二分类Dichotomy的大小和Hypotheses set $\mathcal{H}$与数据集的点数$N$有关，所以可以定义二分类为$\mathcal{H}(x1,x2,\dots,x_N)$。

我们对比Hypotheses set $\mathcal{H}$与$\mathcal{H}(x1,x2,\dots,x_N)$ 这两个集合，前者的元素是所有直线，样本空间几乎是无限大，后者的元素是所有排列组合，样本空间不超过$2^N$。

我们在上文的讨论中发现，当平面上有3个点时，dichotomy的数量与点的摆放位置有关，当排成一条线时，3个点只有6种dichotomy。基于此，我们需要一个与数据集的点摆放位置无关，仅于点的个数$N$有关的函数时，我们需要定义成长函数$m_{\mathcal{H}}(N)$，表示dochotomy最大的数量。因此$m_{\mathcal{H}}(3)=8$。成长函数$m_{\mathcal{H}}(N)$的上界和dochotomy数量的上界一样是$2^N$。

我们可以讨论一下成长函数在不同的情况下的取值。

![1](https://i.loli.net/2020/08/02/IPO6gTELcBCYk5N.png)

