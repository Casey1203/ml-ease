Effective number of hypotheses

回顾上篇，我们说一个Hypotheses set（模型集合）的大小，决定了能否机器学习，即能否找到一个模型，使得它在样本外的数据中的错误率$E_{out}(h)$很低。在上篇中我们提到了模型集合中，可选择的模型个数M的大小时算法能否做到样本外错误率$E_{out}(h)$很低的关键，具体而言
$$
P\left[\left|E_{i n}(g)-E_{o u t}(g)\right|>\epsilon\right] \leq 2 \cdot M \cdot \exp \left(-2 \epsilon^{2} N\right)
$$
我们发现，对于PLA这种模型集合中，对于线性可分的数据集而言，拥有无数条分割线，即$M=\infin$，则表明没有理论保证$E_{in}\approx E_{out}$，不等式右边无限大。事实上，通过概率union的方式是扩大了逃逸概率的上界。