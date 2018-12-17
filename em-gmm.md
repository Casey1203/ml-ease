### 极大似然估计
找到与样本的分布最接近的概率分布模型。例如，给定一组样本$X=\{x_1,x_2,\cdots,x_n\}$，已知它们来自于高斯分布$N(\mu,\sigma)$，估计参数$\mu,\sigma$。

方法是：已知高斯分布的概率密度函数为

$$
f(x)=\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$
则$X$的似然函数为
$$
L ( x ) = \prod _ { i = 1 } ^ { N } \frac { 1 } { \sqrt { 2 \pi } \sigma } e ^ { - \frac { \left( x _ { i } - \mu \right) ^ { 2 } } { 2 \sigma ^ { 2 } } }
$$

对似然函数取对数并且进行化简
$$
\begin{array} { l } { l ( x ) = \log \prod _ { i }^n \frac { 1 } { \sqrt { 2 \pi } \sigma } e ^ { - \frac { \left( x _ { i } - \mu \right) ^ { 2 } } { 2 \sigma ^ { 2 } } } } \\
{ = \sum _ { i }^n \log \frac { 1 } { \sqrt { 2 \pi } \sigma } e ^ { - \frac { ( x_i - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } } } \\
{ = \left( \sum _ { i }^n \log \frac { 1 } { \sqrt { 2 \pi } \sigma } \right) + \left( \sum _ { i }^n - \frac { \left( x _ { i } - \mu \right) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) } \\
{ = - \frac { n } { 2 } \log \left( 2 \pi \sigma ^ { 2 } \right) - \frac { 1 } { 2 \sigma ^ { 2 } } \sum _ { i }^n \left( x _ { i } - \mu \right) ^ { 2 } } \end{array}
$$
将对数似然函数$l ( x ) $对参数$\mu$和$\sigma$分别求偏导数，令它们等于0。
$$
\begin{aligned}
&\frac{\partial l}{\partial \mu}=\frac{1}{\sigma^2}\sum_{i=1}^n{(x_i-\mu)}=0 \\
&\frac{\partial l}{\partial \sigma}=-\frac{n}{2}\frac{4\pi \sigma}{2\pi \sigma^2} + \frac{1}{\sigma^3}\sum_{i}^n{(x_i-\mu)^2}=0
\end{aligned}

$$

整理后可以得到
$$
\begin{array} { c } { \mu = \frac { 1 } { n } \sum _ { i } ^n x _ { i } } \\ { \sigma ^ { 2 } = \frac { 1 } { n } \sum _ { i }^n \left( x _ { i } - \mu \right) ^ { 2 } } \end{array}
$$


### 高斯混合模型GMM
随机变量$X$的分布服从由$K$个高斯分布混合而成的分布。定义如下变量。$\pi_1,\pi_2,\cdots,\pi_K$为选取各个高斯分布的概率。第$k$个高斯分布的均值为$\mu_k$，标准差为$\Sigma_k$。
所以$x$的概率分布可以表示为
$$
p(x_i|\theta) = \sum _ { k = 1 } ^ { K } \pi _ { k } N \left( x _ { i } | \mu _ { k } , \Sigma _ { k } \right )
$$
$\theta$表示$x$的概率分布的参数，在这里表示$\pi, \mu, \Sigma$。

给定一组样本$X=\{x_1,x_2,\cdots,x_n\}$，试估计$\pi$,$\mu$,$\Sigma$。如果样本$x_i$是标量，标准差$\Sigma$是一个数，样本$x_i$是向量，标准差$\Sigma$是方阵。

首先定义对数似然函数
$$
\begin{aligned}
l _ { \pi , \mu , \Sigma } ( x ) &= \log{\prod_i^N{\left(\sum _ { k = 1 } ^ { K } \pi _ { k } N \left( x _ { i } | \mu _ { k } , \Sigma _ { k } \right)\right)}}\\
&=\sum _ { i = 1 } ^ { N } \log \left( \sum _ { k = 1 } ^ { K } \pi _ { k } N \left( x _ { i } | \mu _ { k } , \Sigma _ { k } \right) \right)
\end{aligned}
$$
随机初始化参数$\pi$,$\mu$,$\Sigma$，计算第$i$个样本$x_i$是来自于第$k$个高斯分布生成的概率是
$$
\gamma ( i , k ) = \frac { \pi _ { k } N \left( x _ { i } | \mu _ { k } , \Sigma _ { k } \right) } { \sum _ { j = 1 } ^ { K } \pi _ { j } N \left( x _ { i } | \mu _ { j } , \Sigma _ { j } \right) }
$$

对于第$k$个高斯分布，可以看成是生成了$\{\gamma ( 1, k )x_1, \gamma ( 2 , k )x_2,\cdots, \gamma ( n , k )x_n\}$样本集。利用上面极大似然估计法得到的高斯分布的均值和方差，
$$
\left\{ \begin{array} { l } { \mu = \frac { 1 } { n } \sum _ { i } x _ { i } } \\ { \sigma ^ { 2 } = \frac { 1 } { n } \sum _ { i } \left( x _ { i } - \mu \right) ^ { 2 } } \end{array} \right.
$$
代入样本集中得到新的$\pi$,$\mu$,$\Sigma$
$$
\left\{ \begin{array} { l } 
{ N_k = \sum_{i=1}^N{\gamma(i,k)} } \\
{ \mu_k = \frac{1}{N_k} \sum_{i=1} ^ N \gamma(i,k) x_i }\\
{\Sigma _ { k } = \frac { 1 } { N _ { k } } \sum _ { i = 1 } ^ { N } \gamma ( i , k ) \left( x _ { i } - \mu _ { k } \right) \left( x _ { i } - \mu _ { k } \right) ^ { T }} \\
{\pi _ { k } = \frac { N _ { k } } { N } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \gamma ( i , k )}
\end{array} \right.
$$
其中$ N_k$表示第$k$个高斯分布生成的样本点的个数。$\pi_k$表示大小为$N$的样本集中，有多少样本点来自$k$的占比，来表示概率。值得注意的是，$\sum_{k=1}^KN_k=N$。
对两个过程重复迭代至收敛，得到$\pi$,$\mu$,$\Sigma$的估计值。


### 高斯混合模型的EM算法
下面对上述过程进行推导

首先定义隐变量$z$
$$
z_{i,k} = \left\{ \begin{array} { l } { 1, \quad第i个样本属于第k个高斯分布 } \\ { 0, \quad其他 } \end{array} \right.
$$
这是一个one-hot的向量，用$K$维的向量来表示第$i$个观测的隐变量。

首先是$\gamma ( i , k )$，表示第$i$个样本来自第$k$个高斯分布生成的概率。因此

$$
\begin{aligned}
\gamma ( i , k ) &= \frac{P(z_{i,k}=1, x_i | \theta)}{\sum_{l=1}^K{P(z_{i,l}=1,x_i|\theta)}}  \\
&= \frac{P(x_i|z_{i,k}=1, \theta)P(z_{i,k}=1|\theta)}{\sum_{l=1}^K{P(x_i|z_{i,l}=1, \theta)P(z_{i,l}=1|\theta)}} \\
&= \frac{\pi_k N \left( x _ { i } | \mu _ { k } , \Sigma _ { k } \right)}{\sum _ { l = 1 } ^ { K } \pi _ { l } N \left( x _ { i } | \mu _ { l } , \Sigma _ { l } \right)}
\end{aligned}
$$
分子是第$i$个样本，同时又属于第$k$个高斯分布生成的概率。因为$\gamma(i,k)$是概率，所以需要分母进行归一化。第二个等式是写成了条件概率的形式。因此，根据定义，$P(z_{i,k}=1|\theta)=\pi_k$表示第$i$个样本属于第$k$个高斯分布生成的概率，这里忽略了下标$i$，因为对于任意一个样本点$i$，属于第$k$个高斯分布生成的概率都为$\pi_k$，不用区分样本。其次，$P(x_i|z_{i,k}=1, \theta)$表示给定了样本$x_i$是来自第$k$个高斯分布作为已知条件，因此使用第$k$个高斯分布的参数来生成样本$x_i$，因此$P(x_i|z_{i,k}=1, \theta)=N \left( x _ { i } | \mu _ { k } , \Sigma _ { k } \right)$。因此分子变成了$\pi_k N \left( x _ { i } | \mu _ { k } , \Sigma _ { k } \right)$。分母的变换和分子一致。$\gamma ( i , k )$可以计算出来了。

接下来计算Q函数。

首先表达隐变量的条件概率
$$
P(z_{i,k}=1|x_i;\theta)
$$

样本$x_i$与隐变量$z_{i,k}$的联合概率是
$$
\begin{aligned}
P(x_i,z_{i,k}|\theta) &= P(x_i|z_{i,k};\theta)P(z_{i,k}|\theta) \\
&=  \pi _ { k } N \left( x _ { i } | \mu _ { k } , \Sigma _ { k } \right )
\end{aligned}
$$






$$
\begin{aligned}
P(X,Z|\theta) &= \prod_{i=1}^N{P(x_i,z_{i,1},z_{i,2},\cdots,z_{i,K} | \theta)} \\ 
&= 
\end{aligned}
$$
