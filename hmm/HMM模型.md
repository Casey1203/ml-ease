# HMM模型

隐马尔科夫模型（HMM）可用于标注问题，在语音识别、NLP、生物信息（DNA）、模式识别等领域被实践证明是有效的算法。

HMM是关于时许的概率模型，描述一个过程：由一个隐藏的马尔科夫链随机生成不可观测的状态随机序列，同时每个状态会产生一个观测，进而形成观测随机序列。

由隐藏的马尔科夫随机生成的状态的序列，称为状态序列（state sequence）。每个状态生成一个观测，形成的观测的序列称为观测序列（observation sequence）。这两个序列的每个位置可以看作一个时刻。

![Image text](https://raw.github.com/Casey1203/ml-ease/master/img/hmm.png)