---
layout: post
title: CSS224n之Wrod2Vec理解SkipGram
date: 2017-12-13
categories: blog
tags: [NLP]
author: gafei
---

## 刚开始NLP的相关学习，在《TensorFlow实战》中讲到Word2Vec没有看懂，遂开始学习斯坦福大学的CS224n课程

### 参考文章：  
[理解 Word2Vec 之 Skip-Gram 模型](https://zhuanlan.zhihu.com/p/27234078) by 天雨粟 机器不学习  
[CS224n笔记2 词的向量表示：word2vec](http://www.hankcs.com/nlp/word-vector-representations-word2vec.html) by 码农场  

### Word2Vec和Embeddings
Word2Vec其实就是通过学习文本来用词向量的方式表征词的语义信息，即通过一个嵌入空间使得语义上相似的单词在该空间内距离很近。Embedding其实就是一个映射，将单词从原先所属的空间映射到新的多维空间中，也就是把原先词所在空间嵌入到一个新的空间中去。

### SkipGram
![](http://wx1.sinaimg.cn/large/006Fmjmcly1fgco3v2ca7j30pq0j7drt.jpg)
理解这张图片就能理解什么是SkipGram了，什么是SkipGram，简单来讲就是根据单词推测上下文，如下图所示  

![](http://wx4.sinaimg.cn/large/006Fmjmcly1fgcmzglo19j31ay0n41kx.jpg)  
从左边开始，样本输入是一个one-hot编码后的向量，比如你的词库只有一句话**The dog barked at the mailman**，只有5个单词，那么单词  
**the**的one-hot编码就是  
`[1 0 0 0 0]`  

**dog**的编码是  
`[0 1 0 0 0]`  
依此类推，编码完成后，每一个向量可以唯一表示一个单词，模型的输入如果为一个10000维的向量，那么输出也是一个10000维度（词汇表的大小）的向量，它包含了10000个概率，每一个概率代表着当前词是输入样本中output word的概率大小。  

接下来输入向量*矩阵W，W是存放各个词语特征的矩阵，如果我们现在想用300个特征来表示一个单词（即每个词可以被表示为300维的向量）。那么隐层的权重矩阵应该为5行，300列（隐层有300个结点）。这里每一列代表词库里每一个单词在该特征里的权重，每一行代表该单词的每个特征的权重。所以可以很清楚知道，当每个词语乘上这个矩阵之后，获得的就是这个词语的所有特征，也就是所说的词向量。  

看下面的图片，左右两张图分别从不同角度代表了输入层-隐层的权重矩阵。左图中每一列代表一个10000维的词向量和隐层单个神经元连接的权重向量。从右边的图来看，每一行实际上代表了每个单词的词向量。  
如果两个不同的单词有着非常相似的“上下文”（也就是窗口单词很相似，比如“Kitty climbed the tree”和“Cat climbed the tree”），那么通过我们的模型训练，这两个单词的词向量将非常相似。  

![](https://pic1.zhimg.com/50/v2-c538566f7d627ce7ca40589f15ca8284_hd.jpg)

目标就是通过大量样本训练这个矩阵  

然后继续，将词向量与W'（转置）相乘，得到的是这个词的词向量与其他词的词向量的相似度，在SkipGram中，我们的“相似度”定义为是否为上下文，如果是上下文，那么相似度就高。Softmax之后就可以开始反向优化参数了，Softmax是让大的数更大，小的数更小，类似max。

![](http://wx1.sinaimg.cn/large/006Fmjmcly1fgcnqqwb02j314m0qwwpv.jpg)  

得到相似度（预测概率）之后，定义loss，然后优化loss，这里就不细讲了，在Softmax用的是交叉熵跟SGD。  

![](http://wx1.sinaimg.cn/large/006Fmjmcly1fgcn9ndo8dj316s092jwj.jpg)  

接下来的文章介绍如何用TensorFlow实现Word2Vec中的SG（预测上下文），CBOW（根据上下文预测单词）模型