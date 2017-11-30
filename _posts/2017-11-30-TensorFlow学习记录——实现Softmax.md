---
layout: post
title: TensorFlow学习记录——实现Softmax识别手写数字
date: 2017-11-30
categories: blog
tags: [Machine learning]
author: gafei
---

### 1.MNIST数据集
MNIST手写数字识别是一个机器学习领域中Hello World任务。  
MNIST是一个非常简单的机器视觉数据集，如下图式，它由数万张28*28像素的手写数字组成，作为灰度图。
![](http://oyvmbp6uy.bkt.clouddn.com/20171130_1.png)

### 2.导入数据集
加载数据集，TensorFlow对其进行了封装以便我们使用：
```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

查看数据集的个数，有55000个训练样本，10000个测试样本，5000个验证样本，关于训练集、测试集、验证集的区别，详细看[这篇文章](https://www.jiqizhixin.com/articles/2017-07-24-8)  
```python
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)
```
### 3.设计算法
我们的图像是28*28的灰度图片，由于这个分类任务比较简单，因此舍弃图像的空间信息，将其变为1维784个数据点的数据集。  

因此我们的训练输入数据是55000*784的Tensor，第一维度是图片编号，第二维度是图片像素点的编号，同时标注Label是一个，55000*10的Tensor，10是分成10类，只有一个值为1，其余都是0。比如数字0对应的是
```python
[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```
数字5就是
```python
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
```
算法使用Softmax Regression来训练，它会对每一种类别进行概率估算，取概率最大的数字最为输出结果。  
Softmax Regression可以看作将特征转化为判定为某一类的概率。在这里，我们可以这样想，比如在某个像素点灰度值很大，那它就很有可能是某一个数字，那么这样权重就很大。  
Softmax Regression的具体实现原理：[文章链接](https://www.zhihu.com/question/23765351)
算法具体过程用公式表达的话就是：\[y = softmax(Wx + b)\]

接下来在TensorFlow中实现它，导入TensorFlow库，创建session，创建placeholder作为存放x数据的地方，这里第一个参数是数据类型，第二个是tensor的shape，在这里是`[None, 784]`，None表示不限制输入。
```python
import tensorflow as tf
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
```
接下来给Softmax Regression中的weights和biases创建Variable对象，Variable是用来存储模型参数，是不断更新的，而placeholder是在运算中不会变的。并且不同于存储数据用的tensor一旦用掉会就消失，如placeholder，Variable是持久化的，一直在显存中。  

在这个例子中weights与biases都初始化为0，但是对于复杂的CNN、RNN或者较深的网络，初始化参数比较重要。
```python
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
```
接下来实现Softmax函数，TensorFlow已经封装好了这个函数，并且会自动实现forward和backward。只要定义号loss，训练时会自动求导，梯度下降，从而完成对参数的训练。
```python
y = tf.nn.softmax(tf.matmul(x, W) + b)
```
定义损失函数loss，对于多分类问题常使用交叉熵(cross-entropy)作为loss function。  
我们可以自己写也可以用tf给我们定义好的函数，官方推荐如果使用的loss function是最小化交叉熵，并且，最后一层是要经过softmax函数处理，则最好使用tf.nn.softmax_cross_entropy_with_logits函数，因为它会帮你处理数值不稳定的问题。
```python
y_ = tf.placeholder(tf.float32, [None, 10])
loss = tf.reduce_mean(-tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)))
```
定义y_是输入的真实数据的标注，y是参数计算获得的，由此求误差。  

现在定义一个优化器进行梯度下降计算，直接调用tf.train.GradientDescentOptimizer，学习率0.5，优化目标loss
```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
```
下一步初始化
```python
tf.global_variables_initializer().run()
```
接下来进行迭代运算，迭代iteration_num次，每次训练100张图片，训练一百张图片，这是进行了随机梯度下降，而不训练所有图片的传统梯度下降，对于大部分机器学习问题，我们都只是用一小部分数据进行随机梯度下降，收敛速度快、运算量小、可以跳出局部最优。  
这里设定每迭代100次输出训练准确率
```python
for i in range(iteration_num):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    if i % 100 == 0:
        print("after %d step\ntrain acc is %g" % (i, accuracy.eval({x: mnist.train.images, y_: mnist.train.labels})))
```
最后在测试集中测试，准确率为92%左右。
```python
test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("the test accuracy is : %g" % test_acc)
```
