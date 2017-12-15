---
layout: post
title: Logistic Regression 推导
date: 2017-12-14
categories: blog
tags: [Machine learning]
author: gafei
---
## 上周复习了一下LogisticRegression的推导，这周忘了，记一下以免再忘

### Logistic Regression
LogisticRegression（以下简称“对率回归”）本质是一个分类器  
通过Sigmoid函数计算出最终结果，以0.5为分界线，最终结果大于0.5则属于正类(类别值为1)，反之属于负类(类别值为0)。  

函数为：  
![](http://s3.51cto.com/wyfs02/M02/59/03/wKiom1TEnF3xBqCrAAAtwFC_Y7M318.jpg)

## 推导如何训练参数矩阵θ

将y看作样本x是正例的可能性，那么1-y就是反例的概率  
![](http://s3.51cto.com/wyfs02/M02/59/03/wKiom1TEnF2hTs5PAABLtf3DlpQ603.jpg)

合并一下上面的式子：  
![](http://s3.51cto.com/wyfs02/M00/59/00/wKioL1TEnTnA8ZKSAAA8rx3sZUM132.jpg)

求对数似然：  
![](http://s3.51cto.com/wyfs02/M00/59/03/wKiom1TEnF7glwX2AABxj9lYg18460.jpg)  

对参数求梯度上升：  
![](http://s3.51cto.com/wyfs02/M02/59/00/wKioL1TEnTyxHegQAAAgSFtr9U4431.jpg)  
α是步长（学习率）  

对参数θ求偏导：  
![](http://s3.51cto.com/wyfs02/M01/59/03/wKiom1TEnGHhkXbMAAFCAkpV7Zs421.jpg)  
这里容易理解，对log求导是1/log，对Sigmoid函数求导是F(x)*(1-F(x))  

最后梯度迭代公式：  
![](http://s3.51cto.com/wyfs02/M00/59/03/wKiom1TEnGSDl1JuAAAqXOo511s178.jpg)