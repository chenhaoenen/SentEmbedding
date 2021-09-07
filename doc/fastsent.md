## 简介 
paper: [Learning Distributed Representations of Sentences from Unlabelled Data](https://arxiv.org/abs/1602.03483)

code: https://github.com/fh295/SentenceRepresentation

## method  

### 摘要  
1. Deeper, more complex models are preferable for representations to be used in supervised systems,
2. shallow log-linear models work best for building representation spaces that can be decoded with simple spatial distance metrics.(感觉说的是无监督)
### 提了两个方法
1. SDAE(Sequential Denoising Autoencoders)
2. FastSent

### SDAE
从DAE(denoising autoencoders)衍化而来，其实就是autoencoder,给序列加点噪音，然后recover the original data,该方法常用与CV领域，
文章提供了两种noise function,第一种是以一定概率删除部分token， 第二种是以一定的概率swap两个token,然后recover the original data
源代码我没来得及看,因为是基于theano框架的
### FastSent
该方案是基于SkipThought vector衍化而来，论文之处SkipThought的缺点是训练缓慢，改善的方法是使用BOW的方案。具体的公式如下:
<img src="./images/fastsent1.png"/>
