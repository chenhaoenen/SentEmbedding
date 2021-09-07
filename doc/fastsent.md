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
该方案是基于SkipThought vector衍化而来，论文之处SkipThought的缺点是训练缓慢，改善的方法是使用BOW的方案。具体的公式如下$s_1$：


在和Bert一样的随机mask后，给定一个随机打乱的token,文章的目标是重构被shuffled token正确的顺序：
<!-- $$
\arg \max _{\theta} \sum \log P\left(\operatorname{pos}_{1}=t_{1}, \operatorname{pos}_{2}=t_{2}, \ldots, \operatorname{pos}_{K}=t_{K} \mid t_{1}, t_{2}, \ldots, t_{K}, \theta\right)
$$ --> 

<div align="center"><img style="background: white;" src="..\..\svg\i5wdY0TpCM.svg"></div>
注意：structBert这里也采用了span的方式，但是它不是span mask,而是打乱shuffled span, $K$ 表示每个shuffled subsequence打乱子序列的长度，文章采用如图(a)所示的$ K=3$  trigrams 长度的 subsequence.


<img src="https://render.githubusercontent.com/render/math?math=x_{1,2} = \frac{-b \pm \sqrt{b^2-4ac}}{2b}">
<img src="https://render.githubusercontent.com/render/math?math=x_{1,2} = \frac{-b \pm \sqrt{b^2-4ac}}{2b}">

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}) 
```math
a^2+b^2=c^2
```
$ \sum_{\forall i}{x_i^{2}} $