# HW-STO

- 考虑问题

$$
\min_{w\in \mathbb{R}^d}\frac{1}{n}\sum_{i=1}^n f_i(w)+\lambda \Vert w\Vert_1\\
\text{where } f_i(w) = \log(1+e^{-y_iw^Tx_i}), \lambda>0.
$$

要求：

- 从下面的算法中选两个，写下来并实现。Adadelta, AdagradDA, Adagrad, ProximalAdagrad, Ftrl, Momentum, adam, Momentum, CenteredRMSProp, nesterov, rmsprop, SAG, SAGA, SVRG。

- 可以考虑看caffe，tensorflow的代码实现，但是要自己实现。

- 用下面两个数据集来对算法进行验证：

  - MNIST

  - Covertype

    算法的设置按照下面[这篇文章](https://arxiv.org/abs/1609.08502)section 5来做.

- 尝试几个$\lambda: 10,1,0.1,0.001.$ 生成上面文章的图A.7的图像来对比结果。

- 加分：提出并实现下面的几个算法：

  - stochastic gradient method using line search
  - stochastic gradient method using Barzilar-Borwein step sizes
  - any other better idea 