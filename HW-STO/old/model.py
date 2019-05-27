from time import time
import numpy as np
import tensorflow as tf
from keras import datasets ## 载入数据

mnist = datasets.mnist
train_data, test_data = mnist.load_data()

ld = 0.001  # 还可以是1 0.1 0.001

def sig(w):
    n = len(w)
    w = np.sign(w)
    for i in range(n):
        if w[i] == 0:
            w[i] = np.random.choice([1, -1, 0])
    return w


def f(y, w, x, ld=ld):
    loss = ld * np.linalg.norm(w, ord=1)
    for i in range(len(x)):
        loss += np.log(1 + np.exp(- y[i] * np.dot(w, x[i].flatten())))
    return loss

def df(y, w, x, ld=ld):
    # d = ld * sig(w)
    d = np.zeros(784)
    for i in range(len(x)):
        temp = np.exp(- y[i] * np.dot(w, x[i].flatten()))
        d += y[i] * temp * x[i].flatten() / (1 + temp)
    return d

def adam_train(data, epoches = 20, batch_size = 1, beta_1 = 0.9, beta_2 = 0.999, alpha = 1e2, ld = 1, eps = 1e-8):
    t1 = time()
    t = 0
    w = np.zeros(784)
    # w = np.random.randn(784) / 1e10 # 随机初值
    mt = np.zeros(784)
    vt = 0
    x , y = data
    x = x / 255.
    y %= 2  # 奇偶二值化
    loss = []

    for epoch in range(epoches):
        total_iteration = len(y) // batch_size
        shuffle = np.arange(len(y))
        np.random.shuffle(shuffle)
        x = x[shuffle]
        y = y[shuffle]
        for _ in range(total_iteration):
            t += 1
            x_ = x[_ * batch_size : (_ + 1) * batch_size]
            y_ = y[_ * batch_size: (_ + 1) * batch_size]
            
            g = df( y_, w, x_, ld )

            mt = beta_1 * mt + (1 - beta_1) * g
            vt = beta_2 * vt + (1-beta_2) * np.dot(g, g)


            mt_hat = mt / (1-beta_1**t)
            vt_hat = vt / (1-beta_2**t)

            w -= (alpha * mt_hat) / (np.sqrt(vt_hat)+eps)
        loss.append(f(y_,w,x_))
            
        print(str(epoch+1)+' out of '+str(epoches)+' : finished, the present loss is '+ str(loss[-1]))
    np.savetxt("adam_results/adam_result_with_lambda_equals_to_"+str(ld)+'.txt', w)
    t2 = time()
    print('total time used for '+str(epoches)+' epoches: '+str(t2-t1)+' s.')
    return w

adam_train(train_data,ld = 10)
# total time used for 20 epoches: 24.99453902244568 s.
adam_train(train_data, ld=1)
# total time used for 20 epoches: 24.76461386680603 s.
adam_train(train_data,ld = 0.1)
# total time used for 20 epoches: 20.175593614578247 s.
adam_train(train_data,ld = 0.001)
# total time used for 20 epoches: 24.603734731674194 s.
## w = numpy.loadtxt("filename.txt", delimiter=',')

def Nesterov_train(data, epoches = 20 , batch_size = 1, beta = 0.9, alpha = 1e-3, ld = 1, eps = 1e-2):
    v = np.zeros(784)
    # w = np.random.randn(784) / 1e10
    w = np.zeros(784)

    x, y = data
    x = x / 255.
    y %= 2  # 奇偶二值化
    t1 = time()

    for epoch in range(epoches):
        total_iteration = len(y) // batch_size
        shuffle = np.arange(len(y))
        np.random.shuffle(shuffle)
        x = x[shuffle]
        y = y[shuffle]

        for _ in range(total_iteration):
            x_ = x[_ * batch_size : (_ + 1) * batch_size]
            y_ = y[_ * batch_size: (_ + 1) * batch_size]

            w_ = w + alpha * v
            g = df(y_, w_, x_, ld)
            v = alpha * v - eps * g

            w += v
        print(str(epoch + 1) + ' out of ' + str(epoches) + ' : finished.')
    np.savetxt("Nesterov_results/Nesterov_result_with_lambda_equals_to_" + str(ld)+'.txt', w)
    t2 = time()
    print('total time used for ' + str(epoches) + ' epoches: ' + str(t2 - t1) + ' s.')
    return w

Nesterov_train(train_data, ld=10)
# total time used for 20 epoches: 26.021044969558716 s.
Nesterov_train(train_data, ld=1)
# total time used for 20 epoches: 25.81741690635681 s.
Nesterov_train(train_data, ld=0.1)
# total time used for 20 epoches: 24.883914947509766 s.
Nesterov_train(train_data, ld=0.0001)
# total time used for 20 epoches: 24.954432010650635 s. 