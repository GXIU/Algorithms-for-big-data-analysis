from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import tensorflow as tf 


mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)


ld = 0.001 # 还可以是1 0.1 0.001
alpha = 1e-3
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8
epoches = 20
batch_size = 1000

mt = np.zeros(784)
vt = 0

def sig(w):
    n = len(w)
    w = np.sign(w)
    for i in range(n):
        if w[i] == 0:
            w[i] = np.random.choice([1,-1,0])
    return w
            
def f(y,w,x,ld = ld):
    loss = ld * np.linalg.norm(w,ord = 1)
    for i in range (len(x)):
        loss += np.log(1 + np.exp( - y * np.dot(w,x[i])))
    return loss

def df(y,w,x,ld = ld):
    d = ld * sig(w)
    for i in range(len(x)):
        temp = np.exp(- y[i] * np.dot(w, x[i]))
        d += y[i] * temp * x[i] / ( 1 + temp )
    return d

# adam
def adam_train(epoches = 20 , batch_size = 1000, beta_1 = 0.9, beta_2 = 0.999, alpha = 1e-3, ld = 1, epsilon = 1e-8):
    t = 0
    w = np.random.randn(784)
    mt = np.zeros(784)
    vt = 0
    for epoch in range(epoches):
        total_iteration = 55000 // epoches
        for i in range(total_iteration):
            sample = mnist.train.next_batch(batch_size)
            t += 1
            x , y = sample
            # 变成奇数偶数
            y = y%2
            g = df(y,w,x,ld)

            mt = beta_1 * mt + (1 - beta_1) * g
            vt = beta_2 * vt + (1-beta_2) * np.dot(g,g)

            mt_hat = mt / (1-beta_1**t)
            vt_hat = vt / (1-beta_2**t)

            w -= (alpha * mt_hat) / (np.sqrt(vt_hat)+epsilon)
        print(str(epoch+1)+' out of '+str(epoches)+' : finished.')
    np.savetxt("adam_result_with_lambda_equals_to_"+str(ld), w)
    return w
## w = numpy.loadtxt("filename.txt", delimiter=',')
# def momentum_train(epoches = 20 , batch_size = 1000, lr = , alpha = 1e-3, ld = 1):

