import numpy as np
import tensorflow as tf

# 准备数据

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义目标函数

def fi(w, y, x):
    return np.log(1 + np.exp(-np.dot(y, w) * x))

def regul(w):
    return np.linalg.norm(w, ord = 1)

def objective_function(w,X,Y,lamb):
    assert lamb > 0
    n = len(X)
    obj = lamb * np.linalg.norm(w, ord = 1)
    for i in range(n):
        obj += fi(w, Y[i], X[i]) / n
    return obj
def grad(w,X,Y,lamb):
    return None



# 搭建神经网络

x = tf.placeholder("float", [None, 784])    # 图片的占位符
W = tf.Variable(tf.zeros([784,10]))         # 模型
b = tf.Variable(tf.zeros([10]))             # 标签

## 神经网络

y = tf.nn.softmax(tf.matmul(x,W) + b)

## 训练神经网络

y_ = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)



init = tf.initialize_all_variables()
saver = tf.train.Saver()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print( sess.run(acc, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

