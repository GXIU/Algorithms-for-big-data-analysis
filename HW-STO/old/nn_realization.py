import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import input_data

# 准备数据

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist.train