#!/usr/bin/env python
# coding: utf-8

# Consider deep networks with numerous layers to gain an understanding of how hybridization works. Let's have a look as how we can handle this for substantial chunks of the code by replacing get net() with tf.function (). To begin, we define a basic MLP.

# In[18]:


import tensorflow as tf
from tensorflow.keras.layers import Dense
from d2l import tensorflow as d2l

def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net


# To demonstrate the performance improvement gained by compilation we compare the time needed to evaluate net(x) before and after hybridization. Let us define a class to measure this time first. 

# In[19]:


class Benchmark:
    """For measuring running time."""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')
        
net = get_net()
with Benchmark('Eager Mode'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph Mode'):
    for i in range(1000): net(x)

