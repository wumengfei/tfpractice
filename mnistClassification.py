#coding: utf-8
__author__ = 'Murphy'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one layer and return the output of this layer

        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name='W')

        with tf.name_scope('bias'):
            biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)

        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        return outputs

def compute_accuracy(v_xs,v_ys):
    y_pre = sess.run(prediction,feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

# Make up some real data
# x_data = np.linspace(-1,1,300)[:,np.newaxis]
# noise = np.random.normal(0,0.05,x_data.shape)
# y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,784],name='x_input') # 28*28
    ys = tf.placeholder(tf.float32,[None,10],name='y_input')
# add hidden layer
#l1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)

# add output layer
prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images,mnist.test.labels))

