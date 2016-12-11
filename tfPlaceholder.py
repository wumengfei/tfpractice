#coding: utf-8
__author__ = 'Murphy'

import tensorflow as tf

def run():
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)

    output = tf.mul(input1,input2)

    with tf.Session() as sess:
        print(sess.run(output,feed_dict={input1:[7],input2:[2.0]}))
if __name__ == "__main__":
    run()