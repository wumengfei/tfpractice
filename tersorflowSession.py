#coding: utf-8
__author__ = 'Murphy'

import tensorflow as tf

def run():
    matrix1 = tf.constant([[3,3]])
    matrix2 = tf.constant([[2],
                          [2]])
    product = tf.matmul(matrix1,matrix2)    #matrix multiply np.dot(m1,m2)

    # # method 1
    # sess = tf.Session()
    # result = sess.run(product)
    # print (result)
    # sess.close()

    # method 2  with语句之内 自己会close 不需要人为close
    with tf.Session() as sess:
        result2 = sess.run(product)
        print(result2)

if __name__ == "__main__":
    run()