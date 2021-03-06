import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

x_pl = tf.placeholder(tf.float32, [200, 784])
y_pl = tf.placeholder(tf.float32, [200, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x_pl, w) + b)
cross_entrophy = tf.reduce_mean(-tf.reduce_sum(y_pl * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entrophy)

rating = tf.equal(tf.argmax(y, 1), tf.argmax(y_pl, 1))
accuracy = tf.reduce_mean(tf.cast(rating, tf.float32))

loss_history = []

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(1000):
        print('Iteration', i)
        batch_x, batch_y = mnist.train.next_batch(200)
        _, accuracy1 = sess.run([optimizer, accuracy], feed_dict={x_pl: batch_x, y_pl: batch_y})
        print('Точность', accuracy1)
        loss_history.append(accuracy1)
    print(loss_history)
    plt.plot(loss_history)
    plt.show()