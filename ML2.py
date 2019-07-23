import tensorflow as tf
import matplotlib.pyplot as plt

x = 10
y_true = 2 * x

x_placeholder = tf.placeholder(tf.float32)
y_placeholder = tf.placeholder(tf.float32)

k = tf.Variable(4.0)
y_neuron = x_placeholder * k
loss = tf.reduce_sum((y_placeholder - y_neuron)**2)
optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

loss_history = []

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(0, 1000):
        print('Iteration', i)
        current_loss, current_k, _ = sess.run([loss, k, optimizer], feed_dict={x_placeholder: x, y_placeholder: y_true})
        print('loss =', current_loss)
        print('k =', current_k)
        loss_history.append(current_loss)
    plt.plot(loss_history)
    plt.show()