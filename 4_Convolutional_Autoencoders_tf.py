from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib

data_root = pathlib.Path('C:\\MLProjects\\YOBA')
filenames = list(data_root.glob('*/*'))
filenames = [str(path) for path in filenames]
filenames = filenames*1000

dataset = tf.data.Dataset.from_tensor_slices((filenames))

# step 3: parse every image in the dataset using `map`
def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
    image = tf.cast(image_decoded, tf.float32)
    image = tf.reshape(image,[100 * 100 * 1])
    image /= 255.0
    return image

dataset = dataset.map(_parse_function)

# Training Parameters
learning_rate = 0.01
num_steps = 1000
batch_size = 100
display_step = 100

# Network Parameters
#num_input = 10000

rho = 0.01
beta = 1.0

#переменные для весов свертки и свободных членов:
ae_weights = {'conv1': tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1)),
              'b_conv1': tf.Variable(tf.truncated_normal([4], stddev=0.1)),
              'conv2': tf.Variable(tf.truncated_normal([5, 5, 4, 16], stddev=0.1)),
              'b_hidden': tf.Variable(tf.truncated_normal([16], stddev=0.1)),
              "conv3": tf.Variable(tf.truncated_normal([5, 5, 16, 64], stddev=0.1)),
              "b_hidden3": tf.Variable(tf.truncated_normal([64], stddev=0.1)),
              "deconv0": tf.Variable(tf.truncated_normal([5, 5, 16, 64], stddev=0.1)),
              "b_deconv0": tf.Variable(tf.truncated_normal([16], stddev=0.1)),
              'deconv1': tf.Variable(tf.truncated_normal([5, 5, 4, 16], stddev=0.1)),
              'b_deconv': tf.Variable(tf.truncated_normal([4], stddev=0.1)),
              'deconv2': tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1)),
              'b_visible': tf.Variable(tf.truncated_normal([1], stddev=0.1))}

input_shape = tf.stack([batch_size, 100, 100, 1])
h1_shape = tf.stack([batch_size, 50, 50, 4])
h2_shape = tf.stack([batch_size, 25, 25, 16])

#x_pl = tf.placeholder(tf.float32, [batch_size, num_input])
x_pl = tf.placeholder(tf.float32, [batch_size, 100, 100, 1])
#images = tf.reshape(x_pl, [-1, 100, 100, 1])

# Building the encoder
def encoder(x):
    #создание сверточного слоя (применяет сверточные фильтры):
    conv1_logits = tf.nn.conv2d(x_pl, ae_weights['conv1'], strides=[1, 2, 2, 1], padding='SAME') + ae_weights['b_conv1']
    conv1 = tf.nn.relu(conv1_logits)
    #создание сверточного слоя (применяет сверточные фильтры):
    hidden_logits = tf.nn.conv2d(conv1, ae_weights['conv2'], strides=[1, 2, 2, 1], padding='SAME') + ae_weights['b_hidden']
    hidden = tf.nn.relu(hidden_logits)
    hidden_logits3 = tf.nn.conv2d(hidden, ae_weights["conv3"], strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_hidden3"]
    hidden3 = tf.nn.relu(hidden_logits3)
    return hidden3

# Building the decoder
def decoder(x):
    deconv_h1_logits0 = tf.nn.conv2d_transpose(hidden3, ae_weights["deconv0"], h2_shape, strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_deconv0"]
    deconv_h10 = tf.nn.relu(deconv_h1_logits0)
    #создание сверточного слоя (применяет сверточные фильтры):
    deconv_logits = tf.nn.conv2d_transpose(deconv_h10, ae_weights['deconv1'], h1_shape, strides=[1, 2, 2, 1], padding='SAME') + ae_weights['b_deconv']
    deconv = tf.nn.relu(deconv_logits)
    #создание сверточного слоя (применяет сверточные фильтры):
    visible_logits = tf.nn.conv2d_transpose(deconv, ae_weights['deconv2'], input_shape, strides=[1, 2, 2, 1], padding='SAME') + ae_weights['b_visible']
    visible = tf.nn.relu(visible_logits)
    return visible

# Construct model
encoder_op = encoder(x_pl)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = x_pl

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))

#Определим теперь тензор для регуляризационного слагаемого:
#data_rho = tf.reduce_mean(hidden, 0)
#reg_cost = - tf.reduce_mean(tf.log(data_rho/rho) * rho + tf.log((1-data_rho)/(1-rho)) * (1-rho))

#total_cost = loss + beta * reg_cost

optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

dataset = dataset.batch(batch_size)
iterator = dataset.make_one_shot_iterator()
imagesdata = iterator.get_next()

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

loss_history = []
# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x = sess.run(imagesdata)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={x_pl: batch_x})
        loss_history.append(l)
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((100 * n, 100 * n))
    canvas_recon = np.empty((100 * n, 100 * n))
    for i in range(n):
        # image test set
        batch_x = sess.run(imagesdata)
        # Encode and decode the image
        g = sess.run(decoder_op, feed_dict={x_pl: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original image
            canvas_orig[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100] = \
                batch_x[j].reshape([100, 100])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed image
            canvas_recon[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100] = \
                g[j].reshape([100, 100])

    print("Original Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_orig, origin="upper", cmap="gray")
    plt.show()

    print("Reconstructed Images")
    plt.figure(figsize=(n, n))
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()

    print("loss_history")
    plt.plot(loss_history)
    plt.show()

    print(loss_history)
