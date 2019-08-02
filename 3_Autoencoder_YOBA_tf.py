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

# Parse every image in the dataset using `map`
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
num_steps = 3000
batch_size = 25

display_step = 100
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 10000 # YOBA data input (img shape: 100*100)

# tf Graph input (only pictures)
x_pl = tf.placeholder("float", [None, num_input])

weights = {'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),}
biases = {'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(x_pl)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = x_pl

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

dataset = dataset.batch(batch_size)

# Create iterator and final input tensor
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
        # Get the next batch of YOBA data (only images are needed, not labels)
        batch_x = sess.run(imagesdata)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={x_pl: batch_x})
        # Display logs per step
        loss_history.append(l)
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
    plt.plot(loss_history)
    plt.show()
    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    canvas_orig = np.empty((100 * n, 100 * n))
    canvas_recon = np.empty((100 * n, 100 * n))
    for i in range(n):
        # YOBA test set
        batch_x = sess.run(imagesdata)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={x_pl: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original images
            canvas_orig[i * 100:(i + 1) * 100, j * 100:(j + 1) * 100] = \
                batch_x[j].reshape([100, 100])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed images
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
