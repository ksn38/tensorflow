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

data_root1 = pathlib.Path('C:\\MLProjects\\emptyYOBA')
filenames1 = list(data_root1.glob('*/*'))
filenames1 = [str(path) for path in filenames1]
filenames1 = filenames1*1000
dataset1 = tf.data.Dataset.from_tensor_slices((filenames1))
dataset1 = dataset1.map(_parse_function)

# Training Parameters
learning_rate = 0.01
num_steps = 1000
batch_size = 33
display_step = 100

# Network Parameters
num_input = 10000

rho = 0.01
beta = 1.0

#переменные для весов свертки и свободных членов:
ae_weights = {'encod0': tf.Variable(tf.truncated_normal([1, 1, 1, 1], stddev=0.1)),
              'b_encod0': tf.Variable(tf.truncated_normal([1], stddev=0.1)),
              'encod1': tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1)),
              'b_encod1': tf.Variable(tf.truncated_normal([4], stddev=0.1)),
              "encod2": tf.Variable(tf.truncated_normal([25, 25, 4, 16], stddev=0.1)),
              "b_encod2": tf.Variable(tf.truncated_normal([16], stddev=0.1)),
              'encod3': tf.Variable(tf.truncated_normal([50, 50, 16, 100], stddev=0.1)),
              'b_encod3': tf.Variable(tf.truncated_normal([100], stddev=0.1)),
              "encod4": tf.Variable(tf.truncated_normal([100, 100, 100, 400], stddev=0.1)),
              "b_encod4": tf.Variable(tf.truncated_normal([400], stddev=0.1)),

              "decod4": tf.Variable(tf.truncated_normal([100, 100, 100, 400], stddev=0.1)),
              "b_decod4": tf.Variable(tf.truncated_normal([100], stddev=0.1)),
              'decod3': tf.Variable(tf.truncated_normal([50, 50, 16, 100], stddev=0.1)),
              "b_decod3": tf.Variable(tf.truncated_normal([16], stddev=0.1)),
              "decod2": tf.Variable(tf.truncated_normal([25, 25, 4, 16], stddev=0.1)),
              "b_decod2": tf.Variable(tf.truncated_normal([4], stddev=0.1)),
              'decod1': tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1)),
              'b_decod1': tf.Variable(tf.truncated_normal([1], stddev=0.1)),
              #'decod0': tf.Variable(tf.truncated_normal([1, 1, 1, 1], stddev=0.1)),
              #'b_decod0': tf.Variable(tf.truncated_normal([1], stddev=0.1))
              }


input_shape = tf.stack([batch_size, 100, 100, 1])
h1_shape = tf.stack([batch_size, 50, 50, 4])
h2_shape = tf.stack([batch_size, 25, 25, 16])
h3_shape = tf.stack([batch_size, 10, 10, 100])
h4_shape = tf.stack([batch_size, 5, 5, 400])

x_pl = tf.placeholder(tf.float32, [batch_size, num_input])
#x_pl = tf.placeholder(tf.float32, [batch_size, 100, 100, 1])
x_image = tf.reshape(x_pl, [-1, 100, 100, 1])

# Building the encoder
def encoder(x):
    #создание сверточного слоя (применяет сверточные фильтры):
    encod_h0_logits = tf.nn.conv2d(x, ae_weights['encod0'], strides=[1, 2, 2, 1], padding='SAME') + ae_weights['b_encod0']
    encod = tf.nn.relu(encod_h0_logits)
    encod_h1_logits = tf.nn.conv2d(encod, ae_weights['encod1'], strides=[1, 2, 2, 1], padding='SAME') + ae_weights['b_encod1']
    encod1 = tf.nn.relu(encod_h1_logits)
    encod_h2_logits = tf.nn.conv2d(encod1, ae_weights["encod2"], strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_encod2"]
    encod2 = tf.nn.relu(encod_h2_logits)
    encod_h3_logits = tf.nn.conv2d(encod2, ae_weights["encod3"], strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_encod3"]
    encod3 = tf.nn.relu(encod_h3_logits)
    encod_h4_logits = tf.nn.conv2d(encod3, ae_weights["encod4"], strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_encod4"]
    encod4 = tf.nn.relu(encod_h4_logits)
    return encod4

# Building the decoder
def decoder(x):
    decod_h4_logits = tf.nn.conv2d_transpose(x, ae_weights['decod4'], h4_shape, strides=[1, 2, 2, 1], padding='SAME') + ae_weights['b_decod4']
    decod4 = tf.nn.relu(decod_h4_logits)
    decod_h3_logits = tf.nn.conv2d_transpose(decod4, ae_weights['decod3'], h3_shape, strides=[1, 2, 2, 1], padding='SAME') + ae_weights['b_decod3']
    decod3 = tf.nn.relu(decod_h3_logits)
    decod_h2_logits = tf.nn.conv2d_transpose(decod3, ae_weights["decod2"], h2_shape, strides=[1, 2, 2, 1], padding="SAME") + ae_weights["b_decod2"]
    decod1 = tf.nn.relu(decod_h2_logits)
    decod_h1_logits = tf.nn.conv2d_transpose(decod1, ae_weights['decod1'], h1_shape, strides=[1, 2, 2, 1], padding='SAME') + ae_weights['b_decod1']
    decod = tf.nn.relu(decod_h1_logits)
    #decod_h0_logits = tf.nn.conv2d_transpose(decod, ae_weights['decod0'], input_shape, strides=[1, 2, 2, 1], padding='SAME') + ae_weights['decod0']
    #visible = tf.nn.relu(decod_h0_logits)
    return decod

# Construct model
encoder_op = encoder(x_image)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = x_image

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))

#Определим теперь тензор для регуляризационного слагаемого:
#data_rho = tf.reduce_mean(encod, 0)
#reg_cost = - tf.reduce_mean(tf.log(data_rho/rho) * rho + tf.log((1-data_rho)/(1-rho)) * (1-rho))

#total_cost = loss + beta * reg_cost

optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

dataset = dataset.batch(batch_size)
iterator = dataset.make_one_shot_iterator()
imagesdata = iterator.get_next()

dataset1= dataset1.batch(batch_size)
# step 4: create iterator and final input tensor
iterator1 = dataset1.make_one_shot_iterator()
imagesdata1 = iterator1.get_next()

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
        batch_x = sess.run(imagesdata1)
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