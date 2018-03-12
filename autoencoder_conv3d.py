import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', validation_size=0)

img = mnist.train.images[2]

inputs_ = tf.placeholder(tf.float32, (None, None, 28, 28, 1), name='inputs')
targets_ = tf.placeholder(tf.float32, (None, None, 28, 28, 1), name='targets')



### Encoder
conv1 = tf.layers.conv3d(inputs_, 32, (5,5,5), padding='valid', activation=tf.nn.relu)
#now 24x24x32
conv2 = tf.layers.conv3d(conv1, 16, (5,5,5), padding='valid', activation=tf.nn.relu)
#now 20x20x16
conv3 = tf.layers.conv3d(conv2, 8, (5,5,5), padding='valid', activation=tf.nn.relu)
#now 16x16x8


### Decoder
deconv1 = tf.layers.conv3d_transpose(conv3, 8, (5,5,5), padding='valid', activation=tf.nn.relu, use_bias=False)
#now 20x20x8
deconv2 = tf.layers.conv3d_transpose(deconv1, 16, (5,5,5), padding='valid', activation=tf.nn.relu, use_bias=False)
#now 24x24x16
deconv3 = tf.layers.conv3d_transpose(deconv2, 32, (5,5,5), padding='valid', activation=tf.nn.relu, use_bias=False)
#now 28x28x32

logits = tf.layers.conv3d_transpose(deconv3, 1, (3,3,3), padding='same', activation=None, use_bias=False)
#Now 28x28x1

decoded = tf.nn.sigmoid(logits, name='decoded')

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(0.001).minimize(cost)


sess = tf.Session()


epochs = 1
# the batch size is representing 20 batches of 10 frames of images 10x20
batch_size = 200

noise_factor = 0.5
sess.run(tf.global_variables_initializer())
for e in range(epochs):
    for ii in range(mnist.train.num_examples//batch_size):
        batch = mnist.train.next_batch(batch_size)
        # Get images from the batch
        # reshaping the batch (numberOfFrames == 10, batchSize == 20, imgWidth, imgHeight, channels)
        # this is to match the conv3d inputs
        # not sure why this is not working if its reshaped to (20, 10, 28, 28, 1)
        imgs = batch[0].reshape((10, 20, 28, 28, 1))
        
        # Add random noise to the input images
        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        
        # Noisy images as inputs, original images as targets
        batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: noisy_imgs, targets_: imgs})

        print("Epoch: {}/{}...".format(e+1, epochs),
              "Training loss: {:.4f}".format(batch_cost))

#fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
#in_imgs = mnist.test.images[:10]
#reconstructed = sess.run(decoded, feed_dict={inputs_: in_imgs.reshape((1, 10, 28, 28, 1))})

#for images, row in zip([in_imgs, reconstructed], axes):
#    for img, ax in zip(images, row):
#        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)


#fig.tight_layout(pad=0.1)

#plt.show()


#fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
#in_imgs = mnist.test.images[:10]
#noisy_imgs = in_imgs + noise_factor * np.random.randn(*in_imgs.shape)
#noisy_imgs = np.clip(noisy_imgs, 0., 1.)

#reconstructed = sess.run(decoded, feed_dict={inputs_: noisy_imgs.reshape((10, 1, 28, 28, 1))})

#for images, row in zip([noisy_imgs, reconstructed], axes):
#    for img, ax in zip(images, row):
#        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
#        ax.get_xaxis().set_visible(False)
#        ax.get_yaxis().set_visible(False)

#fig.tight_layout(pad=0.1)


#plt.show()

sess.close()
