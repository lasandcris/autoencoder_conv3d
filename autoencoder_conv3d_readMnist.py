import tensorflow as tf
import os
from glob import glob
from matplotlib import pyplot
import numpy as np
from PIL import Image

def get_image(image_path, width, height):
    image = Image.open(image_path)

    return np.array(image.convert('L'))

def get_batch(image_files, width, height):
	mode = 'L'
	data_batch = np.array([get_image(sample_file, width, height) for sample_file in image_files]).astype(np.float32)
	data_batch = data_batch.reshape(data_batch.shape + (1,))
	return data_batch

def get_batches(batch_size, data_files):
    current_index = 0
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    image_channels = 1
    self_shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, image_channels
    while current_index + batch_size <= self_shape[0]:
        data_batch = get_batch(data_files[current_index:current_index + batch_size],*self_shape[1:3])

        current_index += batch_size

        yield data_batch


def model_inputs():
	inputs_ = tf.placeholder(tf.float32, (None, None, 28, 28, 1), name='inputs')
	targets_ = tf.placeholder(tf.float32, (None, None, 28, 28, 1), name='targets')

	return inputs_, targets_

def network(input_imgs):
	with tf.variable_scope('network'):
		### Encoder
		conv1 = tf.layers.conv3d(input_imgs, 32, (5,5,5), padding='valid', activation=tf.nn.relu)
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

		return logits

def model_loss(input_imgs, target_img):

   	logits = network(input_imgs)

   	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_img, logits=logits)
   	cost = tf.reduce_mean(loss)
   	opt = tf.train.AdamOptimizer(0.001).minimize(cost)

   	return cost, opt

def train(epoch_count, batch_size, files):

    inputs_, targets_ = model_inputs()
        
    cost, opt = model_loss(inputs_, targets_)

    noise_factor = 0.5
    step = 0
    with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    for epoch_i in range(epoch_count):
	        for batch_images in get_batches(batch_size, files):
	        	batch_images = batch_images.reshape((1, 16, 28, 28, 1))
	        	noisy_imgs = batch_images + noise_factor * np.random.randn(*batch_images.shape)
	        	noisy_imgs = np.clip(noisy_imgs, 0., 1.)

	        	batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: noisy_imgs, targets_: batch_images})

	        	if step % 1 == 0:
	        		print("Epoch: {}/{}...".format(epoch_i+1, epoch_count), "Training loss: {:.4f}".format(batch_cost))

	        	step += 1


data_dir = "..\\mnist\\mnist"
files = glob(os.path.join(data_dir, '*.jpg'))

batch_size = 16
epochs = 10

with tf.Graph().as_default():
    train(epochs, batch_size, files)