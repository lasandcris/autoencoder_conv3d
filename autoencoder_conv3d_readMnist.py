import tensorflow as tf
import os
from glob import glob
from matplotlib import pyplot
import numpy as np
from PIL import Image
import math
import collections
import time

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")

def load_examples(input_dir):
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob(os.path.join(input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        #path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        path_queue = tf.train.string_input_producer(input_paths)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 1, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 1])

        width = tf.shape(raw_input)[1] # [height, width, channels]

        a_images = raw_input[:,:width//2,:]
        b_images = raw_input[:,width//2:,:]
        

    inputs, targets = [a_images, b_images]

    def transform(image):
        r = image
        scale_size = 28
        r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))

    inputs_batch = tf.reshape(tensor=inputs_batch, shape=(1,16,28,28,1))
    targets_batch = tf.reshape(tensor=targets_batch, shape=(1,16,28,28,1))


    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def network(input_imgs, target_img):
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

		loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_img, logits=logits)
		cost = tf.reduce_mean(loss)
		opt = tf.train.AdamOptimizer(0.001).minimize(cost)

		return cost, opt

def train(epoch_count, batch_size, files, data_dir):

	examples = load_examples(data_dir)
	print("examples count = %d" % examples.count)

	cost, opt = network(examples.inputs, examples.targets)

	def convert(image):
		return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

	with tf.name_scope("convert_inputs"):
		converted_inputs = convert(examples.inputs)

	with tf.name_scope("convert_targets"):
		converted_targets = convert(examples.targets)

	with tf.name_scope("parameter_count"):
		parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

	output_dir = "\\ouput"
	summary_freq = 100
	trace_freq = 0

	saver = tf.train.Saver(max_to_keep=1)

	logdir = output_dir if (trace_freq > 0 or summary_freq > 0) else None
	sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

	with sv.managed_session() as sess:
		print("parameter_count =", sess.run(parameter_count))

		max_steps = 2**32
		if epoch_count is not None:
			max_steps = examples.steps_per_epoch * epoch_count

		print(examples.steps_per_epoch)

		if max_steps is not None:
			max_steps = max_steps

		# training
		start = time.time()

		for step in range(max_steps):

			fetches = {"train": cost, "global_step": sv.global_step,}

			results, _ = sess.run([cost, opt])
			if step % 500 == 0:
				print("Epoch: {}/{}...".format(step,max_steps),
	              "Training loss: {:.4f}".format(results))
			if step % 1000 == 0:
				print("saving model")
				saver.save(sess, os.path.join(output_dir, "model"), global_step=sv.global_step)



data_dir = "\\sidebyside"
files = glob(os.path.join(data_dir, '*.jpg'))

batch_size = 16
epochs = 2

with tf.Graph().as_default():
    train(epochs, batch_size, files, data_dir)