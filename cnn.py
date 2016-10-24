import tensorflow as tf


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool(x, ksize):
	return tf.nn.max_pool(x, ksize,
		strides=[1, 1, 1, 1], padding='VALID')


class TextCNN(object):
	def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
		filter_sizes, num_filters):
		self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

		W = weight_variable([vocab_size, embedding_size])
		self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
		self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			# Convolution Layer
			filter_shape = [filter_size, embedding_size, 1, num_filters]
			W = weight_variable(filter_shape)
			b = bias_variable([num_filters])
			conv = conv2d(self.embedded_chars_expanded, W)
			# Apply nonlinearity
			h = tf.nn.relu(tf.nn.bias_add(conv, b))
			# Maxpooling over the outputs
			pooled = max_pool(h, [1, sequence_length - filter_size + 1, 1, 1])
			pooled_outputs.append(pooled)
		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(3, pooled_outputs)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		W = weight_variable([num_filters_total, num_classes])
		b = bias_variable([num_classes])
		self.scores = tf.nn.softmax(tf.matmul(self.h_pool_flat, W) + b)
		self.predictions = tf.argmax(self.scores, 1)

		losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
		self.loss = tf.reduce_mean(losses)

		correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
