import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn

from data_utils import *
from cnn import TextCNN


embedding_dim = 128
filter_sizes = "3,4,5"
num_filters = 32

batch_size = 64
num_epochs = 4


def train():
	x_train, y_train, x_test, y_test = load_data_and_labels()

	max_document_length = max([len(x.split(" ")) for x in x_train])
	vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
	x_train = np.array(list(vocab_processor.fit_transform(x_train)))
	x_test = np.array(list(vocab_processor.fit_transform(x_test)))

	sess = tf.Session()
	with sess.as_default():
		cnn = TextCNN(
			sequence_length=x_train.shape[1],
			num_classes=y_train.shape[1],
			vocab_size=len(vocab_processor.vocabulary_),
			embedding_size=embedding_dim,
			filter_sizes=list(map(int, filter_sizes.split(","))),
			num_filters=num_filters)

		optimizer = tf.train.AdamOptimizer(1e-4)
		train_step = optimizer.minimize(cnn.loss)

		sess.run(tf.initialize_all_variables())

		batches = batch_iter(zip(x_train, y_train), batch_size, num_epochs)
		for batch in batches:
			x_batch, y_batch = zip(*batch)
			feed_dict = {
				cnn.input_x: x_batch,
				cnn.input_y: y_batch
			}
			_, accuracy = sess.run([train_step, cnn.accuracy], feed_dict=feed_dict)
			print "accuracy: {}".format(accuracy)

		test_accuracy = sess.run(cnn.accuracy,
			feed_dict={cnn.input_x: x_test, cnn.input_y: y_test})
		print "test accuracy: {}".format(test_accuracy)


if __name__ == "__main__":
	train()