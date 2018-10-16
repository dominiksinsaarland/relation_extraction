import numpy as np
import time
import tensorflow as tf
from nltk import word_tokenize
class preprocessing:
	def __init__(self, FLAGS):
		# preprocessing class, called during training
		self.FLAGS = FLAGS
		self.labels_file = self.FLAGS.labels_file
		with open(self.FLAGS.labels_file) as labs:
			self.labels = {lab.strip(): i for i, lab in enumerate(labs)}

		train_embedded = np.load(self.FLAGS.word_embeddings_train + ".npy")
		test_embedded = np.load(self.FLAGS.word_embeddings_test + ".npy")

		train= self.read_file(self.FLAGS.train_file)
		test= self.read_file(self.FLAGS.test_file)

		# train and test are tuples consisting of (embedded_sentences, labels, position_entities, tokens)
		self.train = (train_embedded, train[0], train[1], train[2])
		self.test = (test_embedded, test[0], test[1], test[2])
		self.max_length = 60
		# batches are built dinamically during training

		print ([np.shape(x) for x in self.train])
		print ([np.shape(x) for x in self.test])

	def create_one_hot(self, length, x):
		# create one hot vector
		zeros = np.zeros(length, dtype=np.int32)
		zeros[x] = 1
		return zeros

	def read_file(self, fn):
		"""
		read file constructed during preprocessing
		returns labels, positions, tokens
		"""
		y, position_queries, sent = [], [], []
		with open(fn) as infile:
			for line in infile:
				line = line.strip().split("\t")
				label = int(line[0])
				y.append(self.create_one_hot(len(self.labels), label))
				position_queries.append((int(line[1]), int(line[2])))
				sent.append(line[3].split())

		return np.array(y), np.array(position_queries), np.array(sent)


if __name__ == "__main__":
	flags = tf.flags

	#flags.DEFINE_float("learning_rate", 0.001, "") // use adaptive learning rate instead, see explanations below
	flags.DEFINE_string("labels_file","/raid/data/dost01/semeval10_data/labels.txt", "files with one label per line")
	flags.DEFINE_string("train_file", "semeval10_meta_info_train.txt", "filename of meta info train (positions queries, labels")
	flags.DEFINE_string("test_file", "semeval10_meta_info_test.txt", "filename of meta info test (positions queries, labels")
	flags.DEFINE_string("word_embeddings_train","semeval_elmo_embeddings_train_sentences", "filename of word embeddings train (npy file")
	flags.DEFINE_string("word_embeddings_test", "semeval_elmo_embeddings_test_sentences", "filename of word embeddings test (npy file")
	FLAGS = flags.FLAGS
	preprocessing = preprocessing(FLAGS)

	
