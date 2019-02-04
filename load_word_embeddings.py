import numpy as np
import time
import tensorflow as tf
from collections import defaultdict
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

		num_labels = defaultdict(int)
		for label in self.train[1]:
			num_labels[np.argmax(label)] += 1

		print (num_labels)
		if FLAGS.mode == "dev":
			split_labels = {i:j//10 for i,j in num_labels.items()}

			t1, t2, t3, t4 = [], [], [], []
			dev1, dev2, dev3, dev4 = [], [], [], []
			train, test = [], []
			for x in zip(*self.train):
				label = np.argmax(x[1])
				x1, x2, x3, x4 = x
				if split_labels[label] > 0:
					dev1.append(x[0])
					dev2.append(x[1])
					dev3.append(x[2])
					dev4.append(x[3])
					split_labels[label] -= 1
				else:
					t1.append(x[0])
					t2.append(x[1])
					t3.append(x[2])
					t4.append(x[3])
			self.train = (np.array(t1), np.array(t2), np.array(t3), np.array(t4))
			self.test = (np.array(dev1), np.array(dev2), np.array(dev3), np.array(dev4))

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

	
