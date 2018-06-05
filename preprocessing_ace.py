import numpy as np
import tensorflow as tf
from nltk import word_tokenize
import ast

class preprocessing:
	def __init__(self, FLAGS):

		# debug locally
	

		self.record_train = "/raid/data/dost01/ace2005/record_train_ace.txt"
		self.record_test = "/raid/data/dost01/ace2005/record_test_ace.txt"
		self.labels_file = "/raid/data/dost01/ace2005/labels_ace2005.txt"
		self.vocab_file = "/raid/data/dost01/embeddings/glove.6B.300d.txt"
		"""

		self.record_train = "/home/dominik/Documents/DFKI/clean_dir/Hiwi-master/NemexRelator2010/features/record_train_ace2005.txt"
		self.record_test = "/home/dominik/Documents/DFKI/clean_dir/Hiwi-master/NemexRelator2010/features/record_test_ace2005.txt"
		self.labels_file = "/home/dominik/Documents/DFKI/clean_dir/Hiwi-master/NemexRelator2010/features/labels_ace2005.txt"
		self.vocab_file = "/home/dominik/Documents/Supertagging/glove.6B.50d.txt"
		"""

		self.FLAGS = FLAGS

		self.read_embeddings()

		self.read_embeddings()


		self.train = self.read_records(self.record_train)

		self.max_length = max([len(x) for x in self.train[0]])
		xs = np.asarray([np.concatenate((tmp, np.zeros(self.max_length - len(tmp)))) for tmp in self.train[0]])
		positions = np.asarray([np.concatenate((tmp, np.zeros(self.max_length - len(tmp)))) for tmp in self.train[2]])
		print (positions[:5])
		self.train = (xs, self.train[1], positions, self.train[3], self.train[4])

		self.test = self.read_records(self.record_test)	
		self.max_length_test = max([len(x) for x in self.test[0]])
		xs = np.asarray([np.concatenate((tmp, np.zeros(self.max_length_test - len(tmp)))) for tmp in self.test[0]])
		positions = np.asarray([np.concatenate((tmp, np.zeros(self.max_length_test - len(tmp)))) for tmp in self.test[2]])
		self.test = (xs, self.test[1], positions, self.test[3], self.test[4])

		"""
		if corpus == "semeval":
			self.record_train = "/raid/data/dost01/semeval10_data/TRAIN_FILE.TXT"
			self.record_test = "/raid/data/dost01/semeval10_data/TEST_FILE_FULL.TXT"
			self.labels_file = "/raid/data/dost01/semeval10_data/labels.txt"
			self.vocab_file = "/raid/data/dost01/embeddings/glove.6B.300d.txt"

			self.read_embeddings()

			self.max_length = max([len(x) for x in self.train[0]])
			xs = np.asarray([np.concatenate((tmp, np.zeros(self.max_length - len(tmp)))) for tmp in self.train[0]])
			positions = np.asarray([np.concatenate((tmp, np.zeros(self.max_length - len(tmp)))) for tmp in self.train[2]])
			self.train = (xs, self.train[1], positions, self.train[3], self.train[4])
		
			self.max_length_test = max([len(x) for x in self.test[0]])
			xs = np.asarray([np.concatenate((tmp, np.zeros(self.max_length_test - len(tmp)))) for tmp in self.test[0]])
			positions = np.asarray([np.concatenate((tmp, np.zeros(self.max_length_test - len(tmp)))) for tmp in self.test[2]])
			self.test = (xs, self.test[1], positions, self.test[3], self.test[4])
		elif corpus == "ace":
			...
		"""
	def read_embeddings(self):
		"""
		read embeddings and store each word in the rows of the matrix at its index
		returns the embedding matrix
		first row is a vector full of zeros for padding, last row is __UNKNOWN__ vector 
		"""
		counter = 0
		vocab_counter = 1
		self.word2id = {}
		# number of words in embedding file + 2, first is __PADDING__, last is __UNKNOWN__ token
		self.embs = np.zeros((400000 + 2, self.FLAGS.embeddings_dim))
		with open(self.vocab_file) as f:
			for line in f:
				split = line.strip().split()
				# header line
				if len(split) == 2:
					continue
				else:
					word, vec = split[0], [float(val) for val in split[1:]]
					self.word2id[word] = vocab_counter
					self.embs[vocab_counter,:] = vec
					vocab_counter += 1
		self.word2id["__PADDING__"] = 0
		self.word2id["__UNKNOWN__"] = vocab_counter
		self.id2word = {j:i for (i,j) in self.word2id.items()}
		self.embs[vocab_counter,:] = np.random.random(self.FLAGS.embeddings_dim)
		return self


	def read_records(self,fn):
		"""
		reads the train and test file and vectorizes sequences with word indexes
		returns X and y
		"""
		with open(self.labels_file) as labs:
			labels = {lab.strip(): i for i, lab in enumerate(labs)}
		NUM_OOV_TOKENS = 0 
		NUM_TOKENS = 0
		X, y = [], []
		X_queries, X_positions, X_position_of_queries = [], [], []

		"""
		max_len = 0
		if is_train:
			with open(fn) as infile:
				for line in infile:
					line = ast.literal_eval(line)
					sent = line[1]
					max_len = max(len(sent), max_len)
			sent_length = max_len
		"""
		with open(fn) as infile:
			for line in infile:
				line = ast.literal_eval(line)
				sent = line[1]
				e1, e2 = line[2]-1, line[3] - 1
				label = line[4]
				word_ids = []
				queries = []
				query_positions = [e1 + 1, e2 + 1]
				for i, w in enumerate(sent):
					if w == "'s":
						continue
					if not w.isalnum():
						continue
					w = w.replace("-", "_").lower()
					w = "".join(["#" if char.isdigit() else char for char in w])

					if w in self.word2id:
						w = self.word2id[w]
					else:
						w = self.word2id["__UNKNOWN__"]

					if i == e1:
						queries.append(w)
					if i == e2:
						queries.append(w)
					word_ids.append(w)
				if len(queries) != 2:
					continue
				positions = list(range(1,len(word_ids) + 1))
				X.append(word_ids)
				X_positions.append(positions)
				X_queries.append(queries)
				X_position_of_queries.append(query_positions)				
				y.append(self.create_one_hot(len(labels), labels[label]))
		return X, np.asarray(y), X_positions, np.asarray(X_queries), np.asarray(X_position_of_queries)

	def create_one_hot(self, length, x):
		# create one hot vector
		zeros = np.zeros(length, dtype=np.int32)
		zeros[x] = 1
		return zeros

if __name__ == "__main__":

	flags = tf.flags

	#flags.DEFINE_float("learning_rate", 0.001, "") // use adaptive learning rate instead, see explanations below
	flags.DEFINE_integer("num_labels", 19, "number of target labels")
	flags.DEFINE_integer("batch_size", 200, "number of batchsize, bigger works better")
	flags.DEFINE_float("dropout", 0.1, "dropout applied after each layer")
	flags.DEFINE_integer("sent_length", 50, "sentence length")
	flags.DEFINE_integer("num_layers", 1, "num layers for encoding/decoding")
	flags.DEFINE_integer("num_heads",1, "num heads per layer")
	flags.DEFINE_integer("num_epochs",20, "")
	flags.DEFINE_integer("min_length", 0, "min length of encoded sentence")
	flags.DEFINE_integer("embeddings_dim", 50, "number of dimensions in word embeddings")
	flags.DEFINE_float("l2_lambda", 0.0001, "")
	flags.DEFINE_integer("max_gradient_norm", 5, "")
	flags.DEFINE_integer("classifier_units", 100, "")

	FLAGS = flags.FLAGS
	preprocessing = preprocessing(FLAGS)
	for i in range(5):
		print ([preprocessing.id2word[w] for (i,w) in np.ndenumerate(preprocessing.train[0][i]) if preprocessing.id2word[w] != "__PADDING__"])
		print ([w for (i,w) in np.ndenumerate(preprocessing.train[2][i]) if w != 0])
		print ([preprocessing.id2word[w] for (i,w) in np.ndenumerate(preprocessing.train[3][i])])
		print ([w for (i,w) in np.ndenumerate(preprocessing.train[4][i])])
	print (np.shape(preprocessing.train[0]), np.shape(preprocessing.train[1]), np.shape(preprocessing.train[2]), np.shape(preprocessing.train[3]), np.shape(preprocessing.train[4]))
	#print ([p for (i,p) in np.ndenumerate(preprocessing.train[1][:5]) if p != 0])
	#print (preprocessing.train[0][:5])
