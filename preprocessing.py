import numpy as np
import tensorflow as tf
from nltk import word_tokenize

class preprocessing:
	def __init__(self, FLAGS):

		# debug locally
		"""
		self.record_train = "/home/dominik/Documents/DFKI/Hiwi-master/NemexRelator2010/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"
		self.record_test = "/home/dominik/Documents/DFKI/Hiwi-master/NemexRelator2010/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"
		self.vocab_file = "/home/dominik/Documents/Supertagging/glove.6B.300d.txt"
		self.labels_file = "../../labels.txt"
		"""



		self.record_train = "/raid/data/dost01/semeval10_data/TRAIN_FILE.TXT"
		self.record_test = "/raid/data/dost01/semeval10_data/TEST_FILE_FULL.TXT"
		self.labels_file = "/raid/data/dost01/semeval10_data/labels.txt"
		self.vocab_file = "/raid/data/dost01/embeddings/glove.6B.300d.txt"


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
			self.record_train = "/raid/data/dost01/ace2005/record_train_ace.txt"
			self.record_test = "/raid/data/dost01/ace2005/record_test_ace.txt"
			self.labels_file = "/raid/data/dost01/2005/labels_ace2005.txt"
			self.vocab_file = "/raid/data/dost01/embeddings/glove.6B.300d.txt"
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

	def read_records(self, fn):
		with open(self.labels_file) as labs:
			labels = {lab.strip(): i for i, lab in enumerate(labs)}
		invs = [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,16,18,17]

		with open(self.labels_file) as labs:
			labels_inv = {lab.strip():i for i, lab in zip(invs, labs)}

		counter = 0
		X, y = [], []
		X_queries, X_positions, X_position_of_queries = [], [], []

		with open(fn) as infile:
			for line in infile:
				if counter % 4 == 0:
					sent = line.strip().split("\t")[1][1:-1]
					sent = word_tokenize(sent)
					context_left = []
					context_middle = []
					context_right = []

					for i, w in enumerate(sent):
						if "".join(sent[i:i+3]) == "<e1>":
							context_left = sent[:i]
							start_e1 = i + 3
							break
					i = start_e1
					for w in sent[start_e1:]:
						if "".join(sent[i:i+3]) == "</e1>":
							e1 = sent[start_e1:i]
							end_e1 = i + 3	
							break					
						i += 1
					i = end_e1
					for w in sent[end_e1:]:
						if "".join(sent[i:i+3]) == "<e2>":
							context_mid = sent[end_e1:i]
							start_e2 = i + 3
							break
						i += 1
					i = start_e2
					for w in sent[start_e2:]:
						if "".join(sent[i:i+3]) == "</e2>":
							e2 = sent[start_e2:i]
							end_e2 = i + 3
						i += 1
					context_right = sent[end_e2:]
					if len(e1) > 1:
						context_left += e1[:-1]
						e1 = [e1[-1]]
					if len(e2) > 1:
						context_mid += e2[:-1]
						e2 = [e2[-1]]
					tmp = []
					"""
					for w in context_mid:
						if w == "'s":
							continue
						if not w.isalnum():
							continue
						w = w.replace("-", "_")
						w = "".join(["#" if char.isdigit() else char for char in w])
						if w in vocab:
							tmp.append(vocab[w])
						else:
							tmp.append(vocab["__UNKNOWN__"])
					e1 = e1[0].lower().replace("-", "_")
					if e1 in vocab:
						q1 = vocab[e1]
					else:
						q1 = vocab["__UNKNOWN__"]
					e2 = e2[0].lower().replace("-", "_")
					if e2 in vocab:
						q2 = vocab[e2]
					else:
						q2 = vocab["__UNKNOWN__"]
					"""
					tmp = context_left + e1 + context_mid + e2 + context_right
					# get rid of sentence ending dot
					tmp = tmp[:-1]
					word_ids = []
					for w in context_left:
						if w == "'s":
							continue
						if not w.isalnum():
							continue
						w = w.replace("-", "_").lower()
						w = "".join(["#" if char.isdigit() else char for char in w])
						if w in self.word2id:
							word_ids.append(self.word2id[w])
						else:
							word_ids.append(self.word2id["__UNKNOWN__"])
					qp_1 = len(word_ids) + 1
					for w in e1:
						if w == "'s":
							continue
						if not w.isalnum():
							continue
						w = w.replace("-", "_").lower()
						w = "".join(["#" if char.isdigit() else char for char in w])
						if w in self.word2id:
							word_ids.append(self.word2id[w])
						else:
							word_ids.append(self.word2id["__UNKNOWN__"])
					for w in context_mid:
						if w == "'s":
							continue
						if not w.isalnum():
							continue
						w = w.replace("-", "_").lower()
						w = "".join(["#" if char.isdigit() else char for char in w])
						if w in self.word2id:
							word_ids.append(self.word2id[w])
						else:
							word_ids.append(self.word2id["__UNKNOWN__"])

					qp_2 = len(word_ids) + 1
					for w in e2:
						if w == "'s":
							continue
						if not w.isalnum():
							continue
						w = w.replace("-", "_").lower()
						w = "".join(["#" if char.isdigit() else char for char in w])
						if w in self.word2id:
							word_ids.append(self.word2id[w])
						else:
							word_ids.append(self.word2id["__UNKNOWN__"])
					for w in context_right:
						if w == "'s":
							continue
						if not w.isalnum():
							continue
						w = w.replace("-", "_").lower()
						w = "".join(["#" if char.isdigit() else char for char in w])
						if w in self.word2id:
							word_ids.append(self.word2id[w])
						else:
							word_ids.append(self.word2id["__UNKNOWN__"])


					e1 = e1[0].lower().replace("-", "_")
					if e1 in self.word2id:
						q1 = self.word2id[e1]
					else:
						q1 = self.word2id["__UNKNOWN__"]
					e2 = e2[0].lower().replace("-", "_")
					if e2 in self.word2id:
						q2 = self.word2id[e2]
					else:
						q2 = self.word2id["__UNKNOWN__"]

					#positions = [i + 2 for i in range(len(tmp))]
					queries = [q1, q2]
					query_positions = [qp_1, qp_2]
					#query_positions = [1, len(tmp) + 2]
					#print (tmp, positions, [e1, e2], query_positions)
				
					tmp = word_ids
					positions = list(range(1,len(tmp) + 1))				
					#print (tmp, positions, queries, query_positions)
					#time.sleep(1)
					X.append(tmp)
					X_positions.append(positions)
					X_queries.append(queries)
					X_position_of_queries.append(query_positions)
					#time.sleep(1)


				elif counter % 4 == 1:
					label = line.strip()
					#print (label, fn))
					y.append(self.create_one_hot(len(labels), labels[label]))
					"""
					if "train" in fn.lower():
					#if "train" in fn.lower() and counter < 4 * 7500:
						if label != "Other":
							y.append(self.create_one_hot(len(labels), labels_inv[label]))
							queries = list(reversed(queries))
							query_positions = list(reversed(query_positions))
							X.append(tmp)
							X_positions.append(positions)
							X_queries.append(queries)
							X_position_of_queries.append(query_positions)
					"""
				counter += 1
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
	#print ([p for (i,p) in np.ndenumerate(preprocessing.train[1][:5]) if p != 0])
	#print (preprocessing.train[0][:5])
