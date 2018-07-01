import numpy as np
import tensorflow as tf
from nltk import word_tokenize
import time
class preprocessing_kbp37:
	def __init__(self, FLAGS):

		# debug locally



		self.vocab_file = "/raid/data/dost01/embeddings/glove.6B." + str(FLAGS.embeddings_dim) + "d.txt"
		self.record_train = "/raid/data/dost01/kbp37/kbp37_train.txt"
		self.record_dev = "/raid/data/dost01/kbp37/kbp37_dev.txt"
		self.record_test = "/raid/data/dost01/kbp37/kbp37_test.txt"
		self.labels_file = "/raid/data/dost01/kbp37/labels.txt"

		"""
		self.vocab_file = "/home/dominik/Documents/Supertagging/glove.6B." + str(FLAGS.embeddings_dim) + "d.txt"
		self.record_train = "/home/dominik/Documents/DFKI/kbp37/kbp37_train.txt"
		self.record_dev = "/home/dominik/Documents/DFKI/kbp37/kbp37_dev.txt"
		self.record_test = "/home/dominik/Documents/DFKI/kbp37/kbp37_test.txt"
		self.labels_file = "/home/dominik/Documents/DFKI/kbp37/labels.txt"
		"""


		self.oov_file = open("OOVS_kbp37.txt", "w")

		self.FLAGS = FLAGS

		self.read_embeddings()

		self.train = self.read_records(self.record_train)
		self.dev = self.read_records(self.record_dev)
		self.train = (self.train[0] + self.dev[0], np.concatenate((self.train[1], self.dev[1])), self.train[2] + self.dev[2], np.concatenate((self.train[3], self.dev[3])), np.concatenate((self.train[4], self.dev[4])))

		self.max_length = max([len(x) for x in self.train[0]])
		xs = np.asarray([np.concatenate((tmp, np.zeros(self.max_length - len(tmp)))) for tmp in self.train[0]])
		positions = np.asarray([np.concatenate((tmp, np.zeros(self.max_length - len(tmp)))) for tmp in self.train[2]])
		print (positions[:5])
		self.train = (xs, self.train[1], positions, self.train[3], self.train[4])

		"""
		# DEV SEST HERE

		self.test = [x[-500:] for x in self.train]
		self.train = [x[:-500] for x in self.train]
		"""


		self.test = self.read_records(self.record_test)
		self.sentences_info = self.test[-1]
		self.max_length_test = max([len(x) for x in self.test[0]])
		xs = np.asarray([np.concatenate((tmp, np.zeros(self.max_length_test - len(tmp)))) for tmp in self.test[0]])
		positions = np.asarray([np.concatenate((tmp, np.zeros(self.max_length_test - len(tmp)))) for tmp in self.test[2]])
		self.test = (xs, self.test[1], positions, self.test[3], self.test[4])
		print ([np.shape(x) for x in self.train])
		print ([np.shape(x) for x in self.test])
		for i in self.sentences_info[:5]:
			print (i)


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
		#c_1, c_2 = 0,0
		with open(self.vocab_file) as f:
			for line in f:
				split = line.strip().split()
				# header line
				if len(split) == 2:
					continue
				else:
					word, vec = split[0], [float(val) for val in split[1:]]
					"""
					if "-" in word:
						#print (word)
						c_1 += 1
					if "_" in word:
						print (word)
						c_2 += 1
					"""
					self.word2id[word] = vocab_counter
					self.embs[vocab_counter,:] = vec
					vocab_counter += 1
		#print (c_1, c_2) // 33402 460
		self.word2id["__PADDING__"] = 0
		self.word2id["__UNKNOWN__"] = vocab_counter
		self.id2word = {j:i for (i,j) in self.word2id.items()}
		self.embs[vocab_counter,:] = np.random.random(self.FLAGS.embeddings_dim)
		return self

	def lookup(self, w):
		w = w.lower()
		#w = w.replace("-", "_").lower()
		#w = "".join(["#" if char.isdigit() else char for char in w])
		if w in self.word2id:
			return [self.word2id[w]]
		elif "".join([c for c in w if c.isalnum()]) in self.word2id:
			return [self.word2id["".join([c for c in w if c.isalnum()])]]

		elif not w.isalnum():
			if w.startswith("http://") or w.startswith("https://"):
				return [self.word2id["url"]]
			elif w.startswith("www."):
				return [self.word2id["url"]]
			split = "".join([c if c.isalnum() else " " for c in w]).split()
			if len(split) > 3:
				print (w)
			if not split:
				return [self.word2id["__UNKNOWN__"]]
			if split[-1] not in self.word2id:
				self.oov_file.write(w + "\n")
				return [self.word2id["__UNKNOWN__"]]
			tmp = []
			for subword in split:
				if len(subword) == 0:
					continue
				if subword in self.word2id:
					tmp.append(self.word2id[subword])
				else:
					self.oov_file.write(w + "\t" + subword + "\n")
					tmp.append(self.word2id["__UNKNOWN__"])
			return tmp
		else:
			self.oov_file.write(w + "\n")
			return [self.word2id["__UNKNOWN__"]]

	def tokenize(self, line):
		sent = line.strip().split()[2:-1]

		for i, w in enumerate(sent):
			if w == "<e1>":
				context_left = sent[:i]
				start_e1 = i + 1
				break
		i = start_e1
		for w in sent[start_e1:]:
			if w == "</e1>":
				e1 = sent[start_e1:i]
				end_e1 = i + 1
				break					
			i += 1
		i = end_e1
		for w in sent[end_e1:]:
			if w == "<e2>":
				context_mid = sent[end_e1:i]
				start_e2 = i + 1
				break
			i += 1
		i = start_e2
		for w in sent[start_e2:]:
			if w == "</e2>":
				e2 = sent[start_e2:i]
				end_e2 = i + 1
			i += 1
		context_right = sent[end_e2:]
		return context_left, e1, context_mid, e2, context_right


	def read_records(self, fn):
		with open(self.labels_file) as labs:
			labels = {lab.strip(): i for i, lab in enumerate(labs)}
		invs = [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,16,18,17]

		with open(self.labels_file) as labs:
			labels_inv = {lab.strip():i for i, lab in zip(invs, labs)}

		counter = 0
		X, y = [], []
		X_queries, X_positions, X_position_of_queries = [], [], []
		sentences = []
		if "train" in fn:
			outfile = open("tokenized_and_lookup_kbp37.txt", "w")

		with open(fn) as infile:
			for line in infile:
				if counter % 4 == 0:
					context_left, e1, context_mid, e2, context_right = self.tokenize(line)
					flag_pass = False

					if not e1 or not e2:
						counter += 1
						flag_pass = True
						continue
					tmp = context_left + e1 + context_mid + e2 + context_right
					tmp = tmp[:-1]
					word_ids = []
					sent_info = []
					for w in context_left:
						"""
						if w == "'s":
							continue
						if not w.replace("-","").isalnum():
							if len(w) > 2:
								print(w)
							continue
						"""
						word_ids += self.lookup(w)
					sent_info.append(len(word_ids))
					for w in e1:
						word_ids += self.lookup(w)
					sent_info.append(len(word_ids))
					# if len(context_left) == 5 and len(e1) == 2, their combined length is 7
					# and the last word so far is the last word of the query which is at position 7
					qp_1 = len(word_ids)
					for w in context_mid:
						word_ids += self.lookup(w)
					sent_info.append(len(word_ids))
					for w in e2:
						word_ids += self.lookup(w)
					sent_info.append(len(word_ids))
					qp_2 = len(word_ids)
					for w in context_right[:-1]:
						word_ids += self.lookup(w)
					sent_info.append(len(word_ids))

					ids_e1 = []
					for w in e1:
						ids_e1 += self.lookup(w)
					if ids_e1[-1] != self.word2id["__UNKNOWN__"]:
						q1 = np.sum([self.embs[w] for w in ids_e1 if w != self.word2id["__UNKNOWN__"]], axis=0)
					else:
						q1 = np.sum([self.embs[w] for w in word_ids[max(0, qp_1 - len(ids_e1) - 3): qp_1+4] if w != self.word2id["__UNKNOWN__"]], axis=0)

					ids_e2 = []
					for w in e2:
						ids_e2 += self.lookup(w)
					if ids_e2[-1] != self.word2id["__UNKNOWN__"]:
						q2 = np.sum([self.embs[w] for w in ids_e2 if w != self.word2id["__UNKNOWN__"]], axis=0)
					else:
						q2 = np.sum([self.embs[w] for w in word_ids[max(0, qp_2 - len(ids_e2) - 3): qp_2+4] if w != self.word2id["__UNKNOWN__"]], axis=0)

					sentences.append((word_ids, sent_info))

					queries = [np.squeeze(q1), np.squeeze(q2)]
					"""
					if np.shape(queries) != (2,50):
						print (np.shape(queries), np.shape(q1), np.shape(np.squeeze(q2)), e1, e2)
						sys.exit(0)
					"""
					query_positions = [qp_1, qp_2]
					#query_positions = [1, len(tmp) + 2]
					#print (tmp, positions, [e1, e2], query_positions)
					tmp = word_ids



					if self.FLAGS.use_whole_sentence == "no":
						# TO GET ONLY [E1, ..., E2]
						#tmp = word_ids[qp_1 - 1:qp_2]
						tmp = word_ids[qp_1:qp_2-len(ids_e2)]
						query_positions = [1, len(tmp)]
						# TO GET ALL, COMMENT AROUND THIS BLOCK

					#positions = list(range(1,len(tmp) + 1))
					positions = list(range(2,len(tmp)))				
					#print (tmp, positions, queries, query_positions)
					#
					X.append(tmp)
					if "train" in fn:
						outfile.write(line + " ".join([self.id2word[w] for w in tmp]) + "\n")
					X_positions.append(positions)
					X_queries.append(queries)
					X_position_of_queries.append(query_positions)
					#


				elif counter % 4 == 1:
					label = line.strip()
					if flag_pass:
						counter += 1
						continue
					#print (label, fn))
					y.append(self.create_one_hot(len(labels), labels[label]))
					"""
					if "train" in fn.lower():
					#if "train" in fn.lower() and counter < 4 * 7500:
						if label != "Other":
							y.append(self.create_one_hot(len(labels), labels_inv[label]))
							#queries = list(reversed(queries))
							queries = [q2, q1]
							query_positions = list(reversed(query_positions))
							X.append(tmp)
							X_positions.append(positions)
							X_queries.append(queries)
							X_position_of_queries.append(query_positions)
					"""
				counter += 1
		print (np.shape(y), np.shape(X_queries), np.shape(X_position_of_queries))
		return X, np.asarray(y), X_positions, np.asarray(X_queries), np.asarray(X_position_of_queries), sentences

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
	flags.DEFINE_string("use_whole_sentence", "no", "")

	FLAGS = flags.FLAGS
	preprocessing = preprocessing_kbp37(FLAGS)
	for i in range(5):
		print ([w for (i,w) in np.ndenumerate(preprocessing.train[0][i]) if preprocessing.id2word[w] != "__PADDING__"])
		print ([preprocessing.id2word[w] for (i,w) in np.ndenumerate(preprocessing.train[0][i]) if preprocessing.id2word[w] != "__PADDING__"])
		print ([w for (i,w) in np.ndenumerate(preprocessing.train[2][i]) if w != 0])
		#print ([preprocessing.id2word[w] for (i,w) in np.ndenumerate(preprocessing.train[3][i])])
		print ([w for (i,w) in np.ndenumerate(preprocessing.train[4][i])])
	#print ([p for (i,p) in np.ndenumerate(preprocessing.train[1][:5]) if p != 0])
	#print (preprocessing.train[0][:5])
