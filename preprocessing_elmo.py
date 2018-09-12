import numpy as np
import tensorflow as tf
from nltk import word_tokenize
import time
from collections import defaultdict
import re

from allennlp.modules.elmo import Elmo, batch_to_ids
import allennlp
from allennlp.commands.elmo import ElmoEmbedder
import tensorflow_hub as hub
class preprocessing:
	def __init__(self, FLAGS):

		# debug locally
		#self.options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
		#self.weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
		#self.ee = ElmoEmbedder()

		"""
		with tf.Graph().as_default() as g:
			self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
			sentences = tf.placeholder(dtype=tf.string, shape=(None, None))
			batch_size = 1
			embeddings = self.elmo(sentences)
		"""

		self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
    

		"""
		self.record_train = "/home/dominik/Documents/DFKI/data/TRAIN_FILE.TXT"
		self.record_test = "/home/dominik/Documents/DFKI/data/TEST_FILE_FULL.TXT"
		self.labels_file = "/home/dominik/Documents/DFKI/data/labels_semeval2010.txt"

		"""

		self.record_train = "/raid/data/dost01/semeval10_data/TRAIN_FILE.TXT"
		self.record_test = "/raid/data/dost01/semeval10_data/TEST_FILE_FULL.TXT"
		self.labels_file = "/raid/data/dost01/semeval10_data/labels.txt"




		self.oov_file = open("OOVS.txt", "w")


		self.FLAGS = FLAGS




		# X, np.asarray(y), X_positions, X_q1, X_q2, Pos_q1, Pos_q2, sentences
		self.train = self.read_records(self.record_train)
		#self.train[0] = self.train[0][:2]
		max_len = max(len(x) for x in self.train[0])
		seq_lengths = [len(x) for x in self.train[0]]
		sents = [x + [""] * (max_len - len(x)) for x in self.train[0]]
		#sents = tf.keras.preprocessing.sequence.pad_sequences(self.train[0],padding="post", value = "")

		config = tf.ConfigProto(allow_soft_placement = True)

		for batch in range(80):
			print (batch)
			batch_sents = sents[batch * 100:(batch + 1) * 100]
			batch_seq_lengths = seq_lengths[batch * 100:(batch + 1) * 100]
			self.embedded = self.elmo(inputs={"tokens": batch_sents, "sequence_len":batch_seq_lengths}, signature="tokens", as_dict=True)["elmo"]
			with tf.Session(config=config) as sess:
				sess.run(tf.global_variables_initializer())
				if batch == 0:
					embedded = sess.run(self.embedded)
				else:
					batch_embedded = sess.run(self.embedded)
					embedded = np.concatenate((embedded, batch_embedded), axis=0)

			print (np.shape(embedded))


		pos = tf.keras.preprocessing.sequence.pad_sequences(self.train[2], padding="post")
		max_len_q1 = max(len(x) for x in self.train[5])
		max_len_q2 = max(len(x) for x in self.train[6])
		maxlen = max(max_len_q1, max_len_q2)
		x_q1 = [x[pos[0]:pos[1]] for x, pos in zip(embedded, self.train[3])]
		x_q2 = [x[pos[0]:pos[1]] for x, pos in zip(embedded, self.train[4])]


		x_q1 = np.array([np.concatenate((x, np.zeros((maxlen - len(x), 1024))), axis=0) for x in x_q1])
		x_q2 = np.array([np.concatenate((x, np.zeros((maxlen - len(x), 1024))), axis=0) for x in x_q2])

		#x_q1 = tf.keras.preprocessing.sequence.pad_sequences(x_q1, padding="post", maxlen=maxlen)
		#x_q2 = tf.keras.preprocessing.sequence.pad_sequences(x_q2, padding="post", maxlen=maxlen)

		pos_q1 = tf.keras.preprocessing.sequence.pad_sequences(self.train[5], padding="post", maxlen=maxlen)
		pos_q2 = tf.keras.preprocessing.sequence.pad_sequences(self.train[6], padding="post", maxlen=maxlen)

		#self.max_len_e1 = len(x_q1[0])
		self.max_len_e1 = maxlen
		length = len(embedded[0])
		for i, x in zip(seq_lengths, embedded):
			x[i:,] = np.zeros((length-i, 1024))
		xs = embedded
		self.max_length = len(xs[0])
		qs = np.concatenate((x_q1, x_q2),axis=1)
		pos_qs = np.concatenate((pos_q1, pos_q2), axis=1)

		self.train = (xs, self.train[1], pos, qs, pos_qs)

		with open("embedded_train.txt", "w") as outfile:
			for x, label, p, q, pos_q in zip(self.train[0], self.train[1], self.train[2], self.train[3], self.train[4]):
				outfile.write("#".join([" ".join(map(str,i)) for i in x]) + "\t" + " ".join(map(str,label)) + "\t" + " ".join(map(str,p)) + "\t" + "#".join([" ".join(map(str,i)) for i in q]) + "\t" + " ".join(map(str,pos_q)) + "\n")


		# DEV SEST HERE
		"""
		self.test = [x[-500:] for x in self.train]
		self.train = [x[:-500] for x in self.train]
		self.sentences_info = self.test[-1]
		"""


		self.test = self.read_records(self.record_test)	


		max_len = max(len(x) for x in self.test[0])
		seq_lengths = [len(x) for x in self.test[0]]
		sents = [x + [""] * (max_len - len(x)) for x in self.test[0]]

		#self.embedded = self.elmo(inputs={"tokens": sents, "sequence_len":seq_lengths}, signature="tokens", as_dict=True)["elmo"]

		for batch in range(28):
			batch_sents = sents[batch * 100:(batch + 1) * 100]
			batch_seq_lengths = seq_lengths[batch * 100:(batch + 1) * 100]
			self.embedded = self.elmo(inputs={"tokens": batch_sents, "sequence_len":batch_seq_lengths}, signature="tokens", as_dict=True)["elmo"]
			with tf.Session(config=config) as sess:
				sess.run(tf.global_variables_initializer())
				if batch == 0:
					embedded = sess.run(self.embedded)
				else:
					batch_embedded = sess.run(self.embedded)
					embedded = np.concatenate((embedded, batch_embedded), axis=0)

		pos = tf.keras.preprocessing.sequence.pad_sequences(self.test[2], padding="post")
		x_q1 = [x[pos[0]:pos[1]] for x, pos in zip(embedded, self.test[3])]
		x_q2 = [x[pos[0]:pos[1]] for x, pos in zip(embedded, self.test[4])]

		x_q1 = np.array([np.concatenate((x, np.zeros((maxlen - len(x), 1024))), axis=0) for x in x_q1])
		x_q2 = np.array([np.concatenate((x, np.zeros((maxlen - len(x), 1024))), axis=0) for x in x_q2])
		#x_q1 = tf.keras.preprocessing.sequence.pad_sequences(x_q1, padding="post", maxlen=maxlen)
		#x_q2 = tf.keras.preprocessing.sequence.pad_sequences(x_q2, padding="post", maxlen=maxlen)

		pos_q1 = tf.keras.preprocessing.sequence.pad_sequences(self.test[5], padding="post", maxlen=maxlen)
		pos_q2 = tf.keras.preprocessing.sequence.pad_sequences(self.test[6], padding="post", maxlen=maxlen)

		qs = np.concatenate((x_q1, x_q2),axis=1)
		pos_qs = np.concatenate((pos_q1, pos_q2), axis=1)
		length = len(embedded[0])
		for i, x in zip(seq_lengths, embedded):
			x[i:,] = np.zeros((length-i, 1024))

		xs = embedded

		self.test = (xs, self.test[1], pos, qs, pos_qs)

		with open("embedded_test.txt", "w") as outfile:
			for x, label, p, q, pos_q in zip(self.test[0], self.test[1], self.test[2], self.test[3], self.test[4]):
				outfile.write("#".join([" ".join(map(str,i)) for i in x]) + "\t" + " ".join(map(str,label)) + "\t" + " ".join(map(str,p)) + "\t" + "#".join([" ".join(map(str,i)) for i in q]) + "\t" + " ".join(map(str,pos_q)) + "\n")


		print ([np.shape(x) for x in self.train])
		print ([np.shape(x) for x in self.test])
		#print (self.sentences_info)

	def tokenize(self, line):
		sent = line.strip().split("\t")[1][1:-1]
		sent = word_tokenize(sent)

		# remove html tags and seperate in "[context_left] [e1] [context_mid] [e2] [context_right]

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
		"""
		if len(e1) > 1:
			print (e1)
		if len(e2) > 2:
			print (e2)
		"""
		print (context_left, e1, context_mid, e2, context_right)
		return context_left, e1, context_mid, e2, context_right

	def read_records(self, fn):
		with open(self.labels_file) as labs:
			labels = {lab.strip(): i for i, lab in enumerate(labs)}
		invs = [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,16,18,17]

		with open(self.labels_file) as labs:
			labels_inv = {lab.strip():i for i, lab in zip(invs, labs)}

		counter = 0
		X, y = [], []
		X_positions = []
		X_q1, X_q2, Pos_q1, Pos_q2 = [], [], [], []
		X_queries, X_positions, X_position_of_queries = [], [], []
		sentences = []
		if "TRAIN" in fn:
			outfile = open("tokenized_and_lookup.txt", "w")

		len_e1 = defaultdict(int)
		len_e2 = defaultdict(int)
		with open(fn) as infile:
			for line in infile:
					
				if counter % 4 == 0:
					context_left, e1, context_mid, e2, context_right = self.tokenize(line)
					start_e1, end_e1, start_e2, end_e2 = len(context_left), len(context_left) + len(e1), len(context_left) + len(e1) + len(context_mid), len(context_left) + len(e1) + len(context_mid) + len(e2)
					sent = context_left + e1 + context_mid + e2 + context_right

					#embedded = self.ee.embed_sentence(sent)[2]
					#q1 = embedded[start_e1:end_e1]
					#q2 = embedded[start_e2:end_e2]
					positions = list(range(1, len(sent) + 1))
					pos_q1 = list(range(start_e1, end_e1 + 1))
					pos_q2 = list(range(start_e2, end_e2 + 1))
					X.append(sent)
					X_positions.append(positions)
					X_q1.append([start_e1, end_e1])
					X_q2.append([start_e2, end_e2])
					Pos_q1.append(pos_q1)
					Pos_q2.append(pos_q2)

				elif counter % 4 == 1:
					"""
					if flag_pass:
						counter += 1
						continue
					"""
					label = line.strip()
					#if pass_flag:
					#	counter += 1
					#	continue
					#print (label, fn))

					y.append(self.create_one_hot(len(labels), labels[label]))

					"""

					if "TRAIN" in fn:
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
		print (len_e1, len_e2)
		#print (np.shape(y), np.shape(X_queries), np.shape(X_position_of_queries))
		return X, np.asarray(y), X_positions, X_q1, X_q2, Pos_q1, Pos_q2, sentences

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
	flags.DEFINE_integer("window_size", 2, "")

	FLAGS = flags.FLAGS
	preprocessing = preprocessing(FLAGS)
	for i in range(5):
		print ([w for (i,w) in np.ndenumerate(preprocessing.train[0][i]) if preprocessing.id2word[w] != "__PADDING__"])
		print ([preprocessing.id2word[w] for (i,w) in np.ndenumerate(preprocessing.train[0][i]) if preprocessing.id2word[w] != "__PADDING__"])
		print ([w for (i,w) in np.ndenumerate(preprocessing.train[2][i]) if w != 0])
		#print ([preprocessing.id2word[w] for (i,w) in np.ndenumerate(preprocessing.train[3][i])])
		print ([w for (i,w) in np.ndenumerate(preprocessing.train[4][i])])
	#print ([p for (i,p) in np.ndenumerate(preprocessing.train[1][:5]) if p != 0])
	#print (preprocessing.train[0][:5])
