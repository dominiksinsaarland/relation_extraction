'''
loosely inspired by
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

import tensorflow as tf
import os
import numpy as np
import ast
from nltk.stem import WordNetLemmatizer
import re
import string
import time
from collections import defaultdict
import random
from nltk import word_tokenize

# hyperparameter set from tensor2tensor, important filter_size (hidden layer of feedforward layers): 4 * embedding size

"""
def transformer_tiny():
  hparams = transformer_base()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 128
  hparams.filter_size = 512
  hparams.num_heads = 4
  return hparams
"""

	

flags = tf.flags

#flags.DEFINE_float("learning_rate", 0.001, "") // use adaptive learning rate instead, see explanations below
flags.DEFINE_integer("num_labels", 19, "number of target labels")
flags.DEFINE_integer("batch_size", 200, "number of batchsize, bigger works better")
flags.DEFINE_float("dropout", 0.1, "dropout applied after each layer")
flags.DEFINE_integer("sent_length", 50, "sentence length")
flags.DEFINE_integer("num_layers", 1, "num layers for encoding/decoding")
flags.DEFINE_integer("num_heads",1, "num heads per layer")
flags.DEFINE_integer("num_epochs",50, "")
flags.DEFINE_integer("min_length", 0, "min length of encoded sentence")
flags.DEFINE_integer("embeddings_dim", 300, "number of dimensions in word embeddings")
flags.DEFINE_integer("warmup_epochs", 5, "")
FLAGS = flags.FLAGS

# path to files


record_train = "/raid/data/dost01/semeval10_data/TRAIN_FILE.TXT"
record_test = "/raid/data/dost01/semeval10_data/TEST_FILE_FULL.TXT"
labels_file = "/raid/data/dost01/semeval10_data/labels.txt"
#vocab_file = "/raid/data/dost01/embeddings/numberbatch-en-17.06.txt"

#record_train = "/raid/data/dost01/semeval10_data/train_glove.txt"
#record_test = "/raid/data/dost01/semeval10_data/test_glove.txt"

vocab_file = "/raid/data/dost01/embeddings/glove.6B.300d.txt"
"""

# run locally on CPU
#vocab_file = "../gridsearch/features/numberbatch-en-17.06.txt"

record_train = "/home/dominik/Documents/DFKI/Hiwi-master/NemexRelator2010/SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT"
record_test = "/home/dominik/Documents/DFKI/Hiwi-master/NemexRelator2010/SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT"

vocab_file = "/home/dominik/Documents/Supertagging/glove.6B.300d.txt"



#record_train = "train_glove.txt"
#record_test = "test_glove.txt"
labels_file = "labels.txt"
"""








# stalin: the next few functions are not important to you, just utils for relation classification
def macro_f1(y_true, y_pred):
	"""
	evaluate macro f1 (prodcues the same results as the official scoring file "semeval2010_task8_scorer-v1.2.pl"
	ignores the Other class
	returns pr, rc and f1 rounded to two decimal points as strings
	"""

	OTHER = "Other"
	d = defaultdict(int)
	for i,j in zip(y_true, y_pred):
		if i == j:
			d[i.split("(")[0] +"_TP"] += 1
		else:
			d[j.split("(")[0] + "_FP"] += 1
			d[i.split("(")[0] + "_FN"] += 1
	TP = 0
	FP = 0
	FN = 0
	items = set()
	for key in d:
		items.add(key.split("_")[0])
	items.remove(OTHER)
	pr, rc, f = 0,0,0
	for item in items:
		t_pr = d[item + "_TP"] / (d[item + "_TP"] + d[item + "_FP"])
		t_rc = d[item + "_TP"] / (d[item + "_TP"] + d[item + "_FN"])
		pr += t_pr
		rc += t_rc
		if t_pr + t_rc == 0:
			print (len(items))
			continue
		f += 2 * t_pr * t_rc / (t_pr + t_rc)
	Pr = pr /len(items)
	Rc = rc / len(items)
	F1 = f / len(items)
	return "{0:.2f}".format(Pr * 100), "{0:.2f}".format(Rc * 100), "{0:.2f}".format(F1 * 100)

"""
def get_vocab(record_train):
	#vocab = {}
	#vocab_counter = 1
	vocab_counts = defaultdict(int)
	with open(record_train) as infile:
		for line in infile:
			line = line.strip().split("\t")
			sent = ast.literal_eval(line[0])
			for w in sent:
				vocab_counts[w] += 1
				#if w not in vocab:
				#	vocab[w] = vocab_counter
				#	vocab_counter += 1
	vocab = {}
	vocab_counter = 1
	for w in vocab_counts:
		if vocab_counts[w] > 4:
			vocab[w] = vocab_counter
			vocab_counter += 1
	vocab["__UNKNOWN__"] = vocab_counter	
	return vocab

def read_embeddings(fn):
	vocab = get_vocab(record_train)
	embs = np.zeros((len(vocab) + 2, FLAGS.embeddings_dim))
	covered = set()
	with open(fn) as f:
		for line in f:
			split = line.strip().split()
			# header line
			if len(split) == 2:
				continue
			w = split[0]
			if w not in vocab:
				continue
			else:
				word, vec = split[0], [float(val) for val in split[1:]]
				#vocab[word] = vocab_counter
				embs[vocab[word],:] = vec
				covered.add(word)
	embs[vocab["__UNKNOWN__"],:] = np.random.random(FLAGS.embeddings_dim)
	for w in vocab:
		if w not in covered:
			vocab[w] = vocab["__UNKNOWN__"]
	return embs, vocab
	
"""
def read_embeddings(fn):
	"""
	read embeddings and store each word in the rows of the matrix at its index
	returns the embedding matrix
	first row is 0 vector for padding, second last row is __UNKNOWN__ vector 
	"""
	counter = 0
	vocab_counter = 1
	vocab = {}
	# number of words in embedding file + 2, first is __PADDING__, last is __UNKNOWN__ token
	embs = np.zeros((417194 + 2, FLAGS.embeddings_dim))
	with open(fn) as f:
		for line in f:
			"""
			counter += 1
			if counter > 1000:
				break
			"""
			split = line.strip().split()
			# header line
			if len(split) == 2:
				continue
			else:
				word, vec = split[0], [float(val) for val in split[1:]]
				vocab[word] = vocab_counter
				embs[vocab_counter,:] = vec
				vocab_counter += 1
	vocab["__PADDING__"] = 0
	vocab["__UNKNOWN__"] = vocab_counter
	embs[vocab_counter,:] = np.random.random(FLAGS.embeddings_dim)
	return embs, vocab
	
def create_one_hot(length, x):
	# create one hot vector
	zeros = np.zeros(length, dtype=np.int32)
	zeros[x] = 1
	return zeros

def read_records(fn, vocab, labels_file):
	with open(labels_file) as labs:
		labels = {lab.strip(): i for i, lab in enumerate(labs)}
	invs = [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,16,18,17]

	with open(labels_file) as labs:
		labels_inv = {lab.strip():i for i, lab in zip(invs, labs)}

	"""
	with open(labels_file) as labs:
		labels_inv = {}
		for i, lab in enumerate(labs):
			if i == 16:
				labels_inv[lab] = 16
			if i == 17:
				labels_inv[lab] = 19
			if i == 18:
				labels_inv[lab] = 18
			

			labels 
	"""
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
					if w in vocab:
						word_ids.append(vocab[w])
					else:
						word_ids.append(vocab["__UNKNOWN__"])
				qp_1 = len(word_ids) + 1
				for w in e1:
					if w == "'s":
						continue
					if not w.isalnum():
						continue
					w = w.replace("-", "_").lower()
					w = "".join(["#" if char.isdigit() else char for char in w])
					if w in vocab:
						word_ids.append(vocab[w])
					else:
						word_ids.append(vocab["__UNKNOWN__"])
				for w in context_mid:
					if w == "'s":
						continue
					if not w.isalnum():
						continue
					w = w.replace("-", "_").lower()
					w = "".join(["#" if char.isdigit() else char for char in w])
					if w in vocab:
						word_ids.append(vocab[w])
					else:
						word_ids.append(vocab["__UNKNOWN__"])

				qp_2 = len(word_ids) + 1
				for w in e2:
					if w == "'s":
						continue
					if not w.isalnum():
						continue
					w = w.replace("-", "_").lower()
					w = "".join(["#" if char.isdigit() else char for char in w])
					if w in vocab:
						word_ids.append(vocab[w])
					else:
						word_ids.append(vocab["__UNKNOWN__"])
				for w in context_right:
					if w == "'s":
						continue
					if not w.isalnum():
						continue
					w = w.replace("-", "_").lower()
					w = "".join(["#" if char.isdigit() else char for char in w])
					if w in vocab:
						word_ids.append(vocab[w])
					else:
						word_ids.append(vocab["__UNKNOWN__"])


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
				y.append(create_one_hot(len(labels), labels[label]))

				if "train" in fn.lower():
				#if "train" in fn.lower() and counter < 4 * 7500:
					if label != "Other":
						y.append(create_one_hot(len(labels), labels_inv[label]))
						queries = list(reversed(queries))
						query_positions = list(reversed(query_positions))
						X.append(tmp)
						X_positions.append(positions)
						X_queries.append(queries)
						X_position_of_queries.append(query_positions)

			counter += 1
	return X, y, X_positions, X_queries, X_position_of_queries


def read_records_ace(fn, vocab, labels_file):
	with open(labels_file) as labs:
		labels = {lab.strip(): i for i, lab in enumerate(labs)}
	X,y = [], []
	X_queries, X_positions, X_position_of_queries = [], [], []
	positions = []
	lemmatizer = WordNetLemmatizer()
	with open(fn) as infile:
		for line in infile:
			# read line
			line = line.strip()
			line = ast.literal_eval(line)
			sent = line[1]
			pos1, pos2 = line[2] - 1, line[3] - 1
			e1, e2 = sent[pos1], sent[pos2]
			label = line[4]
			tmp = []
			flag = False
			for w in sent:
				w = w.replace("-", "_")
				w = "".join(["#" if char.isdigit() else char for char in w])
				if w in vocab:
					tmp.append(vocab[w])
				else:
					tmp.append(vocab["__UNKNOWN__"])

			if e1 in vocab:
				q1 = vocab[e1]
			else:
				q1 = vocab["__UNKNOWN__"]
			if e2 in vocab:
				q2 = vocab[e2]
			else:
				q2 = vocab["__UNKNOWN__"]
			queries = [q1, q2]
			positions = [i + 1 for i in range(len(tmp))]
			query_positions = [pos1 + 1, pos2 + 1]

def read_records_old(fn, vocab, labels_file, sent_length=12):
	"""
	reads the train and test file and vectorizes sequences with word indexes
	returns X and y
	"""
	with open(labels_file) as labs:
		labels = {lab.strip(): i for i, lab in enumerate(labs)}
	invs = [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,16,18,17]

	with open(labels_file) as labs:
		labels_inv = {lab.strip():i for i, lab in zip(invs, labs)}

	
	X,y = [], []
	X_queries, X_positions, X_position_of_queries = [], [], []
	positions = []
	lemmatizer = WordNetLemmatizer()

	with open(fn) as infile:
		for line in infile:
			# read line
			line = line.strip().split("\t")
			sent = ast.literal_eval(line[0])
			es = ast.literal_eval(line[1])
			e1, e2 = es[0], es[1]
			pos = ast.literal_eval(line[2])
			pos1, pos2 = pos[0], pos[1]
			label = line[-1]
			tmp = []

			for w in sent:
				if w in vocab:
					tmp.append(vocab[w])
				else:
					tmp.append(vocab["__UNKNOWN__"])
			if e1 in vocab:
				q1 = vocab[e1]
			else:
				q1 = vocab["__UNKNOWN__"]
			if e2 in vocab:
				q2 = vocab[e2]
			else:
				q2 = vocab["__UNKNOWN__"]
			queries = [q1, q2]
			#tmp.pop(pos2)
			#tmp.pop(pos1)
			query_positions = [pos1 + 1, pos2 + 1]
			#positions = [i + 1 for i in range(pos1)] + [i + 2 for i in range(pos1, pos2 - 1)] + [i + 2 for i in range(pos2, len(sent) - 1)]
			positions = [i + 1 for i in range(len(tmp))]

			if len(tmp) > 30:
				continue

			X.append(tmp)
			y.append(create_one_hot(len(labels), labels[label]))
			X_positions.append(positions)
			X_queries.append(queries)
			X_position_of_queries.append(query_positions)
			if "train" in fn and counter < 4 * 7500:
				if label == "Other":
					continue
				queries = list(reversed(queries))
				query_positions = list(reversed(query_positions))
				y.append(create_one_hot(len(labels), labels_inv[label]))
				X.append(tmp)
				X_positions.append(positions)
				X_queries.append(queries)
				X_position_of_queries.append(query_positions)

			
			if len(positions) != len(tmp):
				print (tmp, positions, queries, query_positions)
				sys.exit(0)
	"""
	with open(fn) as infile:
		for line in infile:
			# read line
			line = line.strip().split("\t")
			sent = ast.literal_eval(line[0])
			es = ast.literal_eval(line[1])
			e1, e2 = es[0], es[1]
			pos = ast.literal_eval(line[2])
			pos1, pos2 = pos[0], pos[1]
			label = line[-1]
			tmp = []
			flag = False
			for w in sent[pos1 + 1: pos2]:
				if w in vocab:
					tmp.append(vocab[w])
				else:
					tmp.append(vocab["__UNKNOWN__"])
			if e1 in vocab:
				q1 = vocab[e1]
			else:
				q1 = vocab["__UNKNOWN__"]
			if e2 in vocab:
				q2 = vocab[e2]
			else:
				q2 = vocab["__UNKNOWN__"]
			queries = [q1, q2]
			if len(tmp) >= FLAGS.sent_length:
				tmp = tmp[:FLAGS.sent_length]
				flag = True
			if len(tmp) >= FLAGS.min_length:
				flag = True
			if flag:
				# nothing to do, just create position embeddings and query positions
				query_positions = [1, len(tmp) + 2]
				positions = [i + 2 for i in range(len(tmp))]
			else:
				# fill the sentence with left context of [e1] and right context of [e2]
				to_fill = FLAGS.min_length - len(tmp)
				left_context = []
				right_context = []
				# if there is enough left and right context, just take from both sides
				if pos1 >= to_fill // 2 and len(sent) - 1 - pos2 >= to_fill // 2:
					for w in sent[pos1 - to_fill // 2:pos1]:
						if w in vocab:
							left_context.append(vocab[w])
						else:
							left_context.append(vocab["__UNKNOWN__"])
					for w in sent[pos2 + 1: pos2 + 1 + to_fill//2]:
						if w in vocab:
							right_context.append(vocab[w])
						else:
							right_context.append(vocab["__UNKNOWN__"])
				# if there is not enough left context (e1 is at the start or first token of sentence):
				# take all left context there is and fill up with right context of [e2]
				elif pos1 < to_fill // 2:
					for w in sent[:pos1]:
						if w in vocab:
							left_context.append(vocab[w])
						else:
							left_context.append(vocab["__UNKNOWN__"])
					end = FLAGS.min_length - len(tmp) - len(left_context)
					end = min(end, len(sent))
					for w in sent[pos2 + 1: 1 + end + pos2]:
						if w in vocab:
							right_context.append(vocab[w])
						else:
							right_context.append(vocab["__UNKNOWN__"])
				# if there is not enough right context, fill up and add as much left context as needed
				elif len(sent) - 1 - pos2 < to_fill // 2:
					#end = to_fill - len(left_context)
					for w in sent[pos2 + 1:]:
						if w in vocab:
							right_context.append(vocab[w])
						else:
							right_context.append(vocab["__UNKNOWN__"])
					start = pos1 - FLAGS.min_length + len(tmp) + len(right_context)
					start = max(0, start)
					for w in sent[start:pos1]:
						if w in vocab:
							left_context.append(vocab[w])
						else:
							left_context.append(vocab["__UNKNOWN__"])
				else:
					print (len(sent), pos1, pos2, tmp)
					sys.exit(0)
				positions = [i + 1 for i in range(len(left_context))]
				positions = positions + [i + 2 + len(left_context) for i in range(len(tmp))]
				positions = positions + [i + 3 + len(left_context) + len(tmp) for i in range(len(right_context))]
				query_positions = [len(left_context) + 1, len(left_context) + len(tmp) + 2]
				tmp = left_context + tmp + right_context
			X.append(tmp)
			y.append(create_one_hot(len(labels), labels[label]))
			X_positions.append(positions)
			X_queries.append(queries)
			X_position_of_queries.append(query_positions)
	"""
	return X, y, X_positions, X_queries, X_position_of_queries

# stalin: things start to become interesting from here!
def position_lookup(x_train, num_units):
	"""
	from the paper:
	In this work, we use sine and cosine functions of different frequencies:
	PE_(pos,2i) = sin(pos/10000 2i/d_model)
	PE_(pos,2i+1) = cos(pos/10000 2i/d_model)
	where pos is the position and i is the dimension. That is, each dimension of the positional encoding
	corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We
	chose this function because we hypothesized it would allow the model to easily learn to attend by
	relative positions, since for any fixed offset k, P E pos+k can be represented as a linear function of
	PE pos .
	"""
	"""
	max_length = max([len(x) for x in x_train]) + 2 
	position_enc = np.array([[pos / np.power(10000, 2.*i/num_units) for i in range(num_units)] for pos in range(max_length)])
	position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
	position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
	# add padding token at row 0, just np.zeros
	position_enc = np.concatenate((np.expand_dims(np.zeros(num_units), 0), position_enc), axis=0)
	return position_enc, max_length
	"""

		#self.position_enc = np.array([np.repeat([pos / np.power(10000, 2*i/self.FLAGS.embeddings_dim) for i in range(self.FLAGS.embeddings_dim // 2)], 2) for pos in range(1, self.preprocessing.max_length + 2)])

	max_length = max([len(x) for x in x_train]) + 2 
	position_enc = np.array([np.repeat([pos / np.power(10000, 2.*i/num_units) for i in range(num_units//2)], 2) for pos in range(1,max_length+1)])
	position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
	position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
	# add padding token at row 0, just np.zeros
	position_enc = np.concatenate((np.expand_dims(np.zeros(num_units), 0), position_enc), axis=0)
	return position_enc, max_length




def pointwise_feedforward(inputs, scope, is_training=True):
	"""
	following section "3.3 Position-wise Feed-Forward Networks" in "attention is all you need":
	FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
	each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This
	consists of two linear transformations with a ReLU activation in between
	"""



	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		h1 = tf.layers.dense(inputs=inputs, units= 4 * FLAGS.embeddings_dim, name="feedforward", activation=tf.nn.relu)
		out = tf.layers.dense(inputs=h1, units= FLAGS.embeddings_dim, name="feedforward2")

	return out

def multihead_attention(queries, keys, scope="multihead_attention", num_units=FLAGS.embeddings_dim, num_heads=FLAGS.num_heads, is_training=True):
	"""
	following section "3.2.2 Multi-Head Attention":
	we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections...
	On each of these projected versions of queries, keys and values we then perform the attention function in parallel...
	These are concatenated and once again projected, resulting in the final values...
	MultiHead(Q, K, V ) = Concat(head_1 , ..., head_h )W_OUT
	where head_i = Attention(QW_i, KW_i, VW_i)
	and Attention is:
	alphas = softmax(matmul(Q, transpose(K)) / sqrt(len(dimensions_keys)))
	alphas is a matrix with dimensions sent_length * sent_length
	attention = matmul(alphas, V)
	attention is a matrix with dimensions sent_length * hidden_dims

	followed by the output projection
	matmul(concat(attentions), W_OUT)
	"""

	for head in range(num_heads):
		with tf.variable_scope(scope + "_" + str(head), reuse=tf.AUTO_REUSE):
			"""
			Q_weights = tf.Variable(tf.random_normal([300, num_units//num_heads]), trainable=True, name="queries_weights")
			K_weights = tf.Variable(tf.random_normal([300, num_units//num_heads]), trainable=True, name="keys_weights")
			V_weights = tf.Variable(tf.random_normal([300, num_units//num_heads]), trainable=True, name="values_weights")
			Q = tf.matmul(queries, Q_weights)
			K = tf.matmul(keys, K_weights, name="keys")
			V = tf.matmul(keys, V_weights)

			print (Q.get_shape(), K.get_shape(), V.get_shape())
			time.sleep(1)
			
			Q = tf.layers.dense(queries, num_units/num_heads, use_bias=False)
			K = tf.layers.dense(keys, num_units/num_heads, use_bias=False, name="keys")
			V = tf.layers.dense(keys, num_units/num_heads, use_bias=False)
			"""
			Q = tf.layers.dense(queries, num_units/num_heads, activation=tf.nn.relu, use_bias=False)
			K = tf.layers.dense(keys, num_units/num_heads, activation=tf.nn.relu, use_bias=False, name="keys")
			V = tf.layers.dense(keys, num_units/num_heads, activation=tf.nn.relu, use_bias=False)
			

			#if scope != "multihead_attention_decoder_self":
			#	K = tf.multiply(K, mask)
			#	V = tf.multiply(V, mask)

			# decoding step for relation classification
			# if num_heads = 3, num_units/num_heads = 300/3 = 100
			# Q = [batch_size, 2, 100] // because we only have two queries
			# K = [batch_size, sent_length, 100]
			# V = [batch_size, sent_length, 100]
			outputs = tf.matmul(Q, K, transpose_b=True)
			# outputs are [batch_size, 2, sent_length]
			# scaling
			outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)
			#mask_softmax = tf.where(tf.greater(outputs, tf.zeros_like(outputs)), x=outputs, y=tf.ones_like(outputs) * -100000)
			mask_softmax = tf.where(tf.equal(outputs, tf.zeros_like(outputs)), x=tf.ones_like(outputs) * -10000000, y=outputs)
			outputs = tf.nn.softmax(mask_softmax, name="attention_softmax")
			# rescale
			outputs = tf.matmul(outputs, V)
			# outputs are [batch_size, 2, 100]
			# concat intermediate results
			if head == 0:
				head_i = outputs
			else:
				head_i = tf.concat([head_i, outputs], axis=-1)
	with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
		multihead = tf.layers.dense(head_i, num_units, activation=tf.nn.relu, use_bias=False)
		#multihead = tf.layers.dense(head_i, num_units, use_bias=False)
	return multihead

def normalize(inputs, epsilon = 1e-6, scope="layer_norm", reuse=None):
	'''Applies layer normalization.
	
	Args:
	  inputs: A tensor with 2 or more dimensions, where the first dimension has
		`batch_size`.
	  epsilon: A floating number. A very small number for preventing ZeroDivision Error.
	  scope: Optional scope for `variable_scope`.
	  reuse: Boolean, whether to reuse the weights of a previous layer
		by the same name.
	  
	Returns:
	  A tensor with the same shape and data dtype as `inputs`.
	'''
	with tf.variable_scope(scope, reuse=reuse):
		inputs_shape = inputs.get_shape()
		params_shape = inputs_shape[-1:]
	
		mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
		print ("scope", scope)
		print ("layer norm shapes", mean.get_shape(), variance.get_shape())
		#beta= tf.Variable(tf.zeros(params_shape))
		#gamma = tf.Variable(tf.ones(params_shape))
		normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
		outputs = normalized		
	return outputs


def add_and_norm(old_inputs, inputs):
	"""
	We employ a residual connection [11] around each of
	the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
	LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
	itself
	"""
	inputs = tf.add(old_inputs, inputs)
	# I think this is what they do too in the paper, but not 100% sure!
	normalized = normalize(inputs)
	return normalized

# load the data
embs, vocab = read_embeddings(vocab_file)

x_train, y_train, x_positions_train, x_queries_train, x_position_of_queries_train = read_records(record_train, vocab, labels_file)
#x_test, y_test, x_positions_test, x_queries_test, x_position_of_queries_test = x_train[-500:], y_train[-500:], x_positions_train[-500:], x_queries_train[-500:], x_position_of_queries_train[-500:]
#x_train, y_train, x_positions_train, x_queries_train, x_position_of_queries_train = x_train[:-500], y_train[:-500], x_positions_train[:-500], x_queries_train[:-500], x_position_of_queries_train[:-500]


x_test, y_test, x_positions_test, x_queries_test, x_position_of_queries_test = read_records(record_test, vocab, labels_file)
result_file = open("dev_results.txt", "w")
id2word = {j:i for i,j in vocab.items()}
"""
for sent, lab, positions, x_queries, x_positions_queries in zip(x_test, y_test, x_positions_test, x_queries_test, x_position_of_queries_test):
	result_file.write(" ".join([id2word[w] for w in sent]) + "\n")
	result_file.write(str(np.argmax(lab)) + "\n")
	result_file.write(" ".join([str(i) for i in positions]) + "\n")
	result_file.write(" ".join([id2word[q] for q in x_queries]) + " " + " ".join([str(i) for i in x_positions_queries]) + "\n")
sys.exit(0)
"""
#

print (len(x_train), len(y_train), len(x_positions_train), len(x_queries_train), len(x_position_of_queries_train))
print (len(y_test))

# get position embeddings
pos_table, max_length = position_lookup(x_train, FLAGS.embeddings_dim)


# ___________________________________________
# FROM HERE, we start to define the model

# initialize some placeholders
# sentence to encode
x = tf.placeholder(tf.int32, shape=[None, None])
# labels
y = tf.placeholder(tf.int32, shape=[None, None])
# positions of tokens in sentence to encode
pos_sent = tf.placeholder(tf.int32, shape=[None, None])
# queries (= [e1, e2])
x_q = tf.placeholder(tf.int32, shape=[None, 2])
# position of queries
x_p = tf.placeholder(tf.int32, shape=[None, 2])

nonpadding = tf.to_float(tf.where(tf.equal(x, tf.zeros_like(x)), x=tf.zeros_like(x),y=tf.ones_like(x)))

# convert embedding matrices to tf.tensors
embeddings = tf.get_variable("embedding", np.shape(embs), initializer=tf.constant_initializer(embs),dtype=tf.float32, trainable=False)
positions = tf.get_variable("positions", np.shape(pos_table), initializer=tf.constant_initializer(pos_table), dtype=tf.float32, trainable=False)

# prepare encoder
"""

mask = tf.where(tf.equal(x, tf.zeros_like(x)))
masked_x = tf.boolean_mask(x, mask)
inputs = tf.nn.embedding_lookup(embeddings, masked_x)
pos_sent_masked = tf.boolean_mask(pos_sent, mask)
pos_inputs = tf.nn.embedding_lookup(positions, pos_sent_masked)
inputs = tf.add(inputs, pos_inputs)

"""


inputs = tf.nn.embedding_lookup(embeddings, x)
mask = tf.where(tf.equal(inputs, tf.zeros_like(inputs)), x=tf.zeros_like(inputs),y=tf.ones_like(inputs))
pos_inputs = tf.nn.embedding_lookup(positions, pos_sent)
inputs = tf.add(inputs, pos_inputs)

# mask padding tokens so we don't add biases to padding timesteps resulting in unexpected results
# if sent_length = 4, this returns [[1,1,...,1],[1,1,1,...,1],[1,1,1,...,1],[1,1,1,...,1],[0,0,0,...,0],...,[0,0,0...0]


# prepare decoder
x_ps = tf.nn.embedding_lookup(positions, x_p)
x_qs = tf.nn.embedding_lookup(embeddings, x_q)
decoder_input = tf.add(x_qs, x_ps)
# not sure wether to add layer normalization over input
#inputs = tf.contrib.layers.layer_norm(inputs)
#decoder_input = tf.contrib.layers.layer_norm(decoder_input)

# and maybe even a dropout
#decoder_input = tf.layers.dropout(decoder_input, rate=FLAGS.dropout)
#inputs = tf.layers.dropout(inputs, rate=FLAGS.dropout)

def encode_sentence(inputs, num_layers, num_heads, is_training=True, dropout_rate=0.1):
	# encoder layers stacked
	for layer in range(num_layers):
		with tf.variable_scope("encoder_layers_%d" % layer, reuse=tf.AUTO_REUSE):
			# attention step
			attention = multihead_attention(inputs, inputs, scope="multihead_attention", num_heads = num_heads, is_training=is_training)
			# residual connection
			"""
			We apply dropout [33] to the output of each sub-layer, before it is added to the
			sub-layer input and normalized. For the base model, we use a rate of P drop = 0.1.
			"""
			attention = tf.layers.dropout(attention, rate=dropout_rate, training=is_training)
			postprocess = add_and_norm(inputs, attention)

			feed_forward = pointwise_feedforward(postprocess, "feedforward_%d" % layer, is_training=is_training)
			feed_forward = tf.layers.dropout(feed_forward, rate=dropout_rate, training=is_training)

			inputs = add_and_norm(postprocess, feed_forward)
			inputs = tf.where(tf.equal(mask, tf.ones_like(mask)), x=inputs, y=tf.zeros_like(mask))
	return inputs

def decode_sentence(decoder_input, encoder_input, num_layers, num_heads, is_training=True, dropout_rate=0.1):
	"""
	In "encoder-decoder attention" layers, queries come from the previous decoder layer,
	and the memory keys and values come from the output of the encoder.
	"""

	for layer in range(num_layers):
		with tf.variable_scope("decoder_layers_%d" % layer, reuse=tf.AUTO_REUSE):
			# self attention first
			attention = multihead_attention(decoder_input, decoder_input, scope="multihead_attention_decoder_self", num_heads = num_heads, is_training=is_training)
			attention = tf.layers.dropout(attention, rate=dropout_rate, training=is_training)
			postprocess = add_and_norm(decoder_input, attention)
			print ("DECODER SHAPE!!", postprocess.get_shape())
			# then multi head attention over encoded input
			attention = multihead_attention(postprocess, encoder_input, scope="multihead_attention_decoder", is_training=is_training)
			attention = tf.layers.dropout(attention, rate=dropout_rate, training=is_training)
			postprocess = add_and_norm(postprocess, attention)			

			# followed by feedforward
			if layer != num_layers - 1:
				feed_forward = pointwise_feedforward(postprocess, "ffn_%d" % layer, is_training=is_training)
				feed_forward = tf.layers.dropout(feed_forward, rate=dropout_rate, training=is_training)
				decoder_input = add_and_norm(postprocess, feed_forward)
			else:
				#pool = tf.reduce_sum(postprocess, -1)
				#pool = tf.reshape(postprocess, [-1, FLAGS.embeddings_dim * 2])
				with tf.variable_scope("classify", reuse=tf.AUTO_REUSE):
					#h1 = tf.layers.dense(inputs=pool, units= 4 * FLAGS.embeddings_dim, name="feedforward", activation=tf.nn.relu)
					h1 = tf.layers.dense(inputs=postprocess, units= 4 * FLAGS.embeddings_dim, name="feedforward", activation=tf.nn.relu)
					h2 = tf.layers.dense(inputs=h1, units= FLAGS.embeddings_dim, name="feedforward2")
					h2 = tf.layers.dropout(h2, rate=dropout_rate, training=is_training)
					"""
					h2 = tf.layers.dense(inputs=pool, units= 450, name="feedforward2", activation=tf.nn.relu)
					h2 = tf.layers.dropout(h2, rate=dropout_rate, training=is_training)
					"""
					h2 = add_and_norm(postprocess, h2)

					pool = tf.reshape(h2, [-1, FLAGS.embeddings_dim * 2])
					h3 = tf.layers.dense(inputs=pool, units=200, activation=tf.nn.relu)
					h3 = tf.layers.dropout(h3, rate=dropout_rate, training=is_training)
					logits = tf.layers.dense(h3, units=19, name="out")
	return logits

# encoder decoder architecture
encoded = encode_sentence(inputs, FLAGS.num_layers, FLAGS.num_heads, dropout_rate=FLAGS.dropout)
print (encoded.get_shape())

logits = decode_sentence(decoder_input, encoded,  FLAGS.num_layers, FLAGS.num_heads, dropout_rate=FLAGS.dropout)


# that was the transformers...
# the rest takes care of the pecularities of classification task:
 # decoded is always tensor with [batch_size, 2, 300] because we have 2 queries 
# so I think it makes sense to reshape to [batch_size, 600] and feed this to classification layer



# cost function, minimizer and predictions
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * 0.005

cost += l2_losses
params = tf.trainable_variables()
gradients = tf.gradients(cost, params)

max_gradient_norm = 3
clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

# optimize the lossfunction

learning_rate = tf.placeholder(tf.float32, shape=[])

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,epsilon=1e-09)
#optimizer= tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

#optimizer =  tf.train.RMSPropOptimizer(0.001)
train_step = optimizer.apply_gradients(zip(clipped_gradients, params))

#inspect_softmax = tf.get_default_graph().get_tensor_by_name("decoder_layers_1/multihead_attention_decoder_0:softmax")
# maybe add l2 loss??
#cost = cost + l2_losses


#print ("\n".join([n.name for n in tf.get_default_graph().as_graph_def().node if "attention_softmax" in n.name or "keys" in n.name]))

#inspect_softmax = tf.get_default_graph().get_tensor_by_name("decoder_layers_0/multihead_attention_decoder_%d/attention_softmax:0" % (FLAGS.num_layers - 1))
#inspect_keys = tf.get_default_graph().get_tensor_by_name("decoder_layers_0/multihead_attention_decoder_0/keys/kernel:0")

"""
# optimizer following the paper
We used the Adam optimizer with beta1 = 0.9, beta2 = 0.98 and epsilon = 10e-9
lrate = d −0.5 * min(step_num^-0.5, stepnum* warmup_STEPS^-1.5)
This corresponds to increasing the learning rate linearly for the first warmup_STEPS training STEPS,
and decreasing it thereafter proportionally to the inverse square root of the step number. We used
warmup_STEPS = 4000.
"""

#learning_rate = tf.placeholder(tf.float32, shape=[])
#train_step = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98,epsilon=1e-09).minimize(cost)

# same thing as above but dropout removed for doing predictions during test
infer_encoded = encode_sentence(inputs, FLAGS.num_layers, FLAGS.num_heads, is_training=False)
infer_logits = decode_sentence(decoder_input, infer_encoded,  FLAGS.num_layers, FLAGS.num_heads, is_training=False)
preds = tf.nn.softmax(infer_logits)
labels_pred = tf.cast(tf.argmax(preds, axis=-1), tf.int32)

# initialize the variables
init = tf.global_variables_initializer()

# some helpers
def get_batch(array, old, new):
	return array[old:new]

with open(labels_file) as labs:
	labels2id = {lab.strip(): i for i, lab in enumerate(labs)}
	id2labels = {j:i for i,j in labels2id.items()}

STEPS = 0
# very arbitrary, not sure what to do
# don't have enough training data anyways...

WARMUP_STEPS = int((len(x_train) / FLAGS.batch_size) * FLAGS.warmup_epochs)
#WARMUP_STEPS = 2000
print (WARMUP_STEPS)
#WARMUP_STEPS = 1000
# ______________________________________
# here we actually perform the training


with tf.Session() as sess:
	sess.run(init)
	for epoch in range(FLAGS.num_epochs):
		# shuffle train set
		# can't do np.permutations because tensors don't have same length, should fix that at some point
		shuffled = list(zip(x_train, y_train, x_positions_train, x_queries_train, x_position_of_queries_train))
		random.shuffle(shuffled)
		x_train, y_train, x_positions_train, x_queries_train, x_position_of_queries_train = zip(*shuffled)

		for batch in range(len(x_train) // FLAGS.batch_size):
			STEPS += 1
			# get the learning rate
			current_lr = (FLAGS.embeddings_dim ** -0.5) * min(STEPS ** -0.5, STEPS * WARMUP_STEPS ** -1.5)
			old = FLAGS.batch_size * batch
			new = FLAGS.batch_size * (batch + 1)
			# load batch data
			xs, ys, this_pos_input, this_q, this_p = get_batch(x_train, old, new), get_batch(y_train,old, new), get_batch(x_positions_train, old, new), get_batch(x_queries_train, old, new), get_batch(x_position_of_queries_train, old, new)
			# dynamically pad data:
			max_len = max([len(i) for i in xs])
			xs = np.asarray([np.concatenate((tmp, np.zeros(max_len - len(tmp)))) for tmp in xs])
			ys = np.asarray(ys)
			this_pos_input = np.asarray([np.concatenate((tmp, np.zeros(max_len - len(tmp)))) for tmp in this_pos_input])
			this_q = np.asarray(this_q)
			this_p = np.asarray(this_p)
			# run the training step
			preds = sess.run(train_step, feed_dict={x: xs, y: ys, pos_sent:this_pos_input, x_q:this_q, x_p:this_p, learning_rate:current_lr})


		# evaluate the training set
		# compute global cost
		"""
		max_len = max([len(i) for i in x_train])
		x_global = np.asarray([np.concatenate((tmp, np.zeros(max_len - len(tmp)))) for tmp in x_train], dtype=np.int32)
		y_global = np.asarray(y_train)
		positions_global = np.asarray([np.concatenate((tmp, np.zeros(max_len - len(tmp)))) for tmp in x_positions_train], dtype=np.int32)
		queries_global = np.asarray(x_queries_train)
		queries_positions_global = np.asarray(x_position_of_queries_train)
		#print (np.shape(x_global), np.shape(y_global), np.shape(positions_global), np.shape(queries_global), np.shape(queries_positions_global))
		this_cost = sess.run(cost, feed_dict = {x: x_global, y:y_global, pos_sent: positions_global, x_q:queries_global, x_p:queries_positions_global})
		# compute accuracy on train set

		labs = sess.run(labels_pred, feed_dict = {x: x_global, y:y_global, pos_sent: positions_global, x_q:queries_global, x_p:queries_positions_global})
		print ("epoch", epoch, "cost", this_cost)
		acc = sum([1 for i,j in zip(labs, y_train) if i == np.argmax(j)])/len(y_train)
		print ("epoch", epoch, "train accuracy", acc)

		labs = []
		for batch in range(len(x_test) // FLAGS.batch_size):
			old = FLAGS.batch_size * batch
			new = FLAGS.batch_size * (batch + 1)
			xs, ys, this_pos_input, this_q, this_p = get_batch(x_test, old, new), get_batch(y_test,old, new), get_batch(x_positions_test, old, new), get_batch(x_queries_test, old, new), get_batch(x_position_of_queries_test, old, new)


			max_len = max([len(i) for i in xs])
			xs = np.asarray([np.concatenate((tmp, np.zeros(max_len - len(tmp)))) for tmp in xs])
			ys = np.asarray(ys)
			this_pos_input = np.asarray([np.concatenate((tmp, np.zeros(max_len - len(tmp)))) for tmp in this_pos_input])
			this_q = np.asarray(this_q)
			this_p = np.asarray(this_p)
			# run the training step
			preds = sess.run(labels_pred, feed_dict={x: xs, y: ys, pos_sent:this_pos_input, x_q:this_q, x_p:this_p, learning_rate:current_lr})
			labs += [id2labels[i] for i in preds] 
		"""
		max_len = max([len(i) for i in x_test])

		x_global = np.asarray([np.concatenate((tmp, np.zeros(max_len - len(tmp)))) for tmp in x_test], dtype=np.int32)
		y_global = np.asarray(y_test)
		positions_global = np.asarray([np.concatenate((tmp, np.zeros(max_len - len(tmp)))) for tmp in x_positions_test], dtype=np.int32)
		queries_global = np.asarray(x_queries_test)
		queries_positions_global = np.asarray(x_position_of_queries_test)
		print (np.shape(x_global), np.shape(y_global), np.shape(positions_global), np.shape(queries_global), np.shape(queries_positions_global))

		# evaluate on test set after every epoch (because development purposes etc...)
		labs = sess.run(labels_pred, feed_dict={x: x_global, y:y_global, pos_sent: positions_global, x_q:queries_global, x_p:queries_positions_global})
		acc = sum([1 for i,j in zip(labs, y_test) if i == np.argmax(j)])/len(y_test)
		print ("epoch", epoch, "test accuracy", acc)

		labs = [id2labels[i] for i in labs]


		true = [id2labels[np.argmax(i)] for i in y_test]
		# to not break the program in the first few epochs because of zero division errors
		try:
			print (macro_f1(true, labs))
		except Exception as inst:
			print (str(inst))
		"""
		x_short = x_global[:10]
		y_short = y_global[:10]
		pos_glob_short = positions_global[:10]
		q_short = queries_global[:10]
		q_pos_short = queries_positions_global[:10]
		"""
		if epoch == FLAGS.num_epochs - 1:
			labs = sess.run(labels_pred, feed_dict={x: x_global, y:y_global, pos_sent: positions_global, x_q:queries_global, x_p:queries_positions_global})
			encs, quers = [[id2word[w] for w in sent if w != 0] for sent in x_global], [[id2word[w] for w in query] for query in queries_global]
			labs_true, labs_pred = [id2labels[np.argmax(j)] for j in y_test], [id2labels[i] for i in labs]

			#print (sess.run(inspect_keys, feed_dict={x: x_short, y:y_short, pos_sent: pos_glob_short, x_q:q_short, x_p:q_pos_short}))
			#print (sess.run(encoded, feed_dict={x: x_short, y:y_short, pos_sent: pos_glob_short, x_q:q_short, x_p:q_pos_short}))
			#print ("encoded")
			#print (sess.run(infer_encoded, feed_dict={x: x_short, y:y_short, pos_sent: pos_glob_short, x_q:q_short, x_p:q_pos_short}))	
			attention_scores = sess.run(inspect_softmax, feed_dict={x: x_global, y:y_global, pos_sent: positions_global, x_q:queries_global, x_p:queries_positions_global})
			pos_encs = [[w for w in sent if w != 0] for sent in positions_global]
			for enc, quer, lab_true, lab_pred, att_scores, pos_enc, pos_quer in zip(encs, quers, labs_true, labs_pred, attention_scores, pos_encs, queries_positions_global):
				pos_q_1 = str(pos_quer[0])
				pos_q_2 = str(pos_quer[1])

				att_score_q1 = att_scores[0][:len(enc)]
				att_score_q2 = att_scores[1][:len(enc)]
				result_file.write("label true " + lab_true + " label predicted " + lab_pred + " " + str(lab_true == lab_pred) + "\n")
				result_file.write("query 1 " + quer[0] + " at position " + pos_q_1 +"\n")
				result_file.write(" ".join([i + "(" + "{0:.2f}".format(j) + ")" for i,j in zip(enc, att_score_q1)]) + "\n")
				result_file.write("attention sum q1: " + str(sum(att_scores[0])) + "\n")
				indices = att_score_q1.argsort()[-3:][::-1]
				result_file.write(quer[0] + " looks at " + " ".join([enc[i] + " {0:.2f}".format(att_score_q1[i]) for i in indices]) + "\n")
				result_file.write("query 2 " + quer[1] + " at position " + pos_q_2 +"\n")
				result_file.write(" ".join([i + "(" + "{0:.2f}".format(j) + ")" for i,j in zip(enc, att_score_q2)]) + "\n")
				result_file.write("attention sum q2: " + str(sum(att_scores[1])) + "\n")

				indices = att_score_q2.argsort()[-3:][::-1]
				result_file.write(quer[1] + " looks mostly at " + " ".join([enc[i] + " {0:.2f}".format(att_score_q2[i]) for i in indices]) + "\n")
				result_file.write("\n\n")

				"""
				print ("label true", lab_true, "label predicted", lab_pred, str(lab_true == lab_pred))
				for i,j in zip(enc, att_score_q1):
					print (i, "(" + "{0:.2f}".format(j) + ")")
				indices = att_score_q1.argsort()[-3:][::-1]
				print (quer[0] + "looks mostly at", [enc[i] + " {0:.2f}".format(att_score_q1[i]) for i in indices])
				for i,j in zip(enc, att_score_q2):
					print (i, "(" + "{0:.2f}".format(j) + ")")
				indices = att_score_q2.argsort()[-3:][::-1]
				print (quer[1] + "looks mostly at", [enc[i] + " {0:.2f}".format(att_score_q2[i]) for i in indices])
				print ("\n")
				print ("encoded sentence", enc)
				print ("positions in sent", pos_enc)
				print("query 1", quer[0], "looks at", "\n", att_score_q1)
				print ("query 2", quer[1], "looks at", "\n", att_score_q2)
				print ("\n")
				"""
		# does something like 6x.xx% macro F1 on the testset after epoch 50 with 1 layer and 2 heads and sent length 25
		# 6x.xx% macro F1 after epoch 50 with 3 layers, 3 heads and sent length 25
		# 6x.xx% macro F1 after epoch 50 with 3 layers, 3 heads, sent length 25 and only going from [e1,...,e2]
		# after fixing bugs
		# ('76.26', '74.27', '75.08'), batchsize 256, epochs 200, num layers 2, num heads 4
		# ('78.75', '78.50', '78.48') batchsize 200, epochs 200, num layers 2, num heads 2, sent min length 4, sent max length 10
		# ('79.59', '78.37', '78.92') batchsize 250, epochs 200, num layers 2, num heads 4, sent min length 4, sent max length 12




