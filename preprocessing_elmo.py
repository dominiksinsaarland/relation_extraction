import numpy as np
import tensorflow as tf
from nltk import word_tokenize
import time
from collections import defaultdict
import re

"""
from allennlp.modules.elmo import Elmo, batch_to_ids
import allennlp
from allennlp.commands.elmo import ElmoEmbedder
"""
import tensorflow_hub as hub
class preprocessing:
	"""
	
	preprocessing module, creates elmo embeddings for the data set
	should be done in a preprocessing step and not during actual training becaue
	batchsizes for transformer models should be around 2k
	- elmo fails to compute 2k in one batch (OOM issues) and one has to mitigate that anyways
	- secondly, creating 2k embeddings takes 10 minutes, clearly dominating the run time of the model (to compute one training step of transformer takes 5 seconds)
	therefore outsourced in this module
	"""

	def __init__(self, FLAGS):
		self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
		self.FLAGS = FLAGS
		self.record_train = self.FLAGS.train_infile
		self.record_test = self.FLAGS.test_infile

		#TODO generate labels file (one label per line)
		self.labels_file = self.FLAGS.labels_file

		#TODO for a new dataset: override the read_records function

		self.train = self.read_records(self.record_train)
		# compute sequence lengths and pad sequences
		max_len = max(len(x) for x in self.train[0])
		seq_lengths = [len(x) for x in self.train[0]]
		sents = [x + [""] * (max_len - len(x)) for x in self.train[0]]
		config = tf.ConfigProto(allow_soft_placement = True)

		# compute elmo embeddings
		for batch in range(len(self.train[0]) // 100 + 1):
			batch_sents = sents[batch * 100:(batch + 1) * 100]
			batch_seq_lengths = seq_lengths[batch * 100:(batch + 1) * 100]
			if batch_sents:
				self.embedded = self.elmo(inputs={"tokens": batch_sents, "sequence_len":batch_seq_lengths}, signature="tokens", as_dict=True)["elmo"]
				with tf.Session(config=config) as sess:
					sess.run(tf.global_variables_initializer())
					if batch == 0:
						embedded = sess.run(self.embedded)
					else:
						batch_embedded = sess.run(self.embedded)
						embedded = np.concatenate((embedded, batch_embedded), axis=0)

		# save elmo embeddings
		np.save(self.FLAGS.word_embeddings_train, embedded)
		outfile_name = self.FLAGS.train_outfile
		with open(outfile_name, "w") as outfile:
			for label, positions, sentence in zip(self.train[1], self.train[2], self.train[0]):
				outfile.write(str(np.argmax(label)) + "\t" + str(positions[0]) + "\t" + str(positions[1]) + "\t" + " ".join(sentence) + "\n")

		self.test = self.read_records(self.record_test)	
		max_len = max(len(x) for x in self.test[0])
		seq_lengths = [len(x) for x in self.test[0]]
		sents = [x + [""] * (max_len - len(x)) for x in self.test[0]]

		for batch in range(len(self.test[0]) // 100 + 1):
			batch_sents = sents[batch * 100:(batch + 1) * 100]
			batch_seq_lengths = seq_lengths[batch * 100:(batch + 1) * 100]
			if batch_sents:
				self.embedded = self.elmo(inputs={"tokens": batch_sents, "sequence_len":batch_seq_lengths}, signature="tokens", as_dict=True)["elmo"]
				with tf.Session(config=config) as sess:
					sess.run(tf.global_variables_initializer())
					if batch == 0:
						embedded = sess.run(self.embedded)
					else:
						batch_embedded = sess.run(self.embedded)
						embedded = np.concatenate((embedded, batch_embedded), axis=0)
		np.save(self.FLAGS.word_embeddings_test, embedded)
		outfile_name = self.FLAGS.test_outfile
		with open(outfile_name, "w") as outfile:
			for label, positions, sentence in zip(self.test[1], self.test[2], self.test[0]):
				outfile.write(str(np.argmax(label)) + "\t" + str(positions[0]) + "\t" + str(positions[1]) + "\t" + " ".join(sentence) + "\n")


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
		return context_left, e1, context_mid, e2, context_right

	def read_records(self, fn):
		"""
		read records semeval
		this is slightly cumbersome for different reasons
		- one example is split in 4 lines (sentence in the first line, label in the second, comments in the third etc.
		- tokenization is annoying, an example sentence: "A misty <e1>ridge</e1> uprises from the <e2>surge</e2>."
		- so one has to get ride of the html tags but remember the positions of e1 and e2 in the sentence
		
		for other corpora (ace and tacred), this is easier because lines are already tokenized and the positions for e1 and e2 are included
		"""

		# load labels
		with open(self.labels_file) as labs:
			labels = {lab.strip(): i for i, lab in enumerate(labs)}
		# counter because every example spans over four lines
		counter = 0
		X, y, entity_positions = [], [], []

		with open(fn) as infile:
			for line in infile:
				# again, cumbersome because examples are split across different lines
				# if counter % 4 == 0, we have a sentence
				if counter % 4 == 0:
					# tokenize (into 5 parts, consider: A misty <e1>ridge</e1> uprises from the <e2>surge</e2>.
					# yields ["A", "misty"], ["ridge"], ["uprises", "from", "the"], ["surge"], []

					context_left, e1, context_mid, e2, context_right = self.tokenize(line)
					position_e1 = len(context_left) + len(e1) - 1
					position_e2 = len(context_left) + len(e1) + len(context_mid) + len(e2) - 1
					entity_positions.append((position_e1, position_e2))
					
					sent = context_left + e1 + context_mid + e2 + context_right
					X.append(sent)
				# if counter % 4 == 0, we have a label
				elif counter % 4 == 1:
					label = line.strip()
					y.append(self.create_one_hot(len(labels), labels[label]))
				counter += 1
		#print (np.shape(y), np.shape(X_queries), np.shape(X_position_of_queries))
		return X, y, entity_positions



	def create_one_hot(self, length, x):
		# create one hot vector
		zeros = np.zeros(length, dtype=np.int32)
		zeros[x] = 1
		return zeros

if __name__ == "__main__":
	flags = tf.flags

	#flags.DEFINE_float("learning_rate", 0.001, "") // use adaptive learning rate instead, see explanations below
	flags.DEFINE_string("train_infile", "/raid/data/dost01/semeval10_data/TRAIN_FILE.TXT", "filename train")
	flags.DEFINE_string("test_infile", "/raid/data/dost01/semeval10_data/TEST_FILE_FULL.TXT","filename test")
	flags.DEFINE_string("labels_file","/raid/data/dost01/semeval10_data/labels.txt", "files with one label per line")
	flags.DEFINE_string("train_outfile", "semeval10_meta_info_train.txt", "filename of meta info train (positions queries, labels")
	flags.DEFINE_string("test_outfile", "semeval10_meta_info_test.txt", "filename of meta info test (positions queries, labels")
	flags.DEFINE_string("word_embeddings_train","semeval_elmo_embeddings_train_sentences", "filename of word embeddings train (npy file")
	flags.DEFINE_string("word_embeddings_test", "semeval_elmo_embeddings_test_sentences", "filename of word embeddings test (npy file")
	FLAGS = flags.FLAGS
	preprocessing = preprocessing(FLAGS)


