import numpy as np
import time
import tensorflow as tf
from nltk import word_tokenize
class preprocessing:

	# after 2000k steps: ('84.20', '86.89', '85.47')
	# same model, same settings, next run:('76.93', '76.91', '76.62')

	# with inverse label: ('84.47', '87.74', '86.01')
	# with ('84.45', '87.79', '86.02')



	def __init__(self, FLAGS):



		self.record_train = "/raid/data/dost01/semeval10_data/TRAIN_FILE.TXT"
		self.record_test = "/raid/data/dost01/semeval10_data/TEST_FILE_FULL.TXT"
		self.labels_file = "/raid/data/dost01/semeval10_data/labels.txt"

		"""
		self.record_train = "/home/dominik/Documents/DFKI/data/TRAIN_FILE.TXT"
		self.record_test = "/home/dominik/Documents/DFKI/data/TEST_FILE_FULL.TXT"
		self.labels_file = "/home/dominik/Documents/DFKI/data/labels_semeval2010.txt"


		"""
		with open(self.labels_file) as labs:
			labels = {lab.strip(): i for i, lab in enumerate(labs)}
		invs = [1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,16,18,17]
		with open(self.labels_file) as labs:
			#labels_inv = {lab.strip():i for i, lab in zip(invs, labs)}
			labels_inv = {i:j for i,j in enumerate(invs)}
		self.FLAGS = FLAGS
		self.train_file = "embedded_train.txt"
		self.test_file = "embedded_test.txt"

		self.train_sentences = self.read_records(self.record_train)
		self.test_sentences = self.read_records(self.record_test)

		self.train = self.read_file(self.train_file, self.train_sentences)



		print ([np.shape(x) for x in self.train])
		pos = tf.keras.preprocessing.sequence.pad_sequences(self.train[2], padding="post")
		maxlen = len(pos[0])
		xs = np.array([np.concatenate((x, np.zeros((maxlen - len(x), 1024))), axis=0) if len(x) > 0 else np.zeros((maxlen, 1024)) for x in self.train[0]])



		if self.FLAGS.use_inverse == "yes":
			inv_x, inv_labs, inv_p, inv_q, inv_pq = [], [], [], [], []
			for x, lab, p, q, p_q in zip(xs, self.train[1], pos, self.train[3], self.train[4]):
				if np.argmax(lab) == 16:
					continue
				inv_x.append(x)
				inv_labs.append(self.create_one_hot(len(lab), labels_inv[np.argmax(lab)]))
				inv_p.append(p)
				inv_q.append(list(reversed(q)))
				inv_pq.append(list(reversed(p_q)))

			xs = np.concatenate((xs,inv_x),axis=0)
			labs = np.concatenate((self.train[1], inv_labs),axis=0)
			pos = np.concatenate((pos, inv_p),axis=0)
			qs = np.concatenate((self.train[3], inv_q),axis=0)
			pos_qs = np.concatenate((self.train[4], inv_pq),axis=0)
			self.train = (xs,labs,pos,qs,pos_qs)
		else:
			self.train = (xs, np.array(self.train[1]), pos, np.array(self.train[3]), np.array(self.train[4]))
		print ([np.shape(x) for x in self.train])
		self.test = self.read_file(self.test_file, self.test_sentences)
		pos = tf.keras.preprocessing.sequence.pad_sequences(self.test[2], padding="post")
		maxlen = len(pos[0])
		xs = np.array([np.concatenate((x, np.zeros((maxlen - len(x), 1024))), axis=0) if len(x) > 0 else np.zeros((maxlen, 1024)) for x in self.test[0]])
		self.test = (xs, np.array(self.test[1]), pos, np.array(self.test[3]), np.array(self.test[4]))
		self.max_length = max(len(self.train[0]), len(self.test[0]))
		print ([np.shape(x) for x in self.test])
	def create_one_hot(self, length, x):
		# create one hot vector
		zeros = np.zeros(length, dtype=np.int32)
		zeros[x] = 1
		return zeros


	def read_file(self, fn,record):
		x,y,pos,q, pos_q = [],[],[],[],[]
		with open(fn) as infile:
			for line,string in zip(infile, record):

				context_left, e1, context_mid, e2, context_right = self.tokenize(string)
				sent = context_left + e1 + context_mid + e2 + context_right 
				line = line.split("\t")
				xs, lab, _, qs, q_pos = line
				#print (q_pos)
				q_pos = list(map(int, q_pos.split()))
				print ("position queries",q_pos)
				start, end = q_pos[:4], q_pos[4:]
				start, end = max(start)-1, max(end)-1
				print ("sentence",sent)
				print ("q1: [", sent[start],"]", sent[start + 1:end],"q2:[",sent[end],"]")

				xs = xs.split("#")
				embedded = [list(map(float,i.split())) for i in xs[start + 1:end]][:20]
				x.append(embedded)
				#print (np.shape(x))
				#pos_q.append([x_start, x_end])
				#pos.append(list(range(2,end-start+1)))
				#pos_q.append([1,end+2])
				pos.append(list(range(2,min(end-start+1, 22))))			
				#pos_q.append([1,min(end+2, 22)])
				pos_q.append([1, len(embedded) + 2])
				print ("query positions",pos_q[-1])
				print ("positions", pos[-1])



				#print (xs[start])
				#print (xs[end])
				q.append([list(map(float,xs[start].split())), list(map(float,xs[end].split()))])
				#
				y.append(list(map(int,lab.split())))
				#print (np.shape(x[-1]))
				#print (start, "end", end)
				#time.sleep(1)
		return x,y,pos,q,pos_q


	def read_records(self, fn):
		records = []
		counter = 0
		with open(fn) as infile:
			for line in infile:					
				if counter % 4 == 0:
					records.append(line)
				counter += 1
		return records

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


if __name__ == "__main__":
	preprocessing("Hi")

	
