from collections import defaultdict
from find_best_gpu import *
import random

def get_batch(tup, old, new):
	"""
	build batches dynamically
	remember: train and test are tuples consisting of (embedded_sentences, labels, position_entities, tokens)
	we need to return: embeddings, queries, positions, position_queries, labels
	"""
	tup = [x[old:new] for x in tup]
	labels = tup[1]
	xs, queries, positions, position_queries = [], [], [], []
	max_len = max([e[1] - e[0] - 1 for e in tup[2]])

	for embedded, entities in zip(tup[0], tup[2]):
		e1, e2 = entities[0], entities[1]
		x = embedded[e1+1:e2]
		pos = list(range(2,len(x) + 2))

		position_queries.append([1, len(pos) + 2])
		pos +=[0] * (max_len - len(x))

		if len(x) < max_len:
			x = np.concatenate((x, np.zeros((max_len-len(x),1024))),axis=0)
		xs.append(x)


		positions.append(pos)
		queries.append([embedded[e1], embedded[e2]])
	return np.array(xs), np.array(labels), np.array(positions), np.array(queries), np.array(position_queries)

def write_html_file(sent,info, q1, q2,predicted, true, entity):
	# write out html file and highlight, what the different queries look at
	"""
	e1 = info[0][info[1][0]:info[1][1]]
	e1 = " ".join([preprocessing.id2word[w] for w in e1])
	e2 = info[0][info[1][2]:info[1][3]]
	e2 = " ".join([preprocessing.id2word[w] for w in e2])
	"""
	e1 = entity[0]
	e2 = entity[1]

	html_string = ""

	# make this: "background-color: rgb(255,255,140);border: 1px solid black" around entities
	#with open("colors_highlighted.html", "a") as outfile:

	if predicted == true:
		html_string += '<p> true label: "' + true + '"; predicted label: "' + predicted + '"; correct prediction</p>'
	else:
		html_string += '<p> true label: "' + true + '"; predicted label: "' + predicted + '"; wrong prediction</p>'

	if FLAGS.use_whole_sentence == "no":
		html_string += "<p>"
		html_string += '<span style="background-color: rgb(255,255,255);border: 1px solid black">' + e1 + " </span>"
		for i,w, c1, c2 in zip(range(len(sent)), sent, q1, q2):
			#if i == pos_q1 - 1:
			#	html_string += ' <span style="background-color: rgb(255,255,' + str(int(255 - c1 * 255)) + ');border: 1px solid black">' + w + " </span>"
			#else:
			html_string += ' <span style="background-color: rgb(255,255,' + str(int(255 - c1 * 255)) + ')">' + w + " </span>"
		html_string += '<span style="background-color: rgb(255,255,255);border: 1px solid black">' + e2 + " </span>"
		html_string += "</p>\n"
		html_string += "<p>"
		html_string += '<span style="background-color: rgb(255,255,255);border: 1px solid black">' + e1 + " </span>"
		for i,w, c1, c2 in zip(range(len(sent)), sent, q1, q2):
			#if i == pos_q2 - 1:
			#	html_string += ' <span style="background-color: rgb(255,' + str(int(255 - c2 * 255)) + ',255);border: 1px solid black">'+ w + " </span>"
			#else:
			html_string += ' <span style="background-color: rgb(255,' + str(int(255 - c2 * 255)) + ',255)">' + w + " </span>"
		html_string += '<span style="background-color: rgb(255,255,255);border: 1px solid black">' + e2 + " </span>"
		html_string += "<br><br></p>\n"
		return html_string	
	else:
		html_string += "<p>"
		for i,w, c1, c2 in zip(range(len(sent)), sent, q1, q2):
			if i == info[1][0]:
				html_string += '<span style="border: 1px solid black">'
			elif i == info[1][1]:
				html_string += "</span>"
			html_string += ' <span style="background-color: rgb(255,255,' + str(int(255 - c1 * 255)) + ')">' + w + " </span>"
		html_string += "</p>\n"
		html_string += "<p>"
		for i,w, c1, c2 in zip(range(len(sent)), sent, q1, q2):
			if i == info[1][2]:
				html_string += '<span style="border: 1px solid black">'
			elif i == info[1][3]:
				html_string += "</span>"
			html_string += ' <span style="background-color: rgb(255,' + str(int(255 - c2 * 255)) + ',255)">' + w + " </span>"

		html_string += "<br><br></p>\n"
		return html_string

def macro_f1(y_true, y_pred):
	"""
	evaluate macro f1 (prodcues the same results as the official scoring file "semeval2010_task8_scorer-v1.2.pl"
	ignores the Other class
	returns pr, rc and f1 rounded to two decimal points as strings
	"""
	if FLAGS.corpus == "semeval":
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


if __name__ == "__main__":

	# find best gpu and set system variables
	BEST_GPU = find_best_gpu()
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = str(BEST_GPU)

	# Import must happen afterwards
	# After changing visible devices there will be only 1 available device which gets id 0 by default.
	# Hence, make sure to always select gpu:0
	device = '/gpu:0'
	"""
	# if one wants to work on cpu, comment out everything above and set device ="/cpu:0"
	device = "/cpu:0"
	"""
	# set some seeds
	seed = 1337
	random.seed(seed)
	import numpy as np
	np.random.seed(seed)
	import tensorflow as tf
	from load_word_embeddings import *
	from transformer_elmo import *
	tf.set_random_seed(seed)

	with tf.device(device):
		# define some hyperparameters
		flags = tf.flags
		flags.DEFINE_integer("num_labels", 19, "number of target labels")
		flags.DEFINE_float("label_smoothing", 0.1, "label smoothing")
		flags.DEFINE_integer("batch_size", 2000, "number of batchsize, bigger works better")
		flags.DEFINE_float("dropout", 0.40, "dropout applied after each layer")
		flags.DEFINE_integer("num_layers", 1, "num layers for encoding/decoding")
		flags.DEFINE_integer("num_heads",8, "num heads per layer")
		flags.DEFINE_integer("embeddings_dim", 1024, "number of dimensions in word embeddings")
		flags.DEFINE_float("l2_lambda", 0.05, "")
		flags.DEFINE_integer("warmup_steps", 500, "")
		flags.DEFINE_integer("max_steps", 1001, "")
		flags.DEFINE_string("corpus", "semeval", "")
		flags.DEFINE_string("iteration", "", "")
		flags.DEFINE_string("resultfile", "default", "")
		flags.DEFINE_float("hidden_units_ffclayer", 2, "")

		flags.DEFINE_string("labels_file","/raid/data/dost01/semeval10_data/labels.txt", "files with one label per line")
		flags.DEFINE_string("train_file", "semeval10_meta_info_train.txt", "filename of meta info train (positions queries, labels")
		flags.DEFINE_string("test_file", "semeval10_meta_info_test.txt", "filename of meta info test (positions queries, labels")
		flags.DEFINE_string("word_embeddings_train","semeval_elmo_embeddings_train_sentences", "filename of word embeddings train (npy file")
		flags.DEFINE_string("word_embeddings_test", "semeval_elmo_embeddings_test_sentences", "filename of word embeddings test (npy file")

		FLAGS = flags.FLAGS
		# initialize preprocessing
		preprocessing = preprocessing(FLAGS)
		
		# for evaluation
		with open(preprocessing.labels_file) as labs:
			labels2id = {lab.strip(): i for i, lab in enumerate(labs)}
			id2labels = {j:i for i,j in labels2id.items()}

		if FLAGS.resultfile == "default":
			f1_results = open("f1_results" + FLAGS.iteration + ".txt", "w")	
		else:
			f1_results = open(FLAGS.resultfile + FLAGS.iteration + ".txt", "w")

		# initialize model
		transformers_model = model(preprocessing, FLAGS)
		init = tf.global_variables_initializer()
		current_step = 0
		# soft placement needed to run on gpu
		config = tf.ConfigProto(allow_soft_placement = True)
		with tf.Session(config=config) as sess:
			sess.run(init)
			while current_step < FLAGS.max_steps:
				# shuffle
				p = np.random.permutation(len(preprocessing.train[0]))
				train = tuple([x[p] for x in preprocessing.train])
				for batch in range(len(train[0]) // FLAGS.batch_size):
					current_step += 1
					current_lr = (FLAGS.embeddings_dim ** -0.5) * min(current_step ** -0.5, current_step * FLAGS.warmup_steps ** -1.5)
					old = FLAGS.batch_size * batch
					new = FLAGS.batch_size * (batch + 1)
					# compute batches dynamically
					this_batch = get_batch(train, old, new)
					# train step
					sess.run(transformers_model.train_step, feed_dict={transformers_model.x: this_batch[0], transformers_model.y: this_batch[1], transformers_model.positions:this_batch[2], transformers_model.queries:this_batch[3],transformers_model.query_positions:this_batch[4], transformers_model.learning_rate:current_lr})

					# eval every 50 steps
					if current_step % 50 == 0:						
						labs = []
						# predictions batchwise too because of memory issues
						for test_batch in range(len(preprocessing.test[0]) // FLAGS.batch_size + 1):
							old = FLAGS.batch_size * test_batch
							new = FLAGS.batch_size * (test_batch + 1)
							test_batch = get_batch(preprocessing.test, old, new)
							this_labs = sess.run(transformers_model.predictions, feed_dict={transformers_model.x: test_batch[0], transformers_model.y: test_batch[1], transformers_model.positions:test_batch[2], transformers_model.queries:test_batch[3], transformers_model.query_positions:test_batch[4]}) 
							labs += list(this_labs)
						
						acc = sum([1 for i,j in zip(labs, preprocessing.test[1]) if i == np.argmax(j)])/len(preprocessing.test[1])
						print ("step:", current_step, "acc", acc)			
						labs = [id2labels[i] for i in labs]
						true = [id2labels[np.argmax(i)] for i in preprocessing.test[1]]
						# try-except block to not break the program in the first few epochs because of zero division errors
						try:
							print (macro_f1(true, labs))
							f1_results.write("step: " + str(current_step) + " acc: " + str(acc) + " f1: " + str(macro_f1(true, labs)) + "\n")
						except Exception as inst:
							print (str(inst))

			# write html file from here
			html_string = "<h2> corpus: " + FLAGS.corpus + "; encode whole sentence " + FLAGS.use_whole_sentence + "</h2>"

			try:

				if FLAGS.corpus == "ace":
					pr, rc, f1 = micro_f1(true, labs)
					html_string += "<p>" + "pr:" + pr + ",rc:" + rc + ",f1:" + f1 + "</p><br>"

				elif FLAGS.corpus == "kbp37":
					pr, rc, f1 = micro_f1(true, labs)
					html_string += "<p> MICRO F1:" + "pr:" + pr + ",rc:" + rc + ",f1:" + f1 + "</p><br>"
					pr, rc, f1 = macro_f1(true, labs)
					html_string += "<p> MACRO F1:" + "pr:" + pr + ",rc:" + rc + ",f1:" + f1 + "</p><br>"
				else:
					pr, rc, f1 = macro_f1(true, labs)
					html_string += "<p>" + "pr:" + pr + ",rc:" + rc + ",f1:" + f1 + "</p><br>"
			except:
				pr, rc, f1 = "0", "0", "0"
				html_string += "<p>" + "pr:" + pr + ",rc:" + rc + ",f1:" + f1 + "</p><br>"
			# convert ids back to words 

			from tokenize_test import *
			encs, sentences_info, entities = read_test_file()	

			#encs = [[preprocessing.id2word[w] for w in sent if w != 0] for sent in preprocessing.test[0]]

			labs_true = true
			labs_pred = labs
			# get the attention scores (for 4 heads)

			attention_scores_0 = sess.run(transformers_model.get_attention_scores_0, feed_dict={transformers_model.x: preprocessing.test[0], transformers_model.y: preprocessing.test[1], transformers_model.positions:preprocessing.test[2], transformers_model.queries:preprocessing.test[3], transformers_model.query_positions:preprocessing.test[4]})
			attention_scores_1 = sess.run(transformers_model.get_attention_scores_1, feed_dict={transformers_model.x: preprocessing.test[0], transformers_model.y: preprocessing.test[1], transformers_model.positions:preprocessing.test[2], transformers_model.queries:preprocessing.test[3], transformers_model.query_positions:preprocessing.test[4]})
			attention_scores_2 = sess.run(transformers_model.get_attention_scores_2, feed_dict={transformers_model.x: preprocessing.test[0], transformers_model.y: preprocessing.test[1], transformers_model.positions:preprocessing.test[2], transformers_model.queries:preprocessing.test[3], transformers_model.query_positions:preprocessing.test[4]})
			attention_scores_3 = sess.run(transformers_model.get_attention_scores_3, feed_dict={transformers_model.x: preprocessing.test[0], transformers_model.y: preprocessing.test[1], transformers_model.positions:preprocessing.test[2], transformers_model.queries:preprocessing.test[3], transformers_model.query_positions:preprocessing.test[4]})
			arr = np.add(attention_scores_0, attention_scores_1)
			arr = np.add(arr, attention_scores_2)
			arr = np.add(arr, attention_scores_3)
			# average attention scores
			attention_scores = arr / 4

			pos_encs = [[w for w in sent if w != 0] for sent in preprocessing.test[2]]

			# iterate through sentences and write the html file

			for enc, sent_info, lab_true, lab_pred, att_scores, pos_enc, pos_quer, entity in zip(encs, sentences_info, labs_true, labs_pred, attention_scores, pos_encs, preprocessing.test[4], entities):
				pos_q_1 = pos_quer[0]
				pos_q_2 = pos_quer[1]

				att_score_q1 = att_scores[0][:len(enc)]
				att_score_q2 = att_scores[1][:len(enc)]

				html_string += write_html_file(enc, sent_info,att_score_q1, att_score_q2, lab_pred, lab_true, entity)
			if FLAGS.use_whole_sentence == "no":
				with open("html_results_only_e1_e2_" + FLAGS.corpus + FLAGS.iteration + ".html", "w") as outfile:
					outfile.write(html_string)
			else:
				with open("html_results_whole_sentence_" + FLAGS.corpus + ".html", "w") as outfile:
					outfile.write(html_string)


	f1_results.close()



