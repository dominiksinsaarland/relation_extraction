from collections import defaultdict
from find_best_gpu import *
import random

def get_batch(tup, old, new, max_len=0):
	"""
	build batches dynamically
	remember: train and test are tuples consisting of (embedded_sentences, labels, position_entities, tokens)
	we need to return: embeddings, queries, positions, position_queries, labels
	"""
	tup = [x[old:new] for x in tup]
	labels = tup[1]
	xs, queries, positions, position_queries = [], [], [], []
	if max_len == 0:
		#max_len = max([len(x) for x in tup[-1]])
		max_len = max([e[1] - e[0] - 1 for e in tup[2]])
	else:
		max_len = max_len
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
		"""
		e1, e2 = entities[0], entities[1]
		if len(embedded) > max_len:
			embedded = embedded[e1:e2+1]
			e1, e2 = 0, len(embedded) - 1
		if len(embedded) < max_len:
			x = np.concatenate((embedded, np.zeros((max_len-len(embedded),1024))),axis=0)
		else:
			x = embedded
		pos = list(range(1,len(embedded) + 1))
		position_queries.append([e1 + 1, e2 + 1])
		pos +=[0] * (max_len - len(embedded))
		positions.append(pos)
		xs.append(x)
		queries.append([embedded[e1], embedded[e2]])
		"""

	return np.array(xs), np.array(labels), np.array(positions), np.array(queries), np.array(position_queries)


def write_html_file_new(sent, q1, q2, predicted, true, entities):
	e1 = entities[0]
	e2 = entities[1]
	html_string = ""

	if predicted == true:
		html_string += '<p> true label: "' + true + '"; predicted label: "' + predicted + '"; correct prediction</p>\n'
	else:
		html_string += '<p> true label: "' + true + '"; predicted label: "' + predicted + '"; wrong prediction</p>\n'
	html_string += "<p>"


	for i,w in zip(range(len(sent)), sent):
		att_score_q1 = q1[:,i]
		# should be 8
		#print ("in function write html file", np.shape(att_score_q1), "i", i)
		if i == 0:
			html_string += '<block style="border: 1px solid black">' + "<b>" + e1 + "</b>" + "</block>"
			html_string += '<block>'
		else:
			html_string += '<block>'
		for c1 in att_score_q1:
			html_string += '<span style="background-color: rgb(255,255,' + str(int(255 - c1 * 255)) + ')"> ' + "%.1f" % c1 + " </span>"
		html_string +=  "<b>" + w + "</b>"  + " </block>"
		if i == len(sent) - 1:
			html_string += '<block style="border: 1px solid black">' + "<b>" + e2 + "</b>" + "</block>"
	html_string += "</p>\n<p>"
	for i,w in zip(range(len(sent)), sent):
		att_score_q2 = q2[:,i]
		if i == 0:
			html_string += '<block style="border: 1px solid black">' + "<b>" + e1 + "</b>" + "</block>"
			html_string += '<block>'
		else:
			html_string += '<block>'
		for c2 in att_score_q2:
			html_string += '<span style="background-color: rgb(255,' + str(int(255 - c2 * 255)) + ',255)"> ' + "%.1f" % c2 + " </span>"
		html_string += "<b>" + w + "</b>" + " </block>"
		if i == len(sent) - 1:
			html_string += '<block style="border: 1px solid black">' + "<b>" + e2 + "</b>" + "</block>"
	html_string += "<br><br></p>\n"
	"""

	for i,w in zip(range(len(sent)), sent):
		att_score_q1 = q1[:,i]
		# should be 8
		#print ("in function write html file", np.shape(att_score_q1), "i", i)
		if i == e1:
			html_string += '<block style="border: 1px solid black">'
		elif i == e2:
			html_string += '<block style="border: 1px solid black">'
		else:
			html_string += '<block>'
		for c1 in att_score_q1:
			html_string += '<span style="background-color: rgb(255,255,' + str(int(255 - c1 * 255)) + ')"> ' + "%.1f" % c1 + " </span>"
		html_string +=  "<b>" + w + "</b>"  + " </block>"
	html_string += "</p>\n<p>"
	for i,w in zip(range(len(sent)), sent):
		att_score_q1 = q2[:,i]
		# should be 8
		#print ("in function write html file", np.shape(att_score_q1), "i", i)
		if i == e1:
			html_string += '<block style="border: 1px solid black">'
		elif i == e2:
			html_string += '<block style="border: 1px solid black">'
		else:
			html_string += '<block>'
		for c2 in att_score_q1:
			html_string += '<span style="background-color: rgb(255,' + str(int(255 - c2 * 255)) + ',255)"> ' + "%.1f" % c2 + " </span>"
		html_string +=  "<b>" + w + "</b>"  + " </block>"
	html_string += "<br><br></p>\n"
	"""

	return html_string



def write_html_file(sent, q1, q2,predicted, true, entity):
	e1 = entity[0]
	e2 = entity[1]
	html_string = ""

	if predicted == true:
		html_string += '<p> true label: "' + true + '"; predicted label: "' + predicted + '"; correct prediction</p>'
	else:
		html_string += '<p> true label: "' + true + '"; predicted label: "' + predicted + '"; wrong prediction</p>'
	q1 = np.average(q1, axis=0)
	q2 = np.average(q2, axis=0)
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
	#tf.set_random_seed(seed)

	with tf.device(device):
		# define some hyperparameters
		flags = tf.flags
		flags.DEFINE_integer("hidden_dim", 1024, "number hidden dimensions")
		flags.DEFINE_integer("num_labels", 19, "number of target labels")
		flags.DEFINE_float("label_smoothing", 0.1, "label smoothing")
		flags.DEFINE_integer("batch_size", 2000, "number of batchsize, bigger works better")
		flags.DEFINE_float("dropout", 0.50, "dropout applied after each layer")
		flags.DEFINE_integer("num_layers", 1, "num layers for encoding/decoding")
		flags.DEFINE_integer("num_heads",4, "num heads per layer")
		flags.DEFINE_integer("embeddings_dim", 1024, "number of dimensions in word embeddings")
		flags.DEFINE_float("l2_lambda", 0.05, "")
		flags.DEFINE_integer("warmup_steps", 250, "")
		flags.DEFINE_integer("max_steps", 1001, "")
		flags.DEFINE_string("corpus", "semeval", "")
		flags.DEFINE_string("iteration", "", "")
		flags.DEFINE_string("resultfile", "default", "")
		flags.DEFINE_float("hidden_units_ffclayer", 2, "")

		flags.DEFINE_string("labels_file","/raid/data/dost01/semeval10_data/labels.txt", "files with one label per line")
		flags.DEFINE_string("train_file", "../clean_dir/semeval10_meta_info_train.txt", "filename of meta info train (positions queries, labels")
		flags.DEFINE_string("test_file", "../clean_dir/semeval10_meta_info_test.txt", "filename of meta info test (positions queries, labels")
		flags.DEFINE_string("word_embeddings_train","../clean_dir/semeval_elmo_embeddings_train_sentences", "filename of word embeddings train (npy file")
		flags.DEFINE_string("word_embeddings_test", "../clean_dir/semeval_elmo_embeddings_test_sentences", "filename of word embeddings test (npy file")


		flags.DEFINE_string("use_types", "no", "")
		flags.DEFINE_string("encoder_attention", "yes", "")
		flags.DEFINE_string("encoder_feedforward", "yes", "")
		flags.DEFINE_string("decoder_self_attention", "yes", "")
		flags.DEFINE_string("decoder_encoder_attention", "yes", "")
		flags.DEFINE_string("decoder_feedforward", "yes", "")

		FLAGS = flags.FLAGS
		# initialize preprocessing
		preprocessing = preprocessing(FLAGS)
		
		# for evaluation
		with open(preprocessing.labels_file) as labs:
			labels2id = {lab.strip(): i for i, lab in enumerate(labs)}
			id2labels = {j:i for i,j in labels2id.items()}


		# for batch weights:
		label_counts = defaultdict(int)
		counter = 0
		with open("../semeval10_data/TRAIN_FILE.TXT") as infile:
			for line in infile:
				if counter % 4 == 1:
					line = line.strip()
					label_counts[line] += 1
				counter += 1
		batch_weights = {}
		for i,j in label_counts.items():
			normalized = 1/(j/(8000/len(label_counts)))
			batch_weights[labels2id[i]] = min(5, normalized)
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
					#print ([np.shape(x) for x in this_batch])
					#this_weights = np.array([batch_weights[np.argmax(i)] for i in this_batch[1]])
					this_weights = np.ones(len(this_batch[0]))
					# train step
					sess.run(transformers_model.train_step, feed_dict={transformers_model.x: this_batch[0], transformers_model.y: this_batch[1], transformers_model.positions:this_batch[2], transformers_model.queries:this_batch[3],transformers_model.query_positions:this_batch[4], transformers_model.learning_rate:current_lr, transformers_model.weights:this_weights})

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
			html_string = "<h2> semeval</h2>\n"
			pr, rc, f1 = macro_f1(true, labs)
			html_string += "<p>" + "pr:" + pr + ",rc:" + rc + ",f1:" + f1 + "</p><br>"

			labs_true = true
			labs_pred = labs
			#max_len = max([len(x) for x in preprocessing.test[-1]])
			attention_scores = np.zeros((len(preprocessing.test[0]), FLAGS.num_heads, 2, 50))
			for test_batch_counter in range(len(preprocessing.test[0]) // FLAGS.batch_size + 1):
				old = FLAGS.batch_size * test_batch_counter
				new = FLAGS.batch_size * (test_batch_counter + 1)
				test_batch = get_batch(preprocessing.test, old, new,max_len = 50)
				#test_batch = get_batch(preprocessing.test, old, new,max_len = max_len)
				"""
				if test_batch_counter == 0:
					attention_scores = [sess.run(transformers_model.get_attention_scores[i], feed_dict={transformers_model.x: test_batch[0], transformers_model.y: test_batch[1], transformers_model.positions:test_batch[2], transformers_model.queries:test_batch[3], transformers_model.query_positions:test_batch[4], transformers_model.types:test_batch[5]}) for i in range(8)]
					print (np.shape(attention_scores))
				else:
					tmp = [sess.run(transformers_model.get_attention_scores[i], feed_dict={transformers_model.x: test_batch[0], transformers_model.y: test_batch[1], transformers_model.positions:test_batch[2], transformers_model.queries:test_batch[3], transformers_model.query_positions:test_batch[4], transformers_model.types:test_batch[5]}) for i in range(8)]
					print (np.shape(tmp))
					attention_scores = np.concatenate((attention_scores, tmp),axis=1)
				"""
				for i in range(FLAGS.num_heads):
					tmp = sess.run(transformers_model.get_attention_scores[i], feed_dict={transformers_model.x: test_batch[0], transformers_model.y: test_batch[1], transformers_model.positions:test_batch[2], transformers_model.queries:test_batch[3], transformers_model.query_positions:test_batch[4]})
					attention_scores[:,i][old:new] = tmp

			#attention_scores = np.mean(attention_scores,axis=0)
			print ("attention scores", np.shape(attention_scores))

			#remember: train and test are tuples consisting of (embedded_sentences, labels, position_entities, tokens)
			print (len(preprocessing.test[-1]), len(labs_true), len(labs_pred), len(attention_scores), len(preprocessing.test[2]))
			for ws, lab_true, lab_pred, att_scores, entities in zip(preprocessing.test[-1], labs_true, labs_pred, attention_scores, preprocessing.test[2]):
				# iterate through sentences and write the html file
				e1,e2 = entities[0], entities[1]

				e1_token, e2_token = ws[e1], ws[e2]
				ws = ws[e1+1:e2]
				att_score_q1 = att_scores[:,0]
				att_score_q2 = att_scores[:,1]
				#html_string += write_html_file_new(ws, att_score_q1, att_score_q2, lab_pred, lab_true, (e1, e2))
				html_string += write_html_file_new(ws, att_score_q1, att_score_q2, lab_pred, lab_true, (e1_token, e2_token))

			"""
			flags.DEFINE_string("encoder_attention", "yes", "")
			flags.DEFINE_string("encoder_feedforward", "yes", "")
			flags.DEFINE_string("decoder_self_attention", "yes", "")
			flags.DEFINE_string("decoder_encoder_attention", "yes", "")
			flags.DEFINE_string("decoder_feedforward", "yes", "")
			"""
			filename = "transformer_htmls_" + FLAGS.encoder_attention + "_" + FLAGS.encoder_feedforward + "_" + FLAGS.decoder_self_attention + "_" + FLAGS.decoder_encoder_attention + "_" + FLAGS.iteration + ".html"
			with open(filename, "w") as outfile:
				outfile.write(html_string)

			html_string = "<h2> semeval</h2>\n"
			pr, rc, f1 = macro_f1(true, labs)
			html_string += "<p>" + "pr:" + pr + ",rc:" + rc + ",f1:" + f1 + "</p><br>"

			labs_true = true
			labs_pred = labs
			for ws, lab_true, lab_pred, att_scores, entities in zip(preprocessing.test[-1], labs_true, labs_pred, attention_scores, preprocessing.test[2]):
				# iterate through sentences and write the html file
				e1,e2 = entities[0], entities[1]
				e1_token, e2_token = ws[e1], ws[e2]
				ws = ws[e1+1:e2]

				att_score_q1 = att_scores[:,0]
				att_score_q2 = att_scores[:,1]
				html_string += write_html_file(ws, att_score_q1, att_score_q2, lab_pred, lab_true, (e1_token, e2_token))

			with open("html_results_heads_collapsed.html", "w") as outfile:
				outfile.write(html_string)


		f1_results.close()
