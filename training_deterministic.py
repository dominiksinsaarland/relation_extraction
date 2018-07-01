from collections import defaultdict
from find_best_gpu import *
import random

def get_batch(tup, old, new):
	# get actual batch and pad down padding tokens to max length of batch
	batch = [x[old:new] for x in tup]
	xs = [[w for w in x if w != 0] for x in batch[0]]
	positions = [[w for w in x if w != 0] for x in batch[2]]
	max_length = max([len(x) for x in xs])
	xs = np.asarray([np.concatenate((tmp, np.zeros(max_length - len(tmp)))) for tmp in xs])
	positions = np.asarray([np.concatenate((tmp, np.zeros(max_length - len(tmp)))) for tmp in positions])
	return [xs, batch[1], positions, batch[3], batch[4]]

def write_html_file(sent,info, q1, q2,predicted, true):
	# write out html file and highlight, what the different queries look at
	e1 = info[0][info[1][0]:info[1][1]]
	e1 = " ".join([preprocessing.id2word[w] for w in e1])
	e2 = info[0][info[1][2]:info[1][3]]
	e2 = " ".join([preprocessing.id2word[w] for w in e2])

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

def micro_f1(y_true, y_pred):
	# micro f1, important for ace and kbp37?
	if FLAGS.corpus == "ace":
		OTHER_RELATION = "NO_RELATION(Arg-1,Arg-1)"
	elif FLAGS.corpus == "kbp37":
		OTHER_RELATION = "no_relation"
	d = defaultdict(int)
	for i,j in zip(y_true, y_pred):
		if i == j:
			d[i +"_TP"] += 1
		else:
			d[j + "_FP"] += 1
			d[i + "_FN"] += 1
	TP = 0
	FP = 0
	FN = 0
	for i,j in d.items():
		if i.endswith("_TP") and i != OTHER_RELATION + "_TP":
			TP += j
		if i.endswith("_FP") and i != OTHER_RELATION + "_FP":
			FP += j

		if i.endswith("_FN") and i != OTHER_RELATION + "_FN":
			FN += j
	Pr = TP/(TP + FP)
	Rc = TP/(TP + FN)
	F1 = (2 * Pr * Rc) / (Pr + Rc) 
	return "{0:.2f}".format(Pr * 100), "{0:.2f}".format(Rc * 100), "{0:.2f}".format(F1 * 100)

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

	elif FLAGS.corpus == "kbp37":
		"""
		evaluate macro f1 (prodcues the same results as the official scoring file "semeval2010_task8_scorer-v1.2.pl"
		ignores the Other class
		returns pr, rc and f1 rounded to two decimal points as strings
		"""

		OTHER = "no_relation"
		d = defaultdict(int)
		for i,j in zip(y_true, y_pred):
			if i == j:
				d[i.split("(")[0] +"__TP"] += 1
			else:
				d[j.split("(")[0] + "__FP"] += 1
				d[i.split("(")[0] + "__FN"] += 1
		TP = 0
		FP = 0
		FN = 0
		items = set()
		for key in d:
			items.add(key.split("__")[0])
		items.remove(OTHER)
		pr, rc, f = 0,0,0
		for item in items:
			t_pr = d[item + "__TP"] / (d[item + "__TP"] + d[item + "__FP"])
			t_rc = d[item + "__TP"] / (d[item + "__TP"] + d[item + "__FN"])
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


	BEST_GPU = find_best_gpu()

	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = str(BEST_GPU)
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	os.environ["inter_op_parallelism_threads"] = str(1)


	# Import must happen afterwards

	# After changing visible devices there will be only 1 available device which gets id 0 by default.
	# Hence, make sure to always select gpu:0
	device = '/cpu:0'


	# set some seeds, still non-deterministic
	seed = 1337

	random.seed(seed)
	import numpy as np
	np.random.seed(seed)
	from preprocessing import *
	from transformers import *
	from preprocessing_ace import *
	from preprocessing_kbp37 import *
	tf.set_random_seed(seed)


	f1_results = open("f1_results.txt", "w")
	with tf.device(device):
		flags = tf.flags

		# some of the flags are not needed anymore
		#flags.DEFINE_float("learning_rate", 0.001, "") // use adaptive learning rate instead, see explanations below
		flags.DEFINE_integer("num_labels", 19, "number of target labels")
		flags.DEFINE_integer("batch_size", 1000, "number of batchsize, bigger works better")
		flags.DEFINE_float("dropout", 0.25, "dropout applied after each layer")
		#flags.DEFINE_integer("sent_length", 50, "sentence length")
		flags.DEFINE_integer("num_layers", 2, "num layers for encoding/decoding")
		flags.DEFINE_integer("num_heads",4, "num heads per layer")
		#flags.DEFINE_integer("num_epochs",20, "")
		#flags.DEFINE_integer("min_length", 0, "min length of encoded sentence")
		flags.DEFINE_integer("embeddings_dim", 100, "number of dimensions in word embeddings")
		flags.DEFINE_float("l2_lambda", 0, "")
		#flags.DEFINE_integer("max_gradient_norm", 5, "")
		#flags.DEFINE_integer("classifier_units", 100, "")
		flags.DEFINE_integer("warmup_steps", 500, "")
		flags.DEFINE_integer("max_steps", 2000, "")
		flags.DEFINE_string("summaries_dir", "test", "")
		flags.DEFINE_string("use_whole_sentence", "no", "")
		flags.DEFINE_string("corpus", "semeval", "")


		FLAGS = flags.FLAGS
		FLAGS.max_steps = FLAGS.warmup_steps * 4 + 1
		# initialize preprocessing
		if FLAGS.corpus == "semeval":
			preprocessing = preprocessing(FLAGS)
		elif FLAGS.corpus == "kbp37":
			FLAGS.num_labels = 37
			preprocessing = preprocessing_kbp37(FLAGS)
			batch_size = 2000
		elif FLAGS.corpus == "ace":
			FLAGS.num_labels = 11
			preprocessing = preprocessing_ace(FLAGS)
			batch_size = 2000

		with open(preprocessing.labels_file) as labs:
			labels2id = {lab.strip(): i for i, lab in enumerate(labs)}
			id2labels = {j:i for i,j in labels2id.items()}

		# initialize model
		transformers_model = model(preprocessing, FLAGS)
		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		current_step = 0

		# soft placement needed to run on gpu
		config = tf.ConfigProto(allow_soft_placement = True)
		config.intra_op_parallelism_threads = 1

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
					# load batch data
					this_batch = get_batch(train, old, new)
					tf.set_random_seed(seed)
					# train step
					summary, _ = sess.run([transformers_model.merged,transformers_model.train_step], feed_dict={transformers_model.x: this_batch[0], transformers_model.y: this_batch[1], transformers_model.positions:this_batch[2], transformers_model.queries:this_batch[3],transformers_model.query_positions:this_batch[4], transformers_model.learning_rate:current_lr})

					# eval every 50 steps
					if current_step % 50 == 0:
						tf.set_random_seed(seed)
						summary, _ = sess.run([transformers_model.merged,transformers_model.predictions], feed_dict={transformers_model.x: this_batch[0], transformers_model.y: this_batch[1], transformers_model.positions:this_batch[2], transformers_model.queries:this_batch[3],transformers_model.query_positions:this_batch[4], transformers_model.learning_rate:current_lr}) 
						tf.set_random_seed(seed)
						transformers_model.train_writer.add_summary(summary, current_step / 50)
						
						# for ace, need to eval in minibatches (37'000 examples in testset)
						if FLAGS.corpus == "ace":
							labs = []
							for test_batch in range(len(preprocessing.test[0]) // FLAGS.batch_size + 1):
								old = FLAGS.batch_size * test_batch
								new = FLAGS.batch_size * (test_batch + 1)
								test_batch = get_batch(preprocessing.test, old, new)
								summary, this_labs = sess.run([transformers_model.merged, transformers_model.predictions], feed_dict={transformers_model.x: test_batch[0], transformers_model.y: test_batch[1], transformers_model.positions:test_batch[2], transformers_model.queries:test_batch[3], transformers_model.query_positions:test_batch[4]}) 
								labs += list(this_labs)
						else:
							summary, labs = sess.run([transformers_model.merged, transformers_model.predictions], feed_dict={transformers_model.x: preprocessing.test[0], transformers_model.y: preprocessing.test[1], transformers_model.positions:preprocessing.test[2], transformers_model.queries:preprocessing.test[3], transformers_model.query_positions:preprocessing.test[4]})

						transformers_model.test_writer.add_summary(summary, current_step / 50)

						
						acc = sum([1 for i,j in zip(labs, preprocessing.test[1]) if i == np.argmax(j)])/len(preprocessing.test[1])
						print ("step:", current_step, "acc", acc)			
						labs = [id2labels[i] for i in labs]
						true = [id2labels[np.argmax(i)] for i in preprocessing.test[1]]
						# to not break the program in the first few epochs because of zero division errors
						try:
							if FLAGS.corpus == "ace":
								print (micro_f1(true, labs))
							elif FLAGS.corpus == "kbp37":
								print (micro_f1(true, labs))
								print (macro_f1(true, labs))
							else:
								print (macro_f1(true, labs))
							f1_results.write("step: " + str(current_step) + " acc: " + str(acc) + " f1: " + str(macro_f1(true, labs)) + "\n")
						except Exception as inst:
							print (str(inst))

			# write html file from here
			html_string = "<h2> corpus: " + FLAGS.corpus + "; encode whole sentence " + FLAGS.use_whole_sentence + "</h2>"

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

			# convert ids back to words 

			encs = [[preprocessing.id2word[w] for w in sent if w != 0] for sent in preprocessing.test[0]]

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

			for enc, sent_info, lab_true, lab_pred, att_scores, pos_enc, pos_quer in zip(encs, preprocessing.sentences_info, labs_true, labs_pred, attention_scores, pos_encs, preprocessing.test[4]):
				pos_q_1 = pos_quer[0]
				pos_q_2 = pos_quer[1]

				att_score_q1 = att_scores[0][:len(enc)]
				att_score_q2 = att_scores[1][:len(enc)]

				html_string += write_html_file(enc, sent_info,att_score_q1, att_score_q2, lab_pred, lab_true)
			if FLAGS.use_whole_sentence == "no":
				with open("html_results_only_e1_e2_" + FLAGS.corpus + ".html", "w") as outfile:
					outfile.write(html_string)
			else:
				with open("html_results_whole_sentence_" + FLAGS.corpus + ".html", "w") as outfile:
					outfile.write(html_string)

	f1_results.close()



