from preprocessing_ace import *
from transformers import *
from collections import defaultdict

def get_batch(tup, old, new):
	return [x[old:new] for x in tup]


def write_html_file(sent, q1, q2, pos_q1, pos_q2, predicted, true):
	html_string = ""

	# make this: "background-color: rgb(255,255,140);border: 1px solid black" around entities
	#with open("colors_highlighted.html", "a") as outfile:
	if predicted == true:
		html_string += '<p> true label: "' + true + '"; predicted label: "' + predicted + '"; correct prediction</p>'
	else:
		html_string += '<p> true label: "' + true + '"; predicted label: "' + predicted + '"; wrong prediction</p>'
	html_string += "<p>"
	for i,w, c1, c2 in zip(range(len(sent)), sent, q1, q2):
		if i == pos_q1 - 1:
			html_string += ' <span style="background-color: rgb(255,255,' + str(int(255 - c1 * 255)) + ');border: 1px solid black">' + w + " </span>"
		else:
			html_string += ' <span style="background-color: rgb(255,255,' + str(int(255 - c1 * 255)) + ')">' + w + " </span>"
	html_string += "</p>\n"
	html_string += "<p>"
	for i,w, c1, c2 in zip(range(len(sent)), sent, q1, q2):
		if i == pos_q2 - 1:
			html_string += ' <span style="background-color: rgb(255,' + str(int(255 - c2 * 255)) + ',255);border: 1px solid black">'+ w + " </span>"
		else:
			html_string += ' <span style="background-color: rgb(255,' + str(int(255 - c2 * 255)) + ',255)">' + w + " </span>"
	html_string += "<br><br></p>\n"
	return html_string	


def micro_f1(y_true, y_pred):
	d = defaultdict(int)
	OTHER_RELATION = "NO_RELATION(Arg-1,Arg-1)"
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
	#return TP, FP, FN, "Precision: " + "{0:.2f}".format(Pr * 100), "Recall: " + "{0:.2f}".format(Rc * 100), "F1 measure: " + "{0:.2f}".format(F1 * 100)
	return "{0:.2f}".format(Pr * 100), "{0:.2f}".format(Rc * 100), "{0:.2f}".format(F1 * 100)

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


if __name__ == "__main__":


	flags = tf.flags

	# some of the flags are not needed anymore
	#flags.DEFINE_float("learning_rate", 0.001, "") // use adaptive learning rate instead, see explanations below
	flags.DEFINE_integer("num_labels", 11, "number of target labels")
	flags.DEFINE_integer("batch_size", 500, "number of batchsize, bigger works better")
	flags.DEFINE_float("dropout", 0.2, "dropout applied after each layer")
	#flags.DEFINE_integer("sent_length", 50, "sentence length")
	flags.DEFINE_integer("num_layers", 1, "num layers for encoding/decoding")
	flags.DEFINE_integer("num_heads",1, "num heads per layer")
	#flags.DEFINE_integer("num_epochs",20, "")
	#flags.DEFINE_integer("min_length", 0, "min length of encoded sentence")
	flags.DEFINE_integer("embeddings_dim", 300, "number of dimensions in word embeddings")
	#flags.DEFINE_float("l2_lambda", 0.0001, "")
	#flags.DEFINE_integer("max_gradient_norm", 5, "")
	#flags.DEFINE_integer("classifier_units", 100, "")
	flags.DEFINE_integer("warmup_steps", 1000, "")
	flags.DEFINE_integer("max_steps", 3000, "")


	FLAGS = flags.FLAGS
	preprocessing = preprocessing(FLAGS)
	with open(preprocessing.labels_file) as labs:
		labels2id = {lab.strip(): i for i, lab in enumerate(labs)}
		id2labels = {j:i for i,j in labels2id.items()}


	"""
	for i in range(5):
		print ([preprocessing.id2word[w] for (i,w) in np.ndenumerate(preprocessing.train[0][i]) if preprocessing.id2word[w] != "__PADDING__"])
		print ([w for (i,w) in np.ndenumerate(preprocessing.train[2][i]) if w != 0])
		print ([preprocessing.id2word[w] for (i,w) in np.ndenumerate(preprocessing.train[3][i])])
		print ([w for (i,w) in np.ndenumerate(preprocessing.train[4][i])])
	"""

	model = model(preprocessing, FLAGS)
	init = tf.global_variables_initializer()
	current_step = 0
	with tf.Session() as sess:
		sess.run(init)
		while current_step < FLAGS.max_steps:
			p = np.random.permutation(len(preprocessing.train[0]))
			#print ([np.shape(x) for x in preprocessing.train])
			train = tuple([x[p] for x in preprocessing.train])

			for batch in range(len(train[0]) // FLAGS.batch_size):
				current_step += 1
				current_lr = (FLAGS.embeddings_dim ** -0.5) * min(current_step ** -0.5, current_step * FLAGS.warmup_steps ** -1.5)
				old = FLAGS.batch_size * batch
				new = FLAGS.batch_size * (batch + 1)
				# load batch data
				this_batch = get_batch(train, old, new)
				sess.run(model.train_step, feed_dict={model.x: this_batch[0], model.y: this_batch[1], model.positions:this_batch[2], model.queries:this_batch[3],model.query_positions:this_batch[4], model.learning_rate:current_lr}) 

				if current_step % 100 == 0:
					labs = []
					for batch in range(len(preprocessing.test[0]) // FLAGS.batch_size):
						old = FLAGS.batch_size * batch
						new = FLAGS.batch_size * (batch + 1)
						this_batch = get_batch(preprocessing.test, old, new)						
						labs += list(sess.run(model.predictions, feed_dict={model.x: this_batch[0], model.y: this_batch[1], model.positions:this_batch[2], model.queries:this_batch[3],model.query_positions:this_batch[4]}))
					acc = sum([1 for i,j in zip(labs, preprocessing.test[1]) if i == np.argmax(j)])/len(preprocessing.test[1])
					print ("step:", current_step, "acc", acc)			
					labs = [id2labels[i] for i in labs]
					true = [id2labels[np.argmax(i)] for i in preprocessing.test[1]]
					# to not break the program in the first few epochs because of zero division errors
					try:
						print (micro_f1(true, labs))
					except Exception as inst:
						print (str(inst))

		# write html file from here

		html_string = ""
		# predictions
		labs = []
		attention_scores = []
		for batch in range(len(preprocessing.test[0]) // FLAGS.batch_size):
			old = FLAGS.batch_size * batch
			new = FLAGS.batch_size * (batch + 1)
			this_batch = get_batch(preprocessing.test, old, new)						
			labs += list(sess.run(model.predictions, feed_dict={model.x: this_batch[0], model.y: this_batch[1], model.positions:this_batch[2], model.queries:this_batch[3],model.query_positions:this_batch[4]}))
			attention_scores += list(sess.run(model.get_attention_scores, feed_dict={model.x: this_batch[0], model.y: this_batch[1], model.positions:this_batch[2], model.queries:this_batch[3],model.query_positions:this_batch[4]}))

		# convert ids back to words 
		encs, quers = [[preprocessing.id2word[w] for w in sent if w != 0] for sent in preprocessing.test[0]], [[preprocessing.id2word[w] for w in query] for query in preprocessing.test[3]]
		labs_true, labs_pred = [id2labels[np.argmax(j)] for j in preprocessing.test[1]], [id2labels[i] for i in labs]
		pos_encs = [[w for w in sent if w != 0] for sent in preprocessing.test[2]]

		# iterate through sentences and write the html file
		for enc, quer, lab_true, lab_pred, att_scores, pos_enc, pos_quer in zip(encs, quers, labs_true, labs_pred, attention_scores, pos_encs, preprocessing.test[4]):
			pos_q_1 = str(pos_quer[0])
			pos_q_2 = str(pos_quer[1])


			att_score_q1 = att_scores[0][:len(enc)]
			att_score_q2 = att_scores[1][:len(enc)]
			"""
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
			html_string += write_html_file(enc, att_score_q1, att_score_q2, pos_q_1, pos_q_2, lab_pred, lab_true)
		
		with open("html_results_ace2005.html", "w") as outfile:
			outfile.write(html_string)



