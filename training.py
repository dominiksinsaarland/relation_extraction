from preprocessing import *
from transformers import *

def get_batch(tup, old, new):
	return [x[old:new] for x in tup]


def write_html_file(sent, q1, q2):
	html_string = ""
	#with open("colors_highlighted.html", "a") as outfile:
	html_string += "<p>"
	for w, c1, c2 in zip(sent, q1, q2):
		html_string += ' <span style="background-color: rgb(255,255,' + str(int(255 - c1 * 255)) + '">' + w + " </span>"
	html_string += "</p>\n"
	html_string += "<p>"
	for w, c1, c2 in zip(sent, q1, q2):
		html_string += ' <span style="background-color: rgb(255,' + str(int(255 - c2 * 255)) + ',255">' + w + " </span>"

	html_string += "</p>\n"
	return html_string	



if __name__ == "__main__":


	flags = tf.flags

	# some of the flags are not needed anymore
	#flags.DEFINE_float("learning_rate", 0.001, "") // use adaptive learning rate instead, see explanations below
	flags.DEFINE_integer("num_labels", 19, "number of target labels")
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
					labs = 	sess.run(model.predictions, feed_dict={model.x: preprocessing.test[0], model.y: preprocessing.test[1], model.positions:preprocessing.test[2], model.queries:preprocessing.test[3], model.query_positions:preprocessing.test[4]}) 
					acc = sum([1 for i,j in zip(labs, preprocessing.test[1]) if i == np.argmax(j)])/len(preprocessing.test[1])
					print ("step:", current_step, "acc", acc)			
					labs = [id2labels[i] for i in labs]
					true = [id2labels[np.argmax(i)] for i in preprocessing.test[1]]
					# to not break the program in the first few epochs because of zero division errors
					try:
						print (macro_f1(true, labs))
					except Exception as inst:
						print (str(inst))

		# write html file from here

		html_string = ""

		# predictions
		labs = 	sess.run(model.predictions, feed_dict={model.x: preprocessing.test[0], model.y: preprocessing.test[1], model.positions:preprocessing.test[2], model.queries:preprocessing.test[3], model.query_positions:preprocessing.test[4]})

		# convert ids back to words 
		encs, quers = [[preprocessing.id2word[w] for w in sent if w != 0] for sent in x_global], [[preprocessing.id2word[w] for w in query] for query in queries_global]
		labs_true, labs_pred = [id2labels[np.argmax(j)] for j in y_test], [id2labels[i] for i in labs]

		# get the attention scores
		attention_scores = sess.run(model.get_attention_scores, feed_dict={model.x: preprocessing.test[0], model.y: preprocessing.test[1], model.positions:preprocessing.test[2], model.queries:preprocessing.test[3], model.query_positions:preprocessing.test[4]})
		pos_encs = [[w for w in sent if w != 0] for sent in positions_global]

		# iterate through sentences and write the html file
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
			html_string += write_html_file(enc, att_score_q1, att_score_q2)
		with open("html_results.html", "w") as outfile:
			outfile.write(html_string)



