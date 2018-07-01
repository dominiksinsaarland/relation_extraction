'''
loosely inspired by
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import tensorflow as tf
import os
import numpy as np
import sys



class model:
	def __init__(self, preprocessing, FLAGS):
		self.preprocessing = preprocessing
		self.FLAGS = FLAGS
		self.create_position_lookup()

		self.x = tf.placeholder(tf.int32, shape=[None, None])
		# labels
		self.y = tf.placeholder(tf.int32, shape=[None, None])
		# positions of tokens in sentence to encode
		self.positions = tf.placeholder(tf.int32, shape=[None, None])
		# queries (= [e1, e2]), already embedded, if query is unknown or multi-word-expression, it gets averaged
		self.queries = tf.placeholder(tf.float32, shape=[None, 2, self.FLAGS.embeddings_dim])
		# position of queries
		self.query_positions = tf.placeholder(tf.int32, shape=[None, 2])

		# convert embedding matrices to tf.tensors
		with tf.variable_scope("embeddings",reuse=tf.AUTO_REUSE):
			self.embeddings = tf.get_variable("embedding", np.shape(self.preprocessing.embs), initializer=tf.constant_initializer(self.preprocessing.embs),dtype=tf.float32, trainable=False)
		self.position_lookup = tf.get_variable("positions", np.shape(self.position_enc), initializer=tf.constant_initializer(self.position_enc), dtype=tf.float32, trainable=False)

		# prepare encoder
		self.inputs = tf.nn.embedding_lookup(self.embeddings, self.x)
		self.mask = tf.to_float(tf.where(tf.equal(self.inputs, tf.zeros_like(self.inputs)), x=tf.zeros_like(self.inputs),y=tf.ones_like(self.inputs)))

		# normalize input
		self.inputs = self.normalize(self.inputs)

		self.position_inputs = tf.nn.embedding_lookup(self.position_lookup, self.positions)

		# add positions to embedded input
		self.inputs = tf.add(self.inputs, self.position_inputs)

		# add dropout
		self.dropout_inputs = tf.layers.dropout(self.inputs, rate=FLAGS.dropout + 0.15, training=True)
		
		# prepare decoder
		self.decoder_inputs = self.queries
		# normalize decoder input
		self.decoder_inputs = self.normalize(self.decoder_inputs)

		# add positions and dropout
		self.decoder_inputs = tf.add(self.decoder_inputs, tf.nn.embedding_lookup(self.position_lookup, self.query_positions))
		self.dropout_decoder_inputs = tf.layers.dropout(self.decoder_inputs, FLAGS.dropout + 0.15, training=True)

		# encode sentence
		self.encoded = self.encode_sentence(self.dropout_inputs, FLAGS.num_layers, FLAGS.num_heads, dropout_rate=FLAGS.dropout)

		# query encoded sentence
		self.logits = self.decode_sentence(self.dropout_decoder_inputs, self.encoded,  FLAGS.num_layers, FLAGS.num_heads, dropout_rate=FLAGS.dropout)

		with tf.name_scope('cross_entropy'):
			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
			if self.FLAGS.l2_lambda != 0:
				self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.FLAGS.l2_lambda
				self.cost += self.l2_losses
			# add l2 loss for classification layer
			l2_loss = tf.losses.get_regularization_loss()
			self.cost += l2_loss
			tf.summary.scalar('cross_entropy', self.cost)
		with tf.name_scope('accuracy'):
			correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.logits, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			tf.summary.scalar('accuracy', accuracy)

		self.learning_rate = tf.placeholder(tf.float32, shape=[])

		# train step
		self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.98,epsilon=1e-09).minimize(self.cost)

		# predictions
		self.inference_encoded_sentence = self.encode_sentence(self.inputs, FLAGS.num_layers, FLAGS.num_heads, is_training=False)
		self.preds = self.decode_sentence(self.decoder_inputs, self.inference_encoded_sentence, FLAGS.num_layers, FLAGS.num_heads, is_training=False)
		
		self.preds = tf.nn.softmax(self.preds)
		self.predictions = tf.cast(tf.argmax(self.preds, axis=-1), tf.int32)
		"""
		with open("variables_graph.txt", "w") as outfile:
			outfile.write("\n".join([n.name for n in tf.get_default_graph().as_graph_def().node]))
		"""

		# get attention scores
		self.get_attention_scores_0 = tf.get_default_graph().get_tensor_by_name("decoder_layers_1/multihead_attention_decoder_1_0/attention_softmax:0")
		self.get_attention_scores_1 = tf.get_default_graph().get_tensor_by_name("decoder_layers_1/multihead_attention_decoder_1_1/attention_softmax:0")
		self.get_attention_scores_2 = tf.get_default_graph().get_tensor_by_name("decoder_layers_1/multihead_attention_decoder_1_2/attention_softmax:0")
		self.get_attention_scores_3 = tf.get_default_graph().get_tensor_by_name("decoder_layers_1/multihead_attention_decoder_1_3/attention_softmax:0")

		# logging progress in tensorboard
		for var in tf.trainable_variables():
			self.variable_summaries(var)
			tf.summary.histogram(var.name, var)
		self.merged = tf.summary.merge_all()
		self.train_writer = tf.summary.FileWriter(self.FLAGS.summaries_dir + '/train', graph=tf.get_default_graph())
		self.test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test', graph=tf.get_default_graph())


	def create_position_lookup(self):
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
		self.position_enc = np.array([np.repeat([pos / np.power(10000, 2*i/self.FLAGS.embeddings_dim) for i in range(self.FLAGS.embeddings_dim // 2)], 2) for pos in range(1, self.preprocessing.max_length + 2)])
		self.position_enc[:, 0::2] = np.sin(self.position_enc[:, 0::2])  # dim 2i
		self.position_enc[:, 1::2] = np.cos(self.position_enc[:, 1::2])  # dim 2i+1
		# add padding token at row 0, just np.zeros
		self.position_enc = np.concatenate((np.expand_dims(np.zeros(self.FLAGS.embeddings_dim), 0), self.position_enc), axis=0)
		return self

	def pointwise_feedforward(self, inputs, scope, is_training=True):
		"""
		following section "3.3 Position-wise Feed-Forward Networks" in "attention is all you need":
		FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
		each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This
		consists of two linear transformations with a ReLU activation in between
		"""

		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			print (inputs.get_shape())
			h1 = tf.layers.dense(inputs=inputs, units= 4 * self.FLAGS.embeddings_dim, kernel_initializer=tf.orthogonal_initializer, name="feedforward", activation=tf.nn.relu)
			print (h1.get_shape())
			out = tf.layers.dense(inputs=h1, units= self.FLAGS.embeddings_dim, kernel_initializer=tf.orthogonal_initializer, name="feedforward2")

		return out

	def add_and_norm(self, old_inputs, inputs):
		"""
		We employ a residual connection [11] around each of
		the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
		LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
		itself
		"""
		inputs = tf.add(old_inputs, inputs)
		return self.normalize(inputs)

	def normalize(self, inputs, epsilon = 1e-9, scope="layer_norm", reuse=None):
		with tf.variable_scope(scope, reuse=reuse):
			inputs_shape = inputs.get_shape()
			params_shape = inputs_shape[-1:]
	
			mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
			normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
			outputs = normalized		
		return outputs

	def multihead_attention(self,queries, keys, scope="multihead_attention",is_training=True):
		num_units = self.FLAGS.embeddings_dim
		num_heads=self.FLAGS.num_heads

		"""
		following section "3.2.2 Multi-Head Attention":
		we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections [...]
		On each of these projected versions of queries, keys and values we then perform the attention function in parallel [...]
		These are concatenated and once again projected, resulting in the final values [...]
		MultiHead(Q, K, V ) = Concat(head_1 , ..., head_h )W_OUT
		where head_i = Attention(QW_i, KW_i, VW_i)
		and Attention is:
		alphas = softmax(matmul(Q, transpose(K)) / sqrt(len(dimensions_keys)))
		alphas is a matrix with dimensions sent_length * sent_length
		attention = matmul(alphas, V)
		attention is a matrix with dimensions sent_length * hidden_dims
		followed by the output projection matmul(concat(attentions), W_OUT)
		"""

		for head in range(num_heads):
			with tf.variable_scope(scope + "_" + str(head), reuse=tf.AUTO_REUSE):
				# different attention heads
				Q = tf.layers.dense(queries, num_units//num_heads, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer,  use_bias=False, name="queries")
				K = tf.layers.dense(keys, num_units//num_heads, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer, use_bias=False, name="keys")
				V = tf.layers.dense(keys, num_units//num_heads, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer, use_bias=False, name="values")

				# attention scores, matmul query and keys
				x = tf.matmul(Q, K, transpose_b=True)
				# outputs are [batch_size, 2, sent_length]

				# scaling down
				x = x / (K.get_shape().as_list()[-1] ** 0.5)

				# mask padding tokens
				mask_softmax = tf.where(tf.equal(x, tf.zeros_like(x)), x=tf.ones_like(x) * -sys.maxsize, y=x)
				x = tf.nn.softmax(mask_softmax, name="attention_softmax")
				x = tf.matmul(x, V)
				# outputs are [batch_size, 2, 100]
				# concat intermediate results
				if head == 0:
					attention_heads = x
				else:
					attention_heads = tf.concat([attention_heads, x], axis=-1)
		# output projection
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			x = tf.layers.dense(attention_heads, num_units, activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.orthogonal_initializer, name="output_projection_matmul")
		return x

	def encode_sentence(self, inputs, num_layers, num_heads, is_training=True, dropout_rate=0.1):
		# encoder layers stacked
		for layer in range(num_layers):
			with tf.variable_scope("encoder_layers_%d" % layer, reuse=tf.AUTO_REUSE):
				# attention step
				print (inputs.get_shape())
				attention = self.multihead_attention(inputs, inputs, scope="encoder_attention_%d" % layer, is_training=is_training)
				# residual connection
				print (attention.get_shape())

				"""
				We apply dropout [33] to the output of each sub-layer, before it is added to the
				sub-layer input and normalized. For the base model, we use a rate of P drop = 0.1.
				"""
				attention = tf.layers.dropout(attention, rate=dropout_rate, training=is_training)
				print (attention.get_shape())

				postprocess = self.add_and_norm(inputs, attention)
				print (postprocess.get_shape())

				# feedforward step

				feed_forward = self.pointwise_feedforward(postprocess, "feedforward_%d" % layer, is_training=is_training)

				feed_forward = tf.layers.dropout(feed_forward, rate=dropout_rate, training=is_training)
				inputs = self.add_and_norm(postprocess, feed_forward)

				# set padding tokens back to zero
				inputs = tf.where(tf.equal(self.mask, tf.ones_like(self.mask)), x=inputs, y=tf.zeros_like(self.mask))



				#inputs = tf.where(tf.equal(self.mask, tf.ones_like(self.mask)), x=postprocess, y=tf.zeros_like(self.mask))
		return inputs

	def variable_summaries(self, var):
		"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
		with tf.name_scope('summaries'):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.summary.scalar('stddev', stddev)
			tf.summary.scalar('max', tf.reduce_max(var))
			tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram(var.name, var)


	def decode_sentence(self, decoder_input, encoder_input, num_layers, num_heads, is_training=True, dropout_rate=0.1):
		"""
		In "encoder-decoder attention" layers, queries come from the previous decoder layer,
		and the memory keys and values come from the output of the encoder.
		"""

		for layer in range(num_layers):
			with tf.variable_scope("decoder_layers_%d" % layer, reuse=tf.AUTO_REUSE):

				# self attention first
				self_attention = self.multihead_attention(decoder_input, decoder_input, scope="self_attention_%d" % layer, is_training=is_training)
				self_attention = tf.layers.dropout(self_attention, rate=dropout_rate, training=is_training)
				postprocess = self.add_and_norm(decoder_input, self_attention)

				# then multi head attention over encoded input
				attention = self.multihead_attention(postprocess, encoder_input, scope="multihead_attention_decoder_%d" % layer, is_training=is_training)
				attention = tf.layers.dropout(attention, rate=dropout_rate, training=is_training)
				postprocess = self.add_and_norm(postprocess, attention)		

				# followed by feedforward
				feed_forward = self.pointwise_feedforward(postprocess, "ffn_%d" % layer, is_training=is_training)
				feed_forward = tf.layers.dropout(feed_forward, rate=dropout_rate, training=is_training)
				decoder_input = self.add_and_norm(postprocess, feed_forward)
		concat = tf.reshape(decoder_input, [-1, self.FLAGS.embeddings_dim * 2])
		with tf.variable_scope("classify", reuse=tf.AUTO_REUSE):
			# classification layer
			"""
			# rnn on top works too, might be very interesting for n-ary relations!
			cell_fw = tf.contrib.rnn.LSTMCell(50)
			cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=1-self.FLAGS.dropout)
			outputs, state = tf.nn.dynamic_rnn(cell_fw, decoder_input, dtype=tf.float32)
			print (decoder_input.get_shape())
			logits = tf.layers.dense(state.h, units=self.FLAGS.num_labels, name="out")
			"""

			logits = tf.layers.dense(concat, units=self.FLAGS.num_labels, name="out", kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.05))

		return logits

