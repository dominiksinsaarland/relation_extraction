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

		# create position embeddings
		self.create_position_lookup()
		self.position_lookup = tf.get_variable("positions", np.shape(self.position_enc), initializer=tf.constant_initializer(self.position_enc), dtype=tf.float32, trainable=False)

		# inputs and positions
		self.x = tf.placeholder(tf.float32, shape=[None, None, 1024])
		self.positions = tf.placeholder(tf.int32, shape=[None, None])

		# labels
		self.y = tf.placeholder(tf.int32, shape=[None, None])

		# queries and their positions
		self.queries = tf.placeholder(tf.float32, shape=[None, 2, 1024])
		self.query_positions = tf.placeholder(tf.int32, shape=[None, 2])


		# prepare encoder
		self.inputs = self.x
		self.mask = tf.to_float(tf.where(tf.equal(self.inputs, tf.zeros_like(self.inputs)), x=tf.zeros_like(self.inputs),y=tf.ones_like(self.inputs)))

		# add position embeddings on top
		self.position_inputs = tf.nn.embedding_lookup(self.position_lookup, self.positions)
		self.inputs = tf.add(self.inputs, self.position_inputs)
		
		# prepare decoder
		self.decoder_inputs = self.queries
		self.position_queries = tf.nn.embedding_lookup(self.position_lookup, self.query_positions)
		self.decoder_inputs = tf.add(self.decoder_inputs, self.position_queries)

		# encode sentence
		self.encoded = self.encode_sentence(self.inputs, FLAGS.num_layers, FLAGS.num_heads, dropout_rate=FLAGS.dropout)

		# query encoded sentence
		self.decoded = self.decode_sentence(self.decoder_inputs, self.encoded,  FLAGS.num_layers, FLAGS.num_heads, dropout_rate=FLAGS.dropout)

		# classify 
		self.logits = self.classify(self.decoded, dropout_rate=FLAGS.dropout)

		# loss function
		with tf.name_scope('cross_entropy'):
			self.cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.y, self.logits, label_smoothing=self.FLAGS.label_smoothing))
			# add l2 loss for classification layer
			l2_loss = tf.losses.get_regularization_loss()
			self.cost += l2_loss

		self.learning_rate = tf.placeholder(tf.float32, shape=[])

		# train step
		self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.98,epsilon=1e-09).minimize(self.cost)

		# predictions
		self.inference_encoded_sentence = self.encode_sentence(self.inputs, FLAGS.num_layers, FLAGS.num_heads, is_training=False, dropout_rate = 0)
		self.inference_decoded_sentence = self.decode_sentence(self.decoder_inputs, self.inference_encoded_sentence, FLAGS.num_layers, FLAGS.num_heads, is_training=False, dropout_rate=0)
		self.preds = tf.nn.softmax(self.classify(self.inference_decoded_sentence, dropout_rate=0, is_training=False))
	
		self.predictions = tf.cast(tf.argmax(self.preds, axis=-1), tf.int32)

		self.get_attention_scores_0 = tf.get_default_graph().get_tensor_by_name("decoder_layers_0/multihead_attention_decoder_0_0/attention_softmax:0")
		self.get_attention_scores_1 = tf.get_default_graph().get_tensor_by_name("decoder_layers_0/multihead_attention_decoder_0_1/attention_softmax:0")
		self.get_attention_scores_2 = tf.get_default_graph().get_tensor_by_name("decoder_layers_0/multihead_attention_decoder_0_2/attention_softmax:0")
		self.get_attention_scores_3 = tf.get_default_graph().get_tensor_by_name("decoder_layers_0/multihead_attention_decoder_0_3/attention_softmax:0")




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
		self.position_enc = np.array([np.repeat([pos / np.power(10000, 2*i/self.FLAGS.embeddings_dim) for i in range(self.FLAGS.embeddings_dim // 2)], 2) for pos in range(1, self.preprocessing.max_length + 10)])
		self.position_enc[:, 0::2] = np.sin(self.position_enc[:, 0::2])  # dim 2i
		self.position_enc[:, 1::2] = np.cos(self.position_enc[:, 1::2])  # dim 2i+1
		# add padding token for row 0, just np.zeros
		self.position_enc = np.concatenate((np.expand_dims(np.zeros(self.FLAGS.embeddings_dim), 0), self.position_enc), axis=0)
		return self

	def pointwise_feedforward(self, inputs, scope):
		"""
		following section "3.3 Position-wise Feed-Forward Networks" in "attention is all you need":
		FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
		each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This
		consists of two linear transformations with a ReLU activation in between
		"""

		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			h1 = tf.layers.dense(inputs=inputs, units= self.FLAGS.hidden_units_ffclayer * self.FLAGS.embeddings_dim, kernel_initializer=tf.contrib.keras.initializers.he_normal(), name="feedforward", activation=tf.nn.relu)
			out = tf.layers.dense(inputs=h1, units= self.FLAGS.embeddings_dim, kernel_initializer=tf.contrib.keras.initializers.he_normal(), name="feedforward2")

		return out

	def add_and_norm(self, old_inputs, inputs):
		"""
		We employ a residual connection [11] around each of
		the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
		LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
		itself
		"""
		inputs = tf.add(old_inputs, inputs)
		return tf.contrib.layers.layer_norm(inputs, trainable=False)

	def normalize(self, inputs, epsilon = 1e-9, scope="layer_norm", reuse=None):
		"""
		works too, but does the same thing as tf.contrib.layers.layer_norm with scalable=False
		"""
		with tf.variable_scope(scope, reuse=reuse):
			inputs_shape = inputs.get_shape()
			params_shape = inputs_shape[-1:]
	
			mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
			normalized = (inputs - mean) / ( (variance + epsilon) ** (.5) )
			outputs = normalized		
		return outputs

	def multihead_attention(self,queries, keys, scope="multihead_attention", is_training=True):
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
				Q = tf.layers.dense(queries, num_units//num_heads , activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer,  use_bias=False, name="queries")
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
			x = tf.layers.dense(attention_heads, num_units, use_bias=False, kernel_initializer=tf.orthogonal_initializer, name="output_projection_matmul")
		return x

	def encode_sentence(self, inputs, num_layers, num_heads, is_training=True, dropout_rate=0.1):
		"""
		encoder step
		performs self attention on the context (everything between e1 and e2)
		returns processed context
		"""
		# encoder layers stacked
		inputs = tf.layers.dropout(inputs, rate=dropout_rate, training=is_training)
		for layer in range(num_layers):
			with tf.variable_scope("encoder_layers_%d" % layer, reuse=tf.AUTO_REUSE):
				# attention step
				print (inputs.get_shape())
				attention = self.multihead_attention(inputs, inputs, scope="encoder_attention_%d" % layer, is_training=is_training)
				# residual connection
				"""
				We apply dropout [33] to the output of each sub-layer, before it is added to the
				sub-layer input and normalized. For the base model, we use a rate of P drop = 0.1.
				"""
				attention = tf.layers.dropout(attention, rate=dropout_rate, training=is_training)
				postprocess = self.add_and_norm(inputs, attention)

				# feedforward step

				feed_forward = self.pointwise_feedforward(postprocess, "feedforward_%d" % layer)
				feed_forward = tf.layers.dropout(feed_forward, rate=dropout_rate, training=is_training)
				inputs = self.add_and_norm(postprocess, feed_forward)

				# set padding tokens back to zero
				inputs = tf.where(tf.equal(self.mask, tf.ones_like(self.mask)), x=inputs, y=tf.zeros_like(self.mask))
		return inputs

	def decode_sentence(self, decoder_input, encoder_input, num_layers, num_heads, is_training=True, dropout_rate=0.1):
		"""
		In "encoder-decoder attention" layers, queries come from the previous decoder layer,
		and the memory keys and values come from the output of the encoder.
		everything else is the same as in the encoding step
		returns queries enriched with context information
		"""
		decoder_input = tf.layers.dropout(decoder_input, rate=dropout_rate, training=is_training)
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
				feed_forward = self.pointwise_feedforward(postprocess, "ffn_%d" % layer)
				feed_forward = tf.layers.dropout(feed_forward, rate=dropout_rate, training=is_training)
				decoder_input = self.add_and_norm(postprocess, feed_forward)
		return decoder_input

	def classify(self, decoder_input, dropout_rate, is_training=True):
		"""
		classification layer
		returns logits
		"""
		concat = tf.reshape(decoder_input, [-1, self.FLAGS.embeddings_dim * 2])
		with tf.variable_scope("classify", reuse=tf.AUTO_REUSE):
			concat = tf.layers.dropout(concat, rate=dropout_rate, training=is_training)
			logits = tf.layers.dense(concat, units=self.FLAGS.num_labels, kernel_initializer=tf.contrib.keras.initializers.he_normal(), name="out", kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.FLAGS.l2_lambda))
		return logits

