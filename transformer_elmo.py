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


# encode whole sentence, query only between e1 and e2!


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
		self.mask = tf.to_float(tf.where(tf.equal(self.positions, tf.zeros_like(self.positions)), x=tf.zeros_like(self.positions),y=tf.ones_like(self.positions)))
		#self.mask = tf.placeholder(tf.float32, shape=[None, None])
		# add position embeddings on top
		self.position_inputs = tf.nn.embedding_lookup(self.position_lookup, self.positions)
		self.inputs = tf.add(self.inputs, self.position_inputs)
		
		# prepare decoder
		if self.FLAGS.use_types == "yes":
	
			with tf.variable_scope("types_embeddings",reuse=tf.AUTO_REUSE):
				self.embeddings = tf.get_variable("types_embeddings", (self.FLAGS.num_types,self.FLAGS.types_embeddings_dim), initializer= tf.initializers.truncated_normal() ,dtype=tf.float32, trainable=True)
			self.types = tf.placeholder(tf.int32, shape=[None, 2])
			self.decoder_inputs = self.queries
			self.position_queries = tf.nn.embedding_lookup(self.position_lookup, self.query_positions)
			self.decoder_inputs = tf.add(self.decoder_inputs, self.position_queries)
			self.types_embedded = tf.nn.embedding_lookup(self.embeddings, self.types)
			self.decoder_inputs = tf.concat([self.decoder_inputs, self.types_embedded], axis=-1)
		else:
			self.decoder_inputs = self.queries
			self.position_queries = tf.nn.embedding_lookup(self.position_lookup, self.query_positions)
			self.decoder_inputs = tf.add(self.decoder_inputs, self.position_queries)

		# encode sentence
		self.encoded = self.encode_sentence(self.inputs, FLAGS.num_layers, FLAGS.num_heads, dropout_rate=FLAGS.dropout)
		#self.encoded = self.inputs
		#self.encoded = self.encode_sentence_small(self.inputs, FLAGS.num_layers, FLAGS.num_heads, dropout_rate=FLAGS.dropout)
		# query encoded sentence
		self.decoded = self.decode_sentence(self.decoder_inputs, self.encoded,  FLAGS.num_layers, FLAGS.num_heads, dropout_rate=FLAGS.dropout)

		# classify 
		self.logits = self.classify(self.decoded, dropout_rate=FLAGS.dropout)

		# loss function
		self.weights = tf.placeholder(tf.float32, shape=[None])
		with tf.name_scope('cross_entropy'):
			self.cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.y, self.logits, label_smoothing=self.FLAGS.label_smoothing, weights=self.weights))
			# add l2 loss for classification layer
			l2_loss = tf.losses.get_regularization_loss()
			self.cost += l2_loss

		self.learning_rate = tf.placeholder(tf.float32, shape=[])

		# train step
		self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.98,epsilon=1e-09).minimize(self.cost)

		# predictions
		#self.inference_encoded_sentence = self.encode_sentence_small(self.inputs, FLAGS.num_layers, FLAGS.num_heads, is_training=False, dropout_rate = 0)
		self.inference_encoded_sentence = self.encode_sentence(self.inputs, FLAGS.num_layers, FLAGS.num_heads, is_training=False, dropout_rate = 0)
		self.inference_decoded_sentence = self.decode_sentence(self.decoder_inputs, self.inference_encoded_sentence, FLAGS.num_layers, FLAGS.num_heads, is_training=False, dropout_rate=0)
		self.preds = tf.nn.softmax(self.classify(self.inference_decoded_sentence, dropout_rate=0, is_training=False))
	
		self.predictions = tf.cast(tf.argmax(self.preds, axis=-1), tf.int32)

		self.get_attention_scores = [tf.get_default_graph().get_tensor_by_name("decoder_layers_0/multihead_attention_decoder_0_" + str(i) + "/attention_softmax:0") for i in range(self.FLAGS.num_heads)]




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
		self.position_enc = np.array([np.repeat([pos / np.power(10000, 2*i/1024) for i in range(1024 // 2)], 2) for pos in range(1, self.preprocessing.max_length + 10)])
		self.position_enc[:, 0::2] = np.sin(self.position_enc[:, 0::2])  # dim 2i
		self.position_enc[:, 1::2] = np.cos(self.position_enc[:, 1::2])  # dim 2i+1
		# add padding token for row 0, just np.zeros
		self.position_enc = np.concatenate((np.expand_dims(np.zeros(1024), 0), self.position_enc), axis=0)
		#self.position_enc = np.zeros(np.shape(self.position_enc))
		#self.position_enc /= 2

		return self

	def pointwise_feedforward(self, inputs, scope):
		"""
		following section "3.3 Position-wise Feed-Forward Networks" in "attention is all you need":
		FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
		each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This
		consists of two linear transformations with a ReLU activation in between
		"""
		num_units = inputs.get_shape().as_list()[-1]
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			h1 = tf.layers.dense(inputs=inputs, units= self.FLAGS.hidden_units_ffclayer * num_units, kernel_initializer=tf.contrib.keras.initializers.he_normal(), name="feedforward", activation=tf.nn.relu)
			out = tf.layers.dense(inputs=h1, units= num_units, kernel_initializer=tf.contrib.keras.initializers.he_normal(), name="feedforward2")

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

	def multihead_attention(self,queries, keys, num_heads, scope="multihead_attention", is_training=True,output_units = 1024):
		num_units = queries.get_shape().as_list()[-1]

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
				
				#Q = tf.layers.dense(queries, self.FLAGS.hidden_dim // num_heads, kernel_initializer=tf.orthogonal_initializer, use_bias=False, name="queries")
				#K = tf.layers.dense(keys, self.FLAGS.hidden_dim // num_heads, kernel_initializer=tf.orthogonal_initializer, use_bias=False, name="keys")
				#V = tf.layers.dense(keys, self.FLAGS.hidden_dim // num_heads, kernel_initializer=tf.orthogonal_initializer, use_bias=False, name="values")
			
				# attention scores, matmul query and keys
				x = tf.matmul(Q, K, transpose_b=True)
				# outputs are [batch_size, 2, sent_length]

				# scaling down
				x = tf.divide(x, K.get_shape().as_list()[-1] ** 0.5)
				# mask padding tokens
				mask_softmax = tf.where(tf.equal(x, tf.zeros_like(x)), x=tf.ones_like(x) * -sys.maxsize, y=x)
				x = tf.nn.softmax(mask_softmax, name="attention_softmax")
				x = tf.matmul(x, V)
				# outputs are [batch_size, 2, 100]
				# concat intermediate results
				print (x.get_shape())
				if head == 0:
					attention_heads = x
				else:
					attention_heads = tf.concat([attention_heads, x], axis=-1)

		# output projection
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			#x = tf.layers.dense(attention_heads, num_units, use_bias=False, kernel_initializer=tf.orthogonal_initializer, name="output_projection_matmul")
			x = tf.layers.dense(attention_heads, num_units,kernel_initializer=tf.contrib.keras.initializers.he_normal(), use_bias=False, name="output_projection_matmul", activation=tf.nn.relu)
			#residual_connection = tf.layers.dense(queries, self.FLAGS.hidden_dim, kernel_initializer=tf.contrib.keras.initializers.he_normal(), use_bias=False, name="output_projection_residual")
		#return x, residual_connection
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
				#attention, residual = self.multihead_attention(inputs, inputs, scope="encoder_attention_%d" % layer, is_training=is_training)
				# residual connection
				"""
				We apply dropout [33] to the output of each sub-layer, before it is added to the
				sub-layer input and normalized. For the base model, we use a rate of P drop = 0.1.
				"""
				if self.FLAGS.encoder_attention == "yes":
					#attention, residual = self.multihead_attention(inputs, inputs, self.FLAGS.num_heads, scope="multihead_attention_decoder_%d" % layer,is_training=is_training)
					attention = self.multihead_attention(inputs, inputs, self.FLAGS.num_heads, scope="multihead_attention_decoder_%d" % layer,is_training=is_training)
					attention = tf.layers.dropout(attention, rate=dropout_rate, training=is_training)
					print (attention.get_shape())
					inputs = self.add_and_norm(inputs, attention)

				if self.FLAGS.encoder_feedforward == "yes":
					feed_forward = self.pointwise_feedforward(inputs, "feedforward_%d" % layer)
					feed_forward = tf.layers.dropout(feed_forward, rate=dropout_rate, training=is_training)
					inputs = self.add_and_norm(inputs, feed_forward)
				inputs = tf.multiply(inputs, tf.expand_dims(self.mask, axis=-1))
		return inputs

	"""
	def tensor_layer(self, inputs):
		with tf.variable_scope("tensor_layer", reuse=tf.AUTO_REUSE):
			vec_1 = inputs[:,:1]
			vec_2 = inputs[:,1:2]
			outer_product = tf.matmul(tf.expand_dims(vec_1, axis=-1), tf.expand_dims(vec_2, axis=1))
			dt, dh = 250, 250

			Wt_1 = tf.get_variable("squared_interaction", (dt, 2*dh, 2*dh), initializer= tf.initializers.truncated_normal , dtype=tf.float32, trainable=True)
			M = tf.tensordot(vec_1,vec_2, axes=0)
			
			pb = 
	"""
	def decode_sentence(self, decoder_input, encoder_input, num_layers, num_heads, is_training=True, dropout_rate=0.1):
		"""
		In "encoder-decoder attention" layers, queries come from the previous decoder layer,
		and the memory keys and values come from the output of the encoder.
		everything else is the same as in the encoding step
		returns queries enriched with context information
		"""
		decoder_input = tf.layers.dropout(decoder_input, rate=dropout_rate, training=is_training)
		#encoder_input = tf.layers.dropout(encoder_input, rate=dropout_rate, training=is_training)
		for layer in range(num_layers):
			with tf.variable_scope("decoder_layers_%d" % layer, reuse=tf.AUTO_REUSE):
				# self attention first
				if self.FLAGS.decoder_self_attention == "yes":
					#self_attention, residual = self.multihead_attention(decoder_input, decoder_input, 1, scope="self_attention_%d" % layer, is_training=is_training)
					self_attention = self.multihead_attention(decoder_input, decoder_input, 1, scope="self_attention_%d" % layer, is_training=is_training)
					self_attention = tf.layers.dropout(self_attention, rate=dropout_rate, training=is_training)
					decoder_input = self.add_and_norm(decoder_input, self_attention)

				# then multi head attention over encoded input
				if self.FLAGS.decoder_encoder_attention == "yes":
					#attention, residual = self.multihead_attention(decoder_input, encoder_input, self.FLAGS.num_heads, scope="multihead_attention_decoder_%d" % layer, is_training=is_training)
					attention = self.multihead_attention(decoder_input, encoder_input, self.FLAGS.num_heads, scope="multihead_attention_decoder_%d" % layer, is_training=is_training)
					attention = tf.layers.dropout(attention, rate=dropout_rate, training=is_training)
					decoder_input = self.add_and_norm(decoder_input, attention)
				if layer == num_layers - 1:
					#postprocess = tf.reshape(postprocess, [-1, self.FLAGS.embeddings_dim * 2])
					decoder_input = tf.reshape(decoder_input, [-1, decoder_input.get_shape().as_list()[-1] * 2])

				if self.FLAGS.decoder_feedforward == "yes":
					feed_forward = self.pointwise_feedforward(decoder_input, "ffn_%d" % layer)
					feed_forward = tf.layers.dropout(feed_forward, rate=dropout_rate, training=is_training)
					decoder_input = self.add_and_norm(decoder_input, feed_forward)
					#decoder_input = feed_forward
				print (decoder_input.get_shape())
		return decoder_input

	def classify(self, decoder_input, dropout_rate, is_training=True):
		"""
		classification layer
		returns logits
		"""
		#decoder_input = tf.reshape(decoder_input, [-1, self.FLAGS.hidden_dim * 2])

		with tf.variable_scope("classify", reuse=tf.AUTO_REUSE):
			#concat = tf.layers.dropout(concat, rate=dropout_rate, training=is_training)
			"""

			num_units = decoder_input.get_shape().as_list()[-1]
			decoder_input = tf.layers.dense(inputs=decoder_input, units= num_units//2, kernel_initializer=tf.contrib.keras.initializers.he_normal(), name="classify", activation=tf.nn.relu)
			"""
			decoder_input = tf.layers.dropout(decoder_input, rate=0.1, training=is_training)
			logits = tf.layers.dense(decoder_input, units=self.FLAGS.num_labels, kernel_initializer=tf.contrib.keras.initializers.he_normal(), name="out", kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.FLAGS.l2_lambda))
		return logits

