'''
loosely inspired by
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import tensorflow as tf
import os
import numpy as np



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
		# queries (= [e1, e2])
		self.queries = tf.placeholder(tf.int32, shape=[None, 2])
		# position of queries
		self.query_positions = tf.placeholder(tf.int32, shape=[None, 2])


		# convert embedding matrices to tf.tensors
		self.embeddings = tf.get_variable("embedding", np.shape(self.preprocessing.embs), initializer=tf.constant_initializer(self.preprocessing.embs),dtype=tf.float32, trainable=False)
		self.position_lookup = tf.get_variable("positions", np.shape(self.position_enc), initializer=tf.constant_initializer(self.position_enc), dtype=tf.float32, trainable=False)

		# prepare encoder
		self.inputs = tf.nn.embedding_lookup(self.embeddings, self.x)
		self.mask = tf.to_float(tf.where(tf.equal(self.inputs, tf.zeros_like(self.inputs)), x=tf.zeros_like(self.inputs),y=tf.ones_like(self.inputs)))

		# normalize input?
		#self.inputs = self.normalize(self.inputs)
		#self.inputs = tf.layers.dropout(self.inputs, rate=0.1, training=True)

		self.position_inputs = tf.nn.embedding_lookup(self.position_lookup, self.positions)
		self.inputs = tf.add(self.inputs, self.position_inputs)



		
		# prepare decoder
		self.decoder_inputs = tf.nn.embedding_lookup(self.embeddings, self.queries)
		
		# normalize input?
		#self.decoder_inputs = self.normalize(self.decoder_inputs)
		#self.decoder_inputs = tf.layers.dropout(self.decoder_inputs, rate=0.1, training=True)

		self.decoder_inputs = tf.add(self.decoder_inputs, tf.nn.embedding_lookup(self.position_lookup, self.query_positions))
		self.decoder_inputs = tf.reduce_sum(self.decoder_inputs, axis=1)

		# encode sentence
		self.encoded = self.encode_sentence(self.inputs, FLAGS.num_layers, FLAGS.num_heads, dropout_rate=FLAGS.dropout)

		# query encoded sentence
		self.logits = self.decode_sentence(self.decoder_inputs, self.encoded,  FLAGS.num_layers, FLAGS.num_heads, dropout_rate=FLAGS.dropout)

		# use crossentropy or mean squared error?
		#self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

		# add l2 losses?
		#self.l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.FLAGS.l2_lambda
		#self.cost += self.l2_losses

		self.cost = tf.reduce_mean(tf.losses.mean_squared_error(self.y, tf.nn.softmax(self.logits)))
		self.learning_rate = tf.placeholder(tf.float32, shape=[])

		"""
		# do gradient clipping?
		params = tf.trainable_variables()
		gradients = tf.gradients(self.cost, params)

		max_gradient_norm = self.FLAGS.max_gradient_norm
		clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.98,epsilon=1e-09)
		self.train_step = optimizer.apply_gradients(zip(clipped_gradients, params))
		"""
		self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.98,epsilon=1e-09).minimize(self.cost)

		# predictions
		self.inference_encoded_sentence = self.encode_sentence(self.inputs, FLAGS.num_layers, FLAGS.num_heads, is_training=False)
		self.preds = self.decode_sentence(self.decoder_inputs, self.inference_encoded_sentence, FLAGS.num_layers, FLAGS.num_heads, is_training=False)
		
		self.preds = tf.nn.softmax(self.preds)
		self.predictions = tf.cast(tf.argmax(self.preds, axis=-1), tf.int32)

		#print ([n.name for n in tf.get_default_graph().as_graph_def().node])
		self.get_attention_scores = tf.get_default_graph().get_tensor_by_name("decoder_layers_0_1/multihead_attention_decoder_0_0/attention_softmax:0")


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
		"""
		if paddings:
			x = tf.multiply(inputs,self.mask)
			non_paddings = tf.reduce_sum(self.mask)
			all_elems = tf.reduce_sum(tf.ones_like(self.mask))
			mean = tf.reduce_sum(x, keep_dims=True) * all_elems / non_paddings
			mean_mask = tf.ones_like(x) * mean
			
			variance_mask = tf.where(tf.equal(x, self.mask), x=mean_mask, y=x)
			variance = tf.reduce_sum(tf.square(variance_mask - mean), axis=[-1], keep_dims=True) / non_paddings
			norm_x = (x - mean) * tf.rsqrt(variance + 1e-9)
		else:
			x = inputs
			mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
			variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
			norm_x = (x - mean) * tf.rsqrt(variance + 1e-9)

		return norm_x
		"""
		return self.normalize(inputs)


	def normalize(self, inputs, epsilon = 1e-6, scope="layer_norm", reuse=None):
		with tf.variable_scope(scope, reuse=reuse):
			inputs_shape = inputs.get_shape()
			params_shape = inputs_shape[-1:]
	
			mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
			print ("scope", scope)
			print ("layer norm shapes", mean.get_shape(), variance.get_shape())
			#beta= tf.Variable(tf.zeros(params_shape))
			#gamma = tf.Variable(tf.ones(params_shape))
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

				Q = tf.layers.dense(queries, num_units/num_heads, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer,  use_bias=False, name="queries")
				K = tf.layers.dense(keys, num_units/num_heads, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer, use_bias=False, name="keys")
				V = tf.layers.dense(keys, num_units/num_heads, activation=tf.nn.relu, kernel_initializer=tf.orthogonal_initializer, use_bias=False, name="values")
			
				# decoding step for relation classification
				# if num_heads = 3, num_units/num_heads = 300/3 = 100
				# Q = [batch_size, 2, 100] // because we only have two queries
				# K = [batch_size, sent_length, 100]
				# V = [batch_size, sent_length, 100]

				x = tf.matmul(Q, K, transpose_b=True)
				# outputs are [batch_size, 2, sent_length]

				# scaling
				x = x / (K.get_shape().as_list()[-1] ** 0.5)

				#mask_softmax = tf.where(tf.greater(outputs, tf.zeros_like(outputs)), x=outputs, y=tf.ones_like(outputs) * -100000)
				mask_softmax = tf.where(tf.equal(x, tf.zeros_like(x)), x=tf.ones_like(x) * -10000000, y=x)
				x = tf.nn.softmax(mask_softmax, name="attention_softmax")
				# rescale
				x = tf.matmul(x, V)
				# outputs are [batch_size, 2, 100]
				# concat intermediate results
				if head == 0:
					head_i = x
				else:
					head_i = tf.concat([head_i, x], axis=-1)
		with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
			x = tf.layers.dense(head_i, num_units, activation=tf.nn.relu, use_bias=False, kernel_initializer=tf.orthogonal_initializer, name="output_projection_matmul")
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
		return inputs

	def decode_sentence(self, decoder_input, encoder_input, num_layers, num_heads, is_training=True, dropout_rate=0.1):
		"""
		In "encoder-decoder attention" layers, queries come from the previous decoder layer,
		and the memory keys and values come from the output of the encoder.
		"""

		for layer in range(num_layers):
			with tf.variable_scope("decoder_layers_%d" % layer, reuse=tf.AUTO_REUSE):
				"""
				# self attention first
				self_attention = self.multihead_attention(decoder_input, decoder_input, scope="self_attention_%d" % layer, is_training=is_training)
				self_attention = tf.layers.dropout(self_attention, rate=dropout_rate, training=is_training)
				postprocess = self.add_and_norm(decoder_input, self_attention)

				# then multi head attention over encoded input
				attention = self.multihead_attention(postprocess, encoder_input, scope="multihead_attention_decoder_%d" % layer, is_training=is_training)
				attention = tf.layers.dropout(attention, rate=dropout_rate, training=is_training)
				postprocess = self.add_and_norm(postprocess, attention)			
				"""

				attention = self.multihead_attention(decoder_input, encoder_input, scope="multihead_attention_decoder_%d" % layer, is_training=is_training)
				postprocess = tf.layers.dropout(attention, rate=dropout_rate, training=is_training)
				concat = tf.reshape(postprocess, [-1, self.FLAGS.embeddings_dim])
				logits = tf.layers.dense(concat, units=self.FLAGS.num_labels, name="out")
				return logits
				# followed by feedforward
				# if not output layer
				if layer != num_layers - 1:
					feed_forward = self.pointwise_feedforward(postprocess, "ffn_%d" % layer, is_training=is_training)
					feed_forward = tf.layers.dropout(feed_forward, rate=dropout_rate, training=is_training)
					decoder_input = self.add_and_norm(postprocess, feed_forward)
				else:
					feed_forward = self.pointwise_feedforward(postprocess, "ffn_%d" % layer, is_training=is_training)
					feed_forward = tf.layers.dropout(feed_forward, rate=dropout_rate, training=is_training)
					out = self.add_and_norm(postprocess, feed_forward)

					with tf.variable_scope("classify", reuse=tf.AUTO_REUSE):
						concat = tf.reshape(out, [-1, self.FLAGS.embeddings_dim * 2])
						#concat = tf.reduce_sum(out, axis=1)
						#h1 = tf.layers.dense(inputs=concat, units=self.FLAGS.classifier_units, activation=tf.nn.relu)
						#h1 = tf.layers.dropout(inputs=h1, rate=dropout_rate, training=is_training)
						logits = tf.layers.dense(concat, units=self.FLAGS.num_labels, name="out")
		return logits

