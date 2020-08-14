#!/usr/bin/env python
# coding: utf-8

from clusc.utils import *
import os

import numpy as np
import tensorflow as tf
import scipy.sparse as sp


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_integer('hidden3', 64, 'Number of units in hidden layer 3.')
flags.DEFINE_integer('discriminator_out', 0, 'discriminator_out.')
flags.DEFINE_float('discriminator_learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate', .5*0.001, 'Initial learning rate.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('seed', 8853, 'seed for fixing the results.')
flags.DEFINE_integer('iterations', 50, 'number of iterations.')

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


class Model(object):
	def __init__(self, **kwargs):
		allowed_kwargs = {'name', 'logging'}
		for kwarg in kwargs.keys():
			assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

		for kwarg in kwargs.keys():
			assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
		name = kwargs.get('name')
		if not name:
			name = self.__class__.__name__.lower()
		self.name = name

		logging = kwargs.get('logging', False)
		self.logging = logging

		self.vars = {}

	def _build(self):
		raise NotImplementedError

	def build(self):
		""" Wrapper for _build() """
		with tf.variable_scope(self.name):
			self._build()
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		self.vars = {var.name: var for var in variables}

	def fit(self):
		pass

	def predict(self):
		pass


class ARGA(Model):
	def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
		super(ARGA, self).__init__(**kwargs)

		self.inputs = placeholders['features']
		self.input_dim = num_features
		self.features_nonzero = features_nonzero
		self.adj = placeholders['adj']
		self.dropout = placeholders['dropout']
		self.build()

	def _build(self):

		with tf.variable_scope('Encoder', reuse=None):
			self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
												  output_dim=FLAGS.hidden1,
												  adj=self.adj,
												  features_nonzero=self.features_nonzero,
												  act=tf.nn.relu,
												  dropout=self.dropout,
												  logging=self.logging,
												  name='e_dense_1')(self.inputs)
												  
												  
			self.noise = gaussian_noise_layer(self.hidden1, 0.1)

			self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
										   output_dim=FLAGS.hidden2,
										   adj=self.adj,
										   act=lambda x: x,
										   dropout=self.dropout,
										   logging=self.logging,
										   name='e_dense_2')(self.noise)


			self.z_mean = self.embeddings

			self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
										  act=lambda x: x,
										  logging=self.logging)(self.embeddings)


class ARVGA(Model):
	def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
		super(ARVGA, self).__init__(**kwargs)

		self.inputs = placeholders['features']
		self.input_dim = num_features
		self.features_nonzero = features_nonzero
		self.n_samples = num_nodes
		self.adj = placeholders['adj']
		self.dropout = placeholders['dropout']
		self.build()

	def _build(self):
		with tf.variable_scope('Encoder'):
			self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
												  output_dim=FLAGS.hidden1,
												  adj=self.adj,
												  features_nonzero=self.features_nonzero,
												  act=tf.nn.relu,
												  dropout=self.dropout,
												  logging=self.logging,
												  name='e_dense_1')(self.inputs)

			self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
										   output_dim=FLAGS.hidden2,
										   adj=self.adj,
										   act=lambda x: x,
										   dropout=self.dropout,
										   logging=self.logging,
										   name='e_dense_2')(self.hidden1)

			self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
											  output_dim=FLAGS.hidden2,
											  adj=self.adj,
											  act=lambda x: x,
											  dropout=self.dropout,
											  logging=self.logging,
											  name='e_dense_3')(self.hidden1)

			self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

			self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
										  act=lambda x: x,
										  logging=self.logging)(self.z)
			self.embeddings = self.z


def dense(x, n1, n2, name):
	"""
	Used to create a dense layer.
	:param x: input tensor to the dense layer
	:param n1: no. of input neurons
	:param n2: no. of output neurons
	:param name: name of the entire dense layer.i.e, variable scope name.
	:return: tensor with shape [batch_size, n2]
	"""
	with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		# np.random.seed(1)
		tf.set_random_seed(1)
		weights = tf.get_variable("weights", shape=[n1, n2],
								  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
		bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
		out = tf.add(tf.matmul(x, weights), bias, name='matmul')
		return out


class Discriminator(Model):
	def __init__(self, **kwargs):
		super(Discriminator, self).__init__(**kwargs)

		self.act = tf.nn.relu

	def construct(self, inputs, reuse = False):
		# with tf.name_scope('Discriminator'):
		with tf.variable_scope('Discriminator'):
			if reuse:
				tf.get_variable_scope().reuse_variables()
			# np.random.seed(1)
			tf.set_random_seed(1)
			dc_den1 = tf.nn.relu(dense(inputs, FLAGS.hidden2, FLAGS.hidden3, name='dc_den1'))
			dc_den2 = tf.nn.relu(dense(dc_den1, FLAGS.hidden3, FLAGS.hidden1, name='dc_den2'))
			output = dense(dc_den2, FLAGS.hidden1, 1, name='dc_output')
			return output
		

def gaussian_noise_layer(input_layer, std):
	noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
	return input_layer + noise


def weight_variable_glorot(input_dim, output_dim, name=""):
	"""Create a weight variable with Glorot & Bengio (AISTATS 2010)
	initialization.
	"""
	init_range = np.sqrt(6.0 / (input_dim + output_dim))
	initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
								maxval=init_range, dtype=tf.float32)
	return tf.Variable(initial, name=name)


def get_layer_uid(layer_name=''):
	"""Helper function, assigns unique layer IDs
	"""
	if layer_name not in _LAYER_UIDS:
		_LAYER_UIDS[layer_name] = 1
		return 1
	else:
		_LAYER_UIDS[layer_name] += 1
		return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
	"""Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
	"""
	noise_shape = [num_nonzero_elems]
	random_tensor = keep_prob
	random_tensor += tf.random_uniform(noise_shape)
	dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
	pre_out = tf.sparse_retain(x, dropout_mask)
	return pre_out * (1./keep_prob)


class Layer(object):
	"""Base layer class. Defines basic API for all layer objects.

	# Properties
		name: String, defines the variable scope of the layer.

	# Methods
		_call(inputs): Defines computation graph of layer
			(i.e. takes input, returns output)
		__call__(inputs): Wrapper for _call()
	"""
	def __init__(self, **kwargs):
		allowed_kwargs = {'name', 'logging'}
		for kwarg in kwargs.keys():
			assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
		name = kwargs.get('name')
		if not name:
			layer = self.__class__.__name__.lower()
			name = layer + '_' + str(get_layer_uid(layer))
		self.name = name
		self.vars = {}
		logging = kwargs.get('logging', False)
		self.logging = logging
		self.issparse = False

	def _call(self, inputs):
		return inputs

	def __call__(self, inputs):
		with tf.name_scope(self.name):
			outputs = self._call(inputs)
			return outputs


class GraphConvolution(Layer):
	"""Basic graph convolution layer for undirected graph without edge labels."""
	def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
		super(GraphConvolution, self).__init__(**kwargs)
		with tf.variable_scope(self.name + '_vars'):
			self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
		self.dropout = dropout
		self.adj = adj
		self.act = act

	def _call(self, inputs):
		x = inputs
		x = tf.nn.dropout(x, 1-self.dropout)
		x = tf.matmul(x, self.vars['weights'])
		x = tf.sparse_tensor_dense_matmul(self.adj, x)
		outputs = self.act(x)
		return outputs


class GraphConvolutionSparse(Layer):
	"""Graph convolution layer for sparse inputs."""
	def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
		super(GraphConvolutionSparse, self).__init__(**kwargs)
		with tf.variable_scope(self.name + '_vars'):
			self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
		self.dropout = dropout
		self.adj = adj
		self.act = act
		self.issparse = True
		self.features_nonzero = features_nonzero

	def _call(self, inputs):
		x = inputs
		x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
		x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
		x = tf.sparse_tensor_dense_matmul(self.adj, x)
		outputs = self.act(x)
		return outputs


class InnerProductDecoder(Layer):
	"""Decoder model layer for link prediction."""
	def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
		super(InnerProductDecoder, self).__init__(**kwargs)
		self.dropout = dropout
		self.act = act

	def _call(self, inputs):
		inputs = tf.nn.dropout(inputs, 1-self.dropout)
		x = tf.transpose(inputs)
		x = tf.matmul(inputs, x)
		x = tf.reshape(x, [-1])
		outputs = self.act(x)
		return outputs


class OptimizerAE(object):
	def __init__(self, preds, labels, pos_weight, norm, d_real, d_fake):
		preds_sub = preds
		labels_sub = labels

		self.real = d_real

		# Discrimminator Loss
		self.dc_loss_real = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.real), logits=self.real, name='dclreal'))

		self.dc_loss_fake = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake, name='dcfake'))
		self.dc_loss = self.dc_loss_fake + self.dc_loss_real

		# Generator loss
		generator_loss = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake, name='gl'))



		self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
		self.generator_loss = generator_loss + self.cost
		self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

		all_variables = tf.trainable_variables()
		dc_var = [var for var in all_variables if 'dc_' in var.name]
		en_var = [var for var in all_variables if 'e_' in var.name]

		with tf.variable_scope(tf.get_variable_scope()):
			self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
															 beta1=0.9, name='adam1').minimize(self.dc_loss, var_list=dc_var) #minimize(dc_loss_real, var_list=dc_var)

			self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
														 beta1=0.9, name='adam2').minimize(self.generator_loss, var_list=en_var)

		self.opt_op = self.optimizer.minimize(self.cost)
		self.grads_vars = self.optimizer.compute_gradients(self.cost)


class OptimizerVAE(object):
	def __init__(self, preds, labels, model, num_nodes, pos_weight, norm, d_real, d_fake):
		preds_sub = preds
		labels_sub = labels

		# Discrimminator Loss
		dc_loss_real = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
		dc_loss_fake = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
		self.dc_loss = dc_loss_fake + dc_loss_real

		# Generator loss
		self.generator_loss = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

		self.cost = norm * tf.reduce_mean(
			tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

		all_variables = tf.trainable_variables()
		dc_var = [var for var in all_variables if 'dc_' in var.op.name]
		en_var = [var for var in all_variables if 'e_' in var.op.name]

		with tf.variable_scope(tf.get_variable_scope(), reuse=False):
			self.discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
																  beta1=0.9, name='adam1').minimize(self.dc_loss, var_list=dc_var)#minimize(dc_loss_real, var_list=dc_var)

			self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.discriminator_learning_rate,
															  beta1=0.9, name='adam2').minimize(self.generator_loss,
																								var_list=en_var)

		self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
		self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

		# Latent loss
		self.log_lik = self.cost
		self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
																   tf.square(tf.exp(model.z_log_std)), 1))
		self.cost -= self.kl

		self.opt_op = self.optimizer.minimize(self.cost)
		self.grads_vars = self.optimizer.compute_gradients(self.cost)


def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj):
	# Construct feed dictionary
	feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
	feed_dict.update({placeholders['dropout']: FLAGS.dropout})

	feed_dict.update({placeholders['dropout']: 0})
	emb = sess.run(model.z_mean, feed_dict=feed_dict)

	z_real_dist = np.random.randn(adj.shape[0], FLAGS.hidden2)
	feed_dict.update({placeholders['real_distribution']: z_real_dist})

	for j in range(5):
		_, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
	d_loss, _ = sess.run([opt.dc_loss, opt.discriminator_optimizer], feed_dict=feed_dict)
	g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)

	avg_cost = reconstruct_loss

	return emb, avg_cost


def get_placeholder(adj):
	placeholders = {
		'features': tf.sparse_placeholder(tf.float32),
		'adj': tf.sparse_placeholder(tf.float32),
		'adj_orig': tf.sparse_placeholder(tf.float32),
		'dropout': tf.placeholder_with_default(0., shape=()),
		'real_distribution': tf.placeholder(dtype=tf.float32, shape=[adj.shape[0], FLAGS.hidden2],
											name='real_distribution')
	}

	return placeholders


def get_model(model_str, placeholders, num_features, num_nodes, features_nonzero):
	discriminator = Discriminator()
	d_real = discriminator.construct(placeholders['real_distribution'])
	model = None
	if model_str == 'arga_ae':
		model = ARGA(placeholders, num_features, features_nonzero)

	elif model_str == 'arga_vae':
		model = ARVGA(placeholders, num_features, num_nodes, features_nonzero)

	return d_real, discriminator, model


def get_optimizer(model_str, model, discriminator, placeholders, pos_weight, norm, d_real,num_nodes):
	if model_str == 'arga_ae':
		d_fake = discriminator.construct(model.embeddings, reuse=True)
		opt = OptimizerAE(preds=model.reconstructions,
						  labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
																	  validate_indices=False), [-1]),
						  pos_weight=pos_weight,
						  norm=norm,
						  d_real=d_real,
						  d_fake=d_fake)
	elif model_str == 'arga_vae':
		opt = OptimizerVAE(preds=model.reconstructions,
						   labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
																	   validate_indices=False), [-1]),
						   model=model, num_nodes=num_nodes,
						   pos_weight=pos_weight,
						   norm=norm,
						   d_real=d_real,
						   d_fake=discriminator.construct(model.embeddings, reuse=True))
	return opt


def sparse_to_tuple(sparse_mx):
	if not sp.isspmatrix_coo(sparse_mx):
		sparse_mx = sparse_mx.tocoo()
	coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
	values = sparse_mx.data
	shape = sparse_mx.shape
	return coords, values, shape


def preprocess_graph(adj):
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
	adj_normalized = adj.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
	return sparse_to_tuple(adj_normalized)


def construct_feed_dict(adj_normalized, adj, features, placeholders):
	# construct feed dictionary
	feed_dict = dict()
	feed_dict.update({placeholders['features']: features})
	feed_dict.update({placeholders['adj']: adj_normalized})
	feed_dict.update({placeholders['adj_orig']: adj})
	return feed_dict

