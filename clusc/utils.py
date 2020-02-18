#!/usr/bin/env python
# coding: utf-8

import os
import string
import inspect
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import tensorflow as tf

from clusc.arga import *
from clusc.input_data import load_data


seed = 8853
np.random.seed(seed)
tf.set_random_seed(seed)


def convert_seq(seq, maxlen, flank_length):
	"""
	Clean, Upper, Uniform
	Add "N"*flank_length
	RevcompSeq
	"""	
	seq = uniform_seq(upper_seq(clean_seq(seq)), maxlen)
	if flank_length >= 0:
		return seq + "N"*flank_length + revcomp_seq(seq)
	else:
		return seq

def clean_seq(seq):
	"""
	Remove non-natural bases
	"""
	char_list = list(string.printable)
	for i in ["A","C","G","T","a","c","g","t"]:
		char_list.remove(i)
	for ch in char_list:
		if ch in seq:
			seq = seq.replace(ch, "N")
	return seq

def upper_seq(seq):
	"""
	Convert DNA sequence to uppercase
	"""
	return seq.upper()

def uniform_seq(seq, maxlen):
	"""
	Uniform sequence to fixed length
	"""
	if len(seq) > maxlen:
		seq = seq[:maxlen]
	else:
		pre = int((maxlen-len(seq))/2)
		suf = maxlen-len(seq)-pre
		seq = "N"*pre + seq + "N"*suf
	return seq

def revcomp_seq(seq):
	"""
	Get reverse complementary sequence
	"""
	return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]

def convertSeqW2V(dataframe, column, window, size):
	sentences = []
	for seq in dataframe[column]:
		for shift in range(0, window):
			sentences.append([word for word in re.findall(r".{"+str(window)+"}", seq[shift:])])

	model = Word2Vec(sentences=sentences, size=size, window=4, min_count=1, negative=5, sg=1, sample=0.001, hs=1, 
					 workers=8, seed=seed)

	w2vFeat = []
	for i in range(0, len(sentences), window):
		w2vSum = np.zeros(shape=(size,))
		for j in range(i, i+window):
			for word in sentences[j]:
				w2vSum += model[word]
		w2vFeat.append(w2vSum)
		
	col = ["w2v_{0}".format(i) for i in range(size)]
	tmp = pd.DataFrame(w2vFeat, columns=col)
	return pd.concat([dataframe, tmp], axis=1)

def get_max_pos(shape, argmax):
	pos = [0 for i in range(len(shape))]
	for i in range(len(pos)):
		arr = 1
		for j in range(i+1, len(pos)):
			arr *= shape[j]
		pos[i] = argmax // arr
		argmax -= arr*pos[i]
	return pos

def is_substring(substr_list, str):
	flag = True
	for substr in substr_list:
		if not (substr in str):
			flag = False
	return flag

def get_files(dir_path, file_ext):
	file_list = []
	file_names = os.listdir(dir_path)
	for fn in file_names:
		if len(file_ext) > 0:
			if is_substring(file_ext, fn):
				file_list.append(os.path.join(dir_path, fn))
		else:
			file_list.append(os.path.join(dir_path, fn))

	if len(file_list) > 0:
		file_list.sort()

	return file_list

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


def retrieve_name(var):
	callers_local_vars = inspect.currentframe().f_back.f_locals.items()
	return [var_name for var_name, var_val in callers_local_vars if var_val is var][-1]

