#!/usr/bin/env python
# coding: utf-8

from clusc.utils import *
from clusc.arga import *

from sklearn.cluster import KMeans


def clustering(emb, n_clusters, ids, **kwargs):
	kmeans = KMeans(n_clusters, random_state=seed).fit(emb)
	predict_labels = kmeans.predict(emb)
	result_pd = pd.DataFrame(data={"ID":ids, "cluster":predict_labels})
	return result_pd


def embedding(data, n_clusters, algorithm="arga", **kwargs):

	if algorithm == "arga":
		model = kwargs.get("model")
		epochs = kwargs.get("epochs")

		tf.reset_default_graph()
		placeholders = get_placeholder(data['adj'])
		d_real, discriminator, ae_model = get_model(model, placeholders, data['num_features'], data['num_nodes'], data['features_nonzero'])
		opt = get_optimizer(model, ae_model, discriminator, placeholders, data['pos_weight'], data['norm'], d_real, data['num_nodes'])
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		for epoch in range(epochs):
			emb, avg_cost = update(ae_model, opt, sess, data['adj_norm'], data['adj_label'], data['features'], placeholders, data['adj'])

	return emb

