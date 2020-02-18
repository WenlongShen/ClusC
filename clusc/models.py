#!/usr/bin/env python
# coding: utf-8

from clusc.utils import *
from clusc.arga import *

import numpy as np
import pandas as pd
import math
import subprocess
import tensorflow as tf


def clustering(graph, features, model, clustering_num, iterations):
	def __init__(self, settings):

		print("Clustering on dataset: %s, model: %s, number of iteration: %3d" % (settings['data_name'], settings['model'], settings['iterations']))

		self.data_name = settings['data_name']
		self.iteration =settings['iterations']
		self.model = settings['model']
		self.n_clusters = settings['clustering_num']

	def erun(self):
		model_str = self.model

		# formatted data
		feas = format_data(self.data_name)

		# Define placeholders
		placeholders = get_placeholder(feas['adj'])

		# construct model
		d_real, discriminator, ae_model = get_model(model_str, placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'])

		# Optimizer
		opt = get_optimizer(model_str, ae_model, discriminator, placeholders, feas['pos_weight'], feas['norm'], d_real, feas['num_nodes'])

		# Initialize session
		sess = tf.Session()
		# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
		sess.run(tf.global_variables_initializer())

		# Train model
		for epoch in range(self.iteration):
			emb, _ = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])

			if (epoch+1) % 2 == 0:
				kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(emb)
				print("Epoch:", '%04d' % (epoch + 1))
				predict_labels = kmeans.predict(emb)
				cm = clustering_metrics(feas['true_labels'], predict_labels)
				cm.evaluationClusterModelFromLabel()


def data_generator(X_train, y_train, batch_size):

	np.random.seed(random_seed)
	while True:
		row = np.random.randint(0, len(X_train), size=batch_size)
		x = X_train[row]
		y = y_train[row]
		yield (x,y)


def model_cnn(inputShape, numClasses, n_filters, n_kernel_size, n_units, gpu=1):

	if len(n_filters) < 2:
		raise Exception("n_filters size less than 2")
	else:
		for i in range(len(n_filters)):
			n_filters[i] = int(round(n_filters[i]))

	if len(n_kernel_size) < 2:
		raise Exception("n_kernel_size size less than 2")
	else:
		for i in range(len(n_kernel_size)):
			n_kernel_size[i] = int(round(n_kernel_size[i]))

	if len(n_units) < 1:
		raise Exception("n_units size less than 1")
	else:
		for i in range(len(n_units)):
			n_units[i] = int(round(n_units[i]))

	gpu = int(round(gpu))
	
	cnnModel = Sequential()
	cnnModel.add(Conv2D(input_shape=inputShape, filters=n_filters[0], kernel_size=(n_kernel_size[0], inputShape[-1]), activation="relu", padding ="same", name="conv2d_1", data_format="channels_first"))
	cnnModel.add(BatchNormalization(name="batch_normalization_1"))
	cnnModel.add(MaxPooling2D(pool_size=(2, 1), strides=None, name="max_pooling2d_1", data_format="channels_first"))
	cnnModel.add(Conv2D(filters=n_filters[1], kernel_size=(n_kernel_size[1], inputShape[-1]), activation="relu", padding ="same", name="conv2d_2", data_format="channels_first"))
	cnnModel.add(BatchNormalization(name="batch_normalization_2"))
	cnnModel.add(MaxPooling2D(pool_size=(2, 1), strides=None, name="max_pooling2d_2", data_format="channels_first"))
	cnnModel.add(Dropout(0.2))

	cnnModel.add(Flatten(name="flatten_1"))
	cnnModel.add(BatchNormalization(name="batch_normalization_3"))
	cnnModel.add(Dense(units=n_units[0], name="dense_1"))
	cnnModel.add(Activation("relu", name="activation_1"))
	cnnModel.add(Dropout(0.2))

	cnnModel.add(BatchNormalization(name="batch_normalization_4"))
	cnnModel.add(Dense(units=numClasses, name="dense_2"))
	cnnModel.add(Activation("sigmoid", name="activation_2"))
	
	if gpu > 1:
		cnnModel = multi_gpu_model(cnnModel, gpus=gpu)
	# categorical_accuracy
	cnnModel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["categorical_accuracy"])
	# print(cnnModel.summary())
	return cnnModel


def optimize_cnn(X_train, y_train, X_val, y_val, n_epoch=100, n_batch_size=256, generator=0, gpu=1):

	inputShape = X_train[0].shape
	numClasses = y_train.shape[1]

	def cnn_cv(n_filters_1, n_filters_2, n_kernel_size_1, n_kernel_size_2, n_units_1):

		K.clear_session()

		n_filters = [n_filters_1, n_filters_2]
		n_kernel_size = [n_kernel_size_1, n_kernel_size_2]
		n_units = [n_units_1]

		n_model = model_cnn(inputShape, numClasses, n_filters, n_kernel_size, n_units, gpu=gpu)

		if generator == 0:
			cnn = n_model.fit(X_train, y_train, batch_size=n_batch_size, validation_data=(X_val, y_val), epochs=n_epoch, verbose=0)
		else:
			cnn = n_model.fit_generator(data_generator(X_train, y_train, n_batch_size), validation_data=(X_val, y_val), epochs=n_epoch, steps_per_epoch=math.ceil(len(X_train)/n_batch_size), verbose=0)
		return 1-cnn.history["val_loss"][-1]

	optimizer = BayesianOptimization(
		f = cnn_cv,
		pbounds = {
			"n_filters_1":(16, 512),
			"n_filters_2":(8, 128),
			"n_kernel_size_1":(4, 32),
			"n_kernel_size_2":(2, 16),
			"n_units_1":(8, 512)
		},
		random_state = random_seed
	)

	optimizer.maximize(n_iter=20, alpha=1e-2, n_restarts_optimizer=5)
	print(optimizer.max)


def inSilicoMutagenesis(seq, pos):
	seqs = []
	for i in pos:
		tmpSeq = []
		tmpSeq.append(seq[:i] + "A" + seq[i+1:])
		tmpSeq.append(seq[:i] + "C" + seq[i+1:])
		tmpSeq.append(seq[:i] + "G" + seq[i+1:])
		tmpSeq.append(seq[:i] + "T" + seq[i+1:])
		seqs.append(tmpSeq)
	seqs = np.array(seqs)
	return seqs

def getMutationScores(seqs, pos, oriData, model, encoding, params):
	pred = model.predict(oriData)
	mutScores = []
	for o, seq in enumerate(seqs): 
		seqMat = inSilicoMutagenesis(seq, pos)
		target = np.argmax(pred[o])
		oriScore = pred[o][target]
		scores = []
		for i in range(0,seqMat.shape[0]):
			tmpScores = []
			for j in range(0,seqMat.shape[1]):
				tmpMat = expand_encoded(expand_encoded(encoding(seqMat[i][j], params[0], params[1])))
				tmpScores.append(model.predict(tmpMat)[0][target]-oriScore)
			scores.append(tmpScores)
		mutScores.append(scores)
	mutScores = np.array(mutScores)
	return mutScores


def getGradients(model, layerName, data):
	loss = model.get_layer(layerName).output
	grads = K.gradients(loss, [model.input])[0]
	fn = K.function([model.input], [loss, grads])
	return fn([data])

def getSeqImportance(model, layerName, data):
	return getGradients(model, layerName, data)[1] * data

def getSeqMotifs(model, layerName, data, seqs, dataPath, method="one", threshold=1):
	"""
	method
		one: select one motif from one seq
		seq: select motifs (score > max_score_within_the_seq*threshold) from one seq
		motif: select motifs (score > max_score_within_the_motif*threshold) from seqs
	"""	
	motifs = np.mean(getGradients(model, layerName, data)[0], axis=-1)
	motifLen = model.get_layer(layerName).get_weights()[0].shape[0]
	motifLen2 = (motifLen - 1) // 2
	if method == "one":
		for i, motif in enumerate(motifs):
			pos = get_max_pos(motif.shape, np.argmax(motif))

			fileName = dataPath + layerName + "_motif_" + str(pos[0])
			fasta = open(fileName+".fasta", "a")
			sites = open(fileName+".sites", "a")

			mot = seqs[i][pos[1]-motifLen2:pos[1]+motifLen-motifLen2]
			if len(mot) == motifLen:
				fasta.write(">%d_%d_%f\n%s\n" % (i, pos[1], motif[pos[0],pos[1]], mot))
				sites.write("%s\n" % (mot))
			fasta.close()
			sites.close()
	if method == "motif":
		max_score = np.zeros(motifs.shape[1])
		for i, nmotif in enumerate(motifs):
			for j, motif in enumerate(nmotif):
				if max_score[j] < motif[np.argmax(motif)]:
					max_score[j] = motif[np.argmax(motif)]
		max_score *= threshold
		for i, nmotif in enumerate(motifs):
			for j, motif in enumerate(nmotif):
				pos = np.argmax(motif)
				if max_score[j] < motif[pos]:
					fileName = dataPath + layerName + "_motif_" + str(j)
					fasta = open(fileName+".fasta", "a")
					sites = open(fileName+".sites", "a")

					mot = seqs[i][pos-motifLen2:pos+motifLen-motifLen2]
					if len(mot) == motifLen:
						fasta.write(">%d_%d_%f\n%s\n" % (i, pos, motif[pos], mot))
						sites.write("%s\n" % (mot))
					fasta.close()
					sites.close()
	if method == "seq":
		for i, nmotif in enumerate(motifs):
			max_score = 0
			for j, motif in enumerate(nmotif):
				if max_score < motif[np.argmax(motif)]:
					max_score = motif[np.argmax(motif)]
			max_score *= threshold
			for j, motif in enumerate(nmotif):
				pos = np.argmax(motif)
				if max_score < motif[pos]:
					fileName = dataPath + layerName + "_motif_" + str(j)
					fasta = open(fileName+".fasta", "a")
					sites = open(fileName+".sites", "a")

					mot = seqs[i][pos-motifLen2:pos+motifLen-motifLen2]
					if len(mot) == motifLen:
						fasta.write(">%d_%d_%f\n%s\n" % (i, pos, motif[pos], mot))
						sites.write("%s\n" % (mot))
					fasta.close()
					sites.close()



# weblogoCmd = 'weblogo -X NO -Y NO --errorbars NO --fineprint "" -C "#008000" A A -C "#0000cc" C C -C "#ffb300" G G -C "#cc0000" T T < %s.fasta > %s.eps' % (fileName, fileName)
# subprocess.call(weblogoCmd, shell=True)
