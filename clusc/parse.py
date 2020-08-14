#!/usr/bin/env python
# coding: utf-8

from clusc.utils import *
from clusc.arga import *

import cooler
import pandas as pd
import numpy as np
import scipy.sparse as sp


def load_data(data_type, infile_name, **kwargs):
	"""
	data_type == loci, net, feat
	"""
	if data_type == "loci":
		return pd.read_csv(infile_name, sep="\t")

	elif data_type == "net":
		loci_num = kwargs.get("loci_num")
		tmp = pd.read_csv(infile_name, sep="\t", header=None)
		tmp = tmp[tmp[2]!=0]
		row = np.array(tmp[0])
		col = np.array(tmp[1])
		data = np.array(tmp[2])
		return sp.coo_matrix((data, (row, col)), shape=(loci_num, loci_num))

	elif data_type == "feat":
		return sp.coo_matrix(pd.read_csv(infile_name, sep="\t").drop("ID", axis=1).values)

	else:
		raise Exception("Invalid data type... We only accepted loci, net, feat right now...")

	return


def format_data(loci_pd, net, feat_mat, algorithm="arga"):

	id_labels = dict()
	ele_num = 0

	for index, row in loci_pd.iterrows():
		id_labels.update({index:row["ID"]})

	adj = np.log1p(net)

	# Store original adjacency matrix (without diagonal entries) for later
	adj_orig = adj
	adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
	adj_orig.eliminate_zeros()

	features = feat_mat

	if FLAGS.features == 0:
		features = sp.identity(features.shape[0])  # featureless

	# Some preprocessing
	adj_norm = preprocess_graph(adj)

	num_nodes = adj.shape[0]

	features = sparse_to_tuple(features.tocoo())
	num_features = features[2][1]
	features_nonzero = features[1].shape[0]

	pos_weight = float(adj.shape[0] * adj.shape[0] - len(adj.data)) / len(adj.data)
	norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - len(adj.data)) * 2)

	adj_label = adj
	adj_label = sparse_to_tuple(adj_label)

	items = [adj, num_features, num_nodes, features_nonzero, pos_weight, norm, adj_norm, adj_label, features, adj_orig, id_labels]
	feas = {}
	for item in items:
		# item_name = [ k for k,v in locals().iteritems() if v == item][0]
		feas[retrieve_name(item)] = item
	feas['num_features'] = num_features
	feas['features_nonzero'] = features_nonzero
	return feas


def preprocess_loci(infile_name, outfile_name, infile_type=None, id_pos=None):
	"""
	user-defined genomic loci or genes
	"""
	loci_chr = []
	loci_start = []
	loci_end = []
	loci_id = []

	if infile_type is None:
		infile_type = infile_name.split(".")[-1]

	if infile_type == "bed":
		with open(infile_name, "r") as inFile:
			line = inFile.readline()
			while line:
				lineArr = line.strip().split("\t")

				loci_chr.append(lineArr[0])
				loci_start.append(lineArr[1])
				loci_end.append(lineArr[2])

				if id_pos == "bed":
					if len(lineArr) >= 3:
						loci_id.append(lineArr[3])
					else:
						raise Exception("Invalid id content... The loci file is not standard bed format...")
				else:
					loci_id.append("{}:{}-{}".format(lineArr[0],lineArr[1],lineArr[2]))

				line = inFile.readline()

		loci_pd = pd.DataFrame(data={"ID":loci_id, "chrom":loci_chr, "start":loci_start, "end":loci_end})
		loci_pd.to_csv(outfile_name, sep="\t", index=False)

		return loci_pd

	else:
		raise Exception("Invalid file type... We only accepted bed format for loci file right now...")

	return


def preprocess_hic(infile_name, outfile_name, loci_pd, infile_type=None):
	"""
	hic data to adjacency matrix
	"""
	if infile_type is None:
		infile_type = infile_name.split(".")[-1]

	if infile_type == "cool":
		hic_data = cooler.Cooler(infile_name)
	else:
		raise Exception("Invalid file type... We only accepted cool format for Hi-C file right now...")

	outfile = open(outfile_name, 'w')
	hic_net = np.zeros((len(loci_pd), len(loci_pd)))

	for index_i, row_i in loci_pd.iterrows():
		for index_j, row_j in loci_pd.iterrows():
			val = hic_data.matrix(balance=True, sparse=True).fetch((row_i["chrom"],row_i["start"],row_i["end"]), \
				(row_j["chrom"],row_j["start"],row_j["end"])).toarray()[0][0]
			if np.isnan(val):
				val = 0

			hic_net[index_i][index_j] = val
			outfile.write("{}\t{}\t{}\n".format(index_i, index_j, val))

	return sp.coo_matrix(hic_net)


def preprocess_feat(infile_name, outfile_name, loci_pd, feat_type=None, **kwargs):
	"""
	convert feature matrix
	"""
	if feat_type is None:
		raise Exception("Please provide feat type... We only accepted category(cat) right now...")

	elif feat_type == "cat":
		cat_file = kwargs.get("cat_file")
		if cat_file is None:
			raise Exception("Please provide cat file, which lists all the features...")
		else:
			cat_dict = {}
			line_num = 0
			outfile = open(outfile_name, "w")
			outfile.write("ID")

			with open(cat_file, "r") as inFile:
				line = inFile.readline().strip()
				while line:
					outfile.write("\t{}".format(line))
					cat_dict[line] = line_num
					line_num += 1
					line = inFile.readline().strip()
			outfile.write("\n")

			
			feat_mat = np.zeros((len(loci_pd), len(cat_dict)))
			for idx, row in loci_pd.iterrows():
				with open(infile_name, "r") as inFile:
					line = inFile.readline().strip()
					while line:
						lineArr = line.split("\t")
						if row["chrom"] == lineArr[0] and int(row["start"]) <= int(lineArr[2]) and int(row["end"]) >= int(lineArr[1]):
							feat_mat[idx][cat_dict[lineArr[3]]] = 1
						line = inFile.readline().strip()

				outfile.write(row["ID"])
				for i in range(0, len(cat_dict)):
					outfile.write("\t{:.0f}".format(feat_mat[idx][i]))
				outfile.write("\n")

			return sp.coo_matrix(feat_mat)

	else:
		raise Exception("Invalid feat type... We only accepted category(cat) right now...")

	return


def noramlization_hic(hic_net, threshold=0.05):
	hic_net = hic_net.toarray()
	hic_net = pow(hic_net, 2)
	hic_net = noramlization(hic_net)
	hic_net = eliminate_to_zeros(hic_net, threshold)
	return sp.coo_matrix(hic_net)

	