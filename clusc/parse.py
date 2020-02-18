#!/usr/bin/env python
# coding: utf-8

from clusc.utils import *

from Bio import SeqIO
import cooler

def parse_cooler(file_path):
	"""
	Hi-C
	"""
	return cooler.Cooler(file_path)


def parse_fasta(file_name, feature_name, maxlen, flank_length):
	"""
	Genome sequences
	"""
	ID=[]
	sequence=[]
	for seq in SeqIO.parse(file_name, "fasta"):
		ID.append(str(seq.id))
		sequence.append(convert_seq(str(seq.seq), maxlen, flank_length))
	return pd.DataFrame(data={"ID":ID, feature_name:sequence})


def parse_bed(file_path):
	"""
	bed file
	"""
	data_list = []
	with open(file_path, "r") as inFile:
		line = inFile.readline()
		while line:
			lineArr = line.strip().split("\t")
			data_list.append([lineArr[0],lineArr[1],lineArr[2]])
			line = inFile.readline()

	return data_list



def parse_signalMatrix(file_name, y, feature_name, maxlen, flank_length):
	inFile = open(file_name, "r")
	line = inFile.readline()
	line = inFile.readline()

	hist = []
	while line:
		X_tmp = []
		lineArr = line.strip().split("\t")
		for i in range(6, len(lineArr)):
			tmp = float(lineArr[i])
			X_tmp.append(tmp)
		for i in range(0, flank_length):
			tmp = 0
			X_tmp.append(tmp)
		for i in range(len(lineArr)-1, 5, -1):
			tmp = float(lineArr[i])
			X_tmp.append(tmp)
		hist.append(X_tmp)
		line = inFile.readline()

	return pd.DataFrame(data={feature_name:hist})


def format_data(hic_data, bed_list, feat_data):

	id_labels = dict()
	edges_weight = []
	arr = np.zeros((len(bed_list), len(bed_list)))
	ele_num = 0

	for id1, item1 in enumerate(bed_list):
		id_labels.update({id1:"{}:{}-{}".format(item1[0],item1[1],item1[2])})
		for id2, item2 in enumerate(bed_list):
			# if id1 > id2:
			# 	continue
			if_val = hic_data.matrix(balance=True, sparse=True).fetch((item1[0],item1[1],item1[2]), (item2[0],item2[1],item2[2])).toarray()[0][0]
			if np.isnan(if_val):
				if_val = 0
			if if_val != 0:
				if id1 != id2:
					ele_num += 1
				if id1 <= id2:
					edges_weight.append(np.log1p(if_val))
			arr[id1][id2] = np.log1p(if_val)

	adj = sp.coo_matrix(arr)

	# Store original adjacency matrix (without diagonal entries) for later
	adj_orig = adj
	adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
	adj_orig.eliminate_zeros()

	features = sp.coo_matrix(feat_data)

	if FLAGS.features == 0:
		features = sp.identity(features.shape[0])  # featureless

	# Some preprocessing
	adj_norm = preprocess_graph(adj)

	num_nodes = adj.shape[0]

	features = sparse_to_tuple(features.tocoo())
	num_features = features[2][1]
	features_nonzero = features[1].shape[0]

	pos_weight = float(adj.shape[0] * adj.shape[0] - ele_num) / ele_num
	norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - ele_num) * 2)

	adj_label = adj
	adj_label = sparse_to_tuple(adj_label)

	items = [adj, num_features, num_nodes, features_nonzero, pos_weight, norm, adj_norm, adj_label, features, adj_orig, id_labels, edges_weight]
	feas = {}
	for item in items:
		# item_name = [ k for k,v in locals().iteritems() if v == item][0]
		feas[retrieve_name(item)] = item
	feas['num_features'] = num_features
	feas['features_nonzero'] = features_nonzero
	return feas

def get_hicnet(bed_list, hic_data):
	id_labels = dict()
	edges_weight = []
	arr = np.zeros((len(bed_list), len(bed_list)))

	for id1, item1 in enumerate(bed_list):
		id_labels.update({id1:"{}:{}-{}".format(item1[0],item1[1],item1[2])})
		for id2, item2 in enumerate(bed_list):
			# if id1 > id2:
			# 	continue
			if_val = hic_data.matrix(balance=True, sparse=True).fetch((item1[0],item1[1],item1[2]), (item2[0],item2[1],item2[2])).toarray()[0][0]
			if np.isnan(if_val):
				if_val = 0
			if if_val != 0:
				edges_weight.append(np.log1p(if_val))
			arr[id1][id2] = np.log1p(if_val)

	return sp.coo_matrix(arr), id_labels, edges_weight


def bed_annotate_cat(bed_list, feat_data, cat_data, outfile_path):

	cat_dict = {}
	line_num = 0
	with open(cat_data, "r") as inFile:
		line = inFile.readline().strip()
		while line:
			cat_dict[line] = line_num
			line_num += 1
			line = inFile.readline().strip()

	outfile = open(outfile_path, 'w')
	feat_mat = np.zeros((len(bed_list), len(cat_dict)))
	for idx, item in enumerate(bed_list):
		with open(feat_data, "r") as inFile:
			line = inFile.readline().strip()
			while line:
				lineArr = line.split("\t")
				if item[0] == lineArr[0] and int(item[1]) <= int(lineArr[2]) and int(item[2]) >= int(lineArr[1]):
					feat_mat[idx][cat_dict[lineArr[3]]] = 1

				line = inFile.readline().strip()
		for i in range(0,len(cat_dict)-1):
			outfile.write("%d\t" % feat_mat[idx][i])
		outfile.write("%d\n" % feat_mat[idx][-1])
	# np.savetxt(outfile_path, feat_mat, fmt='%d', delimiter='\t')
	return feat_mat


