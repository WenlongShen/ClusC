#!/usr/bin/env python
# coding: utf-8

import inspect

import numpy as np
import pandas as pd
import tensorflow as tf


seed = 8853
np.random.seed(seed)
tf.set_random_seed(seed)


def retrieve_name(var):
	callers_local_vars = inspect.currentframe().f_back.f_locals.items()
	return [var_name for var_name, var_val in callers_local_vars if var_val is var][-1]


def noramlization(data):
	min_val = data.min()
	max_val = data.max()
	ranges = max_val - min_val
	norm_data = np.zeros(np.shape(data))
	m, n = data.shape[0], data.shape[1]
	norm_data = data - np.tile(min_val, (m, n))
	norm_data = norm_data / np.tile(ranges, (m, n))
	return norm_data


def eliminate_to_zeros(data, threshold=0.05):
	data[data<threshold] = 0
	return data


def str_to_pos(string_id):
	"""
	chrom:start-end
	"""
	arr1 = string_id.split(":")
	arr2 = arr1[1].split("-")
	return arr1[0],arr2[0],arr2[1]

