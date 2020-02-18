#!/usr/bin/env python
# coding: utf-8

from clusc.utils import *
import matplotlib.pyplot as plt

def plot_heatmap(fig, ax, hic_data, region=None, site_list=None, cmap="Reds"):
	if region == None:
		arr = hic_data.matrix(balance=True, sparse=True)[:].toarray()
	else:
		chrom, start, end = region
		
	im = ax.matshow(np.log1p(arr), cmap=cmap)
	fig.colorbar(im)