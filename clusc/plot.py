#!/usr/bin/env python
# coding: utf-8

from clusc.utils import *

import pygenometracks.tracks as pygtk
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plotting_track(region, track_configs, outfig_name, **kwargs):
	if len(track_configs) < 2:
		raise Exception("Invalid track numbers... We only accepted more than one track right now...")

	figsize=kwargs.get("figsize", (12, 9))
	fig, axs = plt.subplots(len(track_configs), 1, sharex='col', figsize=figsize)
	plt.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9)

	for i in range(0, len(track_configs)):
		ax = axs[i]
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['left'].set_visible(False)

		infile_type = track_configs[i]['file'].split(".")[-1]

		if infile_type == 'cool':
			track_configs[i]['section_name'] = 'HiC'
			track_configs[i]['region'] = "{}:{}-{}".format(region[0], region[1], region[2])
			tk = pygtk.HiCMatrixTrack(track_configs[i])
			tk.plot(ax, region[0], int(region[1])+track_configs[i]['depth'], int(region[2])-track_configs[i]['depth'])
			ax.set_yticks([])

			axins = inset_axes(ax,
					width="1%",
					height="100%",
					loc='lower left',
					bbox_to_anchor=(0, 0, 1, 1),
					bbox_transform=ax.transAxes,
					borderpad=0)
			cobar = fig.colorbar(tk.img, ax=ax, cax=axins)
			cobar.ax.yaxis.set_ticks_position('left')

			if track_configs[i]['title'] != None:
				draw_title(ax, track_configs[i]['title'])

		if infile_type == 'arcs' or infile_type == 'links':
			tk = pygtk.LinksTrack(track_configs[i])
			tk.plot(ax, region[0], region[1], region[2])
			ax.set_yticks([])

			axins = inset_axes(ax,
					width="1%",
					height="100%",
					loc='lower left',
					bbox_to_anchor=(0, 0, 1, 1),
					bbox_transform=ax.transAxes,
					borderpad=0)
			cobar = fig.colorbar(tk.colormap, ax=ax, cax=axins)
			cobar.ax.yaxis.set_ticks_position('left')

			if track_configs[i]['title'] != None:
				draw_title(ax, track_configs[i]['title'])

		if infile_type == 'bigwig' or infile_type == 'bw':
			tk = pygtk.BigWigTrack(track_configs[i])
			tk.plot(ax, region[0], region[1], region[2])

			if track_configs[i]['title'] != None:
				draw_title(ax, track_configs[i]['title'])

		if infile_type == 'gtf':
			track_configs[i]['section_name'] = 'Gene'
			tk = pygtk.GtfTrack(track_configs[i])
			tk.plot(ax, region[0], region[1], region[2])
			ax.set_yticks([])

			if track_configs[i]['title'] != None:
				draw_title(ax, track_configs[i]['title'])

		if infile_type == 'bed':
			tk = pygtk.BedTrack(track_configs[i])
			tk.plot(ax, region[0], region[1], region[2])
			ax.set_yticks([])

			if is_colormap(track_configs[i]['color']):
				axins = inset_axes(ax,
						width="1%",  # width = 1% of parent_bbox width
						height="100%",  # height : 100%
						loc='lower left',
						bbox_to_anchor=(0, 0, 1, 1),
						bbox_transform=ax.transAxes,
						borderpad=0)
				cobar = fig.colorbar(tk.colormap, ax=ax, cax=axins)
				cobar.ax.yaxis.set_ticks_position('left')

			if track_configs[i]['title'] != None:
				draw_title(ax, track_configs[i]['title'])

	plt.savefig(outfig_name)



def plotting_circos(region, track_configs, outfig_name, **kwargs):
	plt.savefig(outfig_name)



def is_colormap(color_name):
	if color_name in dir(plt.cm):
		return True
	else:
		return False


def draw_title(ax, title):
	ax.text(1.1, 0.5, title, 
		horizontalalignment='center', size='large', verticalalignment='center',
		transform=ax.transAxes, wrap=True)
