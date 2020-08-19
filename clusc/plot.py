#!/usr/bin/env python
# coding: utf-8

from clusc.utils import *

import pygenometracks.tracks as pygtk
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plotting_tracks(region, track_configs, outfig_name, **kwargs):
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
			if track_configs[i].get('section_name') == None:
				track_configs[i]['section_name'] = 'HiC'
			if track_configs[i].get('region') == None:
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

			if tk.properties.get('title') != None:
				draw_title(ax, tk.properties['title'])

		if infile_type == 'arcs' or infile_type == 'links':
			if track_configs[i].get('section_name') == None:
				track_configs[i]['section_name'] = 'Links'
			elif track_configs[i].get('section_name') == 'clusters':

				if track_configs[i].get('color') == None \
					or not is_colormap(track_configs[i].get('color')):
					track_configs[i]['color'] = 'gist_rainbow'

				if track_configs[i].get('clusters') == None:
					tmp_pd = pd.read_csv(track_configs[i].get('file'), sep="\t", header=None)
					track_configs[i]['min_value'] = 1
					track_configs[i]['max_value'] = tmp_pd.iloc[:,6].max()
					n_clusters = track_configs[i]['max_value']-track_configs[i]['min_value']+1
				else:			
					track_configs[i]['min_value'] = 1
					track_configs[i]['max_value'] = track_configs[i].get('clusters')
					n_clusters = track_configs[i].get('clusters')

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
			if tk.properties.get('section_name') == 'clusters':
				ticks = np.linspace(tk.properties.get('min_value'),tk.properties.get('max_value'),n_clusters)
				bounds = np.append((ticks-0.5),(tk.properties.get('max_value')+0.5))
				cobar = fig.colorbar(tk.colormap, ax=ax, cax=axins, ticks=ticks, boundaries=bounds)
			else:
				cobar = fig.colorbar(tk.colormap, ax=ax, cax=axins)
			cobar.ax.yaxis.set_ticks_position('left')

			if tk.properties.get('title') != None:
				draw_title(ax, tk.properties['title'])

		if infile_type == 'bigwig' or infile_type == 'bw':
			if track_configs[i].get('section_name') == None:
				track_configs[i]['section_name'] = 'BigWig'
			tk = pygtk.BigWigTrack(track_configs[i])
			tk.plot(ax, region[0], region[1], region[2])

			if tk.properties.get('title') != None:
				draw_title(ax, tk.properties['title'])

		if infile_type == 'gtf':
			if track_configs[i].get('section_name') == None:
				track_configs[i]['section_name'] = 'Gene'
			tk = pygtk.GtfTrack(track_configs[i])
			tk.plot(ax, region[0], region[1], region[2])
			ax.set_yticks([])

			if tk.properties.get('title') != None:
				draw_title(ax, tk.properties['title'])

		if infile_type == 'bed':
			if track_configs[i].get('section_name') == None:
				track_configs[i]['section_name'] = 'Bed'
			elif track_configs[i].get('section_name') == 'clusters':

				if track_configs[i].get('color') == None \
					or not is_colormap(track_configs[i].get('color')):
					track_configs[i]['color'] = 'gist_rainbow'

				if track_configs[i].get('clusters') == None:
					tmp_pd = pd.read_csv(track_configs[i].get('file'), sep="\t", header=None)
					track_configs[i]['min_value'] = 1
					track_configs[i]['max_value'] = tmp_pd.iloc[:,4].max()
					n_clusters = track_configs[i]['max_value']-track_configs[i]['min_value']+1
				else:			
					track_configs[i]['min_value'] = 1
					track_configs[i]['max_value'] = track_configs[i].get('clusters')
					n_clusters = track_configs[i].get('clusters')

			tk = pygtk.BedTrack(track_configs[i])
			tk.plot(ax, region[0], region[1], region[2])
			ax.set_yticks([])

			if is_colormap(tk.properties['color']):
				axins = inset_axes(ax,
						width="1%",  # width = 1% of parent_bbox width
						height="100%",  # height : 100%
						loc='lower left',
						bbox_to_anchor=(0, 0, 1, 1),
						bbox_transform=ax.transAxes,
						borderpad=0)

				if tk.properties.get('section_name') == 'clusters':
					ticks = np.linspace(tk.properties.get('min_value'),tk.properties.get('max_value'),n_clusters)
					bounds = np.append((ticks-0.5),(tk.properties.get('max_value')+0.5))
					cobar = fig.colorbar(tk.colormap, ax=ax, cax=axins, ticks=ticks, boundaries=bounds)
				else:
					cobar = fig.colorbar(tk.colormap, ax=ax, cax=axins)
				cobar.ax.yaxis.set_ticks_position('left')

			if tk.properties.get('title') != None:
				draw_title(ax, tk.properties['title'])

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
