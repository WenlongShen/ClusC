#!/usr/bin/env python
# coding: utf-8

from clusc.utils import *

import pygenometracks.tracks as pygtk
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.cm as mcm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


default_colormap = 'gist_rainbow'

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
				draw_track_title(ax, tk.properties['title'])

		if infile_type == 'arcs' or infile_type == 'links':
			if track_configs[i].get('section_name') == None:
				track_configs[i]['section_name'] = 'Links'
			elif track_configs[i].get('section_name') == 'clusters':

				if track_configs[i].get('color') == None \
					or not is_colormap(track_configs[i].get('color')):
					track_configs[i]['color'] = default_colormap

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
				draw_track_title(ax, tk.properties['title'])

		if infile_type == 'bigwig' or infile_type == 'bw':
			if track_configs[i].get('section_name') == None:
				track_configs[i]['section_name'] = 'BigWig'
			tk = pygtk.BigWigTrack(track_configs[i])
			tk.plot(ax, region[0], region[1], region[2])

			if tk.properties.get('title') != None:
				draw_track_title(ax, tk.properties['title'])

		if infile_type == 'gtf':
			if track_configs[i].get('section_name') == None:
				track_configs[i]['section_name'] = 'Gene'
			tk = pygtk.GtfTrack(track_configs[i])
			tk.plot(ax, region[0], region[1], region[2])
			ax.set_yticks([])

			if tk.properties.get('title') != None:
				draw_track_title(ax, tk.properties['title'])

		if infile_type == 'bed':
			if track_configs[i].get('section_name') == None:
				track_configs[i]['section_name'] = 'Bed'
			elif track_configs[i].get('section_name') == 'clusters':

				if track_configs[i].get('color') == None \
					or not is_colormap(track_configs[i].get('color')):
					track_configs[i]['color'] = default_colormap

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
				draw_track_title(ax, tk.properties['title'])

	plt.savefig(outfig_name)



def plotting_circos(circos_configs, outfig_name, **kwargs):
	figsize=kwargs.get("figsize", (9, 9))
	fig = plt.figure(figsize=figsize)
	ax = fig.add_axes([0,0,1,1], polar=True)
	ax.axis('off')
	ax.set_ylim(0,max(figsize))

	if circos_configs[0].get('type') != 'chrom':
		raise Exception("Invalid first track... We need to configure chrom track first right now...")

	chrom_regions = pd.read_csv(circos_configs[0].get('file'), sep="\t", names=['chrom','start','end'])
	if not is_valid_chrom(chrom_regions):
		raise Exception("Invalid first track... We do not support overlapped genomic regions right now...")

	chrom_regions['length'] = chrom_regions['end']-chrom_regions['start']
	total_len = chrom_regions['length'].sum()
	gap = circos_configs[0].get('gap', np.pi/72)
	len_per_theta = total_len/(np.pi*2-gap*chrom_regions.shape[0])

	cumlen = [0] + list(chrom_regions['length'].cumsum())[:-1]
	chrom_regions['theta_start'] = [np.pi/2-l/len_per_theta-gap*i for i,l in enumerate(cumlen)]
	chrom_regions['theta_end'] = chrom_regions['theta_start']-chrom_regions['length']/len_per_theta

	color,colormap,colorlist = get_colorlist(circos_configs[0].get('color'), 0, chrom_regions.shape[0]-1)

	for index, row in chrom_regions.iterrows():
		if color != None:
			chrom_color = color[index%len(color)]
		else:
			chrom_color = colorlist.to_rgba(index)
		ax.bar((row['theta_start']+row['theta_end'])/2,
			circos_configs[0].get('width', 1),
			color=chrom_color,
			width=(row['theta_end']-row['theta_start']),
			bottom=circos_configs[0].get('radius', 0.9)*max(figsize))

	for i in range(1, len(circos_configs)):
		if circos_configs[i].get('type') == 'highlight':
			tmp_pd = pd.read_csv(circos_configs[i].get('file'), sep="\t", names=['chrom','start','end','name','score','strand'])
			vmin = tmp_pd['score'].min()
			vmax = tmp_pd['score'].max()

			color,colormap,colorlist = get_colorlist(circos_configs[i].get('color'), vmin, vmax)

			for index, row in tmp_pd.iterrows():
				valid,ts,te = get_theta(chrom_regions, row, len_per_theta)
				if not valid:
					continue
				if color != None:
					bar_color = color[(int(row['score'])-vmin)%len(color)]
				else:
					bar_color = colorlist.to_rgba(row['score'])

				ax.bar((ts+te)/2,
					circos_configs[i].get('width', 1),
					color=bar_color,
					width=(te-ts),
					bottom=circos_configs[i].get('radius', 0.9)*max(figsize))


	plt.savefig(outfig_name)


def is_colormap(color_name):
	if color_name in dir(plt.cm):
		return True
	else:
		return False


def colormap_to_rbg(colormap, vmin, vmax):
	cNorm = mc.Normalize(vmin, vmax)
	scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=colormap)
	return scalarMap


def get_colorlist(color_name, vmin, vmax):
	if color_name != None:
		color = color_name
		colormap = None
		colorlist = None
		if isinstance(color, str):
			if is_colormap(color):
				colormap = color
				colorlist = colormap_to_rbg(colormap, vmin, vmax)
				color = None
			else:
				colormap = None
				colorlist = None
				color = [color]
	else:
		colormap = default_colormap
		colorlist = colormap_to_rbg(colormap, vmin, vmax)
		color = None
	return color,colormap,colorlist


def draw_track_title(ax, title):
	ax.text(1.1, 0.5, title, 
		horizontalalignment='center', size='large', verticalalignment='center',
		transform=ax.transAxes, wrap=True)


def is_valid_chrom(chrom_pd):
	for i in range(0, chrom_pd.shape[0]):
		for j in range(i+1, chrom_pd.shape[0]):
			if chrom_pd.loc[i,'chrom'] == chrom_pd.loc[j,'chrom'] \
				and chrom_pd.loc[i,'start'] <= chrom_pd.loc[j,'end'] \
				and chrom_pd.loc[j,'start'] <= chrom_pd.loc[i,'end']:
				return False
	return True


def get_chromID(chrom_pd, regions_pd):
	for index, row in chrom_pd.iterrows():
		if regions_pd['chrom'] == row['chrom'] \
			and regions_pd['start'] >= row['start'] \
			and regions_pd['end'] <= row['end']:
			return index
	return -1


def get_theta(chrom_pd, regions_pd, len_per_theta):
	chromID = get_chromID(chrom_pd, regions_pd)
	if chromID == -1:
		return False,0,0
	else:
		ts = chrom_pd.loc[chromID, 'theta_start']-(regions_pd['start']-chrom_pd.loc[chromID, 'start'])/len_per_theta
		te = chrom_pd.loc[chromID, 'theta_start']-(regions_pd['end']-chrom_pd.loc[chromID, 'start'])/len_per_theta
		return True,ts,te
	
