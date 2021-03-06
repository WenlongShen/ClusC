#!/usr/bin/env python
# coding: utf-8

from clusc.utils import *
from clusc.parse import *

import pygenometracks.tracks as pygtk
import pyBigWig
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.cm as mcm
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import itertools

default_colormap = 'gist_rainbow'

def plot_tracks(region, track_configs, outfig_name, **kwargs):
	if len(track_configs) < 2:
		raise Exception("Invalid track numbers... We only accepted more than one track right now...")

	figsize=kwargs.get("figsize", (16, 9))
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
			if track_configs[i].get('section_name') is None:
				track_configs[i]['section_name'] = 'HiC'
			if track_configs[i].get('region') is None:
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

			if tk.properties.get('title') is not None:
				draw_track_title(ax, tk.properties['title'])

		if infile_type == 'arcs' or infile_type == 'links':
			if track_configs[i].get('section_name') is None:
				track_configs[i]['section_name'] = 'Links'
			elif track_configs[i].get('section_name') == 'clusters':

				if track_configs[i].get('color') is None \
					or not is_colormap(track_configs[i].get('color')):
					track_configs[i]['color'] = default_colormap

				if track_configs[i].get('clusters') is None:
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

			if tk.properties.get('title') is not None:
				draw_track_title(ax, tk.properties['title'])

		if infile_type == 'bigwig' or infile_type == 'bw':
			if track_configs[i].get('section_name') is None:
				track_configs[i]['section_name'] = 'BigWig'
			tk = pygtk.BigWigTrack(track_configs[i])
			tk.plot(ax, region[0], region[1], region[2])

			if tk.properties.get('title') is not None:
				draw_track_title(ax, tk.properties['title'])

		if infile_type == 'gtf':
			if track_configs[i].get('section_name') is None:
				track_configs[i]['section_name'] = 'Gene'
			tk = pygtk.GtfTrack(track_configs[i])
			tk.plot(ax, region[0], region[1], region[2])
			ax.set_yticks([])

			if tk.properties.get('title') is not None:
				draw_track_title(ax, tk.properties['title'])

		if infile_type == 'bed':
			if track_configs[i].get('section_name') is None:
				track_configs[i]['section_name'] = 'Bed'
			elif track_configs[i].get('section_name') == 'clusters':

				if track_configs[i].get('color') is None \
					or not is_colormap(track_configs[i].get('color')):
					track_configs[i]['color'] = default_colormap

				if track_configs[i].get('clusters') is None:
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

			if tk.properties.get('title') is not None:
				draw_track_title(ax, tk.properties['title'])

	plt.savefig(outfig_name)



def plot_circos(circos_configs, outfig_name, **kwargs):
	figsize = kwargs.get("figsize", (9, 9))
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
		if color is not None:
			chrom_color = color[index%len(color)]
		else:
			chrom_color = colorlist.to_rgba(index)
		ax.bar((row['theta_start']+row['theta_end'])/2,
			circos_configs[0].get('width', 1),
			color=chrom_color,
			width=(row['theta_end']-row['theta_start']),
			bottom=circos_configs[0].get('radius', 0.9)*max(figsize))

	if circos_configs[0].get('cytobands_file') is not None:
		tmp_pd = pd.read_csv(circos_configs[0].get('cytobands_file'), sep="\t", names=['chrom','start','end','name','gieStain'])
		tmp_pd = get_valid_regions(chrom_regions, tmp_pd)
		cyto_colors = {"gneg":"#FFFFFF","gpos25":"#E5E5E5","gpos50":"#B3B3B3","gpos75":"#666666",
						"gpos100":"#000000","gvar":"#FFFFFF","stalk":"#CD3333","acen":"#8B2323"}
		for index, row in tmp_pd.iterrows():
			valid,ts,te = get_theta(chrom_regions, row, len_per_theta)
			if not valid:
				continue
			ax.bar((ts+te)/2,
				circos_configs[0].get('width', 1),
				color=cyto_colors[row['gieStain']],
				alpha=0.15,
				width=(te-ts),
				bottom=circos_configs[0].get('radius', 0.9)*max(figsize))

	if circos_configs[0].get('label', True):
		for index, row in chrom_regions.iterrows():
			rotation = get_label_rotation((row['theta_start']+row['theta_end'])/2)
			ax.text(s=row['chrom'],
				x=(row['theta_start']+row['theta_end'])/2,
				y=circos_configs[0].get('radius', 0.9)*max(figsize)*1.12,
				#fontsize=10,
				rotation=rotation,
				ha='center',
				va='center')

	if circos_configs[0].get('tick_unit', 0) != 0:
		for index, row in chrom_regions.iterrows():
			ticks = get_ticks(row, circos_configs[0].get('tick_unit', 0))
			ticks_et = row['theta_start']-(ticks-row['start'])/len_per_theta
			radius = circos_configs[0].get('radius', 0.9)*max(figsize)
			if circos_configs[0].get('tick_orientation', 'outside') == 'outside':
				ax.vlines(ticks_et,
					[radius+circos_configs[0].get('width', 1)]*len(ticks_et),
					[radius+circos_configs[0].get('width', 1)+circos_configs[0].get('tick_length', 0.1)]*len(ticks_et))
			elif circos_configs[0].get('tick_orientation', 'outside') == 'inside':
				ax.vlines(ticks_et,
					[radius]*len(ticks_et),
					[radius-circos_configs[0].get('tick_length', 0.1)]*len(ticks_et))

			if circos_configs[0].get('tick_label') is not None:
				tick_label = circos_configs[0].get('tick_label')
				labels = {"":1,"K":1000,"M":1000000,"G":1000000000}

				for i,tick in enumerate(ticks): 
					label = "{0:.2f}{1}".format(tick/labels[tick_label], tick_label)
					rotation = get_label_rotation(ticks_et[i])
					if circos_configs[0].get('tick_orientation', 'outside') == 'outside':
						ax.text(s=label,
							x=ticks_et[i],
							y=radius+circos_configs[0].get('width', 1)+circos_configs[0].get('tick_length', 0.1)+0.2,
							#fontsize=10,
							rotation=rotation,
							ha='center',
							va='center')
					elif circos_configs[0].get('tick_orientation', 'outside') == 'inside':
						ax.text(s=label,
							x=ticks_et[i],
							y=radius-circos_configs[0].get('tick_length', 0.1)-0.2,
							#fontsize=10,
							rotation=rotation,
							ha='center',
							va='center')

	for i in range(1, len(circos_configs)):
		if circos_configs[i].get('type') == 'highlight':
			tmp_pd = pd.read_csv(circos_configs[i].get('file'), sep="\t", names=['chrom','start','end','name','score','strand'])
			tmp_pd = get_valid_regions(chrom_regions, tmp_pd)
			vmin = tmp_pd['score'].min()
			vmax = tmp_pd['score'].max()

			color,colormap,colorlist = get_colorlist(circos_configs[i].get('color'), vmin, vmax)

			for index, row in tmp_pd.iterrows():
				valid,ts,te = get_theta(chrom_regions, row, len_per_theta)
				if not valid:
					continue
				if color is not None:
					bar_color = color[(int(row['score'])-vmin)%len(color)]
				else:
					bar_color = colorlist.to_rgba(row['score'])

				ax.bar((ts+te)/2,
					circos_configs[i].get('width', 1),
					color=bar_color,
					width=(te-ts),
					bottom=circos_configs[i].get('radius', 0.9)*max(figsize))

		if circos_configs[i].get('type') == 'bar':
			tmp_pd = pd.read_csv(circos_configs[i].get('file'), sep="\t", names=['chrom','start','end','name','score','strand'])
			tmp_pd = get_valid_regions(chrom_regions, tmp_pd)
			tmp_pd['score'] = noramlization_0(tmp_pd['score'])
			
			color,colormap,colorlist = get_colorlist(circos_configs[0].get('color'), 0, chrom_regions.shape[0]-1)
			for index, row in chrom_regions.iterrows():
				if color is not None:
					chrom_color = color[index%len(color)]
				else:
					chrom_color = colorlist.to_rgba(index)
				ax.bar((row['theta_start']+row['theta_end'])/2,
					0.01,
					color=chrom_color,
					width=(row['theta_end']-row['theta_start']),
					bottom=circos_configs[i].get('radius', 0.6)*max(figsize))

			color,colormap,colorlist = get_colorlist(circos_configs[i].get('color'), -1, 1)
			for index, row in tmp_pd.iterrows():
				valid,ts,te = get_theta(chrom_regions, row, len_per_theta)
				if not valid:
					continue
				if color is not None:
					bar_color = color[0]
				else:
					bar_color = colorlist.to_rgba(row['score'])

				ax.bar((ts+te)/2,
					circos_configs[i].get('width', 1)*row['score'],
					color=bar_color,
					width=(te-ts),
					bottom=circos_configs[i].get('radius', 0.6)*max(figsize))

		if circos_configs[i].get('type') == 'bigwig':
			tmp_pd = pd.read_csv(circos_configs[i].get('region_file'), sep="\t", names=['chrom','start','end'])
			tmp_pd = get_valid_regions(chrom_regions, tmp_pd)
			tmp_pd = truncate_regions(tmp_pd, circos_configs[i].get('res'))
			bw = pyBigWig.open(circos_configs[i].get('file'))
			
			color,colormap,colorlist = get_colorlist(circos_configs[0].get('color'), 0, chrom_regions.shape[0]-1)
			for index, row in chrom_regions.iterrows():
				if color is not None:
					chrom_color = color[index%len(color)]
				else:
					chrom_color = colorlist.to_rgba(index)
				ax.bar((row['theta_start']+row['theta_end'])/2,
					0.01,
					color=chrom_color,
					width=(row['theta_end']-row['theta_start']),
					bottom=circos_configs[i].get('radius', 0.6)*max(figsize))

			color,colormap,colorlist = get_colorlist(circos_configs[i].get('color'), 0, chrom_regions.shape[0]-1)
			for index, row in tmp_pd.iterrows():
				if color is not None:
					scores_color = color[index%len(color)]
				else:
					scores_color = colorlist.to_rgba(index)

				valid, pos = get_theta_regions(chrom_regions, row, circos_configs[i].get('res'), len_per_theta)
				pos_tmp = pos[-1]
				pos = pos - circos_configs[i].get('res')/2/len_per_theta
				if pos[-1] < pos_tmp:
					pos = np.delete(pos, -1)
				scores = np.array(bw.stats(row['chrom'],row['start'],row['end'], 
					nBins=int((row['end']-row['start'])/circos_configs[i].get('res')))).astype(float)
				scores = noramlization(scores)*circos_configs[i].get('width', 1) + circos_configs[i].get('radius', 0.5)*max(figsize)
				while len(pos)>len(scores):
					pos = np.delete(pos, -1)
				ax.fill_between(pos, circos_configs[i].get('radius', 0.5)*max(figsize), scores,
					linewidth=0.05,
					color=scores_color,
					facecolor=scores_color)

		if circos_configs[i].get('type') == 'link':
			tmp_pd = pd.read_csv(circos_configs[i].get('file'), sep="\t", names=['chrom1','start1','end1','chrom2','start2','end2','score'])
			vmin = tmp_pd['score'].min()
			vmax = tmp_pd['score'].max()

			color,colormap,colorlist = get_colorlist(circos_configs[i].get('color'), vmin, vmax)

			for index, row in tmp_pd.iterrows():
				tmp_pd_1 = row.loc[['chrom1','start1','end1']].rename({'chrom1':'chrom','start1':'start','end1':'end'})
				tmp_pd_2 = row.loc[['chrom2','start2','end2']].rename({'chrom2':'chrom','start2':'start','end2':'end'})

				valid1,ts1,te1 = get_theta(chrom_regions, tmp_pd_1, len_per_theta)
				valid2,ts2,te2 = get_theta(chrom_regions, tmp_pd_2, len_per_theta)

				if not valid1 or not valid2:
					continue
				if color is not None:
					link_color = color[(int(row['score'])-vmin)%len(color)]
				else:
					link_color = colorlist.to_rgba(row['score'])

				radius = circos_configs[i].get('radius', 0.5)*max(figsize)
				points = [(ts1,radius), ((ts1+te1)/2,radius), (te1,radius),# start1, through point, end1
							(0,0),
							(ts2,radius), ((ts2+te2)/2,radius), (te2,radius),# start2, through point, end2
							(0,0),
							(ts1,radius)]
				codes = [Path.CURVE3]*len(points)
				codes[0] = Path.MOVETO
				path = Path(points, codes)
				patch = PathPatch(path, facecolor=link_color, edgecolor=link_color, alpha=0.3)
				ax.add_patch(patch)

		if circos_configs[i].get('type') == 'hic':
			tmp_pd = pd.read_csv(circos_configs[i].get('region_file'), sep="\t", names=['chrom','start','end'])
			tmp_pd = get_valid_regions(chrom_regions, tmp_pd)
			depth = circos_configs[i].get('depth')/len_per_theta

			for index, row in tmp_pd.iterrows():
				hic_region = (row['chrom'],row['start'],row['end'])
				hic_matrix = get_hic_matrix(circos_configs[i].get('file'), hic_region)
				if circos_configs[i].get('transform', 'log1p') == 'log1p':
					hic_matrix += 1
					norm = mc.LogNorm()
					vmax = np.percentile(hic_matrix.diagonal(1), 80)
					vmin = hic_matrix.min()
				else:
					norm = None
					vmax = np.percentile(hic_matrix.diagonal(1), 80)
					vmin = hic_matrix.min()

				valid, pos = get_theta_regions(chrom_regions, row, circos_configs[i].get('res'), len_per_theta)
				t = np.array([[1,0.5],[-1,0.5]])
				pos_tmp = np.dot(np.array([(i[1],i[0]) for i in itertools.product(pos[::-1],pos)]),t)
				n = len(pos)
				
				X = pos_tmp[:,1].reshape(n,n)
				Y = pos_tmp[:,0].reshape(n,n)
				Y[Y<0] = 0
				Y[Y>depth] = depth
				Y = noramlization(Y)*circos_configs[i].get('width', 1) + circos_configs[i].get('radius', 0.5)*max(figsize)

				color,colormap,colorlist = get_colorlist(circos_configs[i].get('color'), vmin, vmax)
				ax.pcolormesh(X,Y, np.flipud(hic_matrix), 
					cmap=colormap,
					vmin=vmin,
					vmax=vmax, 
					norm=norm)

	plt.savefig(outfig_name)


def plot_heatmap(config, outfig_name, **kwargs):
	figsize = kwargs.get("figsize", (16, 9))
	fig = plt.figure(figsize=figsize)
	ax = fig.add_axes([0,0,1,1])
	ax.axis('off')

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
	if color_name is not None:
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


def get_valid_regions(chrom_pd, regions_pd):
	# We don't support to transform overlapped regions to seperate regions
	for index, row in regions_pd.iterrows():
		chromID = get_chromID(chrom_pd, row)
		if chromID == -1:
			regions_pd.drop(index, inplace=True)
		elif row['start'] < chrom_pd.loc[chromID, 'end'] \
			and row['end'] > chrom_pd.loc[chromID, 'start']:
			if row['start'] < chrom_pd.loc[chromID, 'start']:
				regions_pd.loc[index,'start'] = chrom_pd.loc[chromID, 'start']
			if row['end'] > chrom_pd.loc[chromID, 'end']:
				regions_pd.loc[index,'end'] = chrom_pd.loc[chromID, 'end']
		else:
			regions_pd.drop(index, inplace=True)
	return regions_pd


def truncate_regions(regions_pd, res):
	for index, row in regions_pd.iterrows():
		tmp = (row['end']-row['start']) % res
		regions_pd.loc[index,'end'] = row['end'] - ((row['end']-row['start']) % res)
	return regions_pd


def get_chromID(chrom_pd, region_pd):
	for index, row in chrom_pd.iterrows():
		if region_pd['chrom'] == row['chrom']:
			return index
	return -1


def get_theta(chrom_pd, region_pd, len_per_theta):
	chromID = get_chromID(chrom_pd, region_pd)
	if chromID == -1:
		return False,0,0
	else:
		ts = chrom_pd.loc[chromID, 'theta_start']-(region_pd['start']-chrom_pd.loc[chromID, 'start'])/len_per_theta
		te = chrom_pd.loc[chromID, 'theta_start']-(region_pd['end']-chrom_pd.loc[chromID, 'start'])/len_per_theta
		return True,ts,te


def get_theta_regions(chrom_pd, region_pd, res, len_per_theta):
	valid,ts,te = get_theta(chrom_pd, region_pd, len_per_theta)
	pos = np.arange(ts, te, -res/len_per_theta)
	if len(pos) != int((region_pd['end']-region_pd['start'])/res)+1:
		pos = np.append(pos, te)
	return valid, pos


def get_label_rotation(rad):
	rotation = np.rad2deg(rad)-90
	if rotation < -90 and rotation > -270:
		rotation += 180
	return rotation


def get_ticks(chrom_pd, unit):
	ts,te = 0,0
	if chrom_pd['start']%unit == 0:
		ts = chrom_pd['start']
	else:
		ts = chrom_pd['start'] - chrom_pd['start']%unit + unit
	if chrom_pd['end']%unit == 0:
		te = chrom_pd['end']
	else:
		te = chrom_pd['end'] - chrom_pd['end']%unit
	return np.arange(ts,te+1,unit)

