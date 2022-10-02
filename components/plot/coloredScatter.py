""" Creates a simple scatter plot, colored by some key """

import matplotlib.pylab as plt
import pandas as pd
from components.plot import colorPalette

__author__ = "Matthew Dirks"

def make(df, xColumnName, yColumnName, groupByColumn=None, 
	     title=None, plottingLimits=None, info=None, 
	     groupStyles=None, equalAspectRatio=False, 
	     alpha=1.0, countPoints=True, figsize=(12,8)):
	# Create figure
	fig, ax = plt.subplots(1,1, figsize=figsize)

	# Use colors if provided, else color everything blue
	if (groupByColumn is not None):
		if (groupStyles is None):
			nColors = len(df[groupByColumn].unique())
			colors = colorPalette.getNColors(nColors)
			groupStyles = {name:{'c':color,'marker':'o','lw':0} for name, color in zip(df[groupByColumn].unique(), colors)}

		# Group data
		for idx, (groupName, group) in enumerate(df.groupby(groupByColumn, sort=False)):
			style = groupStyles[groupName]
			alpha = style.get('alpha', alpha) # first use alpha in style object if any, else default to argument alpha

			ax.scatter(group[xColumnName], group[yColumnName], c=style['c'], marker=style['marker'], label=groupName, lw=style['lw'], alpha=alpha)
	else:
		ax.scatter(df[xColumnName], df[yColumnName], c='b', lw=0, alpha=alpha)

	ax.set_xlabel(xColumnName)
	ax.set_ylabel(yColumnName)

	if (equalAspectRatio):
		ax.set_aspect('equal')
		# ax.plot([vmin,vmax],[vmin,vmax],c='#666666',ls='--')

	if (plottingLimits is not None):
		# manual limits
		vmin, vmax = plottingLimits
	else:
		# automatic limits
		vmax = df[[yColumnName,xColumnName]].max().max()
		vmin = df[[yColumnName,xColumnName]].min().min()

	ax.set_xlim([vmin,vmax])
	ax.set_ylim([vmin,vmax])

	# add count of number of points
	if (countPoints):
		ax.text(0.99, 0.01, 'n = %d' % df.shape[0], verticalalignment='bottom', horizontalalignment='right', 
			transform=ax.transAxes, fontsize=15) #color='green', 

	# add count of number of points
	# ax.text(0.99, 0.01, 'nPredictions = %d' % df.shape[0], verticalalignment='bottom', horizontalalignment='right', 
	# 	transform=ax.transAxes, fontsize=15) #color='green', 

	if (groupByColumn is not None):
		# Put a legend to the right of the current axis
		ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

		# Shrink current axis by 30% (to make room for legend)
		box = ax.get_position()
		ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])

	# setup title
	if (title is not None):
		ax.set_title(title)

	if (info is not None):
		fig.suptitle(info)

	return fig

def plot(*args, **kwargs):
	fig = make(*args, **kwargs)
	plt.show()