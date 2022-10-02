from __future__ import division, print_function
from math import ceil

__author__ = "Matthew Dirks"

def makeNiceGridFromNElements(n, targetAspect=2, aspectMin=1, aspectMax=3):
	""" If you want to plot n things in n subplots, this function will tell you how many rows and columns
	of subplots to have, such that you have a visually-appealing grid layout that is optimal if possible.

	We recommend targetAspect be 2 or 3.
	If targetAspect is 2, number of columns will be at most 2 times number of rows.
	If targetAspect is 3, number of columns will be at most 3 times number of rows.

	Shape (nRows and nColumns) are determined by minizing difference to targetAspect ratio,
	except aspect min and max allow a solution to be returned that is not optimal in terms
	of distance to targetAspect ratio but instead factors n perfectly (no leftover subplots).

	Args:
		n: Number of plots.
		targetAspect: maximum allowed ratio between number of columns and number of rows (nColumns/nRows).

	Returns:
		shape: best shape found; tuple of (nColumns, nRows)
		aspect: aspect ratio of best shape found
	"""

	# best shape and aspect 
	best1_shape = None
	best1_aspectDistance = float('inf')
	# second-best shape and aspect
	best2_shape = None
	best2_aspectDistance = float('inf')

	nrows = 0
	while (True):
		nrows += 1
		ncols = n/nrows
		# Note: aspect is monotonically decreasing as nrows increases.
		aspect = (n/nrows)/nrows
		aspectDistance = abs(aspect - targetAspect)

		if (aspect < aspectMin):
			# no more possible solutions
			break 
		else: # aspect >= aspectMin
			# can perfectly factor n while fitting within min/max constraints?
			if (ncols==round(ncols) and aspectDistance < best1_aspectDistance and aspect <= aspectMax):
				best1_aspectDistance = aspectDistance
				best1_shape = (nrows, int(ncols))

			# minimize distance to targetAspect ratio
			if (aspectDistance < best2_aspectDistance):
				best2_aspectDistance = aspectDistance
				best2_shape = (nrows, int(ceil(ncols)))

	if (best1_shape is not None):
		return best1_shape, best1_aspectDistance
	elif (best2_shape is not None):
		return best2_shape, best2_aspectDistance
	else:
		return None, None

def mplSubplots(n):
	""" Make a matplotlib figure with n subplots, arranged in a nice grid. 
	This function is mostly to serve as an example,
	but feel free to use it.
	"""
	import matplotlib.pylab as plt

	(nRows, nCols), _ = makeNiceGridFromNElements(n)

	fig, axs = plt.subplots(nRows, nCols, figsize=(20,10))
	axs = axs.squeeze()
	try:
		axs = [item for sublist in axs for item in sublist]
	except:
		pass

	return fig, axs

if __name__ == '__main__':
	for i in range(20):
		shape, aspect = makeNiceGridFromNElements(i)
		if (shape is None):
			shape = 'NONE'
			aspect = ''
		print(i, '(', shape, ') ratio=', aspect)
