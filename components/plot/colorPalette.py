import matplotlib.pylab as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import numpy as np

__author__ = "Matthew Dirks"

def convertRgbFloatsToHex(colorRgbTuple):
	""" Convert color code from RGB (float between 0 and 1) to hex """
	r,g,b = (int(x*255) for x in colorRgbTuple)
	return "#{0:02x}{1:02x}{2:02x}".format(colorValueClamp(r), colorValueClamp(g), colorValueClamp(b))

c268 = ['#000000','#FFFF00','#1CE6FF','#FF34FF','#FF4A46','#008941','#006FA6','#A30059','#FFDBE5','#7A4900','#0000A6','#63FFAC','#B79762','#004D43','#8FB0FF','#997D87','#5A0007','#809693','#FEFFE6','#1B4400','#4FC601','#3B5DFF','#4A3B53','#FF2F80','#61615A','#BA0900','#6B7900','#00C2A0','#FFAA92','#FF90C9','#B903AA','#D16100','#DDEFFF','#000035','#7B4F4B','#A1C299','#300018','#0AA6D8','#013349','#00846F','#372101','#FFB500','#C2FFED','#A079BF','#CC0744','#C0B9B2','#C2FF99','#001E09','#00489C','#6F0062','#0CBD66','#EEC3FF','#456D75','#B77B68','#7A87A1','#788D66','#885578','#FAD09F','#FF8A9A','#D157A0','#BEC459','#456648','#0086ED','#886F4C','#34362D','#B4A8BD','#00A6AA','#452C2C','#636375','#A3C8C9','#FF913F','#938A81','#575329','#00FECF','#B05B6F','#8CD0FF','#3B9700','#04F757','#C8A1A1','#1E6E00','#7900D7','#A77500','#6367A9','#A05837','#6B002C','#772600','#D790FF','#9B9700','#549E79','#FFF69F','#201625','#72418F','#BC23FF','#99ADC0','#3A2465','#922329','#5B4534','#FDE8DC','#404E55','#0089A3','#CB7E98','#A4E804','#324E72','#6A3A4C','#83AB58','#001C1E','#D1F7CE','#004B28','#C8D0F6','#A3A489','#806C66','#222800','#BF5650','#E83000','#66796D','#DA007C','#FF1A59','#8ADBB4','#1E0200','#5B4E51','#C895C5','#320033','#FF6832','#66E1D3','#CFCDAC','#D0AC94','#7ED379','#012C58','#7A7BFF','#D68E01','#353339','#78AFA1','#FEB2C6','#75797C','#837393','#943A4D','#B5F4FF','#D2DCD5','#9556BD','#6A714A','#001325','#02525F','#0AA3F7','#E98176','#DBD5DD','#5EBCD1','#3D4F44','#7E6405','#02684E','#962B75','#8D8546','#9695C5','#E773CE','#D86A78','#3E89BE','#CA834E','#518A87','#5B113C','#55813B','#E704C4','#00005F','#A97399','#4B8160','#59738A','#FF5DA7','#F7C9BF','#643127','#513A01','#6B94AA','#51A058','#A45B02','#1D1702','#E20027','#E7AB63','#4C6001','#9C6966','#64547B','#97979E','#006A66','#391406','#F4D749','#0045D2','#006C31','#DDB6D0','#7C6571','#9FB2A4','#00D891','#15A08A','#BC65E9','#FFFFFE','#C6DC99','#203B3C','#671190','#6B3A64','#F5E1FF','#FFA0F2','#CCAA35','#374527','#8BB400','#797868','#C6005A','#3B000A','#C86240','#29607C','#402334','#7D5A44','#CCB87C','#B88183','#AA5199','#B5D6C3','#A38469','#9F94F0','#A74571','#B894A6','#71BB8C','#00B433','#789EC9','#6D80BA','#953F00','#5EFF03','#E4FFFC','#1BE177','#BCB1E5','#76912F','#003109','#0060CD','#D20096','#895563','#29201D','#5B3213','#A76F42','#89412E','#1A3A2A','#494B5A','#A88C85','#F4ABAA','#A3F3AB','#00C6C8','#EA8B66','#958A9F','#BDC9D2','#9FA064','#BE4700','#658188','#83A485','#453C23','#47675D','#3A3F00','#061203','#DFFB71','#868E7E','#98D058','#6C8F7D','#D7BFC2','#3C3E6E','#D83D66','#2F5D9B','#6C5E46','#D25B88','#5B656C','#00B57F','#545C46','#866097','#365D25','#252F99','#00CCFF','#674E60','#FC009C','#92896B']

c12 = ['#FF2020','#C16F1C','#FCEA07','#ACF810','#2C9646','#21EDE2','#50A5FA','#0E4BB1','#C29BF9','#8224B5','#FA17FF','#B6298B']
c27 = ['#ff0000', '#c20000', '#850000', '#470000', '#ff7373', '#c25757', '#853c3c', '#472020', '#ffaa00', '#c28100', '#855800', '#473000', '#ffd073', '#c29e57', '#856c3c', '#473a20', '#aaff00', '#81c200', '#588500', '#304700', '#d0ff73', '#9ec257', '#6c853c', '#3a4720', '#00ff00', '#00c200', '#008500', '#004700', '#73ff73', '#57c257', '#3c853c', '#204720', '#00ffaa', '#00c281', '#008558', '#004730', '#73ffd0', '#57c29e', '#3c856c', '#20473a', '#00aaff', '#0081c2', '#005885', '#003047', '#73d0ff', '#579ec2', '#3c6c85', '#203a47', '#0000ff', '#0000c2', '#000085', '#000047', '#7373ff', '#5757c2', '#3c3c85', '#202047', '#aa00ff', '#8100c2', '#580085', '#300047', '#d073ff', '#9e57c2', '#6c3c85', '#3a2047', '#ff00aa', '#c20081', '#850058', '#470030', '#ff73d0', '#c2579e', '#853c6c', '#47203a']
c12_light = ['#FF2020','#C16F1C','#FCEA07','#ACF810','#2C9646','#21EDE2','#50A5FA','#0E4BB1','#C29BF9','#8224B5','#FA17FF','#B6298B']

c24 = ["#023FA5","#7D87B9","#BEC1D4","#D6BCC0","#BB7784","#FFFFFF", "#4A6FE3","#8595E1","#B5BBE3","#E6AFB9","#E07B91","#D33F6A", "#11C638","#8DD593","#C6DEC7","#EAD3C6","#F0B98D","#EF9708", "#0FCFC0","#9CDED6","#D5EAE7","#F3E1EB","#F6C4E1","#F79CD4"]
c28 = ['#ff0000', '#b00000', '#870000', '#550000', '#e4e400', '#baba00', '#878700', '#545400', '#00ff00', '#00b000', '#008700', '#005500', '#00ffff', '#00b0b0', '#008787', '#005555', '#b0b0ff', '#8484ff', '#4949ff', '#0000ff', '#ff00ff', '#b000b0', '#870087', '#550055', '#e4e4e4', '#bababa', '#878787', '#545454']

cPrimaryColors = ['#1313E2', '#11D211', '#E61313', '#FF00A2', '#00FFA8', '#00F6FF', '#FFAE00', '#FFF600', '#D200FF']

def getNColors(nColors=20):
	cm = plt.get_cmap('gist_rainbow')
	cNorm = colors.Normalize(vmin=0, vmax=nColors-1)
	scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

	# old way:
	#ax.set_color_cycle([cm(1.*i/nColors) for i in range(nColors)])
	# new way:
	# ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(nColors)])

	return [scalarMap.to_rgba(i) for i in range(nColors)]

def getNColorsAsHex(nColors):
	""" Returns n colors in hexidecimal format """
	
	# colors are 4-tuples where each value is a float in [0,1]
	_colors = getNColors(nColors)

	# throw away alpha portion
	_colors = [rgba[:3] for rgba in _colors]

	# convert to hex
	_colors = [convertRgbFloatsToHex(rgb) for rgb in _colors]

	return _colors

blues = ['#ffffff','#e5e5ff','#ccccff','#b2b2ff','#9999ff','#7f7fff','#6666ff','#4c4cff','#3232ff','#1919ff','#0000ff','#0000e5','#0000cc','#0000b2','#000099','#00007f','#000066','#00004c','#000033','#000019','#000000']

def get_color_brewer_schemes():
	import json
	with open('colorbrewer.json', 'r') as f:
		colorbrewer_schemes = json.load(f)
	return colorbrewer_schemes