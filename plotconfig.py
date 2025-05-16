import matplotlib.pyplot as plt

# plot configurations
# plt. rcParams ['image.cmap'] = 'gray_r'
plt.rcParams ['image.interpolation'] = 'none'
plt.rcParams ['figure.figsize'] = (5, 5)
plt.rcParams ['image.aspect'] = 'auto'
plt.rcParams ['svg.fonttype'] = 'none'
# Ensure text is saved as text, not paths
# plt.rcParams['pdf.fonttype'] = 'none' # Ensure text is saved as text, not paths
plt.rcParams ['pdf.fonttype'] = 'truetype' # Use 'truetype for PDF font type
# Set the default style
plt.style.use('ggplot')
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Set the font
plt.rcParams ['font.size'] = 11
# plt.rcParams ['font.family'] = 'serif'
plt.rcParams ['font.family'] = 'sans-serif'
# plt.rcParams ['font.family'] = 'monospace'
plt.rcParams ['font.sans-serif'] = ['Arial']
# plt.rcParams["text.usetex"] = True
# Set the DPI (dots per inch)
plt.rcParams ['figure.dpi'] = 150
# Set the line width
plt.rcParams ['lines.linewidth'] = 2
# Set the grid
plt.rcParams ['grid.color'] = '#cccccc'
# Set the axes
plt.rcParams ['axes.facecolor'] = '#ffffff'
plt.rcParams ['axes.edgecolor'] = '#000000'
plt.rcParams ['axes.grid'] = False
plt.rcParams ['axes.axisbelow'] = True
# Set the legend
plt.rcParams ['legend.frameon'] = False
# Set the markers
plt.rcParams ['scatter.marker'] = 'o'
# Set the error bars
plt.rcParams ['errorbar.capsize'] = 3
# Set the histogram bins
plt.rcParams ['hist.bins'] = 'auto'
plt.rcParams ['figure.autolayout'] = True
plt.rcParams ['axes.spines.top'] = False
plt.rcParams ['axes.spines.right'] = False

