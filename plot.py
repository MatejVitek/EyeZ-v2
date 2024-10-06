import itertools
import math
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import os.path as osp


class Figure(object):
	def __init__(self, name, colors, styles, labels, **kw):
		"""
		Wrapper for a figure and its corresponding settings

		:param name: Figure name
		:type  name: str or int or None
		:param colors: Color cycler
		:param labels: Label cycler
		:param save: Filename to save figure to. If None, figure won't be saved.
		:type  save: str or None
		:param str font_family: Name of font family to use for labels
		:param int font_size: Size of label fonts
		:param str xlabel: X label
		:param str ylabel: Y label
		:param x_tick_formatter: Tick formatter function for x axis
		:param y_tick_formatter: Tick formatter function for y axis
		:param tick_formatter: Same tick formatter function for both axes
		:param xlim: Axis limits for x axis
		:param ylim: Axis limits for y axis
		:param lim: Same axis limits for both axes
		:param xticks: Tick positions and labels for x axis
		:param yticks: Tick positions and labels for y axis
		:param ticks: Same tick positions and labels for both axes
		:param bool grid: Should grid be drawn
		:param str grid_which: Which ticks to draw gridlines at
		:param str grid_axis: Which axes should have gridlines
		:param str grid_kw: Other keyword arguments to pass to :py:matplotlib.pyplot.grid
		:param legend_loc: Location of legend
		:param legend_size: Font size for legend
		:param margins: Margins for both axes
		:param pad: Padding for both axes
		"""

		self.name = name
		self.colors = colors
		self.color = next(self.colors)
		self.styles = styles
		self.style = next(self.styles)
		self.labels = labels
		self.label = next(self.labels)
		self.settings = kw
		self._ymin = float('inf')
		self._ymax = float('-inf')

	def __call__(self, *args, **kw):
		return self.plot(*args, **kw)

	def __enter__(self):
		return self.init()

	def init(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if exc_tb is None:
			self.finalize()

	def finalize(self):
		self.save()
		self.close()

	def save(self):
		save = self.settings.get('save')
		separate_legend = self.settings.get('separate_legend')
		if save:
			if separate_legend:
				handles, labels = plt.gca().get_legend_handles_labels()
				plt.figure('Legend')
				plt.legend(handles, labels, ncol=self.settings.get('legend_col', 2), loc='center', fontsize=self.settings.get('legend_size', 20))
				plt.axis('off')
				plt.savefig(osp.join(osp.dirname(save), 'Legend.pdf'), bbox_inches='tight', pad_inches=0)
			print(f"Saving figure to {save}.")
			plt.figure(self.name)
			if separate_legend:
				plt.gca().get_legend().set_visible(False)
			plt.savefig(save, bbox_inches='tight')
			if separate_legend:
				plt.gca().get_legend().set_visible(True)

	def close(self):
		plt.close(self.name)

	def plot(self, x, y, **kw):
		"""
		Plot using :py:matplotlib.pyplot.plot

		:param x: X data
		:param y: Y data
		:param kw: Other keyword arguments passed to :py:matplotlib.pyplot.plot
		"""

		cfg = self.settings

		plt.figure(self.name)
		plt.rcParams['font.family'] = cfg.get('font') or cfg.get('font_family', 'Times New Roman')
		plt.rcParams['font.size'] = cfg.get('font_size', 30)

		plt.plot(
			x, y, kw.pop('style', self.style),
			color=kw.pop('color', self.color),
			linewidth=(kw.get('linewidth') or cfg.get('linewidth', 2)),
			label=kw.pop('label', self.label),
			**kw
		)

		plt.xscale(cfg.get('xscale', 'linear'))
		plt.yscale(cfg.get('yscale', 'linear'))
		plt.xlabel(cfg.get('xlabel', ''))
		plt.ylabel(cfg.get('ylabel', ''))
		plt.gca().xaxis.set_major_formatter(FuncFormatter(
			cfg.get('x_tick_formatter') or
			cfg.get('tick_formatter', def_tick_format)
		))
		plt.gca().yaxis.set_major_formatter(FuncFormatter(
			cfg.get('y_tick_formatter') or
			cfg.get('tick_formatter', def_tick_format)
		))
		plt.xlim(cfg.get('xlim') or cfg.get('lim'))
		ylim = cfg.get('ylim') or cfg.get('lim')
		if ylim == 'auto':
			self._ymin = min(self._ymin, y)
			self._ymax = max(self._ymax, y)
			ytick_size = self._nice_tick_size()
			ymin = max(0, self._ymin - .5 * ytick_size)
			ymax = self._ymax + .5 * ytick_size
			plt.ylim(ymin, ymax)
			plt.gca().yaxis.set_major_locator(MultipleLocator(ytick_size))
		else:
			plt.ylim(ylim)
			plt.yticks(cfg.get('yticks', cfg.get('ticks')))
		plt.xticks(cfg.get('xticks', cfg.get('ticks')))
		plt.grid(
			cfg.get('grid', True),
			which=cfg.get('grid_which', 'major'),
			axis=cfg.get('grid_axis', 'both'),
			**cfg.get('grid_kw', {})
		)
		if self.label:
			handles, labels = plt.gca().get_legend_handles_labels()  #
			by_label = dict(zip(labels, handles))                    # Remove duplicate labels
			plt.legend(by_label.values(), by_label.keys(), loc=cfg.get('legend_loc', 'best'), fontsize=cfg.get('legend_size', 20))

		plt.margins(cfg.get('margins', 0))
		plt.tight_layout(pad=cfg.get('pad', 0))

	def _nice_tick_size(self, min_ticks=3, max_ticks=7):
		diff = self._ymax - self._ymin
		if diff == 0:
			return .1
		return min(
			oom(diff) * np.array([.1, .2, .5, 1, 2, 5]),  # Different possible tick sizes
			key=lambda tick_size: (max(0, min_ticks - (diff // tick_size + 1), (diff // tick_size + 1) - max_ticks), diff // tick_size + 1)  # Return the one closest to the requested number of ticks. If several are in the range, return the one with the fewest ticks.
		)

	def next(self):
		self.color = next(self.colors)
		self.style = next(self.styles)
		self.label = next(self.labels)


class Painter(object):
	def __init__(self, colors='hsv', styles=('-', '--', ':', '-.'), labels=None, k=None, interactive=True, pause_on_end=True, **kw):
		"""
		Class for drawing multiple plots to a figure (or multiple figures)

		:param colors: Name of a colormap or a sequence of colors
		:param styles: Sequence of styles
		:param labels: List of labels for plot legend. If None, no legend will be drawn.
		:param k: Number of colors to sample from the colormap. If None, the entire colormap will be used.
		:type  k: int or None
		:param bool interactive: Should execution continue after drawing
		:param bool pause_on_end: Should execution halt when Painter is finalised (ignored if interactive is False)
		:param kw: Default settings for all figures (see :py:Figure)
		"""

		self.figures = {}
		self.colors = colors
		self.styles = styles
		self.labels = labels
		self.k = k
		self.interactive = interactive
		self.pause = pause_on_end
		self.default_settings = kw

	def __call__(self, *args, **kw):
		return self.draw(*args, **kw)

	def __enter__(self):
		return self.init()

	def init(self):
		if self.interactive:
			plt.ion()
		for figure in self:
			figure.init()
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		if exc_tb is None:
			self.finalize()
		elif self.interactive:
			plt.ioff()
			plt.show()

	def __getitem__(self, name):
		return self.figures[name]

	def __setitem__(self, name, figure):
		self.figures[name] = figure

	def __contains__(self, figure):
		if isinstance(figure, Figure):
			return figure in self.figures.values()
		return figure in self.figures

	def __iter__(self):
		return iter(self.figures.values())

	def finalize(self):
		for figure in self:
			figure.save()
		if self.interactive:
			plt.ioff()
			if self.pause:
				plt.show()
		for figure in self:
			figure.close()

	def add_figure(self, figure=None, colors=None, styles=None, labels=0, k=0, **kw):
		"""
		Add a Figure context to the Painter object

		:param figure: Figure name or number
		:type  figure: str or int or None
		:param colors: Name of a colormap or a sequence of colors. If None, Painter's default will be used.
		:param labels: Sequence of labels. If 0, Painter's default will be used. If None, no legend will be drawn.
		:param k: Number of colors to sample. If 0, Painter's default will be used. If None, all colors will be used.
		:type  k: int or None
		:param kw: Additional figure settings, overriding Painter's defaults (see :py:Figure)

		:return: Added Figure context
		:rtype:  Figure
		"""

		if colors is None:
			colors = self.colors
		if styles is None:
			styles = self.styles
		if labels == 0:
			labels = self.labels
		if k == 0:
			k = self.k
		colors = cycle_colors(colors, k)
		styles = itertools.cycle(styles) if styles else itertools.repeat('-')
		labels = itertools.cycle(labels) if labels else itertools.repeat(None)

		kw.update((k, v) for (k, v) in self.default_settings.items() if k not in kw)
		self[figure] = Figure(figure, colors, styles, labels, **kw)
		return self[figure].init()

	def draw(self, x, y, figure=None, figure_kw=None, **kw):
		"""
		Plot x and y to a figure

		:param x: X data
		:param y: Y data
		:param figure: Figure name or number
		:type  figure: str or int or None
		:param figure_kw: Optional additional figure settings for a new figure (see :py:Figure)
		:type  figure_kw: dict or None
		:param kw: Additional plot settings, overriding Figure defaults (see :py:Figure.plot)
		"""

		if figure not in self:
			if figure_kw:
				self.add_figure(figure, **figure_kw)
			else:
				self.add_figure(figure)
		self[figure].plot(x, y, **kw)
		if self.interactive:
			plt.draw()
			plt.pause(1)
		else:
			plt.show()

	def next(self):
		for figure in self:
			figure.next()


def cycle_colors(colors, k=None):
	# Determine correct k if unspecified
	if k is None:
		if isinstance(colors, str):
			k = 32
		else:
			k = len(colors)

	try:
		# HSV-uniform colormap
		if isinstance(colors, str):
			if colors.lower() in ('hsv', 'uniform', 'hsvuniform', 'hsv-uniform', 'hsv_uniform'):
				colors = [hsv_to_rgb((h, 1, 1)) for h in np.linspace(0, 1, k, endpoint=False)]
			else:
				# Other colormaps
				colors = plt.cm.get_cmap(colors)(np.linspace(0, 1, k, endpoint=False))
		colors = colors(np.linspace(0, 1, k, endpoint=False))
		while True:
			yield from colors
	except (TypeError, AttributeError):
		pass

	# Yield from array of colors (either passed as argument or defined above in HSV)
	while True:
		idx = np.linspace(0, len(colors), k, endpoint=False, dtype=int)
		for i in idx:
			yield colors[i]


def def_tick_format(x, _):
	return np.format_float_positional(x, precision=3, trim='-')


def scientific_tick_format(x, _):
	return np.format_float_scientific(x, precision=3, trim='-')


def exp_format(x, _):
	return rf"$\mathregular{{10^{{{int(round(np.log10(x)))}}}}}$"


def oom(x):
	return 10 ** math.floor(math.log10(x))  # Order of magnitude: oom(0.9) = 0.1, oom(30) = 10
