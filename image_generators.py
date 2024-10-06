import numpy as np
from PIL import Image
import random
import sklearn.utils

from keras.preprocessing import image
from keras.utils import Sequence, to_categorical
from torch.utils.data import Dataset as _TorchDataset, DataLoader as _TorchDataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import resize

from dataset import Dataset


class ImageGenerator(Sequence):
	def __init__(self, data, target_size=(256, 256), color_mode='rgb', squeeze_dims=False, flatten=False, batch_size=32, shuffle=True):
		self.data = data
		self.target_size = target_size
		self.cmap = color_mode
		self.squeeze = squeeze_dims
		self.flatten = flatten
		self.batch_size = batch_size
		self.batch = None
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		return int(np.ceil(len(self.data) / self.batch_size))

	def __getitem__(self, index):
		self._init_batch(min(self.batch_size, len(self.data) - index * self.batch_size))
		self._read_batch(index * self.batch_size)
		return self.batch

	def _init_batch(self, size):
		self.batch = np.empty((size, *self.target_size, 3), dtype=float)

	def _read_batch(self, start):
		for i, sample in enumerate(self.data[start:start + self.batch_size]):
			self._read_sample(i, sample)

	def _read_sample(self, i, sample):
		self.batch[i] = self._load_img(sample.file)

	def _load_img(self, f, target_size=None, color_mode=None, squeeze=None, flatten=None):
		if not target_size:
			target_size = self.target_size
		if not color_mode:
			color_mode = self.cmap
		if squeeze is None:
			squeeze = self.squeeze
		if flatten is None:
			flatten = self.flatten
		x = image.load_img(f, target_size=target_size, color_mode=color_mode)
		x = image.img_to_array(x) / 255
		if flatten:
			x = x.reshape((x.shape[0] * x.shape[1]), 3 if color_mode == 'rgb' else 1)
		if squeeze:
			x = np.squeeze(x)
		return x

	def on_epoch_end(self):
		if self.shuffle:
			random.shuffle(self.data)


class LabeledImageGenerator(ImageGenerator):
	def __init__(self, data, n_classes=None, *args, **kw):
		super().__init__(data, *args, **kw)
		self.n_classes = n_classes
		if self.n_classes is None:
			self.n_classes = max(s.label for s in data)

	def _init_batch(self, size):
		self.batch = (
			np.empty((size, *self.target_size, 3), dtype=float),
			np.empty((size, self.n_classes), dtype=int)
		)

	def _read_sample(self, i, sample):
		self.batch[0][i] = self._load_img(sample.file)
		self.batch[1][i] = to_categorical(sample.label, num_classes=self.n_classes, dtype=int)


class ImageTupleGenerator(ImageGenerator):
	def __init__(self, datasets, *args, **kw):
		sets = [set(_SampleWrapper(s) for s in dataset) for dataset in datasets]
		intersection = set.intersection(*sets)

		self.datasets = [Dataset(data=sorted([s for s in dataset if _SampleWrapper(s) in intersection], key=lambda s: s.basename), **dataset.settings) for dataset in datasets]
		super().__init__(self.datasets[0], *args, **kw)

		assert all(len(d1) == len(d2) for d1, d2 in zip(self.datasets[:-1], self.datasets[1:]))
		discarded = set.union(*sets) - intersection
		print(f"Discarded {len(discarded)} images because their matches were not found: {', '.join(str(s) for s in discarded)}.")
		print(f"Found {len(self.data)} valid image tuples.")

		if isinstance(self.cmap, str):
			self.cmap = [self.cmap] * len(self.datasets)
		if len(self.cmap) != len(self.datasets):
			raise ValueError(f"Length of cmap ({len(self.cmap)}) not equal to number of datasets ({len(self.datasets)}).")
		if isinstance(self.squeeze, bool):
			self.squeeze = [self.squeeze] * len(self.datasets)
		if len(self.squeeze) != len(self.datasets):
			raise ValueError(f"Length of squeeze ({len(self.squeeze)}) not equal to number of datasets ({len(self.datasets)}).")
		if isinstance(self.flatten, bool):
			self.flatten = [self.flatten] * len(self.datasets)
		if len(self.flatten) != len(self.datasets):
			raise ValueError(f"Length of flatten ({len(self.flatten)}) not equal to number of datasets ({len(self.datasets)}).")

	def _init_batch(self, size):
		img_sizes = [(self.target_size[0] * self.target_size[1],) if flatten else self.target_size for flatten in self.flatten]
		img_sizes = [img_size + ((3,) if cmap == 'rgb' else (1,)) for img_size, cmap in zip(img_sizes, self.cmap)]
		img_sizes = [tuple(x for x in img_size if x != 1) if squeeze else img_size for img_size, squeeze in zip(img_sizes, self.squeeze)]
		self.batch = tuple(np.empty((size, *img_size), dtype=float) for img_size in img_sizes)

	def _read_batch(self, start):
		for d, dataset in enumerate(self.datasets):
			for i, sample in enumerate(dataset[start:start + self.batch_size]):
				self._read_sample(d, i, sample)

	def _read_sample(self, d, i, sample):
		self.batch[d][i] = self._load_img(sample.file, color_mode=self.cmap[d], squeeze=self.squeeze[d], flatten=self.flatten[d])

	def on_epoch_end(self):
		if self.shuffle:
			self.datasets[:] = sklearn.utils.shuffle(*self.datasets)


class TorchDataset(_TorchDataset):
	def __init__(self, data, target_size=None, color_mode=None):
		self.data = data
		self.target_size = target_size or (256, 256)
		self.cmap = color_mode or 'rgb'

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		x = read_image(self.data[i].file, ImageReadMode.RGB if self.cmap == 'rgb' else ImageReadMode.GRAY)
		if self.target_size and x.shape[:2] != self.target_size:
			x = resize(x, self.target_size)
		x = x / 255
		return x


class TorchDataLoader(_TorchDataLoader):
	def __init__(self, data, target_size=None, *args, color_mode=None, **kw):
		super().__init__(TorchDataset(data, target_size, color_mode), *args, **kw)


class _SampleWrapper:
	def __init__(self, sample, custom_eq=None):
		self.sample = sample
		self.eq = custom_eq if custom_eq else lambda s, o: all(getattr(s, attr) == getattr(o, attr) for attr in ('id', 'eye', 'direction', 'n'))

	def __eq__(self, other):
		if isinstance(other, _SampleWrapper):
			return self.eq(self.sample, other.sample)
		return NotImplemented

	def __hash__(self):
		return hash((self.sample.id, self.sample.eye, self.sample.direction, self.sample.n))

	def __str__(self):
		return str(self.sample)
