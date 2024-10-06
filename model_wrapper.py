from abc import ABC, abstractmethod
import itertools
import math
import numpy as np
import os
import pickle
import scipy as sp
from time import time
from tqdm import tqdm

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD

from evaluation import Evaluation
from image_generators import ImageGenerator, LabeledImageGenerator, TorchDataLoader


class CVModel(ABC):
	def __init__(self, model):
		self.model = model
		self._verbose = None

	#TODO: When implementing this in EyeZ, should save feature vectors and error rates in addition to the distance matrices
	def evaluate(self, gallery, probe, evaluation=None, impostors=None, plot=None, verbose=1, save=None, use_precomputed=False, times=False, complexity=None, **kw):
		"""
		Evaluate wrapped model

		:param iterable gallery: List of gallery samples
		:param iterable probe: List of probe samples
		:param evaluation: Existing Evaluation to update. If None, new Evaluation will be created.
		:type  evaluation: Evaluation or None
		:param impostors: Dataset for impostor verification attempts in evaluation. If None, full G/P testing will be used.
		:type  impostors: Dataset or None
		:param plot: Plotting function, taking a tuple (x, y, figure). If None, will not plot.
		:type  plot: Callable or None
		:param int verbose: Verbosity level
		:param save: File to save distance matrix info to
		:type  save: str or None
		:param bool use_precomputed: Whether to load saved distance matrix info. If True, should also pass save name.
		:param kw: Keyword arguments to pass to :py:evaluation.compute_error_rates

		:return: Evaluation updated with newly computed metrics
		:rtype:  Evaluation
		"""

		self._verbose = verbose
		self._time = times
		if not evaluation:
			evaluation = Evaluation()

		# Load dist_matrix and imp_matrix info
		if use_precomputed and save and os.path.isfile(save):
			self._print(f"Reading info from {save}.")
			with open(save, 'rb') as f:
				dist_matrix = pickle.load(f)
				imp_matrix = pickle.load(f)
				g_classes = pickle.load(f)
				p_classes = pickle.load(f)
				imp_classes = pickle.load(f)

		# Compute dist_matrix and imp_matrix info
		else:
			dist_matrix, imp_matrix = self.dist_and_imp_matrix(gallery, probe, impostors)
			g_classes = self.classes(gallery)
			p_classes = self.classes(probe)
			imp_classes = self.classes(impostors)

		# Save dist_matrix and imp_matrix info
		if save and not (use_precomputed and os.path.isfile(save)):
			self._print(f"Saving info to {save}.")
			dir = os.path.dirname(save)
			if dir and not os.path.isdir(dir):
				os.makedirs(dir)
			with open(save, 'wb') as f:
				pickle.dump(dist_matrix, f)
				pickle.dump(imp_matrix, f)
				pickle.dump(g_classes, f)
				pickle.dump(p_classes, f)
				pickle.dump(imp_classes, f)

		# Get FAR and FRR
		far, frr, threshold = evaluation.compute_error_rates(
			dist_matrix,
			g_classes,
			p_classes,
			impostor_matrix=imp_matrix,
			impostor_classes=imp_classes,
			**kw
		)

		xvalue = kw.get('xvalue')
		# EER
		eer = evaluation.update_eer()
		self._print(f"EER: {eer}")
		if plot:
			if xvalue is not None:
				plot(xvalue, eer, figure='EER/Data', markersize=12)
			else:
				plot(threshold, far, figure='EER')
				plot(threshold, frr, figure='EER', label=None)
				plot(complexity[0], eer, figure='EER/Complexity', style='o', markersize=.05 * math.sqrt(complexity[1]), color=(*plot['EER/Complexity'].color[:3], .2), label=None)
				plot(complexity[0], eer, figure='EER/Complexity', markersize=8)

		# AUC
		auc = evaluation.update_auc()
		self._print(f"AUC: {auc}")
		if plot:
			if xvalue is not None:
				plot(xvalue, auc, figure='AUC/Data', markersize=12)
			else:
				plot(far, 1 - frr, figure='ROC Curve')
				plot(far, 1 - frr, figure='Semilog ROC Curve')
				plot(complexity[0], auc, figure='AUC/Complexity', style='o', markersize=.05 * math.sqrt(complexity[1]), color=(*plot['AUC/Complexity'].color[:3], .2), label=None)
				plot(complexity[0], auc, figure='AUC/Complexity', markersize=8)

		# VER@1FAR and VER@0.1FAR
		ver1far = evaluation.update_ver1far()
		self._print(f"VER@1FAR: {ver1far}")
		ver01far = evaluation.update_ver01far()
		self._print(f"VER@0.1FAR: {ver01far}")
		if plot:
			if xvalue is not None:
				plot(xvalue, ver1far, figure='VER@1FAR/Data', markersize=12)
				plot(xvalue, ver1far, figure='VER@0.1FAR/Data', markersize=12)
			else:
				plot(complexity[0], ver1far, figure='VER@1FAR/Complexity', style='o', markersize=.05 * math.sqrt(complexity[1]), color=(*plot['VER@1FAR/Complexity'].color[:3], .2), label=None)
				plot(complexity[0], ver1far, figure='VER@1FAR/Complexity', markersize=8)
				plot(complexity[0], ver01far, figure='VER@0.1FAR/Complexity', style='o', markersize=.05 * math.sqrt(complexity[1]), color=(*plot['VER@0.1FAR/Complexity'].color[:3], .2), label=None)
				plot(complexity[0], ver01far, figure='VER@0.1FAR/Complexity', markersize=8)

		return evaluation

	def dist_and_imp_matrix(self, gallery, probe, impostors=None):
		if not impostors:
			return self.dist_matrix(gallery, probe), None
		return self._dist_and_imp_matrix(gallery, probe, impostors)

	@abstractmethod
	def dist_matrix(self, gallery, probe):
		pass

	@abstractmethod
	def _dist_and_imp_matrix(self, gallery, probe, impostors):
		pass

	@staticmethod
	def classes(samples):
		return [s.label for s in samples] if samples else None

	def _print(self, *args, **kw):
		if self._verbose:
			print(*args, **kw)


class DirectDistanceModel(CVModel):
	"""
	Wrapper for models that only calculate distances between images (such as SIFT)

	:param DistModel model: wrapped model
	"""

	def _dist_and_imp_matrix(self, gallery, probe, impostors):
		self._print("Gallery & probe")
		dist_matrix = self.dist_matrix(gallery, probe)
		self._print("Gallery & impostors")
		imp_matrix = self.dist_matrix(gallery, impostors)
		return dist_matrix, imp_matrix

	def dist_matrix(self, gallery, probe):
		self._print("Computing distance matrix")
		dist_matrix = np.empty((len(gallery), len(probe)))

		# To take advantage of caching in DistModel, iterate over square sub-matrices in the distance matrix
		sq = self.model.size // 2
		dm_idx = [
			(g, p)
			for g_s, p_s in itertools.product(range(0, len(gallery), sq), range(0, len(probe), sq))
			for g, p in itertools.product(range(g_s, min(g_s + sq, len(gallery))), range(p_s, min(p_s + sq, len(probe))))
		]
		if self._verbose:
			dm_idx = tqdm(dm_idx)

		for g, p in dm_idx:
			dist_matrix[g, p] = self.model.distance(gallery[g], probe[p])

		if self._time:
			with open('times.tmp', 'a') as f:
				print(np.mean(self.model.feature_times), file=f)
				print(np.mean(self.model.dist_times), file=f)

		m = np.nanmax(dist_matrix)
		m = 1 if np.isnan(m) or m <= 1 else m + 1e-8
		dist_matrix[np.isnan(dist_matrix)] = m

		return dist_matrix


class PredictorModel(CVModel):
	def __init__(self, model, batch_size=32, **kw):
		"""
		Feature extractor wrapper

		:param model: Feature extractor to evaluate (NN should not include the softmax layer). Should have a predict_generator method.
		:param int batch_size: Batch size
		:param input_size: Input size. Will use model.input_shape to be determined automatically if not provided and model.input_shape exists.
		:param distance: Distance metric for feature vector comparison (passed to scipy.spacial.distance.cdist).
		                 Defaults to cosine distance.
		:param distance_normalization: Function for distance normalization.
				                       Defaults to f(d) = d/2 for 'cosine' and no normalization for anything else.
				                       If given, the function should be executable on numpy arrays.
		"""

		# Base model settings
		super().__init__(model)
		self.input_size = kw.pop('input_size', None)
		if not self.input_size:
			try:
				if len(self.model.input_shape) == 2:
					# Custom model
					self.input_size = self.model.input_shape
				else:
					# Keras NN
					self.input_size = self.model.input_shape[1:3]
				if any(size is None for size in self.input_size):
					# Go to default value
					raise TypeError()
			except (AttributeError, TypeError):
				self.input_size = (256, 256)

		# Other settings
		self.batch_size = batch_size
		self.dist = kw.pop('distance') or kw.pop('dist', 'cosine')
		self.dist_norm = kw.pop('distance_normalization', (lambda d: d/2) if self.dist == 'cosine' else None)

	def _dist_and_imp_matrix(self, gallery, probe, impostors):
		g_features = self.predict(gallery, "gallery")
		p_features = self.predict(probe, "probe")
		dist_matrix = self._dist_matrix(g_features, p_features)

		imp_features = self.predict(impostors, "impostor")
		imp_matrix = self._dist_matrix(g_features, imp_features)

		return dist_matrix, imp_matrix

	def dist_matrix(self, gallery, probe):
		t = time()
		g_features = self.predict(gallery, "gallery")
		p_features = self.predict(probe, "probe")
		if self._time:
			t = (time() - t) / (len(gallery) + len(probe))
			with open('times.tmp', 'a') as f:
				print(t, file=f)
		return self._dist_matrix(g_features, p_features)

	def _dist_matrix(self, g_features, p_features):
		t = time()
		dist_matrix = sp.spatial.distance.cdist(g_features, p_features, metric=self.dist)
		dist_matrix[np.isnan(dist_matrix)] = 1.
		if self._time:
			t = (time() - t) / dist_matrix.size
			with open('times.tmp', 'a') as f:
				print(t, file=f)
		return dist_matrix

	def predict(self, data, name="unknown"):
		self._print(f"Predicting {name} features:")
		if hasattr(self.model, 'predict_generator'):
			# Keras
			gen = ImageGenerator(
				data,
				target_size=self.input_size,
				batch_size=self.batch_size,
				shuffle=False
			)
			return self.model.predict_generator(gen, verbose=self._verbose)

		# PyTorch
		gen = TorchDataLoader(
			data,
			target_size=self.input_size,
			batch_size=self.batch_size,
			shuffle=False
		)
		results = []
		for batch in tqdm(gen) if self._verbose else gen:
			batch = batch.to('cuda')
			if batch.ndim < 4:
				batch = batch.unsqueeze(0)
			output = self.model(batch)
			output = output.detach().cpu().numpy()
			results.extend(output)
		return np.array(results)


class TrainablePredictorModel(PredictorModel):
	def __init__(self, model, **kw):
		"""
		Wrapper for a trainable model for CV. Additional parameters (see also :py:PredictorModel.__init__):

		:param int primary_epochs: Epochs to train top layers only
		:param int secondary_epochs: Epochs to train unfrozen layers and top layers
		:param int feature_size: Feature size of custom (FC) feature layer. If 0, no custom feature layer will be added.
		:param first_unfreeze: Index of the first layer to unfreeze in secondary training. If None, skip primary training and train the whole model.
		:type  first_unfreeze: int or None
		:param primary_opt: Optimizer to use in primary training
		:param secondary_opt: Optimizer to use in secondary training
		"""

		super().__init__(model, **kw)
		self.epochs1 = kw.get('primary_epochs') or kw.get('top_epochs') or kw.get('epochs1') or kw.get('epochs', 50)
		self.epochs2 = kw.get('secondary_epochs') or kw.get('unfrozen_epochs') or kw.get('epochs2', 30)
		self.feature_size = kw.get('feature_size', 1024)
		self.first_unfreeze = kw.get('first_unfreeze')
		self.opt1 = kw.get('primary_opt') or kw.get('opt1') or kw.get('opt', 'rmsprop')
		self.opt2 = kw.get('secondary_opt') or kw.get('opt2', SGD(lr=0.0001, momentum=0.9))

		self.base = self.model
		self.base_weights = self.base.get_weights()

		self.model = None
		self.n_classes = None

	def train(self, train_data, validation_data):
		"""
		Train wrapped model on data

		:param train_data: Training data
		:param validation_data: Validation data
		"""

		self._count_classes(train_data + validation_data)
		self._build_model()
		self._train_model(train_data, validation_data)
		self.model.pop()

	def reset(self):
		"""
		Reset model to pre-training state
		"""

		for layer in self.base.layers:
			layer.trainable = True
		self.base.set_weights(self.base_weights)
		self.model = None

	def _count_classes(self, data):
		self.n_classes = len({s.label for s in data})

	def _build_model(self):
		# Add own top layer(s)
		self.model = Sequential()
		self.model.add(self.base)
		if self.feature_size:
			self.model.add(Dense(
				self.feature_size,
				name='top_fc',
				activation='relu'
			))
		self.model.add(Dense(
			self.n_classes,
			name='top_softmax',
			activation='softmax'
		))

	def _train_model(self, t_data, v_data):
		t_gen = LabeledImageGenerator(
			t_data,
			self.n_classes,
			target_size=self.input_size,
			batch_size=self.batch_size
		)
		v_gen = LabeledImageGenerator(
			v_data,
			self.n_classes,
			target_size=self.input_size,
			batch_size=self.batch_size
		)

		if self.first_unfreeze is not None:
			# Freeze base layers
			for layer in self.base.layers:
				layer.trainable = False
			self._print("Training top layers:")
			self._fit_model(t_gen, v_gen, epochs=self.epochs1, opt=self.opt1, loss='categorical_crossentropy')

			# Unfreeze the last few base layers
			for layer in self.base.layers[self.first_unfreeze:]:
				layer.trainable = True
			self._print("Training unfrozen layers:")
			self._fit_model(t_gen, v_gen, epochs=self.epochs2, opt=self.opt2, loss='categorical_crossentropy')

		else:
			self._print("Training model:")
			self._fit_model(t_gen, v_gen, epochs=self.epochs1, opt=self.opt1, loss='categorical_crossentropy')

	def _fit_model(self, t_gen, v_gen, epochs, opt='SGD', loss='categorical_crossentropy'):
		self.model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
		self.model.fit_generator(
			t_gen,
			epochs=epochs,
			validation_data=v_gen
		)
