import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from keras.applications import ResNet50
from keras.layers import Input
from keras.models import load_model, Model
from keras.optimizers import RMSprop, SGD
import torch

from cross_validate import CV
from dataset import Dataset, L, R, C, U
from dist_models import *
from hog import HOGModel
from model_wrapper import *
from naming import NamingParser
from plot import Painter, exp_format, oom
from utils import get_eyez_dir


# Configuration

config = 'compute'
config = 'plot'


# Configuration settings

# How many folds?
K = 10
# Do we want a plot of our models' performances?
PLOT = False
# Do we want to save the evaluation results to a text file?
SAVE = True
# Do we want to load precomputed distance matrices?
LOAD = True
# Do we want to keep the plot open at the end?
PAUSE = False
# Plot size - whole column-width ('large') or half-column ('small')
SIZE = 'large'
SIZE = 'small'
# Compute times for each model?
TIMES = False
# Colormap to use
CMAP = lambda v: plt.cm.plasma(.9 * v)


if config == 'plot':
	K = 1
	PLOT = True
	SAVE = False
	LOAD = True


# Special protocol settings

# Number of view directions in base template for each identity (or list of specific directions)
BASE_DIRS = 4
#BASE_DIRS = (C,)

# Image size to use in model evaluation (None to use default settings)
IMG_SIZE = None
#IMG_SIZE = (400, 400)

# Should dataset be grouped by an attribute (such as age)? If not, set GROUP_BY to None.
GROUP_BY = None
#GROUP_BY = 'age'
#GROUP_BY = 'gender'
# Bins to group into. Ignored if GROUP_BY is None. For more info, see dataset.Dataset.group_by.
BINS = (25, 40)
#BINS = None
# Are we using intergroup evaluation? Ignored if GROUP_BY is None. See cross_validate.CV.cross_validate_grouped.
INTERGROUP = False

# Ignore this
I = 0


# For easier naming of results folder, I should do DATA_PREFIX = EYEZ/path/to/dataset, DATA_NAME = 'Name I want to appear in the results folder', DATA_SUFFIX = 'anything/else'
# Training and testing datasets. If no training is to be done, set train to None.
# DATA_DIR = os.path.join(get_eyez_dir(), 'Recognition', 'Datasets', 'Rot ScleraNet')
# DATA = {'train': None, 'test': 'stage2'}
DATA_DIR = os.path.join(get_eyez_dir(), 'Recognition', 'Datasets', 'SBVPI Scleras')
DATA = {'train': None, 'test': 'Original'}

def main():
	# Define file naming rules
	naming = NamingParser(
		r'ie_d_n',
		eyes=r'LR',
		directions=r'lrsu',
		strict=True
	)
	both_eyes_same_class = False
	mirrored_offset = 0

	train = None
	if DATA['train']:
		train = Dataset(
			os.path.join(DATA_DIR, DATA['train']),
			naming=naming,
			both_eyes_same_class=both_eyes_same_class,
			mirrored_offset=mirrored_offset
		)
		if GROUP_BY:
			train = train.group_by(GROUP_BY, BINS)
	test = Dataset(
		os.path.join(DATA_DIR, DATA['test']),
		naming=naming,
		both_eyes_same_class=both_eyes_same_class,
		mirrored_offset=mirrored_offset
	)
	if GROUP_BY:
		test = test.group_by(GROUP_BY, BINS)

	if IMG_SIZE:
		models = (
			scleranet(image_size=IMG_SIZE),# scleranet(torch_=True, image_size=IMG_SIZE),
			# tinyvit(gaze=True, image_size=IMG_SIZE), tinyvit(gaze=False, image_size=IMG_SIZE),
			# efficientnet(gaze=True, image_size=IMG_SIZE), efficientnet(gaze=False, image_size=IMG_SIZE),
			# torchhub('mobilenetv2', gaze=True, image_size=IMG_SIZE), torchhub('mobilenetv2', gaze=False, image_size=IMG_SIZE),
			# torchhub('mobilenetv3', gaze=True, image_size=IMG_SIZE), torchhub('mobilenetv3', gaze=False, image_size=IMG_SIZE),
			torchhub('squeezenet', gaze=True, image_size=IMG_SIZE, version='1.0'), torchhub('squeezenet', gaze=True, image_size=IMG_SIZE, version='1.1'),
			torchhub('squeezenet', gaze=False, image_size=IMG_SIZE, version='1.0'), torchhub('squeezenet', gaze=False, image_size=IMG_SIZE, version='1.1'),
			# torchhub('squeezenet', gaze=True, image_size=IMG_SIZE, version='1.0'), torchhub('squeezenet', gaze=False, image_size=IMG_SIZE, version='1.1'),
			# torchhub('shufflenet', gaze=True, image_size=IMG_SIZE), torchhub('shufflenet', gaze=False, image_size=IMG_SIZE),
			# torchhub('regnet', gaze=True, image_size=IMG_SIZE), torchhub('regnet', gaze=False, image_size=IMG_SIZE),
			# torchhub('resnet', gaze=True, image_size=IMG_SIZE), torchhub('resnet', gaze=False, image_size=IMG_SIZE),
			*(descriptor(name, image_size=IMG_SIZE) for name in ('sift', 'surf', 'orb')),
			descriptor('sift', True, image_size=IMG_SIZE),
		)
	else:
		models = (
			scleranet(),# scleranet(torch_=True),
			# tinyvit(gaze=True), tinyvit(gaze=False),
			# efficientnet(gaze=True), efficientnet(gaze=False),
			# torchhub('mobilenetv2', gaze=True), torchhub('mobilenetv2', gaze=False),
			# torchhub('mobilenetv3', gaze=True), torchhub('mobilenetv3', gaze=False),
			torchhub('squeezenet', gaze=True, version='1.0'), torchhub('squeezenet', gaze=True, version='1.1'),
			torchhub('squeezenet', gaze=False, version='1.0'), torchhub('squeezenet', gaze=False, version='1.1'),
			# torchhub('squeezenet', gaze=True, version='1.0'), torchhub('squeezenet', gaze=False, version='1.1'),
			# torchhub('shufflenet', gaze=True), torchhub('shufflenet', gaze=False),
			# torchhub('regnet', gaze=True), torchhub('regnet', gaze=False),
			# torchhub('resnet', gaze=True), torchhub('resnet', gaze=False),
			*(descriptor(name) for name in ('sift', 'surf', 'orb')),
			descriptor('sift', True),
		)
	if PLOT and SIZE == 'small':
		labels = (
			"ScNET",
			"GNet1.0", "GNet1.1",
			"SqNet1.0", "SqNet1.1",
			"SIFT", "SURF", "ORB",
			"dSIFT",
		)
	else:
		labels = (
			"ScleraNET",
			"GazeNet 1.0", "GazeNet 1.1",
			"SqueezeNet 1.0", "SqueezeNet 1.1",
			"SIFT", "SURF", "ORB",
			"Dense SIFT",
		)
	complexities = (
		(7725247872, 4739328),  # ScleraNet
		(2411459104, 735424), (871093952, 722496),  # GazeNet 1.0, 1.1
		(2411459104, 735424), (871093952, 722496),  # SqueezeNet 1.0, 1.1
		(155117952, 0), (116196000, 0), (48864000, 0),  # SIFT, SURF, ORB
		(59392000, 0),  # dSIFT
	)

	# This is for plotting grouped
	if GROUP_BY:
		models = (models[I],)
		labels = (labels[I],)

	res_path = [DATA['test']]
	if GROUP_BY:
		res_path.append(f'{GROUP_BY}{"_intergroup" if INTERGROUP else ""}')
	try:
		if BASE_DIRS < 4:
			res_path.append(f'{BASE_DIRS} direction{"s" if BASE_DIRS > 1 else ""} in base')
	except TypeError:
		dirs = {L: 'left', R: 'right', C: 'center', U: 'up'}
		res_path.append(f'{", ".join(dirs[d] for d in sorted(BASE_DIRS))} in base')
	res_dir = os.path.join(get_eyez_dir(), 'Recognition', 'Results', 'SBVPI', *res_path)
	os.makedirs(res_dir, exist_ok=True)

	group_suffix = '_{group}' if GROUP_BY else ''
	fold_suffix = '_fold{fold}' if K > 1 else ''
	# This is also for plotting grouped but doesn't need to be commented out
	label_suffix = f'-{labels[0]}' if GROUP_BY else ''

	font_size = 20 if SIZE == 'large' else 32
	legend_size = 20 if SIZE == 'large' else 20
	size_suffix = '-large' if SIZE == 'large' else ''

	painter = None
	if PLOT:
		if GROUP_BY and K > 1:
			plt_labels = [f"{key} (k = {k})" for key in test.keys() for k in range(K)]
		elif GROUP_BY:
			plt_labels = list(test)
		elif K > 1:
			plt_labels = [f"k = {k}" for k in range(K)]
		else:
			plt_labels = labels
		painter = Painter(
			lim=(0, 1.01),
			xticks=np.linspace(.2, 1, 5),
			yticks=np.linspace(0, 1, 6),
			colors=CMAP,
			k=(len(test) if GROUP_BY else 1) * K * len(models),
			labels=plt_labels,
			font='Times New Roman',
			font_size=font_size,
			legend_size=legend_size,
			pause_on_end=PAUSE,
		)
		painter.init()
		painter.add_figure('EER', xlabel='Threshold', ylabel='FAR/FRR')
		painter.add_figure(
			'ROC Curve',
			save=os.path.join(res_dir, f'Sclera-ROC{label_suffix}{size_suffix}.pdf'),
			xlabel='FAR', ylabel='VER', legend_loc='lower right',
		)
		painter.add_figure(
			'Semilog ROC Curve',
			save=os.path.join(res_dir, f'Sclera-ROC-log{label_suffix}{size_suffix}.pdf'),
			xlabel='FAR', ylabel='VER', legend_loc='lower right',
			xscale='log', xlim=(1e-3, 1.01),
			xticks=(1e-3, 1e-2, 1e-1, 1), x_tick_formatter=exp_format,
		)
		xmin = oom(min(c[0] for c in complexities))
		xmax = 10 * oom(max(c[0] for c in complexities))
		xticks_min = int(math.log10(xmin))
		xticks_max = int(math.log10(xmax))
		for metric in ('VER@0.1FAR', 'VER@1FAR', 'EER', 'AUC'):
			painter.add_figure(
				f'{metric}/Complexity',
				styles='osP*Xv^<>p1234',
				save=os.path.join(res_dir, f'Sclera-FLOPs-{metric}{label_suffix}{size_suffix}.pdf'),
				xlabel='FLOPs', ylabel=metric, separate_legend=True, legend_col=5, #legend_loc='upper right' if metric == 'EER' else 'lower right',
				xscale='log', xlim=(xmin, xmax), ylim='auto',
				xticks=np.logspace(xticks_min, xticks_max, xticks_max - xticks_min + 1), x_tick_formatter=exp_format,
			)

	try:
		res_str = []
		for model, label, complexity in zip(models, labels, complexities):
			evaluation = CV(model)(
				train,
				test,
				K,
				base_split_n=BASE_DIRS,
				plot=painter,
				complexity=complexity,
				closest_only=True,
				intergroup_evaluation=INTERGROUP,
				save=os.path.join(res_dir, 'Distance Matrices', f'{label}{group_suffix}{fold_suffix}.pkl'),
				times=TIMES,
				use_precomputed=LOAD
			)
			if GROUP_BY:
				res_str.append(f"{label}:\n\n" + "\n\n".join(f"{k}:\n{str(v)}" for k, v in evaluation.items()))
			else:
				res_str.append(f"{label}:\n\n{str(evaluation)}")
			print(f"\n{'-' * 40}\n")
			print(res_str[-1])
			print(f"\n{'-' * 40}\n")
			if TIMES and os.path.exists('times.tmp'):
				with open('times.tmp', 'r') as f:
					feature_times = []
					dist_times = []
					for _ in range(K):
						feature_times.append(float(f.readline().strip()))
						dist_times.append(float(f.readline().strip()))
				os.unlink('times.tmp')
				feature_times = np.array(feature_times)
				dist_times = np.array(dist_times)
				with open('times.txt', 'a') as f:
					print(
						label,
						fr"{1000 * feature_times.mean()} \pm {1000 * feature_times.std()}".replace("e", r" \cdot 10^{"),
						fr"{1000 * dist_times.mean()} \pm {1000 * dist_times.std()}".replace("e", r" \cdot 10^{"),
						fr"{1000 * (feature_times.mean() + dist_times.mean())} \pm {1000 * math.sqrt(feature_times.std() ** 2 + dist_times.std() ** 2)}".replace("e", r" \cdot 10^{"),
						sep = " & ",
						end = " \\\\\n",
						file=f
					)
		if SAVE:
			with open(os.path.join(res_dir, f'Evaluation.txt'), 'w', encoding='utf-8') as res_file:
				res_file.write("\n\n\n\n".join(res_str))
				res_file.write("\n")

	finally:
		if PLOT:
			painter.finalize()
		if os.path.exists('times.tmp'):
			os.unlink('times.tmp')


# Configs
BATCH_SIZE = 8
DIST = 'cosine'


def base_nn_config(model, train=False, *args, feature_size=None, first_unfreeze=None, **kw):
	if not train:
		return PredictorModel(model, *args, batch_size=BATCH_SIZE, distance=DIST, **kw)
	return TrainablePredictorModel(
		model,
		*args,
		primary_epochs=100,
		secondary_epochs=50,
		feature_size=feature_size,
		first_unfreeze=first_unfreeze,
		primary_opt=RMSprop(lr=1e-4),
		secondary_opt=SGD(lr=1e-5, momentum=0.5, nesterov=True),
		batch_size=BATCH_SIZE,
		distance=DIST,
		**kw
	)


def resnet50(*args, **kw):
	return base_nn_config(ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg'), *args, first_unfreeze=143, **kw)


def scleranet(layer='final_features', image_size=None, *args, torch_=False, **kw):
	if not torch_:
		model = load_model(os.path.join(get_eyez_dir(), 'Recognition', 'Models', 'Rot ScleraNet', 'id_dir_prediction.75-0.667.hdf5'))
		if not image_size or image_size == (400, 400):
			return base_nn_config(Model(model.input, model.get_layer(layer).output), *args, **kw)
		model.layers.pop(0)
		model = Model(model.input, model.get_layer(layer).output)
		input_ = Input(shape=(*image_size, 3))
		model = Model(input_, model(input_))
	else:
		olddir = os.getcwd()
		os.chdir(os.path.join(os.path.dirname(__file__), '..', 'EyeZ', 'external', 'Lightweight Recognition', 'ScleraNET'))
		sys.path.append('.')
		from model import ScleraNET
		model = ScleraNET(0, gaze=True)
		os.chdir(olddir)
		sys.path.pop()
		print(model.load_state_dict(torch.load(os.path.join(get_eyez_dir(), 'Recognition', 'Models', 'ScleraNET', f'scleranet-gaze-best.pkl')), strict=False))
		if torch.cuda.is_available():
			model.cuda()
		model.eval()
	return base_nn_config(model, *args, input_size=image_size if image_size else (400, 400), **kw)


def tinyvit(image_size=None, gaze=True, *args, **kw):
	olddir = os.getcwd()
	os.chdir(os.path.join(os.path.dirname(__file__), '..', 'EyeZ', 'external', 'TinyViT'))
	sys.path.append('.')
	from argparse import Namespace
	from config import get_config
	from models import build_model
	if not image_size or image_size == (400, 400):
		resolution = 512
		cfg = '384-512'
	else:
		resolution = image_size[0]
		cfg = f'224-{resolution}'
	model_type = 'sclera_vit' if gaze else 'tiny_vit'
	cfg_dir = model_type.replace('_', '')
	model = build_model(get_config(Namespace(cfg=f'configs/matej/{cfg_dir}/{cfg}.yaml', opts=['MODEL.NUM_CLASSES', 0, 'MODEL.TYPE', model_type], batch_size=None, data_path=None, pretrained=None, resume=None, teacher_logits=None, accumulation_steps=None, use_checkpoint=None, disable_amp=None, only_cpu=None, output=None, tag=None, eval=None, throughput=None, local_rank=None)))
	os.chdir(olddir)
	sys.path.pop()
	model_type = 'ScleraViT' if gaze else 'TinyViT'
	saved = torch.load(os.path.join(get_eyez_dir(), 'Recognition', 'Models', 'TinyViT', f'{model_type}-{resolution}', 'default', 'ckpt_best.pth'))
	print(model.load_state_dict(saved['model'], strict=False))
	if torch.cuda.is_available():
		model = model.cuda()
	model.eval()
	return base_nn_config(model, *args, input_size=(resolution, resolution), **kw)


def efficientnet(image_size=(400, 400), gaze=True, *args, **kw):
	olddir = os.getcwd()
	os.chdir(os.path.join(os.path.dirname(__file__), '..', 'EyeZ', 'external', 'EfficientNet'))
	sys.path.append('.')
	model_name = 'sclera' if gaze else 'efficientnet'
	model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=False)
	os.chdir(olddir)
	sys.path.pop()
	last_block = list(model.children())[-1]
	# Hack out the ID head and possibly the gaze head (and the dropout since it's not used in eval anyway)
	def forward(self, input):
		input = self.pooling(input)
		input = self.squeeze(input)
		return input
	last_block.fc = torch.nn.Identity()  # Set them to identity so the model loads without errors
	if gaze:
		last_block.gaze = torch.nn.Identity()  # Set them to identity so the model loads without errors
	last_block.forward = forward.__get__(last_block, torch.nn.Sequential)
	model = torch.nn.Sequential(*list(model.children())[:-1], last_block)
	print(model.load_state_dict(torch.load(os.path.join(get_eyez_dir(), 'Recognition', 'Models', 'EfficientNet', f'{model_name}-best.pkl')), strict=False))
	if torch.cuda.is_available():
		model.cuda()
	model.eval()
	return base_nn_config(model, *args, input_size=image_size, **kw)


def torchhub(module_name, image_size=(400, 400), gaze=True, *args, **kw):
	version = kw.get('version') or kw.get('v') or kw.get('ver') or kw.get('V', None)
	olddir = os.getcwd()
	os.chdir(os.path.join(os.path.dirname(__file__), '..', 'EyeZ', 'external', 'TorchHub'))
	sys.path.append('.')
	import importlib
	module = importlib.import_module(module_name)
	from inspect import signature, Parameter
	model_params = signature(module.build_model).parameters
	model_kw = {arg: value for arg, value in kw.items() if arg in model_params and model_params[arg].kind not in (Parameter.POSITIONAL_ONLY, Parameter.VAR_POSITIONAL)}
	for arg in model_kw:
		del kw[arg]
	model = module.build_model(0, gaze=gaze, **model_kw)
	os.chdir(olddir)
	sys.path.pop()
	model_name = type(model).__name__
	print(model.load_state_dict(torch.load(os.path.join(get_eyez_dir(), 'Recognition', 'Models', model_name, f'{model_name.lower()}{version if version else ""}{"-gaze" if gaze else ""}-best.pkl')), strict=False))
	if torch.cuda.is_available():
		model.cuda()
	model.eval()
	return base_nn_config(model, *args, input_size=image_size, **kw)


def descriptor(*args, **kw):
	return DirectDistanceModel(DescriptorModel(*args, **kw))


def hog(*args, **kw):
	return PredictorModel(HOGModel(*args, **kw), batch_size=BATCH_SIZE, distance=DIST)


# Too slow and doesn't work
def correlation():
	return DirectDistanceModel(CorrelationModel())


if __name__ == '__main__':
	# # Different segmentation results (input datasets)
	# old_dd, old_d, old_s = DATA_DIR, DATA, SIZE
	# SIZE = 'large'
	# DATA = {'train': None, 'test': 'ground_truth'}
	# main()
	# DATA_DIR = os.path.join(get_eyez_dir(), 'Segmentation', 'Vessels', 'Results', 'SBVPI')
	# for data_dir in ('Miura_MC_norm', 'Miura_RLT_norm', 'Coye', 'B-COSFIRE'):
	# 	print("Vessel segmentation experiments:", data_dir)
	# 	DATA = {'train': None, 'test': data_dir}
	# 	main()
	# DATA_DIR = os.path.join(get_eyez_dir(), 'Recognition', 'Datasets', 'SBVPI Scleras)
	# DATA = {'train': None, 'test': 'Original'}
	# main()
	# DATA_DIR, DATA, SIZE = old_dd, old_d, old_s

	# # Different input resolutions
	# old_dd, old_d, old_is, old_s = DATA_DIR, DATA, IMG_SIZE, SIZE
	# SIZE = 'small'
	# DATA_DIR = os.path.join(get_eyez_dir(), 'Recognition', 'Datasets', 'SBVPI Scleras', 'Resized')
	# for resolution in (256, 192, 128, 96, 64):
	# 	print("Resolution experiments:", resolution)
	# 	DATA = {'train': None, 'test': str(resolution)}
	# 	IMG_SIZE = (resolution, resolution)
	# 	main()
	# DATA_DIR, DATA, IMG_SIZE, SIZE = old_dd, old_d, old_is, old_s

	# # Different grouping protocols
	# old_gb, old_b, old_ig, old_i, old_s = GROUP_BY, BINS, INTERGROUP, I, SIZE
	# SIZE = 'small'
	# for GROUP_BY, BINS in (('age', (25, 40)), ('gender', None)):
	# 	for INTERGROUP in (False, True):
	# 		for I in range(7):  # Need to change the range to the number of models
	# 			print("Attribute experiments:", GROUP_BY, INTERGROUP, I)
	# 			main()
	# GROUP_BY, BINS, INTERGROUP, I, SIZE = old_gb, old_b, old_ig, old_i, old_s

	# # Different base sizes
	# old_bd, old_s = BASE_DIRS, SIZE
	# SIZE = 'small'
	# for BASE_DIRS in range(1, 4):
	# 	print("Gallery template experiments:", BASE_DIRS, "directions in base")
	# 	main()
	# print("Gallery template experiments: Center in base")
	# BASE_DIRS = (C,)
	# main()
	# BASE_DIRS, SIZE = old_bd, old_s

	# # For the single run we need both plot sizes
	# for SIZE in (('large', 'small') if config == 'plot' else [SIZE]):
		print(f"Single run with {SIZE} plots")
		main()
