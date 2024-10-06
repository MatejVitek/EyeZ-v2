import matplotlib.pyplot as plt
import numpy as np
import os
import sys

import torch

from cross_validate import CV
from dataset import Dataset
from dist_models import *
from model_wrapper import *
from naming import NamingParser
from plot import Painter
from utils import get_eyez_dir


CMAP = lambda v: plt.cm.plasma(.9 * v)
LIMITED_DATA = 10, 33, 67, 100
DATA_DIR = os.path.join(get_eyez_dir(), 'Recognition', 'Datasets', 'SBVPI Scleras')
DATA = 'Original'


def main():
	naming = NamingParser(
		r'ie_d_n',
		eyes=r'LR',
		directions=r'lrsu',
		strict=True
	)

	test = Dataset(
		os.path.join(DATA_DIR, DATA),
		naming=naming,
		both_eyes_same_class=False,
		mirrored_offset=0
	)

	res_dir = os.path.join(get_eyez_dir(), 'Recognition', 'Results', 'SBVPI', 'SBVPI Scleras', 'limited_data')
	os.makedirs(res_dir, exist_ok=True)

	font_size = 32
	legend_size = 20

	painter = Painter(
		lim=(0, 1.01),
		xticks=np.linspace(.2, 1, 5),
		yticks=np.linspace(0, 1, 6),
		colors=CMAP,
		k=4,
		labels=("GazeNet 1.0", "SqueezeNet 1.0", "GazeNet 1.1", "SqueezeNet 1.1"),
		font='Times New Roman',
		font_size=font_size,
		legend_size=legend_size,
		pause_on_end=False,
	)
	painter.init()
	for metric in ('VER@0.1FAR', 'VER@1FAR', 'EER', 'AUC'):
		painter.add_figure(
			f'{metric}/Data',
			styles='osP*',
			save=os.path.join(res_dir, f'Sclera-LimitedData-{metric}.pdf'),
			xlabel='Training Data Used', ylabel=metric, legend_loc='upper right' if metric == 'EER' else 'lower right',
			xticks=LIMITED_DATA, x_tick_formatter=lambda x, _: str(x) + "%", xlim=(0, 103), ylim='auto', grid_axis='y',
		)

	try:
		res_str = []
		for percent in LIMITED_DATA:
			for version in ('1.0', '1.1'):
				for gaze in (True, False):
					model = torchhub('squeezenet', gaze=gaze, limited_data=percent if percent != 100 else None, version=version)
					label = f"{'Gaze' if gaze else 'Squeeze'}Net {version}"
					evaluation = CV(model)(
						None,
						test,
						1,
						plot=painter,
						closest_only=True,
						xvalue=percent,
						save=os.path.join(res_dir, 'Distance Matrices', f'{percent}-{label}.pkl'),
						use_precomputed=True,
					)
					res_str.append(f"{label}:\n\n{str(evaluation)}")
					print(f"\n{'-' * 40}\n")
					print(res_str[-1])
					print(f"\n{'-' * 40}\n")

	finally:
		painter.finalize()


# Configs
BATCH_SIZE = 16
DIST = 'cosine'


def base_nn_config(model, *args, **kw):
		return PredictorModel(model, *args, batch_size=BATCH_SIZE, distance=DIST, **kw)


def torchhub(module_name, image_size=(400, 400), gaze=True, limited_data=None, *args, **kw):
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
	model_dir = os.path.join('Limited Data', str(limited_data), model_name) if limited_data else model_name
	print(model.load_state_dict(torch.load(os.path.join(get_eyez_dir(), 'Recognition', 'Models', model_dir, f'{model_name.lower()}{version if version else ""}{"-gaze" if gaze else ""}-best.pkl')), strict=False))
	if torch.cuda.is_available():
		model.cuda()
	model.eval()
	return base_nn_config(model, *args, input_size=image_size, **kw)


if __name__ == '__main__':
	main()
