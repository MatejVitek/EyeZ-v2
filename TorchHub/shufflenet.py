from matej.nn.torch import Parallel
import torch


def build_model(n_classes, gaze=False):
	model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
	if not n_classes:
		model.fc = torch.nn.Identity()
		return model

	if n_classes != model.fc.out_features:
		model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
	if gaze:
		model.fc = Parallel(model.fc, torch.nn.Linear(model.fc.in_features, 4))
	return model
