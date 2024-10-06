from matej.nn.torch import Parallel
import torch


def build_model(n_classes, gaze=False):
	model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
	if not n_classes:
		model.classifier = torch.nn.Identity()
		return model

	if n_classes == model.classifier[-1].out_features:
		head = model.classifier[-1]
	else:
		head = torch.nn.Linear(model.last_channel, n_classes)
	if gaze:
		head = Parallel(head, torch.nn.Linear(model.last_channel, 4))
	model.classifier = torch.nn.Sequential(*list(model.classifier)[:-1], head)
	return model
