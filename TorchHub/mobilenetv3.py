from matej.nn.torch import Parallel
import torch
import torchvision


def build_model(n_classes, gaze=False):
	model = torchvision.models.mobilenet_v3_large(pretrained=True)
	if not n_classes:
		model.classifier = torch.nn.Identity()
		return model

	if n_classes == model.classifier[-1].out_features:
		head = model.classifier[-1]
	else:
		head = torch.nn.Linear(model.classifier[-1].in_features, n_classes)
	if gaze:
		head = Parallel(head, torch.nn.Linear(model.classifier[-1].in_features, 4))
	model.classifier = torch.nn.Sequential(*list(model.classifier)[:-1], head)
	return model
