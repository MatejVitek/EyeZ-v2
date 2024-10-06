from matej.nn.torch import Parallel
import torch


def build_model(n_classes, gaze=False, version='1.0'):
	model = torch.hub.load('pytorch/vision:v0.10.0', f'squeezenet{str(version).replace(".", "_")}', pretrained=True)
	if not n_classes:
		model.classifier = torch.nn.Identity()
		return model

	if n_classes != model.classifier[1].out_channels:
		model.classifier[1] = torch.nn.Conv2d(model.classifier[1].in_channels, n_classes, 1)
	if gaze:
		model.classifier = Parallel(model.classifier, torch.nn.Sequential(
			torch.nn.Dropout(.5),
			torch.nn.Conv2d(model.classifier[1].in_channels, 4, 1),
			torch.nn.ReLU(True),
			torch.nn.AdaptiveAvgPool2d((1, 1)),
		))
		def forward(self, input):
			x = self.features(input)
			x = self.classifier(x)
			return torch.flatten(x[0], 1), torch.flatten(x[1], 1)
		model.forward = forward.__get__(model, torch.nn.Module)
	return model
