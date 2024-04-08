
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import flask as Flask
import json

# data transforms
mean =  np.array([0.4664, 0.4891, 0.4104])
std =  np.array([0.1761, 0.1500, 0.1925])

class_index = json.load(open('./class_names.json'))
model_path = 'models/plantvillage.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvNet(nn.Module):
	"""
	This class defines the Convolutional Neural Network architecture used for the PlantVillage dataset.
	"""
	def __init__(self):
		super(ConvNet, self).__init__()
		self.fc_dim = 7744
		self.n_classes = 38
		self.sequential1 = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.sequential2 = nn.Sequential(
			nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.sequential3 = nn.Sequential(
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.sequential4 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		self.sequential5 = nn.Sequential(
			nn.Linear(self.fc_dim, 256),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(128, 64),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(64, self.n_classes),
			# (n + 2p -f)/s + 1
		)
			
	def forward(self, x):
		"""
		Forward pass of the network.

		Parameters
		----------
		x : torch.Tensor
			Input tensor of shape (batch_size, 3, 244, 244)

		Returns
		-------
		torch.Tensor
			Output tensor of shape (batch_size, 38)
		"""
		out = self.sequential1(x)
		out = self.sequential2(out)
		out = self.sequential3(out)
		out = self.sequential4(out)
		out = out.view(out.size(0), -1)
		out = self.sequential5(out)
		return out

plantVillageModel = ConvNet().to(device)
plantVillageModel.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
plantVillageModel.eval()

class ImageClassPredict:
	def __init__(self, model, model_path, image_path):
		self.model = model
		self.model.load_state_dict(torch.load(model_path, map_location=device))
		self.model.eval()
		self.device = torch.device('cpu')
		self.image_transformer = transforms.Compose([
			transforms.CenterCrop(244),
			transforms.ToTensor(),
			transforms.Normalize(mean, std)
		])
		self.model_path = model_path
		self.image = self.load_image(image_path)
		# self.infer()


	def load_image(self, image_path):
		image = Image.open(image_path).convert('RGB')  # Load the image using PIL and convert to RGB
		image = self.image_transformer(image)  # Apply data transformations
		return image


	def infer(self):
		with torch.no_grad():
			output = self.model(self.image.unsqueeze(0))  # Unsqueeze to add batch dimension
		_, prediction = torch.max(output, 1)
		prediction_idx = prediction.item()  # Return the predicted class index
		return self.class_names_map(class_index, prediction_idx)
	
	def class_names_map(self, class_list, prediction):
		return class_list[str(prediction)]



# Image path
img_path = 'data/plantvillage/Squash___Powdery_mildew/01b116bb-2e4a-461a-a772-f7be4a255e31___MD_Powd.M 9996.JPG'

# Initialize the predictor with the model
prediction = ImageClassPredict(ConvNet(), model_path, img_path).infer()

print(prediction)
# Plot the image
im = Image.open(img_path)
plt.figure(figsize=(4, 4))
plt.imshow(im)
plt.title(f'Class: {prediction}')
plt.axis('off')
plt.show()