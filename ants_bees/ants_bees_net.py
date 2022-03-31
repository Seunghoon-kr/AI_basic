import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import os
from PIL import Image
#from sklearn import datasets, model_selection
from ants_bees_split import train_test_split
import numpy as np
import pandas as pd


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 10, 5)
		self.conv2 = nn.Conv2d(10, 20, 5)

		self.fc1 = nn.Linear(20 * 29 * 29, 50) #29 = (((((128-5)+1)/2)-5)+1)/2
		self.fc2 = nn.Linear(50,2)

	def forward(self, x):
		x = F.max_pool2d(F.relu(self.conv1(x)), 2)
		x = F.max_pool2d(F.relu(self.conv2(x)),2)
		x = x.view(-1, 20 * 29 * 29)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x)
