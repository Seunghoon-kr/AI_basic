import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import os
from PIL import Image
import numpy as np
import pandas as pd
#from sklearn import datasets, model_selection
from ants_bees_net import Net

model = Net()

model.load_state_dict(torch.load('./ants_bees_300.pt'))
model.eval()

path = './bee_1.jpg'
img = Image.open(path, 'r')
resize_img = img.resize((128,128))
r,b,g = resize_img.split()
r_resize_img = np.asarray(np.float32(r)/255.0)
g_resize_img = np.asarray(np.float32(g)/255.0)
b_resize_img = np.asarray(np.float32(b)/255.0)
rgb_resize_img = np.asarray([r_resize_img, g_resize_img, b_resize_img])
rgb_resize_img = np.expand_dims(rgb_resize_img, axis=0)

test = torch.from_numpy(rgb_resize_img).float()

outputs = model(test)

_, predicted = torch.max(outputs, 1)

label = ['ants', 'bees']

print(label[predicted[0]])
