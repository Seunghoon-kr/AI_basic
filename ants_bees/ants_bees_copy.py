
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
from ants_bees_split import train_test_split
from ants_bees_net import Net

dirs = ['ants', 'bees']

data = []
label = []

for i, d in enumerate(dirs):

	files = os.listdir('./hymenoptera_data/' + d)

	for f in files:
		path = './hymenoptera_data/' + d + '/' + f
		img = Image.open(path, 'r')
		
		resize_img = img.resize((128,128))
		r,b,g = resize_img.split()
		r_resize_img = np.asarray(np.float32(r)/255.0)
		g_resize_img = np.asarray(np.float32(g)/255.0)
		b_resize_img = np.asarray(np.float32(b)/255.0)
		rgb_resize_img = np.asarray([r_resize_img, g_resize_img, b_resize_img])
		data.append(rgb_resize_img)

		label.append(i)


pd.DataFrame(data[0][0])

label

data = np.array(data, dtype='float32')
label = np.array(label, dtype='int64')

train_X, test_X, train_Y, test_Y = train_test_split(data, label, test_size=0.1)
#train_size = int(0.9 * len(data))
#test_size = len(data) - train_size
#train_X, test_X = random_split(data, [train_size, test_size])
#train_Y = [label[i] for i in train_X.indices]
#test_Y = [label[i] for i in test_X.indices]

print(len(train_X))
print(len(test_X))

train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()

print(train_X.shape)
print(train_Y.shape)

train = TensorDataset(train_X, train_Y)

print(train[0])

train_loader = DataLoader(train, batch_size=32, shuffle=True)


model = Net()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.002)

for epoch in range(200):
	print('epoch: ', epoch)
	total_loss = 0
	for train_x, train_y in train_loader:
		train_x, train_y = Variable(train_x), Variable(train_y)
		optimizer.zero_grad()
		output = model(train_x)
		loss = criterion(output, train_y)
		loss.backward()
		optimizer.step()

		total_loss += loss.item()

	test_X = np.array(test_X, dtype='float32')
	test_Y = np.array(test_Y, dtype='int64')

	test_X = torch.from_numpy(test_X).float()
	test_Y = torch.from_numpy(test_Y).long()

	test_x, test_y = Variable(test_X), Variable(test_Y)
	result = torch.max(model(test_x).data, 1)[1]
	accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())

	print(accuracy)

	if (epoch + 1) % 50 ==0:
		print(epoch+1, total_loss)

print(test_X.shape)
print(test_Y.shape)

torch.save(model.state_dict(), './ants_bees_300.pt')
