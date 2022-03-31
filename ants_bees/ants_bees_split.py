import numpy as np

def train_test_split(X,Y, test_size=0.1, shuffle=True, random_state=1004):

	test_num = int(X.shape[0] * test_size)
	train_num = X.shape[0] - test_num

	if shuffle:
		#np.random.seed(random_state)
		shuffled = np.random.permutation(X.shape[0])
		X = X[shuffled,:]
		Y = Y[shuffled]

	X_train = X[:train_num]
	X_test = X[train_num:]
	Y_train = Y[:train_num]
	Y_test = Y[train_num:]

	return X_train, X_test, Y_train, Y_test
