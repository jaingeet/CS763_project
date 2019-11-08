import pickle
import torch
from torch.autograd import Variable
import numpy as np
from tensorboardX import SummaryWriter
import math
from sklearn.preprocessing import normalize

model_name = './model_basic.pth'

train_X = pickle.load(open('./data/adult/X_train.pkl', 'rb'))
train_Y = pickle.load(open('./data/adult/Y_train.pkl', 'rb'))

test_X = pickle.load(open('./data/adult/X_test.pkl', 'rb'))
test_Y = pickle.load(open('./data/adult/y_test.pkl', 'rb'))

X_train = list(train_X.values)
Y_train = list(train_Y.values)

X_test = list(test_X.values)
Y_test = list(test_Y.values)

N, D_in, H, H_adversary, D_out, D_out_adversary = len(X_train), len(X_train[0]), 264, 100, 1, 1

X_train = Variable(torch.Tensor(X_train))
Y_train = Variable(torch.Tensor(np.expand_dims(np.asarray(Y_train), 1)))
X_test, Y_test = Variable(torch.Tensor(X_test)), Variable(torch.Tensor(np.expand_dims(np.asarray(Y_test), 1)))

keep_prob = .5
learning_rate = 1e-4
num_epochs = 20000

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Dropout(keep_prob),
    torch.nn.Linear(H, H),
    torch.nn.ReLU(),
    torch.nn.Dropout(keep_prob),
    torch.nn.Linear(H, D_out),
    torch.nn.Sigmoid(),
)

# loss_fn = torch.nn.MSELoss(size_average=True)
loss_fn = torch.nn.BCELoss(size_average=True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(num_epochs):
	y_pred = model(X_train)
	
	loss = loss_fn(y_pred, Y_train)

	if t % 100 == 0:
		print(t)
		torch.save(model, model_name)

	# Backprop loss
	loss.backward()
	# Step generator
	optimizer.step()
	# Zero gradients for generator and adversary
	optimizer.zero_grad()

model.eval()

y_pred = model(X_test)
loss = loss_fn(y_pred, Y_test)
print("Loss", loss)