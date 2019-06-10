import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import sys
import time

class Network(nn.Module):
	def __init__(self,hidden_units,dropout_rate):
		super(Network,self).__init__()
		self.hidden_units=hidden_units
		self.fc1=nn.Linear(943,self.hidden_units)
		self.fc2=nn.Linear(self.hidden_units,4760)
		self.dropout=nn.Dropout(p=dropout_rate)

	def forward(self,x):
		x = F.tanh(self.fc1(x))
		x = self.dropout(x)
		x = self.fc2(x)
		return x

def getbatch(bsize):
	seed = np.random.choice(X_train.size()[0], size = bsize, replace = False)
	xb = X_train[seed,:]
	yb = Y_train[seed,:]
	return xb,yb

def train(model,steps,verbose=True,bsize=64,epochs=100):
	lossepoch = criterion(model.forward(X_train),Y_train)
	print(lossepoch.item())
	for epoch in range(epochs):
		for step in range(steps):
			x,y = getbatch(bsize)
			optimizer.zero_grad()
			outputs = model.forward(x)
			loss = criterion(outputs,y)
			loss.backward()
			optimizer.step()
		lossepoch = criterion(model.forward(X_train),Y_train)
		if verbose: print('epoch-{} training loss:{}'.format(epoch+1,lossepoch.item()))


n_hidden = int(sys.argv[1])
dropout_rate = float(sys.argv[2])
X_train = np.load('GTEx_X_float64.npy')
Y_train = np.load('GTEx_Y_0-4760_float64.npy')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
X_train = torch.from_numpy(X_train).type(torch.FloatTensor).to(device)
Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor).to(device)
model = Network(n_hidden,dropout_rate)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.MSELoss()
train(model=model,steps=68)

