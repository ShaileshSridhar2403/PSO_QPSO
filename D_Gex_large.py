import os
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import sys
import time
from sklearn.model_selection import train_test_split
import loadDataset


XFolder = '/home/shailesh/gitHub/PSO_QPSO/datasets/GEO/bgedv2_X_tr_float64'
YFolder = '/home/shailesh/gitHub/PSO_QPSO/datasets/GEO/bgedv2_Y_tr_0-4760_float64'

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

def getbatch(bsize,X_train,Y_train):
	seed = np.random.choice(X_train.size()[0], size = bsize, replace = False)
	xb = X_train[seed,:]
	yb = Y_train[seed,:]
	return xb,yb

def train(model,steps,verbose=True,bsize=64,epochs=100):
	X = loadDataset.loadNextPart(XFolder)
	Y = loadDataset.loadNextPart(YFolder)
	
	while(type(X)!=int):
		Xtrain = torch.from_numpy(X).type(torch.FloatTensor).to(device)
		Ytrain = torch.from_numpy(Y).type(torch.FloatTensor).to(device)
		for epoch in range(epochs):
			for step in range(steps):
#				print('step:',step)
				x,y = getbatch(bsize,Xtrain,Ytrain)
				optimizer.zero_grad()
				outputs = model.forward(x)
				#loss = criterion(outputs,y)
				#loss = torch.sum(torch.sum(torch.pow(torch.sub(outputs,y),2),dim=0)/bsize)/n_outputs
				loss = torch.sum(torch.sum(torch.pow(torch.sub(outputs,y),2),dim=0)/bsize)#new loss according to paper large values of loss
				loss.backward()
				optimizer.step()
				#lossepoch = torch.sum(torch.sum(torch.pow(torch.sub(model.forward(X_train),Y_train),2),dim=0)/2921)/n_outputs
			lossepoch = torch.sum(torch.sum(torch.pow(torch.sub(model.forward(X_test),Y_test),2),dim=0)/2921)#new total loss
			MAE = torch.sum(torch.sum(torch.abs(torch.sub(model.forward(X_test),Y_test)),dim=0)/(0.2*2921))/n_outputs#this should be the MAE
			if verbose: print('epoch-{} training loss:{} MAE:{}'.format(epoch+1,lossepoch.item(),MAE.item()))
		X = loadDataset.loadNextPart(XFolder)
		Y = loadDataset.loadNextPart(YFolder)
			

n_outputs = 4760
n_hidden = int(sys.argv[1])
dropout_rate = float(sys.argv[2])
#X = np.load('/home/shailesh/gitHub/PSO_QPSO/datasets/GEO/bgedv2_X_tr_float64.npy',mmap_mode = 'r')
#Y = np.load('/home/shailesh/gitHub/PSO_QPSO/datasets/GEO/bgedv2_Y_tr_0-4760_float64.npy',mmap_mode = 'r')
##X = np.load('/home/shailesh/gitHub/PSO_QPSO/datasets/GEO/bgedv2_X_te_float64.npy')
#Y = np.load('/home/shailesh/gitHub/PSO_QPSO/datasets/GEO/bgedv2_Y_te_0-4760_float64.npy')
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
X_test = np.load('/home/shailesh/gitHub/PSO_QPSO/datasets/GEO/bgedv2_X_te_float64.npy')
Y_test = np.load('/home/shailesh/gitHub/PSO_QPSO/datasets/GEO/bgedv2_Y_te_0-4760_float64.npy')
#X_train = torch.from_numpy(X_train).type(torch.FloatTensor).to(device)
#Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor).to(device)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor).to(device)
Y_test = torch.from_numpy(Y_test).type(torch.FloatTensor).to(device)


model = Network(n_hidden,dropout_rate)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.00005, momentum=0.4)
#criterion = nn.MSELoss()
train(model=model,steps=68) # why 68

#params = print(list(model.parameters()))
fc1_weight = np.ravel(model.fc1.weight.data.detach().cpu().numpy())
fc1_bias = np.ravel(model.fc1.bias.data.detach().cpu().numpy())
fc2_weight = np.ravel(model.fc2.weight.data.detach().cpu().numpy())
fc2_bias = np.ravel(model.fc2.bias.data.detach().cpu().numpy())

params = np.concatenate((fc1_weight,fc1_bias,fc2_weight,fc2_bias))


np.save("trainedData.npy",params)
print(params,params.shape)