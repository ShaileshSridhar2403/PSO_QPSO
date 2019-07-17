import numpy as np 
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import mnist 
from sklearn.metrics import accuracy_score, precision_score, recall_score

class Network_nn(nn.Module):
	def __init__(self,hidden_units,dropout_rate):
		super(Network_nn,self).__init__()
		self.hidden_units=hidden_units
		self.fc1=nn.Linear(784,self.hidden_units)
		self.fc = nn.Linear(self.hidden_units,self.hidden_units)
		self.fc2=nn.Linear(self.hidden_units,10)
		self.dropout=nn.Dropout(p=dropout_rate)
		self.softmax = nn.Softmax(dim=1)

	def forward(self,x):
		x = F.tanh(self.fc1(x))
		x = F.tanh(self.fc(x))
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.softmax(x)
		return x

def getbatch(bsize):
	seed = np.random.choice(X_train.size()[0], size = bsize, replace = False)
	xb = X_train[seed,:]
#	yb = Y_train[seed,:]	#regression
	yb = Y_train[seed]     #classification
	return xb,yb

def train(model,steps,epochs,bsize,X_train,Y_train):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
	for epoch in range(epochs):
		for step in range(steps):
			x,y = getbatch(bsize)
			optimizer.zero_grad()
			outputs = model.forward(x)
		#loss = criterion(outputs,y)
		#loss = torch.sum(torch.sum(torch.pow(torch.sub(outputs,y),2),dim=0)/bsize)/n_outputs
#		loss = torch.sum(torch.sum(torch.pow(torch.sub(outputs,y),2),dim=0)/bsize)#new loss according to paper large values of loss
			loss = criterion(outputs,y)
			loss.backward()
			optimizer.step()
		outputs = model.forward(X_train)
		lossepoch = criterion(outputs,Y_train)
		print('epoch:',epoch+1,'loss:',lossepoch.item())



n_hidden = 100
dropout_rate = 0.25
num_particles = 50

n_inputs= 784
n_outputs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = Network_nn(n_hidden,dropout_rate)
model.to(device)

a = mnist.train_images()
X_train = mnist.train_images().reshape(a.shape[0], (a.shape[1]*a.shape[2]))
Y_train = mnist.train_labels()
a = mnist.test_images() 
X_test = mnist.test_images().reshape(a.shape[0], (a.shape[1]*a.shape[2]))
Y_test = mnist.test_labels()

X_train = torch.from_numpy(X_train).type(torch.FloatTensor).to(device)
Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor).to(device)
Y_train = Y_train.long()
X_test = torch.from_numpy(X_test).type(torch.FloatTensor).to(device)
Y_test = torch.from_numpy(Y_test).type(torch.FloatTensor).to(device)

train(model,200,200,128,X_train,Y_train)

predict_out = model.forward(X_test)
_, predict_y = torch.max(predict_out, 1)
Y_test = Y_test.cpu()
predict = predict_y.cpu()
print ('prediction accuracy', accuracy_score(Y_test, predict))