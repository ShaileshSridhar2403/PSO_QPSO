#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:43:25 2019

@author: shailesh
"""

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
from sklearn.metrics import accuracy_score, precision_score, recall_score


data=load_iris()

X=data['data']
Y=data['target']

#oneHot = [np.zeros(3) for i in Y]
##for i in range(len(Y)):
##	oneHot[i][Y[i]] = 1
#Y = np.array(oneHot)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

class Network_nn(nn.Module):
	def __init__(self,hidden_units,dropout_rate):
		super(Network_nn,self).__init__()
		self.hidden_units=hidden_units
		self.fc1=nn.Linear(4,self.hidden_units)
		self.fc2=nn.Linear(self.hidden_units,3)
		self.dropout=nn.Dropout(p=dropout_rate)
		self.softmax = nn.Softmax(dim=1)

	def forward(self,x):
		x = F.tanh(self.fc1(x))
		x = self.dropout(x)
		x = self.fc2(x)
		x = self.softmax(x)
		return x

n_hidden = 10
dropout_rate = 0

def getbatch(bsize):
	seed = np.random.choice(X_train.size()[0], size = bsize, replace = False)
	xb = X_train[seed,:]
	yb = Y_train[seed]
	return xb,yb

def matToVec(model):
	fc1_weight = np.ravel(model.fc1.weight.data.detach().cpu().numpy())
	fc1_bias = np.ravel(model.fc1.bias.data.detach().cpu().numpy())
	fc2_weight = np.ravel(model.fc2.weight.data.detach().cpu().numpy())
	fc2_bias = np.ravel(model.fc2.bias.data.detach().cpu().numpy())
	params = np.concatenate((fc1_weight,fc1_bias,fc2_weight,fc2_bias))
	return params

def train(model,steps,verbose=True,bsize=64,epochs=1):
	for epoch in range(epochs):
		print('epoch:',epoch)
		for step in range(steps):
			x,y = getbatch(bsize)
			optimizer.zero_grad()
			outputs = model.forward(x)
			#loss = criterion(outputs,y)
			#loss = torch.sum(torch.sum(torch.pow(torch.sub(outputs,y),2),dim=0)/bsize)/n_outputs
#			loss = torch.sum(torch.sum(torch.pow(torch.sub(outputs,y),2),dim=0)/bsize)#new loss according to paper large values of loss
			loss = criterion(outputs,y)
			loss.backward()
			optimizer.step()
			
			
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = Network_nn(n_hidden,dropout_rate)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.1)
criterion = nn.CrossEntropyLoss()

X_train = torch.from_numpy(X_train).type(torch.FloatTensor).to(device)
Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor).to(device)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor).to(device)
Y_test = torch.from_numpy(Y_test).type(torch.FloatTensor).to(device)

Y_train = Y_train.long()

train(model=model,steps=68)

predict_out = model.forward(X_test)
_, predict_y = torch.max(predict_out, 1)
Y_test = Y_test.cpu()
predict = predict_y.cpu()
print ('prediction accuracy', accuracy_score(Y_test, predict))