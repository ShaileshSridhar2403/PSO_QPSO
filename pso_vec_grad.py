#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 20:21:04 2019

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

def getbatch(bsize):
	seed = np.random.choice(X_train.size()[0], size = bsize, replace = False)
	xb = X_train[seed,:]
	yb = Y_train[seed,:]
	return xb,yb

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
	def vecToMat(self,params):
		#params = np.array(params)
		print('paramLen',len(params))
		params = torch.from_numpy(params).type(torch.FloatTensor).to(device)
		self.fc1.weight.data=params[0:n_inputs*n_hidden].view(n_hidden,n_inputs)
		self.fc1.bias.data=params[n_inputs*n_hidden:n_inputs*n_hidden+n_hidden].view(n_hidden)
		self.fc2.weight.data=params[n_inputs*n_hidden+n_hidden:n_inputs*n_hidden+n_hidden+n_hidden*n_outputs].view(n_outputs,n_hidden)
		self.fc2.bias.data=params[n_inputs*n_hidden+n_hidden+n_hidden*n_outputs:].view(n_outputs)
		
def matToVec(model):
	fc1_weight = np.ravel(model.fc1.weight.data.detach().cpu().numpy())
	fc1_bias = np.ravel(model.fc1.bias.data.detach().cpu().numpy())
	fc2_weight = np.ravel(model.fc2.weight.data.detach().cpu().numpy())
	fc2_bias = np.ravel(model.fc2.bias.data.detach().cpu().numpy())
	params = np.concatenate((fc1_weight,fc1_bias,fc2_weight,fc2_bias))
	return params
	

###################setup(try to get rid of this/ make more generic)##############
n_inputs = 943
n_hidden = 10
n_outputs = 4760
dropout_rate = 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X = np.load('GTEx_X_float64.npy')
Y = np.load('GTEx_Y_0-4760_float64.npy')
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train = torch.from_numpy(X_train).type(torch.FloatTensor).to(device)
Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor).to(device)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor).to(device)
Y_test = torch.from_numpy(Y_test).type(torch.FloatTensor).to(device)
model = Network(n_hidden,dropout_rate)
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
#############setup over#################
def train(model,steps,verbose=True,bsize=64,epochs=1):
	for epoch in range(epochs):
		for step in range(steps):
			x,y = getbatch(bsize)
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
#		if verbose: print('epoch-{} training loss:{} MAE:{}'.format(epoch+1,lossepoch.item(),MAE.item()))
		
		
def findGrad(model,pos_g_best,bsize = 64):
	x,y = getbatch(bsize)
	model.vecToMat(pos_g_best)
	optimizer.zero_grad()
	outputs = model.forward(x)
	#loss = criterion(outputs,y)
	#loss = torch.sum(torch.sum(torch.pow(torch.sub(outputs,y),2),dim=0)/bsize)/n_outputs
	loss = torch.sum(torch.sum(torch.pow(torch.sub(outputs,y),2),dim=0)/bsize)#new loss according to paper large values of loss
	loss.backward()
	optimizer.step()
	
	grad = matToVec(model) - pos_g_best
	return grad

def backPropGrad(model,pos_g_best,steps,bsize=64):
	for step in range(steps):
		x,y = getbatch(bsize)
		optimizer.zero_grad()
		outputs = model.forward(x)
		#loss = criterion(outputs,y)
		#loss = torch.sum(torch.sum(torch.pow(torch.sub(outputs,y),2),dim=0)/bsize)/n_outputs
		loss = torch.sum(torch.sum(torch.pow(torch.sub(outputs,y),2),dim=0)/bsize)#new loss according to paper large values of loss
		loss.backward()
		optimizer.step()
		
	grad = matToVec(model) - pos_g_best
	return grad


def PSO_grad(swarm,localbcosts,w,c1,c2,c3,maxiter,costfunc,verbose,logfile):
	f = open(logfile,'a')
	f.truncate(0)	
	alpha = 1#alpha = 0.0005			
	print('swarm shape',swarm.shape)					
	num_dim = swarm.shape[2]
	num_partcles = swarm.shape[0]
	pos_best_g= np.zeros(num_dim)
	pos_mbest = np.zeros(num_dim)
	print("input size:  ",pos_mbest.size)
	err_best_g= - 1
	i = 0;
	t_init = time.time()
	sat_count = 0
	err_best_g_prev = 0
	t_total_old = time.time()
	while (i<maxiter):
		if sat_count >5000:
			print('saturated')
			return pos_best_g,err_best_g,i
		t_old = time.time()
		pos_best_g,err_best_g,swarm = costfunc(swarm,localbcosts)
		diff = round(float(err_best_g_prev),6)-round(float(err_best_g),6)
		if int(diff)==0:
			sat_count+=1
		else:
			sat_count = 0
		r1 = np.random.random_sample((num_partcles,1))
		r2 = np.random.random_sample((num_partcles,1))
#		vel_cognitive = c1*r1*(swarm[:,1]-swarm[:,0])
#		vel_social = c2*r2*(pos_best_g-swarm[:,0])
#		gradVal = findGrad(model,pos_best_g)
		gradVal = backPropGrad(model,pos_best_g,5)
		swarm[:,2]*=w
		swarm[:,2] += c1*r1*(swarm[:,1]-swarm[:,0]) + c2*r2*(pos_best_g-swarm[:,0]) + c3*gradVal#new velocity
		
		swarm[:,0] += alpha*swarm[:,2]					#new position
		
		i+=1
		err_best_g_prev = err_best_g
		t_new = time.time()
		if verbose: print('iter: {}, best solution: {} time elapsed in secs:{} Tot: {}'.format(i,err_best_g,float(t_new-t_old),float(t_new-t_init)))
		f.write(str(float(err_best_g)) + '\n')
	
	print('\nFINAL SOLUTION:')
	#print('   > {}'.format(self.pos_best_g))
	print('   > {}\n'.format(err_best_g))
	t_total_new = time.time()
	print('total time elapsed:{}secs'.format(t_total_new-t_total_old))
	return pos_best_g,err_best_g,maxiter
		
		
		