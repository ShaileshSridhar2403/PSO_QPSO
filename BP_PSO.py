#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:38:07 2019

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
#from pso_vec_grad_generic import PSO_grad

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
	
	def vecToMat(self,params):
		#params = np.array(params)
		params = torch.from_numpy(params).type(torch.FloatTensor).to(device)
		self.fc1.weight.data=params[0:n_inputs*n_hidden].view(n_hidden,n_inputs)
		self.fc1.bias.data=params[n_inputs*n_hidden:n_inputs*n_hidden+n_hidden].view(n_hidden)
		self.fc2.weight.data=params[n_inputs*n_hidden+n_hidden:n_inputs*n_hidden+n_hidden+n_hidden*n_outputs].view(n_outputs,n_hidden)
		self.fc2.bias.data=params[n_inputs*n_hidden+n_hidden+n_hidden*n_outputs:n_inputs*n_hidden+n_hidden+n_hidden*n_outputs + n_outputs].view(n_outputs)


class Network_PSO(nn.Module):
	def __init__(self,input_units,hidden_units,output_units,dropout_rate):
		super(Network_PSO,self).__init__()
		self.hidden_units=hidden_units
		self.input_units = input_units
		self.output_units = output_units
		self.fc1=nn.Linear(self.input_units,self.hidden_units)
		self.fc2=nn.Linear(self.hidden_units,self.output_units)
		self.dropout=nn.Dropout(p=dropout_rate)
		self.softmax = nn.Softmax(dim=1)

	def forward(self,params,localbcosts):
		errg_best = localbcosts[np.argmin(localbcosts)]
		pos_best_g = np.zeros(params.shape[2])
		criterion = nn.CrossEntropyLoss()
		
		for i in range(params.shape[0]):
			self.vecToMat(params[i][0])
			x = F.tanh(self.fc1(X_train))			#x = torch.tanh(self.fc1(X_train))
			x = self.dropout(x)
			x = self.softmax(self.fc2(x))
			loss = criterion(x,Y_train)
			if loss<localbcosts[i] or localbcosts[i]==-1 :
				localbcosts[i] = loss
				params[i][1] = np.copy(params[i][0])
			if loss<errg_best or errg_best==-1:
				errg_best = loss
				pos_best_g = np.copy(params[i][0])

#			print(MAE_GTx.item())
		return pos_best_g,errg_best,params
	
	def vecToMat(self,params):
		#params = np.array(params)
		params = torch.from_numpy(params).type(torch.FloatTensor).to(device)
		self.fc1.weight.data=params[0:n_inputs*n_hidden].view(n_hidden,n_inputs)
		self.fc1.bias.data=params[n_inputs*n_hidden:n_inputs*n_hidden+n_hidden].view(n_hidden)
		self.fc2.weight.data=params[n_inputs*n_hidden+n_hidden:n_inputs*n_hidden+n_hidden+n_hidden*n_outputs].view(n_outputs,n_hidden)
		self.fc2.bias.data=params[n_inputs*n_hidden+n_hidden+n_hidden*n_outputs:n_inputs*n_hidden+n_hidden+n_hidden*n_outputs + n_outputs].view(n_outputs)
		
		
def setup(n_inputs,n_hidden,n_outputs,dropout_rate):
	model = Network_nn(n_hidden,dropout_rate)
	model.to(device)
	optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.1)
	return model,optimizer

def getbatch(bsize):
	seed = np.random.choice(X_train.size()[0], size = bsize, replace = False)
	xb = X_train[seed,:]
#	yb = Y_train[seed,:]	#regression
	yb = Y_train[seed]     #classification
	return xb,yb

def matToVec(model):
	fc1_weight = np.ravel(model.fc1.weight.data.detach().cpu().numpy())
	fc1_bias = np.ravel(model.fc1.bias.data.detach().cpu().numpy())
	fc2_weight = np.ravel(model.fc2.weight.data.detach().cpu().numpy())
	fc2_bias = np.ravel(model.fc2.bias.data.detach().cpu().numpy())
	params = np.concatenate((fc1_weight,fc1_bias,fc2_weight,fc2_bias))
	return params

def backPropGrad(model,optimizer,pos_g_best,steps,X_train,Y_train,bsize=64):
	criterion = nn.CrossEntropyLoss()
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
		
	grad = matToVec(model) - pos_g_best
	return grad


def PSO_grad(swarm,localbcosts,w,c1,c2,c3,maxiter,costfunc,verbose,logfile):
	model,optimizer = setup(n_inputs,n_hidden,n_outputs,dropout_rate=0)
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
		gradVal = backPropGrad(model,optimizer,pos_best_g,68,X_train,Y_train)
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


data=load_iris()

X=data['data']
Y=data['target']

#oneHot = [np.zeros(3) for i in Y]
##for i in range(len(Y)):
##	oneHot[i][Y[i]] = 1
#Y = np.array(oneHot)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

#n_hidden = int(sys.argv[1])
#dropout_rate = float(sys.argv[2])
#num_particles = int(sys.argv[3])

n_hidden = 10
dropout_rate = 0
num_particles = 1

n_inputs= 4
n_outputs = 3

dimensions = (n_inputs*n_hidden)+(n_hidden*n_outputs)+n_hidden+n_outputs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = Network_PSO(n_inputs,n_hidden,n_outputs,dropout_rate)
model.to(device)

X_train = torch.from_numpy(X_train).type(torch.FloatTensor).to(device)
Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor).to(device)
Y_train = Y_train.long()
X_test = torch.from_numpy(X_test).type(torch.FloatTensor).to(device)
Y_test = torch.from_numpy(Y_test).type(torch.FloatTensor).to(device)

initialcosts =  np.ones(num_particles)*-1
initial = np.random.randn(num_particles,3,dimensions)

logfilePath = 'iris_data.txt'
pos,err,iters= PSO_grad(costfunc=model.forward,swarm=initial,maxiter=1,verbose=True,w = 1,c1=0,c2=0,c3=1,localbcosts=initialcosts,logfile =logfilePath)

model = Network_nn(n_hidden,dropout_rate)

model.vecToMat(pos)
predict_out = model.forward(X_test)
_, predict_y = torch.max(predict_out, 1)
Y_test = Y_test.cpu()
predict = predict_y.cpu()
print ('prediction accuracy', accuracy_score(Y_test, predict))

np.save('results.npy',pos)
