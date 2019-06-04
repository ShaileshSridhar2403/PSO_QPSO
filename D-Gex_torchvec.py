import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import sys
import qpso_vec 
from qpso_vec import QPSO
import time

class Network(nn.Module):
	def __init__(self,hidden_units,dropout_rate):
		super(Network,self).__init__()
		self.hidden_units=hidden_units
		self.fc1=nn.Linear(943,self.hidden_units)
		self.fc2=nn.Linear(self.hidden_units,476)
		self.dropout=nn.Dropout(p=dropout_rate)

	def forward(self,params,localbcosts):
		errg_best = localbcosts[np.argmin(localbcosts)]
		pos_best_g = np.zeros(params.shape[2])
		MAE_GTx = 0
		for i in range(params.shape[0]):
			self.vecToMat(params[i][0])
			x = F.tanh(self.fc1(X_train))
			x = self.dropout(x)
			x = self.fc2(x)
			MAE_GTx = torch.sum(torch.sum(torch.pow(torch.sub(x,Y_train),2),dim=0)/2921)
			if MAE_GTx<localbcosts[i] or localbcosts[i]==-1 :
				localbcosts[i] = MAE_GTx
				params[i][1] = np.copy(params[i][0])
			if MAE_GTx<errg_best or errg_best==-1:
				errg_best = MAE_GTx
				pos_best_g = np.copy(params[i][0])

			#print(MAE_GTx)
		return pos_best_g,errg_best,params
	
	def vecToMat(self,params):
		#params = np.array(params)
		params = torch.from_numpy(params).type(torch.FloatTensor).to(device)
		self.fc1.weight.data=params[0:n_inputs*n_hidden].view(n_hidden,n_inputs)
		self.fc1.bias.data=params[n_inputs*n_hidden:n_inputs*n_hidden+n_hidden].view(n_hidden)
		self.fc2.weight.data=params[n_inputs*n_hidden+n_hidden:n_inputs*n_hidden+n_hidden+n_hidden*n_outputs].view(n_outputs,n_hidden)
		self.fc2.bias.data=params[n_inputs*n_hidden+n_hidden+n_hidden*n_outputs:n_inputs*n_hidden+n_hidden+n_hidden*n_outputs + n_outputs].view(n_outputs)

n_hidden = int(sys.argv[1])
dropout_rate = float(sys.argv[2])
num_particles = int(sys.argv[3])
n_inputs = 943
n_outputs = 476
dimensions = (n_inputs*n_hidden)+(n_hidden*n_outputs)+n_hidden+n_outputs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = Network(n_hidden,dropout_rate)
model.to(device)
X_train = np.load('GTEx_X_float64.npy')
Y_train = np.load('GTEx_Y_0-4760_float64_0.npy')
X_train = torch.from_numpy(X_train).type(torch.FloatTensor).to(device)
Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor).to(device)
initial = np.random.randn(num_particles,2,dimensions)
initialcosts =  np.ones(num_particles)*-1
#initial = np.zeros(dimensions)
#print(initial)
#print(Y_train.size())
#model.forward(initial)
pos=QPSO(costfunc=model.forward,swarm=initial,maxiter=1000,verbose=True,beta=0.77,localbcosts=initialcosts)
model.vecToMat(pos)
#torch.save(model.state_dict(),"/home/azhar/projects/Evolution/Gen1.pt")

