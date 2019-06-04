import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import sys
import QPSO
from QPSO import QPSO
import pyswarms as ps
ypath = 'GTEx_Y_0-4760_float64_0.npy'
num_particles = 20
#from guppy import hpy
MAE_list = []
X_train = np.load('GTEx_X_float64.npy')
Y_train = np.load(ypath)
f =open(ypath.split('.')[0]+'_results.txt','a')
class Network(nn.Module):
    
    def __init__(self,hidden_units,dropout_rate):
        super(Network,self).__init__()
        self.hidden_units=hidden_units
        self.fc1=nn.Linear(943,self.hidden_units)
        self.fc2=nn.Linear(self.hidden_units,476)
        self.dropout=nn.Dropout(p=dropout_rate)
    def forward(self,params):
        global MAE_list
        params = np.array(params)
        params = torch.from_numpy(params).type(torch.FloatTensor).to(device)
        self.fc1.weight.data=params[0:n_inputs*n_hidden].view(n_hidden,n_inputs)
        self.fc1.bias.data=params[n_inputs*n_hidden:n_inputs*n_hidden+n_hidden].view(n_hidden)
        self.fc2.weight.data=params[n_inputs*n_hidden+n_hidden:n_inputs*n_hidden+n_hidden+n_hidden*n_outputs].view(n_outputs,n_hidden)
        self.fc2.bias.data=params[n_inputs*n_hidden+n_hidden+n_hidden*n_outputs:n_inputs*n_hidden+n_hidden+n_hidden*n_outputs + n_outputs].view(n_outputs)
        x = torch.tanh(self.fc1(X_train))
        x = self.dropout(x)
        x = self.fc2(x)
        MSE_GTx = torch.sum(torch.sum(torch.pow(torch.sub(x,Y_train),2),dim=0)/2921)
        MAE_GTx = torch.sum(torch.sum(torch.abs(torch.sub(x,Y_train)),dim=0)/(2921*n_outputs))
        MAE_list.append(float(MAE_GTx))
        if len(MAE_list) == num_particles:
            gBest = str(float(min(MAE_list)))
            f.write(gBest+'\n')
            print("written",gBest)
            MAE_list = []
        return float(MSE_GTx)




n_hidden = int(sys.argv[1])
dropout_rate = float(sys.argv[2])
n_inputs = 943
n_outputs = 476
dimensions = (n_inputs*n_hidden)+(n_hidden*n_outputs)+n_hidden+n_outputs
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = Network(n_hidden,dropout_rate)
model.to(device)
#X_train = np.load('GTEx_X_float64.npy')
#Y_train = np.load('GTEx_Y_0-4760_float64_0.npy')
X_train = torch.from_numpy(X_train).type(torch.FloatTensor).to(device)
Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor).to(device)
initial=np.random.normal(0,1,(dimensions,))
#initial = np.zeros(dimensions)
print(initial)
print(Y_train.size())
#model.forward(initial)
scoreList = []

pos=QPSO(model.forward,initial,num_particles=num_particles,maxiter=1000,verbose=True)

