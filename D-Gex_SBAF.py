import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import sys
import qpso_vec 
from qpso_vec import QPSO
from pso_vec import PSO
#from chaotic_qpso_vec import CQPSO
#from chaotic_pso_vec import CPSO
import time
#import chaosGeneration as cg


#betaFile = open('beta.txt','w')
def dataset_minmax(dataset):
		minmax = list()
		stats = [[min(column), max(column)] for column in zip(*dataset)]
		return stats

def normalize_dataset(dataset, minmax):
		for row in dataset:
			for i in range(len(row)):
					row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def SBAF(act):
	#print(act)
	act = act.cpu()
	act = act.detach().numpy()
	mm = dataset_minmax(act)
	#print(len(mm))
	normalize_dataset(act,mm)
	#print(act)
	act = torch.from_numpy(act).type(torch.FloatTensor).to(device)
	val = 0.5
	y = 1.0 / (1.0 + (0.91 * (torch.pow(act,val)*torch.pow(1-act,1-val))))
	y = torch.abs(y)
	return y

class Network(nn.Module):
	def __init__(self,hidden_units,dropout_rate):
		super(Network,self).__init__()
		self.hidden_units=hidden_units
		self.fc1=nn.Linear(943,self.hidden_units)
		self.bn1 = nn.BatchNorm1d(self.hidden_units)
		self.fc2=nn.Linear(self.hidden_units,4760)
		self.dropout=nn.Dropout(p=dropout_rate)

	def forward(self,params,localbcosts):
		errg_best = localbcosts[np.argmin(localbcosts)]
		pos_best_g = np.zeros(params.shape[2])
		MAE_GTx = 0
		for i in range(params.shape[0]):
			self.vecToMat(params[i][0])
			x = self.fc1(X_train)
			#print('first layer weights')
			#print(self.fc1.weight.data)
			#print('first activation')
			#print(x)
			x = SBAF(x)
			#print('SBAF')
			#print(x)
			x = self.dropout(x)
			x = self.fc2(x)
			#MAE_GTx = torch.sum(torch.sum(torch.pow(torch.sub(x,Y_train),2),dim=0)/2921)/n_outputs
			#print(MAE_GTx)
			MAE_GTx = torch.sum(torch.sum(torch.abs(torch.sub(x,Y_train)),dim=0)/2921)/n_outputs
			print('calculated MAE_GTx')
			print(MAE_GTx)
			if MAE_GTx<localbcosts[i] or localbcosts[i]==-1 :
				localbcosts[i] = MAE_GTx
				params[i][1] = np.copy(params[i][0])
			if MAE_GTx<errg_best or errg_best==-1:
				errg_best = MAE_GTx
				pos_best_g = np.copy(params[i][0])

			#print(MAE_GTx)
		return pos_best_g,float(errg_best),params
	
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
optimization_method = int(sys.argv[4])  #0=QPSO 1=PSO
n_inputs = 943
n_outputs = 4760
dimensions = (n_inputs*n_hidden)+(n_hidden*n_outputs)+n_hidden+n_outputs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(dimensions)
model = Network(n_hidden,dropout_rate)
model.to(device)
X_train = np.load('GTEx_X_float64.npy')
Y_train = np.load('GTEx_Y_0-4760_float64.npy')
X_train = torch.from_numpy(X_train).type(torch.FloatTensor).to(device)
Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor).to(device)
initialcosts =  np.ones(num_particles)*-1
if optimization_method == 0:
	print('QPSO')
#	for beta100 in range(1,101):
#		print('-'*100)
#		print(beta100/100)
#		initial = np.random.randn(num_particles,2,dimensions)
#		initialcosts =  np.ones(num_particles)*-1
#		pos,err,iters=QPSO(costfunc=model.forward,swarm=initial,maxiter=1000,verbose=True,beta = beta100/100,localbcosts=initialcosts,logfile ='GTEx_Y_0-4760_float64.txt' )
#		model.vecToMat(pos)
#		betaFile.write(str(beta100/100)+' '+str(float(err))+' '+str(iters)+'\n')
	#torch.save(model.state_dict(),"/home/azhar/projects/Evolution/Gen1.pt")
#	betaFile.close()
	initial = np.zeros((num_particles,2,dimensions))
	initial[:,0] = np.random.rand(num_particles,dimensions)
	initial[:,1] = np.copy(initial[:,0])
	pos,err,iters=QPSO(costfunc=model.forward,swarm=initial,maxiter=1000,verbose=True,beta = 0.77,localbcosts=initialcosts,logfile ='logqpso.txt' )
	model.vecToMat(pos)
elif optimization_method == 1:
	print('PSO')
	initial = np.zeros((num_particles,3,dimensions))
	initial[:,0] = np.random.rand(num_particles,dimensions)
	initial[:,1] = np.copy(initial[:,0])
	initial[:,0] = 1

#	initialcosts =  np.ones(num_particles)*-1
	pos,err,iters=PSO(costfunc=model.forward,swarm=initial,maxiter=1000,verbose=True,w = 0.5,c1=1,c2=2,localbcosts=initialcosts,logfile ='logpso.txt' )
	model.vecToMat(pos)
	
elif optimization_method == 2:
	print('CQPSO')
	initial = np.zeros((num_particles,2,dimensions))
	initial[:,0] = cg.lorentzMapChaosSeq((num_particles,dimensions))
	initial[:,1] = np.copy(initial[:,0])
	pos,err,iters=CQPSO(costfunc=model.forward,swarm=initial,maxiter=1000,verbose=True,beta = 0.77,localbcosts=initialcosts,logfile ='logcqpso.txt' )
	model.vecToMat(pos)
	
elif optimization_method == 3:
	print('CPSO')
	initial = cg.tentMapChaosSeq((num_particles,3,dimensions))
	pos,err,iters=CPSO(costfunc=model.forward,swarm=initial,maxiter=1000,verbose=True,w = 0.5,c1=1,c2=2,localbcosts=initialcosts,logfile ='logcpso.txt' )
	model.vecToMat(pos)
	
else:
	print('error')



'''
			for debugging 
			print('final output')
			print(x)
			print('Y_train')
			print(Y_train)
			#print(x)
			MAE_GTx = torch.sub(Y_train,x)
			print('difference')
			print(MAE_GTx)
			MAE_GTx = torch.abs(MAE_GTx)
			print('absolute value')
			print(MAE_GTx)
			MAE_GTx = torch.sum(MAE_GTx,dim=0)
			#print(MAE_GTx)
			MAE_GTx = MAE_GTx/2921
			print('sum along rows')
			print(MAE_GTx)
			MAE_GTx = torch.sum(MAE_GTx)/n_outputs
			print('sum along column')
			print(MAE_GTx)
'''