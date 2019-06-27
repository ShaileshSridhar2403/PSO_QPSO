import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import sys
import qpso_vec 
from qpso_vec import QPSO
from pso_vec import PSO
from chaotic_qpso_vec import CQPSO
from chaotic_pso_vec import CPSO
from pso_vec_grad import PSO_grad
import time
import chaosGeneration as cg
import loadDataset

XFolder = '/home/shailesh/gitHub/PSO_QPSO/datasets/GEO/bgedv2_X_tr_float64'
YFolder = '/home/shailesh/gitHub/PSO_QPSO/datasets/GEO/bgedv2_Y_tr_0-4760_float64'
#Xpath = '/home/shailesh/gitHub/PSO_QPSO/datasets/GEO/bgedv2_X_te_float64.npy'
#Ypath = '/home/shailesh/gitHub/PSO_QPSO/datasets/GEO/bgedv2_Y_te_0-4760_float64.npy'

betaFile = open('beta.txt','w')
learnedParams = np.load('/home/shailesh/gitHub/PSO_QPSO/trainedData.npy')

def getbatch(bsize):
	seed = np.random.choice(X_train.size()[0], size = bsize, replace = False)
	xb = X_train[seed,:]
	yb = Y_train[seed,:]
	return xb,yb



class Network_PSO(nn.Module):
	def __init__(self,hidden_units,dropout_rate):
		super(Network_PSO,self).__init__()
		self.hidden_units=hidden_units
		self.fc1=nn.Linear(943,self.hidden_units)
		self.fc2=nn.Linear(self.hidden_units,4760)
		self.dropout=nn.Dropout(p=dropout_rate)

	def forward(self,params,localbcosts):
		errg_best = localbcosts[np.argmin(localbcosts)]
		pos_best_g = np.zeros(params.shape[2])
		MAE_list = np.zeros(params.shape[0])
		X = loadDataset.loadNextPart(XFolder)
		Y = loadDataset.loadNextPart(YFolder)
	
		while(type(X)!=int):
			X_train = torch.from_numpy(X).type(torch.FloatTensor).to(device)
			Y_train = torch.from_numpy(Y).type(torch.FloatTensor).to(device)
			for i in range(params.shape[0]):
				self.vecToMat(params[i][0])
				MAE_GTx = 0
				x = torch.tanh(self.fc1(X_train))			#x = torch.tanh(self.fc1(X_train))
				x = self.dropout(x)
				x = self.fc2(x)
		#		MAE_GTx = torch.sum(torch.sum(torch.pow(torch.sub(x,Y_train),2),dim=0)/2921)
		#		MAE_GTx = torch.sum(torch.sum(torch.abs(torch.sub(x,Y_train)),dim=0)/(2921*n_outputs))
				MAE_list[i] += torch.sum(torch.sum(torch.abs(torch.sub(x,Y_train)),dim=0)/2921)/n_outputs #copy pasted from D-Gex file
		#		MAE_GTx = torch.sum(torch.sum(torch.abs(torch.sub(x,y)),dim=0)/64)/n_outputs #copy pasted from D-Gex file
			X = loadDataset.loadNextPart(XFolder)
			Y = loadDataset.loadNextPart(YFolder)	
			
		
		print(MAE_list)
		for i in range(params.shape[0]):
			if MAE_list[i]<localbcosts[i] or localbcosts[i]==-1 :
				localbcosts[i] = MAE_list[i]
				params[i][1] = np.copy(params[i][0])
		minMAEInd = np.argmin(MAE_list)
		if MAE_list[minMAEInd]<errg_best or errg_best==-1:
			errg_best = MAE_list[minMAEInd]
			pos_best_g = np.copy(params[minMAEInd][0])
			
			
			
#			print(MAE_GTx.item())
		return pos_best_g,errg_best,params
	
	def vecToMat(self,params):
		#params = np.array(params)
		params = torch.from_numpy(params).type(torch.FloatTensor).to(device)
		self.fc1.weight.data=params[0:n_inputs*n_hidden].view(n_hidden,n_inputs)
		self.fc1.bias.data=params[n_inputs*n_hidden:n_inputs*n_hidden+n_hidden].view(n_hidden)
		self.fc2.weight.data=params[n_inputs*n_hidden+n_hidden:n_inputs*n_hidden+n_hidden+n_hidden*n_outputs].view(n_outputs,n_hidden)
		self.fc2.bias.data=params[n_inputs*n_hidden+n_hidden+n_hidden*n_outputs:n_inputs*n_hidden+n_hidden+n_hidden*n_outputs + n_outputs].view(n_outputs)


#def train(model,steps,verbose=True,bsize=64,epochs=1000):
#	
#	for epoch in range(epochs):
#		for step in range(steps):
#			x,y = getbatch(bsize)
#			optimizer.zero_grad()
#			outputs = model.forward(x)
#			#loss = criterion(outputs,y)
#			#loss = torch.sum(torch.sum(torch.pow(torch.sub(outputs,y),2),dim=0)/bsize)/n_outputs
#			loss = torch.sum(torch.sum(torch.pow(torch.sub(outputs,y),2),dim=0)/bsize)#new loss according to paper large values of loss
#			loss.backward()
#			optimizer.step()
#		#lossepoch = torch.sum(torch.sum(torch.pow(torch.sub(model.forward(X_train),Y_train),2),dim=0)/2921)/n_outputs
#		lossepoch = torch.sum(torch.sum(torch.pow(torch.sub(model.forward(X_test),Y_test),2),dim=0)/2921)#new total loss
#		MAE = torch.sum(torch.sum(torch.abs(torch.sub(model.forward(X_test),Y_test)),dim=0)/(0.2*2921))/n_outputs#this should be the MAE
#		if verbose: print('epoch-{} training loss:{} MAE:{}'.format(epoch+1,lossepoch.item(),MAE.item()))
#		
	

n_hidden = int(sys.argv[1])
dropout_rate = float(sys.argv[2])
num_particles = int(sys.argv[3])
optimization_method = int(sys.argv[4])  #0=QPSO 1=PSO
n_inputs = 943
n_outputs = 4760
dimensions = (n_inputs*n_hidden)+(n_hidden*n_outputs)+n_hidden+n_outputs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = Network_PSO(n_hidden,dropout_rate)
model.to(device)
#X_train = np.load(Xpath)
#Y_train = np.load(Ypath)
#X_train = torch.from_numpy(X_train).type(torch.FloatTensor).to(device)
#Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor).to(device)

initialcosts =  np.ones(num_particles)*-1

print('dimenstino',dimensions)
#initial = np.zeros(dimensions)
#print(initial)
#print(Y_train.size())
#model.forward(initial)
if optimization_method == 0:
	print('QPSO')
#	logfilePath = 'Plots/QPSO_'+Xpath.split('.')[0]+'_hidden'+str(n_hidden)+'_particles'+str(num_particles)+'.txt'
	logfilePath= 'Plots/QPSO_test.txt'
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
	initial = np.random.randn(num_particles,2,dimensions)
#	for i in range(initial.shape[0]):
#		initial[i][0] = learnedParams
#		initial[i][1] = learnedParams
#	print(initial[0][0])
#	exit(0)
	pos,err,iters=QPSO(costfunc=model.forward,swarm=initial,maxiter=1000,verbose=True,beta = 0.00000000000000001,localbcosts=initialcosts,logfile =logfilePath )
	model.vecToMat(pos)
elif optimization_method == 1:
#	logfilePath = 'Plots/PSO_'+Xpath.split('.')[0]+'_hidden'+str(n_hidden)+'_particles'+str(num_particles)+'.txt'
	logfilePath= 'Plots/QPSO_test.txt'
	print('PSO')
	initial = np.random.randn(num_particles,3,dimensions)
#	initialcosts =  np.ones(num_particles)*-1
	pos,err,iters=PSO(costfunc=model.forward,swarm=initial,maxiter=1000,verbose=True,w = 0.3,c1=1,c2=3,localbcosts=initialcosts,logfile =logfilePath)
	model.vecToMat(pos)
	print(initial)
elif optimization_method == 2:
	logfilePath = 'Plots/CQPSO_'+Ypath.split('.')[0]+'_hidden'+str(n_hidden)+'_particles'+str(num_particles)+'.txt'
	print('CQPSO')
	initial = cg.tentMapChaosSeq((num_particles,2,dimensions))
	pos,err,iters=CQPSO(costfunc=model.forward,swarm=initial,maxiter=1000,verbose=True,beta = 0.0000000000001,localbcosts=initialcosts,logfile =logfilePath)
	model.vecToMat(pos)
	
elif optimization_method == 3:
	logfilePath = 'Plots/CPSO_'+Ypath.split('.')[0]+'_hidden'+str(n_hidden)+'_particles'+str(num_particles)+'.txt'
	print('CPSO')
	initial = cg.tentMapChaosSeq((num_particles,3,dimensions))
	pos,err,iters=CPSO(costfunc=model.forward,swarm=initial,bounds = [-2.5,2.5],maxiter=1000,verbose=True,w = 0.001,c1=10,c2=1,localbcosts=initialcosts,logfile =logfilePath )
	model.vecToMat(pos)
elif optimization_method == 4:
	logfilePath = 'Plots/PSO_grad_'+Ypath.split('.')[0]+'_hidden'+str(n_hidden)+'_particles'+str(num_particles)+'.txt'
	print('PSO')
	initial = np.random.randn(num_particles,3,dimensions)
#	initialcosts =  np.ones(num_particles)*-1
	pos,err,iters= PSO_grad(costfunc=model.forward,swarm=initial,maxiter=1000,verbose=True,w = 0.3,c1=1,c2=2,c3=0.4,localbcosts=initialcosts,logfile =logfilePath)
	model.vecToMat(pos)
	print(initial)
else:
	print('error')
