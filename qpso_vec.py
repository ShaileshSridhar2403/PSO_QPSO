import numpy as np 
import math
import time

def QPSO(swarm,localbcosts,beta,maxiter,costfunc,verbose):
	num_dim = swarm.shape[1]
	num_partcles = swarm.shape[0]
	pos_best_g= np.zeros(num_dim)
	pos_mbest = np.zeros(num_dim)
	err_best_g= - 1
	i = 0
	t_old = 0
	t_new = 0
	t_total_old = time.time()
	while (i<maxiter):
		t_old = time.time()
		pos_best_g,err_best_g,swarm = costfunc(swarm,localbcosts)
		pos_mbest = np.mean(swarm[:,1],axis=0)
		c1 = np.random.random_sample((num_partcles,1))
		c2 = np.random.random_sample((num_partcles,1))
		u = np.random.random_sample((num_partcles,1))
		k = np.random.random_sample((num_partcles,1))
		
		p = (c1*swarm[:,1]+c2*pos_best_g)/(c1+c2)
		Xfactor = beta*abs(pos_best_g-swarm[:,0])*np.log(1/u)

		swarm[:,0] = p+np.where(k>=0.5,1,-1)*Xfactor
		i=i+1
		t_new = time.time()
		if verbose: print('iter: {}, best solution: {} time elasped in secs:{}'.format(i,err_best_g,int(t_old-t_new)))
	
	print('\nFINAL SOLUTION:')
	#print('   > {}'.format(self.pos_best_g))
	print('   > {}\n'.format(err_best_g))
	t_total_new = time.time()
	print('total time elapsed:{}secs'.format(t_total_new-t_total_old))
	return pos_best_g






