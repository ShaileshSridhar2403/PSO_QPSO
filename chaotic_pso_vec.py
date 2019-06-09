#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 20:21:04 2019

@author: shailesh
"""
import numpy as np 
import math
import time
import chaosGeneration as cg

def CPSO(swarm,localbcosts,w,c1,c2,maxiter,costfunc,verbose,logfile):
	f = open(logfile,'a')
	f.truncate(0)									
	num_dim = swarm.shape[1]
	num_partcles = swarm.shape[0]
	pos_best_g= np.zeros(num_dim)
	pos_mbest = np.zeros(num_dim)
	err_best_g= - 1
	i = 0;
	t_init = time.time()
	sat_count = 0
	err_best_g_prev = 0
	t_total_old = time.time()
	while (i<maxiter):
		if sat_count >49:
			print('saturated')
			return pos_best_g,err_best_g,i
		t_old = time.time()
		pos_best_g,err_best_g,swarm = costfunc(swarm,localbcosts)
		diff = round(int(err_best_g_prev),2)-round(int(err_best_g),2)
		if int(diff)==0:
			sat_count+=1
		else:
			sat_count = 0
		r1 = cg.tentMapChaosSeq_01((num_partcles,1))
		r2 = cg.tentMapChaosSeq_01((num_partcles,1))
#		vel_cognitive = c1*r1*(swarm[:,1]-swarm[:,0])
#		vel_social = c2*r2*(pos_best_g-swarm[:,0])
		swarm[:,2]*=w
		swarm[:,2] += c1*r1*(swarm[:,1]-swarm[:,0]) + c2*r2*(pos_best_g-swarm[:,0])  #new velocity
		
		swarm[:,0] += swarm[:,2]					#new position
		
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
		
		
		