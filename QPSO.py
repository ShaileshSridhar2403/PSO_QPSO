from __future__ import division
import random
import math
import pandas as pd
import numpy as np
import time
import sys

#dataset = pd.read_csv('PSOdata.csv')


# function we are attempting to optimize (minimize)
#def func1(x):
#	total=0
#	total = (dataset["x"]**x[0]*dataset["y"]**x[1]).sum()
#	return total

class Particle:
	def __init__(self,x0,num_dimensions):          
		self.pos_best_i=None          # best position individual 
		self.err_best_i=-1          # best error individual 
		self.err_i=-1               # error individual
		self.num_dimensions=num_dimensions
		self.position_i = np.random.rand(num_dimensions)# particle position

		# evaluate current fitness
	def evaluate(self,costFunc):
		self.err_i=costFunc(self.position_i)

		# check to see if the current position is an individual best
		if self.err_i<self.err_best_i or self.err_best_i==-1:
			self.pos_best_i=self.position_i
			self.err_best_i=self.err_i
	#update position
	def update_position(self,pos_best_g,pos_mbest,beta):
		pos_mbest=np.array(pos_mbest)
		pos_best_g=np.array(pos_best_g)
		c1=random.random()
		c2=random.random()
		k=random.random()
		u=random.random()
		p=np.add(c1*np.array(self.pos_best_i),c2*pos_best_g)/(c1+c2)
		Xfactor=(beta*abs(pos_mbest-np.array(self.position_i))*math.log(1/u))
		if(k>=0.5):
			self.position_i=list(p+Xfactor)
		if(k<0.5):
			self.position_i=list(p-Xfactor)

		'''for i in range(0,self.num_dimensions):
			if self.position_i[i]>bounds[i][1]:
				self.position_i[i]=bounds[i][1]

			# adjust minimum position if neseccary
			if self.position_i[i]<bounds[i][0]:
				self.position_i[i]=bounds[i][0]'''


class QPSO():
    def __init__(self, costFunc, x0, num_particles, maxiter, verbose=False):
        beta=-0.77
        t_old=0
        t_new=0
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        self.pos_best_g=[]                   # best position for group
        pos_mbest=np.empty(num_dimensions)

		# establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0,num_dimensions))
        print('SIZE ',sys.getsizeof(swarm))
		# begin optimization loop
        i=0
        t_old = time.time()
        sat_count = 0
        err_best_g_prev = 0
        diff = 0
        while i<maxiter:
            if sat_count >49:
                print('saturated')
                exit(0)
            pos_mbest=np.empty(num_dimensions)
            if verbose: print('iter: {}, best solution: {} time elapsed in secs:{}'.format(i,err_best_g,int(t_new-t_old)))
            
            t_new = time.time()
			# cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)
                pos_mbest=np.add(pos_mbest,swarm[j].pos_best_i)#calculate mean of best position of all particles

				# determine if current particle is the best (globally)
                if swarm[j].err_i<err_best_g or err_best_g==-1:
                    self.pos_best_g = 0
                    self.pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)
                   
            pos_mbest=pos_mbest/num_particles
            pos_mbest=list(pos_mbest)
            diff = err_best_g_prev-err_best_g
            if int(diff)==0:
                sat_count+=1
            else:
                sat_count = 0
#            print('prev,curr,sat_count',err_best_g_prev,err_best_g,diff,sat_count)
			# cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_position(self.pos_best_g,pos_mbest,beta)
            i+=1
            err_best_g_prev = err_best_g
			
		# print final results
        print('\nFINAL SOLUTION:')
		#print('   > {}'.format(self.pos_best_g))
        print('   > {}\n'.format(err_best_g))
#if __name__ == "__QPSO__":
#	main()

#--- RUN ----------------------------------------------------------------------+

#initial=[0,0]               # initial starting location [x1,x2...]
#bounds=[(-10,10),(-10,10)]
#QPSO(func1, initial,bounds, num_particles=100, maxiter=200,verbose=True)

#--- END ----------------------------------------------------------------------+















