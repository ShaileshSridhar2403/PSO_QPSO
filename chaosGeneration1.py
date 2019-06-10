#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:27:15 2019
@author: shailesh
"""
import numpy as np
import random
lastVal = 0.5
mu = 1.5
sigma=10.0
beta=2.66667
ro=28.
dt=1e-3


def initLastVal():
	global lastVal
	lastVal = 0.5
	
def rangeMatch(seq,original_lLimit,original_rLimit,new_lLimit,new_rLimit):
	seq = new_lLimit + (seq-original_lLimit)/(original_rLimit - original_lLimit)*(new_rLimit-new_lLimit)
	return seq
def tentMap(mu=1.5):
	x_prev = random.random()
	iterr = 20 
	for i in range(iterr):
		if x_prev<0.5:
			x_prev = mu*x_prev
		else :
			x_prev = mu*(1-x_prev)
	#print(x_prev)
	return x_prev

def tentMapChaosSeq(dimensions):		#pass dimensions as tuple
	#global lastVal
	nVals = np.prod(dimensions)
	l = []
	for i in range(nVals):
		val = tentMap(mu)
		l.append(val)
	#	lastVal = val
	l = rangeMatch(np.array(l),0.375,0.75,-20,20)
	return np.resize(l,dimensions)

def tentMapChaosSeq_01(dimensions):		#pass dimensions as tuple
	#global lastVal
	nVals = np.prod(dimensions)
	l = []
	for i in range(nVals):
		val = tentMap(mu)
		l.append(val)
	#	lastVal = val
	l = rangeMatch(np.array(l),0.375,0.75,0,1)
	return np.resize(l,dimensions)

def tentMapVal():
	global lastVal
	val = tentMap(lastVal,mu)
	lastVal = val
	return val

def lorentzMap(dt,sigma=10.0,beta=2.66667,ro=28.):
	x_prev = random.random()
	y_prev = random.random()
	z_prev = random.random()
	iterr = 20
	for i in range(iterr):
		xn = y_prev*dt*sigma + x_prev*(1 - dt*sigma)
		yn = x_prev*dt*(ro-z_prev) + y_prev*(1-dt)
		zn = x_prev*y_prev*dt + z_prev*(1 - dt*beta)
		x_prev, y_prev, z_prev = xn, yn, zn
	return xn # doubt don't know what to return

def lorentzMapChaosSeq(dimensions):		#pass dimensions as tuple
	#global lastVal
	nVals = np.prod(dimensions)
	l = []
	for i in range(nVals):
		val = lorentzMap(dt,sigma,beta,ro)
		l.append(val)
	#	lastVal = val
	l = rangeMatch(np.array(l),0.375,0.75,-20,20)
	return np.resize(l,dimensions)

def lorentzMapChaosSeq_01(dimensions):		#pass dimensions as tuple
	#global lastVal
	nVals = np.prod(dimensions)
	l = []
	for i in range(nVals):
		val = lorentzMap(dt,sigma,beta,ro)
		l.append(val)
	#	lastVal = val
	l = rangeMatch(np.array(l),0.375,0.75,0,1)
	return np.resize(l,dimensions)


	
#need to implement range matching properly for DGEX values right now(mu=1.5):
#	olLimit = 0.375
#	orLimit = 0.75
#	nlLimit = -6
#	nrLimit = 6