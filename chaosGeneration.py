#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:27:15 2019

@author: shailesh
"""
import numpy as np
lastVal = 0.5
mu = 1.5



def initLastVal():
	global lastVal
	lastVal = 0.5
	
def rangeMatch(seq,original_lLimit,original_rLimit,new_lLimit,new_rLimit):
	seq = new_lLimit + (seq-original_lLimit)/(original_rLimit - original_lLimit)*(new_rLimit-new_lLimit)
	return seq
def tentMap(x,mu):
	if x<0.5:
		return mu*x
	else :
		return mu*(1-x)
def tentMapChaosSeq(dimensions):		#pass dimensions as tuple
	global lastVal
	nVals = np.prod(dimensions)
	l = [lastVal]
	for i in range(nVals-1):
		val = tentMap(lastVal,mu)
		l.append(val)
		lastVal = val
	l = rangeMatch(np.array(l),0.375,0.75,-20,20)
	return np.resize(l,dimensions)

def tentMapChaosSeq_01(dimensions):		#pass dimensions as tuple
	global lastVal
	nVals = np.prod(dimensions)
	l = [lastVal]
	for i in range(nVals-1):
		val = tentMap(lastVal,mu)
		l.append(val)
		lastVal = val
	l = rangeMatch(np.array(l),0.375,0.75,0,1)
	return np.resize(l,dimensions)

def tentMapVal():
	global lastVal
	val = tentMap(lastVal,mu)
	lastVal = val
	return val


	
#need to implement range matching properly for DGEX values right now(mu=1.5):
#	olLimit = 0.375
#	orLimit = 0.75
#	nlLimit = -6
#	nrLimit = 6