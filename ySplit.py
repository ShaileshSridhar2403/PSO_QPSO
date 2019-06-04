#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:52:45 2019

@author: shailesh
"""

import numpy as np
fileName = 'GTEx_Y_0-4760_float64.npy'
Y_train = np.load(fileName)

fileSansExtension = fileName.split('.')[0]
n_partitions = 10
new_cols = Y_train.shape[-1]//n_partitions
for i in range(n_partitions):
    
    filePath = fileSansExtension+'_'+str(i)+'.npy'
    f=open(filePath,'w')
    f.close()
    new_mat = Y_train[:,i*new_cols:(i+1)*new_cols]
    np.save(filePath,new_mat)
    
mat =np.load(filePath)    