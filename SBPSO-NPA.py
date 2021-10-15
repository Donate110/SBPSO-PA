#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 10:33:03 2021

@author: Donnas
"""

import random
from random import randint
from random import shuffle
import numpy
import math
import SPoly
import ALLFUNCS
from datetime import datetime
from IPython import get_ipython
import csv

# clears variables to start an independent run
# get_ipython().magic('reset -sf')

# function

     
#___________________________________________________________________________
#   Enter specs
#___________________________________________________________________________
#___________________________________________________________________________

# k_tournament selection
k = 2

# set number of datapoints 
npoints = 500
# Choose number of particles in the swarm 
swarmp = 100
psoswarm = 50

# SBPSO Range
x_low = -10
x_high = 10


# stopping critera
max_it = 500
psoit = 1000
MSE_condition = 1E-10
SSE_condition = 1E-15
relative_err = 1E-15

runs = 5

#___________________________________________________________________________


fc6 = []
fc7 = []
fc8 = []
fc9 = []
fc10 = []

w = 0.7
c1 = 1.4
c2 = 1.4
pen = 0


fields = ["SBPSOit", " Poly set", "SBPSO_MSE","how","k","PSOit", "Coef", "PSO_MSE","testMSE"]

startTime1 = datetime.now()

cfunc6 = SPoly.cf1 
xL6 = 0
xH6 = 3.0
ufunc6 = SPoly.uni_f1 
MD6 = 4+2

c1_F6 = 0.4545
c2_F6 = 0.5431
c3_F6 = 4.8132
c4_F6 = 4.0829
breaker6 = 2**2 +3**2 
     

      
for i in range(runs): 
    print("\n************** Run {0}  for COMBO1 *****************\n".format(i+1))
    fc6.append(ALLFUNCS.SBPSO_combo(ufunc6,cfunc6,MD6,c1_F6,c2_F6,c3_F6,c4_F6,k,w,c1,c2,pen,npoints,swarmp,psoswarm,x_low,x_high,xL6,xH6,max_it,psoit,MSE_condition,SSE_condition,relative_err,breaker6))


with open("COMBO1_c.csv","w") as F6fc:
    F6c_write = csv.writer(F6fc)
    
    F6c_write.writerow(fields)
    F6c_write.writerows(fc6)


cfunc7 = SPoly.cf2
xL7 = -1.0
xH7 = 2.0
ufunc7 = SPoly.uni_f2
MD7 = 3+2

c1_F7 = 0.9648
c2_F7 = 0.5351
c3_F7 = 3.5410
c4_F7 = 3.5410
breaker7 = 0.5**2 + 2**2 + (-1)**2

    
for i in range(runs): 
    print("\n************** Run {0}  for COMBO2 *****************\n".format(i+1))
    fc7.append(ALLFUNCS.SBPSO_combo(ufunc7,cfunc7,MD7,c1_F7,c2_F7,c3_F7,c4_F7,k,w,c1,c2,pen,npoints,swarmp,psoswarm,x_low,x_high,xL7,xH7,max_it,psoit,MSE_condition,SSE_condition,relative_err,breaker7))
    
    

with open("COMBO2_c.csv","w") as F7fc:
    F7c_write = csv.writer(F7fc)
    
    F7c_write.writerow(fields)
    F7c_write.writerows(fc7)

cfunc8 = SPoly.cf7
xL8 = -2.0
xH8 = 2.0
ufunc8 = SPoly.uni_f7
MD8 = 8+2

c1_F8 = 0.6936
c2_F8 = 0.5973
c3_F8 = 4.3133
c4_F8 = 4.7115
breaker8 = 2**2 + 1.5**2 + 2**2 +0.5**2
    
for i in range(runs): 
    print("\n************** Run {0}  for COMBO3 *****************\n".format(i+1))
    fc8.append(ALLFUNCS.SBPSO_combo(ufunc8,cfunc8,MD8,c1_F8,c2_F8,c3_F8,c4_F8,k,w,c1,c2,pen,npoints,swarmp,psoswarm,x_low,x_high,xL8,xH8,max_it,psoit,MSE_condition,SSE_condition,relative_err,breaker8))


with open("COMBO3_c.csv","w") as F8fc:
    F8c_write = csv.writer(F8fc)
    
    F8c_write.writerow(fields)
    F8c_write.writerows(fc8)     


cfunc9 = SPoly.cf5
xL9 = -3.0
xH9 = 5.0
ufunc9 = SPoly.uni_f5
MD9 = 6+2

c1_F9 = 0.4133
c2_F9 = 0.6356
c3_F9 = 4.5221
c4_F9 = 4.4698
breaker9 = 3**2 + (-3**2) + 5**2 + 1 


   
for i in range(runs): 
    print("\n************** Run {0}  for COMBO4 *****************\n".format(i+1))
    fc9.append(ALLFUNCS.SBPSO_combo(ufunc9,cfunc9,MD9,c1_F9,c2_F9,c3_F9,c4_F9,k,w,c1,c2,pen,npoints,swarmp,psoswarm,x_low,x_high,xL9,xH9,max_it,psoit,MSE_condition,SSE_condition,relative_err, breaker9))


with open("COMBO4_c.csv","w") as F9fc:
    F9c_write = csv.writer(F9fc)
    
    F9c_write.writerow(fields)
    F9c_write.writerows(fc9)  




cfunc10 = SPoly.cf6
xL10 = -5.2
xH10 = 0
ufunc10 = SPoly.uni_f6

MD10 = 5+2

c1_F10 = 0.5244
c2_F10 = 0.7465
c3_F10 = 4.5489
c4_F10 = 4.3787
breaker10 = (-2.5**2) + (-5.2**2) + (-1**2)

  
for i in range(runs): 
    print("\n************** Run {0}  for COMBO5 *****************\n".format(i+1))
    fc10.append(ALLFUNCS.SBPSO_combo(ufunc10,cfunc10,MD10,c1_F10,c2_F10,c3_F10,c4_F10,k,w,c1,c2,pen,npoints,swarmp,psoswarm,x_low,x_high,xL10,xH10,max_it,psoit,MSE_condition,SSE_condition,relative_err,breaker10))


with open("COMBO5_c.csv","w") as f10fc:
    f10c_write = csv.writer(f10fc)
    
    f10c_write.writerow(fields)
    f10c_write.writerows(fc10)  


timeEnd = datetime.now() - startTime1
print(timeEnd)



