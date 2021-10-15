#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 09:40:15 2021

@author: Donnas
"""

# Remember indexing starts at 0


import SPoly
from IPython import get_ipython
import csv
from datetime import datetime
import ALLFUNCS

# clears variables to start an independent run
get_ipython().magic('reset -sf')



# Starting conditions

func1 = SPoly.uni_f1
func2 = SPoly.uni_f2
func3 = SPoly.uni_f3
func4 = SPoly.uni_f4
func5 = SPoly.uni_f5

func7 = SPoly.uni_f7


# max degree = highest polynomial order + 2
MD_F1 = 4+2
MD_F2 = 3+2
MD_F3 = 7+2 #uni
MD_F4 = 8+2 #uni
MD_F5 = 6+2 #unit

MD_F7 = 8+2


# control parameters from table 3

c1_F1 = 0.4545
c2_F1 = 0.5431
c3_F1 = 4.8132
c4_F1 = 4.0829

#f2
c1_F2 = 0.9648
c2_F2 = 0.5351
c3_F2 = 3.5410
c4_F2 = 3.5410

#f3
c1_F3 = 0.7226
c2_F3 = 0.6523
c3_F3 = 4.8417
c4_F3 = 3.2346


#f4
c1_F4 = 0.5166
c2_F4 = 0.5498
c3_F4 = 3.8793
c4_F4 = 4.1782

#f5
c1_F5 = 0.4133
c2_F5 = 0.6356
c3_F5 = 4.5221
c4_F5 = 4.4698

#f7
c1_F7 = 0.6936
c2_F7 = 0.5973
c3_F7 = 4.3133
c4_F7 = 4.7115



# k_tournament selection
k = 2
# set number of datapoints 
npoints = 2000
#Choose number of particles in the swarm 
swarmp = 100
# Range
x_low = -10
x_high = 10
# stopping critera
max_it = 500
MSE_condition = 1E-10
relative_err = 1E-10

runs = 1
  
f1k = []  
f2k = []  
f3k = []  
f4k = []  
f5k = [] 

f7k = []


fields = ["Iterations", " Poly set", "MSE", "How", "k", "Test Err"]

startTime1k = datetime.now()

# for i in range(runs): 
#     print("\n************** Run {0}  for Uni1 *****************\n".format(i+1))
#     f1k.append(ALLFUNCS.SBPSO(func1,MD_F1,c1_F1,c2_F1,c3_F1,c4_F1,k,npoints,swarmp,x_low,x_high,max_it,MSE_condition,relative_err))

# timeEnd1k = datetime.now() - startTime1k

# with open("Uni1_k.csv","w") as f1fk:
#     f1k_write = csv.writer(f1fk)
    
#     f1k_write.writerow(fields)
#     f1k_write.writerows(f1k)


# startTime2k = datetime.now()
        
# for i in range(runs): 
#     print("\n************** Run {0} for Uni2 *****************\n".format(i+1))
#     f2k.append(ALLFUNCS.SBPSO(func2,MD_F2,c1_F2,c2_F2,c3_F2,c4_F2,k,npoints,swarmp,x_low,x_high,max_it,MSE_condition,relative_err))

# timeEnd2k = datetime.now() - startTime2k

# with open("Uni2_k.csv","w") as f2fk:
#     f2k_write = csv.writer(f2fk)
    
#     f2k_write.writerow(fields)
#     f2k_write.writerows(f2k)

# startTime3k = datetime.now()
 
# for i in range(runs): 
#     print("\n************** Run {0} for Uni3 *****************\n".format(i+1))

#     f3k.append(ALLFUNCS.SBPSO(func3,MD_F3,c1_F3,c2_F3,c3_F3,c4_F3,k,npoints,swarmp,x_low,x_high,max_it,MSE_condition,relative_err))
        
# timeEnd3k = datetime.now() - startTime3k

# with open("Uni3_k.csv","w") as f3fk:
#     f3k_write = csv.writer(f3fk)
    
#     f3k_write.writerow(fields)
#     f3k_write.writerows(f3k)


# startTime4k = datetime.now()

# for i in range(runs): 
    # print("\n************** Run {0} for Uni4 *****************\n".format(i+1))
    # f4k.append(ALLFUNCS.SBPSO(func4,MD_F4,c1_F4,c2_F4,c3_F4,c4_F4,k,npoints,swarmp,x_low,x_high,max_it,MSE_condition,relative_err))

# timeEnd4k = datetime.now() - startTime4k        

# with open("Uni4_k.csv","w") as f4fk:
#     f4k_write = csv.writer(f4fk)
    
#     f4k_write.writerow(fields)
#     f4k_write.writerows(f4k)
  

# startTime5k = datetime.now()
  
# for i in range(runs): 
#     print("\n************** Run {0} for Uni5 *****************\n".format(i+1))
#     f5k.append(ALLFUNCS.SBPSO(func5,MD_F5,c1_F5,c2_F5,c3_F5,c4_F5,k,npoints,swarmp,x_low,x_high,max_it,MSE_condition,relative_err))
 
# timeEnd5 = datetime.now() - startTime5k              

# with open("Uni5_k.csv","w") as f5fk:
#     f5k_write = csv.writer(f5fk)
    
#     f5k_write.writerow(fields)
#     f5k_write.writerows(f5k)

for i in range(runs): 
    print("\n************** Run {0} for Uni7 *****************\n".format(i+1))
    f7k.append(ALLFUNCS.SBPSO(func7,MD_F7,c1_F7,c2_F7,c3_F7,c4_F7,k,npoints,swarmp,x_low,x_high,max_it,MSE_condition,relative_err))
 
# timeEnd5 = datetime.now() - startTime5k              

# with open("Uni5_k.csv","w") as f5fk:
#     f5k_write = csv.writer(f5fk)
    
#     f5k_write.writerow(fields)
#     f5k_write.writerows(f5k)
    

    

timeEnd = datetime.now() - startTime1k
print(timeEnd)