#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:51:02 2021

@author: Donnas
"""


import numpy
import math
import ALLFUNCS
from datetime import datetime
import DPoly
import csv
import matplotlib.pyplot as plt



# k_tournament selection
k = 2

# set number of datapoints 
npoints = 500
# Choose number of particles in the swarm 
swarmp = 50
QP = 0.3

psoswarm = 50
QPs = 0.3

# SBPSO Range
x_low = -10
x_high = 10


# stopping critera
max_it = 1000
psoit = 500
MSE_condition = 1E-100
SSE_condition = 1E-1    
relative_err = 1E-100

runs = 1
ChangeP = [0.4,0.95]#.2,0.4,0.6,0.8,0.95]


fbase = DPoly.base
func1 = DPoly.case1f1
func2 = DPoly.case1f2
func3 = DPoly.case1f3

ufbase = DPoly.ubase
ufunc1 = DPoly.case1u1
ufunc2 = DPoly.case1u2
ufunc3 = DPoly.case1u3

xLb = -2
xHb = 2
Xfb = {0,1,2}
coefb = [-2,1,2][::-1]

xL1 = -2
xH1 = 2
Xf1 = {1,3,5,7}
coef1 = [1,2,1,1][::-1]

xL2 = -2
xH2 = 2
Xf2 = {0,2,8}
coef2 = [-2,1,2][::-1]

xL3 = -2
xH3 = 2
Xf3 = {0,2,3}
coef3 = [-2,2,1][::-1]

funcs1 = [fbase,func1,func2,func3]
ufuncs1 = [ufbase,ufunc1,ufunc2,ufunc3]
xLow1 = [xLb, xL1, xL2, xL3]
xHi1 = [xHb, xH1, xH2, xH3]
Xfull1 = [Xfb, Xf1, Xf2, Xf3]
coefs1 = [coefb, coef1, coef2, coef3]

cfunc1 = funcs1[0]
ufunc1 = ufuncs1[0]
xLc1 = xLow1[0]
xHc1 = xHi1[0]
Xfc1 = Xfull1[0]
Cc1 = coefs1[0]

d1 = []
p1SBPSO = []
p1coef = []
track1 = []
selectedFunc1 = []



fields = ["SBPSOit","FBreak", "Count","Repick","MSE_coef","MSE_test","ErrIt", "OldErr","ErrBC","ErrAC"]



startTime = datetime.now()

for j in ChangeP:
    for i in range(runs): 
        print("\n************** Run {0}  for Dyna - Case 1 {1} *****************\n".format(i+1,j))
        ans1 =  ALLFUNCS.DyStruct(ufunc1,ufuncs1,cfunc1,funcs1,Xfc1,Xfull1,j,x_low,x_high,xLc1,xLow1,xHc1,xHi1,k,npoints,swarmp,QP,psoswarm,QPs,max_it,psoit,MSE_condition,SSE_condition,relative_err)
        # d1.append([j,ans1])
        p1SBPSO.append([j,sum(ans1[1])/len(ans1[1])])
        p1coef.append([j,sum(ans1[4])/len(ans1[4]),sum(ans1[5])/len(ans1[5])])
        track1.append([j,ans1[2],ans1[3],sum(ans1[7])/len(ans1[7]),sum(ans1[8])/len(ans1[8]),sum(ans1[9])/len(ans1[9])])
        selectedFunc1.append(ans1[10])

        plt.plot(range(ans1[0]),ans1[1], color = "orange")
        plt.xlabel("Iterations")
        plt.ylabel("MSE of SBPSO phase")
        # plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case1/MSE_cSBPSO_{0}_{1}.jpg".format(i + 1, j))
        plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case1/MSE_SBPSO_{0}_{1}.jpg".format(i + 1, j))
        plt.clf()
        plt.plot(range(len(ans1[4])),ans1[4], color = "blue")
        plt.xlabel("Iterations")
        plt.ylabel("ABEBC of PA approximation")
        # plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case1/ABEBC_cPA_{0}_{1}.jpg".format(i + 1, j))
        plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case1/ABEBC_PA_{0}_{1}.jpg".format(i + 1, j))    
        plt.clf()
        plt.plot(range(len(ans1[5])),ans1[5], color = "green")
        plt.xlabel("Iterations")
        plt.ylabel("Test Error of PA approximation")
        # plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case1/Test_cPA_{0}_{1}.jpg".format(i + 1, j))
        plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case1/Test_PA_{0}_{1}.jpg".format(i + 1, j))    
        plt.clf()
        plt.plot(range(len(ans1[9])),ans1[9],color="red")
        plt.xlabel("Iterations")
        plt.ylabel("ABEAC of PA approximation")
        # plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case1/ABEAC_cPA_{0}_{1}.jpg".format(i + 1, j))    
        plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case1/ABEAC_PA_{0}_{1}.jpg".format(i + 1, j))            
        plt.clf()

# with open("Dyna1const.csv","w") as d1f:        
with open("Dyna1_0.4_0.95.csv","w") as d1f:
    d1_write = csv.writer(d1f)
    
    # d1_write.writerow(fields)
    # d1_write.writerows(d1)
    d1_write.writerow(["ChangeP","SBPSO_MSE"])
    d1_write.writerows(p1SBPSO)
    d1_write.writerow(["ChangeP","PSO_MSE","PSO_testErr"])
    d1_write.writerows(p1coef)
    d1_write.writerow(["ChangeP","Count","Repick","MeanOldErr/ABEBC","MeanErrBC_uni","MeanErrAc/ABEAC"])
    d1_write.writerows(track1)
    d1_write.writerow(["Selected Func"])
    d1_write.writerow(selectedFunc1)

func4 = DPoly.case2f1
func5 = DPoly.case2f2
func6 = DPoly.case2f3

ufunc4 = DPoly.ubase
ufunc5 = DPoly.ubase
ufunc6 = DPoly.ubase

xL4 = -4
xH4 = 4
Xf4 = Xfb
coef4 = [-4,2,4][::-1]

xL5 = -2
xH5 = 2
Xf5 = Xfb
coef5 = [2,-1,-2][::-1]

xL6 = -4.2
xH6 = 6
Xf6 = Xfb
coef6 = [-4.2,6,3.3][::-1]


funcs2 = [fbase,func4,func5,func6]
ufuncs2 = [ufbase,ufunc4,ufunc5,ufunc6]
xLow2 = [xLb, xL4, xL5, xL6]
xHi2 = [xHb, xH4, xH5, xH6]
Xfull2 = [Xfb, Xf4, Xf5, Xf6]
coefs2 = [coefb, coef4, coef5, coef6]

cfunc2 = funcs2[0]
ufunc2 = ufuncs2[0]
xLc2 = xLow2[0]
xHc2 = xHi2[0]
Xfc2 = Xfull2[0]
Cc2 = coefs2[0]


d2 = []
p2SBPSO = []
p2coef = []
track2 = []
selectedFunc2 = []
    
for j in ChangeP:
    for i in range(runs): 
        print("\n************** Run {0}  for Dyna - Case 2 {1} *****************\n".format(i+1,j))
        ans2 =  ALLFUNCS.DyStruct(ufunc2,ufuncs2,cfunc2,funcs2,Xfc2,Xfull2,j,x_low,x_high,xLc2,xLow2,xHc2,xHi2,k,npoints,swarmp,QP,psoswarm,QPs,max_it,psoit,MSE_condition,SSE_condition,relative_err)
        # d2.append([j,ans2])
        p2SBPSO.append([j,sum(ans2[1])/len(ans2[1])])
        p2coef.append([j,sum(ans2[4])/len(ans2[4]),sum(ans2[5])/len(ans2[5])])
        track2.append([j,ans2[2],ans2[3],sum(ans2[7])/len(ans2[7]),sum(ans2[8])/len(ans2[8]),sum(ans2[9])/len(ans2[9])])
        selectedFunc2.append(ans2[10])
        
        plt.plot(range(ans2[0]),ans2[1], color="orange")
        plt.xlabel("Iterations")
        plt.ylabel("MSE of SBPSO phase")
        # plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case2/MSE_cSBPSO_{0}_{1}.jpg".format(i + 2, j))
        plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case2/MSE_SBPSO_{0}_{1}.jpg".format(i + 2, j))
        plt.clf()
        plt.plot(range(len(ans2[4])),ans2[4],color="blue")
        plt.xlabel("Iterations")
        plt.ylabel("ABEBC of PA approximation")
        # plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case2/ABEBC_cPA_{0}_{1}.jpg".format(i + 2, j))    
        plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case2/ABEBC_PA_{0}_{1}.jpg".format(i + 2, j))            
        plt.clf()
        plt.plot(range(len(ans2[5])),ans2[5],color="green")
        plt.xlabel("Iterations")
        plt.ylabel("Tesr Error of PA approximation")
        # plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case2/Test_cPA_{0}_{1}.jpg".format(i + 2, j))    
        plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case2/Test_PA_{0}_{1}.jpg".format(i + 2, j))            
        plt.clf()
        plt.plot(range(len(ans2[9])),ans2[9],color="red")
        plt.xlabel("Iterations")
        plt.ylabel("ABEAC of PA approximation")
        # plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case2/ABEAC_cPA_{0}_{1}.jpg".format(i + 2, j))    
        plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case2/ABEAC_PA_{0}_{1}.jpg".format(i + 2, j))            
        plt.clf()


# with open("Dyna2const.csv","w") as d2f:        
with open("Dyna2_0.4_0.95.csv","w") as d2f:
    d2_write = csv.writer(d2f)
    
    # d2_write.writerow(fields)
    # d2_write.writerows(d2)
    d2_write.writerow(["ChangeP","SBPSO_MSE"])
    d2_write.writerows(p2SBPSO)
    d2_write.writerow(["ChangeP","PSO_MSE","PSO_testErr"])
    d2_write.writerows(p2coef)
    d2_write.writerow(["ChangeP","Count","Repick","MeanOldErr","MeanErrBC","MeanErrAc"])
    d2_write.writerows(track2)
    d2_write.writerow(["Selected Func"])
    d2_write.writerow(selectedFunc2)


func7 = DPoly.case3f1
func8 = DPoly.case3f2
func9 = DPoly.case3f3 

ufunc7 = DPoly.case3u1
ufunc8 = DPoly.case3u2
ufunc9 = DPoly.case3u3

d3 = []
p3SBPSO = []
p3coef = []
track3 = []
selectedFunc3 = []

xL7 = -0.5
xH7 = 6
Xf7 = {0,1,2,7}
coef7 = [-0.5,6,1,4][::-1]

xL8 = -2
xH8 = 3
Xf8 = {0,1,3,5}
coef8 = [1,-2,3,0.5][::-1]

xL9 = -5.1
xH9 = 3.7
Xf9 = {0,1,2,4,6}
coef9 = [3.7,-5.1,-0.8,-2.3,-1][::-1]

funcs3 = [fbase,func1,func2,func3,func4,func5,func6,func7,func8,func9]
ufuncs3 = [ufbase,ufunc1,ufunc2,ufunc3,ufunc4,ufunc5,ufunc6,ufunc7,ufunc8,ufunc9]
xLow3 = [xLb, xL1, xL2, xL3,xL4, xL5, xL6,xL7, xL8, xL9]
xHi3 = [xHb, xH1, xH2, xH3, xH4, xH5, xH6, xH7, xH8, xH9]
Xfull3 = [Xfb, Xf1, Xf2, Xf3, Xf4, Xf5, Xf6,Xf7, Xf8, Xf9]
coefs3 = [coefb, coef1, coef2, coef3,coef4, coef5, coef6,coef7, coef8, coef9]

cfunc3 = funcs3[0]
ufunc3= ufuncs3[0]
xLc3 = xLow3[0]
xHc3 = xHi3[0]
Xfc3 = Xfull3[0]
Cc3 = coefs3[0]

for j in ChangeP:
    for i in range(runs): 
        print("\n************** Run {0}  for Dyna - Case 3 {1} *****************\n".format(i+1,j))
        ans3 =  ALLFUNCS.DyStruct(ufunc3,ufuncs3,cfunc3,funcs3,Xfc3,Xfull3,j,x_low,x_high,xLc3,xLow3,xHc3,xHi3,k,npoints,swarmp,QP,psoswarm,QPs,max_it,psoit,MSE_condition,SSE_condition,relative_err)
        # d3.append([j,ans3])
        p3SBPSO.append([j,sum(ans3[1])/len(ans3[1])])
        p3coef.append([j,sum(ans3[4])/len(ans3[4]),sum(ans3[5])/len(ans3[5])])
        track3.append([j,ans3[2],ans3[3],sum(ans3[7])/len(ans3[7]),sum(ans3[8])/len(ans3[8]),sum(ans3[9])/len(ans3[9])])
        selectedFunc3.append(ans3[10])
        
        plt.plot(range(ans3[0]),ans3[1], color="orange")
        plt.xlabel("Iterations")
        plt.ylabel("MSE of SBPSO phase")
        # plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case3/MSE_cSBPSO_{0}_{1}.jpg".format(i + 3, j))
        plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case3/MSE_SBPSO_{0}_{1}.jpg".format(i + 3, j))
        plt.clf()
        plt.plot(range(len(ans3[4])),ans3[4],color="blue")
        plt.xlabel("Iterations")
        plt.ylabel("ABEBC of PA approximation")
        # plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case3/ABEBC_cPA_{0}_{1}.jpg".format(i + 3, j))    
        plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case3/ABEBC_PA_{0}_{1}.jpg".format(i + 3, j))            
        plt.clf()
        plt.plot(range(len(ans3[5])),ans3[5],color="green")
        plt.xlabel("Iterations")
        plt.ylabel("Tesr Error of PA approximation")
        # plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case3/Test_cPA_{0}_{1}.jpg".format(i + 2, j))    
        plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case3/Test_PA_{0}_{1}.jpg".format(i + 2, j))            
        plt.clf()
        plt.plot(range(len(ans3[9])),ans3[9],color="red")
        plt.xlabel("Iterations")
        plt.ylabel("ABEAC of PA approximation")
        # plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case3/ABEAC_cPA_{0}_{1}.jpg".format(i + 2, j))    
        plt.savefig("/Users/Donnas/Documents/Maties/iENG/2021/Skripsie/Code/Dyna/Case3/ABEAC_PA_{0}_{1}.jpg".format(i + 2, j))            
        plt.clf()


# with open("Dyna2const.csv","w") as d1f:        
with open("Dyna3_0.4_0.95.csv","w") as d3f:
    d3_write = csv.writer(d3f)
    
    # d3_write.writerow(fields)
    # d3_write.writerows(d2)
    d3_write.writerow(["ChangeP","SBPSO_MSE"])
    d3_write.writerows(p3SBPSO)
    d3_write.writerow(["ChangeP","PSO_MSE","PSO_testErr"])
    d3_write.writerows(p3coef)
    d3_write.writerow(["ChangeP","Count","Repick","MeanOldErr","MeanErrBC","MeanErrAc"])
    d3_write.writerows(track3)
    d3_write.writerow(["Selected Func"])
    d3_write.writerow(selectedFunc3)


timeEnd = datetime.now() - startTime
print(timeEnd)





