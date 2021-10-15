#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 14:35:10 2021

@author: Donnas
"""

# Skripsie functions

import random 
from random import randint
from random import shuffle
import numpy
import math
from numpy.polynomial import polynomial as P


#______________________#______________________#______________________#______________________
#CLASSES
#______________________#______________________#______________________#______________________
class Problem():
    def __init__(self,function, nx, domMin,domMax):
        #nx = number of dimensions
        self.nx = nx
        self.f = function
        self.domMin = domMin
        self.domMax = domMax

#______________________#______________________#______________________#______________________

# PSO basic (Himmelblau, Rosenbrock, Rastrigen)
#______________________#______________________#______________________#______________________


class Particle():
    def __init__(self,w,c1,c2,problem):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        # dimensions
        self.nx = problem.nx
        
       
        # random initialisation
        x = numpy.random.uniform(problem.domMin,problem.domMax,problem.nx)
        # function value for current position
        fx = problem.f(x)
        
        XP = numpy.random.uniform(problem.domMin,problem.domMax,problem.nx)
        # function value for personal best position
        fXP = problem.f(XP)
        
        # if fx greater than fXP then current position is not the personal best
        self.x = x if fx > fXP else XP
        self.fx = fx if fx > fXP else fXP
        
        # if fx smaller than fXP then current position is the personal best
        self.XP = x if fx <= fXP else XP
        self.fXP = fx if fx <= fXP else fXP
        
        # initialise personal best as global best
        self.XB = self.XP
        self.fXB = self.fXP
        
        # initialise velocity as 0
        self.v = numpy.zeros(problem.nx)
    
    def updatePosition(self):
        self.x = numpy.add(self.x,self.v)
        
            
    def updateVelocity(self):
        r1 = numpy.random.uniform(0,1,self.nx)
        r2 = numpy.random.uniform(0,1,self.nx)
        
        inertia = self.w * self.v
        cognitive = self.c1 * numpy.multiply(r1,numpy.subtract(self.XP,self.x))
        social = self.c2 * numpy.multiply(r2,numpy.subtract(self.XB,self.x))
        
        self.v = numpy.add(cognitive,social)
        self.v = numpy.add(self.v,inertia)
    def updatePersonalBest(self,problem):
        self.XP = self.x if self.fx < self.fXP else self.XP
        self.fXP = self.fx if self.fx < self.fXP else self.fXP
    # def updateBoundsPersonalBest(self,problem):
    #     if inBounds(self.x,problem):
    #         self.XP = self.x if self.fx < self.fXP else self.XP
    #         self.fXP = self.fx if self.fx < self.fXP else self.fXP
    def updateNeighborhoodBest(self,gbest,fgbest):
        self.XB = gbest
        self.fXB = fgbest

#______________________#______________________#______________________#______________________
class Swarm():
    def __init__(self,ns,w,c1,c2,problem):
        self.ns = ns
        self.position = [None] * problem.nx
        self.fvalue = None
        self.particles = []
        # add each particle in the swarm to a list
        for xi in range(0,self.ns):
            self.particles.append(Particle(w,c1,c2,problem))
   
    def updateGlobalBest(self):
        self.position = self.particles[0].XP
        self.fvalue = self.particles[0].fXP
        for xi in self.particles:
            if xi.fXP < self.fvalue:
                self.position = xi.XP
                self.fvalue = xi.fXP
    def printSolution(self):
        print("Paticle at position: {0} \nFunciton value: {1}".format(self.position,self.fvalue))
    def printSwarm(self):
        for xi in self.particles:
            print("position = ",xi.x, " (", xi.fx, ") ", " velocity = ", xi.v, " pbest = ", xi.XP, " (", xi.fXP, ")", " gbest = ",xi.XB, "(",xi.fXB,")","\n")

#______________________#______________________#______________________#______________________
class PSO():
    def __init__(self,seed,ns,problem,w,c1,c2,nt,abs_err):
        self.problem = problem
        #number of iterations
        self.nt = nt
        self.seed = seed
        numpy.random.seed(seed)
        # create swarm with ns particles
        self.swarm = Swarm(ns,w,c1,c2,problem)
        self.fBreak = []
        self.abs_err = abs_err
    
    def runPSO(self):
        self.t = 1
        while (self.t <= self.nt):
            self.swarm.updateGlobalBest()
            for xi in self.swarm.particles:
                xi.updateNeighborhoodBest(self.swarm.position,self.swarm.fvalue)
            for xi in self.swarm.particles:
                xi.updateVelocity()
                xi.updatePosition()
                xi.fx = self.problem.f(xi.x)
                xi.updatePersonalBest(self.problem) 
                #xi.updateBoundsPersonalBest(self.problem) 
                
            self.fBreak.append(xi.fx)    
            # print("fBreak",self.fBreak)
            
    
    # break the code if funciton value doesnt change
            if(self.t > (int(0.3*self.nt))):
                # self.fBreaker = self.fBreak[int(0.3*self.nt)-1:self.t:1]
                self.d = []
                for j in range(int(0.3*self.nt),self.t):
                    # print("fprint", j)
                    self.d.append(abs(float(self.fBreak[j]) - float(self.fBreak[j-1])))
                    
                self.AE = sum(self.d)/len(self.d) 
            
            
                # print("relD {0} @ iter{1}".format(self.d, self.t))
                # print("fREL E:",self.AE)

                if(self.AE < self.abs_err):
                    # print("\nFunction value did not improve after multiple iterations")
                    # print("Algorithm has stopped at iteraton: {0} with an absolute error of: {1}".format(self.t, self.AE))
                    break                     
        
            # print("Diff :",self.relativeDifference)
            # print("Iteration {0} :".format(self.t,self.swarm.fvalue))
            #self.printSolutionQuality()
            self.t += 1
        for xi in self.swarm.particles:
            xi.updateNeighborhoodBest(self.swarm.position,self.swarm.fvalue)
        self.swarm.updateGlobalBest()
   
    def printSolutionQuality(self):
        print(self.swarm.fvalue)
    def printSolution(self):
        print("\nAlgorithm stopped\n")
        self.swarm.printSolution()
    def returnSolution(self):
        self.sol = [self.swarm.position,self.swarm.fvalue]
        return self.sol
  
#______________________#______________________#______________________#______________________
# PSO coefficients
#______________________#______________________#______________________#______________________

class ParticleC():
    def __init__(self,w,c1,c2,problem,X,xset,cfunc,pen):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        # dimensions
        self.nx = problem.nx
        self.X = X
        self.xset = xset
        self.problem = problem
        self.cfunc = cfunc
        self.pen = pen
       
        # random initialisation
        x = numpy.random.uniform(self.problem.domMin,self.problem.domMax,self.problem.nx)
        # function value for current position
        fx = self.problem.f(x, self.X ,self.xset, self.cfunc,self.pen)
        
        XP = numpy.random.uniform(self.problem.domMin,self.problem.domMax,self.problem.nx)
        # function value for personal best position
        fXP = self.problem.f(XP,self.X ,self.xset, self.cfunc, self.pen)
               
        
        
        # if fx greater than fXP then current position is not the personal best
        self.x = x if fx > fXP else XP
        self.fx = fx if fx > fXP else fXP
        
        # if fx smaller than fXP then current position is the personal best
        self.XP = x if fx <= fXP else XP
        self.fXP = fx if fx <= fXP else fXP
        
        # initialise personal best as global best
        self.XB = self.XP
        self.fXB = self.fXP
        
        # initialise velocity as 0
        self.v = numpy.zeros(self.nx)
    
    def updatePosition(self):
        self.x = numpy.add(self.x,self.v) 
            
    def updateVelocity(self):
        r1 = numpy.random.uniform(0,1,self.nx)
        r2 = numpy.random.uniform(0,1,self.nx)
        
        inertia = self.w * self.v
        cognitive = self.c1 * numpy.multiply(r1,numpy.subtract(self.XP,self.x))
        social = self.c2 * numpy.multiply(r2,numpy.subtract(self.XB,self.x))
        
        self.v = numpy.add(cognitive,social)
        self.v = numpy.add(self.v,inertia)
    # def updatePersonalBest(self,problem):
    #     self.XP = self.x if self.fx < self.fXP else self.XP
    #     self.fXP = self.fx if self.fx < self.fXP else self.fXP
    def updateBoundsPersonalBest(self,problem):
        if inBounds(self.x,problem):
            self.XP = self.x if self.fx < self.fXP else self.XP
            self.fXP = self.fx if self.fx < self.fXP else self.fXP
    
    def updateNeighborhoodBest(self,gbest,fgbest):
        self.XB = gbest
        self.fXB = fgbest
        

    
#______________________#______________________#______________________#______________________
class SwarmC():
    def __init__(self,ns,w,c1,c2,problem,X,xset,cfunc,pen):
        self.ns = ns
        self.position = [None] * problem.nx
        self.fvalue = None
        self.particles = []
        self.cfunc = cfunc
        self.pen = pen
        # add each particle in the swarm to a list
        for xi in range(0,self.ns):
            self.particles.append(ParticleC(w,c1,c2,problem,X,xset,cfunc,pen))
   
    def updateGlobalBest(self):
        self.position = self.particles[0].XP
        self.fvalue = self.particles[0].fXP
        for xi in self.particles:
            if xi.fXP < self.fvalue:
                self.position = xi.XP
                self.fvalue = xi.fXP
    def printSolution(self):
        # print("Paticle at position: {0} \nFunciton value: {1}".format(self.position,self.fvalue))
        print("Paticle at position: {0}".format(self.position))
    def printSwarm(self):
        for xi in self.particles:
            print("position = ",xi.x, " (", xi.fx, ") ", " velocity = ", xi.v, " pbest = ", xi.XP, " (", xi.fXP, ")", " gbest = ",xi.XB, "(",xi.fXB,")","\n")
    
  
class PSOC():
    def __init__(self,seed,ns,problem,X,xset,cfunc,pen,w,c1,c2,nt,SSErr,breaker):
        self.problem = problem
        self.X = X
        self.xset = xset
        self.cfunc = cfunc
        self.pen = pen
        #number of iterations
        self.nt = nt
        self.seed = seed
        numpy.random.seed(seed)
        # create swarm with ns particles
        self.swarm = SwarmC(ns,w,c1,c2,problem,self.X,self.xset,self.cfunc, self.pen)
        self.fBreak = []
        self.SSErr = SSErr
        self.breaker = breaker
        
    def runPSO(self):
        self.t = 1
        self.swarm.fvalue = float('Inf')
        
        
        while ((self.t <= self.nt)): #& (self.swarm.fvalue > self.SSErr)):    
            # print("Iteration {0}".format(self.t))
            self.swarm.updateGlobalBest()
            for xi in self.swarm.particles:
                xi.updateNeighborhoodBest(self.swarm.position,self.swarm.fvalue)
            for xi in self.swarm.particles:
                xi.updateVelocity()
                xi.updatePosition()
                # xi.limLow()
                # xi.limHigh()
                # xi.limitToBounds(self.problem)
                xi.fx = self.problem.f(xi.x,self.X,self.xset,self.cfunc,self.pen)
                # xi.updatePersonalBest(self.problem) 
                xi.updateBoundsPersonalBest(self.problem)
                
                
            #save MSE    
            self.fBreak.append((xi.fx)/len(self.xset))    
            #print("fBreak",self.fBreak)
            
    
    # break the code if funciton value doesnt change
            if(self.t > (int(0.3*self.nt))):
                # self.fBreaker = self.fBreak[int(0.3*self.nt)-1:self.t:1]
                self.dif = []
                
                for j in range(self.t-int(0.1*self.nt),self.t):
                    # print("fprint", j)
                    self.dif.append(abs(float(self.fBreak[j]) - float(self.fBreak[j-1])))
                    # print("Relative diff: {0}".format(self.dif))
                self.MAE = sum(self.dif)/len(self.dif) 
                # print("MAE: {0}".format(self.MAE))
        
                if(self.MAE < self.SSErr):
                    print("\nFunction value did not improve after multiple iterations")
                    print("Algorithm has stopped at iteraton: {0} with a SSE of: {1} \nRelative diff MAE :{2}".format(self.t, self.swarm.fvalue,self.MAE))
                    break                     
        
                # print("Diff :",self.dif)
                # print("SSE :",self.MAE)
            print("Iteration {0} + SSE {1}:".format(self.t,self.swarm.fvalue))
            #self.printSolutionQuality()
            
            # Break from while loop if below SSE
            if(abs(self.swarm.fvalue - self.pen*self.breaker) <= self.SSErr):
                print("Algorithm has stopped at iteraton: {0} with a SSE of: {1}".format(self.t, self.swarm.fvalue))
                break
            
            self.t += 1
        for xi in self.swarm.particles:
            xi.updateNeighborhoodBest(self.swarm.position,self.swarm.fvalue)
        self.swarm.updateGlobalBest()
   
    def printSolutionQuality(self):
        print(self.swarm.fvalue)
    def printSolution(self):
        print("\nAlgorithm stopped\n")
        self.swarm.printSolution()
    def returnSolution(self):
        self.sol = [self.swarm.position,self.swarm.fvalue,self.t]
        return self.sol
    def returnPoly(self):
        print("\nPolynomial:\n {0}".format(numpy.poly1d(self.swarm.position)))

#______________________#______________________#______________________#______________________

class ParticleQ():
    def __init__(self,problem,X,xset,cfunc):
        self.w = 0.7
        self.c1 = 1.4
        self.c2 = 1.4
        # dimensions
        self.nx = problem.nx
        self.X = X
        self.xset = xset
        self.problem = problem
        self.cfunc = cfunc
        self.pen = 0
       
        # random initialisation
        x = numpy.random.uniform(self.problem.domMin,self.problem.domMax,self.problem.nx)
        # function value for current position
        fx = self.problem.f(x, self.X ,self.xset, self.cfunc,self.pen)
        
        XP = numpy.random.uniform(self.problem.domMin,self.problem.domMax,self.problem.nx)
        # function value for personal best position
        fXP = self.problem.f(XP,self.X ,self.xset, self.cfunc, self.pen)
               
        
        
        # if fx greater than fXP then current position is not the personal best
        self.x = x if fx > fXP else XP
        self.fx = fx if fx > fXP else fXP
        
        # if fx smaller than fXP then current position is the personal best
        self.XP = x if fx <= fXP else XP
        self.fXP = fx if fx <= fXP else fXP
        
        # initialise personal best as global best
        self.XB = self.XP
        self.fXB = self.fXP
        
        # initialise velocity as 0
        # self.v = numpy.zeros(self.nx)
    
    # def updatePositionUni(self):
    #     # self.x = numpy.add(self.x,self.v) 
    #     x = numpy.random.uniform(0,self.XB)
    
        
    
    # def updatePositionNorm(self):
    #     # self.x = numpy.add(self.x,self.v) 
    #     r = rcloud()
    #     d = diversity()
    #     r = 1
    #     x = numpy.random.normal(self.XP,r)
            
    # def updateVelocity(self):
    #     r1 = numpy.random.uniform(0,1,self.nx)
    #     r2 = numpy.random.uniform(0,1,self.nx)
        
    #     inertia = self.w * self.v
    #     cognitive = self.c1 * numpy.multiply(r1,numpy.subtract(self.XP,self.x))
    #     social = self.c2 * numpy.multiply(r2,numpy.subtract(self.XB,self.x))
        
    #     self.v = numpy.add(cognitive,social)
    #     self.v = numpy.add(self.v,inertia)
    # def updatePersonalBest(self,problem):
    #     self.XP = self.x if self.fx < self.fXP else self.XP
    #     self.fXP = self.fx if self.fx < self.fXP else self.fXP
    def updateBoundsPersonalBest(self,problem):
        if inBounds(self.x,problem):
            self.XP = self.x if self.fx < self.fXP else self.XP
            self.fXP = self.fx if self.fx < self.fXP else self.fXP
    
    def updateNeighborhoodBest(self,gbest,fgbest):
        self.XB = gbest
        self.fXB = fgbest
        

    
#______________________#______________________#______________________#______________________
class SwarmQ():
    def __init__(self,ns,qs,w,c1,c2,problem,X,xset,cfunc,pen):
        self.ns = ns
        self.qs = qs
        self.position = [None] * problem.nx
        self.fvalue = None
        self.particles = []
        self.cfunc = cfunc
        self.pen = pen
        # add each particle in the swarm to a list
        for xi in range(0,self.ns):
            self.particles.append(ParticleC(w,c1,c2,problem,X,xset,cfunc,pen))
   
        # for xq in range(0,self.qs):
            
            
    def updateGlobalBest(self):
        self.position = self.particles[0].XP
        self.fvalue = self.particles[0].fXP
        for xi in self.particles:
            if xi.fXP < self.fvalue:
                self.position = xi.XP
                self.fvalue = xi.fXP
    def printSolution(self):
        # print("Paticle at position: {0} \nFunciton value: {1}".format(self.position,self.fvalue))
        print("Paticle at position: {0}".format(self.position))
    def printSwarm(self):
        for xi in self.particles:
            print("position = ",xi.x, " (", xi.fx, ") ", " velocity = ", xi.v, " pbest = ", xi.XP, " (", xi.fXP, ")", " gbest = ",xi.XB, "(",xi.fXB,")","\n")
    
  
class PSOQ():
    def __init__(self,seed,ns,problem,X,xset,cfunc,nt,SSErr,breaker):
        self.problem = problem
        self.X = X
        self.xset = xset
        self.cfunc = cfunc
        self.pen = 0
        #number of iterations
        self.nt = nt
        self.seed = seed
        numpy.random.seed(seed)
        # create swarm with ns particles
        self.swarm = SwarmC(ns,0.7,1.4,1.4,problem,self.X,self.xset,self.cfunc, self.pen)
        self.fBreak = []
        self.SSErr = SSErr
        self.breaker = breaker
        
    def runPSO(self):
        self.t = 1
        self.swarm.fvalue = float('Inf')
        
        
        while ((self.t <= self.nt)): #& (self.swarm.fvalue > self.SSErr)):    
            # print("Iteration {0}".format(self.t))
            self.swarm.updateGlobalBest()
            for xi in self.swarm.particles:
                xi.updateNeighborhoodBest(self.swarm.position,self.swarm.fvalue)
            for xi in self.swarm.particles:
                xi.updateVelocity()
                xi.updatePosition()
                # xi.limLow()
                # xi.limHigh()
                # xi.limitToBounds(self.problem)
                xi.fx = self.problem.f(xi.x,self.X,self.xset,self.cfunc,self.pen)
                # xi.updatePersonalBest(self.problem) 
                xi.updateBoundsPersonalBest(self.problem)
                
                
            #save MSE    
            self.fBreak.append((xi.fx)/len(self.xset))    
            #print("fBreak",self.fBreak)
            
    
    # break the code if funciton value doesnt change
            if(self.t > (int(0.3*self.nt))):
                # self.fBreaker = self.fBreak[int(0.3*self.nt)-1:self.t:1]
                self.dif = []
                
                for j in range(self.t-int(0.1*self.nt),self.t):
                    # print("fprint", j)
                    self.dif.append(abs(float(self.fBreak[j]) - float(self.fBreak[j-1])))
                    # print("Relative diff: {0}".format(self.dif))
                self.MAE = sum(self.dif)/len(self.dif) 
                print("MAE: {0}".format(self.MAE))
        
                if(self.MAE < self.SSErr):
                    print("\nFunction value did not improve after multiple iterations")
                    print("Algorithm has stopped at iteraton: {0} with a SSE of: {1} \nRelative diff MAE :{2}".format(self.t, self.swarm.fvalue,self.MAE))
                    break                     
        
                # print("Diff :",self.dif)
                # print("SSE :",self.MAE)
            print("Iteration {0} + SSE {1}:".format(self.t,self.swarm.fvalue))
            #self.printSolutionQuality()
            
            # Break from while loop if below SSE
            if(abs(self.swarm.fvalue - self.pen*self.breaker) <= self.SSErr):
                print("Algorithm has stopped at iteraton: {0} with a SSE of: {1}".format(self.t, self.swarm.fvalue))
                break
            
            self.t += 1
        for xi in self.swarm.particles:
            xi.updateNeighborhoodBest(self.swarm.position,self.swarm.fvalue)
        self.swarm.updateGlobalBest()
   
    def printSolutionQuality(self):
        print(self.swarm.fvalue)
    def printSolution(self):
        print("\nAlgorithm stopped\n")
        self.swarm.printSolution()
    def returnSolution(self):
        self.sol = [self.swarm.position,self.swarm.fvalue,self.t]
        return self.sol
    def returnPoly(self):
        print("\nPolynomial:\n {0}".format(numpy.poly1d(self.swarm.position)))


#______________________#______________________#______________________#______________________    
# FUNCTIONS
#______________________#______________________#______________________#______________________

def HIMMELBLAU(x):
    return(((x[0]**2 + x[1] -11)**2 + (x[0] + x[1]**2 - 7)**2))

def ROSENBROCK(x, a = 1, b = 100):
    part1 = (a-x[0])**2
    part2 = (x[1]-x[0]**2)**2
    return part1 + b*part2

def RASTRIGIN(x,A = 10):
    n = len(x)
    sums = 0
    for xi in x:
        sums += (xi**2 - A*numpy.cos(2*numpy.pi*xi))
    return A*n + sums

#______________________#______________________#______________________#______________________
# For uni SBPSO
#______________________#______________________#______________________#______________________
def fitness(x,X,y):
   
    fx = [0 for x in range(len(x))]
    sqr = [0 for x in range(len(x))]
    
    # for each point 
    for i in range(len(x)):
        
        # calculate polynomial value
        for p in X:
            fx[i] =  fx[i] + x[i]**p    
    
        # calculate square error
        sqr[i] = (fx[i]-y[i])**2
    
    # return MSE
    return(sum(sqr)/len(sqr))   
        
#______________________#______________________#______________________#______________________
# For coef SBPSO
#______________________#______________________#______________________#______________________

def fitness2(x,X,y,C):
   
    fx = [0 for x in range(len(x))]
    sqr = [0 for x in range(len(x))]
    # print(X)
    # print(C)
    Xlist = list(X)
    # Clist = list(C)
    # for i in X:
    #     Xlist.append(i)
    if isinstance(C,int):
        Clist = [C]
    else:
        Clist = list(C)
    
    # print("Length of x : {0}\nC: {1}\nX: {2}\n".format(range(len(x)),Clist,Xlist))       
    # for each point 
    for i in range(len(x)):
        for p in range(len(Xlist)):
                fx[i] =  fx[i] + Clist[p]*x[i]**Xlist[p]    
    
        # calculate square error
        sqr[i] = (fx[i]-y[i])**2
    
    # return MSE
    return(sum(sqr)/len(sqr))   
  
#______________________#______________________#______________________#______________________      
# Random generators
#______________________#______________________#______________________#______________________

def rd():
    return random.uniform(0,1)

def SEED():
    return int(10*rd()*rd()*10)

#______________________#______________________#______________________#______________________
# For all SBPSO
#______________________#______________________#______________________#______________________

def MINUS(x_end, x_start):
    nrem = list()
    # add positions in x1 not in x2   
    add = x_end.difference(x_start)
    
    # remove positions in x2 not in x1
    rem = x_start.difference(x_end)
    
    # change coefficients to negative {-}
    
    # NOT SURE IF NEGATIVE VALUES SHOULD BE CREATED
    # LOOP IS CREATED TO ADD THE VALUES WITH NEGATIVE COEFFICIENT
    # IF A NEGATIVE I NEEDS TO BE ADDED, ADD A {-} IN THE BELOW LOOP
    for i in rem:
        nrem.append((-i)) 
    
    return add|set(nrem)

#______________________#______________________#______________________#______________________
# Multiplication of velocity by a scaler
# Samples n random elements from V 
def MULTI(n,V):
  
  numb = math.floor(n*len(V))
  newV = set(random.sample(list(V),numb))
  
  return newV

#______________________#______________________#______________________#______________________
def spes_Add(n,X,A):

  floorBeta = math.floor(n)
  val1 = len(A)
  r = rd()
  
  if(r < (n - floorBeta)):
    val2 = floorBeta + 1
  else:
    val2 = floorBeta 
  
  NBetaA = min(val1,val2)
  
  AddA = set(random.sample(list(A),NBetaA))
  newX = X | AddA
  
  return(newX)

#______________________#______________________#______________________#______________________
# For uni SBPSO
#______________________#______________________#______________________#______________________

def k_tourn(k,A,N,X,x,y):
    
    # k length <= |A|
    kk = min(len(A),k)
    
    e = [0 for x in range(kk)]   
    Xcheck = [set() for x in range(1,kk+1)]   
    score = [0 for x in range(kk)]   
   
    Vtemp = set()
    
    # select N elements
    for i in range(1,N+1):
    
        # start tournament
        for j in range(1,kk+1):
            
            # randomly select j elements from A
            e[j-1] = set(random.choices(list(A),k = j))
            # add elements to X
            Xcheck[j-1] = X|e[j-1]
        
            # evaluate fitness of added element    
            score[j-1] = fitness(x,Xcheck[j-1],y)
        
        # Return position of best fitness, ie. lowest MSE
        select = numpy.argmin(score)
        
        Vtemp = Vtemp|Xcheck[select-1]
        
    return Vtemp 

#______________________#______________________#______________________#______________________
def tourn_Add(n,X,A,k,x,y):

  floorBeta = math.floor(n)
  val1 = len(A)
  r = rd()
  
  if(r < (n - floorBeta)):
    val2 = floorBeta + 1
  else:
    val2 = floorBeta 
  
  NBetaA = min(val1,val2)
  
  # add elements via tournament selection
  AddA = k_tourn(k,A,NBetaA,X,x,y)
  
  newX = X | AddA
  
  return(newX)
#______________________#______________________#______________________#______________________
# For coef SBPSO
#______________________#______________________#______________________#______________________

def k_tourn2(k,A,N,X,x,y,C):
    
    # k length <= |A|
    kk = min(len(A),k)

    e = [0 for x in range(kk)]   
    Xcheck = [X for x in range(kk)]   
    Ccheck = [C for x in range(kk)]
    score = [0 for x in range(kk)]   
    
    
    Vtemp = set()
    
    # select N elements
    for i in range(N):
    # for i in range(2):
        Ccheck = [C for x in range(kk)]
        # start tournament
        for j in range(kk):
            
            # randomly select j elements from A
            e[j] = set(random.sample(list(A),k = (j+1)))
            # add elements to X
            Xcheck[j] = X|e[j]
            
            for t in range(len(Xcheck[j])-len(X)):
                Ccheck[j].append(1)
            print(Ccheck[j])
                       
            # evaluate fitness of added element    
            # print("X: {} Y: {}".format(X,C))    
            score[j] = fitness2(x,Xcheck[j],y,Ccheck[j])
        
    # Return position of best fitness, ie. lowest MSE
        select = numpy.argmin(score)
    
        Vtemp = Vtemp|Xcheck[select]
            
    return Vtemp 
            

#______________________#______________________#______________________#______________________
def tourn_Add2(n,X,A,k,x,y,C):

  floorBeta = math.floor(n)
  val1 = len(A)
  r = rd()
  if isinstance(C,int):
        C = [C]
  else:
        C = list(C)
        
  if(r < (n - floorBeta)):
    val2 = floorBeta + 1
  else:
    val2 = floorBeta 
  
  NBetaA = min(val1,val2)
  
  # add elements via tournament selection
  AddA = k_tourn2(k,A,NBetaA,X,x,y,C)
  
  newX = X | AddA
  
  return(newX)

#______________________#______________________#______________________#______________________
# For all SBPSO
#______________________#______________________#______________________#______________________

def spes_Rem(n,X,S):
  
  floorBeta = math.floor(n)
  val1 = len(S)
  r = rd()
  
  if(r < (n - floorBeta)):
    val2 = floorBeta + 1
  else:
    val2 = floorBeta 
  
  NBetaS = min(val1,val2)
  
  remS = set(random.sample(list(S),NBetaS))
  newX = X - remS 
  
  return(newX)

#______________________#______________________#______________________#______________________
# For coef SBPSO
#______________________#______________________#______________________#______________________

# def add_Coef(X):
#     clen = []
#     c = [i for i in range(len(X))] 
#     for m in X:
#         # print(m)
#         clen.append(len(m))
    
#     # add 1 for each term in X
#     for i in range(len(X)):
#         c[i] = [1 for cc in range(clen[i])]
    
#     return c

def add_COEF(X):
    c = []
    # add 1 for each term in X
    for i in range(len(X)):
        c.append(1)
    
    return c


# Evaluate coef func
def c_funcpen(C,X,xset,cfunc,pen):
    CRev = C[::-1]
    yh = []

    ytr = cfunc(xset) 
    for i in xset:
        total = 0
        SSC = 0
    
        #for each element
        for t in range(len(X)):
            total = total + CRev[t] * i ** list(X)[t]
      
        yh.append(total)
    
        for c in range(len(C)):
            SSC = SSC + CRev[c]**2
    
    SSEpen = sum((yh -ytr)**2) + pen * SSC
    # AE = sum(abs(yh-ytr))
    return SSEpen
    

#______________________#______________________#______________________#______________________    
# For uni SBPSO
#______________________#______________________#______________________#______________________

def SBPSO(func,MD,c1,c2,c3,c4,k,npoints,swarmp,x_low,x_high,max_it,MSE_condition,relative_err):
    
    # create universe
    U = set(range(MD+1))
    
    # create space for variables
    X = [set() for x in range(swarmp)]
    XP =  [set() for x in range(swarmp)]
    XB = set()
        
    V =  [set() for x in range(swarmp)] 
    
    fX =  [set() for x in range(swarmp)]
    
    # initialise personal and global best values to inifinity (for a min problem)
    fXP = [float('inf') for x in range(swarmp)]
    fXB = float('inf')
    
    fBreak = []
   
    savePB = []    
    
    newk = []
    
    for i in range(swarmp):
        newk.append(k)    
    
    # create random dataset
    xset = numpy.random.uniform(x_low, x_high, npoints)
    
    # randomise xset postions
    shuffle(xset)
    
    trainset = xset[range(int(0.7*npoints))]
    ytrain = func(trainset)
    
    testset = xset[range(int(0.7*npoints),npoints)]
    ytest = func(testset)
    
    how = "converge"
    
    
    # create particle list
    for i in range(swarmp):
        # random selector
        nx = randint(1,MD+1)
        
        # initialise particle
        X[i] = set(random.sample(list(U),nx))
    
            
        # fitness of polynomial is the MSE
        fX[i] = fitness(trainset,X[i],ytrain)   
        
           
    # start search
    
    it = 0
    relativeDifference = 1
    reInt = 1
    
    # still need to add MSE condition
    while ((fXB > MSE_condition) & (it < max_it)):        
        
        for i in range(swarmp):
            # if the current position has a better minimum, update personal best    
            if fX[i] < fXP[i]:
                fXP[i] = fX[i]
                XP[i] = X[i] 
            
            # if the persobal best has a better minimum, update global best
            if fXP[i] < fXB:
                fXB = fXP[i]
                XB = XP[i]
                ki = i
        
        # track PB
        savePB.append(XP)     

        # dynamic k - increment by one if PB doenst change
        if (it > 0.1*max_it):
            for i in range(swarmp):   
                if(savePB[it][i] == savePB[it-1][i]):
                    newk[i] += 1
                    # print("plus1")
                    if newk[i] > (MD-2):
                        newk[i] = MD-2
                else:
                    # savek[i] = newk[i]
                    newk[i] = k
                    # print('reset k')
                               
       
            
                
        # update equations
        for i in range(swarmp):
            
            #update velocity
            s1 = MINUS(XP[i],X[i])
            s2 = MINUS(XB,X[i])
            
            A = X[i] | XP[i] | XB
            S = X[i] & XP[i] & XB
                
            s3 = MULTI(c1*rd(),s1)
            s4 = MULTI(c2*rd(),s2)
           
            s5 = tourn_Add(c3*rd(), X[i], A, newk[i], trainset, ytrain)
            s6 = spes_Rem(c4*rd(), X[i], S)   
            
            
            V[i] = s3 | s4 | s5 | s6
            Vlist = list(V[i])
            
            # update postion
            # union between X and V
            for j in range(len(Vlist)):
                if (Vlist[j] < 0):
                    X[i] = X[i] - {abs(Vlist[j])}
                elif (Vlist[j] >= 0): 
                    X[i] = X[i] | {Vlist[j]}
            
            
            # fitness of polynomial is the MSE
            fX[i] = fitness(trainset,X[i],ytrain)
            
            # track glbal best value
        fBreak.append(fXB)
            
        #Debugger for the bias term
        if(it >= (int(0.05*max_it))):
            # print("test debugger {0}".format(fXB))
            if((round(fXB) == 1) and 0 in XB):
                # print("test debugger - 0")
                XB = XB - {0}
                # print("pop test")
                fXB = fitness(trainset,XB,ytrain)
                how = "pop"
            elif(round(fXB) == 1):
                XB = XB | {0}
                fXB = fitness(trainset,XB,ytrain)
                how = "pull"
                
            # Additional stopping criteria if MSE doesn't change for at least 50% of iterations
        if(it >= (int(0.1*max_it))):
            diff = 0
            for j in range(it-(int(0.05*max_it)),it):
                diff = diff + abs(float(fBreak[j+1]) - float(fBreak[j]))
                # print(diff/it)
            relativeDifference = diff/it 
            # print(diff/it)
            if(((relativeDifference) < relative_err) & it >= (int(0.5*max_it))):
                ftest = fitness(testset,XB,ytest) 
                print("\nMSE did not change for multiple iterations")
                print("Algorithm has stopped at iteraton: {0}".format(it+1))
                print("\nFitness (MSE): {0} \nBest polynomial is: {1}".format(fXB,XB))
                return [it,XB,fXB,"break",newk[ki],ftest]
                
             #Debugger trapped search
        if((reInt == 1) & (it >= (int(0.1*max_it))) & (it == round((int(0.35*max_it))) or ((relativeDifference) < relative_err))):
            reInt = 0
            how = "re-int"
            fXP = [float('inf') for x in range(swarmp)]
            fXB = float('inf')
        
            for i in range(swarmp):
                # random selector
                nx = randint(1,MD+1)
    
                # initialise particle
                X[i] = set(random.sample(list(U),nx))
            
                
                # fitness of polynomial is the MSE
                fX[i] = fitness(trainset,X[i],ytrain)  
        
            # test
        print("\nIteration: {0}, MSE: {1} \nPosition: {2}".format(it+1, fXB, XB))
        it += 1
    print("\nAlgorithm has stopped \nBest polynomial set is: {0} \nNumber of iterations: {1} \nFitness (MSE): {2}".format(XB,it,fXB))        
    
    # Determin test error
    ftest = fitness(testset,XB,ytest)   
    
    return [it,XB,fXB,how,newk[ki],ftest]

#______________________#______________________#______________________#______________________    
def SBPSO_combo(ufunc,cfunc,MD,c1,c2,c3,c4,k,w,c_1,c_2,pen,npoints,swarmp,psoswarm,x_low,x_high,xL,xH,max_it,psoit,MSE_condition,SSE_condition,relative_err,breaker):
    
    # create universe
    U = set(range(MD+1))
    
    # create space for variables
    X = [set() for x in range(swarmp)]
    XP =  [set() for x in range(swarmp)]
    XB = set()
    
    # CXB = []
    
        
    V =  [set() for x in range(swarmp)] 
    
    fX =  [float('inf') for x in range(swarmp)]
    # fCX =  [float('inf') for x in range(swarmp)]
    
    # initialise personal and global best values to inifinity (for a min problem)
    fXP = [float('inf') for x in range(swarmp)]
    # fCXP = [float('inf') for x in range(swarmp)]
    fXB = float('inf')
    fCXB = float('inf')
    
    fBreak = []
    # fCBreak = []
    savePB = []    
    
    newk = []
    
    for i in range(swarmp):
        newk.append(k)    
    
        
        
    
    # create random dataset
    xset = numpy.random.uniform(x_low, x_high, npoints)
    
    # randomise xset postions
    shuffle(xset)
    
    trainset = xset[range(int(0.7*npoints))]
    ytrain = ufunc(trainset)
    # ytrainc = cfunc(trainset)
    
    testset = xset[range(int(0.7*npoints),npoints)]
    # ytest = cfunc(testset)
    
    how = "converge"
    
    
    # create particle list
    for i in range(swarmp):
        # random selector
        nx = randint(1,MD+1)
        
        # initialise particle
        X[i] = set(random.sample(list(U),nx))
    
    # add 1s to each X term
    # CX = add_Coef(X)
    
     
    for i in range(swarmp):
        # fitness of polynomial is the MSE
        fX[i] = fitness(trainset,X[i],ytrain)   
        
        
           
    # start search
    
    it = 0
    relativeDifference = 1
    reInt = 1
    
    # SBPSO while loop
    while ((fXB > MSE_condition) & (it < max_it)):        
        
        for i in range(swarmp):
            # if the current position has a better minimum, update personal best    
            if fX[i] < fXP[i]:
                fXP[i] = fX[i]
                XP[i] = X[i]
                # CXP[i] = CX[i] 
            
            # if the persobal best has a better minimum, update global best
            if fXP[i] < fXB:
                fXB = fXP[i]
                XB = XP[i]
                ki = i
                # CXB = CXP[i]
            
        #track PB
        savePB.append(XP)     
    
        # dynamic k - increment by one if PB doenst change
        if (it > 0.1*max_it):
            for i in range(swarmp):   
                if(savePB[it][i] == savePB[it-1][i]):
                    newk[i] += 1
                    # print("plus1")
                    if newk[i] > (MD-2):
                        newk[i] = MD-2
                else:
                    # savek[i] = newk[i]
                    newk[i] = k
                    # print('reset k')
                               
       
            
                
    # update equations
        for i in range(swarmp):
            
            #update velocity
            s1 = MINUS(XP[i],X[i])
            s2 = MINUS(XB,X[i])
            
            A = X[i] | XP[i] | XB
            S = X[i] & XP[i] & XB
                
            s3 = MULTI(c1*rd(),s1)
            s4 = MULTI(c2*rd(),s2)
    
            s5 = tourn_Add(c3*rd(), X[i], A, newk[i], trainset, ytrain)
            s6 = spes_Rem(c4*rd(), X[i], S)   
            
    
            
            V[i] = s3 | s4 | s5 | s6
            Vlist = list(V[i])
            # update postion
            # union between X and V
            for j in range(len(Vlist)):
                if (Vlist[j] < 0):
                    X[i] = X[i] - {abs(Vlist[j])}
                elif (Vlist[j] >= 0): 
                    X[i] = X[i] | {Vlist[j]}
            
            
            # fitness of polynomial is the MSE
            fX[i] = fitness(trainset,X[i],ytrain)
            
        fBreak.append(fXB)
            
        #Debugger for bias term
        if(it >= (int(0.05*max_it))):
            # print("test debugger {0}".format(fXB))
            if((round(fXB) == 1) and 0 in XB):
                # print("test debugger - 0")
                XB = XB - {0}
                # CXB = [1 for i in range(len(XB))]
                # print("pop test")
                fXB = fitness(trainset,XB,ytrain)
                how = "pop"
            elif(round(fXB) == 1):
                XB = XB | {0}
                # CXB = [1 for i in range(len(XB))]
                fXB = fitness(trainset,XB,ytrain)
                how = "pull"
                
            # Additional stopping criteria if MSE doesn't change for at least 50% of iterations
        if(it >= (int(0.1*max_it))):
            diff = 0
            for j in range(it-(int(0.05*max_it)),it):
                diff = diff + abs(float(fBreak[j+1]) - float(fBreak[j]))
                # print(diff/it)
            relativeDifference = diff/it 
            # print(diff/it)
            if(((relativeDifference) < relative_err) & it >= (int(0.5*max_it))):
            # if((relativeDifference) < relative_err):
                print("\nMSE did not change for multiple iterations")
                print("SPBSO algorithm has stopped at iteraton: {0}".format(it+1))
                print("\nFitness (MSE): {0} \nBest polynomial is: {1}".format(fXB,XB))
                break
                # return [it,XB,fXB,"break",newk[ki]]
        
            #Debugger trapped search
        if((reInt == 1) & (it >= (int(0.1*max_it))) & (it == round((int(0.35*max_it))) or ((relativeDifference) < relative_err))):
            reInt = 0
            how = "re-int"
            fXP = [float('inf') for x in range(swarmp)]
            fXB = float('inf')
        
            for i in range(swarmp):
                # random selector
                nx = randint(1,MD+1)
    
                # initialise particle
                X[i] = set(random.sample(list(U),nx))
            
                
                # fitness of polynomial is the MSE
                fX[i] = fitness(trainset,X[i],ytrain)         
            
        
            # test
        print("\nIteration: {0}, MSE: {1} \nPosition: {2}".format(it+1, fXB, XB))
        it += 1
    print("\n SBPSO Algorithm has stopped \nBest polynomial set is: {0} \nNumber of iterations: {1} \nFitness (MSE): {2}".format(XB,it,fXB))        
    SBPSOit = it
    
    CB = add_COEF(XB)
        #PSO while loop to find coefficients of best found structure
    
    itPSO = 0
    testMSE = float('inf')
    
    if(round(fXB) < 1):
        # xt = numpy.random.uniform(xL,xH,int(npoints/2))
        # while ((fCXB > SSE_condition) & (it < 0.1*max_it)):        
        
        
        prb = Problem(c_funcpen,len(XB),xL,xH)
        
        partCOEF = PSOC(SEED(),psoswarm,prb,XB,trainset,cfunc,pen,w,c_1,c_2,psoit,relative_err,breaker)
        partCOEF.runPSO()
        # Return coefficients
        CB = partCOEF.returnSolution()[0]
        SSE = partCOEF.returnSolution()[1]
        #MSE
        fCXB = SSE/(0.7*npoints)
        itPSO = partCOEF.returnSolution()[2]
        # print("SSE - pen: {0}".format(SSE-pen*breaker))
        testE = c_funcpen(CB,XB,testset,cfunc,pen)
        testMSE = testE/(0.3*npoints)            
        
        
        print("\nPSO Algorithm has stopped \nBest Coefficients are: {0}\nFitness (MSE): {1}".format(CB,fCXB))      

   
        
    return [SBPSOit,XB,fXB,how,newk[ki],itPSO,CB,fCXB,testMSE]

#______________________#______________________#______________________#______________________    
def DyStruct(ufunc,ufuncs,cfunc,funcs,Xfc,Xf,ChangeP,x_low,x_high,xLc,xL,xHc,xH,k,npoints,swarmp,QP,psoswarm,QPs,max_it,psoit,MSE_condition,SSE_condition,relative_err):    

    MD = max(Xfc) + 2
    #SBPSO   
    c1 = 0.6936
    c2 = 0.5973
    c3 = 4.3133
    c4 = 4.7115
    
    #PSO
    w = 0.7
    c_1 = 1.4
    c_2 = 1.4
    pen = 0
    
    Qswarm = int(QP*swarmp)
    Nswarm = swarmp - Qswarm
    
    # Qpsoswarm = int(QPs*swarmp)
    # Npsoswarm = psoswarm - Qpsoswarm
        
    
    # create universe
    U = set(range(MD+1))
    # U = set(range(10+1))
    
    # create space for variables
    X = [set() for x in range(swarmp)]
    XP =  [set() for x in range(swarmp)]
    XB = U
         
    V =  [set() for x in range(swarmp)] 
    
    fX =  [float('inf') for x in range(swarmp)]
    
    # initialise personal and global best values to inifinity (for a min problem)
    fXP = [float('inf') for x in range(swarmp)]
    fXB = float('inf')
    fCXB = float('inf')
    
    fBreak = []
    OldErr =[]
    ErrBC =[]
    ErrAC =[]
    savePB = []  
    savefCXB = []
    saveMSE = []
    which = []
    
    
    ErrIt =[]
    
    newk = []
    
    for i in range(swarmp):
        newk.append(k) 
    
    # create random dataset
    xset = numpy.random.uniform(x_low, x_high, npoints)
    
    # randomise xset postions
    shuffle(xset)
    
    trainset = xset[range(int(0.7*npoints))]
    ytrain = ufunc(trainset)
    # ytrainc = cfunc(trainset)
    
    testset = xset[range(int(0.7*npoints),npoints)]
    # ytest = cfunc(testset)
    
    # how = "converge"
    
    
    # create particle list
    for i in range(swarmp):
        # random selector
        nx = randint(1,MD+1)
        
        # initialise particle
        X[i] = set(random.sample(list(U),nx))
    
    XQ = X[Nswarm:swarmp:1]
    X = X[0:Nswarm:1]
    
    # add 1s to each X term
    # CX = add_Coef(X)
    
     
    for i in range(Nswarm):
        # fitness of polynomial is the MSE
        fX[i] = fitness(trainset,X[i],ytrain)   
    for i in range(Nswarm,swarmp):
        # fitness of polynomial is the MSE
        fX[i] = fitness(trainset,XQ[i-Nswarm],ytrain)    
        
        
           
    # start search
    
    it = 0
    # relativeDifference = 1
    # reInt = 1
    Count = 0
    pick = 0
    rePick = 0
    PSOit = 0
    newC = 1
    
    # SBPSO while loop
    # while ((fXB > MSE_condition) & (it < max_it)):        
    while ((it < max_it)):      
        
        # if((it > 0)&(it % 10 == 0)):
        if((it > 0) & (numpy.random.uniform(0,1,1) < ChangeP)):
            OldErr.append(fCXB)
            ErrBC.append(fXB)
            ErrIt.append(it)
            # activate variable to recalc coef
            newC = 1
            
            newpick = random.choice(range(len(funcs)))
            while (newpick == pick):
                newpick = random.choice(range(len(funcs)))
                rePick = rePick + 1        
                print("Picked number {0}".format(pick))   
            if(newpick != pick):
                pick = newpick
                Count = Count + 1
                which.append("Function {0}".format(newpick))
                print("Change in function pick number {0}".format(pick))
            
            # if (pick != pick):
            #     Count = Count + 1
            
            cfunc = funcs[pick]
            ufunc = ufuncs[pick]
            # disFUNC = FUNCs[pick]
            xLc = xL[pick]
            xHc = xH[pick]
            XBc = Xf[pick]
            # Cc = coefs[pick]
            
            # update max degree
            MD = max(XBc) + 2
            # update number of coeficients to look for ie. dimension
            # d = len(XBc)
            # update universe size
            U = set(range(MD+1))
            ytrain = ufunc(trainset)
    
            # initialise quantum particles 
            # XQ = quanS(XQ,diversity(len(Qswarm),XQ))
    
             
            for i in range(Nswarm):
                # fitness of polynomial is the MSE
                fX[i] = fitness(trainset,X[i],ytrain)   
            for i in range(Nswarm,swarmp):
                # fitness of polynomial is the MSE
                fX[i] = fitness(trainset,XQ[i-Nswarm],ytrain)    
        
      
            
            # XP =  [set() for x in range(swarmp)]
            XB = U
            fXP = [float('inf') for x in range(swarmp)]
            fXB = float('inf')
            
            for i in range(swarmp):
                # if the current position has a better minimum, update personal best    
                if(i < Nswarm):
                    if fX[i] < fXP[i]:
                        fXP[i] = fX[i]
                        XP[i] = X[i]
                else:
                               
                    if fX[i] < fXP[i]:
                        fXP[i] = fX[i]
                        XP[i] = XQ[i-Nswarm]
                
          
           
                # if the persobal best has a better minimum, update global best
                if fXP[i] < fXB:
                    fXB = fXP[i]
                    XB = XP[i]
                    # ki = i
            ErrAC.append(fXB)   
                
            # break
    
       
        # break 
        for i in range(swarmp):
            # if the current position has a better minimum, update personal best    
            if(i < Nswarm):
                if fX[i] < fXP[i]:
                    fXP[i] = fX[i]
                    XP[i] = X[i]
            else:
                               
                if fX[i] < fXP[i]:
                    fXP[i] = fX[i]
                    XP[i] = XQ[i-Nswarm]
                
          
       
            # if the persobal best has a better minimum, update global best
            if fXP[i] < fXB:
                fXB = fXP[i]
                XB = XP[i]
                # ki = i
            # ErrAC.append(fXB)    
       
        #track PB
        savePB.append(XP)     
    
        # dynamic k - increment by one if PB doenst change
        if (it > 0.1*max_it):
            for i in range(swarmp):   
                if(savePB[it][i] == savePB[it-1][i]):
                    newk[i] += 1
                    # print("plus1")
                    if newk[i] > (MD-2):
                        newk[i] = MD-2
                else:
                    # savek[i] = newk[i]
                    newk[i] = k
                    # print('reset k')
                               
       
            
                
    # update equations
        for i in range(Nswarm):
            
            #update velocity
            s1 = MINUS(XP[i],X[i])
            s2 = MINUS(XB,X[i])
            
            A = X[i] | XP[i] | XB
            S = X[i] & XP[i] & XB
                
            s3 = MULTI(c1*rd(),s1)
            s4 = MULTI(c2*rd(),s2)
    
            s5 = tourn_Add(c3*rd(), X[i], A, newk[i], trainset, ytrain)
            s6 = spes_Rem(c4*rd(), X[i], S)   
            
    
            
            V[i] = s3 | s4 | s5 | s6
            Vlist = list(V[i])
            
            # update postion NORMAL particles
            # union between X and V
            for j in range(len(Vlist)):
                if (Vlist[j] < 0):
                    X[i] = X[i] - {abs(Vlist[j])}
                elif (Vlist[j] >= 0): 
                    X[i] = X[i] | {Vlist[j]}
                    
                    
        # R = diversity(Qswarm,XQ)    
        # R = numpy.random.uniform(0,max(U),1)    
        # update position QUANTUM particles
        # XQ = quanS(XQ, R, U)
        XQ = quanS(XQ,U)
        
         
        for i in range(Nswarm):
        # fitness of polynomial is the MSE
            fX[i] = fitness(trainset,X[i],ytrain)   
        for i in range(Nswarm,swarmp):
        # fitness of polynomial is the MSE
            fX[i] = fitness(trainset,XQ[i-Nswarm],ytrain)   
            
            
            
        fBreak.append(fXB)
            
        #Debugger for bias term
        if(it >= (int(0.05*max_it))):
            # print("test debugger {0}".format(fXB))
            if((round(fXB) == 1) and 0 in XB):
                # print("test debugger - 0")
                XB = XB - {0}
                # CXB = [1 for i in range(len(XB))]
                # print("pop test")
                fXB = fitness(trainset,XB,ytrain)
                # how = "pop"
            elif(round(fXB) == 1):
                XB = XB | {0}
                # CXB = [1 for i in range(len(XB))]
                fXB = fitness(trainset,XB,ytrain)
                # how = "pull"
         
            # test
        print("\nIteration: {0}, MSE: {1} \nPosition: {2}".format(it+1, fXB, XB))
        it += 1
        # print("\n SBPSO Algorithm has stopped \nBest polynomial set is: {0} \nNumber of iterations: {1} \nFitness (MSE): {2}".format(XB,it,fXB))        
        SBPSOit = it
    
        CB = add_COEF(XB)
        
        #PSO while loop to find coefficients of best found structure
    
        testMSE = float('inf')
    
        if((round(fXB) < 1) & newC == 1):
            # xt = numpy.random.uniform(xL,xH,int(npoints/2))
            # while ((fCXB > SSE_condition) & (it < 0.1*max_it)):        
            
            
            prb = Problem(c_funcpen,len(XB),xLc,xHc)
            
            partCOEF = PSOC(SEED(),psoswarm,prb,XB,trainset,cfunc,pen,w,c_1,c_2,psoit,1E-3,0)
            partCOEF.runPSO()
            # Return coefficients
            CB = partCOEF.returnSolution()[0]
            SSE = partCOEF.returnSolution()[1]
            #MSE
            fCXB = SSE/(0.7*npoints)
            # itPSO = partCOEF.returnSolution()[2]
            # print("SSE - pen: {0}".format(SSE-pen*0))
            testE = c_funcpen(CB,XB, testset,cfunc,pen)
            testMSE = testE/(0.3*npoints)            
        
            PSOit = PSOit + 1
            savefCXB.append(fCXB)    
            saveMSE.append(testMSE)
            
            #deactivate coef calc
            newC = 0
            
            print("\nPSO Algorithm has stopped \nBest Coefficients are: {0}\nFitness (MSE): {1}".format(CB,fCXB))      
    
       
        
    # #return [SBPSOit,XB,fXB,how,newk[ki],itPSO,CB,fCXB,testMSE]
    return [SBPSOit,fBreak, Count, rePick, savefCXB, saveMSE, ErrIt, OldErr, ErrBC, ErrAC,which]

#______________________#______________________#______________________#______________________
def SE(yhat,y):
    return (yhat-y)**2

#______________________#______________________#______________________#______________________
def MSE(yhat,y):
    return sum((yhat-y)**2)/len(y)

#______________________#______________________#______________________#______________________
def Regres(func,SEED,xset, Degree, npoints):

    sqtrain = 0
    # sqtest = 0
    
    # randomise xset postions
    shuffle(xset)
    
    trainset = xset[range(int(0.7*npoints))]
    testset = xset[range(int(0.7*npoints),npoints)]
    
    ytrain = func(trainset)
    ytest = func(testset)
        
    
    regModel = numpy.poly1d(numpy.polyfit(trainset, ytrain, Degree))
        
    for j in range(len(trainset)):
        sqtrain = sqtrain + SE(regModel(trainset[j]),ytrain[j])
    
    # for j in range(len(testset)):    
    #     sqtest = sqtest + SE(reg(testset[j]),ytest[j])
    
    MSE_train = sqtrain/npoints
    # MSE_test = sqtest/npoints
    
    
    
    
    return MSE_train,testset,ytest,regModel

#______________________#______________________#______________________#______________________
def Regres2(func,SEED,xset, Degree, npoints):

    sqtrain = 0
    # sqtest = 0
    
    # randomise xset postions
    shuffle(xset)
    
    trainset = xset[range(int(0.7*npoints))]
    testset = xset[range(int(0.7*npoints),npoints)]
    
    ytrain = func(trainset)
    ytest = func(testset)
        
    
    # regModel = numpy.poly1d(P.polyfit(trainset, ytrain, Degree))
    regModel = numpy.poly1d(numpy.polynomial.polynomial.polyfit(trainset, ytrain, Degree)[::-1])
    # regModel = list(numpy.polynomial.polynomial.polyfit(trainset, ytrain, Degree))
        
    for j in range(len(trainset)):
        sqtrain = sqtrain + SE(regModel(trainset[j]),ytrain[j])
    
    # for j in range(len(testset)):    
    #     sqtest = sqtest + SE(reg(testset[j]),ytest[j])
    
    MSE_train = sqtrain/npoints
    # MSE_test = sqtest/npoints
    
    
    
    # print("MSE {0} \nPolynomial\n\n : {1}".format(MSE_train,numpy.poly1d(regModel)))
    return MSE_train,testset,ytest,regModel

#______________________#______________________#______________________#______________________
def inBounds(x, problem):
    


    feasible = True
    for xj in x:
        if ((xj < problem.domMin) or (xj > problem.domMax)):
            feasible = False
            break
    return feasible


#______________________#______________________#______________________#______________________    
# For Quantum particles
#______________________#______________________#______________________#______________________

def diversity(n,X):
    xsum = 0
    dsum = 0
    for i in X:
        xsum = xsum + sum(i)
        xbar = xsum/n
    
    for i in X:
    
        dsum = dsum + math.sqrt((sum(i)-xbar)**2)
        d = dsum/n
    
    return d    

# def quanS(X,rcloud,U):
def quanS(X,U):       
    
    for i in range(len(X)):
        s = math.floor(numpy.random.uniform(1,max(U),1)) 
        X[i] = set(random.sample(U,s))
        
    return X

def rcloud(nxp,xp,xq,problem):
    nqp = 0
    if inBounds(xq,problem):
        nqp = nqp + 1
        
    dXP = diversity(nxp,xp)
    dQP = diversity(nqp,xq)
    
    return max(dXP,dQP)
        
        
    

# def quanS(X,dist,rcloud):
#     dist = dist.upper()
#     if(dist == "UNI"):
#         D = numpy.random.uniform
#         L = 0
#         H = math.floor(rcloud)
#     elif(dist == "NORM"):
#         D = numpy.random.normal
#     else:
#         print("Invalid Distribution Selection")
#         return
    
#     for i in range(len(X)):
#         X[i] = D(L,H,1)
        
#     return dist

