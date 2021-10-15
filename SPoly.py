#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 08:55:51 2021

@author: Donnas
"""
    
#Static univariate functions

def uni_f1(x):
    # Xfull = {0,1,2,3,4}
    return x**4 + x**2

#f2 form table 1
def uni_f2(x):
    # Xfull = {0,1,2,3}
    return x**3 + x**2 + x

#f3 form table 1
def uni_f3(x):
    # Xfull = {0,1,2,3,4,5,6,7}
    return x**7 + x**5 + x**4 + 1

#f4 form table 1
def uni_f4(x):
    # Xfull = {0,1,2,3,4,5,6,7,8}
    return x**8 + x**7 + x**6 + x**5 + x**4 + x**3 + x**2 + x + 1 


def uni_f5(x):
    # Xfull = {0,1,2,3,4,5,6}
    return x**6 + x**3 + x + 1


def uni_f6(x):
    # Xfull = {0,1,2,3,4,5}
    return x**5 + x**2 + 1


def uni_f7(x):
     # Xfull = {0,1,2,3,4,5,6,7,8}
    return x**8 + x**3 + x + 1

# unif1
def cf1(x):
    return 2*x**4 + 3*x**2 

# unif2    
def cf2(x):
    return 0.5*x**3 + 2*x**2 -1 * x 

# unif4
def cf4(x):
    return 2*x**8 + x**7 + 2.5*x**6 + 3*x**5 - 2*x**4 + 1.5*x**3 - x**2 + 3.5*x - 1 

def cf5(x):
    return 3*x**6 - 3*x**3 + 5*x + 1

def cf6(x):
    return -2.5*x**5 - 5.2*x**2 - 1

def cf7(x):
    return -2*x**8 - 1.5*x**3 + 2*x - 0.5


