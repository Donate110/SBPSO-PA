#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:58:47 2021

@author: Donnas
"""

import ALLFUNCS

def ubase(x):
    return x**4 + x + 1
def base(x):
    return 2*x**4 + x - 2

# case1

def case1u1(x):
    return x**7 + x**5 + x**3 + x


def case1f1(x):
    return x**7 + x**5 + 2*x**3 + x 

def case1u2(x):
    return x**8 + x**2 + 1

def case1f2 (x):
    return 2*x**8 + x**2 - 2

def case1u3(x):
    return x**3 + x**2 +1 

def case1f3(x):
    return x**3 + 2*x**2 - 2

# case 2 - all have same ufunc = ubase

def case2f1(x):
    return 4*x**4 + 2*x**2 - 4

def case2f2 (x):
    return -2*x**4 - x**2 + 2

def case2f3(x):
    return 3.3*x**4 + 6*x**2 - 4.2


# case 3

def case3u1(x):
    return x**7 + x**2 + x + 1

def case3f1(x):
    return 4*x**7 + x**2 - 6*x - 0.5

def case3u2 (x):
    return x**5 + x**3 + x + 1

def case3f2 (x):
    return 0.5*x**5 + 3*x**3 - 2*x + 1

def case3u3(x):
    return x**6  + x**4 + x**2 + x + 1

def case3f3(x):
    return -x**6 - 2.32*x**4 + 0.84*x**2 - 5.11*x + 3.78

