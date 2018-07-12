import numpy as np


def harmonic(n): #original harmonic series
     a=1.0
     for d in range(2, n+1):
         a = a+1.0/(d*d)
     return(a)

print harmonic(100000)*6/(3.1415926**2)
