from __future__ import division
import string, datetime
from numpy import *
import scipy as sp
from pprint import pprint
import scipy.stats as st
from numpy.random import rand
import matplotlib.pyplot as plot
norminv = st.distributions.norm.ppf
norm = st.distributions.norm.cdf
from sobol_lib import *

def getCorrelationMatrix(corr, N):
    return eye(N) * (1.-corr) + corr * ones((N, N))

def defaultTimes(hazardRates, correlation, samples=1):
    N = len(hazardRates)

    hasAntithetic = sum(samples<0) > 0

    if hasAntithetic:
        length = shape(samples)[0]
        samples = samples[0:length/2]

    if shape(samples)==():
        y = getSamplesByType(None, samples, N)
    else:
        y = samples

    if shape(correlation) == ():
        corr = getCorrelationMatrix(correlation, N)
    else:
        corr = correlation

    ch = linalg.cholesky(corr)
    ch = transpose(ch)
    w = norminv(y)
    z = dot(w, ch)

    if hasAntithetic:
        z = append(z, -z, axis=0)

    x = norm(z)
    tau = -log(1-x)/hazardRates
    return tau

def copulaLosses(r, T, notionals, recoveryRates, hazardRates, correlation, samples=1):
    N = size(notionals) 
    tau = defaultTimes(hazardRates, correlation, samples)  
    M = shape(tau)[0]        
    notionalsMatrix = ones((M, N))
    notionalsMatrix[:] = notionals
    recoveryMatrix = ones((M, N))
    recoveryMatrix[:] = 1-recoveryRates
    V = notionalsMatrix*recoveryMatrix
    V = exp(-r*tau) * V
    losses=V
    losses[tau>T] = 0    
    lossSamples = losses.sum(axis=1)  
    return lossSamples

def tranche(K1,K2, r, T, notionals, recoveryRates, hazardRates, correlation, samples=1):        
    losses = copulaLosses(r, T, notionals, recoveryRates, hazardRates, correlation, samples)
    losses = losses-K1
    losses = losses.clip(min=0,max=(K2-K1))
    meanLosses = mean(losses, axis=0)
    return meanLosses

def getSamplesByType(samplingType, M, N):
    if samplingType == "Antithetic":
        z = rand(M,N)
        z = append(z, -z, axis=0)
        return z

    if samplingType == "Sobol":
        z = i4_sobol_generate(N, M, 3).T
        return array(z)

    return rand(M, N)