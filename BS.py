import math
import scipy.optimize
import scipy.stats as stats
from pprint import pprint
from Bisect import bisect
import numpy

def norminv(x):
    return ((1.0/math.sqrt(2.0*math.pi)) * math.exp(-x*x*0.5))

def d1(S0, K, r, T, sigma, q):
    deno = (sigma * math.sqrt(T))
    if (deno==0):
        return 0
    logReturns = math.log(S0/float(K)) if ((S0/float(K)) > 0.0) else 0.0
    return (float(logReturns) + (float(r) - float(q) + float(sigma)*float(sigma)*0.5)*float(T)) / float(deno)
    
def d2(S0, K, r, T, sigma, q):
        return d1(S0, K, r, T, sigma, q)-sigma*math.sqrt(T)
        
def bsformula(callput, S0, K, r, T, sigma, q=0):
    N = stats.norm.cdf
                
    def optionValueOfCall(S0, K, r, T, sigma, q):       
        _d1 = d1(S0, K, r, T, sigma, q)
        _d2 = d2(S0, K, r, T, sigma, q)
        return S0*math.exp(-q*T)*N(_d1)- K*math.exp(-r*T)*N(_d2)
      
    def optionValueOfPut(S0, K, r, T, sigma, q):
        _d1 = d1(S0, K, r, T, sigma, q)
        _d2 = d2(S0, K, r, T, sigma, q)
        return float(K)*math.exp(-float(r)*float(T))*N(-_d2) - float(S0)*math.exp(-float(q)*float(T))*N(-_d1)
        
    def delta(callput, S0, K, r, T, sigma, q):
        _d1 = d1(S0, K, r, T, sigma, q)        
        if callput.lower() == "call":            
            return N(_d1) * math.exp(-q*T)
        else:
            return (N(_d1)-1)* math.exp(-q*T)
    
    def vega(S0, K, r, T, sigma, q):
        _d1 = d1(S0, K, r, T, sigma, q)
        return S0  * math.sqrt(T) * norminv(_d1)  * math.exp(-q*T)
    
    if callput.lower()=="call":
        optionValue = optionValueOfCall(S0, K, r, T, sigma, q)
    else:
        optionValue = optionValueOfPut(S0, K, r, T, sigma, q)
        
    _delta = delta(callput, S0, K, r, T, sigma, q)
    _vega = vega(S0, K, r, T, sigma, q)
    
    return (optionValue, _delta, _vega)
    
def secantMethod(targetfunction, x0,x1,n=100):
    numberOfCalls = 0
    for i in range(n):        
        numberOfCalls+=1
        ans1 = targetfunction(x1)
        ans2 = targetfunction(x0)            
        deno = ans1-ans2
        
        if deno == 0.0:           
            return (x1, numberOfCalls)
        x_temp = x1- (ans1*(x1-x0))/deno
        x0 = x1
        x1 = x_temp        
        
    return (x1, numberOfCalls)
    
def secantMethodBS(price, targetfunction, callput, S0, K, r, T, q, precision):
    
    def getInitialImpVolUsingNewtonsMethod(initialVol):
        return newtonsMethod(callput, S0, K, r, T, q, price, initialVol, precision, 200)[0]
        
    def getValidImpVolBounds():
        # We expect a volatility from 0.00 to 1.00 in typical cases.
        initialVol1 = 0.0
        initialVol2 = 1.0
            
        impv1 = getInitialImpVolUsingNewtonsMethod(initialVol1)
        impv2 = getInitialImpVolUsingNewtonsMethod(initialVol2)
        
        if (((impv2<0) and (impv1>0)) or (impv2>5.0)):            
             # If the right bound gives illegal values (less than 0s), decrement towards the left bound.    
            IMPV_STEP = 0.1
            newImpv = impv2       
            if (impv2>5.0):
                impv2 = 5.0
            else:
                impv2 = initialVol2
            while (newImpv < 0.)  and (impv2>impv1):    
                impv2-=IMPV_STEP
                newImpv = newtonsMethod(callput, S0, K, r, T, q, price, impv2, precision, maxIterations=200)[0]
        
        return (impv1, impv2)
    
    def newTargetFunction(_x, _price=price):
        return targetfunction(_x)/precision-_price/precision
        
    impvBounds = getValidImpVolBounds()
    a = impvBounds[0]
    b = impvBounds[1]
    result = secantMethod(newTargetFunction, a, b)       
        
    return result
    
def newtonsMethod(callput, S0, K, r, T, q, price, initialVol, tolerance=0.01, maxIterations=100):
    iterations=0
    x = initialVol
    prevega=0
    while iterations < maxIterations:
        bsdata = bsformula(callput, S0, K, r, T, x, q)
        optionValue = bsdata[0]
        vega = bsdata[2]
        
        #Take the current implied volatility to prevent division by zero error.
        if (vega==0):   
            return (x, iterations+1)
        prevega = vega
        newt = float(x) - (float(optionValue)-float(price))/float(vega)
        if (abs(newt-x)<tolerance):            
            return (newt, iterations+1)
        
        x = newt        
        iterations += 1
        
    return (float('NaN'), 0)
    
def isNoVolatilityCanBeFound(callput, price, S0, K, T):
    def isOptionValueLessThanIntrinsic(callput, price, S0, K):
        # Check for negative option value.
        if (price<=0):
            return True
            
         # Check if option value is less than intrinsic.
        if (callput.lower()=="call"):
            intrinsic = S0-K
        else:
            intrinsic = K-S0
                   
        if (price < intrinsic):
            return True
                
        return False
        
    def isHaveInvalidInputs(callput, S0, K, T):
        if not (callput.lower()=="call" or callput.lower()=="put"):
            return True

        if (S0<=0 or K<=0 or T<=0):
            return True
        
        return False
        
    invalidInputs = isHaveInvalidInputs(callput, S0, K, T)
    otm = isOptionValueLessThanIntrinsic(callput, price, S0, K)
    return bool(invalidInputs or otm)

def bsimpvol(callput, S0, K, r, T, price, sigma, q=0, priceTolerance=0.01, method='bisect', reportCalls=False):
    def targetfunction(x):
        return bsformula(callput, S0, K, r, T, x, q)[0]
        
    def getReturnData(_impVol, _calls):
        if (reportCalls):
            return (_impVol, _calls)
        else:
            return _impVol

    if (isNoVolatilityCanBeFound(callput, price, S0, K, T)):
        return getReturnData(float('NaN'), 0)
    
    impvol = float('NaN')
    calls = 0
    
    if (method=="bisect"):
        start = 0.50
        result = bisect(price, targetfunction, start, None, [priceTolerance, priceTolerance])
        impvol = result[-1]
        calls = len(result)
    elif (method=="newton"):
        result = newtonsMethod(callput, S0, K, r, T, q, price, sigma, priceTolerance)
        impvol = result[0]
        calls = result[1]            
    else:
        result = secantMethodBS(price, targetfunction, callput, S0, K, r, T, q, priceTolerance)
        impvol = result[0]
        calls = result[1]

    return getReturnData(impvol, calls)