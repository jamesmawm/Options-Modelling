__author__ = 'jamesma'

from scipy import *
import numpy as np

def bisect(target, targetfunction, start=None, bounds=None, tols=[0.01, 0.01], maxiter=100):        
    #Start of inline functions.
    def isBoundValuesDoNotHaveOppositeSigns(fa, fb):
        return bool(not ((fa<0 and fb>0) or (fa>0 and fb<0)))
         
    def isBoundsWithinTolerence(a, b, tol, target, targetfunction):
        fa = targetfunction(a)- target    
        fb = targetfunction(b) - target        
        if isBoundValuesDoNotHaveOppositeSigns(fa, fb):
            return False
            
        if (a > b):
            return bool((a-b)/2.0 < tol)
        else:
            return bool((b-a)/2.0 < tol)
            
    def getInitialBounds(start, bounds, tols):
        if (bounds!=None):
            a = bounds[0];
            b = bounds[1]
        elif (start!=None):
            a = start
            b = start 
            
        if (a>b):
            temp = a
            a = b
            b = temp
                
        return (a, b)
        
    def isRootOfFunction(target, targetfunction, x):
        y = targetfunction(x)  - target
        return bool(y==0)

    def isAnswerFound(y, x, tols, a, b, target, targetfunction):
        return bool((abs(y)<tols[1]) or isBoundsWithinTolerence(a, b, tols[0], target, targetfunction))
        
    def isAnswerFoundByFunction(target, targetfunction, x, tols, a, b):    
        y = targetfunction(x)  - target
        return isAnswerFound(y, x, tols, a, b, target, targetfunction)
                        
    def getMidPoint(a, b):
        return (a+b)/2.0
        
    def checkAndGetBounds(bounds, a, b, target, targetfunction, tols, boundCheckIterations):
        isDoCheckBounds = False
        if (bounds==None):
            fa = targetfunction(a)  - target
            fb = targetfunction(b)  - target
            if isBoundValuesDoNotHaveOppositeSigns(fa, fb):
                a-=tols[0]*2**max(0, boundCheckIterations-1)
                b+=tols[0]*2**max(0, boundCheckIterations-1)
                isDoCheckBounds = True
        return (a, b, isDoCheckBounds)
    #End of inline functions.
    
    # Main route starts here.

    #Check inputs
    if (start==None and bounds==None):
        raise Exception("No inputs supplied!")
        
    result = []
    iterations = 0
         
    #Check if start value contains a solution.
    if (start!=None)  and (bounds==None):
        x = start
        result.append(x)
        if (isRootOfFunction(target, targetfunction, x)):
            return np.array(result)
        
    #Initialize with valid bounds to start the finding process.
    initialBounds = getInitialBounds(start, bounds, tols)
    a = initialBounds[0]
    b = initialBounds[1]
        
    boundCheckIterations=0
    maxBoundCheckArraySize = 50
    isDoCheckBounds = True
        
    #Expand the bounds if the solution out of bounds.
    while (isDoCheckBounds):
        #Limit the number of bound expansion to prevent infinite loops.
        if (len(result)>=maxiter):
            raise Exception("Maximum iteration reached!")
                
        checkedBounds = checkAndGetBounds(bounds, a, b, target, targetfunction, tols, boundCheckIterations)
        a = checkedBounds[0]
        b = checkedBounds[1]
        isDoCheckBounds = checkedBounds[2]            
                        
        #Check left bound.
        result.append(a)
        if (isAnswerFoundByFunction(target, targetfunction, a, tols, a, b)):
            return np.array(result)
                            
        #Check right bound.
        result.append(b)
        if (isAnswerFoundByFunction(target, targetfunction, b, tols, a, b)):
            return np.array(result)
    
        boundCheckIterations += 1
        
    #Check if the expanded bounds are still valid.
    fa = targetfunction(a) - target    
    fb = targetfunction(b) - target        
    if (isBoundValuesDoNotHaveOppositeSigns(fa, fb)):
        if (bounds!=None):
            raise Exception ("Given bounds do not contain a solution.")
        else:
            raise Exception ("Out of bounds (no opposite signs)!")
            
    #Begin searching for a solution by bisection.
    while iterations<(maxiter-boundCheckIterations*2):        
        #Prevent infinite loops.
        if (len(result)>=maxiter):
            raise Exeption("Maximum iteration reached!")       
        
        if (start!=None)  and (bounds!=None) and iterations==0:
            #On the first iteration, if start is supplied (along with bounds) then use it.
            x = start                
            result.append(x)
                        
            #On first iteration, ensure the selected bounds are at the right places.
            fa = targetfunction(a)- target    
            fb = targetfunction(b) - target        
            if (fa > fb):
                a = b                
            else:
                b = a
        elif  (iterations<=1 and boundCheckIterations>0):
            #When the bounds have been expanded, and start is None, check at the bounds.
            #Since a is used by default, assign b to x.
            x = b
        else:
            #Bisect using midpoints of a and b.
            x =getMidPoint(a, b)
            result.append(x)
        
        y = targetfunction(x)  - target    
        if (isAnswerFound(y, x, tols, a, b, target, targetfunction)):
            return np.array(result)
       
        fa = targetfunction(a) - target
        if (sign(fa)==sign(y)):
            a=x
        else:
            b=x
   
        iterations+=1
         
    return np.array(result)
