import math
from BS import  *
import numpy as np

precision = 0.000001

def getBSFormula(callput, S, K, r, T, sigma, q=0., t=0):
    g = "call" if callput==1 else "put"
    return bsformula(g, S, K, r, T,  sigma, q)

def bsDelta(callput, S, K, r, T, sigma, q=0., t=0):
    return getBSFormula(callput, S, K, r, T,  sigma, q)[1]

def bsGamma(callput, S, K, r, T, sigma, q=0., t=0):
    _d1 = d1(S,K,r,T,sigma,q)
    return norminv(_d1)*math.exp(-q*T)/ (S*sigma*math.sqrt(T))

def bsPrice(callput, S, K, r, T, sigma, q=0., t=0):
    return getBSFormula(callput, S, K, r, T,  sigma, q)[0]

def impliedBS(V, callput, S, K, r, T, q=0., t=0):
    g = "call" if callput==1 else "put"
    impv =  bsimpvol(g, S, K, r, T, V, 0.0, q, precision, 'secant', False)
    return impv

def lrtree(callput, S, K, r, T, sigma, q=0., t=0, params={'stepCount':200}):
    n = params['stepCount']
    n = n if (n % 2 == 1) else n+1
    g = "call" if callput==1 else "put"
    dt = (T-t)/float(n)
    R = math.exp((r)*dt);
    discountedDividend = math.exp(-q*dt)
    discountedR = 1/R

    def ppMethod1Inversion(z):
        operator = 1 if z>=0 else -1
        return 0.5 + operator*0.5*math.sqrt(1 - math.exp(-((z/(n+1.0/3.0))**2.0)*(n+1.0/6.0)))

    def getBSNodeValue(stockPriceAtNode, time):
        return bsformula(g, stockPriceAtNode, K, r, time,  sigma, q)[0]

    def compareNodeValues(stockPriceAtNode, bsNodeValue):
        earlyExerciseValue = max(stockPriceAtNode-K,0) if callput==1 else max(K-stockPriceAtNode, 0)
        return max(earlyExerciseValue, bsNodeValue)

    def getNodeValue(stockPriceAtNode):
        bsNodeValue = getBSNodeValue(stockPriceAtNode, T)
        return compareNodeValues(stockPriceAtNode, bsNodeValue)

    def getPenultimateNodeValue(stockPriceAtNode):
        bsNodeValue = getBSNodeValue(stockPriceAtNode, dt)
        return compareNodeValues(stockPriceAtNode, bsNodeValue)

    def generateStockPricesTree(startSt, stepcount):
        stockPrices = []
        for i in range(stepcount+1):
            stockPrices.append([])
            for j in range(i+1):
                stockPrice = startSt * (u**(i-j)) * (d**j)
                stockPrices[i].append(stockPrice)

        return stockPrices

    def generateEmptyTreeArray(treeSize):
        tree=[]
        for i in range(treeSize+1):
            tree.append([])
        return tree

    def getPayoffsOnTree(stockPrices):
        # Get payoff at each node, starting from the back of the tree.
        penultimateIndex = len(stockPrices)-2
        optionValues = generateEmptyTreeArray(len(stockPrices)-1)
        for i in range(penultimateIndex, -1, -1):
            for j in range(i+1):
                stockPrice = stockPrices[i][j]  * discountedDividend**i

                if (i==penultimateIndex):
                    optionValue = getPenultimateNodeValue(stockPrice)
                else:
                    optionValuesAti = optionValues[i+1]
                    payoffAtUpNode = optionValuesAti[j]
                    payoffAtDownNode = optionValuesAti[j+1]
                    payoffAtNode = 1/R*(payoffAtUpNode*p+(1-p)*payoffAtDownNode)
                    optionValue = compareNodeValues(stockPrice, payoffAtNode)

                optionValues[i].append(optionValue)
        return optionValues

    tenor = (T-t)
    _d1 = d1(S, K, r, tenor, sigma, q)
    _d2 = d2(S, K, r, tenor, sigma, q)
    p = ppMethod1Inversion(_d2)
    pNot = ppMethod1Inversion(_d1)

    #Return of tree traveral is not possible.
    if (p >= 1.0) or (p<=0.0):
        return (float('NaN'), float('NaN'), float('NaN'))

    u = R*pNot/float(p)
    d = (R-p*u)/(1-p) if (1-p)!=0 else 0 # Prevent division by zero.

    pretendUpNodeStockPrice = S*u/d
    pretendDownNodeStockPrice = S*d/u

    if (n<=1):
        optionValue = getNodeValue(S)
        Vup = getNodeValue(pretendUpNodeStockPrice)
        Vdown =getNodeValue(pretendDownNodeStockPrice)
    else:
        stockPrices = generateStockPricesTree(S, n)      # Setup tree of terminal stock prices   .
        optionValues = getPayoffsOnTree(stockPrices)    # Generate payoffs on tree.
        optionValue = optionValues[0][0]

        stockPrices = generateStockPricesTree(S/u/d*discountedDividend**-2, n+2)
        optionValues = getPayoffsOnTree(stockPrices)

        Vup =optionValues[2][0]
        Vdown = optionValues[2][2]

    # Get delta
    dS = (pretendUpNodeStockPrice-pretendDownNodeStockPrice)
    if (dS!=0.0):
        lrDelta = (Vup-Vdown)/dS
    else:
        lrDelta = float('NaN')

    #Get gamma
    gammaDeno = ((S+pretendUpNodeStockPrice)/2.-(pretendDownNodeStockPrice+S)/2.)
    dSup = (pretendUpNodeStockPrice-S)
    dSdown = (S-pretendDownNodeStockPrice)
    if (gammaDeno!=0.0):
        gamma = ((Vup-optionValue)/dSup - (optionValue-Vdown)/dSdown)/gammaDeno
    else:
        gamma = float('NaN')

    return (optionValue, lrDelta, gamma)


def lrtreePrice(callput, S, K, r, T, sigma, q=0., t=0, params={'stepCount':200}):
    return lrtree(callput, S, K, r, T, sigma, q, t, params)[0]

def lrtreeDelta(callput, S, K, r, T, sigma, q=0., t=0, params={'stepCount':200}):
    return lrtree(callput, S, K, r, T, sigma, q, t, params)[1]

def lrtreeGamma(callput, S, K, r, T, sigma, q=0., t=0, params={'stepCount':200}):
    return lrtree(callput, S, K, r, T, sigma, q, t, params)[2]