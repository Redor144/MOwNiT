import math
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
import numpy as np
# n - małe liczba węzlów, N - liczba punktów dla których liczymy
def czebyszew(a,b,n):
    X = []
    for k in range(1,n+1):
        xk = 0.5 * (a + b) + 0.5 * (b - a) * math.cos((2 * k - 1) * math.pi / (2 * n))
        X.append(xk)
    return X[::-1]

def evenly(a,b,n):
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def values(f,X):
    n = len(X)
    return [f(X[i]) for i in range(n)]

def max_diff(F,f,X):
    diff = []
    for x in X:
        diff.append(abs(f(x)-F(x)))
    return max(diff)

def max_diff_square(F,f,X):
    sum = 0
    for x in X:
        sum += pow(F(x)-f(x),2)
    return sum

def max_diff_1(X1,X2):
    maksi = float('-inf')
    for i in range(len(X1)):
        maksi= max(maksi,abs(X1[i]-X2[i]))
    return maksi

def max_diff_square_1(X1,X2):
    sum = 0
    for i in range(len(X1)):
        sum += pow(X1[i]-X2[i],2)
    return sum

def gauss_jordan(A, B):
    n = len(B)
    
    for k in range(n):
        # Partial pivoting
        if np.fabs(A[k, k]) < 1e-12:
            for i in range(k + 1, n):
                if np.fabs(A[i, k]) > np.fabs(A[k, k]):
                    # Swap rows
                    A[[k, i]] = A[[i, k]]  
                    B[[k, i]] = B[[i, k]]
                    break
        # Division of the pivot row
        pivot = A[k, k]
        A[k] /= pivot
        B[k] /= pivot
        # Elimination loop
        for i in range(n):
            if i == k or A[i, k] == 0: continue
            factor = A[i, k]
            A[i] -= factor * A[k]
            B[i] -= factor * B[k]
            
    return B

def least_square_approximation(xs, ys, ws, m):
    
    n = len(xs)
    G = np.zeros((m + 1, m + 1), float)
    B = np.zeros(m + 1, float)
    
    sums = [sum(ws[i] * xs[i] ** k for i in range(n)) for k in range(2 * m + 1)]
    
    for j in range(m + 1):
        for k in range(m + 1):
            G[j, k] = sums[j + k]
        
        B[j] = sum(ws[i] * ys[i] * xs[i] ** j for i in range(n))
        
    A = gauss_jordan(G, B)
    return lambda x: sum(A[i] * x ** i for i in range(m + 1))
        

if __name__ == "__main__":
    function = lambda x,m,k: 10 * m + (x**2) / k - 10 * m * math.cos(k * x)
    derivative = lambda x,m,k: 2 * x / k + 10 * m * math.sin(k * x) * k
    m = 2
    k = 2
    a = -3 * math.pi
    b = 3 * math.pi
    x = [a,b]
    N = 1000

    f_to_interpolate = lambda x: function(x,m,k)
    
    print("Podaj liczbę węzłów: ")
    n = int(input())
    # n = 50
    z = int(input("Podaj stopień wielomianu: "))
    # z = 17
    XN = evenly(a,b,n)
    YN = values(f_to_interpolate,XN)
    
    X = evenly(a,b,N)
    Y = values(f_to_interpolate,X)
    W = [1 for i in range(len(XN))]
    f = least_square_approximation(XN,YN,W,z)
    YF = values(f,X)
    # f_quad_natural = quad_n(XN,YN)
    # f_quad_clamped = quad_c(XN,YN)
    # YFN = values(f_quad_natural,X)
    # YFC = values(f_quad_clamped,X)
    print(f"max_dirff: {max_diff_1(Y,YF):.3f}")
    print(f"max_diff_square: {max_diff_square_1(Y,YF)/N:.3f}")
    
    # for i in range(15,51,5):
    XN = evenly(a,b,n)
    YN = values(f_to_interpolate,XN)
    f = least_square_approximation(XN,YN,W,z)
    YF = values(f,X)
    plt.figure(figsize=(10,5))
    plt.plot(X,Y,c='g',label="f(x)")
    plt.plot(X,YF,c='b',label='Funkcja aproksymująca')
    # plt.plot(X,YFC,c='r',label='Clamped Boundry')
    plt.scatter(XN,YN,c='black',label='Węzły')
    # plt.title("Wykres funckji sklejan 2 stopnia")

    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    sns.despine()
    plt.show()
        