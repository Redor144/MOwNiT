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

def quad_n(X,Y):
    n = len(X)
    sigmas = [0] + [2*(Y[i]-Y[i-1])/(X[i]-X[i-1]) for i in range(1,n)]
    b = [0]
    for i in range(1,n):
        b.append(sigmas[i]-b[i-1])
    a = [(b[i+1]-b[i])/(2*(X[i+1]-X[i]))for i in range(n-1)]
    def f(x):
        for i in range(n-1):
            if X[i]<=x<=X[i+1]:
                return a[i]*(x-X[i])**2+b[i]*(x-X[i]) + Y[i]
            elif i==n-2:
                return a[i]*(x-X[i])**2+b[i]*(x-X[i]) + Y[i]
    return f

def quad_c(X,Y):
    n = len(X)
    sigmas = [(Y[1]-Y[0])/(X[1]-X[0])] + [2*(Y[i]-Y[i-1])/(X[i]-X[i-1]) for i in range(1,n)]
    b = [(Y[1]-Y[0])/(X[1]-X[0])]
    for i in range(1,n):
        b.append(sigmas[i]-b[i-1])
    a = [(b[i+1]-b[i])/(2*(X[i+1]-X[i]))for i in range(n-1)]
    def f(x):
        for i in range(n-1):
            if X[i]<=x<=X[i+1]:
                return a[i]*(x-X[i])**2+b[i]*(x-X[i]) + Y[i]
            elif i==n-2:
                return a[i]*(x-X[i])**2+b[i]*(x-X[i]) + Y[i]
    return f

def cubic_n(X,Y):
    n = len(X)
    h = [X[i+1]-X[i] for i in range(n-1)]
    delta = [(Y[i+1]-Y[i])/h[i] for i in range(n-1)]
    
    a1 = [1] + [0 for i in range(n-1)]
    a = [a1]
    for i in range(1,n-1):
        ai = [0 for j in range(i-1)] + [h[i-1],2*(h[i-1]+h[i]),h[i]] + [0 for j in range(i+2,n)]
        a.append(ai)
    an = [0 for i in range(n-1)] + [1]
    a.append(an)
    b = [0] + [delta[i+1]-delta[i] for i in range(n-2)] + [0]
    A = np.array(a)
    B = np.array(b)
    # sigmas = np.linalg.inv(A).dot(B)
    sigmas = np.linalg.solve(A,B)
    
    a = [(sigmas[i+1]-sigmas[i])/h[i] for i in range(n-1)]
    b = [3* sigmas[i]for i in range(n-1)]
    c = [(Y[i+1]-Y[i])/h[i] - h[i]*(sigmas[i+1]+2*sigmas[i]) for i in range(n-1)]
    def f(x):
        for i in range(n-1):
            if X[i]<=x<=X[i+1]:
                return a[i]*(x-X[i])**3 + b[i]*(x-X[i])**2 + c[i]*(x-X[i]) + Y[i]
            elif i==n-2:
                return a[i]*(x-X[i])**3 + b[i]*(x-X[i])**2 + c[i]*(x-X[i]) + Y[i]
    return f

def cubic_c(X,Y):
    n = len(X)
    h = [X[i+1]-X[i] for i in range(n-1)]
    delta = [(Y[i+1]-Y[i])/h[i] for i in range(n-1)]
    
    a1 = [2,1] + [0 for i in range(n-2)]
    a = [a1]
    for i in range(1,n-1):
        ai = [0 for j in range(i-1)] + [h[i-1],2*(h[i-1]+h[i]),h[i]] + [0 for j in range(i+2,n)]
        a.append(ai)
    an = [0 for i in range(n-2)] + [2,1]
    a.append(an)
    b = [0] + [delta[i+1]-delta[i] for i in range(n-2)] + [0]
    A = np.array(a)
    B = np.array(b)
    # sigmas = np.linalg.inv(A).dot(B)
    sigmas = np.linalg.solve(A,B)
    
    a = [(sigmas[i+1]-sigmas[i])/h[i] for i in range(n-1)]
    b = [3* sigmas[i]for i in range(n-1)]
    c = [(Y[i+1]-Y[i])/h[i] - h[i]*(sigmas[i+1]+2*sigmas[i]) for i in range(n-1)]


    def f(x):
        for i in range(n-1):
            if X[i]<=x:
                return a[i]*(x-X[i])**3 + b[i]*(x-X[i])**2 + c[i]*(x-X[i]) + Y[i]
            # elif i==n-2:
            #     return a[i]*(x-X[i])**3 + b[i]*(x-X[i])**2 + c[i]*(x-X[i]) + Y[i]
    return f


def find_best_function():
    function = lambda x,m,k: 10 * m + (x**2) / k - 10 * m * math.cos(k * x)
    m = 2
    k = 2
    a = -3 * math.pi
    b = 3 * math.pi
    x = [a,b]
    N = 1000

    f_to_interpolate = lambda x: function(x,m,k)
    
    X = evenly(a,b,N)
    Y = values(f_to_interpolate,X)
    f_q_c_r_min_diff = (float('inf'),float('inf'))
    f_q_n_r_min_diff = (float('inf'),float('inf'))
    f_c_n_r_min_diff = (float('inf'),float('inf'))
    f_c_c_r_min_diff = (float('inf'),float('inf'))


    for i in range(3,1001):
        XC = czebyszew(a,b,i)
        XR = evenly(a,b,i)
        YC = values(f_to_interpolate,XC)
        YR = values(f_to_interpolate,XR)
        
        f_q_n_r = quad_n(XR,YR)
        # f_q_n_c = quad_n(XC,YC)
        f_q_c_r = quad_c(XR,YR)
        # f_q_c_c = quad_c(XC,YC)
        
        f_c_n_r = cubic_c(XR,YR)
        # f_c_n_c = cubic_n(XC,YC)
        f_c_c_r = cubic_c(XR,YR)
        # f_c_c_c = cubic_c(XC,YC)
        
        f_q_n_r_values = values(f_q_n_r,X)
        # f_q_n_c_values = values(f_q_n_c,X)
        f_q_c_r_values = values(f_q_c_r,X)
        # f_q_c_c_values = values(f_q_c_c,X)
        f_c_n_r_values = values(f_c_n_r,X)
        # f_c_n_c_values = values(f_c_n_c,X)
        f_c_c_r_values = values(f_c_c_r,X)
        # f_c_c_c_values = values(f_c_c_c,X)
        
        # plt.figure()
        # print(XR,YR)
        # print(Y,f_q_n_r_values)
        # print(f_q_n_c_values)
        # plt.plot()
        
        # f_q_n_r_min_diff = (float('inf'),float('inf'))
        if max_diff_square_1(Y,f_q_n_r_values) < f_q_n_r_min_diff[0]:
            f_q_n_r_min_diff=(max_diff_square_1(Y,f_q_n_r_values),i)
            
        # f_q_n_c_min_diff = (float('inf'),float('inf'))
        # if max_diff_square_1(Y,f_q_n_c_values) < f_q_n_c_min_diff[0]:
        #     f_q_n_c_min_diff=(max_diff_square_1(Y,f_q_n_c_values),i)
            
        # f_q_c_r_min_diff = (float('inf'),float('inf'))
        if max_diff_square_1(Y,f_q_c_r_values) < f_q_c_r_min_diff[0]:
            f_q_c_r_min_diff=(max_diff_square_1(Y,f_q_c_r_values),i)
            
        # f_q_c_c_min_diff = (float('inf'),float('inf'))
        # if max_diff_square_1(Y,f_q_c_c_values) < f_q_c_c_min_diff[0]:
        #     f_q_c_c_min_diff=(max_diff_square_1(Y,f_q_c_c_values),i)
        
        # f_c_n_r_min_diff = (float('inf'),float('inf'))
        if max_diff_square_1(Y,f_c_n_r_values) < f_c_n_r_min_diff[0]:
            f_c_n_r_min_diff=(max_diff_square_1(Y,f_c_n_r_values),i)
            
        # f_c_n_c_min_diff = (float('inf'),float('inf'))
        # if max_diff_square_1(Y,f_c_n_c_values) < f_c_n_c_min_diff[0]:
        #     f_c_n_c_min_diff=(max_diff_square_1(Y,f_c_n_c_values),i)
            
        # f_c_c_r_min_diff = (float('inf'),float('inf'))
        if max_diff_square_1(Y,f_c_c_r_values) < f_c_c_r_min_diff[0]:
            f_c_c_r_min_diff=(max_diff_square_1(Y,f_c_c_r_values),i)
            
        # f_c_c_c_min_diff = (float('inf'),float('inf'))
        # if max_diff_square_1(Y,f_c_c_c_values) < f_c_c_c_min_diff[0]:
        #     f_c_c_c_min_diff=(max_diff_square_1(Y,f_c_c_c_values),i)
        print(i)
            
    print("f_q_n_r: ",f_q_n_r_min_diff)
    # print("f_q_n_c: ",f_q_n_c_min_diff)
    print("f_q_c_r: ",f_q_c_r_min_diff)
    # print("f_q_c_c: ",f_q_c_c_min_diff)
    
    print("f_c_n_r: ",f_c_n_r_min_diff)
    # print("f_c_n_c: ",f_c_n_c_min_diff)
    print("f_c_c_r: ",f_c_c_r_min_diff)
    # print("f_c_c_c: ",f_c_c_c_min_diff)
        

if __name__ == "__main__":
    # print("Wyznaczenie funkcji dla konretnej liczby węzłów wybierz 0, wyznacznie funkcji najlepiej przybliżającej wybierz 1: ")
    # to_do = int(input())
    to_do=0
    if not to_do:
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

        # print("Rozmieszczenie węzłów równomierne podaj 0, według Czebyszewa podaj 1: ")
        # rozm = int(input())
        
        # if rozm:
            # XN = czebyszew(a,b,n)
            # print("SS")
        # else:
        XN = evenly(a,b,n)
        
        YN = values(f_to_interpolate,XN)
        
        X = evenly(a,b,N)
        Y = values(f_to_interpolate,X)
        
        print("2 stopień podaj 0, 3 stopeń podaj 1: ")
        stop = int(input())
        
        if not stop:
            f_quad_natural = quad_n(XN,YN)
            f_quad_clamped = quad_c(XN,YN)
            YFN = values(f_quad_natural,X)
            YFC = values(f_quad_clamped,X)
            
            max_diff_natural_q = max_diff_1(Y,YFN)
            max_diff_clamped_q = max_diff_1(Y,YFC)
            
            max_diff_natural_q_s = max_diff_square_1(Y,YFN)/N
            max_diff_clamped_q_s = max_diff_square_1(Y,YFC)/N
            
            print(f"max_diff_natural_q: {max_diff_natural_q}\n max_diff_clamped_q: {max_diff_clamped_q}\n max_diff_natural_q_s: {max_diff_natural_q_s}\n max_diff_clamped_q_s: {max_diff_clamped_q_s}")
            
            plt.figure(figsize=(10,5))
            plt.plot(X,Y,c='g',label="f(x)")
            plt.plot(X,YFN,c='b',label='Free Boundry')
            plt.plot(X,YFC,c='r',label='Clamped Boundry')
            plt.scatter(XN,YN,c='black',label='Węzły')
            # plt.title("Wykres funckji sklejan 2 stopnia")
            
            plt.legend(loc='best')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid()
            sns.despine()
            plt.show()
        else:
            f_cubic_natural = cubic_n(XN,YN)
            f_cubic_clamped = cubic_c(XN,YN)
            YFN = values(f_cubic_natural,X)
            YFC = values(f_cubic_clamped,X)
            plt.figure(figsize=(10,5))
            plt.plot(X,Y,c='g',label="f(x)")
            plt.plot(X,YFN,c='b',label='Free Boundary')
            plt.plot(X,YFC,c='r',label='Clamped Boundary')
            plt.scatter(XN,YN,c='black',label='Węzły')
            
            max_diff_natural_c = max_diff_1(Y,YFN)
            max_diff_clamped_c = max_diff_1(Y,YFC)
            
            max_diff_natural_c_s = max_diff_square_1(Y,YFN)/N
            max_diff_clamped_c_s = max_diff_square_1(Y,YFC)/N
            
            print(f"max_diff_natural_c: {max_diff_natural_c}\n max_diff_clamped_c: {max_diff_clamped_c}\n max_diff_natural_c_s: {max_diff_natural_c_s}\n max_diff_clamped_c_s: {max_diff_clamped_c_s}")
            
            # plt.title("Wykres interpolacji 3 stopnia")
            plt.legend(loc='best')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid()
            sns.despine()
            plt.show()
    # else:
    #     find_best_function()
        