import math
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
import numpy as np

# funckja w przedziale jak w jednym okresie 
# trzeba odpowiednio rzutować albo nasz przedział rzutujemy do przedziału jednego okresu
# i odpowiedznio wtedy mamy rozkład węzłó i wartości rzutowany, albo przerabiamy wzory, żeby były dostowsowane do
# naszego przedziału ale kontrolujemy rozmieszczenie więzłów i wartości w nich
# funckja powinna mieć takie same wartości na krańcach przedziału
# stopień wielomianu != liczba funkcji bazowych
# a0 + suma i=1 do n ai sin() + suma i=0 do n bi cos(x)
# m > 2n+1

from math import sin, cos, pi

def trigonometric_approximation(X, Y, m):
    n = len(X)
    
    def transform_x(x):
        return x/3
    
    def calc_ak(k):
        return 2 / n * sum(Y[i] * cos(k * X[i]) for i in range(n))
    
    def calc_bk(k):
        return 2 / n * sum(Y[i] * sin(k * X[i]) for i in range(n))
    
    X = list(map(transform_x, X))
    ak = list(map(calc_ak, range(m + 1)))
    bk = list(map(calc_bk, range(m + 1)))
    
    def f(x):
        x = transform_x(x)
        return .5 * ak[0] + sum(ak[k] * cos(k * x) + bk[k] * sin(k * x) for k in range(1, m))
    
    return f

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
    
    n = int(input("Podaj liczbę węzłów: "))
    z = int(input("Podaj stopień wielomianu: "))
    # 
    # z = 15
    # n=100
    
    XN = evenly(a,b,n)
    YN = values(f_to_interpolate,XN)
    
    X = evenly(a,b,N)
    Y = values(f_to_interpolate,X)
    f = trigonometric_approximation(XN,YN,z)
    YF = values(f,X)
    print(f"max_dirff: {max_diff_1(Y,YF):.6f}")
    print(f"max_diff_square: {max_diff_square_1(Y,YF)/N:.6f}")
    # for i in [7,10,15,20,25,35,40,50,60,75,85,100]:
    #     arr = []
    #     for j in [3,4,5,7,10,15,25,35,49]:
    #         if j<=(i-1)//2:
    #             XN = evenly(a,b,i)
    #             YN = values(f_to_interpolate,XN)
    #             f = trigonometric_approximation(XN,YN,j)
    #             YF = values(f,X)
    #             arr.append("{:.3f}".format(max_diff_square_1(Y,YF)))
    #     print(arr)
    # print(max(Y))
    plt.figure(figsize=(10,5))
    plt.plot(X,Y,c='g',label="f(x)")
    plt.plot(X,YF,c='b',label='Funkcja aproksymująca')
    plt.scatter(XN,YN,c='black',label='Węzły')

    
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    sns.despine()
    plt.show()