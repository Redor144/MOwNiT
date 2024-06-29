import math
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns

def czebyszew(a,b,n):
    X = []
    for k in range(1,n+1):
        xk = 0.5 * (a + b) + 0.5 * (b - a) * math.cos((2 * k - 1) * math.pi / (2 * n))
        X.append(xk)
    return X

def evenly(a,b,n):
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def values(f,X):
    n = len(X)
    return [f(X[i]) for i in range(n)]

def lagrange(X, Y):
    denom = []
    n = len(X)
    for k in range(n):
        m = 1
        for i in range(n):
            if X[i]==X[k]:
                continue
            m *= (X[k]-X[i])
        denom.append(m)
    def f(x):
        y = 0
        for k in range(n):
            d = 1
            for i in range(n):
                if i == k: continue
                d *= (x - X[i])  
            y += d * Y[k] / denom[k]
        return y
    return f

def hermit(X, Y):
    ms = [len(y) for y in Y]
    m = sum(ms)
    n = m - 1
    xs_ = []
    for i in range(len(X)):
        xs_.extend([X[i]] * 2)
    bs = [[None] * m for _ in range(m)]
    i = 0
    for y_list in Y:
        for j in range(len(y_list)):
            for k in range(j + 1):
                bs[i][k] = y_list[k]
            i += 1
    for j in range(1, m):
        for i in range(j, m):
            if bs[i][j] is not None: 
                continue
            bs[i][j] = (bs[i][j - 1] - bs[i - 1][j - 1]) / (xs_[i] - xs_[i - j])
    bs_ = [bs[i][i] for i in range(m)]
    def f(x):
        x_diffs = [x - xi for xi in X]
        y = bs_[0]
        Pl = 1
        deg = 0
        for i, mi in enumerate(ms):
            for _ in range(mi):
                deg += 1
                Pl *= x_diffs[i]
                y += bs_[deg] * Pl
                if deg == n:
                    return y
    return f

def max_diff(X1,X2):
    maksi = float('-inf')
    for i in range(len(X1)):
        maksi= max(maksi,abs(X1[i]-X2[i]))
    return maksi

def max_diff_square(X1,X2):
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
    d = lambda x: derivative(x,m,k)
    X = evenly(a,b,N)
    Y = values(f_to_interpolate,X)

    print("Podaj jak mają być rozmieszczone węzły (0 - równomiernie, 1 - według Czebyszewa): ")
    flag = int(input())
    print("Podaj liczbę węzłów: ")
    n = int(input())

    if flag:
        XI = czebyszew(a,b,n)
    else:
        XI = evenly(a,b,n)

    YI = values(f_to_interpolate,XI)
    YID = values(d,XI)
    
    YIH = [[YI[i],YID[i]] for i in range(n)]
    
    Hermit = hermit(XI,YIH)
    YIVH = values(Hermit,X)
    Lagrange = lagrange(XI,YI)
    YIVL = values(Lagrange,X)

    plt.figure(figsize=(10,5))
    plt.scatter(XI,YI,c='black',label="Węzły interpolacji")
    plt.plot(X,YIVH,c='b',label="Wielomian Hermita")
    plt.plot(X,Y,c='g',label="Funkcja interpolowana")
    plt.legend(loc='best')

    diff_hermit = max_diff(Y,YIVH)
    diff_lagrange = max_diff(Y,YIVL)
    
    square_diff_hermit = max_diff_square(Y,YIVH)/N
    square_diff_lagrange = max_diff_square(Y,YIVL)/N
    
    print(f"diff_hermit: {diff_hermit}\ndiff_lagrange: {diff_lagrange}\nsquare_diff_hermit: {square_diff_hermit}\nsquare_diff_lagrange: {square_diff_lagrange}")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    sns.despine()
    plt.show()
