import math
import matplotlib.pyplot as plt
from copy import deepcopy
import seaborn as sns
# n - małe liczba węzlów, N - liczba punktów dla których liczymy
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

def newton(X, Y):
    n = len(X)
    diff_quo = deepcopy(Y)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            diff_quo[i] = (diff_quo[i] - diff_quo[i - 1]) / (X[i] - X[i - j])
    print(diff_quo)
    def f(x):
        x_diffs = [1] + [x - X[i] for i in range(n - 1)]
        y = 0
        x_coeff = 1
        for j in range(n):
            x_coeff *= x_diffs[j]
            y += diff_quo[j] * x_coeff
        return y
    return f

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


    min_N_R = float('inf')
    min_w_N_R = 2

    min_N_C = float('inf')
    min_w_N_C = 2

    min_sq_N_R = float('inf')
    min_sq_w_N_R = 2

    min_sq_N_C = float('inf')
    min_sq_w_N_C = 2

    min_L_R = float('inf')
    min_w_L_R = 2

    min_L_C = float('inf')
    min_w_L_C = 2

    min_sq_L_R = float('inf')
    min_sq_w_L_R = 2

    min_sq_L_C = float('inf')
    min_sq_w_L_C = 2


    for i in range(2,300):
        XIC = czebyszew(a,b,i)
        XIR = evenly(a,b,i)
        YIC = values(f_to_interpolate,XIC)
        YIR = values(f_to_interpolate,XIR)
        NewtonR = newton(XIR,YIR)
        NewtonC = newton(XIC,YIC)
        LagrangeR = lagrange(XIR,YIR)
        LagrangeC = lagrange(XIC,YIC)

        YINR = values(NewtonR,X)
        YINC = values(NewtonC,X)

        YILR = values(LagrangeR,X)
        YILC = values(LagrangeC,X)

        new_min_N_R = max_diff_1(YINR,Y)
        if new_min_N_R < min_N_R:
            min_N_R = new_min_N_R
            min_w_N_R = i

        new_min_N_C = max_diff_1(YINC,Y)
        if new_min_N_C < min_N_C:
            min_N_C = new_min_N_C
            min_w_N_C = i

        new_min_sq_N_R = max_diff_square_1(YINR,Y)
        if new_min_sq_N_R < min_sq_N_R:
            min_sq_N_R = new_min_sq_N_R
            min_sq_w_N_R = i

        new_min_sq_N_C = max_diff_square_1(YINC,Y)
        if new_min_sq_N_C < min_sq_N_C:
            min_sq_N_C = new_min_sq_N_C
            min_sq_w_N_C = i

        new_min_L_R = max_diff_1(YILR,Y)
        if new_min_L_R < min_L_R:
            min_L_R = new_min_L_R
            min_w_L_R = i

        new_min_L_C = max_diff_1(YILC,Y)
        if new_min_L_C < min_L_C:
            min_L_C = new_min_L_C
            min_w_L_C = i

        new_min_sq_L_R = max_diff_square_1(YILR,Y)
        if new_min_sq_L_R < min_sq_L_R:
            min_sq_L_R = new_min_sq_L_R
            min_sq_w_L_R = i

        new_min_sq_L_C = max_diff_square_1(YILC,Y)
        if new_min_sq_L_C < min_sq_L_C:
            min_sq_L_C = new_min_sq_L_C
            min_sq_w_L_C = i
    print("min_w_N_R: ", min_w_N_R, "min_N_R: ", min_N_R)

    print("min_w_N_C: ", min_w_N_C, "min_N_C: ", min_N_C)

    print("min_sq_w_N_R: ", min_sq_w_N_R, "min_sq_N_R: ", min_sq_N_R)

    print("min_sq_w_N_C: ", min_sq_w_N_C, "min_sq_N_C: ", min_sq_N_C)

    print("min_w_L_R: ", min_w_L_R, "min_L_R: ", min_L_R)

    print("min_w_L_C: ", min_w_L_C, "min_L_C: ", min_L_C)

    print("min_sq_w_L_R: ", min_sq_w_L_R, "min_sq_L_R: ", min_sq_L_R)

    print("min_sq_w_L_C: ", min_sq_w_L_C, "min_sq_L_C: ", min_sq_L_C)

if __name__ == "__main__":

    print("Wyznaczenia najlepszego wielomianu podaj 0, wyznaczenia wielomanu dla konretnej liczby węzłów wybierz 1: ")
    to_do = int(input())

    if to_do == 1:

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

        print("Podaj jak mają być rozmieszczone węzły (0 - równomiernie, 1 - według Czebyszewa): ")
        flag = int(input())
        print("Podaj liczbę węzłów: ")
        n = int(input())

        if flag:
            XI = czebyszew(a,b,n)
        else:
            XI = evenly(a,b,n)

        YI = values(f_to_interpolate,XI)
        Newton = newton(XI,YI)
        YIVn = values(Newton,X)
        Lagrange = lagrange(XI,YI)
        YIVl = values(Lagrange,X)

        print("Błąd względny Newton: ",max_diff(Newton,f_to_interpolate,X))
        print("Błąd względny Lagrange: ", max_diff(Lagrange,f_to_interpolate,X))
        print("Ten drugi Lagrange: ", max_diff_square(Lagrange,f_to_interpolate,X)/N)
        print("Ten drugi Newton: ",max_diff_square(Newton,f_to_interpolate,X)/N)

        plt.figure(figsize=(10,5))
        plt.scatter(XI,YI,c='black',label="Węzły interpolacji")
        plt.plot(X,YIVn,c='b',label="Wielomian Newtona")
        plt.plot(X,YIVl,c='r',label="Wielomian Lagranga")
        plt.plot(X,Y,c='g',label="Funkcja interpolowana")
        plt.legend(loc='best')

        if flag:
            plt.title("Węzły zgodnie z Czebyszewem")
        else:
            plt.title("Węzły równomierne")

        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        sns.despine()
        plt.show()
    else:
        find_best_function();