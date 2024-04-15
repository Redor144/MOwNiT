# zagadnienie Hermita, z tylko jedną pochodną, styczna jest taka sama więc muszą się gładko stykać
# ważny jest stopień wielomianu to 2n-1 bo mamy wartosci funkcji i pochodnej
# Inerpolacja Hermita realizowana jest na jednym wzorze tzn albo Lagrangea albo Hermita

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
    derivative = lambda x,m,k: 2 * x / k + 10 * m * math.sin(k * x) * k
    derivative_of_f = lambda x: derivative(x,m,k)
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


    for i in range(2,1860):

        XIC = czebyszew(a,b,i)
        XIR = evenly(a,b,i)

        YIC = values(f_to_interpolate,XIC)
        YICD = values(derivative_of_f,XIC)

        YICC = [[YIC[i],YICD[i]] for i in range(len(YIC))]

        YIR = values(f_to_interpolate,XIR)
        YIRD = values(derivative_of_f,XIR)

        YIRR = [[YIR[i],YIRD[i]] for i in range(len(YIR))]

        NewtonR = hermit(XIR,YIRR)
        NewtonC = hermit(XIC,YICC)

        YINR = values(NewtonR,X)
        YINC = values(NewtonC,X)


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
        print(i)
 
    print("min_w_N_R: ", min_w_N_R, "min_N_R: ", min_N_R)

    print("min_w_N_C: ", min_w_N_C, "min_N_C: ", min_N_C)

    print("min_sq_w_N_R: ", min_sq_w_N_R, "min_sq_N_R: ", min_sq_N_R)

    print("min_sq_w_N_C: ", min_sq_w_N_C, "min_sq_N_C: ", min_sq_N_C)

if __name__ == "__main__":

    print("Wyznaczenia najlepszego wielomianu podaj 0, wyznaczenia wielomanu dla konretnej liczby węzłów wybierz 1: ")
    to_do = int(input())
    if to_do == 1:

        function = lambda x,m,k: 10 * m + (x**2) / k - 10 * m * math.cos(k * x)
        derivative = lambda x,m,k: 2 * x / k + 10 * m * math.sin(k * x) * k
        m = 2
        k = 2
        a = -3 * math.pi
        b = 3 * math.pi
        x = [a,b]
        N = 1000

        f_to_interpolate = lambda x: function(x,m,k)
        derivative_of_f = lambda x: derivative(x,m,k)
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
        YID = values(derivative_of_f,XI)
        YtoH = [[YI[i],YID[i]] for i in range(n)]
        Hermit = hermit(XI,YtoH)

        YIVn = values(Hermit,X)

        print("Błąd względny Hermite'a: ",max_diff(Hermit,f_to_interpolate,X))
        print("Ten drugi Hermite'a: ",max_diff_square(Hermit,f_to_interpolate,X)/N)

        plt.figure(figsize=(10,5))
        plt.scatter(XI,YI,c='black',label="Węzły interpolacji")
        plt.plot(X,YIVn,c='b',label="Wielomian Hermita")
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