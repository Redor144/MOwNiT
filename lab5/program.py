import matplotlib.pyplot as plt
import math
import seaborn as sns
from sympy import lambdify, Symbol

# dla danje metody mamy punkt startowy, kryterium i ro dla kryterium parametry do zmian
# dużo wyników może się na mnożyć, wyłapać problemy 
# z bardzo ostraym spadkiem styczna przetnie oś bardzo daleko wyliczenie 
# czegoś takiego może być trudne i trzeba to wyłapać
# jak przyjmować punkty początkowe w siecznych:
# fixujemy a i drugi punkt początkowy zmniejzamy i jeden (i na odwrót b fixujemy)
# możemy zatrzymać się dalego od rzeczywistego punktu zerowergo


def find_roots(func, derivative, start, stop, step, tolerance=1e-8, max_iterations=100_000):
    def newton_method(x0):
        iteracja = 1
        miejsca_zerowe = []

        while iteracja <= max_iterations:
            x1 = x0 - func(x0) / derivative(x0)

            if abs(x1 - x0) < tolerance:
                miejsca_zerowe.append(x1)
                x0 = x1 + 0.1
            else:
                x0 = x1

            iteracja += 1

        return miejsca_zerowe

    miejsca_zerowe = []
    x = start

    while x <= stop:
        miejsca_zerowe += newton_method(x)
        x += step

    miejsca_zerowe = list(set(miejsca_zerowe))
    miejsca_zerowe.sort()

    print("Znalezione miejsca zerowe:")
    for miejsce in miejsca_zerowe:
        print(f"x = {miejsce}")
        

def newton_raphson(f, x0, stop_criterion):
    x_prev = float('inf')
    x_curr = x0
    f_lambd = f
    iters = 0
    
    while not stop_criterion(f_lambd, x_prev, x_curr):
        x_curr, x_prev = init(x_curr), x_curr
        iters += 1
        
    return '{:.6f}'.format(x_curr), iters

def calc_xi2(f, xi0, xi1):
    return xi1 - (xi1 - xi0) / (f(xi1) - f(xi0)) * f(xi1)

def secant_method(f, x0, x1, stop_criterion,max_iter = 12_526_895):
    xi1 = x0
    xi2 = x1
    iters = 0
    
    while not stop_criterion(f, xi1, xi2) and f(xi1)-f(xi2)!=0: 
    # and iters <= max_iter and f(xi1)-f(xi2)!=0:
        xi2, xi1 = calc_xi2(f, xi2, xi1), xi2
        iters += 1
        
    return '{:.6f}'.format(xi2), iters

def printarray(A):
    for a in A:
        print("$"+str(a[0])+"$" + "&" + str(a[1]) + " \\\\")

if __name__ == "__main__":
    n = True
    if n:
        a = 0.1
        b = 1.9
        krok = 0.1
        tolerancja = 1e-8
        maks_iter = 1000

        func = lambda n, m, x: m * x * math.exp(-n) - m * math.exp(-n * x) + 1/m
        derivative = lambda n, m, x: m * n * math.exp(-n*x) + m * math.exp(-n)
        n = 9
        m = 25

        f = lambda x: func(n, m, x)
        d = lambda x : derivative(n, m, x)
        init = lambda x: x - f(x) / d(x)

        step = (b-a) / (999)

        X = [a + i * step for i in range(1000)]
        Y = [f(X[i]) for i in range(1000)]
        YD = [d(X[i]) for i in range(1000)]

        # p = float(input("Podaj dokładność: "))
        # p = .000000000000001

        stop_criterion_init1 = lambda p: lambda _, x_prev, x_curr: abs(x_curr - x_prev) < p
        stop_criterion_init2 = lambda p: lambda f, _, x_curr: abs(f(x_curr)) < p

        # print(secant_method(f, .5, 1, stop_criterion_init1(p)))


        dokladnosc = [1e-02,1e-03,1e-04,1e-05,1e-07,1e-10,1e-15]
        punkty_startowe = [a + i * krok for i in range(19)]


        newton = [[]for i in range(len(punkty_startowe))]

        # for i, b in enumerate(punkty_startowe):
        #     for p in dokladnosc:
        #         newton[i].append(newton_raphson(f,b,stop_criterion_init2(p)))
                
        # newton2 = []
        # popo = 0
        # for row in newton:
        #     x = '{:.1f}'.format(punkty_startowe[popo])+" & "
        #     popo+=1
        #     for i,a in enumerate(row):
        #         x+="$"+str(a[0])+"_{"+str(a[1])+"}$"
        #         if i!=len(row)-1:
        #             x+=" & "
        #         if i==len(row)-1:
        #             x+='\\' + '\\' + '\n' + '\hline'
        #     newton2.append(x)
        
        # def printlatex(A):
        #     for x in A:
        #         print(x)
            
        # printlatex(newton2)


            
        # print(newton)
        # print(newton2)



        # a_stale = [[] for _ in range(len(punkty_startowe)-1)]

        # for i, b in enumerate(punkty_startowe[1:]):
        #     for p in dokladnosc:
        #         a_stale[i].append(newton_raphson(f,punkty_startowe[0],b,stop_criterion_init2(p)))

        # printarray(a_stale)

        # b_stale = [[] for _  in range(len(punkty_startowe)-1)]

        # printarray(b_stale)


        # for i, a in enumerate(punkty_startowe[:len(punkty_startowe)-1]):
        #     for p in dokladnosc:
        #         b_stale[i].append(secant_method(f,a,punkty_startowe[-1],stop_criterion_init1(p)))

        # printarray(b_stale)
        # print(dokladnosc)
        # func = lambda x: 2*x+2
        # derivative = lambda x: 2

        # find_roots(f, d, .5, 1, krok, tolerancja, maks_iter)

        # step = (b-a) / (999)

        # X = [a + i * step for i in range(1000)]
        # Y = [f(X[i]) for i in range(1000)]
        # step = (b-a) / (9999)

        # X = [a + i * step for i in range(10000)]
        # Y = [f(X[i]) for i in range(10000)]

        plt.figure(figsize=(10,5))
        plt.plot(X,Y,c='g',label="Badana funkcja")
        plt.plot(X,YD,c='r',label='Pochodna')
        plt.legend(loc='best')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid()
        sns.despine()
        plt.show()
    else:
        import numpy as np
        from scipy.linalg import solve
        from numpy.linalg import norm

        # Definicja funkcji
        def F(x):
            x1, x2, x3 = x
            return np.array([
                x1**2 + x2**2 - x3**2 - 1,
                x1 - 2*x2**3 + 2*x3**2 + 1,
                2*x1**2 + x2 - 2*x3**2 - 1
            ])

        # Definicja Jacobianu
        def J(x):
            x1, x2, x3 = x
            return np.array([
                [2*x1, 2*x2, -2*x3],
                [1, -6*x2**2, 4*x3],
                [4*x1, 1, -4*x3]
            ])
        
        # Zmodyfikowana metoda Newtona z dodatkową relaksacją i lepszą obsługą osobliwości macierzy
        def newton_method_relaxed(F, J, x0, tol=1e-8, max_iter=2000, epsilon=1e-10, relaxation_factor=0.8):
            x = x0
            for i in range(max_iter):
                Fx = F(x)
                # if norm(Fx) < tol:
                #     return x, i + 1, 'converged', norm(Fx)
                Jx = J(x)
                try:
                    delta_x = solve(Jx, -Fx)
                except np.linalg.LinAlgError:
                    # print(i)
                    return Fx, "ERROR"
                    # Dodanie przesunięcia i próba rozwiązania problemu osobliwości
                    delta_x = solve(Jx + epsilon * np.eye(len(x)), -Fx)
                # x += relaxation_factor * delta_x  # stosowanie czynnika relaksacji
                x+= delta_x
                if norm(delta_x) < tol:
                    return x, i + 1, 'converged', norm(Fx)
            return x, max_iter, 'not converged', norm(Fx)
        
        # Testy z różnymi wektorami początkowymi dla nowego układu równań
#         initial_vectors = [
#             np.array([0.0, 0.0, 0.0]),
#             np.array([1.0, 1.0, 1.0]),
#             np.array([-1.0, 1.0, -0.5]),
#             np.array([0.5, -0.5, 0.5]),
#             np.array([2.0, -2.0, 1.0])
# ]
        initial_vectors1 = [[a/10,b/10,c/10] for a in range(-10,11) for b in range(-10,11) for c in range(-10,11)]
        print(len(initial_vectors1))
        initial_vectors = [np.array(initial_vectors1[i]) for i in range(len(initial_vectors1))] 
        # Ponowne testowanie zmodyfikowanej metody Newtona z różnymi wektorami początkowymi dla nowego układu równań
        relaxed_results = [ (initial_vectors1[i],newton_method_relaxed(F, J, x0)) for i,x0 in enumerate(initial_vectors)]
        # print(relaxed_results)
        # first_m1_1_m1 = []
        # first_m1_1_1 = []
        # first_12_1_m12 = []
        # first_12_1_12 = []
        # errors = []
        # dont_corveges = []
        errorss = 0
        x1=0
        x2=0
        x3=0
        x4=0
        for v in relaxed_results:
            if v[1][1]=="ERROR":
                # errors.append(v)
                errorss+=1
            # elif v[1][2] == "not converged":
            #     dont_corveges.append(v)
            elif np.array_equal(v[1][0],np.array([-1,1,-1])):
                # first_m1_1_m1.append((v[0],v[1][1]))
                x1+=1
            elif np.array_equal(v[1][0],np.array([-1,1,1])):
                # first_m1_1_1.append((v[0],v[1][1]))
                x2+=1
            elif np.array_equal(v[1][0],np.array([1/2,1,-1/2])):
                # first_12_1_m12.append((v[0],v[1][1]))
                x3+=1
            elif np.array_equal(v[1][0],np.array([1/2,1,1/2])):
                # first_12_1_12.append((v[0],v[1][1]))
                x4+=1
        wartosci = [x1,x2,x3,x4]
        kategorie = ['x1','x2','x3','x4']
        plt.figure(figsize=(10, 6))
        plt.bar(kategorie, wartosci, color='blue')

        # Dodanie tytułu i etykiet osi
        # plt.title('Przykładowy Wykres Słupkowy')
        plt.xlabel('rozwiązania')
        plt.ylabel('liczba wektorów')

        # Wyświetlenie wykresu
        plt.show()
        # print("ready")
        # while 1:
        #     n = int(input())
        #     # if n == 0:
        #     #     printarray(errors)
        #     # if n==1:
        #     #     printarray(dont_corveges)
        #     if n==2:     
        #         printarray(first_m1_1_m1)
        #     if n==3:     
        #         printarray(first_m1_1_1)
        #     if n==4:     
        #         printarray(first_12_1_m12)
        #     if n==5:     
        #         printarray(first_12_1_12)
        
        #dla błędnego
        # Metoda Newtona
        # def newton_method(F, J, x0, tol=1e-8, max_iter=50):
        #     x = x0
        #     for i in range(max_iter):
        #         Fx = F(x)
        #         if norm(Fx) < tol:
        #             return x, i + 1, norm(Fx)
        #         Jx = J(x)
        #         delta_x = solve(Jx, -Fx)
        #         x += delta_x
        #         if norm(delta_x) < tol:
        #             return x, i + 1, norm(Fx)
        #     return x, max_iter, norm(Fx)
        # Zmodyfikowana metoda Newtona z obsługą osobliwej macierzy Jacobiego
        # def newton_method_modified(F, J, x0, tol=1e-8, max_iter=1000, epsilon=1e-10):
        #     x = x0
        #     for i in range(max_iter):
        #         Fx = F(x)
        #         if norm(Fx) < tol:
        #             return x, i + 1, 'converged', norm(Fx)
        #         Jx = J(x)
        #         # Dodanie małego przesunięcia do macierzy Jacobiego w przypadku osobliwości
        #         try:
        #             delta_x = solve(Jx, -Fx)
        #         except np.linalg.LinAlgError:
        #             delta_x = solve(Jx + epsilon*np.eye(len(x)), -Fx)
        #         x += delta_x
        #         if norm(delta_x) < tol:
        #             return x, i + 1, 'converged', norm(Fx)
        #     return x, max_iter, 'not converged', norm(Fx)


        # Testy z różnymi wektorami początkowymi
        # initial_vectors = [
        #     np.array([0.0, 0.0, 0.0]),
        #     np.array([1.0, 1.0, 1.0]),
        #     np.array([-1.0, 1.0, 0.5]),
        #     np.array([0.5, -0.5, 0.5]),
        #     np.array([2.0, -2.0, 1.0]),
        #     np.array([0.8 ,0.6, -0.2]),
        #     np.array([1.2, -0.8, -1.1])
        # ]

        # # results = [newton_method(F, J, x0) for x0 in initial_vectors]
        # # results
        # # Ponowne testy z różnymi wektorami początkowymi
        # modified_results = [newton_method_modified(F, J, x0) for x0 in initial_vectors]
        # printarray(modified_results)
        