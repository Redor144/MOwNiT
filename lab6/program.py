import time
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
inpu = int(input("Podaj 0 - obliczenia dla 3 układu, 1 - obliczenia dla 1 i 2 układu: "))
boolean = bool(inpu)

def maximum_error(x_original, x_computed):
    """Oblicza maksymalny błąd między wektorem x_original a x_computed."""
    return np.max(np.abs(x_original - x_computed))

def generate_vector_x(n):
    """Generuje wektor x jako permutację {1, -1} o długości n."""
    return np.random.choice([-1, 1], n)

def compute_vector_b(A, x):
    """Oblicza wektor b = Ax."""
    return np.dot(A, x)

def gauss(A, B, dtype=np.float64):
        n = len(A)
        AB = np.hstack([A, B.reshape((n, 1))]).astype(dtype)

        for i in range(n):
            pivot = AB[i, i]
            for j in range(i + 1, n):
                base = AB[j, i] / pivot
                AB[j] -= base * AB[i]

        X = AB[:, n]
        X[n - 1] /= AB[n - 1, n - 1]
        for i in range(n - 2, -1, -1):
            pivot = AB[i, i]
            X[i] -= (AB[i, i + 1:n] * X[i + 1:n]).sum()
            X[i] /= pivot

        return X

if boolean:

    def generate_matrix_A_with_dtype_1(n, dtype=np.float64):
        """Generuje macierz A o wymiarach nxn zgodnie z zadanymi wzorami z określoną precyzją."""
        A = np.zeros((n, n), dtype=dtype)
        for i in range(n):
            for j in range(n):
                if i == 0:
                    A[i, j] = dtype(1)
                else:
                    A[i, j] = dtype(1 / (i + j + 1))
        return A

    def generate_matrix_A_with_dtype_2(n, dtype=np.float64):
        """Generuje macierz A o wymiarach nxn zgodnie z zadanymi wzorami z określoną precyzją."""
        A = np.zeros((n, n), dtype=dtype)
        for i in range(n):
            for j in range(n):
                if j>=i:
                    A[i, j] = dtype(2*(i+1)/(j+1))
        for i in range(n):
            for j in range(n):
                if j<i:
                    A[i, j] = A[j, i]
        return A
    

    def infinity_norm(A):
        """Oblicza normę nieskończoność macierzy A."""
        return np.max(np.sum(np.abs(A), axis=1))

    def condition_number(A):
        """Oblicza współczynnik uwarunkowania macierzy A używając normy nieskończoność."""
        norm_A = infinity_norm(A)
        try:
            A_inv = np.linalg.inv(A)
            norm_A_inv = infinity_norm(A_inv)
        except np.linalg.LinAlgError:
            return np.inf  # Macierz A jest osobliwa i nie ma odwrotności
        return norm_A * norm_A_inv


    N = []
    diff_float32 = []
    diff_float64 = []

    # Przykładowy rozmiar układu
    for n in range(2, 21):
        
        # x_original = generate_vector_x(n)

        # A_float32 = generate_matrix_A_with_dtype_2(n, dtype=np.float32)
        # # A_float32 = generate_matrix_A_1(n)
        # max_error_float32 = condition_number(A_float32)
        # b_float32 = compute_vector_b(A_float32, x_original)
        # x_computed_float32 = gaussian_elimination(A_float32, b_float32)

        # # Sprawdzenie dla float64 (domyślnie)
        # A_float64 = generate_matrix_A_with_dtype_2(n, dtype=np.float64)
        # # A_float64 = generate_matrix_A_2(n)
        # max_error_float64 = condition_number(A_float64)
        # b_float64 = compute_vector_b(A_float64, x_original)
        # x_computed_float64 = gaussian_elimination(A_float64, b_float64)
    
        # max_error_float32 = maximum_error(x_original, x_computed_float32)
        # max_error_float64 = maximum_error(x_original, x_computed_float64)
        
        A = generate_matrix_A_with_dtype_1(n, dtype=np.float64)
        B = generate_matrix_A_with_dtype_2(n, dtype=np.float64)
        
        # print(str(n) + " & " + '{:.4e}'.format(max_error_float32) + " & " + '{:.4e}'.format(max_error_float64) + " \\\\")
        print(str(n) + " & " + '{:.4e}'.format(condition_number(A)) + " & " + '{:.4e}'.format(condition_number(B)) + " \\\\")
        print(r'\hline')
        # print(n, x_original)'{:.6f}'.format(xi2)
        # print(x_computed_float32, max_error_float32)
        # print(x_computed_float64, max_error_float64)
        N.append(n)
        diff_float32.append(condition_number(A))
        diff_float64.append(condition_number(B))
        # print(n)
        # x_original, x_computed_float32, np.linalg.norm(x_original - x_computed_float32), x_computed_float64, np.linalg.norm(x_original - x_computed_float64)
    plt.figure(figsize=(12, 6))
    plt.scatter(N, diff_float32, label='układ 1', color='orange', alpha=0.7)
    plt.scatter(N, diff_float64, label='układ 2', color='teal', alpha=0.7)

    # # Customize the plot
    # plt.title('Wartości błędów - skala liniowa')
    plt.xlabel('wartość n')
    # plt.ylabel('współczynnik uwarunkowania')
    # plt.legend()
    # plt.grid(True, which="both", ls="--")

    # # Show the plot
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.scatter(N, diff_float32, label='układ 1', color='orange', alpha=0.7)
    # plt.scatter(N, diff_float64, label='układ 2', color='teal', alpha=0.7)

    # # Set logarithmic scale for y-axis
    plt.yscale('log')

    # # Customize the plot
    ##plt.title('Wartości błędów - skala logarytmiczna')
    plt.xlabel('wartość n')
    plt.ylabel('współczynnik uwarunkowania')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # # Show the plot
    plt.show()
    # # Generowanie macierzy A, wektora x i obliczanie b
    # # A = generate_matrix_A(n)
    # # x_original = generate_vector_x(n)
    # # b = compute_vector_b(A, x_original)

    # # # Rozwiązanie Ax = b za pomocą eliminacji Gaussa
    # # x_computed = gaussian_elimination(A, b)

    # # print(x_original, x_computed, np.linalg.norm(x_original - x_computed))

else:
    def generate_matrix_A(n,k=4,m=5):
        A = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i==j:
                    A[i,j] = k
                elif j==i+1:
                    A[i,j] = 1/((i+1)+m)
                elif j==i-1 and i>0:
                    A[i,j] = k/((i+1)+m+1)     
        return A
    def generate_matrix_A_with_dtype_1(n, k=4, m=5, dtype=np.float64):
        """Generuje macierz A o wymiarach nxn zgodnie z zadanymi wzorami z określoną precyzją."""
        A = np.zeros((n, n), dtype=dtype)
        for i in range(n):
            for j in range(n):
                if i==j:
                    A[i,j] = k
                elif j==i+1:
                    A[i,j] = 1/((i+1)+m)
                elif j==i-1 and i>0:
                    A[i,j] = k/((i+1)+m+1)
        return A
    
    def generate_thomas_matrix(n, k=4, m=5, dtype=np.float64):
        a = np.zeros(n-1, dtype=dtype)
        b = np.zeros(n, dtype=dtype)
        c = np.zeros(n-1, dtype=dtype)
        for i in range(n-1):
            c[i] = 1 / (i + 1 +m)
            a[i] = k / (i+3+m)
            b[i] = k
        b[n-1] = k
        return a,b,c
    
    def thomas_algorithm(a, b, c, d, dtype=np.float64):
        n = len(d)
        
        # Ensure the arrays are of the correct dtype
        a = a.astype(dtype)
        b = b.astype(dtype)
        c = c.astype(dtype)
        d = d.astype(dtype)
        
        # Forward sweep
        for i in range(1, n):
            m = a[i-1] / b[i-1]
            b[i] = b[i] - m * c[i-1]
            d[i] = d[i] - m * d[i-1]
        
        # Back substitution
        x = np.zeros(n, dtype=dtype)
        x[-1] = d[-1] / b[-1]
        for i in range(n-2, -1, -1):
            x[i] = (d[i] - c[i] * x[i+1]) / b[i]
        
        return x

    N = []
    time_float32 = []
    time_float64 = []
    
    for n in range(2,1001):
        x_original = generate_vector_x(n)
        
        A_32 = generate_matrix_A_with_dtype_1(n,dtype=np.float32)
        d_32 = compute_vector_b(A_32,x_original)
        # start_time_32 = time.perf_counter()
        # x_computed_32 = gauss(A_32,d_32,dtype=np.float32)
        # end_time_32 = time.perf_counter()
        a_32,b_32,c_32 = generate_thomas_matrix(n,dtype=np.float32)
        start_time_32_thomas = time.perf_counter()
        x_computed_32_thomas = thomas_algorithm(a_32,b_32,c_32,d_32,np.float32)
        end_time_32_thomas = time.perf_counter()
        
        
        A_64 = generate_matrix_A_with_dtype_1(n,dtype=np.float64)
        d_64 = compute_vector_b(A_64,x_original)
        # start_time_64 = time.perf_counter()
        # x_computed_64 = gauss(A_64,d_64,dtype=np.float64)
        # end_time_64 = time.perf_counter()
        a_64,b_64,c_64 = generate_thomas_matrix(n,dtype=np.float64)
        start_time_64_thomas = time.perf_counter()
        x_computed_64_thomas = thomas_algorithm(a_64,b_64,c_64,d_64,np.float64)
        end_time_64_thomas = time.perf_counter()
        
        # max_32_gauss = end_time_32-start_time_32
        max_32_thomas = end_time_32_thomas-start_time_32_thomas
        # max_64_gauss = end_time_64-start_time_64
        max_64_thomas = end_time_64_thomas-start_time_64_thomas
        # print(str(n) + " & " + '{:.4e}'.format(max_32_gauss) + " & " + '{:.4e}'.format(max_32_thomas) + " & " + '{:.4e}'.format(max_64_gauss) + " & " + '{:.4e}'.format(max_64_thomas) + " \\\\")
        # print('\\hline')
        # print('{:.4e}'.format(max_error_float32))
        # a_32,b_32,c_32 = generate_thomas_matrix(n,dtype=np.float32)
        # A_32 = generate_matrix_A_with_dtype_1(n, dtype=np.float32)
        # d_32 = compute_vector_b(A_32,x_original)
        # # scratch_32 = np.zeros(n,dtype=np.float32)
        # start_time_32 = time.perf_counter()
        # x_computed_32 = thomas_algorithm(a_32,b_32,c_32,d_32,np.float32)
        # # x_computed_32 = thomas_algorithm(n,d_32,a_32,b_32,c_32,scratch_32)
        # # x_computed_32 = thomas_algorithm_full_matrix_float32(a,b,c,d_32)
        # # x_computed_32 = thomas_algorithm_diagonals(a_32,b_32,c_32,d_32)
        # end_time_32 = time.perf_counter()

        # a_64,b_64,c_64 = generate_thomas_matrix(n)
        # A_64 = generate_matrix_A_with_dtype_1(n)
        # d_64 = compute_vector_b(A_64,x_original)
        # # scratch_64 = np.zeros(n,dtype=np.float64)
        # start_time_64 = time.perf_counter()
        # x_computed_64 = thomas_algorithm(a_64,b_64,c_64,d_64,np.float64)
        # # x_computed_64 = thomas_algorithm(n,d_64,a_64,b_64,c_64,scratch_64)
        # # x_computed_64 = thomas_algorithm_diagonals(a_64,b_64,c_64,d_64)
        # end_time_64 = time.perf_counter()
        # # print(x_original,x_computed_32)
        
        # N.append(n)
        # time_float32.append(end_time_32-start_time_32)
        # time_float64.append(end_time_64-start_time_64)
        # A_float32 = generate_matrix_A_with_dtype_1(n, dtype=np.float32)
        # # max_error_float32 = condition_number(A_float32)
        # b_float32 = compute_vector_b(A_float32, x_original)
        
        # start_time_32 = time.perf_counter()
        # x_computed_32 = gaussian_elimination(A_float32, b_float32)
        # end_time_32 = time.perf_counter()
        # # Sprawdzenie dla float64 (domyślnie)
        # A_float64 = generate_matrix_A_with_dtype_1(n, dtype=np.float64)
        # # A_float64 = generate_matrix_A_2(n)
        # # max_error_float64 = condition_number(A_float64)
        # # start_time_64 = time.perf_counter()
        # b_float64 = compute_vector_b(A_float64, x_original)
        
        # start_time_64 = time.perf_counter()
        # x_computed_64 = gaussian_elimination(A_float64, b_float64)
        # end_time_64 = time.perf_counter()
        # print(x_original,x_computed_32)
        # max_error_float32 = maximum_error(x_original, x_computed_32)
        # max_error_float64 = maximum_error(x_original, x_computed_64)
        N.append(n)
        time_float32.append(max_32_thomas)
        time_float64.append(max_64_thomas)
        print(n)
        # print(str(n) + " & " + '{:.4e}'.format(max_error_float32) + " & " + '{:.4e}'.format(max_error_float64) + " & " + '{:.4e}'.format(end_time_32-start_time_32) + " & " + '{:.4e}'.format(end_time_64-start_time_64) + " \\\\")
        # print("\hline")
    plt.figure(figsize=(12, 6))
    plt.scatter(N, time_float32, label='float32', color='orange', alpha=0.7)
    plt.scatter(N, time_float64, label='float64', color='teal', alpha=0.7)

    # Customize the plot
    # plt.title('Wartości błędów - skala liniowa')
    plt.xlabel('wartość n')
    plt.ylabel('czas obliczeń [s]')
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Show the plot
    plt.show()