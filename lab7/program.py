# # import numpy as np

# # def create_matrix_A(n, k, m):
# #     A = np.zeros((n, n))
# #     for i in range(n):
# #         for j in range(n):
# #             if i == j:
# #                 A[i, j] = k
# #             else:
# #                 A[i, j] = 1 / (abs(i - j) + m)
# #     return A

# # def create_vector_b(n):
# #     # Creating a permutation of {1, -1} of length n
# #     b = np.random.choice([1, -1], size=n)
# #     return b

# # def jacobi_method(A, b, x0, rho, max_iterations=1000):
# #     n = len(b)
# #     x = x0.copy()
# #     D = np.diag(np.diag(A))
# #     R = A - D
    
# #     for iteration in range(max_iterations):
# #         x_new = np.linalg.inv(D).dot(b - R.dot(x))
        
# #         # Stopping criteria 1
# #         if np.linalg.norm(x_new - x, ord=np.inf) < rho:
# #             break
# #         # Stopping criteria 2
# #         if np.linalg.norm(A.dot(x_new) - b, ord=np.inf) < rho:
# #             break
        
# #         x = x_new
    
# #     return x, iteration

# # def spectral_radius(matrix):
# #     eigenvalues = np.linalg.eigvals(matrix)
# #     return max(abs(eigenvalues))

# # # Example usage
# # n = 5  # Size of the system
# # k = 4  # Diagonal value
# # m = 1  # Parameter for off-diagonal elements
# # rho = 1e-5  # Stopping criteria
# # x0 = np.zeros(n)  # Initial guess

# # A = create_matrix_A(n, k, m)
# # b = create_vector_b(n)

# # # Solving using Jacobi method
# # solution, iterations = jacobi_method(A, b, x0, rho)
# # print(f'Solution: {solution}\nIterations: {iterations}')

# # # Spectral radius of the iteration matrix
# # D = np.diag(np.diag(A))
# # L_plus_U = A - D
# # T = -np.linalg.inv(D).dot(L_plus_U)
# # radius = spectral_radius(T)
# # print(f'Spectral Radius: {radius}')

# import numpy as np

# def generate_matrix_A(n, k, m):
#     A = np.zeros((n, n))
#     for i in range(n):
#         for j in range(n):
#             if i == j:
#                 A[i, j] = k
#             else:
#                 A[i, j] = 1 / (abs(i - j) + m)
#     return A

# def generate_vector_b(A, x):
#     return A @ x

# def jacobi_method(A, b, x0, rho, max_iterations=1000):
#     D = np.diag(np.diag(A))
#     R = A - D
#     D_inv = np.linalg.inv(D)
#     x = x0.copy()
#     for i in range(max_iterations):
#         x_new = D_inv @ (b - R @ x)
#         if np.linalg.norm(x_new - x, ord=np.inf) < rho:
#             return x_new, i + 1
#         if np.linalg.norm(A @ x_new - b, ord=np.inf) < rho:
#             return x_new, i + 1
#         x = x_new
#     return x, max_iterations

# def spectral_radius(matrix):
#     eigenvalues = np.linalg.eigvals(matrix)
#     return max(abs(eigenvalues))

# # Parameters for testing
# n = 5  # Example size
# k = 2  # Example value for k
# m = 1  # Example value for m
# rho = 1e-5  # Example tolerance

# # Generate matrix A and vector b
# A = generate_matrix_A(n, k, m)
# x_true = np.random.choice([1, -1], size=n)
# b = generate_vector_b(A, x_true)
# x0 = np.zeros(n)  # Initial guess

# # Apply Jacobi method
# x_approx, iterations = jacobi_method(A, b, x0, rho)

# # Calculate spectral radius
# D = np.diag(np.diag(A))
# R = A - D
# iteration_matrix = -np.linalg.inv(D) @ R
# spectral_radius_value = spectral_radius(iteration_matrix)

# x_approx, iterations, spectral_radius_value


# import time

# def perform_tests(sizes, k_values, m_values, rhos, max_iterations=1000):
#     results = []

#     for n in sizes:
#         for k in k_values:
#             for m in m_values:
#                 for rho in rhos:
#                     A = generate_matrix_A(n, k, m)
#                     x_true = np.random.choice([1, -1], size=n)
#                     b = generate_vector_b(A, x_true)
#                     x0 = np.zeros(n)  # Initial guess

#                     start_time = time.time()
#                     x_approx, iterations = jacobi_method(A, b, x0, rho, max_iterations)
#                     end_time = time.time()

#                     # Calculate spectral radius
#                     D = np.diag(np.diag(A))
#                     R = A - D
#                     iteration_matrix = -np.linalg.inv(D) @ R
#                     spectral_radius_value = spectral_radius(iteration_matrix)

#                     result = {
#                         "size": n,
#                         "k": k,
#                         "m": m,
#                         "rho": rho,
#                         "iterations": iterations,
#                         "spectral_radius": spectral_radius_value,
#                         "time_seconds": end_time - start_time,
#                         "approx_solution": x_approx,
#                         "true_solution": x_true,
#                         "error": np.linalg.norm(x_approx - x_true, ord=np.inf)
#                     }
#                     results.append(result)
#     return results

# # Define parameter ranges
# sizes = [5, 10, 15]
# k_values = [2, 3, 4]
# m_values = [1, 2, 3]
# rhos = [1e-5, 1e-6, 1e-7]

# # Perform tests
# test_results = perform_tests(sizes, k_values, m_values, rhos)

# import pandas as pd
# # import ace_tools as tools

# # Convert results to DataFrame and display
# df_results = pd.DataFrame(test_results)
# print(df_results.to_string(index=False))
# # tools.display_dataframe_to_user(name="Jacobi Method Test Results", dataframe=df_results)

import numpy as np
import pandas as pd
import time

np.random.seed(42)

def generate_matrix_A(n, k, m):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = k
            else:
                A[i, j] = 1 / (abs(i - j) + m)
    return A

def generate_vector_b(A, x):
    return A @ x

def jacobi_method(A, b, x0, rho, max_iterations=1000):
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.linalg.inv(D)
    x = x0.copy()
    for i in range(max_iterations):
        x_new = D_inv @ (b - R @ x)
        if np.linalg.norm(x_new - x, ord=2) < rho:
            return x_new, i + 1
        # if np.linalg.norm(A @ x_new - b, ord=2) < rho:
        #     return x_new, i + 1
        x = x_new
    return x, max_iterations

def spectral_radius(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return max(abs(eigenvalues))

def perform_tests(sizes, k_values, m_values, rhos, max_iterations=1000):
    results = []
    results2 = []
    for n in sizes:
        print(n)
        for k in k_values:
            for m in m_values:
                result2 = [n]
                A = generate_matrix_A(n, k, m)
                x_true = np.random.choice([1, -1], size=n)
                b = generate_vector_b(A, x_true)
                # x0 = np.zeros(n)
                x0 = np.random.choice([-100,100],size=n)
                # D = np.diag(np.diag(A))
                # R = A - D
                # iteration_matrix = -np.linalg.inv(D) @ R
                # spectral_radius_value = spectral_radius(iteration_matrix)
                for rho in rhos:
                    # A = generate_matrix_A(n, k, m)
                    # x_true = np.random.choice([1, -1], size=n)
                    # b = generate_vector_b(A, x_true)
                    # x0 = np.zeros(n)  # Initial guess
                    # x0 = np.random.choice([-100,100],size=n)
                    # start_time = time.perf_counter()
                    x_approx, iterations = jacobi_method(A, b, x0, rho, max_iterations)
                    # end_time = time.perf_counter()

                    # Calculate spectral radius
                    # D = np.diag(np.diag(A))
                    # R = A - D
                    # iteration_matrix = -np.linalg.inv(D) @ R
                    # spectral_radius_value = spectral_radius(iteration_matrix)

                    # result = {
                    #     "size": n,
                    #     "k": k,
                    #     "m": m,
                    #     "rho": rho,
                    #     "iterations": iterations,
                    #     "spectral_radius": spectral_radius_value,
                    #     "time_seconds": end_time - start_time,
                    #     "approx_solution": x_approx,
                    #     "true_solution": x_true,
                    #     "error": np.linalg.norm(x_approx - x_true, ord=np.inf)
                    # }
                    result = {
                        "rho": rho,
                        "size": n,
                        "iterations": iterations,
                        # "spectral_radius": spectral_radius_value,
                        # "time_seconds": end_time - start_time,
                        "error": np.linalg.norm(x_approx - x_true, ord=2)
                    }
                    result2.append(result['error'])
                    results.append(result)
                # result2.append(result['spectra'])
                results2.append(result2)
    return results, results2

# Define parameter ranges
# sizes = [i for i in range(4000,4001)]
sizes = [i for i in range(600,2001,100)]
k_values = [10]
m_values = [5]
rhos = [1e-1, 1e-2,  1e-3, 1e-5, 1e-10, 1e-15]
# rhos = [1e-1]

# Perform tests
test_results, test_results2 = perform_tests(sizes, k_values, m_values, rhos)

# Convert results to DataFrame and display
df_results = pd.DataFrame(test_results)

to_graph = []

for x in test_results2:
    to_graph.append(x[1])
    # print(str(x[0]) + " & " + '{:.4e}'.format(x[1]) + " \\\\")
    # print(str(x[0]) + " & " + str(x[1]) + " & " + str(x[2]) + " & " + str(x[3]) + " & " + str(x[4]) + " & " + str(x[5]) + " & " + str(x[6]) + " \\\\")
    print(str(x[0]) + " & " + '{:.4e}'.format(x[1]) + " & " + '{:.4e}'.format(x[2]) + " & " + '{:.4e}'.format(x[3]) + " & " + '{:.4e}'.format(x[4]) + " & " + '{:.4e}'.format(x[5]) + " & " + '{:.4e}'.format(x[6]) + " \\\\")
    print("\\hline")
    
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6))
# plt.scatter(sizes, to_graph, color='teal', alpha=0.7)
# # plt.scatter(N, diff_float64, label='układ 2', color='teal', alpha=0.7)

# # # Customize the plot
# # plt.title('Wartości błędów - skala liniowa')
# plt.yscale('log')
# plt.xlabel('wartość n')
# plt.ylabel('promień spektralny')
# plt.grid(True, which="both", ls="--")

# # # Show the plot
# plt.show()
# Display the DataFrame
# print(df_results.to_string(index=False))
# with open('Jacobi_Method_Test_Results.txt', 'w') as f:
#     f.write(df_results.to_string(index=False))
