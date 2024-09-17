import numpy as np

def jacobi_method_precise(A, b, x_init, tol=1e-10, max_iter=10000):
    D = np.diag(A)  
    R = A - np.diagflat(D) 
    x = x_init.copy() 
    for i in range(max_iter):
        x_new = (b - np.dot(R, x)) / D 
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, i 
        x = x_new
    return x, max_iter
A = np.array([[0.52, 0.20, 0.25], 
              [0.30, 0.50, 0.20], 
              [0.18, 0.30, 0.55]])

b = np.array([4800, 5810, 5690])
x_init = np.zeros_like(b)
solution_precise, iterations_precise = jacobi_method_precise(A, b, x_init)
print(f"Solución: {solution_precise}")
print(f"Iteraciones: {iterations_precise}")
