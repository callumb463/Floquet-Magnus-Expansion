import numpy as np
from sympy import bernoulli, factorial
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import customtkinter as ctk
from scipy.integrate import cumulative_trapezoid

def __main__():
    ctk.set_appearance_mode("dark")  # "light" or "system"
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("Floquet-Magnus")
    app.geometry("400x300")
    
    label = ctk.CTkLabel(app, text="Enter Hamiltonian:", fg_color="transparent")
    label.pack()
    hamiltonian = ctk.CTkEntry(app, placeholder_text="1/2 * w^2 * x^2",state="normal")
    hamiltonian.pack()

    app.mainloop()

def H_func(t):
    Sx = (1/np.sqrt(2)) * np.array([[0, 1, 0],
                                    [1, 0, 1],
                                    [0, 1, 0]], dtype=complex)
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0,-1]], dtype=complex)
    # Static Sz + oscillating Sx — these don't commute
    return Sz + 0.3 * np.sin(5*t) * Sx


def floquet_magnus_expansion(H_func, t_grid, order):
    dim = H_func(0).shape[0]
    B = [float(bernoulli(n)/factorial(n)) for n in range(order + 1)]
    H = np.zeros((len(t_grid), dim, dim), dtype=complex)
    for i, t in enumerate(t_grid):
        H[i] = H_func(t)
    F = np.empty(order, dtype=object)
    G = np.empty(order+1, dtype=object)
    W = np.empty((order+1, order+1), dtype=object)
    T = np.empty((order+1, order+1), dtype=object)
    Lambda = np.empty(order, dtype=object)
    

    # Initialize F[0], G[0], T[0, 0], W[n, 0], and Lambda[0]
    F[0] = np.mean(H, axis=0)
    G[0] = H.copy()
    T[0, 0] = F[0]
    W[0, 0] = H.copy()
    for i in range(order+1):
        W[i, 0] = np.zeros((N_t, dim, dim), dtype=complex)
    Lambda[0] = cumulative_trapezoid(H - F[0][np.newaxis, :, :], t_grid, axis=0, initial=0)

    
    #Iterate to compute higher order terms
    for n in range(2, order + 1):
        for j in range(1, n):
            for m in range(1, n-j+1):
                assert T[n-m-1, j-1] is not None, f"T[{n-m-1},{j-1}] accessed but never set"
                assert W[n-m-1, j-1] is not None, f"W[{n-m-1},{j-1}] accessed but never set"
            T[n-1, j] = np.sum(
                [commutator(Lambda[m-1], T[n-m-1, j-1]) for m in range(1, n-j+1)],
                axis=0
            )
            
            W[n-1, j] = np.sum(
                [commutator(Lambda[m-1], W[n-m-1, j-1]) for m in range(1, n-j+1)],
                axis=0
            )
        G[n-1] = sum([B[k] * ((-1j)**k) * (W[n-1, k] + ((-1)**(k+1)) * T[n-1, k]) for k in range(1, n)])
        F[n-1] = np.mean(G[n-1], axis=0)
        T[n-1, 0] = F[n-1]
        Lambda[n-1] = cumulative_trapezoid(
            G[n-1] - F[n-1][np.newaxis, :, :], t_grid, axis=0, initial=0
        )

    return Lambda, F

def exact_evolution(H_func, t_grid, eval_index):
    U = np.eye(H_func(0).shape[0], dtype=complex)
    for i in range(1, eval_index+1):
        dt = t_grid[i] - t_grid[i-1]
        U = expm(-1j * H_func(t_grid[i]) * dt) @ U
    return U

def FM_evolution(Lambda, order, F, t_grid, eval_index):
    Lambda_sum = sum(Lambda[n] for n in range(order))
    F_sum = sum(F[n] for n in range(order))
    kick_op = expm(-1j * Lambda_sum[eval_index])
    floquet_op = expm(-1j * F_sum * t_grid[eval_index])
    U_FM = kick_op @ floquet_op
    return U_FM

def evaluate_accuracy(U1, U2):
    fro_norm = np.linalg.norm(U1 - U2, 'fro')
    fidelity = abs(np.trace(U1.conj().T @ U2))**2 / U1.shape[0]**2
    return fro_norm, fidelity

def plot(Lambda, order):
    fig, axes = plt.subplots(1, figsize=(8, 4))

    peaks = [np.max(np.abs(Lambda[n])) for n in range(order)]
    semilog = axes.semilogy(range(order), peaks, 'o-')
    axes.set_xlabel("Order")
    axes.set_ylabel("Peak")
    axes.set_title("Convergence of Lambda corrections")

    plt.tight_layout()
    plt.show()
    return fig, axes, semilog

def commutator(A, B):
    return A @ B - B @ A

if __name__ == "__main__":
    #Parameters
    period = 2 * np.pi/5
    order = 10
    t_0 = 0
    N_t = 1000

    #Approximation
    t_grid = np.linspace(0, period, N_t)
    Lambda, F = floquet_magnus_expansion(H_func, t_grid, order)
    U_exact = exact_evolution(H_func, t_grid, 400)
    U_FM = FM_evolution(Lambda, order, F, t_grid, 400)

    #Analysis
    plot(Lambda, order)
    fro_norm, fidelity = evaluate_accuracy(U_FM, U_exact)
    print(f"Frobenius Norm: {fro_norm}")
    print(f"Fidelity: {fidelity}") 
    