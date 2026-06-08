import numpy as np
from sympy import bernoulli, factorial
import matplotlib.pyplot as plt
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
    return 0.5 * np.sin(t) * np.array([[2, 1], [0.4, 0.5]])

###STILL NEED TO SORT OUT DIMENSIONS AUTOMATICALLY



###PROBABLY REMAKE FROM SCRATCH
def floquet_magnus_expansion(period, order):
    N_t = 1000
    t_grid = np.linspace(0, period, N_t)
    dim = H_func(0).shape[0]
    H = np.zeros((N_t, dim, dim), dtype=complex)
    for i, t in enumerate(t_grid):
        H[i] = H_func(t)

    F = np.empty(order+1, dtype=object)
    G = np.empty(order+1, dtype=object)
    W = np.empty((order+1, order+1), dtype=object)
    T = np.empty((order+1, order+1), dtype=object)
    Lambda = np.empty(order+1, dtype=object)

    # Initialize all slots to zero arrays
    for i in range(order+1):
        for j in range(order+1):
            T[i, j] = np.zeros((dim, dim), dtype=complex)
            W[i, j] = np.zeros((N_t, dim, dim), dtype=complex)

    F[0] = np.mean(H, axis=0)
    G[0] = H.copy()
    T[0, 0] = F[0]
    W[0, 0] = H.copy()

    Lambda[0] = cumulative_trapezoid(H - F[0][np.newaxis, :, :], t_grid, axis=0, initial=0)

    B = [float(bernoulli(n)/factorial(n)) for n in range(order + 1)]

    for n in range(2, order + 1):
        for j in range(1, n):
            T[n-1, j] = np.sum(
                [commutator(Lambda[m-1], T[n-m-1, j-1]) for m in range(1, n-j+1)],
                axis=0
            )
            
            W[n-1, j] = np.sum(
                [commutator(Lambda[m-1], W[n-m-1, j-1]) for m in range(1, n-j+1)],
                axis=0
            )

        G[n-1] = sum(
            B[j] * ((-1)**j) * (W[n-1, j] + ((-1)**j) * T[n-1, j])
            for j in range(0, n-1)
        )

        F[n-1] = np.mean(G[n-1], axis=0)
        T[n-1, 0] = F[n-1]
        Lambda[n-1] = cumulative_trapezoid(
            G[n-1] - F[n-1][np.newaxis, :, :], t_grid, axis=0, initial=0
        )


    return Lambda

def commutator(A, B):
    return A @ B - B @ A

if __name__ == "__main__":
    #__main__()
    Lambda = floquet_magnus_expansion(2*np.pi, 4)
    print("Lambda_1:\n", Lambda[0])
    print("Lambda_2:\n", Lambda[1])
    print("Lambda_3:\n", Lambda[2])