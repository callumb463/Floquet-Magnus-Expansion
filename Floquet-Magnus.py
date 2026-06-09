import numpy as np
from sympy import bernoulli, factorial
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import customtkinter as ctk
from scipy.integrate import cumulative_trapezoid


def main():
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

def H_func(t, omega, A):
    Sx = (1/np.sqrt(2)) * np.array([[0, 1, 0],
                                    [1, 0, 1],
                                    [0, 1, 0]], dtype=complex)
    Sz = np.array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0,-1]], dtype=complex)
    # Static Sz + oscillating Sx — these don't commute
    return Sz + A * np.sin(omega*t) * Sx


def floquet_magnus_expansion(H_func, t_grid, omega, A, order):
    dim = H_func(0, omega, A).shape[0]
    B = [float(bernoulli(n)/factorial(n)) for n in range(order)]
    H = np.zeros((len(t_grid), dim, dim), dtype=complex)
    for i, t in enumerate(t_grid):
        H[i] = H_func(t, omega, A)
    F = np.empty(order, dtype=object)
    G = np.empty(order, dtype=object)
    W = np.empty((order, order), dtype=object)
    T = np.empty((order, order), dtype=object)
    Lambda = np.empty(order, dtype=object)
    

    # Initialize F[0], G[0], T[0, 0], W[n, 0], and Lambda[0]
    F[0] = F[0] = np.trapezoid(H, t_grid, axis=0) / period
    G[0] = H.copy()
    T[0, 0] = F[0]
    W[0, 0] = H.copy()
    for i in range(1,order):
        W[i, 0] = np.zeros((N_t, dim, dim), dtype=complex)
    Lambda[0] = cumulative_trapezoid(H - F[0][np.newaxis, :, :], t_grid, axis=0, initial=0)

    
    #Iterate to compute higher order terms
    for n in range(2, order + 1):
        for j in range(1, n):
            for m in range(1, n-j+1):
                assert T[n-m-1, j-1] is not None, f"T[{n-m-1},{j-1}] accessed but never set"
                assert W[n-m-1, j-1] is not None, f"W[{n-m-1},{j-1}] accessed but never set"
            T[n-1, j] = np.sum(
                [commutator(T[n-m-1, j-1], Lambda[m-1]) for m in range(1, n-j+1)],
                axis=0
            )
            
            W[n-1, j] = np.sum(
                [commutator(W[n-m-1, j-1], Lambda[m-1]) for m in range(1, n-j+1)],
                axis=0
            )
        G[n-1] = sum([B[k] * ((-1j)**k) * (W[n-1, k] + ((-1)**(k+1)) * T[n-1, k]) for k in range(1, n)])
        F[n-1] = np.trapezoid(G[n-1], t_grid, axis=0) / period
        T[n-1, 0] = F[n-1]
        Lambda[n-1] = cumulative_trapezoid(
            G[n-1] - F[n-1][np.newaxis, :, :], t_grid, axis=0, initial=0
        )

    return Lambda, F

def exact_evolution(H_func, t_grid, omega, A, eval_index):
    U = np.eye(H_func(0, omega, A).shape[0], dtype=complex)
    for i in range(1, eval_index+1):
        dt = t_grid[i] - t_grid[i-1]
        U = expm(-1j * H_func(t_grid[i], omega, A) * dt) @ U
    return U

def FM_evolution(Lambda, F, order, t_grid, eval_index):
    Lambda_sum = sum(Lambda[n] for n in range(order))
    F_sum = sum(F[n] for n in range(order))

    #print(f"order={order}, Lambda_sum max={np.max(np.abs(Lambda_sum[eval_index])):.6f}, F_sum max={np.max(np.abs(F_sum)):.6f}")

    kick_op = expm(-1j * Lambda_sum[eval_index])
    floquet_op = expm(-1j * F_sum * t_grid[eval_index])
    U_FM = kick_op @ floquet_op
    return U_FM

def evaluate_accuracy(U1, U2):
    fro_norm = np.linalg.norm(U1 - U2, 'fro')
    fidelity = abs(np.trace(U1.conj().T @ U2))**2 / U1.shape[0]**2
    return fro_norm, fidelity

def accuracy_plot(ax, Lambda, F, order, t_grid, omega, A):

    #Frobenius vs Numerical U
    eval_indices = range(0, len(t_grid), 50)
    orders_to_test = [1, 2, 5, 10]
    for o in orders_to_test:
        fros = []
        for idx in eval_indices:
            U_exact = exact_evolution(H_func, t_grid, omega, A, idx)
            U_FM = FM_evolution(Lambda, F, o, t_grid, idx)
            fro, _ = evaluate_accuracy(U_FM, U_exact)
            fros.append(fro)
        ax.plot(t_grid[list(eval_indices)], fros, label=f"order={o}")
    ax.set_xlabel("t")
    ax.set_ylabel("Frobenius norm")
    ax.set_title("Norm between Numerical U and FME U")
    ax.legend()
    return

def convergence_plot(ax, order, t_grid, A):
    freq_to_test = [0.1, 1, 5, 10, 20]
    for freq in freq_to_test:
        F_norm = []
        _, F = floquet_magnus_expansion(H_func, t_grid, freq, A, order)
        F_norm.extend([np.linalg.norm(F[n], 'fro') for n in range(order)])
        ax.semilogy(list(range(order)), F_norm, label=f'A/w={A/omega}')
    ax.set_xlabel("order")
    ax.set_ylabel("Norm of F")
    ax.set_title("Convergence of F for Varying Frequencies")
    ax.legend()
    return

def micromotion_dev_plot(ax, Lambda, t_grid):
    Lambda_sum = sum(Lambda[n] for n in range(order))
    P = expm(-1j * Lambda_sum)
    deviation = np.linalg.norm(P-np.eye(P.shape[1])[np.newaxis, :, :], axis=(-2,-1))
    ax.plot(t_grid, deviation)
    ax.set_xlabel("t")
    ax.set_ylabel("Norm of P(t)-I")
    ax.set_title("Deviation of Motion From the Effective Hamiltonian")


def quasienergy_plot(ax, t_grid, omega):
    amp_to_test = np.linspace(0.1, 5, 10)

    quasienergy_array = []
    for amp in amp_to_test:
        _, F = floquet_magnus_expansion(H_func, t_grid, omega, amp, order)
        quasienergies = np.linalg.eigvalsh(sum(F[n] for n in range(len(F))))
        quasienergy_array.append(quasienergies)

    quasienergy_array = np.array(quasienergy_array)
    for n in range(len(quasienergy_array[0])):
        ax.plot(amp_to_test/omega, quasienergy_array[:, n], 'o-', markersize=3)
    ax.set_xlabel("A/w")
    ax.set_ylabel("Quasienergy")
    ax.set_title("Floquet Quasienergy Spectrum vs A/w")


def commutator(A, B):
    return A @ B - B @ A

if __name__ == "__main__":
    #Parameters
    omega = 5
    A = 0.5
    period = 2 * np.pi/omega
    order = 20
    t_0 = 0
    N_t = 5000
    eval_index = 500


    #Approximation
    t_grid = np.linspace(0, period, N_t)
    Lambda, F = floquet_magnus_expansion(H_func, t_grid, omega, A, order)
    U_exact = exact_evolution(H_func, t_grid, omega, A, eval_index)

    fig, axs = plt.subplots(2,2, figsize=(8,6))
    accuracy_plot(axs[0,0], Lambda, F, order, t_grid, omega, A)
    convergence_plot(axs[0,1], order, t_grid, A)
    micromotion_dev_plot(axs[1,0], Lambda, t_grid)
    quasienergy_plot(axs[1,1], t_grid, omega)
    
    fig.tight_layout()
    plt.savefig('Floquet-Magnus_Plots')
    plt.show()