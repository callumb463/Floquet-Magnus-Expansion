from dataclasses import dataclass, field
from typing import Callable
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

@dataclass
class FloquetSystem:
    H_func: Callable          # H_func(t) -> matrix — omega/A baked in via closure
    omega: float
    order: int
    N_t: int = 5000
    Lambda: object = None
    F: object = None
    U_exact: object = None
    U_FM: object = None
    
    @property
    def period(self):
        return 2 * np.pi / self.omega
    
    @property
    def t_grid(self):
        return np.linspace(0, self.period, self.N_t)
    
    def run(self, compute_propagators=True, aht=False):
        self.Lambda, self.F = self.floquet_magnus_expansion(aht)
        if compute_propagators:
            self.U_exact = self.exact_evolution()
            self.U_FM = self.FM_evolution()

    def floquet_magnus_expansion(self, aht):
        period = self.period
        t_grid = self.t_grid
        H_func = self.H_func
        order = self.order

        dim = H_func(0).shape[0]

        B = [float(bernoulli(n)/factorial(n)) for n in range(order)]
        H = np.array([H_func(t) for t in t_grid])
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
            W[i, 0] = np.zeros((len(t_grid), dim, dim), dtype=complex)
        if aht:
            Lambda[0] = np.zeros((len(t_grid), dim, dim), dtype=complex)
        else:
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

    def exact_evolution(self):
        t_grid = self.t_grid
        H_func = self.H_func
        U = np.eye(H_func(0).shape[0], dtype=complex)
        Us = [U]
        for i in range(1, len(self.t_grid)):
            dt = t_grid[i] - t_grid[i-1]
            U = expm(-1j * H_func(t_grid[i]) * dt) @ U
            Us.append(U)
        return Us

    def FM_evolution(self):
        Lambda_sum = sum(self.Lambda[n] for n in range(self.order))
        F_sum = sum(self.F[n] for n in range(self.order))
        Us = []
        for idx in range(len(self.t_grid)):
            kick_op = expm(-1j * Lambda_sum[idx])
            floquet_op = expm(-1j * F_sum * self.t_grid[idx])
            Us.append(kick_op @ floquet_op)
        return Us



def make_spin_drive(omega, A, s):
    Sx, Sy, Sz = spin_matrices(s)
    def H(t):
        return Sz + A * np.cos(omega * t) * Sx
    return H

def evaluate_accuracy(U1, U2):
    fro_norm = np.linalg.norm(U1 - U2, 'fro')
    fidelity = abs(np.trace(U1.conj().T @ U2))**2 / U1.shape[0]**2
    return fro_norm, fidelity

def commutator(A, B):
    return A @ B - B @ A

def spin_matrices(s):
    d = int(2*s + 1)
    m = np.arange(s, -s-1, -1)  # [s, s-1, ..., -s]
    
    Sz = np.diag(m).astype(complex)
    
    # S+ and S- raising/lowering operators
    coeffs = np.sqrt(s*(s+1) - m[:-1]*(m[:-1]-1))
    Sp = np.diag(coeffs, k=1).astype(complex)
    Sm = np.diag(coeffs, k=-1).astype(complex)
    
    Sx = (Sp + Sm) / 2
    Sy = (Sp - Sm) / (2j)
    
    return Sx, Sy, Sz

def accuracy_plot(ax, sys):
    eval_indices = list(range(0, len(sys.t_grid), 50))
    if eval_indices[-1] != len(sys.t_grid) - 1:
        eval_indices.append(len(sys.t_grid) - 1)
    orders_to_test = [1, 2, 5, 10]
    for o in orders_to_test:
        # need a subsys with order=o to get correctly truncated U_FM
        subsys = FloquetSystem(H_func=sys.H_func, omega=sys.omega, order=o, N_t=sys.N_t)
        subsys.run()
        fros = [evaluate_accuracy(subsys.U_FM[idx], sys.U_exact[idx])[0] 
                for idx in eval_indices]
        ax.semilogy(sys.t_grid[list(eval_indices)], fros, label=f"order={o}")
    ax.set_xlabel("t")
    ax.set_ylabel("Frobenius norm")
    ax.set_title("Norm between Numerical U and FME U")
    ax.legend()

def convergence_plot(ax, sys, A, s):
    freq_to_test = [0.1, 1, 5, 10, 20]
    for freq in freq_to_test:
        subsys = FloquetSystem(
            H_func=make_spin_drive(omega=freq, A=A, s=s),
            omega=freq,
            order=sys.order
        )
        subsys.run(compute_propagators=False)
        F_norms = [np.linalg.norm(subsys.F[n], 'fro') for n in range(sys.order)]
        ax.semilogy(list(range(sys.order)), F_norms, label=rf'$A/\omega$={A/freq:.2f}')
    ax.set_xlabel("order")
    ax.set_ylabel("Norm of F")
    ax.set_title(r"Norm of $F_n$ for Varying Frequencies")
    ax.legend()

def micromotion_dev_plot(ax, sys):
    Lambda_sum = sum(sys.Lambda[n] for n in range(sys.order))
    P = expm(-1j * Lambda_sum)
    deviation = np.linalg.norm(P - np.eye(P.shape[1])[np.newaxis, :, :], axis=(-2, -1))
    ax.plot(sys.t_grid, deviation)
    ax.set_xlabel("t")
    ax.set_ylabel(r"Norm of $P(t)-I$")
    ax.set_title("Deviation of the Kick Operator From the Identity")

def quasienergy_plot(ax, sys, A_max=5, n_amps=10, s=0.5):
    amp_to_test = np.linspace(0.1, A_max, n_amps)
    quasienergy_array = []
    for amp in amp_to_test:
        subsys = FloquetSystem(
            H_func=make_spin_drive(omega=sys.omega, A=amp, s=s),
            omega=sys.omega,
            order=sys.order
        )
        subsys.run(compute_propagators=False)
        F_sum = sum(subsys.F[n] for n in range(sys.order))
        quasienergies = np.linalg.eigvalsh(F_sum)
        quasienergy_array.append(quasienergies)
    quasienergy_array = np.array(quasienergy_array)
    for n in range(quasienergy_array.shape[1]):
        ax.plot(amp_to_test / sys.omega, quasienergy_array[:, n], 'o-', markersize=3)
    ax.set_xlabel("A/w")
    ax.set_ylabel("Quasienergy")
    ax.set_title(r"Floquet Quasienergy Spectrum vs $A/\omega$")



if __name__ == "__main__":
    #Parameters
    sys = FloquetSystem(H_func=make_spin_drive(omega=5, A=0.5, s=1), omega=5, order=10)
    sys.run()


    #Plot
    fig, axs = plt.subplots(2,2, figsize=(8,6))
    accuracy_plot(axs[0,0], sys)
    convergence_plot(axs[0,1], sys, 0.5, 1)
    micromotion_dev_plot(axs[1,0], sys)
    quasienergy_plot(axs[1,1], sys, s=1)
    fig.tight_layout()
    plt.savefig('plots/Floquet-Magnus_Plots')