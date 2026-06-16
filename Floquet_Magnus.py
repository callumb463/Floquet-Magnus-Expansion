from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from sympy import bernoulli, factorial
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import customtkinter as ctk
import math
from scipy.integrate import cumulative_trapezoid
from time import process_time, time

@dataclass
class FloquetSystem:
    H_func: Callable          # H_func(t) -> matrix — omega/A baked in via closure
    omega: float
    order: int
    periods: int
    N_t: int = 500
    aht: bool = True
    Lambda: object = None
    F: object = None
    U_exact: object = None
    U_FM: object = None
    kick: object = None
    Floquet: object = None
    
    @property
    def period(self):
        return 2 * np.pi / self.omega
    
    @property
    def t_grid(self):
        return np.linspace(0, self.period*self.periods, self.N_t*self.periods)
    
    def run(self, compute_propagators=True):
        self.Lambda, self.F = self.floquet_magnus_expansion(self.aht)
        if compute_propagators:
            self.U_exact = self.exact_evolution()
            self.U_FM, self.kick, self.Floquet = self.FM_evolution()
            

    def floquet_magnus_expansion(self, aht):
        omega = self.omega
        period = self.period
        N_t = self.N_t
        t_grid_exp = np.linspace(0, period, N_t)
        H_func = self.H_func
        order = self.order

        dim = H_func(0).shape[0]

        B = [float(bernoulli(n)/factorial(n)) for n in range(order)]
        H = np.array([H_func(t) for t in t_grid_exp])
        F = np.empty(order, dtype=object)
        G = np.empty(order, dtype=object)
        W = np.empty((order, order), dtype=object)
        T = np.empty((order, order), dtype=object)
        Lambda = np.empty(order, dtype=object)
        
        # Initialize F[0], G[0], T[0, 0], W[n, 0], and Lambda[0]
        F[0] = F[0] = np.trapezoid(H, t_grid_exp, axis=0) / period
        G[0] = H.copy()
        T[0, 0] = F[0]
        W[0, 0] = H.copy()
        for i in range(1,order):
            W[i, 0] = np.zeros((len(t_grid_exp), dim, dim), dtype=complex)
        if aht:
            Lambda[0] = cumulative_trapezoid(H - F[0][np.newaxis, :, :], t_grid_exp, axis=0, initial=0)
        else:
            H_fft = np.fft.fft(H, axis=0) / N_t
            ms = np.concatenate([np.arange(1, N_t//2 + 1), np.arange(-N_t//2 + 1, 0)])  # m = 1..N/2, -(N/2-1)..-1
            Lambda_init = sum(H_fft[m] / (1j * m * omega) for m in ms)
            Lambda[0] = cumulative_trapezoid(H - F[0][np.newaxis, :, :], t_grid_exp, axis=0, initial=0) + Lambda_init[np.newaxis, :, :]
            
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
            F[n-1] = np.trapezoid(G[n-1], t_grid_exp, axis=0) / period
            T[n-1, 0] = F[n-1]
            Lambda[n-1] = cumulative_trapezoid(
                G[n-1] - F[n-1][np.newaxis, :, :], t_grid_exp, axis=0, initial=0
            )

        return Lambda, F

    def exact_evolution(self):
        t_grid = self.t_grid
        H_func = self.H_func
        U = np.eye(H_func(0).shape[0], dtype=complex)
        Us = [U]
        for i in range(1, len(self.t_grid)):
            dt = t_grid[i] - t_grid[i-1]
            t_mid = (t_grid[i] + t_grid[i-1]) / 2  # midpoint
            U = expm(-1j * H_func(t_mid) * dt) @ U
            Us.append(U)
        return np.array(Us)

    def FM_evolution(self):
        Lambda_sum = sum(self.Lambda[n] for n in range(self.order))
        F_sum = sum(self.F[n] for n in range(self.order))
        Lambda_tiled = np.tile(Lambda_sum, (self.periods, 1, 1))

        kick_op = np.array([expm(-1j * Lambda_tiled[idx]) for idx in range(len(self.t_grid))])
        floquet_op = np.array([expm(-1j * F_sum * t) for t in self.t_grid])
        
        U = kick_op @ floquet_op 
        return U, kick_op, floquet_op



def bloch_siegert_int(omega, A, s):
    Sx, Sy, Sz = spin_matrices(s)
    Sp = Sx + 1j*Sy
    Sm = Sx - 1j*Sy
    def H(t):
        return (A/2) * (Sp*(1 + np.exp(2j*omega*t)) + Sm*(1 + np.exp(-2j*omega*t)))
    return H

def bloch_siegert_lab(omega, A, s):
    Sx, Sy, Sz = spin_matrices(s)
    def H(t):
        return Sz + A*np.cos(omega*t)*Sx
    return H

def evaluate_accuracy(U1, U2):
    fro_norm = np.linalg.norm(U1 - U2, 'fro')
    fidelity = abs(np.trace(U1.conj().T @ U2))**2 / U1.shape[0]**2
    return fro_norm, fidelity

def commutator(A, B):
    return A @ B - B @ A

def expectation(psi, op):
        return np.real(np.einsum('ti,ij,tj->t', psi.conj(), op, psi))

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

def bloch_vector_error(U_approx, U_exact, psi_0, s):
    Sx, Sy, Sz = spin_matrices(s)
    psi_approx = U_approx @ psi_0
    psi_exact  = U_exact  @ psi_0
    
    dx = expectation(psi_approx, Sx) - expectation(psi_exact, Sx)
    dy = expectation(psi_approx, Sy) - expectation(psi_exact, Sy)
    dz = expectation(psi_approx, Sz) - expectation(psi_exact, Sz)
    
    return np.sqrt(dx**2 + dy**2 + dz**2)

def accuracy_plot(ax, sys):
    orders_to_test = [1, 2, 5, 10]
    for o in orders_to_test:
        subsys = FloquetSystem(
            H_func=bloch_siegert_int(omega=sys.omega, A=1, s=1/2),
            omega=sys.omega, 
            order=o, 
            N_t=sys.N_t, 
            periods=1)
        subsys.run()
        
        eval_indices = list(range(0, len(subsys.t_grid), 50))
        if eval_indices[-1] != len(subsys.t_grid) - 1:
            eval_indices.append(len(subsys.t_grid) - 1)
        
        fros = [evaluate_accuracy(subsys.U_FM[idx], subsys.U_exact[idx])[0] 
                for idx in eval_indices]
        t_in_periods = subsys.t_grid[list(eval_indices)] / subsys.period
        ax.semilogy(t_in_periods, fros, label=f"order={o}")
    ax.set_xlabel(r"$t\,/\,T$")
    ax.set_ylabel("Frobenius norm")
    ax.set_title("Norm between Numerical U and FME U")
    ax.legend()

def convergence_plot(ax, sys, A, s):
    freq_to_test = [0.01, 0.1, 1, 10, 100]
    for freq in freq_to_test:
        subsys = FloquetSystem(
            H_func=bloch_siegert_int(omega=freq, A=A, s=s),
            omega=freq,
            order=sys.order,
            periods=1
        )
        subsys.run(compute_propagators=False)
        F_norms = [np.linalg.norm(subsys.F[n], 'fro') for n in range(sys.order)]
        ax.semilogy(list(range(sys.order)), F_norms, label=rf'$A/\omega$={A/freq:.2f}')
    ax.set_xlabel("order $n$")
    ax.set_ylabel(r"$\Vert F_n \Vert$")
    ax.set_title(r"Norm of $F_n$ for Varying $A/\omega$")
    ax.legend()

def micromotion_dev_plot(ax, sys):
    Lambda_sum = sum(sys.Lambda[n] for n in range(sys.order))
    t_grid_one_period = np.linspace(0, sys.period, sys.N_t)
    t_in_periods = t_grid_one_period / sys.period
    P = np.array([expm(-1j * Lambda_sum[idx]) for idx in range(sys.N_t)])
    deviation = np.linalg.norm(P - np.eye(P.shape[1])[np.newaxis, :, :], axis=(-2, -1))
    ax.plot(t_in_periods, deviation)
    ax.set_xlabel(r"$t\,/\,T$")
    ax.set_ylabel(r"$\Vert P(t) - I \Vert$")
    ax.set_title("Deviation of P(t) From the Identity")

def quasienergy_plot(ax, sys, A_max=5, n_amps=10, s=0.5):
    amp_to_test = np.linspace(0.1, A_max, n_amps)
    quasienergy_array = []
    for amp in amp_to_test:
        subsys = FloquetSystem(
            H_func=bloch_siegert_int(omega=sys.omega, A=amp, s=s),
            omega=sys.omega,
            order=sys.order,
            periods=1
        )
        subsys.run(compute_propagators=False)
        F_sum = sum(subsys.F[n] for n in range(sys.order))
        quasienergies = np.linalg.eigvalsh(F_sum) / sys.omega   # rescale to units of omega
        quasienergy_array.append(quasienergies)
    quasienergy_array = np.array(quasienergy_array)
    for n in range(quasienergy_array.shape[1]):
        ax.plot(amp_to_test / sys.omega, quasienergy_array[:, n], 'o-', markersize=3)
    ax.set_xlabel(r"$A\,/\,\omega$")
    ax.set_ylabel(r"Quasienergy $/ \,\omega$")
    ax.set_title(r"Floquet Quasienergy Spectrum vs $A/\omega$")

def comparison_plot(aht_sys, floq_sys):
    psi_0 = [1,0]
    Us = [
        aht_sys.Floquet,                                          # AHT: e^{-iF_AHT t}
        aht_sys.kick @ aht_sys.Floquet,                               # AHT + kick: P(t) e^{-iF_AHT t}
        floq_sys.Floquet,                                         # Floquet: e^{-iF_FM t} (stroboscopic only)
        floq_sys.kick @ floq_sys.Floquet @ floq_sys.kick[0].conj().T,    # Full FM: P(t) e^{-iF_FM t} P(0)†
        aht_sys.U_exact
    ]
    labels = [r"AHT: $e^{-iF_{AHT}t}$", r"AHT + kick: $e^{-i\Lambda(t)}e^{-iF_{AHT}t}$", r"Floquet: $e^{-iF_{FM}t}$",r"Full FM: $e^{-i\Lambda(t)}e^{-iF_{FM}t}$","Exact"]
    fig, axs = plt.subplots(2)
    Sx, Sy, Sz = spin_matrices(0.5)
    for i, (U, label) in enumerate(zip(Us, labels)):
        psi_t = U @ psi_0
        exp_Sz = expectation(psi_t, Sz)
        axs[0].plot(aht_sys.t_grid/aht_sys.period, exp_Sz, label=label)
        axs[0].set_title("Method Comparison")
        axs[0].set_xlabel("t/T")
        axs[0].set_ylabel(r"$<S_z(t)>$")
    axs[0].legend()
    error_aht = bloch_vector_error(Us[0], Us[4], psi_0, s=1/2)
    error_aht_kick = bloch_vector_error(Us[1], Us[4], psi_0, s=1/2)
    axs[1].plot(aht_sys.t_grid/aht_sys.period, error_aht, label=r"AHT: $e^{-iF_{AHT}t}$")
    axs[1].plot(aht_sys.t_grid/aht_sys.period, error_aht_kick, label=r"AHT + kick: $e^{-i\Lambda(t)}e^{-iF_{AHT}t}$")
    axs[1].set_title("Bloch Vector Error")
    axs[1].set_xlabel("t/T")
    axs[1].set_ylabel(r"$|\Delta\vec{r}|$")
    axs[1].legend()
    fig.tight_layout()
    fig.savefig('plots/Comparison', dpi=300)

def transition_plot(axs, eps, aht_sys, floq_sys):
    psi_0 = np.array([1., 0.], dtype=complex)
    floquet = floq_sys.Floquet
    floquet_kick = floq_sys.kick
    aht = aht_sys.Floquet
    aht_kick = aht_sys.kick
    #effective Hamiltonian
    psi_ef = floquet @ psi_0
    P_ef = np.abs(psi_ef[:, 1])**2
    #full propagator with kick
    psi_fm = (floquet_kick @ floquet) @ (floquet_kick[0].conj().T @ psi_0)
    P_fm = np.abs(psi_fm[:, 1])**2
    #AHT propagator with kick
    psi_aht_kick = (aht_kick @ aht) @ psi_0
    P_aht_kick = np.abs(psi_aht_kick[:,1])**2
    #Exact
    psi_exact = aht_sys.U_exact @ psi_0
    P_exact = np.abs(psi_exact[:, 1])**2

    axs[0].plot(aht_sys.t_grid / aht_sys.period, P_ef,    label="Ef")
    axs[0].plot(aht_sys.t_grid / aht_sys.period, P_fm,    label="FM")
    axs[0].plot(aht_sys.t_grid / aht_sys.period, P_aht_kick,    label="AHT + Kick")
    axs[0].plot(aht_sys.t_grid / aht_sys.period, P_exact, label="Exact")
    axs[0].set_xlabel("t/T")
    axs[0].set_ylabel("Transition Probability")
    axs[0].set_title(rf"($\varepsilon = {eps}$)")
    axs[0].legend()
    axs[1].plot(aht_sys.t_grid / aht_sys.period, P_ef-P_exact,    label="Ef")
    axs[1].plot(aht_sys.t_grid / aht_sys.period, P_fm-P_exact,    label="FM")
    axs[1].plot(aht_sys.t_grid / aht_sys.period, P_aht_kick-P_exact,    label="AHT + Kick")
    axs[1].set_xlabel("t/T")
    axs[1].set_ylabel("Probability Difference")
    axs[1].set_title(f"Error")
    axs[1].legend()

def H_eff_error(sys):
    psi_0 = np.array([1., 0.], dtype=complex)
    floquet = sys.Floquet
    psi_ef = floquet @ psi_0
    P_ef = np.abs(psi_ef[:, 1])**2
    psi_exact = sys.U_exact @ psi_0
    P_exact = np.abs(psi_exact[:, 1])**2
    error = np.sqrt(np.mean(np.square(P_ef-P_exact)))
    return error

def phase_plot():
    periods = 3
    spin = 0.5
    fig, axs = plt.subplots(1)
    epsilons = np.linspace(0.1, 3, 20)
    omegas = np.linspace(0.1, 8, 20)
    orders = [2,3]
    error_grids = {}
    for order in orders:
        grid = np.zeros((len(omegas), len(epsilons)))
        for i, eps in enumerate(epsilons):
            for j, omega in enumerate(omegas):
                floq_sys = FloquetSystem(
                    H_func=bloch_siegert_int(omega=omega, A=eps, s=spin),
                    omega=omega,
                    order=order,
                    periods=math.ceil(periods/eps),
                    aht=False
                )
                floq_sys.run()
                grid[j, i] = H_eff_error(floq_sys)
        error_grids[order] = grid

    np.savez('error_grids.npz',
         grid_low=error_grids[2],
         grid_high=error_grids[3],
         epsilons=epsilons,
         omegas=omegas)
    

def main():
    #Plots full motion for each operator
    omega = 1
    spin = 0.5
    order = 3
    periods = 3
    epsilon = 1

    phase_plot()


    """
    fig, axs = plt.subplots(2,3, figsize=(16,9))
    epsilons = [0.2, 1, 3]
    for i,eps in enumerate(epsilons):
        aht_sys = FloquetSystem(H_func=bloch_siegert_int(omega=omega, A=eps, s=spin), omega=omega, order=order, periods=math.ceil(periods/eps), aht=True)
        aht_sys.run()
        floq_sys = FloquetSystem(H_func=bloch_siegert_int(omega=omega, A=eps, s=spin), omega=omega, order=order, periods=math.ceil(periods/eps), aht=False)
        floq_sys.run()
        transition_plot(axs[:,i], eps, aht_sys, floq_sys)
    fig.suptitle("Transition Probability Plots")
    fig.tight_layout()
    fig.savefig('plots/Transition_Probability', dpi=200)
    
    eps = 1
    aht_sys = FloquetSystem(H_func=bloch_siegert_int(omega=omega, A=eps, s=spin), omega=omega, order=order, periods=2, aht=True)
    aht_sys.run()
    floq_sys = FloquetSystem(H_func=bloch_siegert_int(omega=omega, A=eps, s=spin), omega=omega, order=order, periods=2, aht=False)
    floq_sys.run()

    comparison_plot(aht_sys, floq_sys)
    #Plots various qualities of the system
    fig, axs = plt.subplots(2,2, figsize=(8,6))
    accuracy_plot(axs[0,0], floq_sys)
    convergence_plot(axs[0,1], floq_sys, 0.5, 1)
    micromotion_dev_plot(axs[1,0], floq_sys)
    quasienergy_plot(axs[1,1], floq_sys, s=0.5)
    fig.tight_layout()
    plt.savefig('plots/Floquet-Magnus_Plots')
    """
    #Times the computation
    

if __name__ == "__main__":
    start_ptime = process_time()
    start_rtime = time()

    main()

    end_ptime = process_time()
    end_rtime = time()
    print(f'Program processed in {(end_ptime-start_ptime)*1000} ms')
    print(f'Took {end_rtime-start_rtime:.3f} seconds')
    