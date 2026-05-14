import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qiskit.quantum_info import Statevector

from quantum_evolution.hamiltonians import heisenberg_xxz_hamiltonian, ising_and_transverse_hamiltonian
from quantum_evolution.observables import magnetization_operator
from quantum_evolution.trotter import trotter_evolution

def simulation():
    N=6

    models = [(heisenberg_xxz_hamiltonian(N, 1.0, 0.5), "Heisenberg XXZ Model"),
              (ising_and_transverse_hamiltonian(N, 1.0, 0.5), "Transverse Field Ising Model")]
    
    obs = magnetization_operator(N)
   
    T = 5.0
    trials=5
    step_list=[20,200]
    exact_steps = 300
    
    psi_0 = Statevector(np.random.rand(2**N) + 1j*np.random.rand(2**N))
    psi_0 /= np.linalg.norm(psi_0)

    plt.rcParams.update({
        "figure.titlesize": 20,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13
    })
    fig, axes = plt.subplots(1,2, figsize=(14,5))

    for ax, (H_op, title) in zip(axes, models):
        paulis = H_op.to_list()

        #Exact evolution
        H_mat = H_op.to_matrix()
        dt = T / exact_steps
        psi_exact = psi_0.copy()
        exact_obs = []
        U = expm(-1j * H_mat * dt)
        for _ in range(exact_steps+1):
            exact_obs.append(np.real(np.vdot(psi_exact.data, (obs @ psi_exact).data)))
            psi_exact = Statevector(U @ psi_exact.data)
        
        exact_times = np.linspace(0, T, exact_steps + 1)
        ax.plot(exact_times, exact_obs, "--", color="black", label="Exact")
        
        #Trotter

        colors = ["red", "blue"]

        for steps, color in zip(step_list,colors):
            dt = T / steps
            indices = np.linspace(0, steps, 10, dtype=int)
            times = np.linspace(0, T, steps + 1)[indices]
            results = []

            for _ in range(trials):
                results.append(trotter_evolution(paulis, psi_0, T, dt, obs, N, 4096))
            results = np.array(results)
            mean = np.mean(results, axis=0)[indices]
            std = np.std(results, axis=0)[indices]

            ax.errorbar(times, mean, yerr=std, fmt='-o', capsize=3, label=f"{steps} steps", color=color)

        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Magnetization")
        ax.legend()
    
    plt.suptitle("Effect of Trotter Step Count (N=6)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulation()
