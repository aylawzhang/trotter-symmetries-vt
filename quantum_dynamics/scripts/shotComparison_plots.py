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

    models = [(heisenberg_xxz_hamiltonian(N, 1.0, 1.0), "Heisenberg XXZ Model"),
              (ising_and_transverse_hamiltonian(N, 1.0, 1.0), "Transverse Field Ising Model")]
    obs = magnetization_operator(N)
   
    T = 0.5
    # N_shots=[1024,4096]
    trials=5
    points=10
    trotter_steps=100
    
    psi_0 = Statevector(np.random.rand(2**N) + 1j*np.random.rand(2**N))
    psi_0 /= np.linalg.norm(psi_0)

    indices = np.linspace(0, trotter_steps, points, dtype=int)
    times = np.linspace(0, T, trotter_steps + 1)[indices]
    fig, axes = plt.subplots(1,2, figsize=(14,5))

    for ax, (H_op, title) in zip(axes, models):
        paulis = H_op.to_list()

        #Exact evolution
        H_mat = H_op.to_matrix()
        dt = T / trotter_steps
        psi_exact = psi_0.copy()
        exact_obs = []
        U = expm(-1j * H_mat * dt)
        for _ in range(trotter_steps+1):
            exact_obs.append(np.real(np.vdot(psi_exact.data, (obs @ psi_exact).data)))
            psi_exact = Statevector(U @ psi_exact.data)
        exact_plot = np.array(exact_obs)[indices]

        #Trotter
        low_shots = []
        high_shots = []

        for _ in range(trials):
            low_shots.append(trotter_evolution(paulis, psi_0, T, dt, obs, N,1024))
            high_shots.append(trotter_evolution(paulis, psi_0, T, dt, obs, N,4096))

        low_shots = np.array(low_shots)
        high_shots = np.array(high_shots)

        low_mean = np.mean(low_shots, axis=0)[indices]
        low_std  = np.std(low_shots, axis=0)[indices]
        high_mean = np.mean(high_shots, axis=0)[indices]
        high_std  = np.std(high_shots, axis=0)[indices]

        ax.plot(times, exact_plot, "--", color="black", linewidth=2, label="Exact")
        ax.errorbar(times, low_mean, yerr=low_std, label="1024 shots", fmt='-o', capsize=3)
        ax.errorbar(times, high_mean, yerr=high_std, label="4096 shots", fmt='-s', capsize=3)
        
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Magnetization")
        ax.legend()

    plt.suptitle(f"Shot Noise Effects on Trotterization", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulation()
