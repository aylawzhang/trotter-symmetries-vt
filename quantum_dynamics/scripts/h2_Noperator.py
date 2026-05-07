import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qiskit.quantum_info import Statevector

from quantum_evolution.hamiltonians import h2_hamiltonian
from quantum_evolution.observables import number_operator, IIZI_observable
from quantum_evolution.trotter import trotter_evolution
from quantum_evolution.qdrift import qdrift_evolution

def simulation(initial_state):
    H_op,N = h2_hamiltonian(1.0)
    N_op=number_operator(N)
    # N_op=IIZI_observable(N)
    paulis = H_op.to_list()
    T = 0.5

    shots=2048
    trials=5
    points=10
    trotter_steps=100
    
    psi_0 = Statevector.from_label(initial_state)

    indices = np.linspace(0, trotter_steps, points, dtype=int)
    times = np.linspace(0, T, trotter_steps + 1)[indices]

    all_trotter = []
    all_qdrift = []

    plt.figure()

    #exact evolution
    H_mat = H_op.to_matrix()
    dt = T / trotter_steps
    psi_exact = psi_0.copy()
    exact_mags = []
    U = expm(-1j * H_mat * dt)
    for _ in range(trotter_steps+1):
        exact_mags.append(np.real(np.vdot(psi_exact.data, (N_op @ psi_exact).data)))
        psi_exact = Statevector(U @ psi_exact.data)
    exact_mags_plot = np.array(exact_mags)[indices]
    plt.plot(times, exact_mags_plot, "--", label=f"Exact", linewidth=2, color='black')

    #trotter
    for _ in range(trials):
        dt = T / trotter_steps
        
        #trotter
        mags_t = trotter_evolution(paulis, psi_0, T, dt, N_op, N,shots)
        all_trotter.append(mags_t)
    
        #qdrift    
        mags_q = qdrift_evolution(paulis, psi_0, T, dt, N_op, N,shots)
        all_qdrift.append(mags_q)

    all_trotter = np.array(all_trotter)
    all_qdrift = np.array(all_qdrift)
    trot_mean = np.mean(all_trotter, axis=0)[indices]
    trot_std  = np.std(all_trotter, axis=0)[indices]
    q_mean = np.mean(all_qdrift, axis=0)[indices]
    q_std  = np.std(all_qdrift, axis=0)[indices]

    plt.errorbar(times, trot_mean, yerr=trot_std, label="Trotter (Avg + Std)", fmt='-o', capsize=3)
    plt.errorbar(times, q_mean, yerr=q_std, label="qDRIFT (Avg + Std)", fmt='-s', capsize=3)

    plt.xlabel("Time")
    plt.ylabel("Number Operator")
    plt.title("H2 Molecule Time Evolution (|"+initial_state+">)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    simulation("1100")
    simulation("1010")