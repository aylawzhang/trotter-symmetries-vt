import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qiskit.quantum_info import Statevector

from quantum_evolution.hamiltonians import h2_hamiltonian
from quantum_evolution.observables import number_operator, IIZI_observable
from quantum_evolution.trotter import random_trotter, symmetry_trotter
from quantum_evolution.qdrift import random_qdrift, symmetry_qdrift

def simulation(initial_state):
    H_op,N = h2_hamiltonian(1.0)
    # N_op=number_operator(N)
    N_op=IIZI_observable(N)
    paulis = H_op.to_list()
    T = 0.5

    shots=2048
    trials=5
    points=10
    trotter_steps=100
    
    psi_0 = Statevector.from_label(initial_state)

    indices = np.linspace(0, trotter_steps, points, dtype=int)
    times = np.linspace(0, T, trotter_steps + 1)[indices]

    trotter_ctrl = []
    trotter_symm = []
    qdrift_ctrl = []
    qdrift_symm = []

    plt.figure()

    #exact evolution
    H_mat = H_op.to_matrix()
    dt = T / trotter_steps
    psi_exact = psi_0.copy()
    exact_obs = []
    U = expm(-1j * H_mat * dt)
    for _ in range(trotter_steps+1):
        exact_obs.append(np.real(np.vdot(psi_exact.data, (N_op @ psi_exact).data)))
        psi_exact = Statevector(U @ psi_exact.data)
    exact_mags_plot = np.array(exact_obs)[indices]
    plt.plot(times, exact_mags_plot, "--", label=f"Exact", linewidth=2, color='black')

    #trotter
    for _ in range(trials):
        dt = T / trotter_steps
        
        #trotter control
        t_rand = random_trotter(paulis, psi_0, T, dt, N_op, N,shots)
        trotter_ctrl.append(t_rand)

        #trotter symmetry
        t_symm = symmetry_trotter(paulis, psi_0, T, dt, N_op, N,shots)
        trotter_symm.append(t_symm)
    
        #qdrift control
        q_ctrl = random_qdrift(paulis, psi_0, T, dt, N_op, N,shots)
        qdrift_ctrl.append(q_ctrl)
        
        #qdrift symmetry
        q_symm = symmetry_qdrift(paulis, psi_0, T, dt, N_op, N,shots)
        qdrift_symm.append(q_symm)

    trotter_ctrl = np.array(trotter_ctrl)
    trotter_symm = np.array(trotter_symm)
    qdrift_ctrl = np.array(qdrift_ctrl)
    qdrift_symm = np.array(qdrift_symm)

    tc_mean = np.mean(trotter_ctrl, axis=0)[indices]
    tc_std  = np.std(trotter_ctrl, axis=0)[indices]
    qc_mean = np.mean(qdrift_ctrl, axis=0)[indices]
    qc_std  = np.std(qdrift_ctrl, axis=0)[indices]
    ts_mean = np.mean(trotter_symm, axis=0)[indices]
    ts_std  = np.std(trotter_symm, axis=0)[indices]
    qs_mean = np.mean(qdrift_symm, axis=0)[indices]
    qs_std  = np.std(qdrift_symm, axis=0)[indices]

    plt.errorbar(times, tc_mean, yerr=tc_std, label="Control Trotter", fmt='-o', capsize=3)
    plt.errorbar(times, qc_mean, yerr=qc_std, label="Control qDRIFT", fmt='-s', capsize=3)
    plt.errorbar(times, ts_mean, yerr=ts_std, label="Symmetry-Aware Trotter", fmt='-o', capsize=3)
    plt.errorbar(times, qs_mean, yerr=qs_std, label="Symmetry-Aware qDRIFT", fmt='-o', capsize=3)

    plt.xlabel("Time")
    plt.ylabel("IIZI Operator")
    plt.title("H2 Molecule Time Evolution (|"+initial_state+">)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # simulation("1100")
    simulation("1010")