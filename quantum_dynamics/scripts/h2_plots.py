import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qiskit.quantum_info import Statevector

from quantum_evolution.hamiltonians import h2_hamiltonian
from quantum_evolution.observables import number_operator, Z_operator
from quantum_evolution.trotter import trotter_evolution
from quantum_evolution.qdrift import naive_qdrift, symmetry_qdrift

def simulation():
    H_op,N = h2_hamiltonian(1.0)
    paulis = H_op.to_list()

    N_op=number_operator(N)
    Z_op=Z_operator(N)
   
    T = 0.5
    shots=2048
    trials=5
    points=10
    trotter_steps=100
    
    hf_state = "0" * (N//2) + "1" * (N//2)
    psi_0 = Statevector.from_label(hf_state)

    configs = [
        ("Number Operator (1024 shots)", N_op, 1024),
        ("Z Operator (1024 shots)", Z_op, 1024),
        ("Number Operator (4096 shots)", N_op, 4096),
        ("Z Operator (4096 shots)", Z_op, 4096),
    ]

    indices = np.linspace(0, trotter_steps, points, dtype=int)
    times = np.linspace(0, T, trotter_steps + 1)[indices]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (title, obs, shots) in zip(axes, configs):

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
        ax.plot(times, exact_plot, "--", color="black", linewidth=2, label="Exact")

        #Approximation methods
        trotter_ctrl = []
        qdrift_naive = []
        qdrift_symm = []

        for _ in range(trials):
            dt = T / trotter_steps
            
            #trotter control
            t_rand = trotter_evolution(paulis, psi_0, T, dt, obs, N,shots)
            trotter_ctrl.append(t_rand)

            #qdrift control
            q_naive = naive_qdrift(paulis, psi_0, T, dt, obs, N,shots)
            qdrift_naive.append(q_naive)
            
            #qdrift symmetry
            q_symm = symmetry_qdrift(paulis, psi_0, T, dt, obs, N,shots)
            qdrift_symm.append(q_symm)
            
            print("trial done")

        trotter_ctrl = np.array(trotter_ctrl)
        qdrift_naive = np.array(qdrift_naive)
        qdrift_symm = np.array(qdrift_symm)

        tc_mean = np.mean(trotter_ctrl, axis=0)[indices]
        tc_std  = np.std(trotter_ctrl, axis=0)[indices]
        qn_mean = np.mean(qdrift_naive, axis=0)[indices]
        qn_std  = np.std(qdrift_naive, axis=0)[indices]
        qs_mean = np.mean(qdrift_symm, axis=0)[indices]
        qs_std  = np.std(qdrift_symm, axis=0)[indices]

        ax.errorbar(times, tc_mean, yerr=tc_std, label="Trotter", fmt='-o', capsize=3)
        ax.errorbar(times, qn_mean, yerr=qn_std, label="Naive qDRIFT", fmt='-s', capsize=3)
        ax.errorbar(times, qs_mean, yerr=qs_std, label="Symmetry qDRIFT", fmt='-^', capsize=3)
        
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Expectation Value")
        ax.legend()

        print("config done")

    plt.suptitle(f"H2 Time Evolution from Hartree-Fock State |{hf_state}>", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulation()
