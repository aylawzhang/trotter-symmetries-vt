import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from qiskit.quantum_info import Statevector

from quantum_evolution.hamiltonians import ising_and_transverse_hamiltonian
from quantum_evolution.observables import magnetization_operator
from quantum_evolution.trotter import trotter_evolution
from quantum_evolution.qdrift import qdrift_evolution

def simulation():
    N = 4
    J = 1.0
    delta = 1.0
    h=1.0
    T = 0.5
    shot_values=[512]
    
    psi_0 = Statevector(np.random.rand(2**N) + 1j*np.random.rand(2**N))
    psi_0 /= np.linalg.norm(psi_0)

    H_op = ising_and_transverse_hamiltonian(N, J,h)
    M_op = magnetization_operator(N)
    paulis = H_op.to_list()

    # step_values = [10,50,100]
    step_values=[100]

    plt.figure()

    #exact evol
    start_exact = time.time()
    H_mat = H_op.to_matrix()
    steps = step_values[-1]
    dt = T / steps
    psi_exact = psi_0.copy()
    exact_mags = []

    for t in range(steps + 1):
        exact_mags.append(np.real(np.vdot(psi_exact.data, (M_op @ psi_exact).data)))
        if t < steps:
            psi_exact = Statevector(expm(-1j * H_mat * dt) @ psi_exact.data)
    end_exact = time.time()
    plt.plot(np.linspace(0, T, len(exact_mags)), exact_mags, "--", label=f"exact ({end_exact-start_exact:.4f}s)", linewidth=2)
    
    #trotter
    for N_shots in shot_values:
        for steps in step_values:
            dt = T / steps
            start_trot = time.time()
            mags = trotter_evolution(paulis, psi_0, T, dt, M_op, N,N_shots)
            end_trot = time.time()
            plt.plot(np.linspace(0, T, steps + 1), mags, label=f"trotter {N_shots} shots ({end_trot-start_trot:.4f}s)")
    
    #qdrift
    steps = step_values[-1]
    dt = T / steps
    
    for N_shots in shot_values:
        start_q = time.time()
        q_avg = qdrift_evolution(paulis, psi_0, T, dt, M_op, N,N_shots)
        end_q = time.time()

        plt.plot(np.linspace(0, T, steps + 1), q_avg, label=f"qdrift avg {N_shots} shots ({end_q-start_q:.4f}s)", linewidth=2)    

    plt.xlabel("Time")
    plt.ylabel("Magnetization")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    simulation()