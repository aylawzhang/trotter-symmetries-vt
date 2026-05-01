import numpy as np
from qiskit.quantum_info import Pauli
from .observables import sample_shots, evaluate_observable

def qdrift_evolution(paulis, psi_0, T, dt, obs, N, N_shots):
    """
    Performs qDRIFT evolution. 
    paulis (list): Hamiltonian terms as (label, coeff)
    psi_0 (Statevector): Initial state
    T (float): Total evolution time
    dt (float): Time step size
    obs (SparsePauliOp): Observable to measure
    N (int): Number of qubits
    N_shots (int): Number of random evolutions
    Returns: np.ndarray (averaged measurements over time)
    """
    coeffs = np.array([abs(item[1]) for item in paulis])
    lam = np.sum(coeffs)
    probabilities = coeffs/lam

    ops=[Pauli(p[0]) for p in paulis]
    signs = [np.sign(np.real(p[1])) for p in paulis]
    
    steps = int(T/dt)
    tau = T*lam/steps

    avg_measurement = np.zeros(steps + 1)

    for _ in range(N_shots):
        psi=psi_0.copy()
        single_run=[]
        initial=sample_shots(psi,1)
        single_run.append(evaluate_observable(initial,N,obs))
        for _ in range(steps):
            idx = np.random.choice(len(paulis), p=probabilities)
            theta = signs[idx] * tau
            psi = np.cos(theta) * psi - 1j * np.sin(theta) * psi.evolve(ops[idx])
            
            samples=sample_shots(psi,1)
            single_run.append(evaluate_observable(samples, N, obs))        
        avg_measurement += np.array(single_run)
    return avg_measurement/N_shots