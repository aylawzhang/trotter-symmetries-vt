import numpy as np
import random
from collections import defaultdict
from qiskit.quantum_info import Pauli, SparsePauliOp
from .observables import sample_shots, evaluate_observable, number_operator

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
    lam = np.sum(np.abs(coeffs))
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

def naive_qdrift(paulis, psi_0, T, dt, obs, N, N_shots):
    """
    Control for qDRIFT evolution; randomizes order of Paulis.
    paulis (list): Hamiltonian terms as (label, coeff)
    psi_0 (Statevector): Initial state
    T (float): Total evolution time
    dt (float): Time step size
    obs (SparsePauliOp): Observable to measure
    N (int): Number of qubits
    N_shots (int): Number of random evolutions
    Returns: np.ndarray (averaged measurements over time)
    """
    paulis = list(paulis) 
    random.shuffle(paulis)

    coeffs = np.array([abs(item[1]) for item in paulis])
    lam = np.sum(np.abs(coeffs))
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

def group_paulis(paulis, number_op, tol=1e-10):
    """
    Helper function that groups commuting Pauli terms while preserving number operator.
    paulis (list): Hamiltonian terms as (label, coeff)
    number_op (SparsePauliOp): Number operator used for commutation check
    tol (float): Numerical tolerance for commutation test
    Returns: list (groups of commuting Pauli terms)
    """
    groups = []
    iz_group = []

    buckets = defaultdict(list)

    for label, coeff in paulis:
        p = Pauli(label)
        if all(c in ["I", "Z"] for c in label):
            iz_group.append((p, coeff))
        else:
            buckets[round(abs(np.real(coeff)), 10)].append((p, coeff))

    if iz_group:
        groups.append(iz_group)

    for bucket in buckets.values():
        for p, coeff in bucket:
            added = False
            for group in groups[1:]:
                if all(p.commutes(q) for q, _ in group):
                    test_group = group + [(p, coeff)]
                    op = SparsePauliOp([x.to_label() for x, _ in test_group],[c for _, c in test_group])
                    comm = op @ number_op - number_op @ op

                    if np.allclose(comm.to_matrix(), 0, atol=tol):
                        group.append((p, coeff))
                        added = True
                        break

            if not added:
                groups.append([(p, coeff)])

    return groups

def symmetry_qdrift(paulis, psi_0, T, dt, obs, N, N_shots):
    """
    Performs qDRIFT evolution by grouping Pauli terms with Z and I together and X and Y together.
    paulis (list): Hamiltonian terms as (label, coeff)
    psi_0 (Statevector): Initial state
    T (float): Total evolution time
    dt (float): Time step size
    obs (SparsePauliOp): Observable to measure
    N (int): Number of qubits
    N_shots (int): Number of random evolutions
    Returns: np.ndarray (averaged measurements over time)
    """

    grouped_paulis = group_paulis(paulis, number_operator(N))
    
    group_norms = [sum(abs(c) for _,c in group) for group in grouped_paulis]
    lam = sum(group_norms)
    probs = [g/lam for g in group_norms]
    
    steps = int(T/dt)
    tau = T*lam/steps
    avg_measurement = np.zeros(steps + 1)

    for _ in range(N_shots):
        psi=psi_0.copy()
        single_run=[]
        initial=sample_shots(psi,1)
        single_run.append(evaluate_observable(initial,N,obs))
        
        for _ in range(steps):
            idx = np.random.choice(len(grouped_paulis), p=probs)
            selected_group = grouped_paulis[idx]
            current_norm = group_norms[idx]
            for pauli, coeff in selected_group:
                angle = np.real(coeff) * (tau / current_norm)
                psi = np.cos(angle) * psi - 1j * np.sin(angle) * psi.evolve(pauli)
            samples=sample_shots(psi,1)
            single_run.append(evaluate_observable(samples, N, obs))        
        avg_measurement += np.array(single_run)
    return avg_measurement/N_shots