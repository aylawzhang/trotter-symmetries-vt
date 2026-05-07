import numpy as np
import random
from qiskit.quantum_info import Pauli
from .observables import sample_shots, evaluate_observable
from .trotter import check_internal_commutation, check_obs_commutation

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

def random_qdrift(paulis, psi_0, T, dt, obs, N, N_shots):
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

    term_A=[] #Z and I
    term_B=[] #X and Y

    for label, coeff in paulis:
        p_obj = Pauli(label)
        if all(c in ['I', 'Z'] for c in label):
            term_A.append((p_obj, coeff))
        else:
            term_B.append((p_obj, coeff))

    A_check = check_internal_commutation(term_A) and check_obs_commutation(term_A, obs)
    B_check = check_internal_commutation(term_B) and check_obs_commutation(term_B, obs)

    if not A_check:
        print("Warning: Commutation check failed for Group A. Approximation error may occur.")
    if not B_check:
        print("Warning: Commutation check failed for Group B. Approximation error may occur.")
    
    norm_A = sum(abs(c) for _, c in term_A)
    norm_B = sum(abs(c) for _, c in term_B)
    lam = norm_A + norm_B

    probs = [norm_A / lam, norm_B / lam]
    groups = [term_A, term_B]
    group_norms = [norm_A, norm_B]
    
    steps = int(T/dt)
    tau = T*lam/steps
    avg_measurement = np.zeros(steps + 1)

    for _ in range(N_shots):
        psi=psi_0.copy()
        single_run=[]
        initial=sample_shots(psi,1)
        single_run.append(evaluate_observable(initial,N,obs))
        
        for _ in range(steps):
            idx = np.random.choice([0, 1], p=probs)
            selected_group = groups[idx]
            current_norm = group_norms[idx]
            for pauli, coeff in selected_group:
                angle = np.real(coeff) * (tau / current_norm)
                psi = np.cos(angle) * psi - 1j * np.sin(angle) * psi.evolve(pauli)
            samples=sample_shots(psi,1)
            single_run.append(evaluate_observable(samples, N, obs))        
        avg_measurement += np.array(single_run)
    return avg_measurement/N_shots