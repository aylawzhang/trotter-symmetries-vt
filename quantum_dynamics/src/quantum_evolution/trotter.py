import numpy as np
import random
from qiskit.quantum_info import Pauli, SparsePauliOp
from .observables import sample_shots, evaluate_observable

def trotter_evolution(paulis, psi_0, T, dt, obs, N, N_shots):
    """
    Performs first-order Trotter evolution and measures an observable.
    paulis (list): Hamiltonian terms as (label, coeff)
    psi_0 (Statevector): Initial state
    T (float): Total evolution time
    dt (float): Time step size
    obs (SparsePauliOp): Observable to measure
    N (int): Number of qubits
    N_shots (int): Measurement shots per time step
    Returns: list (measured values over time)
    """
    steps = int(T / dt)
    psi = psi_0.copy()
    measurements = []

    gates =[]
    for label,coeff in paulis:
        op = Pauli(label) 
        theta = np.real(coeff) * dt
        gates.append((op, theta))

    for _ in range(steps+1):
        samples = sample_shots(psi, N_shots)
        est = evaluate_observable(samples,N,obs)
        measurements.append(est)
        if _ < steps:
            for op,theta in gates:
                psi = (np.cos(theta) * psi) - (1j * np.sin(theta) * psi.evolve(op))
    return measurements

def random_trotter(paulis, psi_0, T, dt, obs, N, N_shots):
    """
    Control approach for first-order Trotter evolution. Randomly shuffles Pauli list.
    paulis (list): Hamiltonian terms as (label, coeff)
    psi_0 (Statevector): Initial state
    T (float): Total evolution time
    dt (float): Time step size
    obs (SparsePauliOp): Observable to measure
    N (int): Number of qubits
    N_shots (int): Measurement shots per time step
    Returns: list (measured values over time)
    """
    paulis = list(paulis) 
    random.shuffle(paulis)
    
    steps = int(T / dt)
    psi = psi_0.copy()
    measurements = []

    gates =[]
    for label,coeff in paulis:
        op = Pauli(label) 
        theta = np.real(coeff) * dt
        gates.append((op, theta))

    for _ in range(steps+1):
        samples = sample_shots(psi, N_shots)
        est = evaluate_observable(samples,N,obs)
        measurements.append(est)
        if _ < steps:
            for op,theta in gates:
                psi = (np.cos(theta) * psi) - (1j * np.sin(theta) * psi.evolve(op))
    return measurements

def check_internal_commutation(group):
    """
    Verifies if all Hamiltonian terms within a group commute with each other.
    group (list): Hamiltonian terms as (Pauli, coeff)
    Returns: bool (True if all terms commute, False otherwise)
    """
    for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if not group[i][0].commutes(group[j][0]):
                    return False
    return True

def check_obs_commutation(group, obs):
    """
    Verifies if all terms in a group commute with the observable being measured.
    group (list): Hamiltonian terms as (Pauli, coeff)
    obs (SparsePauliOp): The observable to check against
    Returns: bool (True if all terms commute with the observable, False otherwise)
    """
    labels = []
    coeffs = []
    for pauli, coeff in group:
        labels.append(pauli.to_label())
        coeffs.append(coeff)
    group_op = SparsePauliOp(labels, coeffs)
    commutator = group_op @ obs - obs @ group_op
    return np.allclose(commutator.to_matrix(), 0, atol=1e-10)

def symmetry_trotter(paulis, psi_0, T, dt, obs, N, N_shots):
    """
    Symmetry-informed Trotterization approach. Splits paulis into group A (Z and I) and group B (X and Y), 
    checks commutation within group and with observable. 
    paulis (list): Hamiltonian terms as (label, coeff)
    psi_0 (Statevector): Initial state
    T (float): Total evolution time
    dt (float): Time step size
    obs (SparsePauliOp): Observable to measure
    N (int): Number of qubits
    N_shots (int): Measurement shots per time step
    Returns: list (measured values over time)
    """

    gates_A=[]
    gates_B=[]
    for label, coeff in paulis:
        gate = (Pauli(label), np.real(coeff) * dt)
        if all(c in ['I', 'Z'] for c in label):
            gates_A.append(gate)
        else:
            gates_B.append(gate)

    A_check = check_internal_commutation(gates_A) and check_obs_commutation(gates_A,obs)
    B_check = check_internal_commutation(gates_B) and check_obs_commutation(gates_B,obs)

    if not A_check:
        print("Warning: Commutation check failed for Group A. Approximation error may occur.")
    if not B_check:
        print("Warning: Commutation check failed for Group B. Approximation error may occur.")

    steps = int(T / dt)
    psi = psi_0.copy()
    measurements = []

    for _ in range(steps+1):
        samples = sample_shots(psi, N_shots)
        est = evaluate_observable(samples,N,obs)
        measurements.append(est)
        if _ < steps:
            for op,theta in gates_A:
                psi = (np.cos(theta) * psi) - (1j * np.sin(theta) * psi.evolve(op))
            for op,theta in gates_B:
                psi = (np.cos(theta) * psi) - (1j * np.sin(theta) * psi.evolve(op))
    return measurements