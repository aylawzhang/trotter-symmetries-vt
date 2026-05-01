import numpy as np
from qiskit.quantum_info import Pauli
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