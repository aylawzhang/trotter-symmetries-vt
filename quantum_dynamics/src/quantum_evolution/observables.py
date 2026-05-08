import numpy as np
from qiskit.quantum_info import SparsePauliOp, PauliList, Pauli, Statevector

def magnetization_operator(N):
    """
    Creates the average magnetization operator (sum of Zi / N)
    N (int): Number of qubits
    Returns: SparsePauliOp
    """
    terms=[]
    for i in range(N):
        op = ["I"]*N
        op[i] = "Z"
        terms.append(("".join(op), 1.0/N))
    return SparsePauliOp.from_list(terms)

def Z_operator(N):
    """
    Creates the Z observable where the Z is at the N/2 position.
    N (int): Number of qubits
    Returns: SparsePauliOp
    """
    op = ["I"] * N
    op[(N//2)-1] = "Z"
    return SparsePauliOp.from_list([("".join(op), 1.0)])

def number_operator(nq):
    """
    Creates the number operator observable N = sum(I - Zi) / 2
    nq (int): Number of qubits
    Returns: SparsePauliOp
    """
    pauli_list = []
    
    identity_str = "I" * nq
    pauli_list.append((identity_str, float(nq)/2))
    for i in range(nq):
        op = list("I" * nq)
        op[i] = "Z"
        pauli_list.append(("".join(op), -0.5))
    return SparsePauliOp.from_list(pauli_list)

def sample_shots(psi, N_shots):
    """
    Simulates measurement by sampling from the statevector probability distribution
    psi (Statevector): quantum state
    N_shots (int): Number of samples to take
    Returns: np.ndarray (indices of sampled states)
    """
    probs = np.abs(psi.data)**2
    states = np.arange(len(probs))
    samples = np.random.choice(states, size=N_shots, p=probs)
    return samples

def evaluate_observable(samples, N, obs):
    """
    Estimates the expectation value of a diagonal operator (Z and I only)
    samples (np.ndarray): Sampled state indices
    N (int): Number of qubits
    obs (SparsePauliOp): Diagonal observable to measure
    Returns: float
    """
    total_val = 0
    obs_dict = obs.to_list()
    
    for s in samples:
        bitstring = format(s, f"0{N}b")
        sample_contribution = 0
        for label, coeff in obs_dict:
            term_val = coeff
            for i, pauli in enumerate(reversed(label)): 
                if pauli == "Z" and bitstring[-(i+1)] == '1':
                    term_val *= -1
            sample_contribution += term_val
        total_val += np.real(sample_contribution)
    
    return total_val / len(samples)