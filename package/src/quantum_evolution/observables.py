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
        terms.append(("".join(reversed(op)), 1.0/N))
    return SparsePauliOp.from_list(terms)

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