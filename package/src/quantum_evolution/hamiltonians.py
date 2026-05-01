import numpy as np
from qiskit.quantum_info import SparsePauliOp, PauliList, Pauli, Statevector

def ising_and_transverse_hamiltonian(N, J, h=0.0):
    """
    Constructs Ising Hamiltonian with transverse field
    N (int): Number of qubits
    J (float): Interaction strength (ZZ)
    h (float): Transverse field strength (X)
    Returns: SparsePauliOp
    """
    terms =[]
    for i in range(N-1):
        op = ["I"]*N
        op[i], op[i+1] = "Z", "Z"
        terms.append(("".join(reversed(op)), J))
    if h != 0:
        for i in range(N):
            op = ["I"]*N
            op[i] ="X"
            terms.append(("".join(reversed(op)), h))
    return SparsePauliOp.from_list(terms)

def heisenberg_xxz_hamiltonian(N, J, delta):
    """
    Constructs the XXZ Heisenberg Hamiltonian
    N (int): Number of qubits
    J (float): Exchange interaction strength (XX, YY)
    delta (float): Anisotropy strength (ZZ)
    Returns: SparsePauliOp
    """
    terms = []
    for i in range(N-1):
        for label, coeff in [("X", J),("Y", J),("Z", delta)]:
            op = ["I"]*N
            op[i], op[i+1] = label,label
            terms.append(("".join(reversed(op)), coeff))
    return SparsePauliOp.from_list(terms)

