import numpy as np
from qiskit.quantum_info import SparsePauliOp, PauliList, Pauli, Statevector
import pennylane as qml

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

def h2_hamiltonian(bond_length_A):
    symbols = ["H", "H"]
    bohr_dist = bond_length_A * 1.889726
    geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bohr_dist]])
    molecule = qml.qchem.Molecule(symbols, geometry)
    hamiltonian, qubits = qml.qchem.molecular_hamiltonian(molecule)
    terms = []
    coeffs, ops = hamiltonian.terms()
    
    for coeff, op in zip(coeffs, ops):
        pauli_string = ["I"] * qubits
        for pw in op.pauli_rep.keys():
            for wire, p_char in pw.items():
                pauli_string[wire] = p_char
            
        terms.append(("".join(reversed(pauli_string)), float(coeff)))
        
    return SparsePauliOp.from_list(terms), qubits