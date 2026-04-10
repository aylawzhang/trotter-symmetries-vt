import numpy as np
import time
from scipy.linalg import expm
from qiskit.quantum_info import SparsePauliOp

def ising_and_transverse_hamiltonian(N, J, h=0.0):
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
    terms = []
    for i in range(N-1):
        for label, coeff in [("X", J),("Y", J),("Z", delta)]:
            op = ["I"]*N
            op[i], op[i+1] = label,label
            terms.append(("".join(reversed(op)), coeff))
    return SparsePauliOp.from_list(terms)

def magnetization_operator(N):
    terms=[]
    for i in range(N):
        op = ["I"]*N
        op[i] = "Z"
        terms.append(("".join(reversed(op)), 1.0))
    return SparsePauliOp.from_list(terms)

def trotter_evolution(H, psi_0, T, dt, M_op, N):
    steps = int(T / dt)
    psi = psi_0.copy()
    mags = []
    M_mat = M_op.to_matrix()
    gates =[]
    for label,coeff in H.to_list():
        gate = expm(-1j*coeff * SparsePauliOp(label).to_matrix() *dt)
        gates.append(gate)
    for _ in range(steps+1):
        mags.append(np.real(np.vdot(psi, M_mat@psi)) / N)
        if _ < steps:
            for U in gates:
                psi = U@psi
    return mags

def qdrift_single_run(H, psi_0, T, dt, M_op, N):
    pauli_list = H.to_list()
    coeffs = np.array([abs(item[1]) for item in pauli_list])
    lam = np.sum(coeffs)
    M_mat = M_op.to_matrix()
    term_mats = [SparsePauliOp(p[0]).to_matrix() for p in pauli_list]
    steps = int(T/dt)
    probabilities = coeffs/lam
    psi = psi_0.copy()
    tau = T/steps
    for _ in range(steps):
        idx = np.random.choice(len(pauli_list), p=probabilities)
        coeff = pauli_list[idx][1]
        gate = expm(-1j * np.sign(coeff)*lam*tau*term_mats[idx])
        psi = gate@psi
    return np.real(np.vdot(psi, M_mat @ psi))/N

def test_evolution():
    N = 4
    J = 1.0
    delta = 0.5
    T = 0.5
    dt = 0.01
    trials = 500
    
    psi_0 = np.random.rand(2**N) + 1j * np.random.rand(2**N)
    psi_0 /= np.linalg.norm(psi_0)
    
    H_op = heisenberg_xxz_hamiltonian(N, J, delta)
    M_op = magnetization_operator(N)

    start = time.time()
    psi_exact = expm(-1j *H_op.to_matrix() *T) @psi_0
    exact_mag = np.real(np.vdot(psi_exact, M_op.to_matrix() @ psi_exact)) / N
    time_exact = time.time() -start

    start = time.time()
    t_mags = trotter_evolution(H_op, psi_0, T, dt, M_op, N)
    time_trotter = time.time() -start

    start = time.time()
    q_results = [qdrift_single_run(H_op, psi_0, T, dt, M_op, N) for _ in range(trials)]
    time_qdrift = time.time() -start

    print(f"Exact: {exact_mag:.6f} ({time_exact:.4f}s)")
    print(f"Trotter: {t_mags[-1]:.6f} ({time_trotter:.4f}s)")
    print(f"qDRIFT Avg: {np.mean(q_results):.6f} ({time_qdrift:.4f}s)")

if __name__ == "__main__":
    test_evolution()