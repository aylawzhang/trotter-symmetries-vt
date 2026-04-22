import numpy as np
import time
from scipy.linalg import expm
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp, PauliList, Pauli, Statevector

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
        terms.append(("".join(reversed(op)), 1.0/N))
    return SparsePauliOp.from_list(terms)

def sample_shots(psi, N_shots):
    probs = np.abs(psi.data)**2
    states = np.arange(len(probs))
    samples = np.random.choice(states, size=N_shots, p=probs)
    return samples

def samples_to_mag(samples, N):
    total = 0
    for s in samples:
        bitstring = format(s, f"0{N}b")
        z_vals = [(1 if b == '0' else -1) for b in bitstring]
        total += np.mean(z_vals)
    return total / len(samples)

def trotter_evolution(H, paulis, psi_0, T, dt, M_op, N, N_shots):
    steps = int(T / dt)
    psi = psi_0.copy()
    mags = []

    gates =[]
    for label,coeff in paulis:
        op = Pauli(label) 
        theta = np.real(coeff) * dt
        gates.append((op, theta))
    for _ in range(steps+1):
        samples = sample_shots(psi, N_shots)
        mag_est = samples_to_mag(samples, N)
        mags.append(mag_est)
        if _ < steps:
            for op,theta in gates:
                psi = (np.cos(theta) * psi) - (1j * np.sin(theta) * psi.evolve(op))
    return mags

def qdrift_single_run(H, paulis, psi_0, T, dt, M_op, N, N_shots):
    coeffs = np.array([abs(item[1]) for item in paulis])
    lam = np.sum(coeffs)
    probabilities = coeffs/lam

    ops=[Pauli(p[0]) for p in paulis]
    signs = [np.sign(np.real(p[1])) for p in paulis]
    
    steps = int(T/dt)
    psi = psi_0.copy()
    tau = T*lam/steps

    mags=[]
    samples=sample_shots(psi,N_shots)
    mags.append(samples_to_mag(samples,N))
    for _ in range(steps):
        idx = np.random.choice(len(paulis), p=probabilities)
        theta = signs[idx] * tau
        psi = np.cos(theta) * psi - 1j * np.sin(theta) * psi.evolve(ops[idx])
        samples=sample_shots(psi,N_shots)
        mags.append(samples_to_mag(samples,N))
    return mags

def test_evolution():
    N = 4
    J = 1.0
    delta = 1.0
    h=1.0
    T = 0.5
    trials = 100
    N_shots=1024
    
    psi_0 = Statevector(np.random.rand(2**N) + 1j*np.random.rand(2**N))
    psi_0 /= np.linalg.norm(psi_0)

    H_op = ising_and_transverse_hamiltonian(N, J,h)
    M_op = magnetization_operator(N)
    paulis = H_op.to_list()

    # step_values = [10,50,100]
    step_values=[100]

    plt.figure()

    #exact evol
    start_exact = time.time()
    H_mat = H_op.to_matrix()
    steps = step_values[-1]
    dt = T / steps
    psi_exact = psi_0.copy()
    exact_mags = []

    for t in range(steps + 1):
        exact_mags.append(np.real(np.vdot(psi_exact.data, (M_op @ psi_exact).data)))
        if t < steps:
            psi_exact = Statevector(expm(-1j * H_mat * dt) @ psi_exact.data)
    end_exact = time.time()
    plt.plot(np.linspace(0, T, len(exact_mags)), exact_mags, "--", label=f"exact ({end_exact-start_exact:.4f}s)", linewidth=2)
    
    #trotter
    for steps in step_values:
        dt = T / steps
        start_trot = time.time()
        mags = trotter_evolution(H_op, paulis, psi_0, T, dt, M_op, N,N_shots)
        end_trot = time.time()
        plt.plot(np.linspace(0, T, steps + 1), mags, label=f"trotter {steps} steps ({end_trot-start_trot:.4f}s)")
    
    #qdrift
    steps = step_values[-1]
    dt = T / steps
    q_avg = np.zeros(steps + 1)

    start_q = time.time()
    for _ in range(trials):
        q_run = qdrift_single_run(H_op, paulis, psi_0, T, dt, M_op, N,N_shots)
        q_avg += np.array(q_run)
    q_avg /= trials
    end_q = time.time()

    plt.plot(np.linspace(0, T, steps + 1), q_avg, label=f"qdrift avg ({end_q-start_q:.4f}s)", linewidth=2)    
    plt.xlabel("Time")
    plt.ylabel("Magnetization")
    plt.legend()
    plt.show()

test_evolution()