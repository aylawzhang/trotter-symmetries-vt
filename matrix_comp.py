import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

paulix=np.array([[0,1],[1,0]])
pauliy=np.array([[0,-1j],[1j,0]])
pauliz=np.array([[1,0],[0,-1]])
identity=np.array([[1,0],[0,1]])

def heisenberg_xxz_hamiltonian(N, J, delta):
    coeffs=[]
    terms=[]
    for i in range(N-1):
        for pauli, coeff in [(paulix,J), (pauliy,J), (pauliz,delta)]:
            ops = [identity]*N
            ops[i] = pauli
            ops[i+1] = pauli
            term = ops[0]
            for op in ops[1:]:
                term = np.kron(term, op)
            coeffs.append(coeff)
            terms.append(term)
    return coeffs,terms

def magnetization(psi, M, N):
    return np.real(np.vdot(psi, M @ psi)) / N

def time_evol_operator(H,t):
    return expm(-1j * H * t)


def magnetization_operator(N):
    M = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N):
        ops=[identity]*N
        ops[i]=pauliz
        term=ops[0]
        for o in ops[1:]:
            term=np.kron(term, o) 
        M+=term
    return M

def qdrift_evolution(N, coeffs, terms, steps, time_val, psi):
    lam = sum([abs(x) for x in coeffs])
    tau = time_val * lam / steps
    qdrift_gates = [expm(-1j * np.sign(c) * tau * T) for c, T in zip(coeffs, terms)]
    probabilities = np.array([abs(c) for c in coeffs]) / lam

    psi = psi.copy()
    M_operator = magnetization_operator(N)

    mags = [magnetization(psi, M_operator, N)]

    indices = np.random.choice(len(qdrift_gates), size=steps, p=probabilities)

    for i in range(steps):
        psi = qdrift_gates[indices[i]] @ psi
        mags.append(magnetization(psi, M_operator, N))

    return mags

def transverse_ising_hamiltonian(N,h,J):
    H = np.zeros((2**N, 2**N), dtype=complex)
    
    terms=[]
    coeffs=[]
    for i in range(N-1):
        operators=[identity for _ in range(N)]
        operators[i]=pauliz
        operators[i+1]=pauliz
        term=operators[0]
        for t in operators[1:]:
            term=np.kron(term, t) 
        terms.append(term)
        coeffs.append(J)
    
    for i in range(N):
        operators=[identity for _ in range(N)]
        operators[i]=paulix
        term=operators[0]
        for t in operators[1:]:
            term=np.kron(term, t) 
        terms.append(term)
        coeffs.append(h)
    
    return coeffs,terms

def exact_evolution(N, coeffs, terms, total_time, total_steps,M_operator):

    dt = total_time / total_steps

    # build full Hamiltonian ONCE (minimal change)
    H = np.zeros_like(terms[0], dtype=complex)
    for c, t in zip(coeffs, terms):
        H += c * t

    U_exact = time_evol_operator(H, dt)

    psi = psi0.copy()

    magnetizations = [magnetization(psi, M_operator, N)]

    for _ in range(total_steps):
        psi = U_exact @ psi
        magnetizations.append(magnetization(psi, M_operator, N))

    return magnetizations

def transverse_ising_trotter(N, time, steps, coeffs, terms,M_operator):
    dt = time / steps

    psi = psi0.copy()

    gates = [expm(-1j * coeffs[i] * dt * terms[i]) for i in range(len(terms))]

    magnetizations = [magnetization(psi, M_operator, N)]

    for _ in range(steps):
        for U in gates:
            psi = U @ psi
        magnetizations.append(magnetization(psi, M_operator, N))

    return magnetizations
        
N=5
time=0.5
h=1.0
J=1.0

STEP_VALUES = [10,50,100]

coeffs,terms = transverse_ising_hamiltonian(N, h, J)
M_op=magnetization_operator(N)

# initial state
psi0 = np.random.rand(2**N) + 1j * np.random.rand(2**N)
psi0 /= np.linalg.norm(psi0)

plt.figure(figsize=(10,6))

# -------------------------
# EXACT
# -------------------------
steps = STEP_VALUES[-1]
dt = time / steps

start = time_module = __import__("time").time()
exact_mag = exact_evolution(N, coeffs,terms,time, steps,M_op)
end = __import__("time").time()

plt.plot(np.linspace(0, time, steps+1),
         exact_mag,
         "--",
         label=f"Exact ({end-start:.4f}s)",
         linewidth=2)

# -------------------------
# TROTTER
# -------------------------
for S in STEP_VALUES:
    start = __import__("time").time()
    mag = transverse_ising_trotter(N, time, S, coeffs,terms,M_op)
    end = __import__("time").time()

    plt.plot(np.linspace(0, time, S+1),
             mag,
             label=f"Trotter S={S} ({end-start:.4f}s)")

# -------------------------
# QDRIFT
# -------------------------
steps = STEP_VALUES[-1]
trials = 300
q_avg = np.zeros(steps+1)

coeffs, terms = transverse_ising_hamiltonian(N, J, 1.0)

start = __import__("time").time()

for _ in range(trials):
    mags = qdrift_evolution(N, coeffs, terms, steps, time, psi0)
    q_avg += np.array(mags)

q_avg /= trials

end = __import__("time").time()

plt.plot(np.linspace(0, time, steps+1),
         q_avg,
         label=f"QDrift ({end-start:.4f}s)",
         linewidth=2)

# -------------------------
# PLOT
# -------------------------
plt.xlabel("Time")
plt.ylabel("Magnetization")
plt.title("Matrix vs Trotter vs QDrift")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.show()