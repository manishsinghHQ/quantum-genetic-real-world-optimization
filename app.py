import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Qiskit
from qiskit import QuantumCircuit, Aer, execute

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="QGA Optimization Lab", layout="wide")

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Controls")

n_assets = st.sidebar.slider("Number of Assets", 5, 30, 10)
generations = st.sidebar.slider("Generations", 10, 100, 30)
risk_lambda = st.sidebar.slider("Risk Factor (λ)", 0.0, 1.0, 0.5)
algorithm = st.sidebar.selectbox("Algorithm", ["QGA", "GA", "PSO", "DE"])
mode = st.sidebar.radio("Execution Mode", ["Simulator"])

# =========================
# DATA GENERATION
# =========================
np.random.seed(42)
returns = np.random.uniform(0.01, 0.2, n_assets)
risk = np.random.uniform(0.01, 0.1, n_assets)

def fitness(solution):
    return np.sum(solution * returns) - risk_lambda * np.sum(solution * risk)

# =========================
# QGA IMPLEMENTATION
# =========================
def run_qga():
    n = n_assets
    pop_size = 20

    theta = np.full((pop_size, n), np.pi / 4)
    best = None
    best_fit = -np.inf
    history = []

    for gen in range(generations):
        population = []

        for i in range(pop_size):
            probs = np.sin(theta[i]) ** 2
            individual = (np.random.rand(n) < probs).astype(int)
            population.append(individual)

        fitness_vals = np.array([fitness(ind) for ind in population])
        idx = np.argmax(fitness_vals)

        if fitness_vals[idx] > best_fit:
            best_fit = fitness_vals[idx]
            best = population[idx]

        # Update theta
        for i in range(pop_size):
            for j in range(n):
                if population[i][j] != best[j]:
                    theta[i][j] += 0.05 if best[j] == 1 else -0.05

        history.append(best_fit)

    return best_fit, history

# =========================
# GA IMPLEMENTATION
# =========================
def run_ga():
    pop_size = 20
    pop = np.random.randint(0, 2, (pop_size, n_assets))
    history = []

    for _ in range(generations):
        fitness_vals = np.array([fitness(ind) for ind in pop])
        best_idx = np.argmax(fitness_vals)

        history.append(fitness_vals[best_idx])

        # Selection
        parents = pop[np.argsort(fitness_vals)[-10:]]

        # Crossover
        children = []
        for _ in range(pop_size):
            p1, p2 = parents[np.random.randint(0, 10, 2)]
            point = np.random.randint(1, n_assets)
            child = np.concatenate([p1[:point], p2[point:]])
            children.append(child)

        pop = np.array(children)

    return max(history), history

# =========================
# PSO IMPLEMENTATION
# =========================
def run_pso():
    pop_size = 20
    X = np.random.rand(pop_size, n_assets)
    V = np.random.rand(pop_size, n_assets)

    best_local = X.copy()
    best_global = None
    best_fit = -np.inf
    history = []

    for _ in range(generations):
        for i in range(pop_size):
            sol = (X[i] > 0.5).astype(int)
            fit = fitness(sol)

            if fit > best_fit:
                best_fit = fit
                best_global = X[i]

        history.append(best_fit)

        V = 0.5 * V + np.random.rand() * (best_local - X) + np.random.rand() * (best_global - X)
        X = X + V

    return best_fit, history

# =========================
# DE IMPLEMENTATION
# =========================
def run_de():
    pop_size = 20
    pop = np.random.rand(pop_size, n_assets)
    history = []

    for _ in range(generations):
        new_pop = []

        for i in range(pop_size):
            a, b, c = pop[np.random.choice(pop_size, 3, replace=False)]
            mutant = a + 0.5 * (b - c)
            trial = np.clip(mutant, 0, 1)

            sol = (trial > 0.5).astype(int)
            if fitness(sol) > fitness((pop[i] > 0.5).astype(int)):
                new_pop.append(trial)
            else:
                new_pop.append(pop[i])

        pop = np.array(new_pop)

        fits = [fitness((p > 0.5).astype(int)) for p in pop]
        history.append(max(fits))

    return max(history), history

# =========================
# QISKIT DEMO
# =========================
def run_qiskit_demo(n):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.measure_all()

    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=512).result()
    counts = result.get_counts()

    return qc, counts

# =========================
# MAIN UI
# =========================
st.title("⚛️ Quantum Genetic Algorithm Optimization Lab")

if st.button("🚀 Run Optimization"):
    start = time.time()

    if algorithm == "QGA":
        best, history = run_qga()
    elif algorithm == "GA":
        best, history = run_ga()
    elif algorithm == "PSO":
        best, history = run_pso()
    else:
        best, history = run_de()

    end = time.time()

    st.success(f"Best Fitness: {best:.4f}")
    st.info(f"Execution Time: {end - start:.2f} sec")

    # Plot
    fig, ax = plt.subplots()
    ax.plot(history)
    ax.set_title("Convergence Curve")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    st.pyplot(fig)

    # Qiskit Visualization
    st.subheader("⚛️ Quantum Circuit Demo")
    qc, counts = run_qiskit_demo(min(n_assets, 5))
    st.pyplot(qc.draw(output='mpl'))

    st.write("Measurement Counts:", counts)

# =========================
# BENCHMARK BUTTON
# =========================
if st.button("📊 Compare All Algorithms"):
    results = {}

    for algo, func in {
        "QGA": run_qga,
        "GA": run_ga,
        "PSO": run_pso,
        "DE": run_de
    }.items():
        best, _ = func()
        results[algo] = best

    df = pd.DataFrame(list(results.items()), columns=["Algorithm", "Fitness"])
    st.bar_chart(df.set_index("Algorithm"))
