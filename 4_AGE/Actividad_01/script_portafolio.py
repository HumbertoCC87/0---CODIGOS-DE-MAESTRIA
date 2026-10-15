#Maestria en inteligencia artificial
#Algoritmos Evolutivos

#Este script implementa la Opción 7.1: Optimización Evolutiva de Carteras de Inversión con Penalización por Violación de Restricciones y Costo de Transacción.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ==============================================================================
# CONFIGURACIÓN Y PARÁMETROS DEL MODELO (Opción 7.1)
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_FRONTIER = os.path.join(SCRIPT_DIR, 'grafico_frontera_riesgo.png')
FILE_WEIGHTS = os.path.join(SCRIPT_DIR, 'grafico_pesos_cartera.png')
FILE_JSON = os.path.join(SCRIPT_DIR, 'resultados_portafolio.json')

TICKERS = ['AAPL', 'MSFT', 'NVDA', 'PEP', 'WMT', 'KO', 'JPM', 'BRK-B', 'JNJ', 'CAT', 'XOM']
NUM_ASSETS = len(TICKERS)
PERIOD = "3y"
TRADING_DAYS = 252

# Ecuación (39) Parámetros
LAMBDA = 0.5           # Aversión al riesgo balanceada
TRANSACTION_COST = 0.001 # Costo de transacción base c(w) = 0.1%
RHO = 100.0            # Coeficiente de penalización por violación de restricciones

# Parámetros del Algoritmo Evolutivo (GA)
POP_SIZE = 50
GENERATIONS = 100
CX_PROB = 0.8
MUT_PROB = 0.3
MUT_SIGMA = 0.05
ELITISM_COUNT = 2
SEED = 42

np.random.seed(SEED)

# ==============================================================================
# 1. DESCARGA DE DATOS Y CÁLCULOS FINANCIEROS ANUALIZADOS
# ==============================================================================
def download_data(tickers, period=PERIOD):
    print(f"[1/5] Descargando datos de yfinance para {len(tickers)} activos ({period})...")
    data = yf.download(tickers, period=period, auto_adjust=False)
    
    if 'Adj Close' in data:
        prices = data['Adj Close']
    elif 'Close' in data:
        prices = data['Close']
    else:
        prices = data
        
    prices = prices[tickers].dropna()
    return prices

def calculate_financial_metrics(prices):
    print("[2/5] Calculando rendimientos diarios, vector mu y matriz de covarianzas Sigma...")
    daily_returns = prices.pct_change().dropna()
    mu = daily_returns.mean().values * TRADING_DAYS
    sigma = daily_returns.cov().values * TRADING_DAYS
    return mu, sigma, daily_returns

# ==============================================================================
# 2. FORMULACIÓN DE LA FUNCIÓN DE EVALUACIÓN J(w) Y DIVERSIDAD D_port
# ==============================================================================
def compute_penalty_V(w):
    sum_violation = (np.sum(w) - 1.0) ** 2
    neg_violation = np.sum(np.maximum(0.0, -w) ** 2)
    return sum_violation + neg_violation

def evaluate_J(w, mu, sigma, lambd=LAMBDA, c=TRANSACTION_COST, rho=RHO):
    w = np.array(w)
    variance = float(w.T @ sigma @ w)
    ret = float(mu.T @ w)
    V = compute_penalty_V(w)
    J = lambd * variance - (1.0 - lambd) * ret + c + rho * V
    return J

def compute_population_diversity(pop):
    N = len(pop)
    if N <= 1:
        return 0.0
    total_dist = 0.0
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(pop[i] - pop[j])
            total_dist += dist
            count += 1
    return total_dist / count if count > 0 else 0.0

# ==============================================================================
# 3. ALGORITMO EVOLUTIVO EN REALES & OPERADORES
# ==============================================================================
def repair_weights(w):
    w_repaired = np.maximum(0.0, w)
    s = np.sum(w_repaired)
    if s > 0:
        return w_repaired / s
    else:
        return np.ones(len(w)) / len(w)

def generate_random_feasible_portfolio(n_assets):
    w = np.random.uniform(0.01, 1.0, n_assets)
    return w / np.sum(w)

def tournament_selection(pop, fitnesses, k=3):
    selected_idx = np.random.choice(len(pop), size=k, replace=False)
    best_idx = selected_idx[np.argmin(fitnesses[selected_idx])]
    return pop[best_idx].copy()

def arithmetic_crossover(parent1, parent2):
    alpha = np.random.uniform(0.0, 1.0)
    child1 = alpha * parent1 + (1.0 - alpha) * parent2
    child2 = (1.0 - alpha) * parent1 + alpha * parent2
    return repair_weights(child1), repair_weights(child2)

def gaussian_mutation(w, prob=MUT_PROB, sigma=MUT_SIGMA):
    w_mut = w.copy()
    for i in range(len(w_mut)):
        if np.random.rand() < prob:
            w_mut[i] += np.random.normal(0.0, sigma)
    return repair_weights(w_mut)

def run_evolutionary_algorithm(mu, sigma, pop_size=POP_SIZE, generations=GENERATIONS):
    print(f"[3/5] Ejecutando Optimización Evolutiva (N={pop_size}, Gen={generations})...")
    pop = np.array([generate_random_feasible_portfolio(NUM_ASSETS) for _ in range(pop_size)])
    best_history = []
    
    for gen in range(generations):
        fitnesses = np.array([evaluate_J(ind, mu, sigma) for ind in pop])
        best_j = np.min(fitnesses)
        best_history.append(best_j)
        
        sorted_indices = np.argsort(fitnesses)
        new_pop = [pop[idx].copy() for idx in sorted_indices[:ELITISM_COUNT]]
        
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop, fitnesses)
            p2 = tournament_selection(pop, fitnesses)
            
            if np.random.rand() < CX_PROB:
                c1, c2 = arithmetic_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
                
            c1 = gaussian_mutation(c1)
            c2 = gaussian_mutation(c2)
            
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
                
        pop = np.array(new_pop)
        
    final_fitnesses = np.array([evaluate_J(ind, mu, sigma) for ind in pop])
    best_idx = np.argmin(final_fitnesses)
    best_portfolio = pop[best_idx]
    diversity = compute_population_diversity(pop)
    
    return best_portfolio, pop, best_history, diversity

# ==============================================================================
# 4. GENERACIÓN DE VISUALIZACIONES Y SALIDAS
# ==============================================================================
def plot_risk_return_frontier(mu, sigma, w_1N, w_rand, w_opt, final_pop, output_filename=FILE_FRONTIER):
    plt.figure(figsize=(10, 6), dpi=300)
    
    num_samples = 1000
    random_samples = np.array([generate_random_feasible_portfolio(NUM_ASSETS) for _ in range(num_samples)])
    sample_returns = [float(mu.T @ w) for w in random_samples]
    sample_variances = [float(w.T @ sigma @ w) for w in random_samples]
    
    plt.scatter(sample_variances, sample_returns, c='lightgray', alpha=0.5, s=15, label='Espacio de Carteras Aleatorias')
    
    pop_returns = [float(mu.T @ w) for w in final_pop]
    pop_variances = [float(w.T @ sigma @ w) for w in final_pop]
    plt.scatter(pop_variances, pop_returns, c='lightgreen', edgecolors='green', s=30, alpha=0.7, label='Población Final GA')
    
    ret_1N = float(mu.T @ w_1N)
    var_1N = float(w_1N.T @ sigma @ w_1N)
    plt.scatter([var_1N], [ret_1N], c='red', s=140, marker='^', zorder=5, label='Cartera Equitativa (1/N)')
    
    ret_rand = float(mu.T @ w_rand)
    var_rand = float(w_rand.T @ sigma @ w_rand)
    plt.scatter([var_rand], [ret_rand], c='orange', s=140, marker='s', zorder=5, label='Cartera Aleatoria Factible')
    
    ret_opt = float(mu.T @ w_opt)
    var_opt = float(w_opt.T @ sigma @ w_opt)
    plt.scatter([var_opt], [ret_opt], c='gold', edgecolors='black', linewidths=1.5, s=220, marker='*', zorder=6, label='Cartera Optimizada GA')
    
    plt.title('Frontera de Riesgo vs. Rendimiento Esperado (Opción 7.1)', fontsize=14, fontweight='bold')
    plt.xlabel('Riesgo Anualizado / Varianza ($\sigma_p^2$)', fontsize=12)
    plt.ylabel('Rendimiento Esperado Anualizado ($\mu_p$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"[4/5] Gráfico guardado: '{output_filename}'")

def plot_weight_allocation(tickers, w_1N, w_opt, output_filename=FILE_WEIGHTS):
    x = np.arange(len(tickers))
    width = 0.35
    
    plt.figure(figsize=(12, 6), dpi=300)
    plt.bar(x - width/2, w_1N, width, label='Cartera Equitativa (1/N)', color='#4c72b0', alpha=0.85)
    plt.bar(x + width/2, w_opt, width, label='Cartera Optimizada GA', color='#55a868', alpha=0.85)
    
    plt.xlabel('Activos (Tickers S&P 500)', fontsize=12, fontweight='bold')
    plt.ylabel('Peso Asignado ($w_i$)', fontsize=12, fontweight='bold')
    plt.title('Asignación de Pesos por Activo: Cartera Optimizada GA vs. 1/N', fontsize=14, fontweight='bold')
    plt.xticks(x, tickers, fontsize=11, fontweight='bold')
    plt.ylim(0, max(max(w_opt), max(w_1N)) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    for i in range(len(tickers)):
        plt.text(x[i] + width/2, w_opt[i] + 0.005, f"{w_opt[i]:.1%}", ha='center', va='bottom', fontsize=9, rotation=45)
        
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"[4/5] Gráfico guardado: '{output_filename}'")

# ==============================================================================
# 5. IMPRESIÓN EN CONSOLA Y SALIDA DE RESULTADOS
# ==============================================================================
def print_markdown_results(tickers, mu, sigma, w_1N, w_rand, w_opt, pop_rand, final_pop_ga):
    ret_1N = float(mu.T @ w_1N)
    var_1N = float(w_1N.T @ sigma @ w_1N)
    vol_1N = np.sqrt(var_1N)
    j_1N = evaluate_J(w_1N, mu, sigma)
    div_1N = 0.0
    
    ret_rand = float(mu.T @ w_rand)
    var_rand = float(w_rand.T @ sigma @ w_rand)
    vol_rand = np.sqrt(var_rand)
    j_rand = evaluate_J(w_rand, mu, sigma)
    div_rand = compute_population_diversity(pop_rand)
    
    ret_opt = float(mu.T @ w_opt)
    var_opt = float(w_opt.T @ sigma @ w_opt)
    vol_opt = np.sqrt(var_opt)
    j_opt = evaluate_J(w_opt, mu, sigma)
    div_ga = compute_population_diversity(final_pop_ga)

    print("\n" + "="*80)
    print("RESULTADOS NUMÉRICOS EN FORMATO MARKDOWN ESTRICTO")
    print("="*80 + "\n")

    print(r"### Tabla 1: Vector $\mu$ (Rendimiento Esperado Anualizado por Activo)" + "\n")
    print(r"| Activo (Ticker) | Rendimiento Anualizado ($\mu_i$) | Rendimiento Porcentual (%) |")
    print("| :--- | :---: | :---: |")
    for t, m in zip(tickers, mu):
        print(f"| {t} | {m:.6f} | {m*100:.2f}% |")
    print("\n")

    print(r"### Tabla 2: Matriz de Covarianzas Anualizada $\Sigma$ (11 x 11)" + "\n")
    header = "| Activo | " + " | ".join(tickers) + " |"
    divider = "| :--- | " + " | ".join([":---:"] * len(tickers)) + " |"
    print(header)
    print(divider)
    for i, t in enumerate(tickers):
        row_vals = " | ".join([f"{sigma[i, j]:.6f}" for j in range(len(tickers))])
        print(f"| **{t}** | {row_vals} |")
    print("\n")

    print(r"### Tabla 3: Cuadro Comparativo de Desempeño entre Carteras" + "\n")
    print(r"| Cartera / Estrategia | Rendimiento Esperado ($\mu_p$) | Riesgo / Varianza ($\sigma_p^2$) | Volatilidad ($\sigma_p$) | Función de Evaluación $J(w)$ | Diversidad Poblacional $D_{port}$ |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    print(f"| **Línea Base 1: Cartera Equitativa (1/N)** | {ret_1N:.6f} ({ret_1N*100:.2f}%) | {var_1N:.6f} | {vol_1N:.6f} ({vol_1N*100:.2f}%) | {j_1N:.6f} | {div_1N:.6f} |")
    print(f"| **Línea Base 2: Muestra Aleatoria Factible** | {ret_rand:.6f} ({ret_rand*100:.2f}%) | {var_rand:.6f} | {vol_rand:.6f} ({vol_rand*100:.2f}%) | {j_rand:.6f} | {div_rand:.6f} |")
    print(f"| **Cartera Optimizada (Algoritmo Evolutivo)** | {ret_opt:.6f} ({ret_opt*100:.2f}%) | {var_opt:.6f} | {vol_opt:.6f} ({vol_opt*100:.2f}%) | **{j_opt:.6f}** | {div_ga:.6f} |")
    print("\n")

    results_json = {
        "tickers": tickers,
        "vector_mu": {t: float(m) for t, m in zip(tickers, mu)},
        "matriz_sigma": sigma.tolist(),
        "cartera_1_N": {
            "pesos": {t: float(w) for t, w in zip(tickers, w_1N)},
            "mu_p": ret_1N,
            "sigma_p_2": var_1N,
            "sigma_p": vol_1N,
            "J_w": j_1N,
            "D_port": div_1N
        },
        "cartera_aleatoria": {
            "pesos": {t: float(w) for t, w in zip(tickers, w_rand)},
            "mu_p": ret_rand,
            "sigma_p_2": var_rand,
            "sigma_p": vol_rand,
            "J_w": j_rand,
            "D_port": div_rand
        },
        "cartera_optimizada_ga": {
            "pesos": {t: float(w) for t, w in zip(tickers, w_opt)},
            "mu_p": ret_opt,
            "sigma_p_2": var_opt,
            "sigma_p": vol_opt,
            "J_w": j_opt,
            "D_port": div_ga
        }
    }
    
    with open(FILE_JSON, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False)
    print(f"[5/5] Resultados numéricos guardados en '{FILE_JSON}'.")

def main():
    prices = download_data(TICKERS, period=PERIOD)
    mu, sigma, daily_returns = calculate_financial_metrics(prices)
    w_1N = np.ones(NUM_ASSETS) / NUM_ASSETS
    
    pop_rand = np.array([generate_random_feasible_portfolio(NUM_ASSETS) for _ in range(POP_SIZE)])
    w_rand = pop_rand[0]
    
    w_opt, final_pop_ga, best_history, div_ga = run_evolutionary_algorithm(mu, sigma)
    
    plot_risk_return_frontier(mu, sigma, w_1N, w_rand, w_opt, final_pop_ga)
    plot_weight_allocation(TICKERS, w_1N, w_opt)
    
    print_markdown_results(TICKERS, mu, sigma, w_1N, w_rand, w_opt, pop_rand, final_pop_ga)

if __name__ == "__main__":
    main()
