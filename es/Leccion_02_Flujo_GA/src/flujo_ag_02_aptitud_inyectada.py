"""
Lección 02 - Paso 2: El problema se convierte en un argumento
====================================================
NUEVO EN ESTE PASO: la función de aptitud es inyectada en lugar de estar soldada.

En el paso 1, la palabra `objective` aparece dentro de Individuo. Esa única referencia
suelda el flujo a un problema: para resolver un segundo problema copiarías el archivo.
El libro separa `fitness.py` de `individual.py` exactamente por esta razón.

Así que la función sale y se convierte en un argumento. Cada individuo lleva la
función que lo juzga, los operadores la pasan, y el controlador toma el
problema que se le pide resolver. Luego, al mismo controlador se le entregan dos
problemas diferentes - una maximización multimodal, una minimización escrita como
una maximización - y el cuerpo del ciclo no se toca entre ellos.

CAMBIOS RESPECTO A flujo_ag_01_las_cinco_fases.py
Introdúcelos en este orden:
    1. los dos problemas        sine_landscape() y closeness_to_target(), lado a lado
    2. Individuo(genes, f)      un individuo lleva la función que lo juzga
    3. los operadores de const. create_random() toma f; cruza() y mutar() la heredan
    4. run(fitness_function)    el controlador toma el problema; su cuerpo nunca nombra uno
    5. las dos ejecuciones      mismo controlador, ambos problemas, nada intermedio cambió

Ejecútalo:  python flujo_ag_02_aptitud_inyectada.py

Un controlador, dos problemas: el seno reproduce el paso 1; el objetivo
aterriza a 0.0038 de +4.200. La única diferencia es qué función se pasó.
"""
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 52
POPULATION_SIZE = 10
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.1
MAX_GENERATIONS = 10
GENE_MIN, GENE_MAX = -10.0, 10.0
TOURNAMENT_SIZE, BLEND_ALPHA = 3, 1.0
MUTATION_MU, MUTATION_SIGMA = 0.0, 1.0
TARGET = 4.2
FIGURES = Path(__file__).resolve().parent.parent / "figures"

Fitness = Callable[[float], float]


# --- NUEVO (1) los dos problemas ----------------------------------------------
def sine_landscape(x: float) -> float:
    """El objetivo de la Lección 01, ahora solo un problema entre otros.

    Args:
        x: un real en [-10, 10].

    Returns:
        sen(x) - 0.2 * |x|. Máximo global cerca de x = +1.372, f = +0.706.

    Example:
        Esta corrida reproduce el paso 1: x = +1.372, f = +0.706.
    """
    return np.sin(x) - 0.2 * abs(x)


def closeness_to_target(x: float) -> float:
    """MINIMIZAR la distancia a TARGET - escrita como una maximización.

    Un algoritmo genético siempre maximiza; no hay variante minimizadora. A
    un problema que naturalmente es una minimización se le inyecta con su signo
    invertido, y ese es todo el truco. La mejor aptitud alcanzable aquí es 0.

    Args:
        x: un real en [-10, 10].

    Returns:
        -(x - 4.2)^2. Cero solo en x = 4.2.

    Example:
        Esta corrida aterriza a 0.0038 de +4.200.
    """
    return -(x - TARGET) ** 2
# ------------------------------------------------------------------------------


def acotar(g: float, low: float = GENE_MIN, high: float = GENE_MAX) -> float:
    """Proyecta un gen al intervalo cerrado del paisaje.

    Args:
        g: gen crudo.
        low, high: extremos cerrados.

    Returns:
        g proyectado a [low, high].
    """
    return max(low, min(high, g))


class Individuo:
    # --- NUEVO (2) Individuo(genes, f) ----------------------------------------
    """Una solución candidata: un cromosoma, su juez y el veredicto.

    Mantener `fitness_function` en el individuo significa que los operadores no
    necesitan que se les pase: un hijo es juzgado por lo que haya juzgado a su
    padre. Soldar la función aquí, como hizo el paso 1, es lo que hizo que todo
    el archivo fuera de un solo propósito.

    Args:
        gene_list: cromosoma; aquí un solo real.
        fitness_function: juez inyectado; no está soldado al archivo.
    """

    def __init__(self, gene_list: List[float], fitness_function: Fitness) -> None:
        self.gene_list = gene_list
        self.fitness_function = fitness_function
        self.aptitud = fitness_function(gene_list[0])
    # --------------------------------------------------------------------------

    @property
    def gen(self) -> float:
        """El único gen: este paisaje es unidimensional."""
        return self.gene_list[0]

    def __repr__(self) -> str:
        return f"x={self.gen:+.3f} f={self.aptitud:+.3f}"


def seleccion_torneo(population: List[Individuo], size: int) -> List[Individuo]:
    """Sin cambios, y vale la pena decir por qué: la selección compara valores
    de aptitud que ya existen. Nunca necesita saber de dónde vinieron.

    Args:
        population: lista de Individuo.
        size: competidores por torneo.

    Returns:
        Lista de la misma longitud, hecha de copias.
    """
    return [max([random.choice(population) for _ in range(size)],
                key=lambda i: i.aptitud) for _ in range(len(population))]


# --- NUEVO (3) los operadores de construcción ---------------------------------
def create_random(fitness_function: Fitness) -> Individuo:
    """Extrae un individuo y le pega el juez que usará toda su descendencia.

    Args:
        fitness_function: problema a resolver.

    Returns:
        Individuo uniforme en [-10, 10], juzgado por `fitness_function`.
    """
    return Individuo([random.uniform(GENE_MIN, GENE_MAX)], fitness_function)


def cruza(p1: Individuo, p2: Individuo) -> Tuple[Individuo, Individuo]:
    """Cruza de mezcla; el hijo hereda el juez del primer padre.

    Args:
        p1, p2: progenitores (mismo juez, por construcción del controlador).

    Returns:
        Dos Individuo nuevos, acotados, con `p1.fitness_function`.
    """
    shift = (1 + 2 * BLEND_ALPHA) * random.random() - BLEND_ALPHA
    g1 = acotar((1 - shift) * p1.gen + shift * p2.gen)
    g2 = acotar(shift * p1.gen + (1 - shift) * p2.gen)
    return (Individuo([g1], p1.fitness_function),
            Individuo([g2], p1.fitness_function))


def mutar(ind: Individuo) -> Individuo:
    """Suma ruido gaussiano y conserva el juez del padre.

    Args:
        ind: candidato a perturbar.

    Returns:
        Un Individuo nuevo con `ind.fitness_function`.
    """
    return Individuo([acotar(ind.gen + random.gauss(MUTATION_MU, MUTATION_SIGMA))],
                     ind.fitness_function)
# ------------------------------------------------------------------------------


def evolve_one_generation(
        population: List[Individuo]) -> Tuple[List[Individuo], Dict[str, int]]:
    """SELECCIONAR -> CRUZAR -> MUTAR -> reemplazar, una vez.

    Args:
        population: generación actual.

    Returns:
        (nueva población, censo de pares/cruzados/mutados/creados).
    """
    census = {"pairs": 0, "crossed": 0, "mutated": 0, "created": 0}

    selected = seleccion_torneo(population, TOURNAMENT_SIZE)             # SELECCIONAR

    crossed: List[Individuo] = []                                        # CRUZAR
    # Pares consecutivos: el torneo ya barajó el orden al copiar.
    for p1, p2 in zip(selected[::2], selected[1::2]):
        census["pairs"] += 1
        if random.random() < CROSSOVER_PROBABILITY:
            crossed.extend(cruza(p1, p2))
            census["crossed"] += 1
            census["created"] += 2
        else:
            crossed.extend([p1, p2])

    mutated: List[Individuo] = []                                        # MUTAR
    for ind in crossed:
        if random.random() < MUTATION_PROBABILITY:
            mutated.append(mutar(ind))
            census["mutated"] += 1
            census["created"] += 1
        else:
            mutated.append(ind)

    return mutated, census                                             # reemplazar


# --- NUEVO (4) run(fitness_function) ------------------------------------------
def run(fitness_function: Fitness) -> Tuple[List[List[Individuo]],
                                            List[Dict[str, int]]]:
    """INICIALIZAR, luego evolucionar hasta que la condición de parada diga detenerse.

    Lee el cuerpo y nota lo que falta: sin seno, sin objetivo, sin mención de
    qué se está optimizando. Esa ausencia es el punto del paso.

    Args:
        fitness_function: el problema; el único argumento que cambia entre corridas.

    Returns:
        (lista de generaciones incluyendo la 0, lista de censos).

    Example:
        seno -> x = +1.372; cercanía -> 0.0038 de +4.200.
    """
    random.seed(SEMILLA)
    population = [create_random(fitness_function) for _ in range(POPULATION_SIZE)]

    generations = [population]
    censuses: List[Dict[str, int]] = []
    for _ in range(MAX_GENERATIONS):
        population, census = evolve_one_generation(population)
        generations.append(population)
        censuses.append(census)
    return generations, censuses
# ------------------------------------------------------------------------------


# --- NUEVO (5) las dos ejecuciones --------------------------------------------
PROBLEMAS = [("paisaje de seno", sine_landscape),
            ("cercania al objetivo", closeness_to_target)]

results = []
for name, fitness_function in PROBLEMAS:
    generations, censuses = run(fitness_function)
    best = max(generations[-1], key=lambda i: i.aptitud)
    results.append((name, generations, best))
    print(f"{name:>20s}: {MAX_GENERATIONS} generaciones -> {best}")

sine_best = results[0][2]
target_best = results[1][2]
gap = abs(target_best.gen - TARGET)
print(f"\nLa primera ejecución encontró el pico del seno en x={sine_best.gen:+.3f}; la segunda")
print(f"aterrizó a {gap:.4f} de distancia del objetivo {TARGET:+.3f}. Misma semilla, mismo")
print("tamaño de población, mismos operadores, mismas diez generaciones - y la única")
print("diferencia entre las dos ejecuciones es qué función se pasó.")
print("\nLa ejecución del seno también reproduce exactamente el paso 1: sacar la función")
print("cambió el código, no el algoritmo. Y nota lo que los dos paisajes")
print("son: el segundo tiene un pico, así que cada comparación que hace la selección apunta")
print("hacia el mismo lado; el primero tiene cuatro. El flujo no puede distinguirlos.")
# ------------------------------------------------------------------------------

grid = np.linspace(GENE_MIN, GENE_MAX, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, generations, best) in zip(axes, results):
    _, fitness_function = next(p for p in PROBLEMAS if p[0] == name)
    pop = generations[-1]
    ax.plot(grid, [fitness_function(v) for v in grid], "--",
            color="tab:blue", alpha=0.6)
    ax.plot([i.gen for i in pop], [i.aptitud for i in pop],
            "o", color="tab:orange", label="población final")
    ax.plot([best.gen], [best.aptitud], "s", color="tab:green",
            markersize=9, label="mejor")
    ax.set_title(f"{name}\nmejor x={best.gen:+.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("aptitud")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower center")
fig.suptitle("Un controlador, dos problemas, sin cambios en el ciclo")
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "flujo_ag_02_aptitud_inyectada.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURES}/flujo_ag_02_aptitud_inyectada.png")

