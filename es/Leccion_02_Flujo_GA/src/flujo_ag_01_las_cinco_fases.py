"""
Lección 02 - Paso 1: El flujo, nombrado
====================================
NUEVO EN ESTE PASO: evolve_one_generation(), run(), y un rastro por fase.

La Lección 01 terminó con un algoritmo genético que funcionaba. Su cuerpo de ciclo, sin embargo,
era una serie de bloques de comentarios sin nombre, y no puedes discutir "el flujo" cuando el
flujo no tiene nombre y no tiene límite. Este paso lo nombra.

Nada a continuación cambia el algoritmo. Una generación de evolución se convierte en una
función, la condición de parada se mueve a un lugar, y el controlador imprime lo que
cada fase le entregó a la siguiente. La prueba de que es el mismo algoritmo está en la
salida: misma semilla, misma respuesta que el paso 6 de la Lección 01, al dígito.

CAMBIOS RESPECTO A primer_ejemplo_06_el_ciclo_completo.py de la Lección 01
Introdúcelos en este orden:
    1. evolve_one_generation()  una generación como una función; también cuenta lo que hizo
    2. run()                    el controlador: INICIALIZAR, luego evolucionar hasta la condición de parada
    3. el rastro de fases       imprimir el censo cada generación, y comprobar contra la Lección 01

Ejecútalo:  python flujo_ag_01_las_cinco_fases.py

Misma semilla, misma respuesta que el paso 6 de la Lección 01: x = +1.372,
f = +0.706. La corrida costó 107 evaluaciones, no las 110 obvias.
"""
import random
from pathlib import Path
from typing import Dict, List, Tuple

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
FIGURES = Path(__file__).resolve().parent.parent / "figures"

# El paso 6 de la Lección 01 imprimió esto, con esta semilla. Es el valor de referencia
# que esta refactorización tiene que reproducir; no se usa para nada más.
LESSON_01_STEP_6_GENE = +1.372


def objective(x: float) -> float:
    """La función que queremos MAXIMIZAR.

    Args:
        x: un real en [-10, 10].

    Returns:
        sen(x) - 0.2 * |x|. Máximo global cerca de x = +1.372, f = +0.706.

    Example:
        Con SEMILLA = 52 este controlador reproduce x = +1.372, f = +0.706.
    """
    return np.sin(x) - 0.2 * abs(x)


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
    """Una solución candidata: un cromosoma más la aptitud que produce.

    La aptitud se evalúa aquí, en el constructor, y en ningún otro lugar. Eso
    vale la pena notarlo: EVALUAR es una fase del flujo sin código propio,
    y la única forma de contar las evaluaciones es contar los individuos creados.

    Args:
        gene_list: cromosoma; aquí un solo real.

    Example:
        Diez individuos iniciales = diez evaluaciones, no once.
    """

    def __init__(self, gene_list: List[float]) -> None:
        self.gene_list = gene_list
        self.aptitud = objective(gene_list[0])

    @property
    def gen(self) -> float:
        """El único gen: este paisaje es unidimensional."""
        return self.gene_list[0]

    def __repr__(self) -> str:
        return f"x={self.gen:+.3f} f={self.aptitud:+.3f}"


def create_random() -> Individuo:
    """Extrae un individuo uniforme de [-10, 10].

    Returns:
        Individuo con un gen en [GENE_MIN, GENE_MAX].
    """
    return Individuo([random.uniform(GENE_MIN, GENE_MAX)])


def seleccion_torneo(population: List[Individuo], size: int) -> List[Individuo]:
    """Un torneo por cada lugar de la nueva generación.

    Args:
        population: lista de Individuo.
        size: competidores por torneo (aquí 3).

    Returns:
        Lista de la misma longitud, hecha de copias (las mismas instancias).
    """
    return [max([random.choice(population) for _ in range(size)],
                key=lambda i: i.aptitud) for _ in range(len(population))]


def cruza(p1: Individuo, p2: Individuo) -> Tuple[Individuo, Individuo]:
    """Cruza de mezcla (BLX-alfa) entre dos Individuo.

    Args:
        p1, p2: progenitores.

    Returns:
        Dos Individuo nuevos, acotados a [-10, 10]. Cada uno cuenta como
        una evaluación porque el constructor llama a objective().
    """
    shift = (1 + 2 * BLEND_ALPHA) * random.random() - BLEND_ALPHA
    g1 = acotar((1 - shift) * p1.gen + shift * p2.gen)
    g2 = acotar(shift * p1.gen + (1 - shift) * p2.gen)
    return Individuo([g1]), Individuo([g2])


def mutar(ind: Individuo) -> Individuo:
    """Suma ruido gaussiano a un gen y reconstruye el Individuo.

    Args:
        ind: candidato a perturbar.

    Returns:
        Un Individuo nuevo (una evaluación más).
    """
    return Individuo([acotar(ind.gen + random.gauss(MUTATION_MU, MUTATION_SIGMA))])


# --- NUEVO (1) evolve_one_generation() ------------------------------------------
def evolve_one_generation(
        population: List[Individuo]) -> Tuple[List[Individuo], Dict[str, int]]:
    """SELECCIONAR -> CRUZAR -> MUTAR -> reemplazar, una vez.

    Devolver el censo junto con la nueva población cuesta tres contadores y
    compra toda la lección: de aquí en adelante el flujo puede ser observado en lugar de
    creído. Nada aquí decide cuándo detenerse - ese es trabajo del controlador.

    Args:
        population: generación actual.

    Returns:
        (nueva población, censo con pares/cruzados/mutados/creados).

    Example:
        Esta corrida crea 107 individuos, no 110: un seleccionado que no
        se cruza ni muta es el mismo objeto y no se reevalúa.
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
# ------------------------------------------------------------------------------


# --- NUEVO (2) run() ------------------------------------------------------------
def run() -> Tuple[List[List[Individuo]], List[Dict[str, int]]]:
    """INICIALIZAR, luego evolucionar hasta que la condición de parada diga detenerse.

    La condición de parada vive aquí y solo aquí. En este momento es la
    más cruda que el libro lista - un número fijo de generaciones - y ponerla
    en un lugar visible es lo que la hace reemplazable más adelante.

    Returns:
        (lista de generaciones incluyendo la 0, lista de censos).

    Example:
        Diez generaciones, SEMILLA = 52: x = +1.372, f = +0.706.
    """
    random.seed(SEMILLA)
    population = [create_random() for _ in range(POPULATION_SIZE)]

    generations = [population]
    censuses: List[Dict[str, int]] = []
    for _ in range(MAX_GENERATIONS):
        population, census = evolve_one_generation(population)
        generations.append(population)
        censuses.append(census)
    return generations, censuses
# ------------------------------------------------------------------------------


generations, censuses = run()

# --- NUEVO (3) el rastro de fases ---------------------------------------------
print("El flujo, una línea por generación. SELECCIONAR nunca crea nada: copia")
print("referencias, por lo que solo CRUZAR y MUTAR añaden nuevos individuos,")
print("y cada nuevo individuo es exactamente una evaluación de aptitud.\n")
print(" gen | pares cruzados | mutados | individuos nuevos | mejor aptitud")
print("-----+----------------+---------+-------------------+--------------")
best0 = max(generations[0], key=lambda i: i.aptitud)
print(f"   0 |              - |       - |"
      f" {POPULATION_SIZE:17d} | {best0.aptitud:+.4f}")
for g, census in enumerate(censuses, start=1):
    best = max(generations[g], key=lambda i: i.aptitud)
    print(f" {g:3d} |"
          f" {census['crossed']:9d} / {census['pairs']:2d} |"
          f" {census['mutated']:4d}/{POPULATION_SIZE:2d} |"
          f" {census['created']:17d} | {best.aptitud:+.4f}")

evaluations = POPULATION_SIZE + sum(c["created"] for c in censuses)
naive = POPULATION_SIZE * (MAX_GENERATIONS + 1)
print(f"\nEsta ejecución costó {evaluations} evaluaciones de aptitud: una por individuo jamás")
print(f"creado. La suposición obvia es {naive} = {POPULATION_SIZE} x "
      f"{MAX_GENERATIONS + 1}, y es incorrecta, porque un")
print("individuo que es seleccionado y luego ni cruzado ni mutado es el")
print("mismo objeto y nunca es reevaluado. El verdadero costo de una ejecución son los datos,")
print("no la aritmética - por lo cual el censo de arriba tiene que ser contado.")

best = max(generations[-1], key=lambda i: i.aptitud)
print(f"\nSolución: {best}")
unchanged = abs(best.gen - LESSON_01_STEP_6_GENE) < 5e-4
print(f"Lección 01 paso 6, misma semilla, reportó x={LESSON_01_STEP_6_GENE:+.3f}.")
print("Misma semilla, misma respuesta: nombrar el flujo no cambió nada." if unchanged
      else "La respuesta se movió: esta refactorización NO conservó el comportamiento.")
# ------------------------------------------------------------------------------

grid = np.linspace(GENE_MIN, GENE_MAX, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (label, pop) in zip(axes, [("generación 0", generations[0]),
                                   (f"generación {MAX_GENERATIONS}", generations[-1])]):
    ax.plot(grid, objective(grid), "--", color="tab:blue", alpha=0.6)
    ax.plot([i.gen for i in pop], [i.aptitud for i in pop],
            "o", color="tab:orange", label="población")
    b = max(pop, key=lambda i: i.aptitud)
    ax.plot([b.gen], [b.aptitud], "s", color="tab:green", markersize=9, label="mejor")
    ax.set_title(label)
    ax.set_xlabel("x")
    ax.grid(True, linestyle=":", alpha=0.5)
axes[0].set_ylabel("f(x)")
axes[0].legend()
fig.suptitle("El flujo lleva a la población de dispersa a establecida")
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "flujo_ag_01_las_cinco_fases.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURES}/flujo_ag_01_las_cinco_fases.png")

