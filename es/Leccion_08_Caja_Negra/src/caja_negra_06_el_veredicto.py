"""
Lección 08 - Paso 6: El veredicto, doce veces más
=================================================
NUEVO EN ESTE PASO: el Monte Carlo y las dos lecturas.

La ejecución del paso 5 pareció un éxito, y una ejecución es una anécdota. Este paso
repite la búsqueda completa doce veces, desde doce semillas, y pone a los
doce campeones lado a lado. Nada sobre el algoritmo cambia: mismo
cromosoma, mismos operadores, mismas perillas, mismas 100 generaciones. Solo la semilla
se mueve.

El veredicto se divide en dos. DÓNDE: cada uno de los doce campeones se asienta en
x = -10.95 - doce búsquedas independientes, todas iniciadas dentro de [-10, 10], todas
caminan al mismo lugar fuera de él y se estacionan dentro de 0.008 el uno del otro. QUÉ:
las aptitudes que reportan van de 5.8e-09 a 1.2e-04, un factor de 21,000
entre la respuesta más débil y la más fuerte a la misma pregunta. Un
algoritmo que está de acuerdo consigo mismo sobre a dónde ir, y difiere por cuatro
órdenes de magnitud sobre lo que encontró allí, no está convergiendo en un valor.
Está escalando algo sin un techo visible. El paso 7 abre la caja.

DESAPARECE EN ESTE PASO: los diagnósticos de una sola ejecución (una ejecución ya no es la
unidad de evidencia) y la figura de la curva de éxito.

CAMBIOS RESPECTO A caja_negra_05_la_busqueda.py
Introdúcelos en este orden:
    1. run_search()        la búsqueda del paso 5 como función: entra semilla, sale campeón
    2. el Monte Carlo      doce ejecuciones, doce semillas, doce campeones
    3. las dos lecturas    dónde se sientan los campeones vs qué reportan

Ejecútalo:  python caja_negra_06_el_veredicto.py

Los doce campeones se sientan en un intervalo estrecho de x y su aptitud difiere en 21,504×. Eso no es convergencia a un máximo finito: es una búsqueda escalando algo sin techo.
"""
import math
import random
from pathlib import Path
from typing import Any, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

POPULATION_SIZE = 200
GENERATIONS = 100
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.2
ELITE_SIZE = 3
RUNS = 12
A_MIN, A_MAX = 0.0, 1.0
B_MIN, B_MAX = 0.0, 1.0
X_MIN, X_MAX = -100.0, 100.0
N_SET = range(0, 21)
FUN_SET = ("sin", "cos")
INIT_X_MIN, INIT_X_MAX = -10.0, 10.0
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def complicated_one(a: float, b: float, x: float, n: int, fun_name: str) -> float:
    """La caja negra, sin cambios desde el paso 1. A un paso de ser leída.

    Args:
        a: real acotado a [0, 1].
        b: real acotado a [0, 1].
        x: real acotado a [-100, 100].
        n: entero legal 0..20.
        fun_name: etiqueta ``sin`` o ``cos``.

    Returns:
        Un flotante. Doce búsquedas independientes la llaman 12 × ~20,000 veces.

    Example:
        Las aptitudes de los doce campeones van de ~5.8e-09 a ~1.2e-04,
        un factor 21,504×, todos cerca de x = -10.95.
    """
    total = 0.0
    for _ in range(10, 10 + n + 1):
        if fun_name == "cos":
            trig = math.cos(math.log2(b ** n + 1) * math.pi * x) / (n + 1) + math.sin(x) ** n
        elif fun_name == "sin":
            trig = math.sin(math.log2(b ** n + 1) * math.pi * x) / (n + 1) + math.cos(x) ** n
        else:
            raise ValueError(f"Función desconocida: {fun_name}")
        resid = trig - math.log2(n + 1)
        div = ((n + 1) ** 2) * (1 + a + b) * (120 - x ** 2) * resid + 1 / 2
        total += ((x * n + math.log(n + 1)) / div) / (10 ** 15)
    return total


def clamp(gene: float, low: float, high: float) -> float:
    """Empujar un gen real de vuelta dentro de su intervalo.

    Args:
        gene: valor propuesto.
        low: borde inferior legal.
        high: borde superior legal.

    Returns:
        El gen recortado a [low, high].

    Example:
        Los doce campeones caben en x ≈ -10.95, lejos del borde ±100.
        clamp no es lo que los detuvo ahí.
    """
    return max(low, min(high, gene))


def closest(value: float, allowed: Sequence[int]) -> int:
    """Ajustar un número real al punto legal más cercano de una malla.

    Args:
        value: real propuesto.
        allowed: puntos legales 0..20.

    Returns:
        El entero legal más cercano.

    Example:
        Los campeones coinciden en n tanto como en x: la malla no explica
        el factor 21,504× de aptitud.
    """
    return min(allowed, key=lambda candidate: abs(candidate - value))


class Individuo:
    """Un cromosoma de cinco genes que no puede existir en un estado ilegal.

    La clase cuenta sus propias instanciaciones: esa cuenta ES el
    presupuesto de evaluaciones de aptitud de la ejecución en curso.

    Args:
        gene_list: cinco valores crudos (a, b, x, n, fun_name).

    Returns:
        Un Individuo legal. ``counter`` se reinicia en cada ``run_search``.

    Example:
        Doce campeones, todos cerca de x = -10.95, aptitudes separadas por
        21,504×. Una ejecución ya no es la unidad de evidencia.
    """

    counter = 0

    def __init__(self, gene_list: Sequence[Any]) -> None:
        self.__class__.counter += 1
        a_raw, b_raw, x_raw, n_raw, fun_name = gene_list
        self.gene_list: List[Any] = [
            clamp(float(a_raw), A_MIN, A_MAX),
            clamp(float(b_raw), B_MIN, B_MAX),
            clamp(float(x_raw), X_MIN, X_MAX),
            closest(float(n_raw), N_SET),
            fun_name,
        ]
        self.aptitud: float = complicated_one(*self.gene_list)

    def __str__(self) -> str:
        a, b, x, n, fun_name = self.gene_list
        return f"[a={a:.4f} b={b:.4f} x={x:+.4f} n={n:2d} fun={fun_name}]"


def create_random() -> Individuo:
    """Un individuo aleatorio, cada gen extraído de la forma que su propio tipo permite.

    Returns:
        Un Individuo legal con x en [-10, 10].

    Example:
        Cada semilla 0..11 arranca dentro de [-10, 10] y camina al mismo
        vecindario x = -10.95. El acuerdo es de lugar, no de valor.
    """
    return Individuo([
        random.uniform(A_MIN, A_MAX),
        random.uniform(B_MIN, B_MAX),
        random.uniform(INIT_X_MIN, INIT_X_MAX),
        random.choice(N_SET),
        random.choice(FUN_SET),
    ])


def blend(g1: float, g2: float, alpha: float) -> tuple[float, float]:
    """La cruza de mezcla de la Lección 04 para un par de valores brutos de gen.

    Args:
        g1: gen del primer padre.
        g2: gen del segundo padre.
        alpha: 0.3 para a, 0.5 para b y n, 1.0 para x.

    Returns:
        Dos valores hijos, aún sin reparar.

    Example:
        Mismos alfas en las doce semillas. El factor 21,504× no viene de
        cambiar el operador; viene de estacionarse a distancias distintas.
    """
    low = min(g1, g2) - alpha * abs(g2 - g1)
    high = max(g1, g2) + alpha * abs(g2 - g1)
    return (low + random.random() * (high - low),
            low + random.random() * (high - low))


def cruza(parent1: Individuo, parent2: Individuo
              ) -> tuple[Individuo, Individuo]:
    """La cruza por gen del libro: una moneda por gen, un alfa por gen.

    Args:
        parent1: primer padre.
        parent2: segundo padre.

    Returns:
        Dos hijos legales.

    Example:
        Nada del operador cambia entre las doce ejecuciones. Solo la semilla
        se mueve, y aun así las aptitudes difieren 21,504×.
    """
    a1, b1, x1, n1, fun1 = parent1.gene_list
    a2, b2, x2, n2, fun2 = parent2.gene_list
    if random.random() < 0.5:
        a1, a2 = blend(a1, a2, 0.3)
    if random.random() < 0.5:
        b1, b2 = blend(b1, b2, 0.5)
    if random.random() < 0.5:
        x1, x2 = blend(x1, x2, 1.0)
    if random.random() < 0.5:
        n1, n2 = blend(n1, n2, 0.5)
    if random.random() < 0.5:
        fun1, fun2 = fun2, fun1
    return (Individuo([a1, b1, x1, n1, fun1]),
            Individuo([a2, b2, x2, n2, fun2]))


def mutar(individual: Individuo) -> Individuo:
    """La mutación por gen del libro: una moneda por gen, un sigma por gen.

    Args:
        individual: individuo legal de partida.

    Returns:
        Un Individuo legal.

    Example:
        Sigma 1.0 en x basta para que las doce búsquedas salgan de [-10, 10]
        y se estacionen dentro de 0.008 una de otra.
    """
    a, b, x, n, fun_name = individual.gene_list
    if random.random() < 0.5:
        a += random.gauss(0.0, 0.2)
    if random.random() < 0.5:
        b += random.gauss(0.0, 0.2)
    if random.random() < 0.5:
        x += random.gauss(0.0, 1.0)
    if random.random() < 0.5:
        n += random.gauss(0.0, 1.0)
    if random.random() < 0.5:
        fun_name = random.choice(FUN_SET)
    return Individuo([a, b, x, n, fun_name])


def select_rank_with_elite(population: List[Individuo],
                           elite_size: int) -> List[Individuo]:
    """La selección por rango de la Lección 03 con el top `elite_size` copiado directamente.

    Args:
        population: generación actual.
        elite_size: copias garantizadas del frente; el libro usa 3.

    Returns:
        Una nueva población del mismo tamaño.

    Example:
        Élites de 3 en las doce semillas. El acuerdo de lugar no es un
        accidente de una sola ruleta.
    """
    ordered = sorted(population, key=lambda ind: -ind.aptitud)
    size = len(population)
    weights = [1 - i / size for i in range(size)]
    total = sum(weights)
    selected: List[Individuo] = list(ordered[:elite_size])
    for _ in range(size - elite_size):
        point = random.random() * total
        cumulative = 0.0
        for ind, weight in zip(ordered, weights):
            cumulative += weight
            if cumulative > point:
                selected.append(ind)
                break
    return selected


# --- CAMBIADO --- el ciclo generacional del paso 5 se vuelve una función: una semilla
# entra, un campeón (y su factura de evaluación) sale. Nada más se mueve.
def run_search(seed: int) -> tuple[Individuo, int]:
    """Una búsqueda completa: 100 generaciones desde `semilla`, se devuelve el mejor de la historia.

    Args:
        seed: semilla de ``random``; la ejecución i usa la semilla i.

    Returns:
        ``(campeón, evaluaciones)``. Las evaluaciones salen de ``Individuo.counter``.

    Example:
        Semillas 0..11: todos los campeones en x ≈ -10.95, aptitudes
        separadas por 21,504×. Una semilla es una anécdota; doce son el
        veredicto.
    """
    random.seed(seed)
    Individuo.counter = 0
    population = [create_random() for _ in range(POPULATION_SIZE)]
    best_ever = max(population, key=lambda ind: ind.aptitud)
    for _ in range(GENERATIONS):
        selected = select_rank_with_elite(population, ELITE_SIZE)
        crossed: List[Individuo] = []
        for parent1, parent2 in zip(selected[::2], selected[1::2]):
            if random.random() < CROSSOVER_PROB:
                crossed.extend(cruza(parent1, parent2))
            else:
                crossed.extend([parent1, parent2])
        population = [mutar(ind) if random.random() < MUTATION_PROB else ind
                      for ind in crossed]
        generation_best = max(population, key=lambda ind: ind.aptitud)
        if generation_best.aptitud > best_ever.aptitud:
            best_ever = generation_best
    return best_ever, Individuo.counter


print("=" * 78)
print("Lección 08 - Paso 6: el veredicto, doce veces más")
print("=" * 78)

# --- NUEVO (2) el Monte Carlo --------------------------------------------------
# Doce búsquedas independientes, semillas 0..11. Mismo algoritmo, mismas perillas; solo
# la semilla se mueve. Cada línea abajo es una ejecución entera del paso 5 comprimida a su
# campeón.
champions: List[Individuo] = []
total_evaluations = 0
print(f"\n{RUNS} ejecuciones x {GENERATIONS} generaciones (población "
      f"{POPULATION_SIZE}, pc = {CROSSOVER_PROB}, pm = {MUTATION_PROB})")
print("-" * 78)
print(f"  {'semilla':>7}  {'campeón':<44}  {'aptitud':>10}  {'evals':>7}")
for seed in range(RUNS):
    champion, evaluations = run_search(seed)
    champions.append(champion)
    total_evaluations += evaluations
    print(f"  {seed:>7}  {champion!s:<44}  {champion.aptitud:>10.3e}  "
          f"{evaluations:>7,}")
print(f"  evaluaciones totales de aptitud: {total_evaluations:,}")
# ------------------------------------------------------------------------------

# --- NUEVO (3) las dos lecturas -------------------------------------------------
# Lectura uno, DÓNDE: los genes x de los doce campeones.
# Lectura dos, QUÉ: las aptitudes que reportan.
champion_xs = [ind.gene_list[2] for ind in champions]
champion_fits = [ind.aptitud for ind in champions]
x_spread = max(champion_xs) - min(champion_xs)
fit_ratio = max(champion_fits) / min(champion_fits)
print("\nLas dos lecturas")
print("-" * 78)
print(f"  DÓNDE se sientan los campeones:  x en [{min(champion_xs):+.4f}, "
      f"{max(champion_xs):+.4f}]  (dispersión {x_spread:.4f})")
print(f"  QUÉ reportan:         aptitud en [{min(champion_fits):.3e}, "
      f"{max(champion_fits):.3e}]  (proporción {fit_ratio:,.0f}x)")
print("\n  Unánimes acerca de dónde, separados por cuatro órdenes de magnitud acerca de qué.")
print("  Una búsqueda convergiendo en un máximo no hace eso. Una búsqueda escalando")
print("  algo no acotado hace exactamente eso - y el paso 7 abre la caja.")
# ------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.scatter(champion_xs, champion_fits, s=60, color="tab:blue", zorder=3)
for seed, ind in enumerate(champions):
    ax.annotate(f"{seed}", (ind.gene_list[2], ind.aptitud),
                textcoords="offset points", xytext=(5, 5), fontsize=8,
                color="tab:gray")
ax.set_yscale("log")
ax.set_xlabel("gen x del campeón")
ax.set_ylabel("aptitud del campeón (escala log)")
ax.set_title(f"Doce campeones: unánimes en dónde "
             f"(dispersión {x_spread:.3f}), separados por {fit_ratio:,.0f}x en qué",
             fontsize=10)
ax.grid(True, linestyle=":", alpha=0.5)
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "caja_negra_06_el_veredicto.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURES}/caja_negra_06_el_veredicto.png")

