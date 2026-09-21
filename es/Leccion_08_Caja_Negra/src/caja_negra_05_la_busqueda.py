"""
Lección 08 - Paso 5: La búsqueda, y una respuesta que no se queda quieta
========================================================================
NUEVO EN ESTE PASO: select_rank_with_elite(), el ciclo generacional, y los
diagnósticos.

Cada pieza está finalmente sobre la mesa: un cromosoma que no puede ser ilegal
(paso 3), operadores afinados gen por gen (paso 4). Este paso ensambla el
algoritmo genético del libro y lo deja ejecutar. La selección es la selección
por rango de la Lección 03 con una élite de 3 - la elección del libro - y el ciclo es
el de la Lección 02: seleccionar, cruzar pares con probabilidad 0.8, mutar con probabilidad 0.2.
El libro ejecuta este ciclo por 1,000 generaciones; 100 son suficientes para demostrar el
punto, así que 100 serán.

Y el punto es incómodo. La ejecución reporta una mejor aptitud de 2.278e-09
en x = -10.9536 - un cromosoma estacionado muy fuera del intervalo [-10, 10]
del cual se extrajo la población inicial, en una dirección que la cuadrícula del paso 2 nunca
alcanzó. Gastó 20,204 evaluaciones para llegar allí: 0.02% del presupuesto de la cuadrícula del
libro del paso 2. Por cada medida que el curso ha usado hasta ahora, esto es un
éxito. Y sin embargo la curva del mejor de la historia sigue subiendo cuando el presupuesto se
agota, y el 97.5% de la población final vive fuera del intervalo en el que los comenzamos.
El algoritmo es unánime acerca de DÓNDE ir. El paso 6 pregunta si
está de acuerdo consigo mismo acerca de QUÉ encontró.

DESAPARECE EN ESTE PASO: el censo de reparación y la demo de deriva (el punto del paso 4
está claro), y la muestra de un solo disparo de los operadores.

CAMBIOS RESPECTO A caja_negra_04_los_operadores.py
Introdúcelos en este orden:
    1. select_rank_with_elite()   método de la Lección 03, élite de 3: elección del libro
    2. el ciclo generacional      100 generaciones (el libro ejecutó 1,000)
    3. los diagnósticos           mejor vs promedio, el censo de x, el censo de evaluaciones

Ejecútalo:  python caja_negra_05_la_busqueda.py

La mejor aptitud alcanza 2.278e-09, pero la curva sigue subiendo y el 97.5% de la población final ha salido del intervalo inicial de x. El algoritmo es unánime sobre DÓNDE ir, no sobre QUÉ encontró.
"""
import math
import random
from pathlib import Path
from typing import Any, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 19
POPULATION_SIZE = 200
GENERATIONS = 100                 # el libro usó 1,000; 100 prueban el punto
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.2
ELITE_SIZE = 3
A_MIN, A_MAX = 0.0, 1.0
B_MIN, B_MAX = 0.0, 1.0
X_MIN, X_MAX = -100.0, 100.0
N_SET = range(0, 21)
FUN_SET = ("sin", "cos")
INIT_X_MIN, INIT_X_MAX = -10.0, 10.0
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def complicated_one(a: float, b: float, x: float, n: int, fun_name: str) -> float:
    """La caja negra, sin cambios desde el paso 1. Todavía nadie la ha leído.

    Args:
        a: real acotado a [0, 1].
        b: real acotado a [0, 1].
        x: real acotado a [-100, 100].
        n: entero legal 0..20.
        fun_name: etiqueta ``sin`` o ``cos``.

    Returns:
        Un flotante. ``Individuo.counter`` cuenta estas llamadas.

    Example:
        El campeón de SEMILLA = 19 reporta 2.278e-09 en x = -10.9536,
        fuera del [-10, 10] inicial, después de 20,204 evaluaciones.
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
        El campeón se estaciona en x = -10.9536, todavía dentro de [-100, 100].
        clamp no lo detuvo: el intervalo declarado es más ancho que el inicial.
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
        El campeón del paso 5 guarda n = 3, un punto de la malla. La
        reparación silenciosa del paso 4 sigue vigente.
    """
    return min(allowed, key=lambda candidate: abs(candidate - value))


class Individuo:
    """Un cromosoma de cinco genes que no puede existir en un estado ilegal.

    Toda la reparación vive en el constructor, así que los operadores debajo pueden tratar
    cada gen como un número real. La clase cuenta sus propias instanciaciones:
    desde este paso en adelante esa cuenta ES el presupuesto de evaluaciones de aptitud.

    Args:
        gene_list: cinco valores crudos (a, b, x, n, fun_name).

    Returns:
        Un Individuo legal. Incrementa ``counter``.

    Example:
        Tras 100 generaciones, ``Individuo.counter`` es 20,204: 0.02% del
        presupuesto de la cuadrícula del libro y la única línea de costo que hay.
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
        La población inicial cabe en [-10, 10]. Al terminar, el 97.5% vive
        fuera: la mutación caminó y nadie la detuvo.
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
        x con alfa = 1.0 es el gen que sale de [-10, 10]: el 97.5% de la
        población final ya no vive donde nació.
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
        Probabilidad 0.8 por par, 100 generaciones. La cruza de x (alfa 1.0)
        es la que permite abandonar el intervalo inicial.
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
        Un Individuo legal, a veces idéntico.

    Example:
        Sigma 1.0 en x es el paso que saca a la población de [-10, 10].
        El campeón termina en x = -10.9536.
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


# --- NUEVO (1) select_rank_with_elite() -----------------------------------------
def select_rank_with_elite(population: List[Individuo],
                           elite_size: int) -> List[Individuo]:
    """La selección por rango de la Lección 03 con el top `elite_size` copiado directamente.

    La selección del libro, sin cambios: ordenar el mejor primero, darle a la posición i el
    peso 1 - i/N, y girar la rueda N - elite_size veces. Los espacios de élite
    se llenan antes de cualquier sorteo, por lo que el mejor individuo no se puede perder - la
    garantía más barata del curso, y en este paisaje la que soporta la carga.

    Args:
        population: generación actual.
        elite_size: copias garantizadas del frente; el libro usa 3.

    Returns:
        Una nueva población del mismo tamaño, con la élite al frente.

    Example:
        Élites de 3 y 100 generaciones bastan para estacionar el campeón en
        x = -10.9536 con aptitud 2.278e-09. La curva sigue subiendo.
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
# ------------------------------------------------------------------------------


# --- NUEVO (2) el ciclo generacional --------------------------------------------
# El ciclo de la Lección 02 con las perillas del libro: seleccionar una nueva población, cruzar los
# pares con probabilidad 0.8, mutar a los sobrevivientes con probabilidad 0.2.
random.seed(SEMILLA)
Individuo.counter = 0
population = [create_random() for _ in range(POPULATION_SIZE)]
best_ever = max(population, key=lambda ind: ind.aptitud)
ever_history: List[float] = []
average_history: List[float] = []
for generation in range(GENERATIONS):
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
    ever_history.append(best_ever.aptitud)
    average_history.append(sum(ind.aptitud for ind in population)
                           / len(population))
# ------------------------------------------------------------------------------

print("=" * 78)
print("Lección 08 - Paso 5: la búsqueda, y una respuesta que no se queda quieta")
print("=" * 78)
print(f"\n{GENERATIONS} generaciones, población {POPULATION_SIZE}, "
      f"pc = {CROSSOVER_PROB}, pm = {MUTATION_PROB}, élite {ELITE_SIZE}")
print("-" * 78)
print(f"  mejor de la historia: {best_ever}  aptitud {best_ever.aptitud:.3e}")

# --- NUEVO (3) los diagnósticos --------------------------------------------------
# Tres lecturas de la misma ejecución. Ninguna de ellas es el veredicto por sí sola;
# juntas son la pregunta que el paso 6 tiene que responder.
print("\nLos diagnósticos")
print("-" * 78)

# (a) mejor vs promedio: ¿qué tanto sobresale el campeón por encima de su propia generación?
final_average = average_history[-1]
print(f"  mejor de la historia:             {best_ever.aptitud:+.3e}")
print(f"  media de población final:         {final_average:+.3e}")
print(f"  curva mejor historia en generación 1:    {ever_history[0]:+.3e}")
print(f"  curva mejor historia en generación {GENERATIONS}: {ever_history[-1]:+.3e}"
      "  (aún subiendo)")

# (b) el censo de x: ¿dónde vive realmente la población final?
final_xs = [ind.gene_list[2] for ind in population]
outside = sum(1 for value in final_xs if not INIT_X_MIN <= value <= INIT_X_MAX)
print(f"\n  rango de x de población final: [{min(final_xs):+.2f}, {max(final_xs):+.2f}]")
print(f"  fuera del intervalo inicial [{INIT_X_MIN:+.0f}, {INIT_X_MAX:+.0f}]: "
      f"{outside}/{len(final_xs)} ({outside / len(final_xs):.1%})")

# (c) el censo de evaluaciones: ¿qué costó la respuesta?
print(f"\n  evaluaciones de aptitud usadas: {Individuo.counter:,}")
print(f"  presupuesto de cuadrícula del libro (paso 2): 105,000,000 "
      f"({Individuo.counter / 105_000_000:.2%} de él gastado)")
# ------------------------------------------------------------------------------

fig, (ax_curve, ax_census) = plt.subplots(1, 2, figsize=(11.5, 4.0))
ax_curve.plot(range(1, GENERATIONS + 1), ever_history, color="tab:blue",
              linewidth=1.6, label="mejor de la historia")
ax_curve.set_yscale("log")
ax_curve.set_title("La curva de éxito: mejor aptitud hasta ahora", fontsize=10)
ax_curve.set_xlabel("generación")
ax_curve.set_ylabel("aptitud (escala log)")
ax_curve.grid(True, linestyle=":", alpha=0.5)
ax_curve.legend(fontsize=8)

ax_census.hist(final_xs, bins=40, color="tab:blue", edgecolor="white")
ax_census.axvspan(INIT_X_MIN, INIT_X_MAX, color="tab:green", alpha=0.15,
                  label="rango de x inicial [-10, 10]")
ax_census.set_title("Dónde vive la población final", fontsize=10)
ax_census.set_xlabel("gen x")
ax_census.set_ylabel("cuenta")
ax_census.legend(fontsize=8)
ax_census.grid(True, axis="y", linestyle=":", alpha=0.5)

fig.suptitle("Un éxito de libro de texto - que ha abandonado el mapa que le dimos")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "caja_negra_05_la_busqueda.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURES}/caja_negra_05_la_busqueda.png")

