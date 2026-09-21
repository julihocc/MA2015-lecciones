"""
Lección 08 - Paso 4: Los operadores, y lo que cuesta la reparación silenciosa
=============================================================================
NUEVO EN ESTE PASO: cruza(), mutar(), el censo de reparación, y la demo de
deriva.

El paso 3 puso toda la legalidad en el constructor, por lo que los operadores pueden escribirse
como si cada gen fuera un número real. Este paso los escribe - el diseño del libro,
y uno deliberado: cada gen tiene su propia moneda y su propio tamaño de paso,
porque los genes son diferentes tipos de objeto. Mezcla alfa 0.3 para
a, 0.5 para b, 1.0 para x; sigma gaussiano 0.2 para a y b, 1.0 para x y n;
el gen etiqueta muta volviéndose a elegir, la única mutación que admite un conjunto.

Luego la factura que prometió el paso 3. La reparación es silenciosa, y el silencio tiene un precio:
medir con qué frecuencia una mutación de n realmente mueve n. La respuesta es 58.5% -
al otro 41.5% de mutaciones de n se les devuelve la misma n por el ajuste, y
no se le dice a nadie. Y en el límite (el ejemplo de mutación imposible del libro):
haz la malla más gruesa que el paso del operador y la mutación se vuelve
ABSOLUTAMENTE silenciosa - mil mutaciones, y el gen discreto está exactamente
donde empezó, mientras su gemelo de valor real se aleja a la deriva. El operador no
falló ruidosamente. Simplemente nunca sucedió.

DESAPARECE EN ESTE PASO: repair_report() y la muestra de genes ilegales (el punto del paso 3
está claro), y el censo de población inicial (la figura del paso 3).

CAMBIOS RESPECTO A caja_negra_03_el_cromosoma.py
Introdúcelos en este orden:
    1. cruza()             las monedas por gen del libro: cada gen, su propio alfa
    2. mutar()             sigmas por gen; n mutado como un real, luego ajustado
    3. el censo de rep.    contar lo que el constructor deshizo silenciosamente
    4. la demo de deriva   una malla más gruesa que el paso: silencio absoluto

Ejecútalo:  python caja_negra_04_los_operadores.py

El 41.5% de las mutaciones enteras forzadas vuelven a su n original sin cambios. Si la malla es más gruesa que el paso, la mutación no es débil: está ausente.
"""
import math
import random
from pathlib import Path
from typing import Any, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 19
POPULATION_SIZE = 200
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
        Un flotante. Cada Individuo nuevo la llama una vez.

    Example:
        Una mutación que el constructor deshace igual cuesta esta llamada:
        el 41.5% de las mutaciones de n pagan y no se mueven.
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
        ``clamp(1.7, 0.0, 1.0)`` es 1.0. Los operadores pueden proponer cualquier
        real; el constructor es quien tiene los bordes.
    """
    return max(low, min(high, gene))


def closest(value: float, allowed: Sequence[int]) -> int:
    """Ajustar un número real al punto legal más cercano de una malla.

    Args:
        value: real propuesto, a menudo n + gauss(0, 1).
        allowed: puntos legales; aquí 0..20, o la malla de espaciado 2 de la deriva.

    Returns:
        El entero legal más cercano.

    Example:
        Con sigma = 1.0, el 41.5% de las mutaciones de n caen dentro de ±0.5 y
        ``closest`` las devuelve al n de partida.
    """
    return min(allowed, key=lambda candidate: abs(candidate - value))


class Individuo:
    """Un cromosoma de cinco genes que no puede existir en un estado ilegal.

    Toda la reparación vive en el constructor, así que los operadores debajo pueden tratar
    cada gen como un número real. La clase cuenta sus propias instanciaciones:
    desde el paso 5 en adelante esa cuenta ES el presupuesto de evaluaciones de aptitud.

    Args:
        gene_list: cinco valores crudos (a, b, x, n, fun_name).

    Returns:
        Un Individuo legal. Si n no se movió, nadie se entera.

    Example:
        Mutar n con sigma = 1.0 y luego construir: 41.5% de las veces el
        ``gene_list`` guardado es el mismo n de entrada.
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
        Los 1000 individuos del censo de mutación salen de aquí. El 10%
        inicial de x no cambia: los operadores aún no han caminado.
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
        alpha: factor de mezcla; 0.3 para a, 0.5 para b y n, 1.0 para x.

    Returns:
        Dos valores hijos, posiblemente fuera del intervalo legal.

    Example:
        x recibe alfa = 1.0 (el del curso); a recibe 0.3 porque su intervalo
        es [0, 1] y un alfa grande lo empuja fuera a cada rato.
    """
    low = min(g1, g2) - alpha * abs(g2 - g1)
    high = max(g1, g2) + alpha * abs(g2 - g1)
    return (low + random.random() * (high - low),
            low + random.random() * (high - low))


# --- NUEVO (1) cruza() ------------------------------------------------------
def cruza(parent1: Individuo, parent2: Individuo
              ) -> tuple[Individuo, Individuo]:
    """La cruza por gen del libro: una moneda por gen, un alfa por gen.

    La Lección 04 cruzó cada gen con el mismo operador. Aquí x recibe la
    mezcla completa (alfa = 1.0) que el curso ha usado todo el tiempo, a y b reciben
    mezclas más suaves, n se mezcla como un real y se deja que el constructor lo ajuste, y
    el gen etiqueta solo se puede intercambiar - no hay punto medio entre 'sin'
    y 'cos', por lo que la cruza para un conjunto es un intercambio o nada.

    Args:
        parent1: primer padre legal.
        parent2: segundo padre legal.

    Returns:
        Dos hijos legales. La reparación, si ocurrió, no se anuncia.

    Example:
        Con SEMILLA = 19, una cruza de dos aleatorios produce hijos cuyos
        genes a, b y x ya no coinciden con ninguno de los padres: cada
        moneda y cada alfa es propia.
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
# ------------------------------------------------------------------------------

# --- NUEVO (2) mutar() ---------------------------------------------------------
def mutar(individual: Individuo) -> Individuo:
    """La mutación por gen del libro: una moneda por gen, un sigma por gen.

    Los genes reales reciben pasos gaussianos dimensionados a sus intervalos. El gen de
    malla recibe el mismo trato que un real - n + gauss(0, 1) - y el
    constructor ajusta el resultado de vuelta en la malla. Esa última línea es
    donde el censo de reparación de abajo encuentra sus números.

    Args:
        individual: individuo legal de partida.

    Returns:
        Un Individuo legal, a veces idéntico al de entrada.

    Example:
        De 1000 llamadas completas, una fracción devuelve genes idénticos:
        cada moneda falló o la reparación deshizo el único cambio. El 41.5%
        de las mutaciones forzadas de n no se mueven.
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
# ------------------------------------------------------------------------------

print("=" * 78)
print("Lección 08 - Paso 4: los operadores, y lo que cuesta la reparación silenciosa")
print("=" * 78)

# Una cruza, una mutación, para que las monedas por gen sean visibles una vez.
random.seed(SEMILLA)
parent1, parent2 = create_random(), create_random()
child1, child2 = cruza(parent1, parent2)
print("\nUna cruza (monedas y alfas difieren por gen)")
print("-" * 78)
print(f"  padre 1:  {parent1}")
print(f"  padre 2:  {parent2}")
print(f"  hijo 1:   {child1}")
print(f"  hijo 2:   {child2}")
print(f"\nUna mutación del padre 1: {mutar(parent1)}")

# --- NUEVO (3) el censo de rep. ------------------------------------------------
# Forzar a que dispare la mutación de n, y contar cuán a menudo n realmente cambia.
# Una gaussiana con sigma 1.0 cae dentro de +-0.5 de su centro el 38.3% de las
# veces, y cada una de esas cae de vuelta en el n desde el que comenzó - el
# ajuste es invisible desde el exterior.
TRIALS = 5000
random.seed(2)
moved = 0
for _ in range(TRIALS):
    n_before = random.choice(N_SET)
    n_after = closest(n_before + random.gauss(0.0, 1.0), N_SET)
    if n_after != n_before:
        moved += 1
snapped_back = TRIALS - moved
print(f"\nEl censo de reparación: {TRIALS} mutaciones forzadas de n, sigma = 1.0")
print("-" * 78)
print(f"  n realmente cambió:            {moved:5d}  ({moved / TRIALS:.1%})")
print(f"  volvió a su estado original:   {snapped_back:5d}  ({snapped_back / TRIALS:.1%})")

# Y el operador completo: de 1000 mutaciones completas, ¿cuántos individuos salen
# con genes idénticos - cada moneda falló o fue reparada?
random.seed(1)
population = [create_random() for _ in range(1000)]
identical = sum(1 for ind in population
                if mutar(ind).gene_list == ind.gene_list)
print(f"\n  De 1000 llamadas completas a mutar(), {identical} devolvieron un individuo con genes "
      f"idénticos ({identical / 1000:.1%}).")
print("  Cada uno de esos costó una evaluación de aptitud y no cambió nada. "
      "La reparación no")
print("  se anuncia a sí misma; simplemente hace que el operador sea más silencioso que lo que "
      "dicen sus parámetros.")
# ------------------------------------------------------------------------------

# --- NUEVO (4) la demo de deriva ---------------------------------------------------
# El ejemplo de mutación imposible del libro, y el caso límite del censo:
# una malla de espaciado 2, un operador que da pasos de máximo 1. El ajuste solo
# puede devolver el punto de partida.
LATTICE = range(-1000, 1001, 2)      # espaciado 2
DRIFT_STEPS = 1000
random.seed(5)
real_walk, discrete_walk = [0.0], [0]
for _ in range(DRIFT_STEPS):
    real_walk.append(real_walk[-1] + random.uniform(-1, 1))
    discrete_walk.append(closest(discrete_walk[-1] + random.uniform(-1, 1),
                                 LATTICE))
print(f"\nLa demo de deriva: {DRIFT_STEPS} mutaciones uniform(-1, +1)")
print("-" * 78)
print(f"  gen real:                    derivó a     {real_walk[-1]:+.2f}")
print(f"  gen de malla (espaciado 2):  terminó en  {discrete_walk[-1]}")
print("\n  Mismo operador, mismos sorteos. El gen real camina aleatoriamente; el gen de malla "
      "nunca")
print("  se mueve, porque cada paso es menor a la mitad del espaciado de la malla, "
      "así que el ajuste")
print("  deshace cada uno de ellos. Nada lanzó excepción, nada advirtió. Cuando la malla es "
      "más gruesa")
print("  que el paso, la mutación no es débil - está ausente.")
# ------------------------------------------------------------------------------

fig, (ax_snap, ax_drift) = plt.subplots(1, 2, figsize=(11.5, 4.0))
# Repetir el patrón de sorteo del censo (semilla 2) para obtener los deltas antes/después.
random.seed(2)
deltas = []
for _ in range(TRIALS):
    n0 = random.choice(N_SET)
    deltas.append(closest(n0 + random.gauss(0.0, 1.0), N_SET) - n0)
ax_snap.hist(deltas, bins=np.arange(-6.5, 6.5, 1.0), color="tab:blue",
             edgecolor="white")
ax_snap.bar([0], [deltas.count(0)], width=1.0, color="tab:red", alpha=0.75,
            label="volvió al mismo n")
ax_snap.set_title("Dónde aterrizaron realmente 5000 mutaciones de n", fontsize=10)
ax_snap.set_xlabel("cambio en n después del ajuste")
ax_snap.set_ylabel("cuenta")
ax_snap.legend(fontsize=8)
ax_snap.grid(True, axis="y", linestyle=":", alpha=0.5)

ax_drift.plot(real_walk, linewidth=1.2, color="tab:blue",
              label=f"gen real (termina {real_walk[-1]:+.2f})")
ax_drift.plot(discrete_walk, linewidth=1.2, color="tab:red",
              label=f"gen malla, espacio 2 (termina {discrete_walk[-1]})")
ax_drift.set_title("Mismo operador, mismos sorteos, 1000 mutaciones", fontsize=10)
ax_drift.set_xlabel("número de mutación")
ax_drift.set_ylabel("valor del gen")
ax_drift.legend(fontsize=8)
ax_drift.grid(True, linestyle=":", alpha=0.5)

fig.suptitle("Reparación silenciosa: el operador es más silencioso de lo que dicen sus parámetros")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "caja_negra_04_los_operadores.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURES}/caja_negra_04_los_operadores.png")

