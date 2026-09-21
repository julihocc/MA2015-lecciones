"""
Lección 08 - Paso 2: La cuadrícula, tasada y luego refinada
===========================================================
NUEVO EN ESTE PASO: la comprobación por fuerza bruta de las Lecciones 01-06, costeada honestamente en
este problema y luego interrogada.

Cada lección anterior verificó sus ejecuciones contra un óptimo de fuerza bruta en una cuadrícula.
Esa comprobación es lo que hizo que "¿tuvo éxito esta ejecución?" fuera respondible. Este paso intenta
comprarlo aquí, y falla dos veces.

El primer fracaso es aritmético: la propia cuadrícula del libro sobre estos cinco genes es de
105 millones de evaluaciones, y medimos - no adivinamos - cuánto cuesta eso.

El segundo fracaso es peor, y es el que hay que enseñar. Incluso una cuadrícula que PODEMOS
pagar da una respuesta que no se queda quieta. Refina el eje x por diez y el
máximo reportado no converge a un límite; se multiplica. Una secuencia que
crece por 2x, luego 21x, luego 1387x no es una estimación acercándose a un valor. Es
la cuadrícula diciéndonos, en el único idioma que tiene una cuadrícula, que sea lo que sea que esté
muestreando no es un máximo. Todavía no se nos permite abrir la caja, así que tomamos
eso como dato y lo llevamos adelante.

CAMBIOS RESPECTO A caja_negra_01_sin_imagen.py
Introdúcelos en este orden:
    1. evaluation_cost()      medir el precio de una llamada antes de citar cualquier total
    2. BOOK_GRID              la fuerza bruta del libro, tasada en lugar de ejecutada
    3. grid_maximum()         la búsqueda exhaustiva que realmente podemos pagar
    4. REFINEMENTS            ejecutarla a cuatro resoluciones y leer la secuencia

Ejecútalo:  python caja_negra_02_la_cuadricula.py

La cuadrícula del libro cuesta 105,525,000 llamadas. Refinar x hace que el máximo reportado crezca 61,099 veces en lugar de converger: la fuerza bruta no tiene un techo que alcanzar.
"""
import math
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

A_MIN, A_MAX = 0.0, 1.0
B_MIN, B_MAX = 0.0, 1.0
X_MIN, X_MAX = -100.0, 100.0
N_SET = range(0, 21)
FUN_SET = ("sin", "cos")
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
        Un flotante. El ciclo interno corre n+1 veces, así que el precio no es constante.

    Example:
        La cuadrícula del libro pide 105,525,000 de estas llamadas. Tasarlas es
        el punto de este paso; ejecutarlas no cabe en una clase.
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


print("=" * 78)
print("Lección 08 - Paso 2: la cuadrícula, tasada y luego refinada")
print("=" * 78)

# --- NUEVO (1) evaluation_cost() ------------------------------------------------
def evaluation_cost(samples: int = 20000, seed: int = 7) -> float:
    """Segundos por llamada a la caja, medidos en esta máquina, en este momento.

    Citar "105 millones de evaluaciones" no significa nada sin un precio por
    evaluación, y el precio no es una constante del problema: el ciclo interno de la
    caja se ejecuta n+1 veces, por lo que una llamada con n = 20 cuesta veinte veces una llamada con
    n = 0. Por lo tanto, cronometramos una muestra extraída de la forma en que la extraerá la cuadrícula -
    n uniforme sobre todo su conjunto - y reportamos la media.

    Args:
        samples: llamadas cronometradas; 20000 suaviza el ruido del reloj.
        seed: semilla de numpy para que la muestra de (a, b, x, n, fun) sea reproducible.

    Returns:
        Segundos por llamada, media de ``samples`` evaluaciones.

    Example:
        Multiplicado por 105,525,000 da el costo de la cuadrícula del libro en
        esta máquina. El número de segundos cambia de laptop; el producto de
        ejes no.
    """
    rng = np.random.default_rng(seed)
    a_s = rng.uniform(A_MIN, A_MAX, samples)
    b_s = rng.uniform(B_MIN, B_MAX, samples)
    x_s = rng.uniform(X_MIN, X_MAX, samples)
    n_s = rng.integers(min(N_SET), max(N_SET) + 1, samples)
    f_s = rng.integers(0, len(FUN_SET), samples)
    start = time.perf_counter()
    for a, b, x, n, fi in zip(a_s, b_s, x_s, n_s, f_s):
        complicated_one(float(a), float(b), float(x), int(n), FUN_SET[int(fi)])
    return (time.perf_counter() - start) / samples
# ------------------------------------------------------------------------------

# --- NUEVO (2) BOOK_GRID --------------------------------------------------------
# Fuerza bruta del Capítulo 8 de Gridin, transcrita como tamaños en lugar de bucles:
#   a in arange(0, 1, .02)      b in arange(0, 1, .02)
#   x in arange(-100, 101, .2)  n in range(0, 21)      fun in {cos, sin}
# El libro lo ejecuta. Nosotros lo tasamos, porque tasarlo es la lección.
BOOK_GRID: List[Tuple[str, int, str]] = [
    ("a", len(np.arange(0.0, 1.0, 0.02)), "paso 0.02"),
    ("b", len(np.arange(0.0, 1.0, 0.02)), "paso 0.02"),
    ("x", len(np.arange(-100.0, 101.0, 0.2)), "paso 0.2"),
    ("n", len(N_SET), "cada valor legal"),
    ("fun_name", len(FUN_SET), "cada valor legal"),
]
# ------------------------------------------------------------------------------

seconds_per_eval = evaluation_cost()
print(f"\nCosto medido de una llamada a la caja: {seconds_per_eval * 1e6:.2f} microsegundos")
print(f"  (media sobre 20000 llamadas con n extraído uniformemente de {min(N_SET)}..{max(N_SET)})")

print("\nLa cuadrícula de fuerza bruta del libro, tasada")
print("-" * 78)
book_total = 1
for name, count, note in BOOK_GRID:
    book_total *= count
    print(f"  {name:<9} {count:>6d} puntos   ({note})")
book_seconds = book_total * seconds_per_eval
print(f"  {'':<9} {'':>6}   producto = {book_total:,} evaluaciones")
print(f"  Al precio medido eso es {book_seconds:,.0f} s = {book_seconds / 60:.1f} minutos,")
print("  para un problema, en una clase, con cinco genes. La cuadrícula de la Lección 02 era de 20,001")
print(f"  puntos y terminó al instante; esta es {book_total / 20001:,.0f} veces más grande, porque")
print("  una cuadrícula cuesta el PRODUCTO de sus ejes y este problema tiene cinco de ellos.")

# --- NUEVO (3) grid_maximum() ---------------------------------------------------
def grid_maximum(a_values: Sequence[float], b_values: Sequence[float],
                 x_values: Sequence[float], n_values: Sequence[int],
                 fun_values: Sequence[str]) -> Tuple[float, Tuple, int]:
    """Búsqueda exhaustiva sobre una cuadrícula explícita. Retorna (mejor, dónde, evaluaciones).

    Deliberadamente escrito de la manera ingenua. No hay astucia que agregar: toda la
    afirmación de la fuerza bruta es que busca en todas partes, y todo su costo es
    que "todas partes" es un producto.

    Args:
        a_values: puntos legales de a.
        b_values: puntos legales de b.
        x_values: puntos legales de x; este es el eje que se refina.
        n_values: enteros legales de n.
        fun_values: etiquetas legales de fun_name.

    Returns:
        ``(mejor, (a, b, x, n, fun), evaluaciones)``. El tercer valor es el
        producto de las longitudes, no una estimación.

    Example:
        Con a y b en {0.0, 0.5}, n = 3 y ambas etiquetas, refinar x de 201
        a 200001 puntos hace crecer el máximo reportado 61,099 veces. Una
        estimación convergente no hace eso.
    """
    best = -math.inf
    where: Tuple = ()
    evaluations = 0
    for a in a_values:
        for b in b_values:
            for n in n_values:
                for fun_name in fun_values:
                    for x in x_values:
                        evaluations += 1
                        value = complicated_one(float(a), float(b), float(x), int(n), fun_name)
                        if value > best:
                            best, where = value, (float(a), float(b), float(x), int(n), fun_name)
    return best, where, evaluations
# ------------------------------------------------------------------------------

# --- NUEVO (4) REFINEMENTS ------------------------------------------------------
# Una cuadrícula que podemos pagar: los dos genes continuos a y b tienen dos valores cada uno, los
# genes discretos están fijados, y solo se refina x. Esto es generoso con la cuadrícula -
# le estamos permitiendo gastar todo su presupuesto en el único eje que tiene más probabilidades de
# importar, lo cual es el mejor caso para el método, no el peor.
GRID_A = (0.0, 0.5)
GRID_B = (0.0, 0.5)
GRID_N = (3,)
REFINEMENTS = (201, 2001, 20001, 200001)
# ------------------------------------------------------------------------------

print("\nRefinando un eje: a en {0.0, 0.5}, b en {0.0, 0.5}, n = 3, ambas etiquetas")
print("-" * 78)
print(f"  {'puntos x':>9} {'espaciado':>9} {'evaluaciones':>12} {'max reportado':>18} "
      f"{'en x':>10} {'vs anterior':>12}")
ladder: List[Tuple[float, float, float]] = []
previous = None
for points in REFINEMENTS:
    xs = np.linspace(X_MIN, X_MAX, points)
    best, where, evaluations = grid_maximum(GRID_A, GRID_B, xs, GRID_N, FUN_SET)
    spacing = (X_MAX - X_MIN) / (points - 1)
    ratio = best / previous if previous else float("nan")
    ladder.append((spacing, best, evaluations))
    shown = f"{ratio:11.1f}x" if previous else f"{'-':>12}"
    print(f"  {points:9d} {spacing:9.4f} {evaluations:12d} {best:18.6e} "
          f"{where[2]:10.3f} {shown}")
    previous = best

# Cada oración a continuación se calcula a partir de la escalera recién impresa.
growth = [ladder[i][1] / ladder[i - 1][1] for i in range(1, len(ladder))]
total_growth = ladder[-1][1] / ladder[0][1]
print(f"\n  Cada refinamiento de diez veces multiplicó el máximo reportado por "
      f"{', '.join(f'{g:.1f}' for g in growth)}.")
print(f"  A lo largo de toda la escalera la respuesta creció {total_growth:,.0f} veces y el crecimiento")
print(f"  {'se aceleró' if growth[-1] > growth[0] else 'se desaceleró'} en lugar de estabilizarse.")
print("\n  Una estimación convergente tiene razones que se acercan a 1. Estas van hacia el otro")
print("  lado. Sea lo que sea que la cuadrícula esté escalando, no tiene una cima que la cuadrícula pueda alcanzar,")
print("  y el número que imprime es una declaración sobre el espaciado, no sobre la")
print("  función. Cualquier lección anterior habría citado la fila más fina como 'el")
print("  óptimo'. Aquí ese sería un número inventado.")

print("\nLo que esto nos cuesta")
print("-" * 78)
print("  Las Lecciones 01-06 respondieron '¿tuvo éxito la ejecución?' comparando la ejecución contra")
print("  la cuadrícula. Ambas mitades de eso ahora se han ido: la cuadrícula honesta es inasequible,")
print("  y la cuadrícula asequible no tiene respuesta que dar. Los pasos 3 a 5 construyen la búsqueda")
print("  de todos modos, y el paso 6 tiene que encontrar alguna otra manera de juzgarla.")

fig, ax = plt.subplots(figsize=(7.5, 5))
spacings = [s for s, _, _ in ladder]
values = [v for _, v, _ in ladder]
ax.loglog(spacings, values, "o-", color="tab:red", linewidth=1.6, markersize=7)
for spacing, value, evaluations in ladder:
    ax.annotate(f"{evaluations:,} evals", (spacing, value),
                textcoords="offset points", xytext=(8, -12), fontsize=8)
ax.invert_xaxis()
ax.set_xlabel("espaciado de cuadrícula en x (más fino hacia la derecha)")
ax.set_ylabel("máximo que la cuadrícula reporta")
ax.set_title("Una respuesta de fuerza bruta que se niega a converger")
ax.grid(True, which="both", linestyle=":", alpha=0.5)
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "caja_negra_02_la_cuadricula.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURES}/caja_negra_02_la_cuadricula.png")

