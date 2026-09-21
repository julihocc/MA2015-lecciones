"""
Lección 08 - Paso 3: Un cromosoma, tres tipos de gen
====================================================
NUEVO EN ESTE PASO: Individual, y las dos funciones de reparación que lo hacen legal.

Hasta ahora un cromosoma era un número real, o un vector de números reales, y
"legal" significaba "dentro de [-10, 10]". Aquí un individuo porta tres tipos
diferentes de gen a la vez:

    a, b, x   valores reales dentro de un intervalo -> se pueden empujar al borde
    n         un punto en una malla entera          -> se puede ajustar al más cercano
    fun_name  miembro de un conjunto sin orden      -> solo se puede reemplazar

Nada en el curso hasta ahora aplica a los últimos dos. La cruza de mezcla de la Lección 04
y la mutación gaussiana de la Lección 05 producen reales arbitrarios; entregados a este
problema producen n = 3.47 y ningún valor de fun_name en absoluto. La decisión de diseño
que toma este paso - y ES una decisión, no un hecho - es dejar que los
operadores sigan siendo descuidados y poner TODA la legalidad en el constructor. Un Individuo
no puede existir en un estado ilegal, por lo que ningún operador tiene que comprobarlo nunca.

El precio de esa decisión se mide aquí y se paga en el paso 4: la reparación es
silenciosa. Un gen que el operador movió y el constructor devolvió a su lugar se ve exactamente
como un gen que el operador nunca tocó.

DESAPARECE EN ESTE PASO: grid_maximum() y la escalera de refinamiento. El paso 2 resolvió
esa pregunta; la cuadrícula no vuelve.

CAMBIOS RESPECTO A caja_negra_02_la_cuadricula.py
Introdúcelos en este orden:
    1. clamp()                un gen real acotado tiene bordes a los que puede ser empujado
    2. closest()              un gen de malla no tiene intermedios; ajustar al más cercano
    3. Individual             el constructor repara, así que los operadores no lo necesitan
    4. create_random()        la población inicial, y dónde vive realmente
    5. repair_report()        medir con qué frecuencia se activa la reparación, ya que es silenciosa

Ejecútalo:  python caja_negra_03_el_cromosoma.py

El constructor repara los cinco ejemplos ilegales. La población inicial muestrea solo el 10% del rango declarado de x: lo que está afuera solo se alcanza caminando.
"""
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 19
POPULATION_SIZE = 200
A_MIN, A_MAX = 0.0, 1.0
B_MIN, B_MAX = 0.0, 1.0
X_MIN, X_MAX = -100.0, 100.0
N_SET = range(0, 21)
FUN_SET = ("sin", "cos")
# El inicializador deliberadamente NO cubre el rango declarado de x. Esta es la
# elección del libro y se mantiene, porque el paso 6 tiene que notarlo.
INIT_X_MIN, INIT_X_MAX = -10.0, 10.0
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def complicated_one(a: float, b: float, x: float, n: int, fun_name: str) -> float:
    """La caja negra, sin cambios desde el paso 1.

    Args:
        a: real acotado a [0, 1].
        b: real acotado a [0, 1].
        x: real acotado a [-100, 100].
        n: entero legal 0..20.
        fun_name: etiqueta ``sin`` o ``cos``.

    Returns:
        Un flotante. El constructor llama esto una vez por individuo, legal o no.

    Example:
        Un cromosoma con n = 0 sigue devolviendo 0.0: esa hoja sigue muerta.
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
print("Lección 08 - Paso 3: un cromosoma, tres tipos de gen")
print("=" * 78)

# --- NUEVO (1) clamp() ----------------------------------------------------------
def clamp(gene: float, low: float, high: float) -> float:
    """Empujar un gen real de vuelta dentro de su intervalo.

    La Lección 04 ya necesitaba esto para un gen en un intervalo. Nada nuevo
    excepto que ahora hay tres de ellos con tres intervalos diferentes, por lo que
    los límites son argumentos en lugar de constantes del módulo.

    Args:
        gene: valor propuesto, posiblemente fuera de [low, high].
        low: borde inferior del intervalo legal.
        high: borde superior del intervalo legal.

    Returns:
        El gen recortado al intervalo.

    Example:
        ``clamp(1.7, 0.0, 1.0)`` es 1.0; ``clamp(250.0, -100.0, 100.0)`` es 100.0.
        Esos son dos de los cinco ejemplos ilegales que repara el constructor.
    """
    return max(low, min(high, gene))
# ------------------------------------------------------------------------------

# --- NUEVO (2) closest() --------------------------------------------------------
def closest(value: float, allowed: Sequence[int]) -> int:
    """Ajustar un número real al punto legal más cercano de una malla.

    Un gen entero no es un gen real con un paso de redondeo añadido; es un
    objeto diferente. No hay n = 3.47 que evaluar, por lo que cualquier operador que
    produzca uno no ha producido nada, y algo tiene que decidir qué significaba.
    El vecino más cercano es la respuesta barata y obvia - y el paso 4 muestra lo que
    cuesta cuando la malla es más gruesa que el tamaño del paso del operador.

    Args:
        value: real propuesto por un operador descuidado.
        allowed: puntos legales de la malla, aquí 0..20.

    Returns:
        El entero de ``allowed`` más cercano a ``value``.

    Example:
        ``closest(3.47, N_SET)`` es 3; ``closest(99.0, N_SET)`` es 20.
        Los cinco ejemplos ilegales salen legales y nadie lanza excepción.
    """
    return min(allowed, key=lambda candidate: abs(candidate - value))
# ------------------------------------------------------------------------------

# --- NUEVO (3) Individual -------------------------------------------------------
class Individuo:
    """Un cromosoma de cinco genes que no puede existir en un estado ilegal.

    Toda la reparación vive aquí, en un solo lugar, para que la cruza y la mutación puedan
    escribirse como si cada gen fuera un número real y aún así nunca producir un
    individuo inválido. La clase cuenta sus propias instanciaciones porque desde el
    paso 5 en adelante el número de Individuos creados ES el número de evaluaciones de
    aptitud gastadas, y ese censo es la única línea de presupuesto que tiene esta lección.

    Args:
        gene_list: cinco valores crudos (a, b, x, n, fun_name). El constructor
            recorta, ajusta y evalúa; no hace falta que el que llama repare.

    Returns:
        Un Individuo legal con ``gene_list`` y ``aptitud``.

    Example:
        ``Individuo([1.7, 0.5, 3.0, 3.47, "cos"])`` guarda a=1.0 y n=3.
        Los cinco ejemplos ilegales del paso 3 salen legales.
    """

    counter = 0

    def __init__(self, gene_list: Sequence[Any]) -> None:
        self.__class__.counter += 1
        a_raw, b_raw, x_raw, n_raw, fun_name = gene_list
        self.raw_gene_list: List[Any] = list(gene_list)
        self.gene_list: List[Any] = [
            clamp(float(a_raw), A_MIN, A_MAX),
            clamp(float(b_raw), B_MIN, B_MAX),
            clamp(float(x_raw), X_MIN, X_MAX),
            closest(float(n_raw), N_SET),
            fun_name,
        ]
        a, b, x, n, fun_name = self.gene_list
        self.aptitud: float = complicated_one(a, b, x, n, fun_name)

    def __str__(self) -> str:
        a, b, x, n, fun_name = self.gene_list
        return f"[a={a:.4f} b={b:.4f} x={x:+.4f} n={n:2d} fun={fun_name}]"
# ------------------------------------------------------------------------------

# --- NUEVO (4) create_random() --------------------------------------------------
def create_random() -> Individuo:
    """Un individuo aleatorio, cada gen extraído de la forma que su propio tipo permite.

    Nota la asimetría: a y b se extraen de todo su rango declarado, pero x
    se extrae de [-10, 10] de un declarado [-100, 100]. El gen de malla se
    extrae eligiendo un punto legal, nunca extrayendo un real y ajustando, y
    el gen etiqueta se extrae eligiendo un miembro. No hay otra manera de
    extraerlos - que es el punto.

    Returns:
        Un Individuo legal. x vive en [-10, 10], no en el rango declarado.

    Example:
        Con SEMILLA = 19 y 200 individuos, x cubre el 10% del intervalo
        declarado [-100, 100]. Lo que está afuera solo se alcanza por mutación.
    """
    return Individuo([
        random.uniform(A_MIN, A_MAX),
        random.uniform(B_MIN, B_MAX),
        random.uniform(INIT_X_MIN, INIT_X_MAX),
        random.choice(N_SET),
        random.choice(FUN_SET),
    ])
# ------------------------------------------------------------------------------

# --- NUEVO (5) repair_report() --------------------------------------------------
def repair_report(candidates: Sequence[Sequence[Any]]) -> List[Tuple[str, Any, Any]]:
    """Construir cada candidato y registrar qué genes cambió el constructor.

    La reparación es invisible desde afuera: el que llama entrega una lista de genes y recibe
    de vuelta un Individuo legal, sin señal de que algo fue alterado. Eso es
    conveniente y es peligroso, por lo que esta función existe para hacer la parte silenciosa
    audible al menos una vez.

    Args:
        candidates: listas de cinco genes, posiblemente ilegales.

    Returns:
        Filas ``(nombre, crudo, guardado)`` solo para los genes que cambiaron.

    Example:
        Los cinco ejemplos ilegales producen cinco filas de reparación y cero
        excepciones. El silencio es la decisión de diseño de este paso.
    """
    rows: List[Tuple[str, Any, Any]] = []
    for gene_list in candidates:
        individual = Individuo(gene_list)
        for name, raw, kept in zip("a b x n fun".split(), gene_list, individual.gene_list):
            if raw != kept:
                rows.append((name, raw, kept))
    return rows
# ------------------------------------------------------------------------------

print("\nLo que le hace el constructor a genes que un operador podría entregarle")
print("-" * 78)
ILLEGAL = [
    [1.7, 0.5, 3.0, 3, "cos"],        # a por encima de su techo
    [0.5, -0.4, 3.0, 3, "cos"],       # b por debajo de su piso
    [0.5, 0.5, 250.0, 3, "cos"],      # x muy fuera del rango declarado
    [0.5, 0.5, 3.0, 3.47, "cos"],     # n entre dos puntos de malla
    [0.5, 0.5, 3.0, 99.0, "cos"],     # n más allá del final de la malla
]
for name, raw, kept in repair_report(ILLEGAL):
    print(f"  gen {name:<3} entregado {str(raw):>8}  ->  guardado como {str(kept):>8}")
print("\n  Cinco listas de genes ilegales entraron; cinco Individuos legales salieron, y ni")
print("  uno de ellos lanzó excepción. Esa es la decisión de diseño de este paso, declarada tan")
print("  sin rodeos como puede ser declarada.")

random.seed(SEMILLA)
Individuo.counter = 0
population = [create_random() for _ in range(POPULATION_SIZE)]

print(f"\nLa población inicial ({POPULATION_SIZE} individuos, SEMILLA = {SEMILLA})")
print("-" * 78)
xs = [ind.gene_list[2] for ind in population]
ns = [ind.gene_list[3] for ind in population]
funs = Counter(ind.gene_list[4] for ind in population)
fits = [ind.aptitud for ind in population]
declared_width = X_MAX - X_MIN
covered = max(xs) - min(xs)
print(f"  x  abarca [{min(xs):+.3f}, {max(xs):+.3f}] - {covered / declared_width * 100:.1f}% del "
      f"declarado [{X_MIN:+.0f}, {X_MAX:+.0f}]")
print(f"  n  toma {len(set(ns))} de los {len(N_SET)} valores legales; "
      f"el más común es n={Counter(ns).most_common(1)[0][0]} "
      f"({Counter(ns).most_common(1)[0][1]} individuos)")
print(f"  fun_name: " + ", ".join(f"{k}={v}" for k, v in sorted(funs.items())))
dead = [ind for ind in population if ind.aptitud == 0.0]
on_dead_sheet = [ind for ind in population if ind.gene_list[3] == 0]
print(f"\n  {len(dead)} de {POPULATION_SIZE} individuos tienen aptitud exactamente 0.0, y "
      f"{len(on_dead_sheet)} de ellos")
print(f"  están sentados en la hoja n = 0 que el paso 1 descubrió que es idénticamente cero.")
print(f"  Mejor aptitud en la población inicial: {max(fits):.6e}")
print(f"  Evaluaciones de aptitud gastadas hasta ahora: {Individuo.counter}")

print("\nLa parte que no es gratis")
print("-" * 78)
print(f"  El inicializador muestrea x sobre [{INIT_X_MIN:+.0f}, {INIT_X_MAX:+.0f}] mientras la caja acepta")
print(f"  [{X_MIN:+.0f}, {X_MAX:+.0f}]. Eso es {(INIT_X_MAX - INIT_X_MIN) / declared_width * 100:.0f}% del eje declarado, y es la elección del")
print("  libro, conservada deliberadamente. Lo que sea que esté afuera solo se puede alcanzar por")
print("  mutación caminando allí un paso a la vez. El paso 6 verifica si alguna vez lo hizo.")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
axes[0].hist(xs, bins=25, color="tab:blue", edgecolor="white")
axes[0].axvspan(X_MIN, INIT_X_MIN, color="tab:red", alpha=0.12)
axes[0].axvspan(INIT_X_MAX, X_MAX, color="tab:red", alpha=0.12, label="declarado pero nunca muestreado")
axes[0].set_xlim(X_MIN, X_MAX)
axes[0].set_title("x: real, y solo se usa el 10%", fontsize=10)
axes[0].set_xlabel("x")
axes[0].legend(fontsize=8)
axes[1].hist(ns, bins=np.arange(min(N_SET) - 0.5, max(N_SET) + 1.5, 1.0),
             color="tab:green", edgecolor="white")
axes[1].set_title("n: una malla, 21 puntos legales", fontsize=10)
axes[1].set_xlabel("n")
axes[2].bar(list(sorted(funs)), [funs[k] for k in sorted(funs)], color="tab:purple")
axes[2].set_title("fun_name: conjunto, sin orden, sin punto medio", fontsize=10)
axes[2].set_xlabel("fun_name")
for ax in axes:
    ax.set_ylabel("individuos")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
fig.suptitle(f"Un cromosoma, tres tipos de gen (población inicial, SEMILLA = {SEMILLA})")
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "caja_negra_03_el_cromosoma.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURES}/caja_negra_03_el_cromosoma.png")

