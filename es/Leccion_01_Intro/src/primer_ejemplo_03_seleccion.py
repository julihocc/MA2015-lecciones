"""
Lección 01 - Paso 3: Selección
===============================
NUEVO EN ESTE PASO: seleccion_torneo().

La población del paso 2 es aleatoria y en su mayoría mala. La selección es la
primera fuerza de la evolución: decide quién llega a reproducirse. Aquí no se
crea nada nuevo: solo elegimos, con repetición, entre lo que ya existe.

CAMBIOS RESPECTO A primer_ejemplo_02_poblacion_aleatoria.py
Introdúcelos en este orden:
    1. seleccion_torneo()     la primera fuerza de la evolución: quién se reproduce

Ejecútalo:  python primer_ejemplo_03_seleccion.py

La selección solo copia: no aparece ningún valor nuevo de x. En esta
corrida, 4 de 10 individuos se extinguen.
"""
import random
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 52
TAMANO_POBLACION = 10
GEN_MIN, GEN_MAX = -10.0, 10.0
TAMANO_TORNEO = 3
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


def objetivo(x):
    """La función que queremos MAXIMIZAR.

    Argumentos:
        x: un real o un arreglo de numpy.

    Devuelve:
        sen(x) - 0.2 * |x|. Máximo global cerca de x = +1.372, f = +0.706.

    Ejemplo:
        objetivo(+1.372) ≈ +0.706.
    """
    return np.sin(x) - 0.2 * abs(x)


class Individuo:
    """Una solución candidata: un cromosoma y su aptitud.

    Argumentos:
        lista_genes: cromosoma; aquí un solo real en [-10, 10].

    Ejemplo:
        Con SEMILLA = 52, 4 de 10 individuos se extinguen tras el torneo.
    """

    def __init__(self, lista_genes):
        self.lista_genes = lista_genes
        self.aptitud = objetivo(lista_genes[0])

    @property
    def gen(self):
        """El único gen: este paisaje es unidimensional."""
        return self.lista_genes[0]

    def __repr__(self):
        return f"x={self.gen:+.3f} f={self.aptitud:+.3f}"


def crear_aleatorio():
    """Extrae un individuo uniforme de [-10, 10].

    Devuelve:
        Individuo con un gen en [GEN_MIN, GEN_MAX].
    """
    return Individuo([random.uniform(GEN_MIN, GEN_MAX)])


# --- NUEVO (1) seleccion_torneo() ---------------------------------------------
def seleccion_torneo(poblacion, tamano):
    """Corre un torneo por cada lugar de la nueva generación.

    Cada torneo toma `tamano` individuos al azar y se queda con el más apto.
    Un individuo fuerte gana varios torneos y aparece varias veces; uno débil
    probablemente no aparece nunca. `tamano` por sí solo controla qué tan
    codiciosa es la selección. No inventa genes: solo copia referencias.

    Argumentos:
        poblacion: lista de Individuo.
        tamano: competidores por torneo (aquí 3).

    Devuelve:
        Lista de la misma longitud, hecha de copias (las mismas instancias).

    Ejemplo:
        Con SEMILLA = 52 y tamano = 3, 4 de 10 individuos se extinguen.
    """
    descendencia = []
    for _ in range(len(poblacion)):
        candidatos = [random.choice(poblacion) for _ in range(tamano)]
        descendencia.append(max(candidatos, key=lambda ind: ind.aptitud))
    return descendencia
# ------------------------------------------------------------------------------


random.seed(SEMILLA)
poblacion = [crear_aleatorio() for _ in range(TAMANO_POBLACION)]
descendencia = seleccion_torneo(poblacion, TAMANO_TORNEO)

print("ANTES de la selección:")
for ind in sorted(poblacion, key=lambda i: -i.aptitud):
    print("   ", ind)

print("\nDESPUÉS de la selección (los mismos individuos, otras multiplicidades):")
# id() y no el valor de x: la selección devuelve las mismas instancias.
conteos = Counter(id(ind) for ind in descendencia)
for ind in sorted(poblacion, key=lambda i: -i.aptitud):
    n = conteos.get(id(ind), 0)
    print(f"    {ind}   seleccionado {n} vez(ces)  {'#' * n}")

extintos = [ind for ind in poblacion if conteos.get(id(ind), 0) == 0]
print("\nObserva: no apareció ningún valor nuevo de x. La selección solo copia.")
print(f"{len(extintos)} de {TAMANO_POBLACION} individuos fueron seleccionados cero veces.")
print("No queda ninguna copia seleccionada que pueda reproducirse directamente;")
print("la mutación todavía podría regenerar valores iguales o cercanos.")

ordenados = sorted(poblacion, key=lambda i: i.aptitud)
xs = [ind.aptitud for ind in ordenados]
ys = [conteos.get(id(ind), 0) for ind in ordenados]
colores = ["tab:red" if n == 0 else "tab:blue" for n in ys]

plt.figure(figsize=(8, 4))
plt.vlines(xs, 0, ys, colors=colores, linewidth=2, zorder=2)
plt.scatter(xs, ys, c=colores, s=90, zorder=3, edgecolors="white", linewidths=0.6)
plt.scatter([], [], color="tab:red", s=90, label="extintos (0 copias)")
plt.scatter([], [], color="tab:blue", s=90, label="seleccionados")
plt.title("La selección solo copia; los puntos rojos ya no están")
plt.xlabel("Aptitud f(x)")
plt.ylabel("Veces seleccionado (multiplicidad)")
plt.yticks(range(0, max(ys) + 2))
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5, zorder=0)
FIGURAS.mkdir(exist_ok=True)
plt.savefig(FIGURAS / "primer_ejemplo_03_seleccion.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigura guardada en {FIGURAS}/primer_ejemplo_03_seleccion.png")

