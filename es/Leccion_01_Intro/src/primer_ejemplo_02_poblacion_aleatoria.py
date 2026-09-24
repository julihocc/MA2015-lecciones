"""
Lección 01 - Paso 2: Una población
===================================
NUEVO EN ESTE PASO: la clase Individuo y una población aleatoria.

Un algoritmo genético no sigue la pista de una sola solución candidata, sigue
muchas. Aquí simplemente esparcimos diez al azar y vemos dónde caen.

CAMBIOS RESPECTO A primer_ejemplo_01_el_paisaje.py
Introdúcelos en este orden:
    1. Individuo              una solución candidata: sus genes y la aptitud que producen
    2. crear_aleatorio()      extraer un individuo uniformemente del espacio de búsqueda
    3. la población           construir diez y graficar dónde cayeron

Ejecútalo:  python primer_ejemplo_02_poblacion_aleatoria.py

Diez suposiciones a ciegas con SEMILLA = 52. Ninguna cae cerca del óptimo
global x = +1.372; el AG todavía no busca, solo muestrea.
"""
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 52
TAMANO_POBLACION = 10
GEN_MIN, GEN_MAX = -10.0, 10.0
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


def objetivo(x: float | np.ndarray) -> float | np.ndarray:
    """La función que queremos MAXIMIZAR.

    Argumentos:
        x: un real o un arreglo de numpy.

    Devuelve:
        sen(x) - 0.2 * |x|. Máximo global cerca de x = +1.372, f = +0.706.

    Ejemplo:
        objetivo(+1.372) ≈ +0.706.
    """
    return np.sin(x) - 0.2 * abs(x)


# --- NUEVO (1) Individuo ------------------------------------------------------
class Individuo:
    """Una solución candidata: un cromosoma y su aptitud.

    La aptitud se calcula una sola vez, al crear el individuo, de modo que
    nunca puede quedar desfasada respecto a los genes.

    Argumentos:
        lista_genes: cromosoma; aquí un solo real en [-10, 10].

    Ejemplo:
        Con SEMILLA = 52, diez individuos aleatorios no rozan x = +1.372.
    """

    def __init__(self, lista_genes: list[float]) -> None:
        self.lista_genes = lista_genes
        self.aptitud = objetivo(lista_genes[0])

    @property
    def gen(self) -> float:
        """El único gen: este paisaje es unidimensional."""
        return self.lista_genes[0]

    def __repr__(self) -> str:
        return f"x={self.gen:+.3f} f={self.aptitud:+.3f}"
# ------------------------------------------------------------------------------


# --- NUEVO (2) crear_aleatorio() ----------------------------------------------
def crear_aleatorio() -> Individuo:
    """Extrae un individuo uniforme del intervalo cerrado [-10, 10].

    Uniforme a ciegas: todavía no hay presión ni memoria. Diez de estas
    extraídas no caen cerca del óptimo; eso es lo que prueba este paso.

    Devuelve:
        Individuo con un gen en [GEN_MIN, GEN_MAX].

    Ejemplo:
        Con SEMILLA = 52, el mejor de diez sigue lejos de x = +1.372.
    """
    return Individuo([random.uniform(GEN_MIN, GEN_MAX)])
# ------------------------------------------------------------------------------


# --- NUEVO (3) la población ---------------------------------------------------
random.seed(SEMILLA)
poblacion = [crear_aleatorio() for _ in range(TAMANO_POBLACION)]

print("Población inicial (diez suposiciones a ciegas):")
for ind in poblacion:
    print("   ", ind)

mejor = max(poblacion, key=lambda i: i.aptitud)
print(f"\nMejor de la población inicial: {mejor}")

malla = np.linspace(GEN_MIN, GEN_MAX, 400)
plt.figure(figsize=(8, 4))
plt.plot(malla, objetivo(malla), "--", color="tab:blue")
plt.plot([i.gen for i in poblacion], [i.aptitud for i in poblacion],
         "o", color="#A65300", label="población")
plt.plot([mejor.gen], [mejor.aptitud], "s", color="tab:green", markersize=9, label="mejor")
plt.title("Generación 0: población aleatoria")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.legend(); plt.grid(True, linestyle=":", alpha=0.5)
FIGURAS.mkdir(exist_ok=True)
plt.savefig(FIGURAS / "primer_ejemplo_02_poblacion_aleatoria.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigura guardada en {FIGURAS}/primer_ejemplo_02_poblacion_aleatoria.png")
# ------------------------------------------------------------------------------

