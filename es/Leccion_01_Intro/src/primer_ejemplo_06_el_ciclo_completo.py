"""
Lección 01 - Paso 6: El algoritmo genético completo
===================================================
NUEVO EN ESTE PASO: el ciclo generacional que junta los pasos 3, 4 y 5.

    inicializar -> [ SELECCIONAR -> CRUZAR -> MUTAR -> reemplazar ] x N

Nada de lo que sigue es nuevo salvo el ciclo mismo.

CAMBIOS RESPECTO A primer_ejemplo_05_mutacion.py
Introdúcelos en este orden:
    1. el ciclo generacional  llamar a SELECCIONAR, CRUZAR y MUTAR en orden, N veces
    2. la gráfica de convergencia  registrar la mejor aptitud de cada generación y graficarla

Ejecútalo:  python primer_ejemplo_06_el_ciclo_completo.py

Diez generaciones con SEMILLA = 52 encuentran x = +1.372, f = +0.706, cerca de
la referencia muestreada de la malla del paso 1.
"""
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 52
TAMANO_POBLACION = 10
PROB_CRUZA = 0.8
PROB_MUTACION = 0.1
MAX_GENERACIONES = 10
GEN_MIN, GEN_MAX = -10.0, 10.0
TAMANO_TORNEO, ALFA_MEZCLA = 3, 1.0
MUTACION_MU, MUTACION_SIGMA = 0.0, 1.0
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


def objetivo(x: float | np.ndarray) -> float | np.ndarray:
    """La función que queremos MAXIMIZAR.

    Argumentos:
        x: un real o un arreglo de numpy.

    Devuelve:
        sen(x) - 0.2 * |x|. La malla densa muestra su cima más alta cerca de
        x = +1.372, f = +0.706; no es una prueba exacta del continuo.

    Ejemplo:
        Con SEMILLA = 52, el ciclo llega a x = +1.372, f = +0.706.
    """
    return np.sin(x) - 0.2 * abs(x)


def acotar(g: float, minimo: float = GEN_MIN, maximo: float = GEN_MAX) -> float:
    """Proyecta un gen al intervalo cerrado del paisaje.

    Argumentos:
        g: gen crudo.
        minimo, maximo: extremos cerrados.

    Devuelve:
        g proyectado a [minimo, maximo].
    """
    return max(minimo, min(maximo, g))


class Individuo:
    """Una solución candidata: un cromosoma y su aptitud.

    Argumentos:
        lista_genes: cromosoma; aquí un solo real en [-10, 10].
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


def crear_aleatorio() -> Individuo:
    """Extrae un individuo uniforme de [-10, 10].

    Devuelve:
        Individuo con un gen en [GEN_MIN, GEN_MAX].
    """
    return Individuo([random.uniform(GEN_MIN, GEN_MAX)])


def seleccion_torneo(poblacion: list[Individuo], tamano: int) -> list[Individuo]:
    """Un torneo por cada lugar de la nueva generación.

    Argumentos:
        poblacion: lista de Individuo.
        tamano: competidores por torneo (aquí 3).

    Devuelve:
        Lista de la misma longitud, hecha de copias.

    Ejemplo:
        En el paso 3, tamano = 3 extinguió a 4 de 10 individuos.
    """
    return [max([random.choice(poblacion) for _ in range(tamano)],
                key=lambda i: i.aptitud) for _ in range(len(poblacion))]


def cruza(p1: Individuo, p2: Individuo) -> tuple[Individuo, Individuo]:
    """Variante complementaria de cruza por mezcla entre dos Individuo.

    Argumentos:
        p1, p2: progenitores.

    Devuelve:
        Dos Individuo nuevos, acotados a [-10, 10].
    """
    desplazamiento = (1 + 2 * ALFA_MEZCLA) * random.random() - ALFA_MEZCLA
    g1 = acotar((1 - desplazamiento) * p1.gen + desplazamiento * p2.gen)
    g2 = acotar(desplazamiento * p1.gen + (1 - desplazamiento) * p2.gen)
    return Individuo([g1]), Individuo([g2])


def mutar(ind: Individuo) -> Individuo:
    """Suma ruido gaussiano a un gen y reconstruye el Individuo.

    Argumentos:
        ind: candidato a perturbar.

    Devuelve:
        Un Individuo nuevo.
    """
    return Individuo([acotar(ind.gen + random.gauss(MUTACION_MU, MUTACION_SIGMA))])


# --- NUEVO (1) el ciclo generacional ------------------------------------------
random.seed(SEMILLA)
poblacion = [crear_aleatorio() for _ in range(TAMANO_POBLACION)]
historia = [max(poblacion, key=lambda i: i.aptitud).aptitud]
print(f"Generación  0 | mejor {historia[0]:+.4f}")

for generacion in range(1, MAX_GENERACIONES + 1):

    descendencia = seleccion_torneo(poblacion, TAMANO_TORNEO)        # SELECCIONAR

    cruzados = []                                                    # CRUZAR
    # Pares consecutivos: la selección ya barajó el orden al copiar.
    for p1, p2 in zip(descendencia[::2], descendencia[1::2]):
        if random.random() < PROB_CRUZA:
            cruzados.extend(cruza(p1, p2))
        else:
            cruzados.extend([p1, p2])

    poblacion = [mutar(i) if random.random() < PROB_MUTACION else i
                 for i in cruzados]                                  # MUTAR + reemplazar

    mejor = max(poblacion, key=lambda i: i.aptitud)
    historia.append(mejor.aptitud)
    print(f"Generación {generacion:2d} | mejor {mejor.aptitud:+.4f}")
# ------------------------------------------------------------------------------

mejor = max(poblacion, key=lambda i: i.aptitud)
_malla = np.linspace(GEN_MIN, GEN_MAX, 400)
_x_verdadero = float(_malla[np.argmax(objetivo(_malla))])
print(f"\nSolución: {mejor}   (la referencia de malla está cerca de "
      f"x={_x_verdadero:+.3f}, f={objetivo(_x_verdadero):+.3f})")

# --- NUEVO (2) la gráfica de convergencia -------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(range(len(historia)), historia, "o-", color="tab:green")
plt.title("Mejor aptitud por generación")
plt.xlabel("generación"); plt.ylabel("mejor f(x)")
plt.grid(True, linestyle=":", alpha=0.5)
FIGURAS.mkdir(exist_ok=True)
plt.savefig(FIGURAS / "primer_ejemplo_06_el_ciclo_completo.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Figura guardada en {FIGURAS}/primer_ejemplo_06_el_ciclo_completo.png")
# ------------------------------------------------------------------------------

