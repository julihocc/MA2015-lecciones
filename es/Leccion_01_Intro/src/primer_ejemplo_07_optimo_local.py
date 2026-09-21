"""
Lección 01 - Paso 7: Cuando falla
==================================
NUEVO EN ESTE PASO: nada. Es el paso 6 con UN carácter cambiado.

    SEMILLA = 52  ->  SEMILLA = 16

Mismo algoritmo, mismos parámetros, mismo número de generaciones. Solo cambia
la población inicial aleatoria, y el resultado es un fracaso: la corrida se
asienta en un máximo local y nunca lo abandona.

Esto no es un error que haya que corregir. Es el comportamiento honesto del
método, y es la razón de que existan las Lecciones 03, 04, 05 y 07: la presión
de selección, la elección de operadores y la afinación de parámetros son todos
intentos de hacer que este fracaso sea menos probable.

CAMBIOS RESPECTO A primer_ejemplo_06_el_ciclo_completo.py
Introdúcelos en este orden:
    1. SEMILLA                cambiar 52 por 16; nada más en el archivo se mueve

Ejecútalo:  python primer_ejemplo_07_optimo_local.py
Compáralo:  python primer_ejemplo_06_el_ciclo_completo.py

Misma receta, SEMILLA = 16 en lugar de 52. La corrida se asienta en
x = -4.417, f = +0.073 y nunca sale de esa colina local.
"""
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 16   # --- CAMBIADO --- la única diferencia con el paso 6
TAMANO_POBLACION = 10
PROB_CRUZA = 0.8
PROB_MUTACION = 0.1
MAX_GENERACIONES = 10
GEN_MIN, GEN_MAX = -10.0, 10.0
TAMANO_TORNEO, ALFA_MEZCLA = 3, 1.0
MUTACION_MU, MUTACION_SIGMA = 0.0, 1.0
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


def objetivo(x):
    """La función que queremos MAXIMIZAR.

    Argumentos:
        x: un real o un arreglo de numpy.

    Devuelve:
        sen(x) - 0.2 * |x|. Máximo global cerca de x = +1.372, f = +0.706.

    Ejemplo:
        Con SEMILLA = 16 el ciclo se queda en x = -4.417, f = +0.073.
    """
    return np.sin(x) - 0.2 * abs(x)


def acotar(g, minimo=GEN_MIN, maximo=GEN_MAX):
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


def seleccion_torneo(poblacion, tamano):
    """Un torneo por cada lugar de la nueva generación.

    Argumentos:
        poblacion: lista de Individuo.
        tamano: competidores por torneo (aquí 3).

    Devuelve:
        Lista de la misma longitud, hecha de copias.
    """
    return [max([random.choice(poblacion) for _ in range(tamano)],
                key=lambda i: i.aptitud) for _ in range(len(poblacion))]


def cruza(p1, p2):
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


def mutar(ind):
    """Suma ruido gaussiano a un gen y reconstruye el Individuo.

    Argumentos:
        ind: candidato a perturbar.

    Devuelve:
        Un Individuo nuevo. En esta corrida, sigma = 1.0 no produjo un escape.
    """
    return Individuo([acotar(ind.gen + random.gauss(MUTACION_MU, MUTACION_SIGMA))])


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

mejor = max(poblacion, key=lambda i: i.aptitud)
_malla = np.linspace(GEN_MIN, GEN_MAX, 400)
_x_verdadero = float(_malla[np.argmax(objetivo(_malla))])
print(f"\nSolución: {mejor}")
print(f"Referencia de malla: x={_x_verdadero:+.3f} f={objetivo(_x_verdadero):+.3f}   <- nunca la encontró")
print("\nLa corrida convergió en pocas generaciones y ahí dejó de mejorar.")
print("Todos los individuos están en la misma colina, la cruza tiene poco")
print("alcance y esta secuencia aleatoria de diez generaciones no produjo")
print("un escape.")

plt.figure(figsize=(8, 4))
plt.plot(range(len(historia)), historia, "o-", color="tab:green")
plt.title("Mejor aptitud por generación")
plt.xlabel("generación"); plt.ylabel("mejor f(x)")
plt.grid(True, linestyle=":", alpha=0.5)
FIGURAS.mkdir(exist_ok=True)
plt.savefig(FIGURAS / "primer_ejemplo_07_optimo_local.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Figura guardada en {FIGURAS}/primer_ejemplo_07_optimo_local.png")

