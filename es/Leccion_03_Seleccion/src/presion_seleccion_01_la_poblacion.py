"""
Lección 03 - Paso 1: La población, y cómo medir lo que hace la selección
===========================================================================
La selección nunca inventa nada. Toma una población de N y devuelve una
población de N, construida solo con copias de lo que ya estaba allí. Así que lo único
que vale la pena medir es la CONTABILIDAD: quién fue copiado, cuántas veces y
quién desapareció.

Este paso no construye más que ese aparato de medición, y lo aplica a la
selección más barata posible - la que copia a todos exactamente una vez. Cada
método en los pasos 2 al 6 se mide frente a esta línea base.

La población es la misma de diez individuos de la Lección 01 (mismo objetivo, misma
SEMILLA = 52), de modo que los números continúan en lugar de reiniciar.

Ejecútalo:  python presion_seleccion_01_la_poblacion.py

Copiar a todos una vez: 0 extintos, diversidad 6.665, 1 copia del mejor.
La aptitud es negativa casi en todas partes, así que aún no hay ruleta.
"""
import random
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 52
TAMANO_POBLACION = 10
GEN_MIN, GEN_MAX = -10.0, 10.0
FIGURAS = Path(__file__).resolve().parent.parent / "figures"


def objetivo(x):
    """La función que queremos MAXIMIZAR - el paisaje de la Lección 01, sin cambios.

    Funciona con un solo flotante y con un arreglo de numpy, para que la misma línea dibuje la
    curva y evalúe a un individuo.

    Args:
        x: un real o un arreglo de numpy.

    Returns:
        sen(x) - 0.2 * |x|. Máximo global cerca de x = +1.372, f = +0.706.
    """
    return np.sin(x) - 0.2 * abs(x)


class Individuo:
    """Una solución candidata. El nombre solo está ahí para hacer legibles las
    tablas de selección: cada método abajo reporta quién fue copiado, por nombre.

    Args:
        nombre: etiqueta A..J, en orden de creación.
        lista_genes: cromosoma; aquí un solo real.

    Example:
        Con SEMILLA = 52, diez individuos; diversidad 6.665.
    """

    def __init__(self, nombre: str, lista_genes: list[float]) -> None:
        self.nombre = nombre
        self.lista_genes = lista_genes
        self.aptitud = float(objetivo(lista_genes[0]))

    @property
    def gen(self) -> float:
        """El único gen: este paisaje es unidimensional."""
        return self.lista_genes[0]

    def __repr__(self) -> str:
        return f"{self.nombre}  x={self.gen:+.3f}  f={self.aptitud:+.3f}"


def crear_poblacion(tamano: int = TAMANO_POBLACION) -> list[Individuo]:
    """Diez individuos aleatorios, nombrados de la A a la J en orden de creación.

    random.seed() se llama aquí en lugar de a nivel de módulo para que cada
    experimento en esta lección comience con la misma población.

    Args:
        tamano: cuántos individuos (aquí 10).

    Returns:
        Lista A..J, SEMILLA = 52.

    Example:
        Diversidad 6.665; el mejor espera 1 copia bajo "no hacer nada".
    """
    random.seed(SEMILLA)
    genes = [random.uniform(GEN_MIN, GEN_MAX) for _ in range(tamano)]
    return [Individuo(chr(ord("A") + i), [g]) for i, g in enumerate(genes)]


def conteo_seleccion(poblacion: list[Individuo],
                     seleccionados: list[Individuo]) -> list[int]:
    """Cuántas copias de cada individuo produjo una selección.

    Contar por id() funciona porque la selección devuelve referencias a los mismos
    objetos que se le dieron - no copia genes, solo apunta a individuos.

    Args:
        poblacion: los N candidatos originales.
        seleccionados: las N copias que devolvió el operador.

    Returns:
        Lista de N enteros: copias de cada individuo, en el orden de `poblacion`.
    """
    conteos = Counter(id(ind) for ind in seleccionados)
    return [conteos.get(id(ind), 0) for ind in poblacion]


def contar_extintos(copias: list[int]) -> int:
    """Individuos seleccionados cero veces. Sus genes se pierden para siempre.

    Args:
        copias: conteos por individuo, en el orden de la población.

    Returns:
        Cuántos cayeron en 0. En esta línea base: 0 de 10.
    """
    return sum(1 for c in copias if c == 0)


def dispersion_genetica(individuos: list[Individuo]) -> float:
    """Diversidad, como la desviación estándar de los genes.

    Un número es rudimentario, pero es el número que importa: cuando llega a
    cero, la cruza ya no puede producir nada nuevo y la búsqueda termina.

    Args:
        individuos: población o generación seleccionada.

    Returns:
        Desviación estándar poblacional de los genes. Aquí 6.665.
    """
    return statistics.pstdev([ind.gen for ind in individuos])


def imprimir_tabla_copias(poblacion: list[Individuo], copias: list[int]) -> None:
    """La tabla que imprime cada paso de esta lección: quién sobrevivió, con qué frecuencia.

    Args:
        poblacion: candidatos nombrados A..J.
        copias: conteos alineados con `poblacion`.
    """
    for ind, n in zip(poblacion, copias):
        print(f"    {ind}   copias: {n}  {'#' * n}")


def seleccionar_nada(poblacion: list[Individuo]) -> list[Individuo]:
    """La línea base de cero presión: todos se reproducen exactamente una vez.

    Este no es un operador útil - es la vara de medir. Un método de selección es
    interesante exactamente en la medida en que sus números difieren de estos.

    Args:
        poblacion: los N candidatos.

    Returns:
        Una copia superficial: mismas instancias, una de cada una.

    Example:
        0 extintos, diversidad 6.665, 1 copia del mejor.
    """
    return list(poblacion)


poblacion = crear_poblacion()

print("La población (ordenada por aptitud, el mejor primero):")
for ind in sorted(poblacion, key=lambda i: -i.aptitud):
    print("   ", ind)

mejor = max(poblacion, key=lambda i: i.aptitud)
peor = min(poblacion, key=lambda i: i.aptitud)
print(f"\nEl mejor es {mejor.nombre} en f={mejor.aptitud:+.3f}, "
      f"el peor es {peor.nombre} en f={peor.aptitud:+.3f}.")
print(f"Diversidad (desviación estándar de los genes): {dispersion_genetica(poblacion):.3f}")

print("\nLa línea base - selección que copia a todos una vez:")
copias = conteo_seleccion(poblacion, seleccionar_nada(poblacion))
imprimir_tabla_copias(poblacion, copias)
print(f"    extintos: {contar_extintos(copias)} de {TAMANO_POBLACION}"
      f"   |   diversidad: {dispersion_genetica(seleccionar_nada(poblacion)):.3f}"
      f"   |   copias del mejor: {copias[poblacion.index(mejor)]}")
print("No pasa nada, que es el punto: sin presión, sin pérdida.")

# Un hecho sobre esta población decide todo el paso 2.
no_negativos = [ind for ind in poblacion if ind.aptitud >= 0]
print(f"\nNota: {len(no_negativos)} de {TAMANO_POBLACION} individuos tienen "
      f"aptitud >= 0.")
print("Todas las aptitudes aquí son negativas, porque f(x) = sin(x) - 0.2|x| es")
print("negativa en casi todas partes en [-10, +10]. No se puede construir una ruleta")
print("a partir de anchos de sector negativos, por lo que el primer método en el paso 2 no")
print("funciona en esta población hasta que hagamos algo al respecto.")

fig, (ax_curve, ax_bars) = plt.subplots(1, 2, figsize=(11, 4))

x = np.linspace(GEN_MIN, GEN_MAX, 400)
ax_curve.plot(x, objetivo(x), color="tab:blue", linewidth=1.2)
ax_curve.scatter([i.gen for i in poblacion], [i.aptitud for i in poblacion],
                 color="tab:red", zorder=3)
for ind in poblacion:
    ax_curve.annotate(ind.nombre, (ind.gen, ind.aptitud),
                      textcoords="offset points", xytext=(4, 5), fontsize=9)
ax_curve.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
ax_curve.set_title("Los diez individuos en el paisaje de la Lección 01")
ax_curve.set_xlabel("x")
ax_curve.set_ylabel("f(x)")
ax_curve.grid(True, linestyle=":", alpha=0.5)

ordenado = sorted(poblacion, key=lambda i: i.aptitud)
ax_bars.barh([i.nombre for i in ordenado], [i.aptitud for i in ordenado],
             color="tab:red")
ax_bars.axvline(0.0, color="grey", linewidth=0.8, linestyle="--")
ax_bars.set_title("Aptitud bruta: todos los valores son negativos")
ax_bars.set_xlabel("f(x)")
ax_bars.grid(True, axis="x", linestyle=":", alpha=0.5)

FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "presion_seleccion_01_la_poblacion.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/presion_seleccion_01_la_poblacion.png")

