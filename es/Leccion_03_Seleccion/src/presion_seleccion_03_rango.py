"""
Lección 03 - Paso 3: Selección por rango, o presión que no depende de la escala
=============================================================================
NUEVO EN ESTE PASO: seleccion_rango().

El paso 2 dejó una incomodidad: la misma ruleta, en la misma población, aplica
una presión muy diferente dependiendo de un piso que inventamos. La selección
por rango descarta los valores de aptitud y mantiene solo su ORDEN. Así, los
anchos de sector dependen únicamente del tamaño de la población, por lo que la
presión es la misma independientemente de la escala de aptitud - y, en
particular, no se puede cambiar sumando una constante.

CAMBIOS RESPECTO A presion_seleccion_02_proporcional.py
Introdúcelos en este orden:
    1. seleccion_rango()         anchos de sector a partir solo del orden: sin valores, sin piso
    2. la fila de rango          mídela junto a los dos pisos del paso 2
    3. la figura de sectores     los tres conjuntos de anchos de sector, individuo por individuo

Ejecútalo:  python presion_seleccion_03_rango.py

El orden por sí solo fija la presión. El mejor individuo todavía se
pierde en el 13.6% de las muestras.
"""

import random
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 52
TAMANO_POBLACION = 10
GEN_MIN, GEN_MAX = -10.0, 10.0
PISO_APTITUD_ESTRICTO = 0.001    # el peor individuo conserva casi ningún sector
PISO_APTITUD_GENEROSO = 3.0      # el peor individuo conserva un sector ancho
MUESTRAS = 2000                  # extracciones por medición, para promedios estables
FIGURAS = Path(__file__).resolve().parent.parent / "figures"


def objetivo(x):
    """La función que queremos MAXIMIZAR.

    Args:
        x: un real o un arreglo de numpy.

    Returns:
        sen(x) - 0.2 * |x|.
    """
    return np.sin(x) - 0.2 * abs(x)


class Individuo:
    """Una solución candidata. El nombre hace legibles las tablas.

    Args:
        nombre: etiqueta A..J.
        lista_genes: cromosoma; aquí un solo real.
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
    """Diez individuos aleatorios, nombrados de la A a la J.

    random.seed() vive aquí para que cada experimento arranque igual.

    Args:
        tamano: cuántos individuos (aquí 10).

    Returns:
        Lista A..J, SEMILLA = 52. Diversidad 6.665.
    """
    random.seed(SEMILLA)
    genes = [random.uniform(GEN_MIN, GEN_MAX) for _ in range(tamano)]
    return [Individuo(chr(ord("A") + i), [g]) for i, g in enumerate(genes)]


def conteo_seleccion(poblacion: list[Individuo],
                     seleccionados: list[Individuo]) -> list[int]:
    """Cuántas copias de cada individuo produjo una selección.

    Contar por id() funciona porque la selección devuelve referencias.

    Args:
        poblacion: los N candidatos originales.
        seleccionados: las N copias que devolvió el operador.

    Returns:
        Lista de N enteros, en el orden de poblacion.
    """
    conteos = Counter(id(ind) for ind in seleccionados)
    return [conteos.get(id(ind), 0) for ind in poblacion]


def contar_extintos(copias: list[int]) -> int:
    """Individuos seleccionados cero veces.

    Args:
        copias: conteos por individuo.

    Returns:
        Cuántos cayeron en 0.
    """
    return sum(1 for c in copias if c == 0)


def dispersion_genetica(individuos: list[Individuo]) -> float:
    """Diversidad, como la desviación estándar de los genes.

    Args:
        individuos: población o generación seleccionada.

    Returns:
        Desviación estándar poblacional. Línea base: 6.665.
    """
    return statistics.pstdev([ind.gen for ind in individuos])


def imprimir_tabla_copias(poblacion: list[Individuo], copias: list[int]) -> None:
    """Quién sobrevivió, con qué frecuencia.

    Args:
        poblacion: candidatos nombrados A..J.
        copias: conteos alineados con poblacion.
    """
    for ind, n in zip(poblacion, copias):
        print(f"    {ind}   copias: {n}  {'#' * n}")


def seleccionar_nada(poblacion: list[Individuo]) -> list[Individuo]:
    """Línea base de cero presión: todos se reproducen una vez.

    Args:
        poblacion: los N candidatos.

    Returns:
        Copia superficial: mismas instancias, una de cada una.
    """
    return list(poblacion)


def desplazar_a_positivo(valores: list[float], piso: float) -> list[float]:
    """Mueve las aptitudes para que el peor caiga exactamente en piso.

    Args:
        valores: aptitudes crudas, aquí todas negativas.
        piso: valor del peor tras el desplazamiento.

    Returns:
        Pesos no negativos. Piso 0.001 vs 3.0 cambia 2.35 a 1.30 copias.
    """
    mas_bajo = min(valores)
    return [valor - mas_bajo + piso for valor in valores]


def seleccion_proporcional(poblacion: list[Individuo],
                           piso: float) -> list[Individuo]:
    """Ruleta proporcional a la aptitud desplazada.

    Args:
        poblacion: los N candidatos.
        piso: el parámetro oculto; no lo da el problema.

    Returns:
        N referencias. Piso 0.001 vs 3.0: 2.35 vs 1.30 copias del mejor.
    """
    pesos = desplazar_a_positivo([ind.aptitud for ind in poblacion], piso)
    total = sum(pesos)
    seleccionados: list[Individuo] = []
    for _ in range(len(poblacion)):
        punto = random.random() * total
        acumulado = 0.0
        for ind, peso in zip(poblacion, pesos):
            acumulado += peso
            if acumulado > punto:
                seleccionados.append(ind)
                break
    return seleccionados


Selector = Callable[[list[Individuo]], list[Individuo]]


def reporte_presion(etiqueta: str, selector: Selector,
                    poblacion: list[Individuo],
                    muestras: int = MUESTRAS) -> dict:
    """Ejecuta un selector `muestras` veces y resume la presión.

    Args:
        etiqueta: nombre de la fila.
        selector: operador que toma N y devuelve N.
        poblacion: los diez individuos fijos.
        muestras: extracciones (aquí 2000).

    Returns:
        Dict con copies, sd, extinct, lost, fitness, spread, best_copies.
    """
    mejor = max(poblacion, key=lambda ind: ind.aptitud)
    random.seed(SEMILLA)
    copias_mejor, extinciones, aptitudes, dispersiones = [], [], [], []
    for _ in range(muestras):
        seleccionados = selector(poblacion)
        copias = conteo_seleccion(poblacion, seleccionados)
        copias_mejor.append(copias[poblacion.index(mejor)])
        extinciones.append(contar_extintos(copias))
        aptitudes.append(statistics.mean(ind.aptitud for ind in seleccionados))
        dispersiones.append(dispersion_genetica(seleccionados))
    return {
        "label": etiqueta,
        "copies": statistics.mean(copias_mejor),
        "sd": statistics.pstdev(copias_mejor),
        "extinct": statistics.mean(extinciones),
        "lost": 100 * sum(1 for c in copias_mejor if c == 0) / muestras,
        "fitness": statistics.mean(aptitudes),
        "spread": statistics.mean(dispersiones),
        "best_copies": copias_mejor,
    }


def imprimir_encabezado_presion() -> None:
    """Encabezado de la tabla de comparación de los pasos 2 al 6."""
    print(f"{'método':26s} {'copias':>6} {'sd':>5} {'extint':>7} "
          f"{'perd':>6} {'f med':>8} {'dispers':>7}")
    print(f"{'':26s} {'del me':>6} {'':>5} {'de 10':>7} "
          f"{'mejor':>6} {'selecc':>8} {'genes':>7}")


def imprimir_fila_presion(fila: dict) -> None:
    """Una fila de la tabla de presión.

    Args:
        fila: dict devuelto por reporte_presion().
    """
    print(f"{fila['label']:26s} {fila['copies']:6.2f} {fila['sd']:5.2f} "
          f"{fila['extinct']:7.2f} {fila['lost']:5.1f}% {fila['fitness']:+8.3f} "
          f"{fila['spread']:7.3f}")


# --- NUEVO (1) seleccion_rango() ----------------------------------------------
def seleccion_rango(poblacion: list[Individuo]) -> list[Individuo]:
    """Selección por rango: los anchos de sector provienen del ORDEN, no de los valores.

    Ordena el mejor primero y dale a la posición i el ancho 1 - i/N. Los anchos nunca
    dependen de qué tan separados estén los valores de aptitud, así que un valor atípico
    no puede tragar la ruleta - y no se necesita desplazamiento ni piso en absoluto.

    Args:
        poblacion: los N candidatos.

    Returns:
        N referencias. El mejor se pierde aún en el 13.6% de las muestras.
    """
    ordenado = sorted(poblacion, key=lambda ind: -ind.aptitud)
    tamano = len(poblacion)
    pesos = [1 - i / tamano for i in range(tamano)]
    total = sum(pesos)
    seleccionados: list[Individuo] = []
    for _ in range(tamano):
        punto = random.random() * total
        acumulado = 0.0
        for ind, peso in zip(ordenado, pesos):
            acumulado += peso
            if acumulado > punto:
                seleccionados.append(ind)
                break
    return seleccionados
# ------------------------------------------------------------------------------


poblacion = crear_poblacion()
mejor = max(poblacion, key=lambda ind: ind.aptitud)

# --- NUEVO (2) la fila de rango -----------------------------------------------
print("Anchos de sector: la proporcional depende del piso, el rango no.")
for piso in (PISO_APTITUD_ESTRICTO, PISO_APTITUD_GENEROSO):
    pesos = desplazar_a_positivo([ind.aptitud for ind in poblacion], piso)
    proporciones = [100 * p / sum(pesos) for p in pesos]
    proporcion_mejor, proporcion_peor = proporciones[poblacion.index(mejor)], min(proporciones)
    print(f"  proporcional, piso {piso:<6} mejor {proporcion_mejor:5.2f}%   "
          f"peor {proporcion_peor:5.2f}%   proporción {proporcion_mejor / proporcion_peor:7.1f}")

pesos_rango = [1 - i / TAMANO_POBLACION for i in range(TAMANO_POBLACION)]
proporciones_rango = [100 * p / sum(pesos_rango) for p in pesos_rango]
print(f"  rango, sin piso en absoluto   mejor {proporciones_rango[0]:5.2f}%   "
      f"peor {proporciones_rango[-1]:5.2f}%   "
      f"proporción {proporciones_rango[0] / proporciones_rango[-1]:7.1f}")
print(f"\nEsa última proporción es exactamente el tamaño de la población, {TAMANO_POBLACION}: el")
print("mejor individuo obtiene el peso 1 y el peor obtiene el peso 1/N, cualesquiera")
print("que sean los valores de aptitud. La selección por rango tiene una presión, y está")
print("fijada por N en lugar de elegida por accidente.")

print("\nUna extracción por rango:")
seleccionados = seleccion_rango(poblacion)
copias = conteo_seleccion(poblacion, seleccionados)
imprimir_tabla_copias(poblacion, copias)
print(f"    extintos en esta única extracción: {contar_extintos(copias)} de "
      f"{TAMANO_POBLACION}   |   diversidad: {dispersion_genetica(seleccionados):.3f} "
      f"(era {dispersion_genetica(poblacion):.3f})")

print(f"\nMedido sobre {MUESTRAS} extracciones:")
filas = [
    reporte_presion("sin seleccion", seleccionar_nada, poblacion),
    reporte_presion(f"proporcional, piso {PISO_APTITUD_ESTRICTO}",
                    lambda p: seleccion_proporcional(p, PISO_APTITUD_ESTRICTO),
                    poblacion),
    reporte_presion(f"proporcional, piso {PISO_APTITUD_GENEROSO}",
                    lambda p: seleccion_proporcional(p, PISO_APTITUD_GENEROSO),
                    poblacion),
    reporte_presion("rango", seleccion_rango, poblacion),
]
imprimir_encabezado_presion()
for fila in filas:
    imprimir_fila_presion(fila)

clasificado_por_presion = sorted(filas[1:], key=lambda r: r["copies"])
print("\nPresión, de menor a mayor (copias esperadas del mejor individuo):")
print("  " + "  <  ".join(f"{r['label']} ({r['copies']:.2f})"
                          for r in clasificado_por_presion))
fila_rango = filas[3]
print(f"\nEl rango se sitúa entre los dos pisos que inventamos, que es el punto:")
print("es una cantidad definida de presión en lugar de una cantidad que se mueve")
print("cuando alguien reescala la función de aptitud. Lo que NO hace es")
print(f"proteger al mejor individuo - todavía se pierde en {fila_rango['lost']:.1f}% ")
print("de las extracciones. Ese es el paso 4.")
# ------------------------------------------------------------------------------

# --- NUEVO (3) la figura de sectores ------------------------------------------
ordenado = sorted(poblacion, key=lambda ind: -ind.aptitud)
etiquetas = [ind.nombre for ind in ordenado]
posiciones = np.arange(len(ordenado))

series = []
for piso in (PISO_APTITUD_ESTRICTO, PISO_APTITUD_GENEROSO):
    pesos = desplazar_a_positivo([ind.aptitud for ind in ordenado], piso)
    series.append((f"proporcional, piso {piso}",
                   [100 * p / sum(pesos) for p in pesos]))
series.append(("rango", proporciones_rango))

fig, ax = plt.subplots(figsize=(9, 4))
ancho = 0.27
for i, (etiqueta, proporciones) in enumerate(series):
    ax.bar(posiciones + (i - 1) * ancho, proporciones, ancho, label=etiqueta)
ax.set_xticks(posiciones)
ax.set_xticklabels(etiquetas)
ax.set_xlabel("individuo, el mejor a la izquierda")
ax.set_ylabel("% de la ruleta")
ax.set_title("Ancho de sector por individuo: el piso remodela la ruleta, "
             "el rango la fija")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.legend()

FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "presion_seleccion_03_rango.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/presion_seleccion_03_rango.png")
# ------------------------------------------------------------------------------

