"""
Lección 03 - Paso 2: Selección proporcional, y el piso que nadie menciona
=========================================================================
NUEVO EN ESTE PASO: desplazar_a_positivo(), seleccion_proporcional(), reporte_presion().

La imagen estándar de la selección es una ruleta cuyos sectores son tan anchos
como la aptitud de los individuos. En esta población la imagen no encaja: todas
las aptitudes son negativas, y un sector no puede tener un ancho negativo. La
reparación habitual es desplazar todos los valores hacia arriba hasta que el
peor sea no negativo - y esa reparación introduce silenciosamente un parámetro,
el piso, que resulta controlar la presión de selección más fuertemente que
los valores de aptitud.

CAMBIOS RESPECTO A presion_seleccion_01_la_poblacion.py
Introdúcelos en este orden:
    1. desplazar_a_positivo()    una ruleta necesita sectores no negativos; el piso es una elección libre
    2. seleccion_proporcional()  la ruleta en sí: un giro por ranura de la nueva generación
    3. reporte_presion()         la medición que reusa cada paso posterior: muchos giros, una fila
    4. los dos pisos             girar la misma ruleta bajo piso 0.001 y piso 3.0
    5. la figura de la ruleta    ambas ruletas desenrolladas, para poder comparar los anchos de sector

Ejecútalo:  python presion_seleccion_02_proporcional.py

El piso es la presión: el mejor espera 2.35 copias en el piso 0.001 pero
1.30 en el piso 3.0. Misma población, misma ruleta.
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
    """La función que queremos MAXIMIZAR - el paisaje de la Lección 01, sin cambios.

    Funciona con un solo flotante y con un arreglo de numpy, para que la misma línea dibuje la
    curva y evalúe a un individuo.

    Args:
        x: un real o un arreglo de numpy.

    Returns:
        sen(x) - 0.2 * |x|.
    """
    return np.sin(x) - 0.2 * abs(x)


class Individuo:
    """Una solución candidata. El nombre solo está ahí para hacer legibles las
    tablas de selección: cada método abajo reporta quién fue copiado, por nombre.

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
    """Diez individuos aleatorios, nombrados de la A a la J en orden de creación.

    random.seed() se llama aquí en lugar de a nivel de módulo para que cada
    experimento en esta lección comience con la misma población.

    Args:
        tamano: cuántos individuos (aquí 10).

    Returns:
        Lista A..J, SEMILLA = 52.
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
        Lista de N enteros, en el orden de `poblacion`.
    """
    conteos = Counter(id(ind) for ind in seleccionados)
    return [conteos.get(id(ind), 0) for ind in poblacion]


def contar_extintos(copias: list[int]) -> int:
    """Individuos seleccionados cero veces. Sus genes se pierden para siempre.

    Args:
        copias: conteos por individuo.

    Returns:
        Cuántos cayeron en 0.
    """
    return sum(1 for c in copias if c == 0)


def dispersion_genetica(individuos: list[Individuo]) -> float:
    """Diversidad, como la desviación estándar de los genes.

    Un número es rudimentario, pero es el número que importa: cuando llega a
    cero, la cruza ya no puede producir nada nuevo y la búsqueda termina.

    Args:
        individuos: población o generación seleccionada.

    Returns:
        Desviación estándar poblacional de los genes.
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
    """
    return list(poblacion)


# --- NUEVO (1) desplazar_a_positivo() -----------------------------------------
def desplazar_a_positivo(valores: list[float], piso: float) -> list[float]:
    """Mueve una lista de valores de aptitud para que el peor caiga exactamente en `piso`.

    Una ruleta no puede tener sectores negativos, y en esta población cada aptitud
    es negativa, por lo que algo como esto es inevitable. Lo que es fácil de perder de vista es
    que `piso` es un parámetro libre: el problema no lo proporciona, y el
    resto de este paso muestra que decide la presión de selección.

    Args:
        valores: aptitudes crudas, aquí todas negativas.
        piso: valor que se le asigna al peor tras el desplazamiento.

    Returns:
        Pesos no negativos, el mínimo igual a `piso`.

    Example:
        Piso 0.001: el mejor espera 2.35 copias. Piso 3.0: 1.30 copias.
    """
    mas_bajo = min(valores)
    return [valor - mas_bajo + piso for valor in valores]
# ------------------------------------------------------------------------------


# --- NUEVO (2) seleccion_proporcional() ---------------------------------------
def seleccion_proporcional(poblacion: list[Individuo],
                           piso: float) -> list[Individuo]:
    """Selección proporcional a la aptitud: la ruleta.

    Cada individuo posee un sector tan ancho como su aptitud desplazada. Un giro por
    ranura de la nueva generación: lanza un punto a la ruleta y recorre las
    anchuras acumuladas hasta pasarlo.

    Args:
        poblacion: los N candidatos.
        piso: el parámetro oculto; no lo da el problema.

    Returns:
        N referencias a individuos de `poblacion`.

    Example:
        Piso 0.001 vs 3.0: 2.35 vs 1.30 copias esperadas del mejor.
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
# ------------------------------------------------------------------------------


# --- NUEVO (3) reporte_presion() ----------------------------------------------
Selector = Callable[[list[Individuo]], list[Individuo]]


def reporte_presion(etiqueta: str, selector: Selector,
                    poblacion: list[Individuo],
                    muestras: int = MUESTRAS) -> dict:
    """Ejecuta un selector `muestras` veces y resume la presión que aplica.

    Cada método en esta lección se mide con esta única función, siempre desde
    la misma semilla, de modo que las filas de las tablas de comparación son comparables por
    construcción en lugar de por esperanza. Los conteos brutos por extracción se conservan porque
    el paso 5 trata sobre su dispersión, no su media.

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
    """Un encabezado para cada tabla de comparación en los pasos 2 al 6."""
    # Las columnas coinciden con las claves de reporte_presion(); no se reordenan.
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
# ------------------------------------------------------------------------------


# --- NUEVO (4) los dos pisos --------------------------------------------------
poblacion = crear_poblacion()
mejor = max(poblacion, key=lambda ind: ind.aptitud)

print("La ruleta se construye con la aptitud DESPLAZADA, y el desplazamiento necesita un piso.")
for piso in (PISO_APTITUD_ESTRICTO, PISO_APTITUD_GENEROSO):
    pesos = desplazar_a_positivo([ind.aptitud for ind in poblacion], piso)
    proporciones = [100 * p / sum(pesos) for p in pesos]
    proporcion_mejor = proporciones[poblacion.index(mejor)]
    print(f"\n  piso = {piso}:   anchos de sector, en % de la ruleta")
    for ind, proporcion in zip(poblacion, proporciones):
        print(f"    {ind}   sector: {proporcion:5.2f}%  {'=' * round(proporcion)}")
    print(f"    sector más ancho / sector más estrecho = "
          f"{max(proporciones) / min(proporciones):.1f}")
    print(f"    el mejor individuo, {mejor.nombre}, espera "
          f"{TAMANO_POBLACION * proporcion_mejor / 100:.2f} copias de sí mismo")

print(f"\nUn giro por ranura, con piso = {PISO_APTITUD_ESTRICTO}:")
seleccionados = seleccion_proporcional(poblacion, PISO_APTITUD_ESTRICTO)
copias = conteo_seleccion(poblacion, seleccionados)
imprimir_tabla_copias(poblacion, copias)
print(f"    extintos en esta única extracción: {contar_extintos(copias)} de "
      f"{TAMANO_POBLACION}   |   diversidad: {dispersion_genetica(seleccionados):.3f} "
      f"(era {dispersion_genetica(poblacion):.3f})")

print(f"\nLa misma ruleta, girada {MUESTRAS} veces bajo cada piso:")
filas = [
    reporte_presion("sin seleccion", seleccionar_nada, poblacion),
    reporte_presion(f"proporcional, piso {PISO_APTITUD_ESTRICTO}",
                    lambda p: seleccion_proporcional(p, PISO_APTITUD_ESTRICTO),
                    poblacion),
    reporte_presion(f"proporcional, piso {PISO_APTITUD_GENEROSO}",
                    lambda p: seleccion_proporcional(p, PISO_APTITUD_GENEROSO),
                    poblacion),
]
imprimir_encabezado_presion()
for fila in filas:
    imprimir_fila_presion(fila)

linea_base, estricto, generoso = filas
print("\nMismo método, misma población, misma semilla: solo se movió el piso, y")
print(f"  - las copias esperadas del mejor individuo pasaron de "
      f"{estricto['copies']:.2f} a {generoso['copies']:.2f}")
print(f"  - la probabilidad de perder al mejor individuo por completo pasó de "
      f"{estricto['lost']:.1f}% a {generoso['lost']:.1f}%")
print(f"  - la diversidad de la generación seleccionada pasó de "
      f"{estricto['spread']:.3f} a {generoso['spread']:.3f}, "
      f"frente a {linea_base['spread']:.3f} sin ninguna selección")
print("Nada en el problema dice qué piso es correcto. Esa es la verdadera")
print("debilidad de la selección proporcional: la presión que aplica depende de")
print("un desplazamiento arbitrario tanto como de la población. El paso 3 elimina el")
print("desplazamiento por completo.")
# ------------------------------------------------------------------------------

# --- NUEVO (5) la figura de la ruleta -----------------------------------------
fig, ejes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
for ax, piso in zip(ejes, (PISO_APTITUD_ESTRICTO, PISO_APTITUD_GENEROSO)):
    pesos = desplazar_a_positivo([ind.aptitud for ind in poblacion], piso)
    izquierda = 0.0
    for ind, peso in zip(poblacion, pesos):
        ancho = 100 * peso / sum(pesos)
        ax.barh([0], [ancho], left=[izquierda], height=0.6, edgecolor="white")
        if ancho > 4:
            ax.text(izquierda + ancho / 2, 0, ind.nombre,
                    ha="center", va="center", fontsize=9)
        izquierda += ancho
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_title(f"la ruleta desenrollada, piso = {piso}")
ejes[-1].set_xlabel("% de la ruleta")

FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "presion_seleccion_02_proporcional.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/presion_seleccion_02_proporcional.png")
# ------------------------------------------------------------------------------

