"""
Lección 03 - Paso 5: Muestreo universal estocástico, o la misma ruleta girada una vez
=====================================================================================
NUEVO EN ESTE PASO: seleccion_muestreo_universal_estocastico().

La selección proporcional gira la ruleta N veces, por lo que la suerte se acumula: un
individuo que posea un cuarto de la ruleta puede terminar sin nada, y
otro puede ganar cuatro lugares seguidos. SUS mantiene exactamente la misma ruleta y el
mismo número esperado de copias, pero toma las N muestras de una vez, con punteros
espaciados uniformemente. La expectativa no se mueve. La dispersión colapsa.

CAMBIOS RESPECTO A presion_seleccion_04_elitismo.py
Introdúcelos en este orden:
    1. seleccion_muestreo_universal_estocastico()   una extracción aleatoria, N punteros uniformemente espaciados
    2. las filas de SUS                             misma expectativa que la ruleta, mucha menos dispersión
    3. la figura de dispersión                      cuántas copias del mejor, ruleta contra SUS

Ejecútalo:  python presion_seleccion_05_sus.py

Misma ruleta, misma expectativa, la dispersión colapsa. Y no arregla
el piso: 2.35 vs 1.31 copias, la misma división del paso 2.
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
TAMANOS_ELITE = (1, 2)           # cuántos individuos se saltan el sorteo
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


def seleccion_rango(poblacion: list[Individuo]) -> list[Individuo]:
    """Anchos de sector a partir solo del orden: sin valores, sin piso.

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


def seleccion_rango_con_elite(poblacion: list[Individuo],
                              tamano_elite: int) -> list[Individuo]:
    """Los mejores se copian antes del sorteo, así no se pueden perder.

    Args:
        poblacion: los N candidatos.
        tamano_elite: cuántos mejores se copian antes del sorteo.

    Returns:
        N referencias. Con elite(1) la pérdida del mejor cae de 13.6% a 0.0%.
    """
    ordenado = sorted(poblacion, key=lambda ind: -ind.aptitud)
    tamano = len(poblacion)
    pesos = [1 - i / tamano for i in range(tamano)]
    total = sum(pesos)
    seleccionados: list[Individuo] = list(ordenado[:tamano_elite])
    for _ in range(tamano - tamano_elite):
        punto = random.random() * total
        acumulado = 0.0
        for ind, peso in zip(ordenado, pesos):
            acumulado += peso
            if acumulado > punto:
                seleccionados.append(ind)
                break
    return seleccionados


# --- NUEVO (1) seleccion_muestreo_universal_estocastico() ---------------------
def seleccion_muestreo_universal_estocastico(poblacion: list[Individuo],
                                             piso: float) -> list[Individuo]:
    """SUS: una extracción aleatoria, luego N punteros espaciados uniformemente alrededor de la ruleta.

    La ruleta es exactamente la de seleccion_proporcional(). La diferencia es que
    los punteros no pueden agruparse: un sector más ancho que el espacio tiene
    garantizada al menos una copia, y nadie puede tener suerte dos veces seguidas.

    Args:
        poblacion: los N candidatos.
        piso: el mismo de la ruleta proporcional; SUS no lo elimina.

    Returns:
        N referencias. Expectativa 2.35 vs 1.31 según el piso; menor dispersión.
    """
    ordenado = sorted(poblacion, key=lambda ind: -ind.aptitud)
    pesos = desplazar_a_positivo([ind.aptitud for ind in ordenado], piso)
    tamano = len(poblacion)
    espaciado = sum(pesos) / tamano
    # Un solo dado; los demás punteros van a paso fijo. Por eso no se agrupan.
    inicio = random.uniform(0, espaciado)
    seleccionados: list[Individuo] = []
    indice = 0
    acumulado = pesos[0]
    for ranura in range(tamano):
        puntero = inicio + ranura * espaciado
        while acumulado < puntero and indice < tamano - 1:
            indice += 1
            acumulado += pesos[indice]
        seleccionados.append(ordenado[indice])
    return seleccionados
# ------------------------------------------------------------------------------


poblacion = crear_poblacion()
mejor = max(poblacion, key=lambda ind: ind.aptitud)

# --- NUEVO (2) las filas de SUS -----------------------------------------------
print(f"Una extracción SUS, piso = {PISO_APTITUD_ESTRICTO}:")
seleccionados = seleccion_muestreo_universal_estocastico(poblacion,
                                               PISO_APTITUD_ESTRICTO)
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
    reporte_presion(f"rango + elite({TAMANOS_ELITE[0]})",
                    lambda p: seleccion_rango_con_elite(p, TAMANOS_ELITE[0]),
                    poblacion),
    reporte_presion(f"rango + elite({TAMANOS_ELITE[1]})",
                    lambda p: seleccion_rango_con_elite(p, TAMANOS_ELITE[1]),
                    poblacion),
    reporte_presion(f"SUS, piso {PISO_APTITUD_ESTRICTO}",
                    lambda p: seleccion_muestreo_universal_estocastico(
                        p, PISO_APTITUD_ESTRICTO), poblacion),
    reporte_presion(f"SUS, piso {PISO_APTITUD_GENEROSO}",
                    lambda p: seleccion_muestreo_universal_estocastico(
                        p, PISO_APTITUD_GENEROSO), poblacion),
]
imprimir_encabezado_presion()
for fila in filas:
    imprimir_fila_presion(fila)

por_etiqueta = {fila["label"]: fila for fila in filas}
ruleta = por_etiqueta[f"proporcional, piso {PISO_APTITUD_ESTRICTO}"]
sus = por_etiqueta[f"SUS, piso {PISO_APTITUD_ESTRICTO}"]

print(f"\nLa misma ruleta, piso {PISO_APTITUD_ESTRICTO}, muestreada de dos formas:")
print(f"    copias esperadas del mejor    {ruleta['copies']:.2f} vs "
      f"{sus['copies']:.2f}   (difieren en "
      f"{abs(ruleta['copies'] - sus['copies']):.2f})")
print(f"    dispersión de ese conteo (sd) {ruleta['sd']:.2f} vs {sus['sd']:.2f}"
      f"   (SUS mantiene el {100 * sus['sd'] / ruleta['sd']:.0f}% de ella)")
print(f"    extracciones que pierden al mejor {ruleta['lost']:.1f}% vs "
      f"{sus['lost']:.1f}%")
print(f"    extintos por extracción           {ruleta['extinct']:.2f} vs "
      f"{sus['extinct']:.2f}")
print("Esta es la afirmación completa sobre SUS, y vale la pena hacerla con precisión: no")
print("selecciona mejores individuos, selecciona a los MISMOS individuos")
print("de forma más confiable. El ruido de muestreo no es presión de selección, y eliminarlo")
print("es gratis.")

sus_generoso = por_etiqueta[f"SUS, piso {PISO_APTITUD_GENEROSO}"]
print(f"\nLo que SUS no arregla: el piso. Las copias esperadas del mejor son")
print(f"{sus['copies']:.2f} en el piso {PISO_APTITUD_ESTRICTO} y "
      f"{sus_generoso['copies']:.2f} en el piso {PISO_APTITUD_GENEROSO}, la "
      f"misma división que encontró el paso 2.")
print("SUS repara cómo se muestrea la ruleta, no cómo se construye la ruleta.")
# ------------------------------------------------------------------------------

# --- NUEVO (3) la figura de dispersión ----------------------------------------
fig, (ax_wheel, ax_hist) = plt.subplots(1, 2, figsize=(12, 4.5))

pesos = desplazar_a_positivo(
    [ind.aptitud for ind in sorted(poblacion, key=lambda i: -i.aptitud)],
    PISO_APTITUD_ESTRICTO)
ordenado = sorted(poblacion, key=lambda ind: -ind.aptitud)
izquierda = 0.0
for ind, peso in zip(ordenado, pesos):
    ancho = 100 * peso / sum(pesos)
    ax_wheel.barh([0], [ancho], left=[izquierda], height=0.6, edgecolor="white")
    if ancho > 4:
        ax_wheel.text(izquierda + ancho / 2, 0, ind.nombre, ha="center",
                      va="center", fontsize=9)
    izquierda += ancho
espaciado = 100 / TAMANO_POBLACION
random.seed(SEMILLA)
inicio = random.uniform(0, espaciado)
for ranura in range(TAMANO_POBLACION):
    ax_wheel.axvline(inicio + ranura * espaciado, color="black", linewidth=2)
ax_wheel.set_yticks([])
ax_wheel.set_xlim(0, 100)
ax_wheel.set_xlabel("% de la ruleta")
ax_wheel.set_title(f"SUS: una extracción fija los {TAMANO_POBLACION} punteros")

rango_copias = range(0, max(max(ruleta["best_copies"]), max(sus["best_copies"])) + 1)
ancho = 0.4
ax_hist.bar([c - ancho / 2 for c in rango_copias],
            [100 * ruleta["best_copies"].count(c) / MUESTRAS for c in rango_copias],
            ancho, label=f"proporcional ({ruleta['sd']:.2f} sd)")
ax_hist.bar([c + ancho / 2 for c in rango_copias],
            [100 * sus["best_copies"].count(c) / MUESTRAS for c in rango_copias],
            ancho, label=f"SUS ({sus['sd']:.2f} sd)")
ax_hist.axvline(ruleta["copies"], color="grey", linestyle="--",
                label=f"esperadas {ruleta['copies']:.2f} copias")
ax_hist.set_xticks(list(rango_copias))
ax_hist.set_xlabel(f"copias del mejor individuo, {mejor.nombre}, en una extracción")
ax_hist.set_ylabel(f"% de {MUESTRAS} extracciones")
ax_hist.set_title("misma expectativa, diferente dispersión")
ax_hist.grid(True, axis="y", linestyle=":", alpha=0.5)
ax_hist.legend()

FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "presion_seleccion_05_sus.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/presion_seleccion_05_sus.png")
# ------------------------------------------------------------------------------

