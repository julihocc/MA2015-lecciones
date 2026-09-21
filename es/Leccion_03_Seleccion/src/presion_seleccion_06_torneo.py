"""
Lección 03 - Paso 6: Selección por torneo, y los cinco métodos en una escala
==========================================================================
NUEVO EN ESTE PASO: seleccion_torneo(), y la comparación para la que era la lección.

La selección por torneo no necesita ruleta, ni ordenamiento, ni suma, ni piso: elige a unos
cuantos individuos al azar, quédate con el más apto, repite. Este es el método que usó
la Lección 01, y con la misma semilla este archivo reproduce la extracción de la Lección 01 exactamente. Su
verdadera virtud es la única cosa que ninguno de los otros cuatro ofrece: un solo número entero que
establece la presión de selección, desde ninguna en absoluto hasta casi total.

Con ese dial en mano, los cinco métodos por fin pueden ponerse en una sola escala - que
es el punto de la lección, y la última tabla de este paso.

CAMBIOS RESPECTO A presion_seleccion_05_sus.py
Introdúcelos en este orden:
    1. seleccion_torneo()         sin ruleta, sin ordenar, sin piso - solo comparaciones locales
    2. la extracción de Lección 01 la misma semilla y el mismo resultado que la Lección 01, paso 3
    3. el dial de presión         recorre el tamaño del torneo y observa subir la presión
    4. la comparación completa    cada método de los pasos 2 al 6 en una tabla, ordenados por presión
    5. la figura del dial         el dial, y la presión contra diversidad para cada método

Ejecútalo:  python presion_seleccion_06_torneo.py

Un entero fija la presión: k=3 da 2.74 copias, k=5 da 4.13, k=10 da
6.52. En las 13 configuraciones, presión vs calidad +0.95, vs diversidad -0.92.
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
TAMANOS_TORNEO = (1, 2, 3, 5, 10) # el dial de presión, de ninguna a casi total
TAMANO_TORNEO_LECCION_01 = 3      # el valor que usó la Lección 01
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


def seleccion_muestreo_universal_estocastico(poblacion: list[Individuo],
                                             piso: float) -> list[Individuo]:
    """Una extracción aleatoria, N punteros uniformemente espaciados.

    Args:
        poblacion: los N candidatos.
        piso: el mismo de la ruleta proporcional; SUS no lo elimina.

    Returns:
        N referencias. Expectativa 2.35 vs 1.31 según el piso.
    """
    ordenado = sorted(poblacion, key=lambda ind: -ind.aptitud)
    pesos = desplazar_a_positivo([ind.aptitud for ind in ordenado], piso)
    tamano = len(poblacion)
    espaciado = sum(pesos) / tamano
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


# --- NUEVO (1) seleccion_torneo() ---------------------------------------------
def seleccion_torneo(poblacion: list[Individuo],
                     tamano: int) -> list[Individuo]:
    """Selección por torneo: `tamano` candidatos aleatorios por ranura, el más apto gana.

    Sin ordenar, sin sumas, sin desplazamiento, sin piso - solo comparaciones locales, lo cual
    es por lo que la Lección 01 pudo usarlo antes de que existiera toda esta maquinaria y por qué
    la mayoría de las bibliotecas lo usan por defecto. `tamano` es el dial de presión: en el tamaño 1 es
    puro azar, en el tamaño N el mejor individuo gana casi todas las ranuras.

    Con reemplazo, población N y rango r=1 para el mejor, una ranura selecciona
    el rango r con probabilidad ((N-r+1)/N)^tamano - ((N-r)/N)^tamano. Estas
    probabilidades telescopan a uno; multiplicar por N da las copias esperadas.

    Args:
        poblacion: los N candidatos.
        tamano: competidores por torneo; el dial de presión.

    Returns:
        N referencias. k=3 da 2.74 copias del mejor; k=10 da 6.52.
    """
    seleccionados: list[Individuo] = []
    for _ in range(len(poblacion)):
        candidatos = [random.choice(poblacion) for _ in range(tamano)]
        seleccionados.append(max(candidatos, key=lambda ind: ind.aptitud))
    return seleccionados
# ------------------------------------------------------------------------------


poblacion = crear_poblacion()
mejor = max(poblacion, key=lambda ind: ind.aptitud)

# --- NUEVO (2) la extracción de Lección 01 ------------------------------------
print(f"Una extracción por torneo, tamaño {TAMANO_TORNEO_LECCION_01}, desde SEMILLA = "
      f"{SEMILLA} - la extracción que mostró la Lección 01:")
seleccionados = seleccion_torneo(poblacion, TAMANO_TORNEO_LECCION_01)
copias = conteo_seleccion(poblacion, seleccionados)
imprimir_tabla_copias(poblacion, copias)
ganador = poblacion[copias.index(max(copias))]
print(f"    {contar_extintos(copias)} de {TAMANO_POBLACION} individuos se "
      f"extinguieron, y {ganador.nombre} (x={ganador.gen:+.3f}) tomó "
      f"{max(copias)} de las {TAMANO_POBLACION} ranuras.")
print("    Esos son los mismos números que imprimió el paso 3 de la Lección 01, porque es")
print("    la misma población, la misma semilla y el mismo operador. Esta")
print("    lección está midiendo lo que esa lección solo mostró una vez.")
# ------------------------------------------------------------------------------

# --- NUEVO (3) el dial de presión ---------------------------------------------
print(f"\nEl dial: tamaño de torneo contra presión, sobre {MUESTRAS} extracciones cada uno:")
dial = [reporte_presion(f"torneo k={k}",
                        lambda p, k=k: seleccion_torneo(p, k), poblacion)
        for k in TAMANOS_TORNEO]
imprimir_encabezado_presion()
for fila in dial:
    imprimir_fila_presion(fila)

print(f"\nk = {TAMANOS_TORNEO[0]} no es selección en absoluto: cada ranura es una")
print(f"sola elección aleatoria, por lo que el mejor individuo espera "
      f"{dial[0]['copies']:.2f} copias, lo mismo que sin selección, y se")
print(f"pierde en el {dial[0]['lost']:.1f}% de las extracciones. En k = "
      f"{TAMANOS_TORNEO[-1]} el mejor individuo toma "
      f"{dial[-1]['copies']:.2f} de las {TAMANO_POBLACION} ranuras y")
print(f"{dial[-1]['extinct']:.2f} individuos de {TAMANO_POBLACION} mueren por "
      f"extracción. Un número entero abarca todo el rango.")
# ------------------------------------------------------------------------------

# --- NUEVO (4) la comparación completa ----------------------------------------
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
] + dial

# ¿Qué tan ancha es la banda que el piso por sí solo abre, en copias del mejor?
# Calculado a partir de las dos filas proporcionales anteriores, antes de que se ordene la tabla,
# para que la oración impresa al final no pueda desviarse de la tabla.
_proporcional = [fila for fila in filas if fila["label"].startswith("proporcional")]
rango_por_piso = abs(_proporcional[0]["copies"] - _proporcional[1]["copies"])

print(f"\nLos cinco métodos, {MUESTRAS} extracciones cada uno, ordenados por presión:")
filas.sort(key=lambda fila: fila["copies"])
imprimir_encabezado_presion()
for fila in filas:
    imprimir_fila_presion(fila)

presion = [fila["copies"] for fila in filas]
diversidad = [fila["spread"] for fila in filas]
calidad = [fila["fitness"] for fila in filas]
print(f"\nA través de esas {len(filas)} configuraciones:")
print(f"    presión contra calidad     (aptitud media): "
      f"correlación {statistics.correlation(presion, calidad):+.2f}")
print(f"    presión contra diversidad  (dispersión gen): "
      f"correlación {statistics.correlation(presion, diversidad):+.2f}")
print("Esos dos números son la lección. No hay método aquí que compre")
print("aptitud sin pagar en diversidad, y no hay método que mantenga")
print("diversidad sin renunciar a aptitud. En lo que difieren los cinco métodos")
print("no es en el intercambio - es en cuánto de él obtienes por accidente:")
print(f"    proporcional intercambia a una tasa fijada por el piso "
      f"({rango_por_piso:.2f} copias del mejor, para la misma población)")
print("    rango intercambia a una tasa fijada por el tamaño de la población, y solo eso")
print("    elitismo elimina el riesgo de ir hacia atrás, y añade presión")
print("    SUS elimina el ruido de muestreo sin tocar la tasa")
print("    torneo pone la tasa en un número entero que tú eliges a propósito")
# ------------------------------------------------------------------------------

# --- NUEVO (5) la figura del dial ---------------------------------------------
fig, (ax_dial, ax_scatter) = plt.subplots(1, 2, figsize=(12, 4.5))

ax_dial.plot(TAMANOS_TORNEO, [fila["copies"] for fila in dial], "o-",
             color="tab:blue", label="copias del mejor individuo")
ax_dial.plot(TAMANOS_TORNEO, [fila["extinct"] for fila in dial], "s--",
             color="tab:red", label="individuos que se extinguen")
ax_dial.set_xlabel("tamaño de torneo k")
ax_dial.set_ylabel(f"de un total de {TAMANO_POBLACION}")
ax_dial.set_title("el dial: un entero de ninguna presión a casi total")
ax_dial.grid(True, linestyle=":", alpha=0.5)
ax_dial.legend()

ax_scatter.scatter(presion, diversidad, color="tab:purple", zorder=3)
for fila in filas:
    ax_scatter.annotate(fila["label"], (fila["copies"], fila["spread"]),
                        textcoords="offset points", xytext=(5, 4), fontsize=7)
ax_scatter.axhline(dispersion_genetica(poblacion), color="grey", linestyle="--",
                   label=f"sin seleccion ({dispersion_genetica(poblacion):.2f})")
ax_scatter.set_xlabel("presión: copias esperadas del mejor individuo")
ax_scatter.set_ylabel("diversidad: desviación estándar de los genes")
ax_scatter.set_title("cada método paga la presión con diversidad")
ax_scatter.grid(True, linestyle=":", alpha=0.5)
ax_scatter.legend()

FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "presion_seleccion_06_torneo.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/presion_seleccion_06_torneo.png")
# ------------------------------------------------------------------------------

