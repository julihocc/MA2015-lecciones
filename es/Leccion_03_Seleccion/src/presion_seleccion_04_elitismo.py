"""
Lección 03 - Paso 4: Elitismo, la garantía más barata del curso
=================================================================
NUEVO EN ESTE PASO: seleccion_rango_con_elite().

Los pasos 2 y 3 comparten un defecto fácil de pasar por alto: el mejor individuo
se extrae al azar como todos los demás, por lo que a veces no se extrae en absoluto. Una
generación puede ser estrictamente peor que su generación padre. El elitismo arregla eso
copiando a los mejores individuos en la nueva generación antes de que ocurra cualquier
extracción. Es una línea de código, elimina la regresión por completo, y
cuesta diversidad - las tres cosas las mide este paso.

CAMBIOS RESPECTO A presion_seleccion_03_rango.py
Introdúcelos en este orden:
    1. seleccion_rango_con_elite()   los individuos superiores se saltan el sorteo por completo
    2. las filas de élite            lo que vale la garantía y lo que cuesta
    3. la figura de elitismo         probabilidad de perder al mejor, y diversidad, por método

Ejecútalo:  python presion_seleccion_04_elitism.py

Una línea reduce la pérdida del mejor de 13.6% a 0.0%, sube la presión
de 1.80 a 2.64 copias y cuesta diversidad (5.448 a 5.210).
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


# --- NUEVO (1) seleccion_rango_con_elite() ------------------------------------
def seleccion_rango_con_elite(poblacion: list[Individuo],
                              tamano_elite: int) -> list[Individuo]:
    """Selección por rango con los `tamano_elite` mejores individuos copiados directamente.

    Los lugares de élite se llenan antes de que ocurra cualquier extracción aleatoria, por lo que el mejor
    individuo no se puede perder. Los N - tamano_elite lugares restantes son selección
    por rango ordinaria, por lo que el cuerpo repite el bucle de seleccion_rango().

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
# ------------------------------------------------------------------------------


poblacion = crear_poblacion()
mejor = max(poblacion, key=lambda ind: ind.aptitud)

# --- NUEVO (2) las filas de élite ---------------------------------------------
print("Una extracción por rango con un lugar de élite:")
seleccionados = seleccion_rango_con_elite(poblacion, TAMANOS_ELITE[0])
copias = conteo_seleccion(poblacion, seleccionados)
imprimir_tabla_copias(poblacion, copias)
print(f"    el lugar de élite es {mejor.nombre}, el mejor individuo, y se "
      f"llena antes de cualquier sorteo")

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
for tamano_elite in TAMANOS_ELITE:
    filas.append(reporte_presion(
        f"rango + elite({tamano_elite})",
        lambda p, e=tamano_elite: seleccion_rango_con_elite(p, e), poblacion))
imprimir_encabezado_presion()
for fila in filas:
    imprimir_fila_presion(fila)

fila_rango = filas[3]
filas_elite = filas[4:]
print(f"\nLa garantía, verificada sobre {MUESTRAS} extracciones. Para cada extracción, toma la")
print("mejor aptitud presente en la generación que produjo, y quédate con la peor")
print("de esas - la generación con más mala suerte que el método puede darte:")
for etiqueta, selector in [("rango", seleccion_rango)] + [
        (f"rango + elite({e})", lambda p, e=e: seleccion_rango_con_elite(p, e))
        for e in TAMANOS_ELITE]:
    random.seed(SEMILLA)
    mas_mala_suerte = min(max(ind.aptitud for ind in selector(poblacion))
                          for _ in range(MUESTRAS))
    print(f"    {etiqueta:20s} mejor de la peor generación: {mas_mala_suerte:+.3f}")
print(f"    el mejor de la población propia es {mejor.aptitud:+.3f}")
print("Con un lugar de élite los dos números son idénticos, en cada extracción: la")
print("mejor aptitud de la población nunca puede volver a bajar. Sin él puede,")
print(f"y lo hace, en el {fila_rango['lost']:.1f}% de las extracciones.")

print("\nLo que cuesta:")
for fila in filas_elite:
    print(f"    {fila['label']}: las copias esperadas del mejor individuo suben "
          f"{fila_rango['copies']:.2f} -> {fila['copies']:.2f}, "
          f"extinciones {fila_rango['extinct']:.2f} -> {fila['extinct']:.2f}, "
          f"diversidad {fila_rango['spread']:.3f} -> {fila['spread']:.3f}")
print("Así que el elitismo no es gratis y no es solo una red de seguridad: también eleva")
print("la presión, porque los lugares de élite son lugares que el resto no puede ganar.")
print("En esta población la diversidad que cuesta es pequeña, pero el mecanismo")
print("es el que termina toda ejecución: la Lección 06 mide a dónde lleva esto.")
# ------------------------------------------------------------------------------

# --- NUEVO (3) la figura de elitismo ------------------------------------------
etiquetas = [fila["label"] for fila in filas]
posiciones = np.arange(len(filas))

fig, (ax_lost, ax_div) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_lost.bar(posiciones, [fila["lost"] for fila in filas], color="tab:red")
ax_lost.set_xticks(posiciones)
ax_lost.set_xticklabels(etiquetas, rotation=30, ha="right", fontsize=8)
ax_lost.set_ylabel("% de extracciones")
ax_lost.set_title("extracciones que pierden al mejor individuo")
ax_lost.grid(True, axis="y", linestyle=":", alpha=0.5)

ax_div.bar(posiciones, [fila["spread"] for fila in filas], color="tab:blue")
ax_div.axhline(dispersion_genetica(poblacion), color="grey", linestyle="--",
               label=f"sin seleccion ({dispersion_genetica(poblacion):.2f})")
ax_div.set_xticks(posiciones)
ax_div.set_xticklabels(etiquetas, rotation=30, ha="right", fontsize=8)
ax_div.set_ylabel("desviación estándar de los genes")
ax_div.set_title("diversidad de la generación seleccionada")
ax_div.grid(True, axis="y", linestyle=":", alpha=0.5)
ax_div.legend()

FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "presion_seleccion_04_elitismo.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/presion_seleccion_04_elitismo.png")
# ------------------------------------------------------------------------------

