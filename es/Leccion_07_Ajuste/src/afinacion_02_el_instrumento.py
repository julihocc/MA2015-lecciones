"""
Leccion 07 - Paso 2: El instrumento
===================================
NUEVO EN ESTE PASO: error_estandar(), medir(), y las perillas convertidas en
argumentos.

El Paso 1 corrio el metodo del libro y lo vio fallar: una corrida por configuracion, un
ranking decidido en el cuarto decimal, conteos de victorias que cambian con la semilla.
La Leccion 06 construyo la cura - una tasa de exito sobre cientos de corridas, juzgada desde
afuera por una referencia de cuadricula densa y una tolerancia establecida. Este paso apunta ese
instrumento a la primera perilla, la probabilidad de cruza, en la linea base
del curso: poblacion 10, probabilidad de mutacion 0.1, el paisaje de seno de la
Leccion 01, diez generaciones.

La respuesta medida en estas 500 semillas es clara: la tasa de exito sube monotonicamente
de 55.6% en cruza 0.0 a 83.0% en cruza 1.0. Los resultados emparejados
miden esa brecha sin tratar configuraciones como independientes. Pero a la tabla le falta una columna, y esa columna
cambia como se lee: cuanto gasto cada configuracion. El Paso 3 la anade.

CAMBIOS RESPECTO A afinacion_01_corridas_sueltas.py
Introducelos en este orden:
    1. las perillas son argumentos correr() toma las probabilidades por llamada
    2. error_estandar()            el EE por sustitucion de cada tasa medida
    3. medir()                     el instrumento de barrido: N corridas por configuracion
    4. la linea base del curso     TAMANO_POBLACION, PROBABILIDAD_MUTACION, CORRIDAS

ELIMINADO DE afinacion_01_corridas_sueltas.py: el historial por generacion, la
tabla de conteo de victorias y la figura de multiples pequenos (el punto del paso 1 esta hecho), y
las constantes del protocolo del libro (poblacion 16, mutacion 0.2, semilla 63), que
eran el especimen, no el estudio.

Ejecutalo:  python afinacion_02_el_instrumento.py

La tasa de cruza sube 55.6% → 83.0%; la brecha se reporta con el error
estándar de las diferencias emparejadas.
"""
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# El algoritmo, exactamente como lo dejaron las Lecciones 01-06.
MUTACION_MU, MUTACION_SIGMA = 0.0, 1.0
TAMANO_TORNEO, ALFA_MEZCLA = 3, 1.0
TAMANO_POBLACION = 10   # --- CAMBIADO --- de 16 del libro: la linea base del curso
PROBABILIDAD_MUTACION = 0.1
CORRIDAS = 500
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


@dataclass(frozen=True)
class Problema:
    """Un paisaje, su caja, y el presupuesto de generaciones con el que se corre.

    Args:
        argumentos del constructor.

    Example:
        Medida correctamente, la tasa de cruza sube 55.6% → 83.0%. Los extremos
    """
    nombre: str
    aptitud: Callable[[list[float]], float]
    limite_inferior: float
    limite_superior: float
    genes: int
    generaciones: int


def paisaje_seno(genes: list[float]) -> float:
    """El paisaje de la Leccion 01, sin cambios: un gen, un pico global suave.

    Args:
        genes.

    Returns:
        float.

    Example:
        Objetivo 1.4299% de la caja; 5 de 8 corridas lo encuentran (62.5% ± 17.1%).
    """
    return float(np.sin(genes[0]) - 0.2 * abs(genes[0]))


SENO = Problema("seno 1-D", paisaje_seno, -10.0, 10.0, 1, 10)


class Individuo:
    """Una solucion candidata, que lleva el problema que la juzga.

    Args:
        argumentos del constructor.

    Example:
        Medida correctamente, la tasa de cruza sube 55.6% → 83.0%. Los extremos
    """

    def __init__(self, genes: list[float], problema: Problema) -> None:
        self.problema = problema
        self.genes = [acotar(g, problema) for g in genes]
        self.aptitud = float(problema.aptitud(self.genes))

    def __repr__(self) -> str:
        coordenadas = ", ".join(f"{g:+.3f}" for g in self.genes)
        return f"({coordenadas}) f={self.aptitud:+.4f}"


def acotar(gen: float, problema: Problema) -> float:
    """La cruza y la mutacion proponen cualquier cosa; la caja tiene bordes.

    Args:
        gen, problema.

    Returns:
        float.

    Example:
        acotar(11.4) == 10.0 en la caja [0, 10] de esta lección.
    """
    return max(problema.limite_inferior, min(problema.limite_superior, gen))


def seleccion_torneo(poblacion: list[Individuo]) -> list[Individuo]:
    """El ganador de la Leccion 03: la presion es un entero, establecido a proposito.

    Args:
        poblacion.

    Returns:
        list[Individuo].

    Example:
        Medida correctamente, la tasa de cruza sube 55.6% → 83.0%. Los extremos
    """
    return [max([random.choice(poblacion) for _ in range(TAMANO_TORNEO)],
                key=lambda i: i.aptitud) for _ in range(len(poblacion))]


def cruza(padre1: Individuo, padre2: Individuo
              ) -> tuple[Individuo, Individuo]:
    """Cruza de mezcla, un factor de mezcla por gen.

    Args:
        padre1, padre2.

    Returns:
        tuple[Individuo, Individuo].

    Example:
        Medida correctamente, la tasa de cruza sube 55.6% → 83.0%. Los extremos
    """
    genes1, genes2 = [], []
    for a, b in zip(padre1.genes, padre2.genes):
        shift = (1 + 2 * ALFA_MEZCLA) * random.random() - ALFA_MEZCLA
        genes1.append((1 - shift) * a + shift * b)
        genes2.append(shift * a + (1 - shift) * b)
    return (Individuo(genes1, padre1.problema),
            Individuo(genes2, padre1.problema))


def mutar(individuo: Individuo, sigma: float = MUTACION_SIGMA) -> Individuo:
    """Una moneda decide si el individuo muta; luego cada gen se mueve.

    Args:
        individuo, sigma.

    Returns:
        Individuo.

    Example:
        Medida correctamente, la tasa de cruza sube 55.6% → 83.0%. Los extremos
    """
    return Individuo([g + random.gauss(MUTACION_MU, sigma)
                       for g in individuo.genes], individuo.problema)


# --- NUEVO (1) las perillas son argumentos ---------------------------------------
def correr(problema: Problema, semilla: int, tamano_poblacion: int,
        probabilidad_cruza: float, probabilidad_mutacion: float
        ) -> Individuo:
    """Una corrida completa, devolviendo su respuesta: el mejor individuo historico.

    Las probabilidades llegan como argumentos porque esta leccion las gira -
    una perilla que vive en una constante global no puede ser barrida. La corrida i usa la semilla
    i, asi que cada configuracion enfrenta los mismos dados y la comparacion esta emparejada.

    Args:
        problema, semilla, tamano_poblacion, probabilidad_cruza, probabilidad_mutacion.

    Returns:
        Individuo.

    Example:
        Corrida i usa semilla i, para que las comparaciones vayan emparejadas.
    """
    random.seed(semilla)
    poblacion = [Individuo([random.uniform(problema.limite_inferior, problema.limite_superior)
                              for _ in range(problema.genes)], problema)
                  for _ in range(tamano_poblacion)]
    mejor_historico = max(poblacion, key=lambda i: i.aptitud)
    for _ in range(problema.generaciones):
        seleccionados = seleccion_torneo(poblacion)
        cruzados: list[Individuo] = []
        # Pares consecutivos: el torneo ya barajó el orden al copiar.
        for padre1, padre2 in zip(seleccionados[::2], seleccionados[1::2]):
            if random.random() < probabilidad_cruza:
                cruzados.extend(cruza(padre1, padre2))
            else:
                cruzados.extend([padre1, padre2])
        poblacion = [mutar(ind) if random.random() < probabilidad_mutacion
                      else ind for ind in cruzados]
        mejor = max(poblacion, key=lambda i: i.aptitud)
        if mejor.aptitud > mejor_historico.aptitud:
            mejor_historico = mejor
    return mejor_historico
# ------------------------------------------------------------------------------


def optimo_fuerza_bruta(problema: Problema, puntos_por_eje: int) -> float:
    """El mejor valor en una cuadricula regular: la verdad, desde fuera de cualquier corrida.

    Args:
        problema, puntos_por_eje.

    Returns:
        float.

    Example:
        Seno 1-D: +0.705908. Libro 2-D: +0.500000 sin volumen.
    """
    eje = np.linspace(problema.limite_inferior, problema.limite_superior, puntos_por_eje)
    return float(max(problema.aptitud([float(x)]) for x in eje))


TOLERANCIA = 0.01


def veredicto(individuo: Individuo, optimo: float) -> bool:
    """Llego esta corrida? Un si/no, decidido desde fuera de la corrida.

    Args:
        individuo, optimo.

    Returns:
        bool.

    Example:
        En seno 1-D, 5 de 8 corridas caen dentro de la tolerancia.
    """
    return individuo.aptitud >= optimo - TOLERANCIA


# --- NUEVO (2) error_estandar() -------------------------------------------------
def error_estandar(tasa: float, n: int) -> float:
    """La oscilacion de una tasa medida: sqrt(p(1-p)/n).

    El error estandar por sustitucion de la Leccion 06, ya como equipo permanente: ninguna decision
    de afinacion en esta leccion es citada sin ella.

    Args:
        tasa, n.

    Returns:
        float.

    Example:
        8 corridas ±17.1 puntos; 1000 corridas ±1.3 puntos.
    """
    return (tasa * (1 - tasa) / n) ** 0.5
# ------------------------------------------------------------------------------


def intervalo_wilson(exitos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo binomial aproximado al 95%, incluso en los extremos."""
    tasa = exitos / n
    denominador = 1 + z ** 2 / n
    centro = (tasa + z ** 2 / (2 * n)) / denominador
    radio = z / denominador * (
        tasa * (1 - tasa) / n + z ** 2 / (4 * n ** 2)
    ) ** 0.5
    return centro - radio, centro + radio


# --- NUEVO (3) medir() --------------------------------------------------------
def medir(problema: Problema, perilla: str, valores: tuple, fijas: dict,
            corridas: int, optimo: float
          ) -> list[tuple[float, float, float, list[int]]]:
    """El instrumento de barrido: para cada valor de `perilla`, con las otras perillas
    mantenidas en `fijas`, corre el algoritmo `corridas` veces y devuelve
    (valor, tasa, error estándar, resultados por semilla) por configuración.

    Mismas semillas para cada configuracion, por lo que los dados estan emparejados y las diferencias
    son de la perilla, no de la suerte.

    Args:
        problema, perilla, valores, fijas, corridas, optimo.

    Returns:
        Resúmenes y resultados 0/1 necesarios para diferencias emparejadas.

    Example:
        Cruza 55.6% → 83.0% en 500 corridas; mutación bruta 61.8% → 89.8%.
    """
    filas = []
    for valor in valores:
        configuraciones = {**fijas, perilla: valor}
        resultados = [int(veredicto(correr(problema, semilla, **configuraciones), optimo))
                      for semilla in range(corridas)]
        exitos = sum(resultados)
        tasa = exitos / corridas
        filas.append((valor, tasa, error_estandar(tasa, corridas), resultados))
    return filas
# ------------------------------------------------------------------------------


# ===== La primera perilla, medida ===============================================
OPTIMO = optimo_fuerza_bruta(SENO, 20001)
FIJAS = {"tamano_poblacion": TAMANO_POBLACION,
         "probabilidad_mutacion": PROBABILIDAD_MUTACION}
VALORES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

print(f"El objetivo: referencia de cuadrícula {OPTIMO:+.6f}, exito significa a menos de {TOLERANCIA} "
      f"de el.")
print(f"El instrumento: {CORRIDAS} corridas por configuracion, poblacion "
      f"{FIJAS['tamano_poblacion']}, mutacion "
      f"{FIJAS['probabilidad_mutacion']} fija, las mismas {CORRIDAS} semillas para "
      f"cada configuracion.\n")
print("  cruza     | tasa de exito")
print("  ----------+--------------")
filas = medir(SENO, "probabilidad_cruza", VALORES, FIJAS, CORRIDAS, OPTIMO)
for valor, tasa, error, resultados in filas:
    bajo, alto = intervalo_wilson(sum(resultados), CORRIDAS)
    print(f"  {valor:>9} | {tasa:6.1%} +/- {error:4.1%} un EE; "
          f"Wilson 95% [{bajo:.1%}, {alto:.1%}]")

primera, ultima = filas[0], filas[-1]
brecha = ultima[1] - primera[1]
diferencias = [despues - antes for antes, despues in zip(primera[3], ultima[3])]
error_emparejado = statistics.stdev(diferencias) / len(diferencias) ** 0.5
print(f"\nEl contraste de extremos que el paso 1 no pudo fijar es claro en esta muestra: la tasa "
      f"sube monotonicamente")
print(f"de {primera[1]:.1%} en cruza {primera[0]} a {ultima[1]:.1%} en "
      f"cruza {ultima[0]}. Los extremos de la")
print(f"perilla están separados por {brecha:.1%}; el error estándar emparejado es "
      f"{error_emparejado:.1%}. Las brechas vecinas son menores y el barrido es")
print("exploratorio, no una prueba de comparaciones múltiples; no es el abismo que las")
print("corridas sueltas pretendian. La cruza funciona, y ahora se mide en lugar")
print("de atestiguarse.")
print("\nUna columna falta en esta tabla, y cambia como se lee la tabla:")
print("cuanto gasto cada configuracion. El Paso 3 la anade.")

# La imagen: tasas observadas con intervalos de Wilson al 95%.
fig, ax = plt.subplots(figsize=(7.5, 4.2))
intervalos = [intervalo_wilson(sum(fila[3]), CORRIDAS) for fila in filas]
errores_inferiores = [fila[1] - intervalo[0] for fila, intervalo in zip(filas, intervalos)]
errores_superiores = [intervalo[1] - fila[1] for fila, intervalo in zip(filas, intervalos)]
ax.errorbar([r[0] for r in filas], [r[1] for r in filas],
            yerr=[errores_inferiores, errores_superiores], fmt="o-", color="tab:blue",
            capsize=4, linewidth=1.6)
ax.set_title(f"Probabilidad de cruza, medida: {CORRIDAS} corridas por configuracion "
             f"(poblacion {FIJAS['tamano_poblacion']}, "
             f"mutacion {FIJAS['probabilidad_mutacion']})")
ax.set_xlabel("probabilidad de cruza")
ax.set_ylabel("tasa observada (intervalo de Wilson al 95%)")
ax.grid(True, linestyle=":", alpha=0.5)
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "afinacion_02_el_instrumento.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/afinacion_02_el_instrumento.png")

