"""
Leccion 07 - Paso 1: El metodo del libro, fielmente
==================================================
NUEVO EN ESTE PASO: nada es nuevo para los estudiantes. Este es el algoritmo de
las Lecciones 01-06, sin cambios, manejado de la forma en que lo hace el
Capitulo 7 de Gridin: elige una perilla, corre una configuracion una vez, mira
la respuesta, elige un ganador.

El Capitulo 7 es el capitulo de laboratorio del libro - 42 figuras, tres
perillas (probabilidad de cruza, probabilidad de mutacion, tamano de poblacion),
cada una girada en su propio paisaje, cada configuracion mostrada como una
corrida, generacion por generacion. Es la forma natural de afinar, y es
exactamente el metodo que la Leccion 06 demolio. Este paso mantiene el
protocolo del libro y sus numeros - poblacion 16, probabilidad de mutacion 0.2,
probabilidades de cruza 0.0, 0.4, 0.7, semilla 63 - en el paisaje del curso, y
lo lee de la forma en que lo lee el libro. Luego lo lee siete veces mas.

El paisaje es el seno de la Leccion 01, un gen, optimo +0.705908 conocido por
fuerza bruta, "exito" significando a menos de 0.01 de el. Un paisaje para toda
la leccion, a proposito: el libro gira cada perilla en una funcion diferente,
por lo que su capitulo nunca compara nada con nada. Aqui cada configuracion
enfrenta el mismo problema, por lo que las perillas pueden ser comparadas -
primero mal (este paso), luego correctamente (el resto de la leccion).

Ejecutalo:  python afinacion_01_corridas_sueltas.py

Una corrida por config corona pc = 0.7 por 0.0021; sobre 8 semillas las
victorias van 0/3/5 y la división más estrecha es 0.000004.
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
# El protocolo del libro: su poblacion, su probabilidad de mutacion, sus tres
# configuraciones de cruza, su semilla.
TAMANO_POBLACION = 16
PROBABILIDAD_MUTACION = 0.2
CONFIGURACIONES_CRUZA = (0.0, 0.4, 0.7)
SEMILLA = 63
SEMILLAS = 8
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


@dataclass(frozen=True)
class Problema:
    """Un paisaje, su caja, y el presupuesto de generaciones con el que se corre.

    Args:
        argumentos del constructor.

    Example:
        Una corrida por config corona pc = 0.7 por 0.0021; sobre 8 semillas las victorias van 0/3/5.
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
        Una corrida por config corona pc = 0.7 por 0.0021; sobre 8 semillas las victorias van 0/3/5.
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
        Una corrida por config corona pc = 0.7 por 0.0021; sobre 8 semillas las victorias van 0/3/5.
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
        Una corrida por config corona pc = 0.7 por 0.0021; sobre 8 semillas las victorias van 0/3/5.
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
        Una corrida por config corona pc = 0.7 por 0.0021; sobre 8 semillas las victorias van 0/3/5.
    """
    return Individuo([g + random.gauss(MUTACION_MU, sigma)
                       for g in individuo.genes], individuo.problema)


def correr(problema: Problema, semilla: int, probabilidad_cruza: float
        ) -> list[list[Individuo]]:
    """Una corrida completa, a la manera del libro: una configuracion, una semilla, una respuesta.

    Args:
        problema, semilla, probabilidad_cruza.

    Returns:
        list[list[Individuo]].

    Example:
        Corrida i usa semilla i, para que las comparaciones vayan emparejadas.
    """
    random.seed(semilla)
    poblacion = [Individuo([random.uniform(problema.limite_inferior, problema.limite_superior)
                              for _ in range(problema.genes)], problema)
                  for _ in range(TAMANO_POBLACION)]
    generaciones = [poblacion]
    for _ in range(problema.generaciones):
        seleccionados = seleccion_torneo(poblacion)
        cruzados: list[Individuo] = []
        # Pares consecutivos: el torneo ya barajó el orden al copiar.
        for padre1, padre2 in zip(seleccionados[::2], seleccionados[1::2]):
            if random.random() < probabilidad_cruza:
                cruzados.extend(cruza(padre1, padre2))
            else:
                cruzados.extend([padre1, padre2])
        poblacion = [mutar(ind) if random.random() < PROBABILIDAD_MUTACION
                      else ind for ind in cruzados]
        generaciones.append(poblacion)
    return generaciones


def mejor_historico(generaciones: list[list[Individuo]]) -> Individuo:
    """El mejor individuo que la corrida haya tenido, que es la respuesta de la corrida.

    Args:
        generaciones.

    Returns:
        Individuo.

    Example:
        Una corrida por config corona pc = 0.7 por 0.0021; sobre 8 semillas las victorias van 0/3/5.
    """
    return max((ind for pob in generaciones for ind in pob),
               key=lambda i: i.aptitud)


def historial(generaciones: list[list[Individuo]]) -> list[float]:
    """Mejor aptitud hasta ahora despues de cada generacion - la curva de convergencia.

    Args:
        generaciones.

    Returns:
        list[float].

    Example:
        Una corrida por config corona pc = 0.7 por 0.0021; sobre 8 semillas las victorias van 0/3/5.
    """
    mejor, filas = -float("inf"), []
    for poblacion in generaciones:
        mejor = max(mejor, max(ind.aptitud for ind in poblacion))
        filas.append(mejor)
    return filas


# ===== Parte 1: el metodo del libro, una vez ========================================
print(f"El protocolo del libro: poblacion {TAMANO_POBLACION}, mutacion "
      f"{PROBABILIDAD_MUTACION}, semilla {SEMILLA},")
print(f"una corrida por configuracion de cruza, {SENO.generaciones} generaciones cada una:\n")
print("  cruza     | mejor historico")
print("  ----------+-----------------")
sueltas = {}
for pc in CONFIGURACIONES_CRUZA:
    campeon = mejor_historico(correr(SENO, SEMILLA, pc))
    sueltas[pc] = campeon.aptitud
    print(f"  {pc:>9} | {campeon.aptitud:+15.4f}")
ganador = max(sueltas, key=sueltas.get)
print(f"\nVeredicto, estilo libro: probabilidad de cruza {ganador} gana. Escribelo "
      "y sigue adelante.")
margen = max(sueltas.values()) - min(sueltas.values())
print(f"(El margen sobre la peor configuracion: {margen:.4f}.)")

# ===== Parte 2: el mismo protocolo, siete veces mas ============================
print(f"\nEl mismo protocolo, {SEMILLAS} semillas:\n")
encabezado = "  semilla |" + "|".join(f"  pc = {pc:>3}  " for pc in CONFIGURACIONES_CRUZA)
print(encabezado + "  ganador")
print("  --------+" + "+".join("-" * 12 for _ in CONFIGURACIONES_CRUZA) + "--------")
victorias = {pc: 0 for pc in CONFIGURACIONES_CRUZA}
curvas = {pc: [] for pc in CONFIGURACIONES_CRUZA}
brechas, deficiencias = [], []
for semilla in range(SEMILLAS):
    respuestas = {}
    for pc in CONFIGURACIONES_CRUZA:
        generaciones = correr(SENO, semilla, pc)
        respuestas[pc] = mejor_historico(generaciones).aptitud
        curvas[pc].append(historial(generaciones))
    ganador_ronda = max(respuestas, key=respuestas.get)
    victorias[ganador_ronda] += 1
    posiciones = sorted(respuestas.values(), reverse=True)
    brechas.append(posiciones[0] - posiciones[1])
    deficiencias.append(posiciones[0] - posiciones[-1])
    fila = "  ".join(f"{respuestas[pc]:+9.4f} " for pc in CONFIGURACIONES_CRUZA)
    print(f"  {semilla:>7} | {fila}  pc = {ganador_ronda}")
print("\nConteos de victorias: " + ", ".join(f"pc = {pc}: {victorias[pc]} de {SEMILLAS}"
                                   for pc in CONFIGURACIONES_CRUZA))
mas_ajustado = min(range(SEMILLAS), key=lambda s: brechas[s])
mas_aspero = max(range(SEMILLAS), key=lambda s: deficiencias[s])
print(f"\nLee la tabla. En la semilla {mas_ajustado} las dos configuraciones principales estan separadas por "
      f"{brechas[mas_ajustado]:.6f}; en la semilla {mas_aspero} las mismas")
print(f"tres configuraciones se esparcen por {deficiencias[mas_aspero]:.4f}. La perilla "
      f"no cambio - la semilla lo hizo.")
print("Un metodo que no puede reproducir su propio ranking en el mismo problema no esta midiendo")
print("la perilla. Esta leyendo ruido.")
print("\nLa Leccion 06 construyo el instrumento exactamente para esto. El Paso 2 lo retoma.")

# La imagen: un panel por semilla, tres curvas de convergencia cada uno. El ranking
# que parecia un hecho en la unica corrida del libro se voltea de panel a panel.
fig, ejes = plt.subplots(2, 4, figsize=(11.5, 4.6), sharex=True, sharey=True)
for semilla, ax in enumerate(ejes.flat):
    for pc in CONFIGURACIONES_CRUZA:
        ax.plot(curvas[pc][semilla], linewidth=1.4, label=f"pc = {pc}")
    ax.set_title(f"semilla {semilla}", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    if semilla % 4 == 0:
        ax.set_ylabel("mejor hasta ahora", fontsize=8)
ejes[1, 1].legend(fontsize=7, loc="lower right")
fig.suptitle("Una corrida por configuracion, ocho veces: el ranking es inestable")
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "afinacion_01_corridas_sueltas.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/afinacion_01_corridas_sueltas.png")

