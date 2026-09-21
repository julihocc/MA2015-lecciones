"""
Lección 06 - Paso 1: Una corrida, y lo que vale
===============================================
NUEVO EN ESTE PASO: no hay nada nuevo para los estudiantes. Este es el
algoritmo de las Lecciones 01-05, sin cambios, apuntado al paisaje bidimensional
con el que Gridin abre el Capítulo 6 - y luego ejecutado ocho veces en lugar de una.

Cada lección hasta ahora terminó leyendo una sola corrida. La Lección 01 leyó
una corrida y concluyó que el algoritmo funciona; la Lección 02 leyó una corrida
y concluyó que una regla de paro es gratuita; la Lección 03 y la Lección 05
ambas cerraron con una pregunta abierta que no pudieron responder, y ambas
preguntas tenían la misma forma: ¿ayuda este cambio, a lo largo de las corridas?
Esta lección es la maquinaria para responder eso, y comienza mostrando por qué
la pregunta no puede evadirse.

Un algoritmo genético es una variable aleatoria. Toma una semilla y devuelve una
respuesta, y la respuesta cambia cuando la semilla lo hace - no por un error de
redondeo, sino por más que todo el rango de mejora que logra la corrida. La curva
por generación impresa a continuación parece una prueba de convergencia. Es una
sola muestra.

El problema es del libro: f(x, y) = sin(x)cos(x) - (|(x + 50)(y - 10)| / 10)^0.1
en [-100, 100]^2. Vale la pena leerlo antes de ejecutar nada. El primer término
oscila entre -0.5 y +0.5. El segundo término es una penalización que desaparece
solo en las dos líneas x = -50 e y = 10, y se acerca a cero tan lentamente - una
décima potencia - que estar cerca casi no vale nada. El paso 2 mide lo que
eso cuesta.

Los operadores son del curso, no del libro: torneo de 3 (Lección 03),
cruza mezclada con alfa = 1.0 (Lección 04), mutación gaussiana con sigma = 1.0
detrás de una moneda por individuo (Lección 05). Mantenerlos significa que cada
número en esta lección continúa las anteriores en lugar de reiniciarlas.

Ejecútalo:  python tasa_exito_01_una_corrida.py

La dispersión entre semillas es 0.8946, 1.5 veces la ganancia de una corrida
(+0.5822). Semilla 3 reporta -0.0129; semilla 4, -0.9076.
"""
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# El algoritmo, exactamente como lo dejaron las Lecciones 01-05.
TAMANO_POBLACION = 10
PROBABILIDAD_CRUZA = 0.8
PROBABILIDAD_MUTACION = 0.1
MUTACION_MU, MUTACION_SIGMA = 0.0, 1.0
TAMANO_TORNEO, ALFA_MEZCLA = 3, 1.0
# La corrida i usa la semilla i. Esa convención se mantiene en toda la lección: hace que
# cualquier subconjunto de corridas sea reproducible por sí mismo, y permite que un
# paso posterior reproduzca el bloque de corridas de otra lección exactamente
# pidiendo las mismas semillas.
CORRIDAS = 8
FIGURAS = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class Problema:
    """Un paisaje, su caja y el presupuesto de generaciones con el que se ejecuta.

    La Lección 02 hizo que la función de aptitud fuera un argumento. Aquí los límites,
    el número de genes y el presupuesto viajan con él, porque esta lección compara
    corridas y una comparación solo es justa si a ambos lados se les dio el mismo
    presupuesto.

    Args:
        argumentos del constructor.

    Example:
        La dispersión entre semillas es 0.8946, 1.5 veces la ganancia de una corrida (+0.5822).
    """
    nombre: str
    aptitud: Callable[[list[float]], float]
    bajo: float
    alto: float
    genes: int
    generaciones: int


def libro_2d(genes: list[float]) -> float:
    """El paisaje del Capítulo 6 de Gridin. El término de penalización es la mitad interesante.

    Args:
        genes.

    Returns:
        float.

    Example:
        Óptimo +0.500000, pero 0.003197% de la cuadrícula; 0 de 8 corridas aciertan.
    """
    x, y = genes
    return np.sin(x) * np.cos(x) - pow(abs((x + 50.0) * (y - 10.0)) / 10.0, 0.1)


DOS_D = Problema("libro 2-D", libro_2d, -100.0, 100.0, 2, 25)


class Individuo:
    """Una solución candidata, que lleva el problema que la juzga.

    Args:
        argumentos del constructor.

    Example:
        La dispersión entre semillas es 0.8946, 1.5 veces la ganancia de una corrida (+0.5822).
    """

    def __init__(self, genes: list[float], problema: Problema) -> None:
        self.problema = problema
        self.genes = [acotar(g, problema) for g in genes]
        self.aptitud = float(problema.aptitud(self.genes))

    def __repr__(self) -> str:
        coordenadas = ", ".join(f"{g:+.3f}" for g in self.genes)
        return f"({coordenadas}) f={self.aptitud:+.4f}"


def acotar(gen: float, problema: Problema) -> float:
    """La cruza y la mutación proponen cualquier cosa; la caja tiene bordes.

    Args:
        gen, problema.

    Returns:
        float.

    Example:
        acotar(11.4) == 10.0 en la caja [0, 10] de esta lección.
    """
    return max(problema.bajo, min(problema.alto, gen))


def seleccion_torneo(poblacion: list[Individuo]) -> list[Individuo]:
    """El ganador de la Lección 03: la presión es un número entero, fijado a propósito.

    Args:
        poblacion.

    Returns:
        list[Individuo].

    Example:
        La dispersión entre semillas es 0.8946, 1.5 veces la ganancia de una corrida (+0.5822).
    """
    return [max([random.choice(poblacion) for _ in range(TAMANO_TORNEO)],
                key=lambda i: i.aptitud) for _ in range(len(poblacion))]


def cruza(padre1: Individuo, padre2: Individuo
          ) -> tuple[Individuo, Individuo]:
    """Cruza mezclada (blend crossover), un factor de mezcla por gen.

    Args:
        padre1, padre2.

    Returns:
        tuple[Individuo, Individuo].

    Example:
        La dispersión entre semillas es 0.8946, 1.5 veces la ganancia de una corrida (+0.5822).
    """
    genes1, genes2 = [], []
    for a, b in zip(padre1.genes, padre2.genes):
        cambio = (1 + 2 * ALFA_MEZCLA) * random.random() - ALFA_MEZCLA
        genes1.append((1 - cambio) * a + cambio * b)
        genes2.append(cambio * a + (1 - cambio) * b)
    return (Individuo(genes1, padre1.problema),
            Individuo(genes2, padre1.problema))


def mutar(individuo: Individuo, sigma: float = MUTACION_SIGMA) -> Individuo:
    """Una moneda decide si el individuo muta; luego cada gen se mueve.

    Esta es la convención de la Lección 05 (probabilidad por gen 1.0), e importa
    aquí por una razón que no tiene nada que ver con la biología: es el operador
    cuyo flujo aleatorio midió la Lección 05, por lo que el paso 5 puede reproducir
    sus corridas.

    Args:
        individuo, sigma.

    Returns:
        Individuo.

    Example:
        La dispersión entre semillas es 0.8946, 1.5 veces la ganancia de una corrida (+0.5822).
    """
    return Individuo([g + random.gauss(MUTACION_MU, sigma)
                       for g in individuo.genes], individuo.problema)


def evolucionar_una_generacion(poblacion: list[Individuo]) -> list[Individuo]:
    """SELECCION_TORNEO -> CRUZA -> MUTAR -> reemplazar, una vez.

    Args:
        poblacion.

    Returns:
        list[Individuo].

    Example:
        La dispersión entre semillas es 0.8946, 1.5 veces la ganancia de una corrida (+0.5822).
    """
    seleccionados = seleccion_torneo(poblacion)
    cruzados: list[Individuo] = []
    # Pares consecutivos: el torneo ya barajó el orden al copiar.
    for padre1, padre2 in zip(seleccionados[::2], seleccionados[1::2]):
        if random.random() < PROBABILIDAD_CRUZA:
            cruzados.extend(cruza(padre1, padre2))
        else:
            cruzados.extend([padre1, padre2])
    return [mutar(ind) if random.random() < PROBABILIDAD_MUTACION else ind
            for ind in cruzados]


def corrida(problema: Problema, semilla: int) -> list[list[Individuo]]:
    """Una corrida completa: inicializar, luego evolucionar por el presupuesto del problema.

    Args:
        problema, semilla.

    Returns:
        list[list[Individuo]].

    Example:
        La dispersión entre 8 semillas es 0.8946, 1.5× la ganancia de una corrida.
    """
    random.seed(semilla)
    poblacion = [Individuo([random.uniform(problema.bajo, problema.alto)
                              for _ in range(problema.genes)], problema)
                  for _ in range(TAMANO_POBLACION)]
    generaciones = [poblacion]
    for _ in range(problema.generaciones):
        poblacion = evolucionar_una_generacion(poblacion)
        generaciones.append(poblacion)
    return generaciones


def mejor_historico(generaciones: list[list[Individuo]]) -> Individuo:
    """El mejor individuo que haya tenido la corrida, que es la respuesta de la corrida.

    Args:
        generaciones.

    Returns:
        Individuo.

    Example:
        La dispersión entre semillas es 0.8946, 1.5 veces la ganancia de una corrida (+0.5822).
    """
    return max((ind for pop in generaciones for ind in pop),
               key=lambda i: i.aptitud)


def historia(generaciones: list[list[Individuo]]) -> list[tuple[float, float, float]]:
    """Promedio, mejor de la generación y mejor histórico, una fila por generación.

    Args:
        generaciones.

    Returns:
        list[tuple[float, float, float]].

    Example:
        La dispersión entre semillas es 0.8946, 1.5 veces la ganancia de una corrida (+0.5822).
    """
    filas, historico = [], -float("inf")
    for poblacion in generaciones:
        aptitudes = [ind.aptitud for ind in poblacion]
        historico = max(historico, max(aptitudes))
        filas.append((statistics.fmean(aptitudes), max(aptitudes), historico))
    return filas


# ===== Parte 1: leer una corrida de la manera en que cada lección lo ha hecho ==
print(f"Una corrida en el paisaje {DOS_D.nombre}, semilla 0, "
      f"{DOS_D.generaciones} generaciones:\n")
filas = historia(corrida(DOS_D, 0))
print("  generación | promedio | mejor de gen | mejor histórico")
print("  -----------+----------+--------------+----------------")
for generacion, (promedio, mejor, historico) in enumerate(filas):
    print(f"  {generacion:>10} | {promedio:+8.4f} | {mejor:+12.4f} | {historico:+15.4f}")
primer_historico, ultimo_historico = filas[0][2], filas[-1][2]
print(f"\nLa corrida mejoró su mejor histórico de {primer_historico:+.4f} a "
      f"{ultimo_historico:+.4f}, una ganancia de {ultimo_historico - primer_historico:+.4f}.")
print("La aptitud promedio la sigue hacia arriba. La curva es monótona, la población")
print("alcanza a su campeón, y todas las lecciones hasta ahora habrían dejado de")
print("leer aquí.")

# ===== Parte 2: la misma configuración, ocho veces ============================
print(f"\nLa misma configuración, sin cambiar nada más que la semilla, {CORRIDAS} veces:\n")
print("  semilla | mejor histórico | dónde")
print("  --------+-----------------+---------------------------")
respuestas = []
for semilla in range(CORRIDAS):
    campeon = mejor_historico(corrida(DOS_D, semilla))
    respuestas.append(campeon.aptitud)
    donde = ", ".join(f"{g:+.2f}" for g in campeon.genes)
    print(f"  {semilla:>7} | {campeon.aptitud:+15.4f} | ({donde})")

diferencia = max(respuestas) - min(respuestas)
ganancia = ultimo_historico - primer_historico
print(f"\nMejor respuesta {max(respuestas):+.4f}, peor {min(respuestas):+.4f}, "
      f"dispersión {diferencia:.4f}.")
print(f"Media {statistics.fmean(respuestas):+.4f}, desviación estándar "
      f"{statistics.stdev(respuestas):.4f}.")
print(f"\nLa dispersión entre semillas es {diferencia / ganancia:.1f} veces la mejora")
print(f"que logró la corrida rastreada ({ganancia:+.4f}). Qué semilla resultó que usaste")
print("decide más sobre la respuesta que la búsqueda misma.")
peor_semilla = min(range(CORRIDAS), key=lambda s: respuestas[s])
mejor_semilla = max(range(CORRIDAS), key=lambda s: respuestas[s])
print(f"\nLa semilla {mejor_semilla} haría que reportaras {max(respuestas):+.4f}; la semilla "
      f"{peor_semilla} haría que reportaras {min(respuestas):+.4f}.")
print("Ambos reportes serían ciertos, y ninguno trataría sobre el algoritmo.")
print("\nEntonces: ¿cuántas corridas, y qué se debe contar exactamente? El paso 2 tiene que")
print("establecer qué cuenta como éxito antes de que pueda contarse en absoluto.")

# La imagen: las tres curvas de la corrida rastreada, y las ocho curvas del mejor histórico.
fig, (ax_uno, ax_muchos) = plt.subplots(1, 2, figsize=(11.5, 4.0))
ax_uno.plot([r[0] for r in filas], label="promedio de generación")
ax_uno.plot([r[1] for r in filas], label="mejor de generación")
ax_uno.plot([r[2] for r in filas], label="mejor histórico", linewidth=2.0)
ax_uno.set_title("Una corrida, semilla 0: la imagen que leyó cada lección hasta ahora",
                 fontsize=10)
ax_uno.set_xlabel("generación")
ax_uno.set_ylabel("aptitud")
ax_uno.grid(True, linestyle=":", alpha=0.5)
ax_uno.legend(fontsize=8, loc="lower right")

for semilla in range(CORRIDAS):
    ax_muchos.plot([r[2] for r in historia(corrida(DOS_D, semilla))],
                   label=f"semilla {semilla}", linewidth=1.3)
ax_muchos.set_title(f"Mejor histórico, {CORRIDAS} semillas: dispersión {diferencia:.2f} "
                    f"vs ganancia de una corrida {ganancia:.2f}", fontsize=10)
ax_muchos.set_xlabel("generación")
ax_muchos.set_ylabel("mejor aptitud hasta ahora")
ax_muchos.grid(True, linestyle=":", alpha=0.5)
ax_muchos.legend(fontsize=7, ncol=2, loc="lower right")

fig.suptitle("Un algoritmo genético es una variable aleatoria, no un procedimiento")
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "tasa_exito_01_una_corrida.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/tasa_exito_01_una_corrida.png")

