"""
Lección 06 - Paso 4: El presupuesto
===================================
NUEVO EN ESTE PASO: el contador de evaluaciones, el barrido de población, y
exito_ciegas() - el punto de referencia que nadie ha ejecutado todavía.

El paso 3 midió la efectividad: una tasa de éxito del 79.4% con población 10, y
cerró con la pregunta que una tasa no puede responder - ¿cuánto costaron esas evaluaciones
y estuvieron bien gastadas? Este paso las cuenta. El Capítulo 6 de Gridin
lo hace con un atributo de clase en Individuo que cuenta cada evaluación de
aptitud, y con un barrido de tamaños de población que grafica la aptitud final
contra el número de individuos evaluados.

Dos medidas surgen de esto, y tiran en direcciones opuestas. Aumentar la
población de 10 a 30 eleva la tasa de éxito del 80% al 99% - y aumenta
el costo de una corrida de 100 evaluaciones a 300. La efectividad sube,
la eficiencia baja, monótonamente, ambas. No hay una configuración
que sea mejor en ambas; solo hay una elección, y la Lección 07 existe para hacerla.

Luego el veredicto honesto. Gasta el mismo presupuesto en extracciones uniformes a ciegas y
el margen del AG en este paisaje resulta ser de unos pocos puntos, no un
abismo - porque un objetivo de 0.286 de ancho es fácil para todos. Esa es una propiedad
del problema, no un defecto del algoritmo: en los problemas combinatorios
de las Lecciones 09 a 11, una extracción a ciegas esencialmente nunca acierta, y los operadores
son la única opción viable.

CAMBIOS RESPECTO A tasa_exito_03_mil_corridas.py
Introdúcelos en este orden:
    1. el contador de evaluaciones Individuo cuenta cada evaluación de aptitud
    2. el barrido de población     TAMANOS_POBLACION, re-ejecutando el instrumento por tamaño
    3. exito_ciegas()              el mismo presupuesto gastado en extracciones a ciegas: 1 - (1 - q) ** E
    4. CORRIDAS = 500              cambiado de 1000: cinco poblaciones, 500 corridas cada una

ELIMINADO DE tasa_exito_03_mil_corridas.py: la búsqueda de picos y el
histograma (la distribución es la historia del paso 3; esta es sobre el costo).

Ejecútalo:  python tasa_exito_04_el_presupuesto.py

Población 10 a 30: éxito 80.0% a 99.4%; el rendimiento cae 8.00 a 3.31
éxitos por cada 1000 evaluaciones.
"""
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# El algoritmo, exactamente como lo dejaron las Lecciones 01-05.
PROBABILIDAD_CRUZA = 0.8
PROBABILIDAD_MUTACION = 0.1
MUTACION_MU, MUTACION_SIGMA = 0.0, 1.0
TAMANO_TORNEO, ALFA_MEZCLA = 3, 1.0
# La corrida i usa la semilla i, para toda la lección.
CORRIDAS = 500   # --- CAMBIADO --- de 1000: cinco poblaciones, 500 corridas cada una
FIGURAS = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class Problema:
    """Un paisaje, su caja y el presupuesto de generaciones con el que se ejecuta.

    Args:
        argumentos del constructor.

    Example:
        Población 10 a 30: éxito 80.0% a 99.4%; el rendimiento cae 8.00 a 3.31 por 1000 evals.
    """
    nombre: str
    aptitud: Callable[[list[float]], float]
    bajo: float
    alto: float
    genes: int
    generaciones: int


def paisaje_seno(genes: list[float]) -> float:
    """El paisaje de la Lección 01, sin cambios: un gen, un pico global suave.

    Args:
        genes.

    Returns:
        float.

    Example:
        Objetivo 1.4299% de la caja; 5 de 8 corridas lo encuentran (62.5% ± 17.1%).
    """
    return float(np.sin(genes[0]) - 0.2 * abs(genes[0]))


SENO = Problema("seno 1-D", paisaje_seno, -10.0, 10.0, 1, 10)


# --- NUEVO (1) el contador de evaluaciones ------------------------------------
class Individuo:
    """Una solución candidata, que lleva el problema que la juzga.

    `evaluaciones` cuenta cada evaluación de aptitud que realiza el algoritmo,
    a lo largo de todo el experimento: se incrementa cada vez que se crea un Individuo,
    y se reinicia a cero antes de cada corrida. La aptitud es lo único por lo que un
    AG paga, por lo que este contador ES el costo de una corrida.

    Args:
        argumentos del constructor.

    Example:
        Población 10 a 30: éxito 80.0% a 99.4%; el rendimiento cae 8.00 a 3.31 por 1000 evals.
    """

    evaluaciones = 0

    def __init__(self, genes: list[float], problema: Problema) -> None:
        self.problema = problema
        self.genes = [acotar(g, problema) for g in genes]
        self.aptitud = float(problema.aptitud(self.genes))
        Individuo.evaluaciones += 1

    def __repr__(self) -> str:
        coordenadas = ", ".join(f"{g:+.3f}" for g in self.genes)
        return f"({coordenadas}) f={self.aptitud:+.4f}"
# ------------------------------------------------------------------------------


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


def seleccion_torneo(poblacion: list[Individuo],
                     tamano: int) -> list[Individuo]:
    """El ganador de la Lección 03: la presión es un número entero, fijado a propósito.

    Args:
        poblacion, tamano.

    Returns:
        list[Individuo].

    Example:
        Población 10 a 30: éxito 80.0% a 99.4%; el rendimiento cae 8.00 a 3.31 por 1000 evals.
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
        Población 10 a 30: éxito 80.0% a 99.4%; el rendimiento cae 8.00 a 3.31 por 1000 evals.
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

    Args:
        individuo, sigma.

    Returns:
        Individuo.

    Example:
        Población 10 a 30: éxito 80.0% a 99.4%; el rendimiento cae 8.00 a 3.31 por 1000 evals.
    """
    return Individuo([g + random.gauss(MUTACION_MU, sigma)
                       for g in individuo.genes], individuo.problema)


def evolucionar_una_generacion(poblacion: list[Individuo],
                               tamano: int) -> list[Individuo]:
    """SELECCION_TORNEO -> CRUZA -> MUTAR -> reemplazar, una vez.

    Args:
        poblacion, tamano.

    Returns:
        list[Individuo].

    Example:
        Población 10 a 30: éxito 80.0% a 99.4%; el rendimiento cae 8.00 a 3.31 por 1000 evals.
    """
    seleccionados = seleccion_torneo(poblacion, tamano)
    cruzados: list[Individuo] = []
    # Pares consecutivos: el torneo ya barajó el orden al copiar.
    for padre1, padre2 in zip(seleccionados[::2], seleccionados[1::2]):
        if random.random() < PROBABILIDAD_CRUZA:
            cruzados.extend(cruza(padre1, padre2))
        else:
            cruzados.extend([padre1, padre2])
    return [mutar(ind) if random.random() < PROBABILIDAD_MUTACION else ind
            for ind in cruzados]


def corrida(problema: Problema, semilla: int, tamano_poblacion: int
            ) -> list[list[Individuo]]:
    """Una corrida completa: inicializar, luego evolucionar por el presupuesto del problema.

    Args:
        problema, semilla, tamano_poblacion.

    Returns:
        list[list[Individuo]].

    Example:
        La dispersión entre 8 semillas es 0.8946, 1.5× la ganancia de una corrida.
    """
    random.seed(semilla)
    poblacion = [Individuo([random.uniform(problema.bajo, problema.alto)
                              for _ in range(problema.genes)], problema)
                  for _ in range(tamano_poblacion)]
    generaciones = [poblacion]
    for _ in range(problema.generaciones):
        poblacion = evolucionar_una_generacion(poblacion, tamano_poblacion)
        generaciones.append(poblacion)
    return generaciones


def mejor_historico(generaciones: list[list[Individuo]]) -> Individuo:
    """El mejor individuo que haya tenido la corrida, que es la respuesta de la corrida.

    Args:
        generaciones.

    Returns:
        Individuo.

    Example:
        Población 10 a 30: éxito 80.0% a 99.4%; el rendimiento cae 8.00 a 3.31 por 1000 evals.
    """
    return max((ind for pop in generaciones for ind in pop),
               key=lambda i: i.aptitud)


def optimo_fuerza_bruta(problema: Problema, puntos_por_eje: int) -> float:
    """El mejor valor en una cuadrícula regular: la verdad, desde fuera de cualquier corrida.

    Args:
        problema, puntos_por_eje.

    Returns:
        float.

    Example:
        Seno 1-D: +0.705908. Libro 2-D: +0.500000 sin volumen.
    """
    eje = np.linspace(problema.bajo, problema.alto, puntos_por_eje)
    return float(max(problema.aptitud([float(x)]) for x in eje))


TOLERANCIA = 0.01


def veredicto(individuo: Individuo, optimo: float) -> bool:
    """¿Llegó esta corrida? Un sí/no, decidido desde fuera de la corrida.

    Args:
        individuo, optimo.

    Returns:
        bool.

    Example:
        En seno 1-D, 5 de 8 corridas caen dentro de la tolerancia.
    """
    return individuo.aptitud >= optimo - TOLERANCIA


def proporcion_dentro_de_tolerancia(problema: Problema, optimo: float,
                                    puntos_por_eje: int) -> float:
    """Fracción de una cuadrícula densa dentro de TOLERANCIA de la referencia.

    Aproxima la probabilidad continua uniforme de acierto; no es una medida
    exacta del conjunto objetivo.
    """
    eje = np.linspace(problema.bajo, problema.alto, puntos_por_eje)
    valores = np.array([problema.aptitud([float(x)]) for x in eje])
    return float(np.mean(valores >= optimo - TOLERANCIA))


def error_estandar(tasa: float, n: int) -> float:
    """El bamboleo de una tasa medida: sqrt(p(1-p)/n).

    Args:
        tasa, n.

    Returns:
        float.

    Example:
        8 corridas ±17.1 puntos; 1000 corridas ±1.3 puntos.
    """
    return (tasa * (1 - tasa) / n) ** 0.5


def intervalo_wilson(exitos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo binomial aproximado al 95% que funciona en 0% y 100%."""
    tasa = exitos / n
    denominador = 1 + z ** 2 / n
    centro = (tasa + z ** 2 / (2 * n)) / denominador
    radio = z / denominador * (
        tasa * (1 - tasa) / n + z ** 2 / (4 * n ** 2)
    ) ** 0.5
    return centro - radio, centro + radio


# --- NUEVO (3) exito_ciegas() -------------------------------------------------
def exito_ciegas(proporcion: float, evaluaciones: float) -> float:
    """La tasa de éxito de gastar `evaluaciones` en extracciones uniformes a ciegas.

    Cada extracción alcanza el objetivo con probabilidad `proporcion`, de manera independiente,
    por lo que la probabilidad de que al menos una de E extracciones acierte es 1 - (1 - proporcion) ** E.
    Este es el punto de referencia que el AG tiene que superar para justificar su maquinaria:
    el mismo presupuesto, sin operadores, sin memoria, sin población.

    Args:
        proporcion, evaluaciones.

    Returns:
        float.

    Example:
        Con igual presupuesto, ciegas llegan 76.2% y 98.7%.
    """
    return 1.0 - (1.0 - proporcion) ** evaluaciones
# ------------------------------------------------------------------------------


optimo = optimo_fuerza_bruta(SENO, 20001)
proporcion = proporcion_dentro_de_tolerancia(SENO, optimo, 20001)
print(f"El objetivo: referencia de cuadrícula {optimo:+.6f}; éxito significa "
      f"estar a {TOLERANCIA}; la cuadrícula estima una proporción ciega de "
      f"{proporcion:.2%}.\n")

# --- NUEVO (2) el barrido de población ----------------------------------------
# El barrido del libro: las mismas semillas, el mismo presupuesto de generaciones, y solo
# cambiando el tamaño de la población. Cada tamaño re-ejecuta todo el instrumento de
# Monte Carlo, contando lo que costó cada corrida.
TAMANOS_POBLACION = (10, 15, 20, 25, 30)

print(f"{CORRIDAS} corridas por tamaño de población, {SENO.generaciones} generaciones "
      f"cada una:\n")
print("  población  | tasa (un EE)  | intervalo Wilson 95% | evaluaciones/corr | éxitos por")
print("             |               |                      |   (media, máx)    | 1000 evaluaciones")
print("  -----------+---------------+----------------------+-------------------+----------------")
barrido = []
for tamano in TAMANOS_POBLACION:
    exitos, costos = 0, []
    for semilla in range(CORRIDAS):
        Individuo.evaluaciones = 0
        campeon = mejor_historico(corrida(SENO, semilla, tamano))
        costos.append(Individuo.evaluaciones)
        exitos += veredicto(campeon, optimo)
    tasa = exitos / CORRIDAS
    costo_medio = statistics.fmean(costos)
    intervalo = intervalo_wilson(exitos, CORRIDAS)
    ciegas_promedio = statistics.fmean(exito_ciegas(proporcion, costo) for costo in costos)
    barrido.append((tamano, exitos, tasa, intervalo, costo_medio, max(costos), ciegas_promedio))
    print(f"  {tamano:>10} | {tasa:6.1%} +/-{error_estandar(tasa, CORRIDAS):4.1%} "
          f"| [{intervalo[0]:.1%}, {intervalo[1]:.1%}]    "
          f"| {costo_medio:6.1f}, {max(costos):>4}    | {1000 * tasa / costo_medio:9.2f}")
# ------------------------------------------------------------------------------

# ===== Parte 1: la efectividad sube, la eficiencia baja =======================
(primer_tamano, _, primera_tasa, _, primer_costo, _, _), \
    (ultimo_tamano, _, ultima_tasa, _, ultimo_costo, _, _) \
    = barrido[0], barrido[-1]
print(f"\nLee la tabla en ambos sentidos. Desde población {primer_tamano} a "
      f"{ultimo_tamano}:")
print(f"    la tasa de éxito sube {primera_tasa:.1%} -> {ultima_tasa:.1%} "
      f"(+{ultima_tasa - primera_tasa:.1%}),")
print(f"    el costo de una corrida sube {primer_costo:.0f} -> {ultimo_costo:.0f} "
      f"evaluaciones (x{ultimo_costo / primer_costo:.1f}),")
print(f"    y el rendimiento baja {1000 * primera_tasa / primer_costo:.2f} -> "
      f"{1000 * ultima_tasa / ultimo_costo:.2f} éxitos por 1000 evaluaciones.")
print("La efectividad y la eficiencia tiran en direcciones opuestas, "
      "monótonamente, ambas.")
print("No hay una configuración que sea mejor en ambas - solo hay una elección,")
print("y la Lección 07 existe para hacerla.")

# ===== Parte 2: el punto de referencia que nadie ha ejecutado todavía =========
print(f"\nAhora gasta los mismos presupuestos a ciegas. Una extracción uniforme acierta el objetivo "
      f"{proporcion:.2%} del")
print("tiempo; E extracciones independientes aciertan al menos una vez con probabilidad "
      "1 - (1 - q) ** E:\n")
print("  presup. (evaluaciones) | extra. a ciegas | el AG con ese presup. | margen")
print("  -----------------------+-----------------+-----------------------+-------")
for tamano, _, tasa, _, costo_medio, _, ciegas in (barrido[0], barrido[-1]):
    print(f"  {costo_medio:>22.0f} | {ciegas:14.1%} | {tasa:20.1%} | "
          f"{tasa - ciegas:+5.1%}")
print(f"\nEn un objetivo de esta anchura - {proporcion:.2%} de toda la caja - la maquinaria "
      "de las Lecciones 01-05")
print("compra unos pocos puntos por encima de tirar los dados, no un abismo. Esa es una propiedad "
      "del problema,")
print("no un defecto del algoritmo: donde el objetivo no tiene un ancho que valga "
      "la pena nombrar - las")
print("mochilas, horarios y recorridos de las Lecciones 09 a 11 - una extracción a ciegas "
      "esencialmente nunca")
print("acierta, y los operadores son la única opción viable.")

# La imagen: tasa contra costo, con el punto de referencia a ciegas; y el rendimiento.
fig, (ax_tasa, ax_rendimiento) = plt.subplots(1, 2, figsize=(11.5, 4.0))
tamanos = [fila[0] for fila in barrido]
tasas = [fila[2] for fila in barrido]
costos = [fila[4] for fila in barrido]
intervalos = [fila[3] for fila in barrido]
errores = ([tasa - intervalo[0] for tasa, intervalo in zip(tasas, intervalos)],
           [intervalo[1] - tasa for tasa, intervalo in zip(tasas, intervalos)])

linea_presupuesto = np.linspace(min(costos) * 0.9, max(costos) * 1.05, 200)
ax_tasa.plot(linea_presupuesto, [exito_ciegas(proporcion, e) for e in linea_presupuesto],
             color="tab:gray", linestyle="--", linewidth=1.4,
             label="extr. a ciegas, mismo presupuesto")
ax_tasa.errorbar(costos, tasas, yerr=errores, fmt="o-", color="tab:blue",
                 capsize=4, linewidth=1.6, label="el AG, intervalos de Wilson al 95%")
for tamano, _, tasa, _, costo, _, _ in barrido:
    ax_tasa.annotate(f"pob {tamano}", xy=(costo, tasa),
                     xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=8)
ax_tasa.set_title("Tasa de éxito contra costo: el margen sobre las extracciones a ciegas es estrecho",
                  fontsize=10)
ax_tasa.set_xlabel("evaluaciones por corrida")
ax_tasa.set_ylabel("tasa de éxito")
ax_tasa.grid(True, linestyle=":", alpha=0.5)
ax_tasa.legend(fontsize=8, loc="lower right")

ax_rendimiento.plot(tamanos, [1000 * fila[2] / fila[4] for fila in barrido], "o-",
              color="tab:red", linewidth=1.6)
ax_rendimiento.set_title("El rendimiento baja a medida que la población crece", fontsize=10)
ax_rendimiento.set_xlabel("tamaño de población")
ax_rendimiento.set_ylabel("éxitos por 1000 evaluaciones")
ax_rendimiento.grid(True, linestyle=":", alpha=0.5)

fig.suptitle("La efectividad sube, la eficiencia baja - solo hay una elección")
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "tasa_exito_04_el_presupuesto.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/tasa_exito_04_el_presupuesto.png")

