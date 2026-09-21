"""
Leccion 07 - Paso 4: Mutacion, y donde se esconde la curva de campana
=====================================================================
NUEVO EN ESTE PASO: la segunda perilla. El instrumento no cambia - ese es
el punto de haberlo construido.

Todos los libros de texto dibujan la curva de mutacion como una campana: muy poca mutacion y
la busqueda se estanca, demasiada y degenera en una caminata aleatoria, con un
punto optimo en el medio. El Capitulo 7 de Gridin muestra corridas sueltas en 0.0, 0.3 y
1.0 y hace gestos hacia la misma forma. Este paso mide la perilla correctamente -
siete configuraciones, 500 corridas cada una, cruza fija en 0.8 - y la curva de campana
no esta ahi. La tasa de exito bruta sube monotonicamente hasta
la mutacion 1.0.

Luego, la columna de margen dice la verdad. Con el precio frente a la busqueda ciega al
mismo presupuesto, la curva SI se dobla: negativa en mutacion 0.0 (el AG sin
mutacion es peor que tirar dados), un pico ancho alrededor de 0.2, negativa
otra vez en 1.0. La curva de campana que prometen los libros de texto existe - pero vive en
el margen sobre la busqueda ciega, donde se pone precio al presupuesto, no en la tasa
bruta, donde cada evaluacion extra parece progreso.

CAMBIOS RESPECTO A afinacion_03_la_perilla_del_presupuesto.py
Introducelos en este orden:
    1. la segunda perilla VALORES_PM barridos, cruza fija en 0.8

Ejecutalo:  python afinacion_04_mutacion.py

La tasa bruta es monótona hasta mutación 1.0 (61.8% → 89.8%). El margen se
dobla: -10.9% en 0.0, pico +5.9% cerca de 0.2, -3.7% en 1.0.
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
TAMANO_POBLACION = 10
PROBABILIDAD_CRUZA = 0.8
CORRIDAS = 500
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


@dataclass(frozen=True)
class Problema:
    """Un paisaje, su caja, y el presupuesto de generaciones con el que se corre.

    Args:
        argumentos del constructor.

    Example:
        La tasa bruta es monótona hasta mutación 1.0 (61.8% → 89.8%). El margen se
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

    `evaluaciones` cuenta cada evaluacion de aptitud: el costo de una corrida, y de
    una configuracion de perilla.

    Args:
        argumentos del constructor.

    Example:
        La tasa bruta es monótona hasta mutación 1.0 (61.8% → 89.8%). El margen se
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
        La tasa bruta es monótona hasta mutación 1.0 (61.8% → 89.8%). El margen se
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
        La tasa bruta es monótona hasta mutación 1.0 (61.8% → 89.8%). El margen se
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
        La tasa bruta es monótona hasta mutación 1.0 (61.8% → 89.8%). El margen se
    """
    return Individuo([g + random.gauss(MUTACION_MU, sigma)
                       for g in individuo.genes], individuo.problema)


def correr(problema: Problema, semilla: int, tamano_poblacion: int,
        probabilidad_cruza: float, probabilidad_mutacion: float
        ) -> Individuo:
    """Una corrida completa, devolviendo su respuesta: el mejor individuo historico.

    La corrida i usa la semilla i, asi que cada configuracion enfrenta los mismos dados.

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


def cuota_dentro_de_tolerancia(problema: Problema, optimo: float,
                           puntos_por_eje: int) -> float:
    """Que fraccion de la caja puntua dentro de TOLERANCIA del optimo -
    tambien la probabilidad de acierto de una extraccion ciega."""
    eje = np.linspace(problema.limite_inferior, problema.limite_superior, puntos_por_eje)
    valores = np.array([problema.aptitud([float(x)]) for x in eje])
    return float(np.mean(valores >= optimo - TOLERANCIA))


def error_estandar(tasa: float, n: int) -> float:
    """La oscilacion de una tasa medida: sqrt(p(1-p)/n).

    Args:
        tasa, n.

    Returns:
        float.

    Example:
        8 corridas ±17.1 puntos; 1000 corridas ±1.3 puntos.
    """
    return (tasa * (1 - tasa) / n) ** 0.5


def medir(problema: Problema, perilla: str, valores: tuple, fijas: dict,
            corridas: int, optimo: float) -> list[tuple]:
    """El instrumento de barrido: por configuracion, (valor, tasa de exito, error estandar,
    evaluaciones promedio por corrida). Mismas semillas para cada configuracion."""
    filas = []
    for valor in valores:
        configuraciones = {**fijas, perilla: valor}
        exitos, costos = 0, []
        for semilla in range(corridas):
            Individuo.evaluaciones = 0
            campeon = correr(problema, semilla, **configuraciones)
            costos.append(Individuo.evaluaciones)
            exitos += veredicto(campeon, optimo)
        tasa = exitos / corridas
        filas.append((valor, tasa, error_estandar(tasa, corridas),
                     statistics.fmean(costos)))
    return filas


def exito_ciego(cuota: float, evaluaciones: float) -> float:
    """La tasa de exito de gastar `evaluaciones` extracciones uniformes ciegas:
    1 - (1 - cuota) ** E. El punto de referencia que una configuracion tiene que superar
    para justificar su presupuesto."""
    return 1.0 - (1.0 - cuota) ** evaluaciones


OPTIMO = optimo_fuerza_bruta(SENO, 20001)
CUOTA = cuota_dentro_de_tolerancia(SENO, OPTIMO, 20001)

# --- CAMBIADO --- la segunda perilla: mutacion barrida, cruza fija en 0.8
FIJAS = {"tamano_poblacion": TAMANO_POBLACION,
         "probabilidad_cruza": PROBABILIDAD_CRUZA}
PERILLA = "probabilidad_mutacion"
VALORES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)

print(f"El objetivo: optimo {OPTIMO:+.6f}; una extraccion ciega lo acierta "
      f"{CUOTA:.2%} de las veces.")
print(f"El instrumento: {CORRIDAS} corridas por configuracion, poblacion "
      f"{FIJAS['tamano_poblacion']}, cruza "
      f"{FIJAS['probabilidad_cruza']} fija.\n")
print("  mutacion | tasa de exito | evals/cor | ciego en mismo | margen")
print("           |               |           | presupuesto    |")
print("  ---------+---------------+-----------+----------------+--------")
filas = medir(SENO, PERILLA, VALORES, FIJAS, CORRIDAS, OPTIMO)
for valor, tasa, error, costo in filas:
    ciego = exito_ciego(CUOTA, costo)
    print(f"  {valor:>8} | {tasa:6.1%} +/-{error:4.1%} | {costo:9.1f} | "
          f"{ciego:14.1%} | {tasa - ciego:+5.1%}")

tasas = [r[1] for r in filas]
margenes = [r[1] - exito_ciego(CUOTA, r[3]) for r in filas]
mejor_margen = max(range(len(filas)), key=lambda i: margenes[i])
print(f"\nLa tasa bruta sube monotonicamente, {tasas[0]:.1%} a {tasas[-1]:.1%}"
      f" - sin curva de campana. Lee")
print("solo esa columna y la mutacion 1.0 gana, y la forma del libro de texto")
print("no esta en ningun lado. Pero la columna de margen se dobla: negativa en mutacion 0.0")
print(f"({margenes[0]:+.1%} - sin mutacion el AG es PEOR que las extracciones "
      f"ciegas a igual presupuesto),")
print(f"un pico ancho alrededor de mutacion {filas[mejor_margen][0]} "
      f"({margenes[mejor_margen]:+.1%}), negativa de nuevo en "
      f"1.0 ({margenes[-1]:+.1%}).")
print("\nLa curva de campana que prometen los libros de texto existe. Vive en el margen")
print("sobre la busqueda ciega, donde se valora el presupuesto - no en la tasa bruta,")
print("donde cada evaluacion extra parece progreso.")
print("\nY la mutacion 0.0 es la advertencia que hizo la Leccion 05, ahora con precio: sin")
print("mutacion la poblacion solo puede remezclar sus genes iniciales, y una vez que")
print("converge, vuelve a gastar su presupuesto re-evaluando casi duplicados.")
print("\nLa tercera perilla es la perilla de presupuesto mas pura de todas. Paso 5.")

# La imagen: la tasa sube y nunca se dobla; el margen si.
fig, (ax_tasa, ax_margen) = plt.subplots(1, 2, figsize=(11.5, 4.0))
valores = [r[0] for r in filas]
ax_tasa.errorbar(valores, tasas, yerr=[r[2] for r in filas], fmt="o-",
                 color="tab:blue", capsize=4, linewidth=1.6)
ax_tasa.set_title("La tasa bruta: monotona - no hay curva de campana aqui", fontsize=10)
ax_tasa.set_xlabel("probabilidad de mutacion")
ax_tasa.set_ylabel("tasa de exito")
ax_tasa.grid(True, linestyle=":", alpha=0.5)

ax_margen.axhline(0.0, color="black", linewidth=0.8)
ax_margen.plot(valores, margenes, "o-", color="tab:red", linewidth=1.6)
ax_margen.annotate(f"pico cerca de {filas[mejor_margen][0]}",
                   xy=(filas[mejor_margen][0], margenes[mejor_margen]),
                   xytext=(14, -4), textcoords="offset points", ha="left",
                   fontsize=8)
ax_margen.set_title("El margen sobre busqueda ciega: la curva de campana vive aqui",
                    fontsize=10)
ax_margen.set_xlabel("probabilidad de mutacion")
ax_margen.set_ylabel("tasa de exito menos ciego, mismo presupuesto")
ax_margen.grid(True, linestyle=":", alpha=0.5)

fig.suptitle("Mutacion: medida en la moneda equivocada, parece gratis")
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "afinacion_04_mutacion.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/afinacion_04_mutacion.png")

