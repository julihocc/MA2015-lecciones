"""
Leccion 07 - Paso 5: Poblacion, la perilla de presupuesto mas pura
==================================================================
NUEVO EN ESTE PASO: la tercera perilla - los propios cuatro tamanos de poblacion
del libro, medidos en los terminos del curso.

La cruza y la mutacion al menos fingen tratar sobre estrategia de busqueda.
El tamano de la poblacion no finge nada: una poblacion mas grande es mas evaluaciones,
punto. Es la perilla del presupuesto sin ningun disfraz, y el paso 4 de la Leccion 06
ya la atrapo una vez - la efectividad sube, la eficiencia cae. Este paso
la vuelve a medir dentro del estudio de afinacion, con las configuraciones del propio
libro (6, 10, 20, 50, mutacion 0.2 del libro), para cerrar el patron antes del
mapa final.

El patron se mantiene, y se agudiza. La poblacion 50 tiene exito en todas y cada una de
las 500 corridas - una puntuacion perfecta, comprada a 550 evaluaciones la corrida, donde el
rendimiento ha colapsado a una quinta parte de lo que la poblacion 6 entrega por 1000
evaluaciones. Y con el precio frente a la busqueda ciega, el margen alcanza su punto maximo en
la poblacion 10 y se desvanece en 50: a ese presupuesto, el AG y tirar dados
son la misma maquina. "¿Que poblacion es la mejor?" es una pregunta sin
respuesta hasta que se nombra una moneda - y nombrar monedas es la leccion.

CAMBIOS RESPECTO A afinacion_04_mutacion.py
Introducelos en este orden:
    1. la tercera perilla TAMANOS_POBLACION barridos, cruza 0.8 y mutacion 0.2 fijas

Ejecutalo:  python afinacion_05_poblacion.py

Población 50 registra 500/500 éxitos (intervalo de Wilson al 95%: 99.2%–100%) a 550 evals/corrida. El
rendimiento colapsa 9.25 → 1.82 éxitos por 1000 evals.
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
PROBABILIDAD_CRUZA = 0.8
PROBABILIDAD_MUTACION = 0.2
CORRIDAS = 500
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


@dataclass(frozen=True)
class Problema:
    """Un paisaje, su caja, y el presupuesto de generaciones con el que se corre.

    Args:
        argumentos del constructor.

    Example:
        Población 50 registra 500/500 éxitos; eso no prueba una tasa verdadera de 100%.
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
        Población 50 registra 500/500 éxitos; eso no prueba una tasa verdadera de 100%.
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
        Población 50 registra 500/500 éxitos; eso no prueba una tasa verdadera de 100%.
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
        Población 50 registra 500/500 éxitos; eso no prueba una tasa verdadera de 100%.
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
        Población 50 registra 500/500 éxitos; eso no prueba una tasa verdadera de 100%.
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


def intervalo_wilson(exitos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo aproximado al 95% que conserva incertidumbre en 0% y 100%."""
    tasa = exitos / n
    denominador = 1 + z ** 2 / n
    centro = (tasa + z ** 2 / (2 * n)) / denominador
    radio = z / denominador * (
        tasa * (1 - tasa) / n + z ** 2 / (4 * n ** 2)
    ) ** 0.5
    return centro - radio, centro + radio


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

# --- CAMBIADO --- la tercera perilla: los tamanos de poblacion del libro, cruza 0.8
# y la mutacion 0.2 del libro fijas
FIJAS = {"probabilidad_cruza": PROBABILIDAD_CRUZA,
         "probabilidad_mutacion": PROBABILIDAD_MUTACION}
PERILLA = "tamano_poblacion"
VALORES = (6, 10, 20, 50)

print(f"El objetivo: optimo {OPTIMO:+.6f}; una extraccion ciega lo acierta "
      f"{CUOTA:.2%} de las veces.")
print(f"El instrumento: {CORRIDAS} corridas por configuracion, cruza "
      f"{FIJAS['probabilidad_cruza']}, mutacion "
      f"{FIJAS['probabilidad_mutacion']} fija - los propios cuatro tamanos "
      f"del libro.\n")
print("  poblacion  | tasa de exito | evals/cor | ciego en mismo | margen |"
      " exitos por")
print("             |               |           | presupuesto    |        |"
      " 1000 evals")
print("  -----------+---------------+-----------+----------------+--------+"
      "--------------")
filas = medir(SENO, PERILLA, VALORES, FIJAS, CORRIDAS, OPTIMO)
for valor, tasa, error, costo in filas:
    ciego = exito_ciego(CUOTA, costo)
    print(f"  {valor:>10} | {tasa:6.1%} +/-{error:4.1%} | {costo:9.1f} | "
          f"{ciego:14.1%} | {tasa - ciego:+5.1%} | {1000 * tasa / costo:8.2f}")

primera, ultima = filas[0], filas[-1]
intervalo_ultima = intervalo_wilson(round(ultima[1] * CORRIDAS), CORRIDAS)
margenes = [r[1] - exito_ciego(CUOTA, r[3]) for r in filas]
mejor_margen = max(range(len(filas)), key=lambda i: margenes[i])
print(f"\nLa poblacion {ultima[0]} registra {CORRIDAS}/{CORRIDAS} éxitos; su intervalo "
      f"de Wilson al 95% es [{intervalo_ultima[0]:.1%}, {intervalo_ultima[1]:.1%}]. Esa puntuación observada se compra")
print(f"a {ultima[3]:.0f} evaluaciones la corrida. La poblacion {primera[0]} tiene exito "
      f"{primera[1]:.1%} de las veces a")
print(f"{primera[3]:.0f}. El rendimiento cae de {1000 * primera[1] / primera[3]:.2f} "
      f"a {1000 * ultima[1] / ultima[3]:.2f} exitos por")
print("1000 evaluaciones: un colapso de cinco veces para una subida de "
      f"{primera[1]:.1%} a {ultima[1]:.1%}.")
print(f"\nY el margen sobre la busqueda ciega alcanza su punto maximo en la poblacion "
      f"{filas[mejor_margen][0]} ({margenes[mejor_margen]:+.1%}) y desaparece en "
      f"{ultima[0]} ({margenes[-1]:+.1%}):")
print("a ese presupuesto el AG y tirar dados son la misma maquina.")
print("\nEntonces: ¿que poblacion es la mejor? No hay respuesta hasta que se")
print("nombre una moneda. La efectividad dice 50. La eficiencia dice 6. El margen dice 10.")
print("Y las perillas no actuan una a la vez - el paso 6 las mapea juntas.")

# La imagen: la tasa sube hacia la curva ciega; el rendimiento colapsa.
fig, (ax_tasa, ax_rendimiento) = plt.subplots(1, 2, figsize=(11.5, 4.0))
costos = [r[3] for r in filas]
tasas = [r[1] for r in filas]
linea_presupuesto = np.linspace(min(costos) * 0.9, max(costos) * 1.02, 300)
ax_tasa.plot(linea_presupuesto, [exito_ciego(CUOTA, e) for e in linea_presupuesto],
             color="tab:gray", linestyle="--", linewidth=1.4,
             label="extracciones ciegas, mismo presupuesto")
ax_tasa.errorbar(costos, tasas, yerr=[r[2] for r in filas], fmt="o-",
                 color="tab:blue", capsize=4, linewidth=1.6,
                 label="el AG, poblacion 6-50")
for valor, tasa, _, costo in filas:
    ax_tasa.annotate(f"pob {valor}", xy=(costo, tasa), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8)
ax_tasa.set_title("En la poblacion 50 el AG se fusiona con la curva ciega",
                  fontsize=10)
ax_tasa.set_xlabel("evaluaciones por corrida")
ax_tasa.set_ylabel("tasa de exito")
ax_tasa.grid(True, linestyle=":", alpha=0.5)
ax_tasa.legend(fontsize=8, loc="center right")

ax_rendimiento.plot([r[0] for r in filas], [1000 * r[1] / r[3] for r in filas], "o-",
              color="tab:red", linewidth=1.6)
ax_rendimiento.set_title("El rendimiento colapsa a medida que la poblacion crece", fontsize=10)
ax_rendimiento.set_xlabel("tamano de la poblacion")
ax_rendimiento.set_ylabel("exitos por 1000 evaluaciones")
ax_rendimiento.grid(True, linestyle=":", alpha=0.5)

fig.suptitle("Poblacion: la perilla de presupuesto sin disfraz")
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "afinacion_05_poblacion.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/afinacion_05_poblacion.png")

