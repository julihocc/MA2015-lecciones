"""
Leccion 07 - Paso 3: La perilla del presupuesto
===============================================
NUEVO EN ESTE PASO: el contador de evaluaciones, exito_ciego(), y la columna
de margen.

El Paso 2 midio la perilla de cruza y encontro una subida monotona limpia: 55.6%
en 0.0 a 83.0% en 1.0. Cerro con la columna faltante. La cruza no
viene gratis - cada par cruzado crea dos nuevos individuos, y cada
nuevo individuo es una evaluacion de aptitud, la unica moneda que gasta un AG. Asi
que el dial que "mejora" el algoritmo tambien infla su presupuesto, y la
pregunta se vuelve: ¿cuanto de la subida es mejor busqueda, y cuanto es
simplemente mas gasto?

El instrumento para esa pregunta es el de la Leccion 06: contar las evaluaciones, luego
fijar el precio de cada configuracion contra extracciones uniformes ciegas dado el mismo presupuesto. La
respuesta, en este paisaje, es incomoda. La perilla es una perilla de presupuesto: el
margen sobre la busqueda ciega se reduce monotonicamente a medida que se sube la cruza,
y las cien evaluaciones que separan los dos extremos del dial le compran al
AG menos de lo que le habrian comprado cien extracciones ciegas.

CAMBIOS RESPECTO A afinacion_02_el_instrumento.py
Introducelos en este orden:
    1. el contador de evaluaciones Individuo cuenta; medir() recolecta por configuracion
    2. exito_ciego()               el punto de referencia de igual presupuesto: 1 - (1 - q) ** E
    3. la columna de margen        tasa menos ciego con el mismo presupuesto

Ejecutalo:  python afinacion_03_la_perilla_del_presupuesto.py

Cruza 0.0 gasta 20 evals/corrida, 1.0 gasta 120. El margen sobre ciego cae
+30.6 → +0.8 puntos: la perilla es de presupuesto.
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
PROBABILIDAD_MUTACION = 0.1
CORRIDAS = 500
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


@dataclass(frozen=True)
class Problema:
    """Un paisaje, su caja, y el presupuesto de generaciones con el que se corre.

    Args:
        argumentos del constructor.

    Example:
        Cruza 0.0 gasta 20 evals/corrida, 1.0 gasta 120. El margen sobre ciego cae
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


# --- NUEVO (1) el contador de evaluaciones -------------------------------------------
class Individuo:
    """Una solucion candidata, que lleva el problema que la juzga.

    `evaluaciones` cuenta cada evaluacion de aptitud que el algoritmo realiza:
    se incrementa cada vez que se crea un Individuo, y se reinicia a cero
    antes de cada corrida. La aptitud es lo unico por lo que un AG paga, asi que este
    contador ES el costo de una corrida - y de una configuracion de perilla.

    Args:
        argumentos del constructor.

    Example:
        Cruza 0.0 gasta 20 evals/corrida, 1.0 gasta 120. El margen sobre ciego cae
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
        Cruza 0.0 gasta 20 evals/corrida, 1.0 gasta 120. El margen sobre ciego cae
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
        Cruza 0.0 gasta 20 evals/corrida, 1.0 gasta 120. El margen sobre ciego cae
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
        Cruza 0.0 gasta 20 evals/corrida, 1.0 gasta 120. El margen sobre ciego cae
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
    """El instrumento de barrido, ahora con la columna de costo: por configuracion devuelve
    (valor, tasa de exito, error estandar, evaluaciones promedio por corrida)."""
    filas = []
    for valor in valores:
        configuraciones = {**fijas, perilla: valor}
        exitos, costos = 0, []
        for semilla in range(corridas):
            Individuo.evaluaciones = 0          # NUEVO (1): recolectado por corrida
            campeon = correr(problema, semilla, **configuraciones)
            costos.append(Individuo.evaluaciones)
            exitos += veredicto(campeon, optimo)
        tasa = exitos / corridas
        filas.append((valor, tasa, error_estandar(tasa, corridas),
                     statistics.fmean(costos)))
    return filas


# --- NUEVO (2) exito_ciego() --------------------------------------------------
def exito_ciego(cuota: float, evaluaciones: float) -> float:
    """La tasa de exito de gastar `evaluaciones` extracciones uniformes ciegas.

    Cada extraccion da en el blanco con probabilidad `cuota`, de forma independiente, por lo que la
    probabilidad de que al menos una de E extracciones acierte es 1 - (1 - cuota) ** E.
    Este es el punto de referencia que una configuracion de perilla tiene que superar para justificar su presupuesto:
    el mismo dinero, sin operadores, sin memoria, sin poblacion.

    Args:
        cuota, evaluaciones.

    Returns:
        float.

    Example:
        Cruza 0.0 gasta 20 evals; 100 extra ciegas compran 57.2 puntos.
    """
    return 1.0 - (1.0 - cuota) ** evaluaciones
# ------------------------------------------------------------------------------


OPTIMO = optimo_fuerza_bruta(SENO, 20001)
CUOTA = cuota_dentro_de_tolerancia(SENO, OPTIMO, 20001)
FIJAS = {"tamano_poblacion": TAMANO_POBLACION,
         "probabilidad_mutacion": PROBABILIDAD_MUTACION}
VALORES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

print(f"El objetivo: optimo {OPTIMO:+.6f}; una extraccion ciega lo acierta "
      f"{CUOTA:.2%} de las veces.")
print(f"El instrumento: {CORRIDAS} corridas por configuracion, poblacion "
      f"{FIJAS['tamano_poblacion']}, mutacion "
      f"{FIJAS['probabilidad_mutacion']} fija - y ahora cada corrida tiene "
      f"precio.\n")

# --- NUEVO (3) la columna de margen ------------------------------------------------
print("  cruza     | tasa de exito | evals/cor | ciego en mismo | margen")
print("            |               |           | presupuesto    |")
print("  ----------+---------------+-----------+----------------+--------")
filas = medir(SENO, "probabilidad_cruza", VALORES, FIJAS, CORRIDAS, OPTIMO)
for valor, tasa, error, costo in filas:
    ciego = exito_ciego(CUOTA, costo)
    print(f"  {valor:>9} | {tasa:6.1%} +/-{error:4.1%} | {costo:9.1f} | "
          f"{ciego:14.1%} | {tasa - ciego:+5.1%}")
# ------------------------------------------------------------------------------

primera, ultima = filas[0], filas[-1]
evals_extra = ultima[3] - primera[3]
ganancia_ag = ultima[1] - primera[1]
ganancia_ciega = exito_ciego(CUOTA, ultima[3]) - exito_ciego(CUOTA, primera[3])
print(f"\nCada giro de la perilla compra evaluaciones: cruza {primera[0]} "
      f"gasta {primera[3]:.0f} por corrida,")
print(f"cruza {ultima[0]} gasta {ultima[3]:.0f}. El margen sobre la busqueda ciega "
      f"a igual presupuesto")
print(f"se reduce monotonicamente, de {primera[1] - exito_ciego(CUOTA, primera[3]):+.1%} "
      f"a {ultima[1] - exito_ciego(CUOTA, ultima[3]):+.1%}.")
print(f"\nY las {evals_extra:.0f} evaluaciones entre los dos extremos le compran al "
      f"AG {ganancia_ag:.1%} - mientras que las")
print(f"mismas {evals_extra:.0f} gastadas a ciegas habrian comprado {ganancia_ciega:.1%}. "
      f"En este paisaje,")
print("el dial de cruza no es un dial de calidad. Es un dial de presupuesto, y uno")
print("despilfarrador.")
print("\nAntes de descartar la cruza, recuerda la moneda: este objetivo es")
print(f"amplio ({CUOTA:.2%} de la caja), asi que el dinero a ciegas llega lejos aqui. Donde no")
print("llega a ninguna parte - los problemas combinatorios de las Lecciones 09 a 11 - los")
print("operadores son la unica opcion. Y la segunda perilla cuenta una")
print("historia diferente. Paso 4.")

# La imagen: la subida contra el costo, con la curva ciega debajo; y
# el margen, cayendo.
fig, (ax_tasa, ax_margen) = plt.subplots(1, 2, figsize=(11.5, 4.0))
costos = [r[3] for r in filas]
tasas = [r[1] for r in filas]
linea_presupuesto = np.linspace(min(costos) * 0.9, max(costos) * 1.05, 200)
ax_tasa.plot(linea_presupuesto, [exito_ciego(CUOTA, e) for e in linea_presupuesto],
             color="tab:gray", linestyle="--", linewidth=1.4,
             label="extracciones ciegas, mismo presupuesto")
ax_tasa.errorbar(costos, tasas, yerr=[r[2] for r in filas], fmt="o-",
                 color="tab:blue", capsize=4, linewidth=1.6,
                 label="el AG, cruza 0.0-1.0")
for valor, tasa, _, costo in filas:
    ax_tasa.annotate(f"pc {valor}", xy=(costo, tasa), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8)
ax_tasa.set_title("La subida es principalmente gasto", fontsize=10)
ax_tasa.set_xlabel("evaluaciones por corrida")
ax_tasa.set_ylabel("tasa de exito")
ax_tasa.grid(True, linestyle=":", alpha=0.5)
ax_tasa.legend(fontsize=8, loc="lower right")

margenes = [r[1] - exito_ciego(CUOTA, r[3]) for r in filas]
ax_margen.bar([str(r[0]) for r in filas], margenes, color="tab:red", alpha=0.8)
ax_margen.set_title("El margen sobre ciegas se reduce al abrir la perilla",
                    fontsize=10)
ax_margen.set_xlabel("probabilidad de cruza")
ax_margen.set_ylabel("tasa de exito menos ciego, mismo presupuesto")
ax_margen.grid(True, axis="y", linestyle=":", alpha=0.5)

fig.suptitle("La perilla de cruza es una perilla de presupuesto")
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "afinacion_03_la_perilla_del_presupuesto.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/afinacion_03_la_perilla_del_presupuesto.png")

