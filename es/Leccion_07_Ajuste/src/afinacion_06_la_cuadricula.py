"""
Leccion 07 - Paso 6: La cuadricula, y el veredicto
==================================================
NUEVO EN ESTE PASO: las dos perillas mapeadas juntas, y el veredicto de la leccion.

Los pasos 2 a 5 giraron una perilla a la vez y cada perilla confeso ser una
perilla de presupuesto. La pregunta final honesta es si interactuan - si
la mejor cruza depende de la configuracion de mutacion, de la manera que los libros de texto
asumen cuando recomiendan pares como "0.8 y 0.1". Este paso mide la
cuadricula completa: seis configuraciones de cruza por seis configuraciones de mutacion, poblacion
10, 100 corridas por celda, cada celda valorada contra la busqueda ciega en su propio
presupuesto.

Salen a la luz tres cosas. La esquina con ambas perillas en cero recupera la busqueda
ciega casi exactamente - la seleccion por si sola ES diez extracciones ciegas, y el
instrumento esta calibrado por ello. La mejor tasa bruta se encuentra en la esquina
mas cara, y su margen sobre ciegas a ese presupuesto es un error de
redondeo. Y el mapa de margenes muestra donde los operadores realmente ganan su
sustento: las celdas baratas, la region de bajo presupuesto donde el dinero a ciegas aun no ha
alcanzado. En este paisaje las perillas interactuan principalmente a traves del presupuesto.
Ese es el veredicto - y es un veredicto SOBRE ESTE PROBLEMA, que es por lo que
la afinacion es un tema de laboratorio y por lo que la Leccion 12, el AG adaptativo, existe.

CAMBIOS RESPECTO A afinacion_05_poblacion.py
Introducelos en este orden:
    1. la cuadricula  cada celda (cruza, mutacion) medida y valorada
    2. CORRIDAS = 100 cambiado de 500: treinta y seis celdas tienen que encajar en el reloj

ELIMINADO DE afinacion_05_poblacion.py: la llamada de barrido de una sola perilla y su
figura (el estudio de poblacion es del paso 5, y sus numeros estan en el README).

Ejecutalo:  python afinacion_06_la_cuadricula.py

La esquina (0, 0) recupera la búsqueda ciega; la mejor celda (1.0, 0.5)
puntúa 90.0% a 160 evals. Solo 20 de 36 celdas superan su presupuesto.
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
CORRIDAS = 100   # --- CAMBIADO --- de 500: treinta y seis celdas tienen que encajar en el reloj
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


@dataclass(frozen=True)
class Problema:
    """Un paisaje, su caja, y el presupuesto de generaciones con el que se corre.

    Args:
        argumentos del constructor.

    Example:
        La esquina (0, 0) recupera la búsqueda ciega; la mejor celda (1.0, 0.5) puntúa 90.0%.
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
        La esquina (0, 0) recupera la búsqueda ciega; la mejor celda (1.0, 0.5) puntúa 90.0%.
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
        La esquina (0, 0) recupera la búsqueda ciega; la mejor celda (1.0, 0.5) puntúa 90.0%.
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
        La esquina (0, 0) recupera la búsqueda ciega; la mejor celda (1.0, 0.5) puntúa 90.0%.
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
        La esquina (0, 0) recupera la búsqueda ciega; la mejor celda (1.0, 0.5) puntúa 90.0%.
    """
    return Individuo([g + random.gauss(MUTACION_MU, sigma)
                       for g in individuo.genes], individuo.problema)


def correr(problema: Problema, semilla: int, tamano_poblacion: int,
        probabilidad_cruza: float, probabilidad_mutacion: float
        ) -> Individuo:
    """Una corrida completa, devolviendo su respuesta: el mejor individuo historico.

    La corrida i usa la semilla i, asi que cada celda enfrenta los mismos dados.

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


def exito_ciego(cuota: float, evaluaciones: float) -> float:
    """La tasa de exito de gastar `evaluaciones` extracciones uniformes ciegas:
    1 - (1 - cuota) ** E. El punto de referencia que una celda tiene que superar para justificar su
    presupuesto."""
    return 1.0 - (1.0 - cuota) ** evaluaciones


OPTIMO = optimo_fuerza_bruta(SENO, 20001)
CUOTA = cuota_dentro_de_tolerancia(SENO, OPTIMO, 20001)
VALORES_PC = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
VALORES_PM = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)

print(f"El objetivo: optimo {OPTIMO:+.6f}; una extraccion ciega lo acierta "
      f"{CUOTA:.2%} de las veces.")
print(f"La cuadricula: {len(VALORES_PC)} x {len(VALORES_PM)} celdas, poblacion "
      f"{TAMANO_POBLACION}, {CORRIDAS} corridas por celda, cada celda valorada.\n")

# --- NUEVO (1) la cuadricula ----------------------------------------------------------
tasas, costos = {}, {}
for pc in VALORES_PC:
    for pm in VALORES_PM:
        exitos, costos_celda = 0, []
        for semilla in range(CORRIDAS):
            Individuo.evaluaciones = 0
            campeon = correr(SENO, semilla, TAMANO_POBLACION, pc, pm)
            costos_celda.append(Individuo.evaluaciones)
            exitos += veredicto(campeon, OPTIMO)
        tasas[(pc, pm)] = exitos / CORRIDAS
        costos[(pc, pm)] = statistics.fmean(costos_celda)
# ------------------------------------------------------------------------------

print("  tasa de exito  |" + "|".join(f" pm = {pm:>4} " for pm in VALORES_PM))
print("  ---------------+" + "+".join("-" * 10 for _ in VALORES_PM))
for pc in VALORES_PC:
    print(f"  pc = {pc:>4}    |" + "|".join(f" {tasas[(pc, pm)]:7.1%}  "
                                             for pm in VALORES_PM))

cero = (0.0, 0.0)
ciego_cero = exito_ciego(CUOTA, costos[cero])
ee_cero = error_estandar(tasas[cero], CORRIDAS)
print(f"\nLa esquina de calibracion, pc = 0 y pm = 0: el AG puntua "
      f"{tasas[cero]:.1%} +/- {ee_cero:.1%}.")
print(f"Sin cruza y sin mutacion, la seleccion por si sola solo puede re-muestrear "
      f"la poblacion")
print(f"inicial y reportar su mejor - literalmente {costos[cero]:.0f} extracciones ciegas, "
      f"que puntuan {ciego_cero:.1%}.")
print("Dentro de un EE por sustitución de una celda de 100 corridas, el instrumento mide el AG")
print("y los dados como la misma maquina. Esta calibrado.")

mejor = max(tasas, key=tasas.get)
ciego_mejor = exito_ciego(CUOTA, costos[mejor])
print(f"\nLa mejor tasa bruta se encuentra en pc = {mejor[0]}, pm = {mejor[1]}: "
      f"{tasas[mejor]:.1%}. Es tambien la")
print(f"celda mas cara, a {costos[mejor]:.0f} evaluaciones una corrida - y "
      f"extracciones ciegas a ese presupuesto")
print(f"puntuan {ciego_mejor:.1%}. El margen del campeon: "
      f"{tasas[mejor] - ciego_mejor:+.2%} - dentro del redondeo a cero.")
positivas = sum(1 for celda in tasas
               if tasas[celda] > exito_ciego(CUOTA, costos[celda]))
print(f"\nDe las {len(tasas)} celdas, {positivas} superan a la busqueda ciega en su "
      f"propio presupuesto. El mapa")
print("muestra donde: las celdas baratas. Los operadores ganan su sustento donde el")
print("presupuesto es pequeno; dondequiera que crezca el presupuesto, el dinero a ciegas los alcanza.")
print("\nEse es el veredicto de la leccion, y es un veredicto SOBRE ESTE")
print("PROBLEMA. En un paisaje cuyo objetivo es amplio, las perillas son perillas de presupuesto")
print("y la afinacion es principalmente contabilidad. En los objetivos estrechos de las Lecciones 09")
print("a 11 la misma medicion vale la pena volver a ejecutarla - y las configuraciones")
print("que corona alli seran diferentes. Una configuracion fija es siempre un")
print("compromiso a traves de los problemas y a traves de la corrida misma, que es")
print("exactamente por lo que la Leccion 12, el AG adaptativo, existe.")

# La imagen: el mapa de tasas y el mapa de margenes, lado a lado.
cuadricula_tasas = np.array([[tasas[(pc, pm)] for pm in VALORES_PM]
                      for pc in VALORES_PC])
cuadricula_margenes = np.array([[tasas[(pc, pm)] - exito_ciego(CUOTA, costos[(pc, pm)])
                         for pm in VALORES_PM] for pc in VALORES_PC])
fig, (ax_tasa, ax_margen) = plt.subplots(1, 2, figsize=(11.5, 4.2))
for ax, cuadricula, titulo, fmt in (
        (ax_tasa, cuadricula_tasas, "tasa de exito", ".0%"),
        (ax_margen, cuadricula_margenes, "margen sobre ciegas, mismo presupuesto", "+.0%")):
    imagen = ax.imshow(cuadricula, cmap="RdYlGn", aspect="auto",
                      vmin=-np.max(np.abs(cuadricula)) if cuadricula is cuadricula_margenes else 0,
                      vmax=np.max(np.abs(cuadricula)) if cuadricula is cuadricula_margenes else 1)
    for i, pc in enumerate(VALORES_PC):
        for j, pm in enumerate(VALORES_PM):
            ax.text(j, i, format(cuadricula[i, j], fmt), ha="center", va="center",
                    fontsize=7)
    ax.set_xticks(range(len(VALORES_PM)), [str(v) for v in VALORES_PM])
    ax.set_yticks(range(len(VALORES_PC)), [str(v) for v in VALORES_PC])
    ax.set_xlabel("probabilidad de mutacion")
    ax.set_ylabel("probabilidad de cruza")
    ax.set_title(titulo, fontsize=10)
    fig.colorbar(imagen, ax=ax, shrink=0.85)
ax_tasa.set_title("tasa de exito: la cresta es el presupuesto", fontsize=10)
ax_margen.set_title("margen: los operadores lo ganan en celdas baratas",
                    fontsize=10)
fig.suptitle("Las dos perillas juntas, medidas y con precio")
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "afinacion_06_la_cuadricula.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/afinacion_06_la_cuadricula.png")

