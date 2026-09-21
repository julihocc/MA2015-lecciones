"""
Lección 06 - Paso 2: Qué cuenta como éxito
==========================================
NUEVO EN ESTE PASO: optimo_fuerza_bruta(), TOLERANCIA y veredicto(),
proporcion_dentro_de_tolerancia(), y un segundo problema - el paisaje de un gen
de la Lección 01.

El paso 1 mostró ocho respuestas a una pregunta. Antes de que puedan promediarse
en algo, se debe definir una palabra: éxito. Una corrida no puede juzgarse a sí misma -
reporta lo mejor que encontró, nunca qué tan bueno fue eso - por lo que el juicio
tiene que venir de afuera, y el paso 5 de la Lección 02 construyó exactamente
ese instrumento: fuerza bruta sobre el paisaje en una cuadrícula, y medir cada corrida
contra el resultado. También dejó una advertencia adjunta, en sus propias líneas finales:
esa referencia "funciona porque x es un número; a partir de la Lección 08 no lo es,
y la respuesta honesta a '¿tuvo éxito esta corrida?' se vuelve genuinamente difícil de obtener".

Este paso es donde se cobra esa factura, una lección antes de lo anunciado,
porque el propio paisaje del Capítulo 6 tiene dos genes en lugar de uno. Dos cosas
se rompen a la vez, y solo una de ellas es la obvia.

La obvia es el costo: una cuadrícula lo suficientemente fina para resolver un gen
necesita que su resolución se eleve a la potencia del número de genes. La cuadrícula de
20,001 puntos de la Lección 02 se convierte en 400 millones para dos genes y es inescribible para diez.

La otra es peor, y es el mismo defecto que la Lección 05 encontró en la escalera de
la Lección 02, usando ropa diferente. El óptimo del paisaje 2-D existe, y
no tiene volumen: el término de penalización desaparece solo en dos líneas, y se acerca
a cero como una décima potencia, por lo que para puntuar dentro de 0.01 del óptimo un punto
debe estar dentro de 1e-20 de una línea. La cuadrícula encuentra ese óptimo solo porque la
línea resulta pasar a través de ella. Ninguna búsqueda aterrizará en ella, y ningún veredicto
construido sobre ella puede distinguir una buena corrida de una mala - reporta fracaso para ambas.

Así que la lección se mueve a un paisaje donde la palabra significa algo:
f(x) = sin(x) - 0.2|x| de la Lección 01, cuyo óptimo es un pico suave con un ancho medible.
Todo de aquí en adelante se mide allí. El problema 2-D se queda en el archivo como
el espécimen - la razón de la mudanza tiene que permanecer visible.

CAMBIOS RESPECTO A tasa_exito_01_una_corrida.py
Introdúcelos en este orden:
    1. optimo_fuerza_bruta()               el instrumento de la Lección 02: la verdad, desde fuera de la corrida
    2. TOLERANCIA                          qué tan cerca cuenta como llegada, y veredicto() lo aplica
    3. proporcion_dentro_de_tolerancia()   la comprobación que enseñó la Lección 05: ¿tiene volumen el objetivo?
    4. SENO                                el paisaje de un gen de la Lección 01, donde el veredicto sobrevive

Ejecútalo:  python tasa_exito_02_el_veredicto.py

El óptimo 2-D es real (+0.500000) pero ocupa 0.003197% de la cuadrícula, así
que 0 de 8 corridas tienen éxito. En seno 1-D, 5 de 8: 62.5% ± 17.1%.
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
# La corrida i usa la semilla i, para toda la lección.
CORRIDAS = 8
FIGURAS = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class Problema:
    """Un paisaje, su caja y el presupuesto de generaciones con el que se ejecuta.

    Args:
        argumentos del constructor.

    Example:
        El óptimo 2-D es real (+0.500000) pero ocupa 0.003197% de la cuadrícula, así
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


# --- NUEVO (4) SENO -----------------------------------------------------------
def paisaje_seno(genes: list[float]) -> float:
    """El paisaje de la Lección 01, sin cambios: un gen, un pico global suave.

    Elegido sobre la función 2-D del libro por una razón medida, impresa abajo:
    su óptimo ocupa un tramo de x lo suficientemente ancho para ser encontrado. Todo lo que
    el resto de esta lección cuenta se cuenta aquí.

    Args:
        genes.

    Returns:
        float.

    Example:
        Objetivo 1.4299% de la caja; 5 de 8 corridas lo encuentran (62.5% ± 17.1%).
    """
    return float(np.sin(genes[0]) - 0.2 * abs(genes[0]))


# Diez generaciones en lugar de veinticinco: suficiente para tener éxito la mayor parte del tiempo
# y fallar con la frecuencia suficiente para que valga la pena contar las fallas. Un problema que
# siempre se resuelve no mide nada.
SENO = Problema("seno 1-D", paisaje_seno, -10.0, 10.0, 1, 10)
# ------------------------------------------------------------------------------


class Individuo:
    """Una solución candidata, que lleva el problema que la juzga.

    Args:
        argumentos del constructor.

    Example:
        El óptimo 2-D es real (+0.500000) pero ocupa 0.003197% de la cuadrícula, así
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
        El óptimo 2-D es real (+0.500000) pero ocupa 0.003197% de la cuadrícula, así
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
        El óptimo 2-D es real (+0.500000) pero ocupa 0.003197% de la cuadrícula, así
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
        El óptimo 2-D es real (+0.500000) pero ocupa 0.003197% de la cuadrícula, así
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
        El óptimo 2-D es real (+0.500000) pero ocupa 0.003197% de la cuadrícula, así
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
        El óptimo 2-D es real (+0.500000) pero ocupa 0.003197% de la cuadrícula, así
    """
    return max((ind for pop in generaciones for ind in pop),
               key=lambda i: i.aptitud)


# --- NUEVO (1) optimo_fuerza_bruta() ------------------------------------------
def optimo_fuerza_bruta(problema: Problema, puntos_por_eje: int) -> float:
    """El mejor valor en una cuadrícula regular sobre toda la caja.

    El instrumento del paso 5 de la Lección 02, generalizado a más de un gen - que es
    donde comienza a doler. La cuadrícula tiene puntos_por_eje ** problema.genes
    puntos, por lo que aumentar la resolución de un problema de dos genes cuesta el cuadrado de
    lo que cuesta para un gen. Ese exponente es toda la dificultad, y es
    por qué esta función toma la resolución como argumento en lugar de ocultarla.

    Args:
        problema, puntos_por_eje.

    Returns:
        float.

    Example:
        Seno 1-D: +0.705908. Libro 2-D: +0.500000 sin volumen.
    """
    eje = np.linspace(problema.bajo, problema.alto, puntos_por_eje)
    if problema.genes == 1:
        return float(max(problema.aptitud([float(x)]) for x in eje))
    malla = np.meshgrid(*[eje] * problema.genes, indexing="ij")
    x, y = malla
    valores = np.sin(x) * np.cos(x) - np.power(
        np.abs((x + 50.0) * (y - 10.0)) / 10.0, 0.1)
    return float(np.max(valores))
# ------------------------------------------------------------------------------


# --- NUEVO (2) TOLERANCIA -----------------------------------------------------
# Qué tan cerca del óptimo cuenta como haberlo encontrado. No hay un valor
# de principios; solo hay un valor declarado en voz alta, para que una tasa de éxito citada
# más tarde pueda leerse como "dentro de 0.01 del óptimo" en lugar de como un hecho sobre
# el algoritmo. Cámbialo y cada tasa en esta lección cambia con él.
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
# ------------------------------------------------------------------------------


# --- NUEVO (3) proporcion_dentro_de_tolerancia() ------------------------------
def proporcion_dentro_de_tolerancia(problema: Problema, optimo: float,
                                    puntos_por_eje: int) -> float:
    """¿Qué fracción de la caja puntúa dentro de TOLERANCIA del óptimo?

    La Lección 05 encontró que la escalera de la Lección 02 declaraba un óptimo que ocupaba un
    solo punto, de modo que ninguna corrida podía alcanzarlo y no se podía culpar a ningún operador.
    Esta es esa comprobación, hecha rutina: antes de creer en una tasa de éxito, mide
    cuánto del espacio de búsqueda cuenta como éxito. Una proporción de cero significa que el
    veredicto no está midiendo la búsqueda.

    Args:
        problema, optimo, puntos_por_eje.

    Returns:
        float.

    Example:
        El óptimo 2-D es real (+0.500000) pero ocupa 0.003197% de la cuadrícula, así
    """
    eje = np.linspace(problema.bajo, problema.alto, puntos_por_eje)
    if problema.genes == 1:
        valores = np.array([problema.aptitud([float(x)]) for x in eje])
    else:
        x, y = np.meshgrid(*[eje] * problema.genes, indexing="ij")
        valores = np.sin(x) * np.cos(x) - np.power(
            np.abs((x + 50.0) * (y - 10.0)) / 10.0, 0.1)
    return float(np.mean(valores >= optimo - TOLERANCIA))


def en_la_linea_singular(puntos_por_eje: int, optimo: float
                         ) -> tuple[int, int]:
    """De los puntos de la cuadrícula 2-D que cuentan como éxito, ¿cuántos se asientan en y = 10?

    La respuesta decide si la proporción anterior mide una región o un artefacto
    de dónde casualmente caen las líneas de la cuadrícula.

    Args:
        puntos_por_eje, optimo.

    Returns:
        tuple[int, int].

    Example:
        El óptimo 2-D es real (+0.500000) pero ocupa 0.003197% de la cuadrícula, así
    """
    eje = np.linspace(DOS_D.bajo, DOS_D.alto, puntos_por_eje)
    x, y = np.meshgrid(eje, eje, indexing="ij")
    valores = np.sin(x) * np.cos(x) - np.power(
        np.abs((x + 50.0) * (y - 10.0)) / 10.0, 0.1)
    ganadores = valores >= optimo - TOLERANCIA
    return int(ganadores.sum()), int((ganadores & (y == 10.0)).sum())
# ------------------------------------------------------------------------------


# ===== Parte 1: el instrumento, y lo que cuestan dos genes ====================
print("El instrumento de la Lección 02, aplicado a un paisaje de dos genes.\n")
print("  puntos por eje  | puntos de cuadrícula | óptimo encontrado")
print("  ----------------+----------------------+------------------")
for resolucion in (201, 801, 2001):
    print(f"  {resolucion:>15} | {resolucion ** DOS_D.genes:>20,} | "
          f"{optimo_fuerza_bruta(DOS_D, resolucion):+.6f}")
optimo_2d = optimo_fuerza_bruta(DOS_D, 2001)
print(f"\nLa cuadrícula dice {optimo_2d:+.6f}, y tiene razón. En la línea y = 10")
print("el término de penalización es |0| ** 0.1 = 0 exactamente, por lo que el paisaje allí es")
print("0.5 * sin(2x), cuyo máximo es +0.500000. El óptimo es real.")
cuadricula_leccion_02 = 20001
print(f"\nEl costo de encontrarlo de esa manera: la Lección 02 usó una cuadrícula de "
      f"{cuadricula_leccion_02:,} puntos")
print(f"para un gen. La misma resolución cuesta "
      f"{cuadricula_leccion_02 ** 2:,} puntos para dos genes,")
print(f"y alrededor de 10 ** {10 * np.log10(cuadricula_leccion_02):.0f} para diez. La fuerza "
      "bruta no es un método; es un lujo")
print("que los problemas didácticos unidimensionales resultan poder permitirse.")

# ===== Parte 2: el veredicto que no se puede dar ==============================
proporcion_2d = proporcion_dentro_de_tolerancia(DOS_D, optimo_2d, 2001)
print(f"\nEl costo no es el problema real. De los {2001 ** 2:,} puntos de la cuadrícula, "
      f"{proporcion_2d:.6%} puntúan")
print(f"dentro de {TOLERANCIA} del óptimo. Resuélvelo a mano y verás por qué: "
      "f >= 0.49 necesita")
umbral = TOLERANCIA ** 10
print(f"la penalización por debajo de {TOLERANCIA}, y una décima potencia por debajo de {TOLERANCIA} "
      f"necesita")
print(f"|(x + 50)(y - 10)| / 10 < {umbral:.0e}. En x = +22.5, eso es una "
      f"banda en y de")
print(f"semi-ancho {10 * umbral / abs(22.5 + 50.0):.1e}, que ninguna cuadrícula y "
      "ninguna búsqueda resuelve.")
ganadores, en_linea = en_la_linea_singular(2001, optimo_2d)
print(f"\nEntonces, ¿dónde están esos {ganadores} puntos ganadores de la cuadrícula? "
      f"{en_linea} de {ganadores} de ellos tienen y = 10.0")
print("exactamente. La cuadrícula ve el óptimo solo porque una de sus propias líneas")
print("corre a lo largo de él.")
print("\nEse es el hallazgo de la Lección 05 nuevamente en dos dimensiones: un óptimo sin")
print("volumen. La Lección 05 pudo reparar su escalera con una constante. Aquí la")
print("forma de la penalización ES el problema, por lo que el veredicto es lo que tiene que irse.")
print(f"\nLas {CORRIDAS} corridas del paso 1, juzgadas contra {optimo_2d:+.4f}:")
respuestas_2d = [mejor_historico(corrida(DOS_D, semilla)) for semilla in range(CORRIDAS)]
exitos_2d = sum(veredicto(a, optimo_2d) for a in respuestas_2d)
print(f"    {exitos_2d} de {CORRIDAS} tuvieron éxito. Mejor corrida "
      f"{max(a.aptitud for a in respuestas_2d):+.4f}, peor "
      f"{min(a.aptitud for a in respuestas_2d):+.4f}.")
print("Un veredicto que devuelve la misma respuesta para la mejor corrida y la peor")
print("no está midiendo las corridas.")

# ===== Parte 3: un paisaje donde la palabra significa algo ====================
optimo_1d = optimo_fuerza_bruta(SENO, cuadricula_leccion_02)
proporcion_1d = proporcion_dentro_de_tolerancia(SENO, optimo_1d, cuadricula_leccion_02)
ancho_1d = proporcion_1d * (SENO.alto - SENO.bajo)
print(f"\nEl paisaje de la Lección 01, {SENO.nombre}, en la misma cuadrícula que usó la Lección 02 "
      f"({cuadricula_leccion_02:,} puntos):")
print(f"    óptimo {optimo_1d:+.6f}")
print(f"    proporción de la caja dentro de {TOLERANCIA}: {proporcion_1d:.4%}, "
      f"un tramo de x {ancho_1d:.3f} de ancho")
print(f"    un gen extraído uniformemente aterriza en él una vez de cada "
      f"{1 / proporcion_1d:.0f} extracciones")
print("\nEse es un objetivo. Tiene ancho, una extracción a ciegas lo alcanza a una tasa que puedes")
print("escribir, y una corrida que lo alcanza hizo algo que la extracción no hizo.")
print(f"\nLas mismas {CORRIDAS} semillas, en este paisaje, juzgadas de la misma manera:\n")
print("  semilla | mejor histórico |   déficit  | veredicto")
print("  --------+-----------------+------------+----------")
respuestas_1d = [mejor_historico(corrida(SENO, semilla)) for semilla in range(CORRIDAS)]
for semilla, campeon in enumerate(respuestas_1d):
    acierto = veredicto(campeon, optimo_1d)
    print(f"  {semilla:>7} | {campeon.aptitud:+15.4f} | "
          f"{optimo_1d - campeon.aptitud:+10.4f} | "
          f"{'LO ENCONTRO' if acierto else 'fallo'}")
exitos_1d = sum(veredicto(a, optimo_1d) for a in respuestas_1d)
tasa = exitos_1d / CORRIDAS
bamboleo = (CORRIDAS * tasa * (1 - tasa)) ** 0.5
print(f"\n{exitos_1d} de {CORRIDAS} corridas encontraron el óptimo, una tasa de "
      f"{tasa:.1%}. Ese es un número que vale")
print(f"la pena tener, y vale muy poco: si {tasa:.1%} fuera la tasa verdadera, "
      f"el conteo en")
print(f"{CORRIDAS} corridas aún se movería por {bamboleo:.1f} corridas de un experimento "
      "al siguiente, una")
print("desviación estándar. El paso 3 lo ejecuta correctamente.")

# La imagen: el objetivo 1-D tiene ancho; el 2-D es el borde de un acantilado.
fig, (ax_seno, ax_cuspide) = plt.subplots(1, 2, figsize=(11.5, 4.0))
fina = np.linspace(SENO.bajo, SENO.alto, 4001)
valores = np.array([paisaje_seno([float(x)]) for x in fina])
ax_seno.plot(fina, valores, color="tab:blue", linewidth=1.4)
ax_seno.axhline(optimo_1d - TOLERANCIA, color="tab:green", linestyle="--",
                linewidth=1.0, label=f"óptimo - {TOLERANCIA}")
ax_seno.fill_between(fina, optimo_1d - TOLERANCIA, valores,
                     where=valores >= optimo_1d - TOLERANCIA,
                     color="tab:green", alpha=0.6,
                     label=f"objetivo, {ancho_1d:.3f} de ancho")
for campeon in respuestas_1d:
    ax_seno.plot([campeon.genes[0]], [campeon.aptitud], "o",
                 color="tab:red", markersize=5)
ax_seno.set_title(f"seno 1-D: el objetivo tiene ancho ({proporcion_1d:.2%} de la caja)",
                  fontsize=10)
ax_seno.set_xlabel("x")
ax_seno.set_ylabel("aptitud")
ax_seno.grid(True, linestyle=":", alpha=0.5)
ax_seno.legend(fontsize=8, loc="lower center")

ys = np.linspace(9.0, 11.0, 4001)
corte_x = 22.5
cuspide = np.sin(corte_x) * np.cos(corte_x) - np.power(
    np.abs((corte_x + 50.0) * (ys - 10.0)) / 10.0, 0.1)
ax_cuspide.plot(ys, cuspide, color="tab:red", linewidth=1.4)
ax_cuspide.axhline(optimo_2d - TOLERANCIA, color="tab:green", linestyle="--",
                linewidth=1.0, label=f"óptimo - {TOLERANCIA}")
ax_cuspide.set_title(f"libro 2-D, corte en x = {corte_x}: el objetivo es una línea",
                  fontsize=10)
ax_cuspide.set_xlabel("y")
ax_cuspide.set_ylabel("aptitud")
ax_cuspide.grid(True, linestyle=":", alpha=0.5)
ax_cuspide.legend(fontsize=8, loc="lower center")

fig.suptitle("Antes de contar éxitos, comprueba que el éxito es alcanzable")
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "tasa_exito_02_el_veredicto.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/tasa_exito_02_el_veredicto.png")

