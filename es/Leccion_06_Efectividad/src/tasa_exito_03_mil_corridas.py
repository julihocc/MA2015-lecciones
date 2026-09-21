"""
Lección 06 - Paso 3: Mil corridas
=================================
NUEVO EN ESTE PASO: tasa_exito(), error_estandar(), intervalo_wilson(), y
CORRIDAS = 1000.

El paso 2 terminó con una tasa de éxito medida en ocho corridas - 62.5% - y una
advertencia sobre lo que vale tal tasa: si 62.5% fuera la tasa verdadera, el
conteo en 8 corridas aún divagaría por 1.4 corridas de un experimento al
siguiente. Este paso paga esa advertencia. Ejecuta el experimento mil
veces, de la manera en que lo hace el Capítulo 6 de Gridin, y reemplaza la anécdota con una
medición: una tasa, su error estándar, un intervalo de Wilson al 95%, y la distribución completa de
respuestas detrás de ella.

Dos cosas surgen de las mil corridas que ninguna muestra pequeña podría mostrar.
La estimación de 8 corridas difiere de la estimación de 1000 por casi exactamente
la desviación estándar estimada que el paso 2 anticipó: ese EE era el
instrumento correcto todo el tiempo. Y la distribución de la mejor aptitud histórica no
es una campana alrededor del óptimo: se dispara en el pico global, y se dispara
nuevamente en el óptimo local en el que tropezó el paso 7 de la Lección 01. Ese fracaso
nunca fue una semilla desafortunada. Es un porcentaje fijo de todo lo que este
algoritmo hace, y este paso lo mide.

CAMBIOS RESPECTO A tasa_exito_02_el_veredicto.py
Introdúcelos en este orden:
    1. tasa_exito()        el instrumento Monte Carlo: N corridas, cuenta los veredictos
    2. error_estandar()    lo que compra N: sqrt(p(1-p)/N), el bamboleo de una tasa medida
    3. intervalo_wilson()  intervalo binomial al 95% que funciona en 0% y 100%
    4. CORRIDAS = 1000     cambiado de 8: el millar del libro, a unos pocos segundos

ELIMINADO DE tasa_exito_02_el_veredicto.py: el espécimen 2-D (libro_2d, DOS_D,
en_la_linea_singular, y las ramas de dos genes de los instrumentos de cuadrícula). Su
lección está enseñada; todo de aquí en adelante se mide en SENO.

Ejecútalo:  python tasa_exito_03_mil_corridas.py

La tasa observada es 79.4% con un error estándar de 1.3%. La estimación de 8 corridas difiere por
16.9 puntos. 71 corridas (7.1%) mueren en el pico local de x = -4.51.
"""
import random
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
# La corrida i usa la semilla i, para toda la lección - así que las primeras ocho de estas
# mil corridas SON el experimento del paso 2, semilla por semilla.
CORRIDAS = 1000   # --- CAMBIADO --- (4) de 8: el millar del libro, a unos pocos segundos
FIGURAS = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class Problema:
    """Un paisaje, su caja y el presupuesto de generaciones con el que se ejecuta.

    Args:
        argumentos del constructor.

    Example:
        La tasa observada es 79.4% y un error estándar es 1.3%. La estimación de 8 corridas falló por
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


# Diez generaciones: suficiente para tener éxito la mayor parte del tiempo y fallar con la
# frecuencia suficiente para que valga la pena contar las fallas.
SENO = Problema("seno 1-D", paisaje_seno, -10.0, 10.0, 1, 10)


class Individuo:
    """Una solución candidata, que lleva el problema que la juzga.

    Args:
        argumentos del constructor.

    Example:
        La tasa observada es 79.4% y un error estándar es 1.3%. La estimación de 8 corridas falló por
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
        La tasa observada es 79.4% y un error estándar es 1.3%. La estimación de 8 corridas falló por
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
        La tasa observada es 79.4% y un error estándar es 1.3%. La estimación de 8 corridas falló por
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
        La tasa observada es 79.4% y un error estándar es 1.3%. La estimación de 8 corridas falló por
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
        La tasa observada es 79.4% y un error estándar es 1.3%. La estimación de 8 corridas falló por
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
        La tasa observada es 79.4% y un error estándar es 1.3%. La estimación de 8 corridas falló por
    """
    return max((ind for pop in generaciones for ind in pop),
               key=lambda i: i.aptitud)


def optimo_fuerza_bruta(problema: Problema, puntos_por_eje: int) -> float:
    """El mejor valor en una cuadrícula regular sobre toda la caja.

    El instrumento del paso 5 de la Lección 02, de vuelta en la dimensión uno donde es
    asequible: la verdad sobre el paisaje, desde fuera de cualquier corrida.

    Args:
        problema, puntos_por_eje.

    Returns:
        float.

    Example:
        Seno 1-D: +0.705908. Libro 2-D: +0.500000 sin volumen.
    """
    eje = np.linspace(problema.bajo, problema.alto, puntos_por_eje)
    return float(max(problema.aptitud([float(x)]) for x in eje))


# Qué tan cerca del óptimo cuenta como haberlo encontrado. Declarado en voz alta, para que una
# tasa de éxito pueda leerse como "dentro de 0.01 del óptimo" en lugar de como un
# hecho sobre el algoritmo.
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
    """¿Qué fracción de la caja puntúa dentro de TOLERANCIA del óptimo?

    También la probabilidad de acierto de una extracción a ciegas - el punto de referencia
    en el que el paso 4 gasta el mismo presupuesto.

    Args:
        problema, optimo, puntos_por_eje.

    Returns:
        float.

    Example:
        La tasa observada es 79.4% y un error estándar es 1.3%. La estimación de 8 corridas falló por
    """
    eje = np.linspace(problema.bajo, problema.alto, puntos_por_eje)
    valores = np.array([problema.aptitud([float(x)]) for x in eje])
    return float(np.mean(valores >= optimo - TOLERANCIA))


# --- NUEVO (1) tasa_exito() ---------------------------------------------------
def tasa_exito(problema: Problema, corridas: int, optimo: float
               ) -> tuple[list[Individuo], int]:
    """El instrumento Monte Carlo: ejecuta el algoritmo `corridas` veces, juzga
    la respuesta de cada corrida desde fuera, y cuenta los éxitos.

    Devuelve las respuestas mismas, no solo el conteo, porque el conteo
    nunca es toda la historia - la distribución de los fracasos es donde el
    carácter del algoritmo se muestra.

    Args:
        problema, corridas, optimo.

    Returns:
        tuple[list[Individuo], int].

    Example:
        1000 corridas: 79.4% ± 1.3%. 71 (7.1%) mueren en x = -4.51.
    """
    respuestas = [mejor_historico(corrida(problema, semilla)) for semilla in range(corridas)]
    exitos = sum(veredicto(a, optimo) for a in respuestas)
    return respuestas, exitos
# ------------------------------------------------------------------------------


# --- NUEVO (2) error_estandar() -----------------------------------------------
def error_estandar(tasa: float, n: int) -> float:
    """El bamboleo de una tasa medida: sqrt(p(1-p)/n).

    Una tasa de éxito es una moneda que se mide lanzándola. Esto es qué tan lejos
    vaga la proporción medida de la verdadera, una desviación estándar -
    y se encoge con la raíz cuadrada de n, que es por lo que la certeza es
    cara: cuatro veces las corridas compran solo la mitad del bamboleo.

    Args:
        tasa, n.

    Returns:
        float.

    Example:
        8 corridas ±17.1 puntos; 1000 corridas ±1.3 puntos.
    """
    return (tasa * (1 - tasa) / n) ** 0.5
# ------------------------------------------------------------------------------


# --- NUEVO (3) intervalo_wilson() --------------------------------------------
def intervalo_wilson(exitos: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo binomial bilateral que no degenera en 0% ni en 100%."""
    tasa = exitos / n
    denominador = 1 + z ** 2 / n
    centro = (tasa + z ** 2 / (2 * n)) / denominador
    radio = z / denominador * (
        tasa * (1 - tasa) / n + z ** 2 / (4 * n ** 2)
    ) ** 0.5
    return centro - radio, centro + radio
# ------------------------------------------------------------------------------


# ===== Parte 1: la medición, y el destino de la estimación del paso 2 ========
optimo = optimo_fuerza_bruta(SENO, 20001)
print(f"El objetivo (cuadrícula de {20001:,} puntos): óptimo {optimo:+.6f}, "
      f"éxito significa dentro de {TOLERANCIA} de él.\n")
print(f"{CORRIDAS:,} corridas, población {TAMANO_POBLACION}, "
      f"{SENO.generaciones} generaciones cada una...\n")
respuestas, exitos = tasa_exito(SENO, CORRIDAS, optimo)
tasa = exitos / CORRIDAS
bajo, alto = intervalo_wilson(exitos, CORRIDAS)
print(f"    {exitos} de {CORRIDAS:,} corridas cumplieron la tolerancia de la referencia de cuadrícula.")
print(f"    tasa de éxito {tasa:.1%} +/- {error_estandar(tasa, CORRIDAS):.1%} "
      f"(un error estándar)")
print(f"    intervalo de Wilson al 95% [{bajo:.1%}, {alto:.1%}]")

# Las primeras ocho de estas corridas son el experimento del paso 2, semilla por semilla.
corridas_paso2 = 8
exitos_paso2 = sum(veredicto(a, optimo) for a in respuestas[:corridas_paso2])
tasa_paso2 = exitos_paso2 / corridas_paso2
bamboleo_paso2 = error_estandar(tasa_paso2, corridas_paso2)
fallo = tasa - tasa_paso2
print(f"\nEl paso 2 ejecutó las primeras {corridas_paso2} de estas mismas semillas y reportó "
      f"{tasa_paso2:.1%} +/- {bamboleo_paso2:.1%}.")
print(f"La estimación de 1000 corridas es {tasa:.1%}. La de 8 corridas difiere por "
      f"{fallo:.1%} -")
print(f"eso es {fallo / bamboleo_paso2:.1f} de sus propias desviaciones estándar. El "
      "bamboleo que calculó el paso 2")
print("informó sobre dispersión muestral; no era un intervalo de confianza.")

# ===== Parte 2: la distribución no es una campana =============================
# Encuentra los picos del paisaje en una cuadrícula fina, para que las afirmaciones a continuación
# sean medidas, no recordadas de la Lección 01.
fina = np.linspace(SENO.bajo, SENO.alto, 200001)
valores_finos = np.array([paisaje_seno([float(x)]) for x in fina])
es_pico = (valores_finos[1:-1] > valores_finos[:-2]) \
    & (valores_finos[1:-1] > valores_finos[2:]) & (valores_finos[1:-1] > -0.5)
pico_x, pico_f = fina[1:-1][es_pico], valores_finos[1:-1][es_pico]
no_global = np.abs(pico_f - optimo) > TOLERANCIA
local_mas_alto = int(np.argmax(pico_f[no_global]))
local_x, local_f = float(pico_x[no_global][local_mas_alto]), \
    float(pico_f[no_global][local_mas_alto])

aptitudes = [a.aptitud for a in respuestas]
en_global = exitos
en_local = sum(1 for f in aptitudes if abs(f - local_f) < TOLERANCIA)
en_medio = CORRIDAS - en_global - en_local
print(f"\nDónde aterrizaron las {CORRIDAS:,} respuestas:")
print(f"    {en_global:>4} corridas ({en_global / CORRIDAS:5.1%})  dentro de {TOLERANCIA} "
      f"de la referencia de cuadrícula densa (f = {optimo:+.4f})")
print(f"    {en_local:>4} corridas ({en_local / CORRIDAS:5.1%})  murieron en el pico "
      f"local en x = {local_x:+.2f} (f = {local_f:+.4f})")
print(f"    {en_medio:>4} corridas ({en_medio / CORRIDAS:5.1%})  en medio - "
      f"casi aciertos en la colina final")
print(f"\nEl paso 7 de la Lección 01 cambió una semilla y vio la corrida estancarse cerca "
      f"de x = {local_x:+.2f}.")
print(f"Esa nunca fue una semilla desafortunada: es {en_local / CORRIDAS:.1%} de "
      f"todo lo que este algoritmo hace.")
print(f"La peor corrida de las mil terminó en f = {min(aptitudes):+.4f}.")

# ===== Parte 3: lo que la tasa no puede decir =================================
proporcion = proporcion_dentro_de_tolerancia(SENO, optimo, 20001)
print(f"\nUn gen extraído uniformemente de la caja aterriza en el objetivo "
      f"{proporcion:.2%} del tiempo.")
print(f"Cada corrida evaluó alrededor de cien individuos (el paso 4 los cuenta "
      f"exactamente)")
print(f"para convertir {proporcion:.2%} en {tasa:.1%}. Si esas evaluaciones fueron "
      f"bien gastadas no es")
print("una pregunta que la tasa pueda responder. El paso 4 cuenta el costo.")

# La imagen: el histograma del libro, anotado con lo que son los picos.
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.hist(aptitudes, bins=16, facecolor="tab:blue", alpha=0.75, edgecolor="black")
ax.axvline(optimo - TOLERANCIA, color="tab:green", linestyle="--",
           linewidth=1.2, label=f"óptimo - {TOLERANCIA}")
ax.annotate(f"{en_global} de {CORRIDAS:,} corridas aterrizan aquí",
            xy=(optimo, en_global), xytext=(0.32, 0.72),
            textcoords="axes fraction", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"))
ax.annotate(f"{en_local} corridas murieron en\nel pico local",
            xy=(local_f, en_local), xytext=(0.13, 0.45),
            textcoords="axes fraction", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"))
ax.set_title(f"Mejor aptitud histórica a lo largo de {CORRIDAS:,} corridas independientes "
             f"(población {TAMANO_POBLACION}, {SENO.generaciones} generaciones)")
ax.set_xlabel("mejor aptitud alcanzada")
ax.set_ylabel("número de corridas")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.legend(fontsize=9)
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "tasa_exito_03_mil_corridas.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/tasa_exito_03_mil_corridas.png")

