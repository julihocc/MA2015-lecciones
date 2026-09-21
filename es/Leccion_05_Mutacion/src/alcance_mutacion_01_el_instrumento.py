"""
Lección 05 - Paso 1: El alcance de una mutación, y cómo medirlo
=====================================================================
La selección descarta y la cruza recombina; ambos trabajan solo con valores que la
población ya posee. La mutación es el único operador que puede producir un valor
que nadie tenía. Por lo tanto, toda esta lección trata sobre una pregunta: ¿QUÉ TAN LEJOS
puede llegar una mutación?

El paso 5 de la Lección 01 hizo esa pregunta una vez, sobre una población atrapada en una colina
local a seis unidades de la colina global, y obtuvo dos respuestas: sigma = 1.0 nunca
llegó, sigma = 3.0 llegó 110 veces de 2000. Este paso convierte ese experimento
único en un instrumento - una función que reporta toda la distribución
de desplazamiento, no solo el conteo de llegadas - y cada paso posterior lo reutiliza.

Los dos números que imprimió la Lección 01 salen de aquí sin cambios. Esa identidad es
la prueba de que nada fue redefinido silenciosamente en el camino.

Ejecútalo:  python alcance_mutacion_01_el_instrumento.py

Los dos números de la Lección 01 reaparecen: sigma 1.0 llega 0 de 2000,
sigma 3.0 llega 110 de 2000. En sigma 6.0 el 19.4% de las propuestas se acotan.
"""
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 52
GEN_MIN, GEN_MAX = -10.0, 10.0
MUTACION_MU = 0.0
INTENTOS = 2000
# La situación en la que nos dejó la Lección 01: una población convergente en la colina local
# cerca de x = -4.6, cuyo pico vale f = +0.074. La colina global está en x = +1.38
# y vale f = +0.706. "Llegar" significa aterrizar dentro del RADIO_COLINA de ella,
# que es el ancho de la cuenca de esa colina.
ATRAPADO_EN = -4.6
COLINA_GLOBAL = 1.38
RADIO_COLINA = 1.5
SIGMAS = (0.5, 1.0, 3.0, 6.0)
FIGURAS = Path(__file__).resolve().parent.parent / "figures"


def objetivo(x: float) -> float:
    """El paisaje de la Lección 01, sin cambios. Funciona en un flotante y en un arreglo.

    Args:
        x.

    Returns:
        float.

    Example:
        El máximo global del seno está cerca de x = +1.372, f = +0.706.
    """
    return np.sin(x) - 0.2 * abs(x)


def acotar(gen: float, bajo: float = GEN_MIN, alto: float = GEN_MAX) -> float:
    """La mutación puede proponer cualquier número real; el espacio de búsqueda tiene bordes.

    Args:
        gen, bajo, alto.

    Returns:
        float.

    Example:
        acotar(11.4) == 10.0 en la caja [0, 10] de esta lección.
    """
    return max(bajo, min(alto, gen))


class Individuo:
    """Una solución candidata: un cromosoma y su puntaje.

    Args:
        argumentos del constructor.

    Example:
        Los dos números de la Lección 01 reaparecen: sigma 1.0 llega 0 de 2000,
    """

    def __init__(self, lista_genes: list[float]) -> None:
        self.lista_genes = lista_genes
        self.aptitud = float(objetivo(lista_genes[0]))

    @property
    def gen(self) -> float:
        return self.lista_genes[0]

    def __repr__(self) -> str:
        return f"x={self.gen:+.3f} f={self.aptitud:+.3f}"


def mutar_gaussiana(gen: float, mu: float, sigma: float) -> float:
    """Agrega ruido extraído de N(mu, sigma) a un gen - el operador de la Lección 01.

    sigma es el tamaño de paso de la búsqueda, y todo lo que esta lección mide
    es una consecuencia de ello.

    Args:
        gen, mu, sigma.

    Returns:
        float.

    Example:
        Desde x = -4.6, sigma = 1.0 no cubre x = +1.38; sigma = 3.0 sí.
    """
    return acotar(gen + random.gauss(mu, sigma))


def reporte_alcance(inicio: float, sigma: float, intentos: int = INTENTOS) -> dict:
    """Dispara `intentos` mutaciones únicas desde `inicio` y describe dónde aterrizan.

    Este es el instrumento. Vuelve a inicializar la semilla antes de cada medición, así que dos filas
    de la tabla abajo difieren solo en sigma y nunca en las extracciones que consumieron -
    lo cual es lo que hace que las filas sean comparables.

    `acotados` cuenta las propuestas que cayeron fuera de [GEN_MIN, GEN_MAX] y
    tuvieron que ser jaladas de vuelta al borde. Se reporta porque un salto truncado
    es un salto que la columna de sigma ya no describe.

    Args:
        inicio, sigma, intentos.

    Returns:
        dict.

    Example:
        sigma 1.0 llega 0 de 2000; sigma 3.0 llega 110 de 2000.
    """
    random.seed(SEMILLA)
    aterrizajes, distancias, acotados = [], [], 0
    for _ in range(intentos):
        propuesta = inicio + random.gauss(MUTACION_MU, sigma)
        aterrizaje = acotar(propuesta)
        if aterrizaje != propuesta:
            acotados += 1
        aterrizajes.append(aterrizaje)
        distancias.append(abs(aterrizaje - inicio))
    llegadas = sum(1 for x in aterrizajes if abs(x - COLINA_GLOBAL) < RADIO_COLINA)
    cuesta_arriba = sum(1 for x in aterrizajes if objetivo(x) > objetivo(inicio))
    return {"sigma": sigma,
            "aterrizajes": aterrizajes,
            "distancia_media": statistics.fmean(distancias),
            "distancia_mediana": statistics.median(distancias),
            "distancia_maxima": max(distancias),
            "acotados": acotados,
            "llegadas": llegadas,
            "cuesta_arriba": cuesta_arriba,
            "intentos": intentos}


print("Dónde está atrapada la población, y a dónde necesita llegar:")
print(f"    atrapada en x = {ATRAPADO_EN:+.2f}, f = {objetivo(ATRAPADO_EN):+.4f}")
print(f"    colina global x = {COLINA_GLOBAL:+.2f}, f = {objetivo(COLINA_GLOBAL):+.4f}")
print(f"    distancia a cruzar: {abs(COLINA_GLOBAL - ATRAPADO_EN):.1f} unidades")

reportes = [reporte_alcance(ATRAPADO_EN, sigma) for sigma in SIGMAS]

print(f"\nUna mutación, {INTENTOS} veces, desde x = {ATRAPADO_EN:+.2f}:\n")
print(" sigma | media |dx| | mediana |dx| | max |dx| | acotados | c. arriba | "
      "llegaron a colina global")
print("-------+------------+--------------+----------+----------+-----------+"
      "-------------------------")
for r in reportes:
    print(f" {r['sigma']:5.1f} |{r['distancia_media']:11.3f} |"
          f"{r['distancia_mediana']:13.3f} |{r['distancia_maxima']:9.3f} |"
          f"{r['acotados']:9d} |{r['cuesta_arriba']:10d} |"
          f" {r['llegadas']:5d} de {r['intentos']}"
          f"  ({100 * r['llegadas'] / r['intentos']:5.2f}%)")

# La comprobación de identidad. Si estas dos filas alguna vez dejan de coincidir con la Lección 01, algo
# en el operador o en la semilla cambió y el resto de la lección es insegura.
por_sigma = {r["sigma"]: r for r in reportes}
print(f"\nEl paso 5 de la Lección 01 reportó {por_sigma[1.0]['llegadas']} llegadas con "
      f"sigma = 1.0 y {por_sigma[3.0]['llegadas']} con sigma = 3.0.")
print("Esas son las dos filas de arriba, producidas por la misma semilla y el mismo")
print("operador: el instrumento mide lo que midió la Lección 01, y más.")

# Para qué son las columnas extra.
limpios = [r for r in reportes if r["acotados"] == 0]
print("\nLee la columna de media |dx| contra sigma. Un paso gaussiano de desviación")
print("estándar s viaja 0.798 * s en promedio, así que en las filas donde ninguna")
print("propuesta chocó contra un muro la proporción debería ser esa constante:")
for r in limpios:
    print(f"    sigma = {r['sigma']:<4} media |dx| = {r['distancia_media']:.3f}"
          f"   proporción = {r['distancia_media'] / r['sigma']:.3f}")
peor = max(reportes, key=lambda r: r["acotados"])
print(f"    sigma = {peor['sigma']:<4} media |dx| = {peor['distancia_media']:.3f}"
      f"   proporción = {peor['distancia_media'] / peor['sigma']:.3f}"
      f"   <- {peor['acotados']} de {peor['intentos']} acotados")
print(f"La última fila se queda corta de 0.798 porque el "
      f"{100 * peor['acotados'] / peor['intentos']:.1f}% de sus propuestas aterrizaron")
print(f"fuera de [{GEN_MIN:+.0f}, {GEN_MAX:+.0f}] y fueron jaladas de vuelta al "
      "borde. Esas mutaciones no")
print("viajaron tan lejos como dice sigma, y parte de las llegadas de esa fila es el")
print("muro en lugar de la búsqueda.")

# El intercambio del que realmente trata la tabla.
mejores_llegadas = max(reportes, key=lambda r: r["llegadas"])
mejor_c_arriba = max(reportes, key=lambda r: r["cuesta_arriba"])
print(f"\nLas llegadas suben con sigma: "
      + ", ".join(f"{r['llegadas']}" for r in reportes)
      + f" para sigma = "
      + ", ".join(f"{r['sigma']}" for r in reportes) + ".")
print(f"La mejor tasa de llegadas en esta tabla es sigma = {mejores_llegadas['sigma']}, "
      f"la más grande probada.")
print("Así que el alcance simplemente se puede comprar - y el precio está en la columna cuesta arriba.")
print(f"La mayor cantidad de movimientos cuesta arriba, {mejor_c_arriba['cuesta_arriba']} de "
      f"{mejor_c_arriba['intentos']}, pertenece a sigma = {mejor_c_arriba['sigma']},")
print(f"que llegó {mejor_c_arriba['llegadas']} veces. Un sigma pequeño es bueno para "
      "mejorar donde")
print("ya está y es inútil para ir a cualquier otra parte; un sigma grande es lo")
print("inverso. Ese es todo el intercambio, y ningún número por sí solo lo resuelve.")

# La imagen: donde aterrizan realmente 2000 mutaciones, para los dos sigmas de
# la Lección 01, dibujados contra el paisaje en el que aterrizan.
fig, ejes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
cuadricula = np.linspace(GEN_MIN, GEN_MAX, 800)
for eje, sigma in zip(ejes, (1.0, 3.0)):
    r = por_sigma[sigma]
    eje.hist(r["aterrizajes"], bins=80, range=(GEN_MIN, GEN_MAX),
            color="tab:orange", alpha=0.75,
            label=f"{r['intentos']} aterrizajes, sigma = {sigma}")
    gemelo = eje.twinx()
    gemelo.plot(cuadricula, objetivo(cuadricula), color="tab:blue", linewidth=1.2)
    gemelo.set_ylabel("f(x)")
    gemelo.axvspan(COLINA_GLOBAL - RADIO_COLINA, COLINA_GLOBAL + RADIO_COLINA,
                 color="tab:green", alpha=0.15)
    eje.axvline(ATRAPADO_EN, color="tab:red", linestyle="--", linewidth=1)
    eje.set_ylabel("aterrizajes")
    eje.set_title(f"sigma = {sigma}: {r['llegadas']} de {r['intentos']} aterrizan en "
                 f"la colina global (banda verde)", fontsize=10)
    eje.legend(loc="upper left", fontsize=8)
ejes[-1].set_xlabel("x")
fig.suptitle("El alcance de una mutación, disparada 2000 veces desde la colina local")
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "alcance_mutacion_01_el_instrumento.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/alcance_mutacion_01_el_instrumento.png")

