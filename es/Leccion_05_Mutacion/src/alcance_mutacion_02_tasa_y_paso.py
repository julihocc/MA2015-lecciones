"""
Lección 05 - Paso 2: La mutación tiene dos diales, no uno
====================================================
NUEVO EN ESTE PASO: un individuo con valor vectorial, el operador de desviación
aleatoria del libro, y el instrumento reajustado para medirlo.

El paso 1 midió un solo gen, por lo que sigma era lo único que había para ajustar.
Un cromosoma real tiene muchos genes, y el operador tiene entonces un segundo dial: la
probabilidad por gen p de que un gen sea tocado en absoluto. El
`mutation_random_deviation(ind, mu, sigma, p)` de Gridin lleva ambos.

Los dos diales no son intercambiables. Su producto establece cuánto movimiento total
compra una mutación; p por sí solo decide si ese movimiento llega como
un salto largo o como diez saltos cortos, y esas son búsquedas diferentes.

CAMBIOS RESPECTO A alcance_mutacion_01_el_instrumento.py
Introdúcelos en este orden:
    1. CANTIDAD_GENES                 una tasa no tiene en qué actuar hasta que hay más de un gen
    2. Individuo.aptitud              CAMBIADO: califica un vector sumando el objetivo sobre sus genes
    3. mutar_desviacion_aleatoria()   el operador del libro - mu, sigma, y la probabilidad por gen p
    4. reporte_regimen()              el instrumento reajustado para un vector: cuántos genes se movieron, y qué tan lejos

Ejecútalo:  python alcance_mutacion_02_tasa_y_paso.py

Los genes tocados nunca difieren de p·n por más de 0.015. Tres regímenes
con el mismo p·sigma = 0.30 tienen un salto más largo que difiere por 7.1.
"""
import copy
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 52
GEN_MIN, GEN_MAX = -10.0, 10.0
MUTACION_MU = 0.0
INTENTOS = 2000
# --- NUEVO (1) CANTIDAD_GENES -------------------------------------------------
# Diez genes en lugar de uno. Nada sobre el operador necesita esto - es la
# MEDICIÓN lo que lo necesita: con un solo gen, p solo decide con qué frecuencia ocurre la
# mutación en absoluto, y su efecto es indistinguible de bajar la
# probabilidad de mutación de todo el individuo.
CANTIDAD_GENES = 10
# ------------------------------------------------------------------------------
# Cada fila es un régimen de mutación: (p, sigma). Las tres primeras comparten el mismo
# producto p * sigma y son el punto del paso; las dos últimas lo elevan.
REGIMENES = ((0.1, 3.0), (0.3, 1.0), (1.0, 0.3), (0.3, 3.0), (1.0, 1.0))
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
        Los genes tocados nunca difieren de p·n por más de 0.015. Tres regímenes
    """

    def __init__(self, lista_genes: list[float]) -> None:
        self.lista_genes = lista_genes
        self.aptitud = sum(float(objetivo(g)) for g in lista_genes)   # --- CAMBIADO --- un vector se califica sumando sobre sus genes

    @property
    def gen(self) -> float:
        return self.lista_genes[0]

    def __repr__(self) -> str:
        return ("[" + ", ".join(f"{g:+.2f}" for g in self.lista_genes)
                + f"] f={self.aptitud:+.3f}")


def mutar_gaussiana(gen: float, mu: float, sigma: float) -> float:
    """Agrega ruido extraído de N(mu, sigma) a un gen - el operador de la Lección 01.

    Args:
        gen, mu, sigma.

    Returns:
        float.

    Example:
        Desde x = -4.6, sigma = 1.0 no cubre x = +1.38; sigma = 3.0 sí.
    """
    return acotar(gen + random.gauss(mu, sigma))


# --- NUEVO (3) mutar_desviacion_aleatoria() -----------------------------------
def mutar_desviacion_aleatoria(lista_genes: list[float], mu: float, sigma: float,
                            p: float) -> list[float]:
    """El operador de Gridin: recorre el cromosoma y perturba cada gen con prob p.

    La moneda se lanza una vez POR GEN, no una vez por individuo. Esa es la
    diferencia entre "este individuo muta" y "este gen muta", y
    es por eso que un individuo puede regresar de la mutación completamente sin cambios
    a pesar de que la mutación se aplicó en él.

    Args:
        lista_genes, mu, sigma, p.

    Returns:
        list[float].

    Example:
        En una permutación produce 1935 inválidos de 2000.
    """
    mutante = copy.deepcopy(lista_genes)
    for i in range(len(mutante)):
        if random.random() < p:
            mutante[i] = mutar_gaussiana(mutante[i], mu, sigma)
    return mutante
# ------------------------------------------------------------------------------


# --- NUEVO (4) reporte_regimen() ----------------------------------------------
def reporte_regimen(lista_genes: list[float], p: float, sigma: float,
                  intentos: int = INTENTOS) -> dict:
    """Dispara `intentos` mutaciones a un individuo y describe lo que se movió.

    Tres cantidades, y responden a tres preguntas diferentes:
      tocados        cuántos genes seleccionó la moneda        -> el dial de tasa
      paso           qué tan lejos se movió realmente un gen tocado -> el dial de tamaño
      viaje          la suma de |dx| a lo largo del cromosoma       -> el presupuesto completo

    Volver a inicializar la semilla antes de cada régimen mantiene las filas comparables, exactamente como en el
    paso 1.

    Args:
        lista_genes, p, sigma, intentos.

    Returns:
        dict.

    Example:
        Los genes tocados nunca difieren de p·n por más de 0.015. Tres regímenes
    """
    random.seed(SEMILLA)
    tocados, viaje, pasos, acotados, intactos = [], [], [], 0, 0
    for _ in range(intentos):
        mutante = mutar_desviacion_aleatoria(lista_genes, MUTACION_MU, sigma, p)
        movimientos = [abs(a - b) for a, b in zip(lista_genes, mutante)]
        movidos = [d for d in movimientos if d > 0.0]
        acotados += sum(1 for g in mutante if g in (GEN_MIN, GEN_MAX))
        tocados.append(len(movidos))
        viaje.append(sum(movimientos))
        pasos.extend(movidos)
        if not movidos:
            intactos += 1
    return {"p": p, "sigma": sigma, "presupuesto": p * sigma,
            "media_tocados": statistics.fmean(tocados),
            "esperados_tocados": p * len(lista_genes),
            "media_paso": statistics.fmean(pasos) if pasos else 0.0,
            "max_paso": max(pasos) if pasos else 0.0,
            "media_viaje": statistics.fmean(viaje),
            "pasos": pasos,
            "acotados": acotados,
            "proporcion_acotados": acotados / max(1, sum(tocados)),
            "intactos": intactos,
            "intentos": intentos}
# ------------------------------------------------------------------------------


random.seed(SEMILLA)
sujeto = Individuo([random.uniform(GEN_MIN, GEN_MAX)
                      for _ in range(CANTIDAD_GENES)])
print(f"El individuo siendo mutado ({CANTIDAD_GENES} genes):")
print("   ", sujeto)

reportes = [reporte_regimen(sujeto.lista_genes, p, sigma) for p, sigma in REGIMENES]

print(f"\nCada régimen aplicado {INTENTOS} veces a ese único individuo:\n")
print("    p | sigma | p*sigma | genes tocados (p*n) | media paso | max paso |"
      " viaje total | sin cambios")
print("------+-------+---------+---------------------+------------+----------+"
      "-------------+------------")
for r in reportes:
    print(f" {r['p']:4.1f} | {r['sigma']:5.1f} | {r['presupuesto']:7.2f} |"
          f"     {r['media_tocados']:6.3f} ({r['esperados_tocados']:4.1f}) |"
          f" {r['media_paso']:10.3f} | {r['max_paso']:8.3f} |"
          f" {r['media_viaje']:11.3f} | {r['intactos']:11d}")

# Afirmación 1: el dial de tasa hace exactamente lo que dice.
peor_brecha = max(abs(r["media_tocados"] - r["esperados_tocados"]) for r in reportes)
print(f"\nLa columna de genes tocados nunca difiere de p * {CANTIDAD_GENES} por más "
      f"de {peor_brecha:.3f}.")
print("p no es una fuerza, es un conteo de cabezas: dice cuántos de los diez")
print("genes participan, y no dice nada en absoluto sobre lo que les sucede.")

# Afirmación 2: el mismo presupuesto, gastado de tres maneras diferentes.
presupuestos = {}
for r in reportes:
    presupuestos.setdefault(round(r["presupuesto"], 6), []).append(r)
trio = max(presupuestos.values(), key=len)
trio = sorted(trio, key=lambda r: r["p"])
print(f"\nLas primeras {len(trio)} filas comparten el mismo producto "
      f"p * sigma = {trio[0]['presupuesto']:.2f}. Ese producto es el presupuesto:")
for r in trio:
    print(f"    p = {r['p']:<4} sigma = {r['sigma']:<4} -> viaje total "
          f"{r['media_viaje']:.3f}, salto individual más largo {r['max_paso']:.3f}")
dispersion = max(r["max_paso"] for r in trio) / min(r["max_paso"] for r in trio)
viajes = [r["media_viaje"] for r in trio]
print(f"El viaje total se mantiene entre {min(viajes):.3f} y {max(viajes):.3f} - "
      f"un rango de {100 * (max(viajes) / min(viajes) - 1):.0f}%.")
print(f"El salto individual más largo no lo hace: difiere por un factor de "
      f"{dispersion:.1f}.")
print("Aproximadamente el mismo movimiento, y solo uno de estos regímenes puede llevar un")
print("solo gen a través de un valle. Eso es lo que p decide.")

# Afirmación 3: por qué los presupuestos no son exactamente iguales.
concentrado = min(trio, key=lambda r: r["p"])
difuso = max(trio, key=lambda r: r["p"])
deficit = difuso["media_viaje"] - concentrado["media_viaje"]
print(f"\nNo son exactamente iguales, y la brecha es instructiva: "
      f"{deficit:.3f} de viaje")
print(f"falta en p = {concentrado['p']}. Los saltos largos son los que "
      f"caen por el borde de")
print(f"[{GEN_MIN:+.0f}, {GEN_MAX:+.0f}] y se acotan: "
      f"el {100 * concentrado['proporcion_acotados']:.1f}% de los genes que ese régimen "
      f"tocó")
print(f"terminaron clavados a un muro, contra "
      f"el {100 * difuso['proporcion_acotados']:.1f}% en p = {difuso['p']}, cuyos "
      f"pasos rara vez")
print("llegan tan lejos. Un espacio acotado grava silenciosamente el alcance, y grava al")
print("régimen concentrado más duramente.")

# Afirmación 4: la otra cosa que hace una moneda por gen.
bajo = min(reportes, key=lambda r: r["p"])
print(f"\nY la última columna: en p = {bajo['p']}, {bajo['intactos']} de "
      f"{bajo['intentos']} mutaciones no cambiaron nada en absoluto")
print(f"({100 * bajo['intactos'] / bajo['intentos']:.1f}%), porque la moneda cayó en "
      f"cruz {CANTIDAD_GENES} veces seguidas.")
print(f"Su tasa predicha es (1 - p)^{CANTIDAD_GENES} = "
      f"{(1 - bajo['p']) ** CANTIDAD_GENES:.4f}. Una baja tasa de mutación no hace que la")
print("mutación sea suave; la hace rara, lo cual es una cosa diferente.")

# La imagen: el mismo presupuesto de viaje, tres formas.
fig, ejes = plt.subplots(1, len(trio), figsize=(4.0 * len(trio), 3.4),
                         sharey=True)
for eje, r in zip(ejes, trio):
    eje.hist(r["pasos"], bins=60, range=(0, 12), color="tab:orange", alpha=0.85)
    eje.set_yscale("log")
    eje.axvline(r["media_paso"], color="tab:blue", linestyle="--", linewidth=1.2,
               label=f"media paso {r['media_paso']:.2f}")
    eje.set_title(f"p = {r['p']}, sigma = {r['sigma']}\n"
                 f"viaje {r['media_viaje']:.2f}, más largo {r['max_paso']:.2f}",
                 fontsize=9)
    eje.set_xlabel("|dx| de un gen tocado")
    eje.grid(True, linestyle=":", alpha=0.5)
    eje.legend(fontsize=8)
ejes[0].set_ylabel("conteo de genes tocados (log)")
fig.suptitle(f"Un presupuesto de viaje (p * sigma = {trio[0]['presupuesto']:.2f}), "
             "tres formas de gastarlo")
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "alcance_mutacion_02_tasa_y_paso.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/alcance_mutacion_02_tasa_y_paso.png")

