"""
Lección 05 - Paso 3: Mutación que se niega a empeorar las cosas
===============================================================
NUEVO EN ESTE PASO: la variante de desviación aleatoria guiada por aptitud, y el
contador que le pone precio.

Cada mutación hasta ahora ha sido ciega: mueve los genes y acepta lo que sea que
diga el objetivo sobre el resultado. La mayor parte del tiempo el resultado es peor -
el paso 1 ya mostró que en sigma = 3.0 solo 144 de 2000 mutaciones fueron
cuesta arriba. El último operador de Gridin las rechaza: reintenta, hasta MAX_INTENTOS
veces, y devuelve el primer mutante que supera a su padre.

Funciona. También cuesta, y el costo es la parte interesante, porque se
paga en la única moneda que un algoritmo genético realmente gasta: llamadas a la
función de aptitud.

CAMBIOS RESPECTO A alcance_mutacion_02_tasa_y_paso.py
Introdúcelos en este orden:
    1. MAX_INTENTOS                   cuántos intentos se le permite a una mutación
    2. mutar_guiada_por_aptitud()     reintenta hasta que el mutante supere a su padre, o se rinde
    3. comparar()                     mide ambos operadores por mutación Y por llamada de aptitud

Ejecútalo:  python alcance_mutacion_03_guiada_por_aptitud.py

El filtro convierte un cambio de aptitud de -1.0337 en +0.0548 y gasta
5603 evaluaciones contra 2000: por mil llamadas llega 48.7 vs 51.0 de la ciega.
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
CANTIDAD_GENES = 10
REGIMENES = ((0.1, 3.0), (0.3, 1.0), (1.0, 0.3), (0.3, 3.0), (1.0, 1.0))
# --- NUEVO (1) MAX_INTENTOS ---------------------------------------------------
# El valor por defecto de Gridin. Es un presupuesto, no una garantía: si ninguno de los tres
# intentos mejora al padre, el padre se devuelve sin cambios y las
# tres evaluaciones de aptitud se pierden de todos modos.
MAX_INTENTOS = 3
# ------------------------------------------------------------------------------
# De vuelta al individuo de un solo gen del paso 1 y su población atrapada, porque
# la pregunta aquí no es cómo se distribuye el movimiento a través de un cromosoma, sino
# si filtrar el movimiento por aptitud ayuda a una búsqueda que necesita escapar.
ATRAPADO_EN = -4.6
COLINA_GLOBAL = 1.38
RADIO_COLINA = 1.5
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
        El filtro convierte un cambio de aptitud de -1.0337 en +0.0548 y gasta
    """

    def __init__(self, lista_genes: list[float]) -> None:
        self.lista_genes = lista_genes
        self.aptitud = sum(float(objetivo(g)) for g in lista_genes)

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


def mutar_desviacion_aleatoria(lista_genes: list[float], mu: float, sigma: float,
                            p: float) -> list[float]:
    """El operador de Gridin: recorre el cromosoma y perturba cada gen con prob p.

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


# --- NUEVO (2) mutar_guiada_por_aptitud() -------------------------------------
def mutar_guiada_por_aptitud(individuo: Individuo, mu: float, sigma: float,
                          p: float, max_intentos: int = MAX_INTENTOS
                          ) -> tuple[Individuo, int]:
    """Reintenta desviación aleatoria hasta que el mutante supere a su padre, o se rinde.

    El segundo valor de retorno es el número de evaluaciones de aptitud que esta
    sola mutación consumió. Devolverlo es la razón completa por la que existe este paso:
    la ventaja del operador es obvia y su precio no lo es, y una figura de mérito
    que ignora el precio siempre lo preferirá.

    Nota lo que la prueba de aceptación puede y no puede ver. Compara al mutante
    solo con su propio padre, por lo que nunca puede aceptar un paso que baja para
    luego subir a algún lugar mejor - y en una región plana del paisaje
    no puede aceptar nada en absoluto, porque nada allí es estrictamente mejor.

    Args:
        individuo, mu, sigma, p, max_intentos.

    Returns:
        tuple[Individuo, int].

    Example:
        Cambia -1.0337 a +0.0548 y gasta 5603 evaluaciones contra 2000.
    """
    evaluaciones = 0
    for _ in range(max_intentos):
        mutante = Individuo(mutar_desviacion_aleatoria(individuo.lista_genes,
                                                    mu, sigma, p))
        evaluaciones += 1
        if mutante.aptitud > individuo.aptitud:
            return mutante, evaluaciones
    return individuo, evaluaciones
# ------------------------------------------------------------------------------


# --- NUEVO (3) comparar() -----------------------------------------------------
def comparar(inicio: float, sigma: float, intentos: int = INTENTOS) -> dict:
    """Ejecuta ambos operadores desde el mismo gen, bajo la misma semilla, y ponles precio.

    Cada cifra por mutación se reporta dos veces: una por mutación, que es cómo
    suele juzgarse a un operador, y una por cada mil evaluaciones de aptitud,
    que es lo que realmente paga una ejecución. p = 1.0 aquí para que "una mutación" signifique
    lo mismo para ambos operadores en un cromosoma de un solo gen.

    Args:
        inicio, sigma, intentos.

    Returns:
        dict.

    Example:
        El filtro convierte un cambio de aptitud de -1.0337 en +0.0548 y gasta
    """
    padre = Individuo([inicio])

    random.seed(SEMILLA)
    deltas_ciega, llegadas_ciega, evaluaciones_ciega = [], 0, 0
    for _ in range(intentos):
        mutante = Individuo(mutar_desviacion_aleatoria([inicio], MUTACION_MU,
                                                    sigma, 1.0))
        evaluaciones_ciega += 1
        deltas_ciega.append(mutante.aptitud - padre.aptitud)
        if abs(mutante.gen - COLINA_GLOBAL) < RADIO_COLINA:
            llegadas_ciega += 1

    random.seed(SEMILLA)
    deltas_guiada, llegadas_guiada, evaluaciones_guiada, aceptados = [], 0, 0, 0
    for _ in range(intentos):
        mutante, evaluaciones = mutar_guiada_por_aptitud(padre, MUTACION_MU,
                                                    sigma, 1.0)
        evaluaciones_guiada += evaluaciones
        deltas_guiada.append(mutante.aptitud - padre.aptitud)
        aceptados += mutante is not padre
        if abs(mutante.gen - COLINA_GLOBAL) < RADIO_COLINA:
            llegadas_guiada += 1

    return {"sigma": sigma, "intentos": intentos,
            "ciega_media_delta": statistics.fmean(deltas_ciega),
            "ciega_cuesta_arriba": sum(1 for d in deltas_ciega if d > 0),
            "ciega_llegadas": llegadas_ciega,
            "ciega_evaluaciones": evaluaciones_ciega,
            "guiada_media_delta": statistics.fmean(deltas_guiada),
            "guiada_aceptados": aceptados,
            "guiada_llegadas": llegadas_guiada,
            "guiada_evaluaciones": evaluaciones_guiada,
            "ciega_deltas": deltas_ciega,
            "guiada_deltas": deltas_guiada}
# ------------------------------------------------------------------------------


print(f"Mutando un solo gen sentado en x = {ATRAPADO_EN:+.2f} "
      f"(f = {objetivo(ATRAPADO_EN):+.4f}),")
print(f"{INTENTOS} veces con cada operador, en MAX_INTENTOS = {MAX_INTENTOS}.\n")

resultados = [comparar(ATRAPADO_EN, sigma) for sigma in (1.0, 3.0)]

FILA = " {:>5} | {:<14} | {:>18} | {:>18} | {:>8} | {:>13}"
encabezado = FILA.format("sigma", "operador", "cambio medio en f",
                    "aceptado o c. arr.", "llegadas", "llamadas aptitud")
print(encabezado)
print("-" * len(encabezado))
for r in resultados:
    print(FILA.format(f"{r['sigma']:.1f}", "ciega",
                     f"{r['ciega_media_delta']:+.4f}",
                     f"{r['ciega_cuesta_arriba']} de {r['intentos']}",
                     r["ciega_llegadas"], r["ciega_evaluaciones"]))
    print(FILA.format("", "guiada por aptitud",
                     f"{r['guiada_media_delta']:+.4f}",
                     f"{r['guiada_aceptados']} de {r['intentos']}",
                     r["guiada_llegadas"], r["guiada_evaluaciones"]))

print("\nLos conteos cuesta arriba difieren un poco de la tabla del paso 1 (140 y 144)")
print("porque mutar_desviacion_aleatoria() lanza una moneda por gen antes de extraer")
print("el ruido, por lo que los dos operadores consumen el flujo aleatorio de manera diferente.")
print("Dentro de este script ambas columnas fueron extraídas de la misma semilla, que es")
print("lo que la comparación necesita.")

for r in resultados:
    sigma = r["sigma"]
    print(f"\n--- sigma = {sigma} " + "-" * (58 - len(str(sigma))))
    print(f"La mutación ciega cambia la aptitud por {r['ciega_media_delta']:+.4f} en "
          f"promedio; el filtro convierte eso")
    print(f"en {r['guiada_media_delta']:+.4f}. Está haciendo exactamente lo que "
          "dice hacer.")
    esperado = 1 - (1 - r["ciega_cuesta_arriba"] / r["intentos"]) ** MAX_INTENTOS
    print(f"Aceptó {r['guiada_aceptados']} de {r['intentos']} mutaciones "
          f"({100 * r['guiada_aceptados'] / r['intentos']:.1f}%), contra "
          f"{r['ciega_cuesta_arriba']} extracciones cuesta arriba")
    print(f"({100 * r['ciega_cuesta_arriba'] / r['intentos']:.1f}%) para la mutación ciega. "
          f"Tres intentos independientes de un evento del {100 * r['ciega_cuesta_arriba'] / r['intentos']:.1f}%")
    observado = r["guiada_aceptados"] / r["intentos"]
    print(f"tienen éxito el {100 * esperado:.1f}% de las veces; la columna de aceptación "
          f"dice {100 * observado:.1f}%,")
    print(f"una brecha de {100 * abs(observado - esperado):.1f} puntos. El filtro "
          "no añade nueva habilidad - solo")
    print("compra más boletos en la misma lotería.")
    tasa_ciega = 1000 * r["ciega_llegadas"] / r["ciega_evaluaciones"]
    tasa_guiada = 1000 * r["guiada_llegadas"] / r["guiada_evaluaciones"]
    print(f"\nLlegadas en la colina global: {r['ciega_llegadas']} ciegas, "
          f"{r['guiada_llegadas']} guiadas por aptitud - el filtro")
    if r["guiada_llegadas"] > r["ciega_llegadas"]:
        veredicto = "alcanza la colina lejana más a menudo"
    elif r["guiada_llegadas"] < r["ciega_llegadas"]:
        veredicto = "alcanza la colina lejana menos a menudo"
    else:
        veredicto = "no cambia con qué frecuencia se alcanza la colina lejana"
    print(f"{veredicto}. Pero gastó "
          f"{r['guiada_evaluaciones']} llamadas de aptitud contra")
    print(f"{r['ciega_evaluaciones']}, así que por cada mil llamadas las tasas son "
          f"{tasa_ciega:.1f} ciega y {tasa_guiada:.1f} guiada.")
    if tasa_ciega == tasa_guiada == 0.0:
        print("Ninguno de los operadores llega jamás allí, así que no hay nada que")
        print("comparar: en este sigma la brecha es simplemente demasiado ancha para un solo salto,")
        print("y ninguna cantidad de filtrado crea un alcance que el tamaño del paso")
        print("no tiene.")
    elif tasa_guiada < tasa_ciega:
        print("Por mutación el filtro gana; por evaluación de aptitud PIERDE.")
    else:
        print("El filtro gana en ambas cuentas aquí.")

print("\nEse es el veredicto sobre este operador, y no es el que su nombre")
print("sugiere. Reintentar hasta mejorar es una excelente manera de hacer ver")
print("bien a una sola mutación y una mala manera de gastar un presupuesto de evaluaciones - y en")
print("las Lecciones 06 y 07, el presupuesto de evaluación es lo que se mide.")
print("\nHay un segundo límite, estructural en lugar de estadístico. La prueba es")
print(f"`mutante.aptitud > padre.aptitud`, sin tolerancia, por lo que en una región")
print("donde el paisaje es plano nada es nunca estrictamente mejor y el")
print("operador devuelve el padre en cada ocasión, habiendo pagado MAX_INTENTOS")
print("evaluaciones por ello. El paso 4 se ejecuta en exactamente ese tipo de paisaje.")

# La imagen: la distribución de cambio de aptitud, ciega contra filtrada.
fig, ejes = plt.subplots(1, len(resultados), figsize=(5.2 * len(resultados), 3.6))
for eje, r in zip(ejes, resultados):
    bins = np.linspace(min(min(r["ciega_deltas"]), min(r["guiada_deltas"])),
                       max(max(r["ciega_deltas"]), max(r["guiada_deltas"])), 60)
    eje.hist(r["ciega_deltas"], bins=bins, color="tab:red", alpha=0.6,
            label=f"ciega (media {r['ciega_media_delta']:+.3f})")
    eje.hist(r["guiada_deltas"], bins=bins, color="tab:green", alpha=0.6,
            label=f"guiada por aptitud (media {r['guiada_media_delta']:+.3f})")
    eje.set_yscale("log")
    eje.axvline(0.0, color="grey", linewidth=0.9, linestyle="--")
    eje.set_title(f"sigma = {r['sigma']}", fontsize=10)
    eje.set_xlabel("cambio en aptitud después de una mutación")
    eje.set_ylabel("conteo (log)")
    eje.grid(True, linestyle=":", alpha=0.5)
    eje.legend(fontsize=8, loc="upper left")
fig.suptitle("El filtro elimina la cola izquierda - y lo paga en llamadas de aptitud")
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "alcance_mutacion_03_guiada_por_aptitud.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/alcance_mutacion_03_guiada_por_aptitud.png")

