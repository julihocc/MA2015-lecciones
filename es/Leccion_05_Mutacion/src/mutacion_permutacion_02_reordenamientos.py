"""
Lección 05 - Paso 2 del segundo ejemplo: tres reordenamientos más grandes
======================================================================
NUEVO EN ESTE PASO: inversión, desplazamiento y barajar, y la comparación que pone
a los cuatro operadores de permutación en una sola escala.

El intercambio es el movimiento legal mínimo. Gridin da tres más grandes, y los tres
funcionan de la misma manera: elige dos posiciones y reordena el bloque entre
ellas. La inversión lo invierte (le da la vuelta), el desplazamiento lo rota por un lugar, barajar
lo aleatoriza. La pregunta que responde este paso es la que el primer ejemplo hizo sobre
sigma - ¿qué tan lejos llega cada uno de ellos? - y la respuesta necesita dos números,
no uno.

Los dos números no concuerdan, y ese es el resultado. El operador que perturba
la mayor cantidad de posiciones mueve los símbolos lo menos posible. El operador que suena más
destructivo no hace nada en absoluto el 13.6% de las veces.

CAMBIOS RESPECTO A mutacion_permutacion_01_ruido_ilegal.py
Introdúcelos en este orden:
    1. mutacion_inversion()      invierte el bloque: los mismos símbolos, leídos hacia atrás
    2. mutacion_desplazamiento() rota el bloque por un lugar: todos se mueven, apenas
    3. mutacion_barajar()        aleatoriza el bloque, lo que a veces significa no cambiarlo
    4. comparar_operadores()     los cuatro operadores legales en una escala, y la comprobación exacta

Ejecútalo:  python mutacion_permutacion_02_reordenamientos.py

Por posiciones: desplazamiento 4.663 > inversión 4.230 > barajar 3.708 >
intercambio 2.000. Barajar devuelve el individuo intacto 271 de 2000 veces.
"""
import copy
import random
import statistics
from math import copysign
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 52
LONGITUD_CROMOSOMA = 10
INTENTOS = 2000
MUTACION_MU, MUTACION_SIGMA = 0.0, 1.0
FIGURAS = Path(__file__).resolve().parent.parent / "figures"


def es_permutacion(lista_genes: list, alfabeto: list) -> bool:
    """Un cromosoma es legal solo si es un reordenamiento del alfabeto.

    Args:
        lista_genes, alfabeto.

    Returns:
        bool.

    Example:
        1935 de 2000 mutantes con ruido gaussiano fallan esta prueba.
    """
    return sorted(lista_genes) == sorted(alfabeto)


def posiciones_cambiadas(original: list, mutante: list) -> int:
    """Cuántas posiciones contienen un símbolo diferente después de la mutación.

    Args:
        original, mutante.

    Returns:
        int.

    Example:
        Por posiciones: desplazamiento 4.663 > inversión 4.230 > barajar 3.708 > intercambio 2.000.
    """
    return sum(1 for a, b in zip(original, mutante) if a != b)


def viaje(original: list, mutante: list) -> float:
    """Distancia media, en posiciones, que se movió un símbolo.

    Args:
        original, mutante.

    Returns:
        float.

    Example:
        Por posiciones: desplazamiento 4.663 > inversión 4.230 > barajar 3.708 > intercambio 2.000.
    """
    donde = {simbolo: indice for indice, simbolo in enumerate(mutante)}
    return statistics.fmean(abs(donde[simbolo] - indice)
                            for indice, simbolo in enumerate(original))


def mutacion_inversion_bit(lista_genes: list[int]) -> list[int]:
    """El operador de Gridin para un cromosoma BINARIO: elige una posición, inviértela.

    Args:
        lista_genes.

    Returns:
        list[int].

    Example:
        Por posiciones: desplazamiento 4.663 > inversión 4.230 > barajar 3.708 > intercambio 2.000.
    """
    mutante = copy.deepcopy(lista_genes)
    posicion = random.randint(0, len(mutante) - 1)
    mutante[posicion] = (mutante[posicion] + 1) % 2
    return mutante


def mutacion_intercambio(lista_genes: list, posiciones: tuple[int, int] | None = None
                      ) -> list:
    """Intercambia los símbolos en dos posiciones: el movimiento legal mínimo.

    Args:
        lista_genes, posiciones.

    Returns:
        list.

    Example:
        Cambia exactamente 2 posiciones; viaje idéntico al desplazamiento en 90 de 90 pares.
    """
    mutante = copy.deepcopy(lista_genes)
    if posiciones is None:
        posiciones = random.sample(range(len(mutante)), 2)
    i, j = posiciones
    mutante[i], mutante[j] = mutante[j], mutante[i]
    return mutante


# --- NUEVO (1) mutacion_inversion() -------------------------------------------
def mutacion_inversion(lista_genes: list, posiciones: tuple[int, int] | None = None
                       ) -> list:
    """Invierte el bloque entre dos posiciones.

    El contenido del bloque queda intacto como un CONJUNTO; solo cambia la dirección de
    lectura. En una ruta este es el movimiento clásico: deshacer un cruce del camino
    sin perturbar nada fuera del tramo invertido. Nota que mueve los
    dos extremos del bloque más lejos y deja el medio casi donde estaba.

    Args:
        lista_genes, posiciones.

    Returns:
        list.

    Example:
        Mayor viaje (1.308) entre los cuatro reordenamientos.
    """
    mutante = copy.deepcopy(lista_genes)
    fuente = copy.deepcopy(lista_genes)
    if posiciones is None:
        posiciones = random.sample(range(len(mutante)), 2)
    bajo, alto = sorted(posiciones)
    for desplazamiento in range(0, alto - bajo + 1):
        mutante[bajo + desplazamiento] = fuente[alto - desplazamiento]
    return mutante
# ------------------------------------------------------------------------------


# --- NUEVO (2) mutacion_desplazamiento() --------------------------------------
def mutacion_desplazamiento(lista_genes: list, posiciones: tuple[int, int] | None = None
                   ) -> list:
    """Toma el símbolo en una posición y re-insértalo en otra.

    Todo entre las dos posiciones se desliza un lugar para hacer espacio. Así que el
    número de posiciones que esto perturba es todo el tramo - lo cual suena violento -
    mientras que cada símbolo excepto uno se mueve exactamente un lugar, lo cual no lo es.

    Args:
        lista_genes, posiciones.

    Returns:
        list.

    Example:
        Más posiciones perturbadas: 4.663.
    """
    mutante = copy.deepcopy(lista_genes)
    if posiciones is None:
        posiciones = random.sample(range(len(mutante)), 2)
    fuente, destino = posiciones
    viajando = mutante[fuente]
    direccion = int(copysign(1, destino - fuente))
    for indice in range(fuente, destino, direccion):
        mutante[indice] = mutante[indice + direccion]
    mutante[destino] = viajando
    return mutante
# ------------------------------------------------------------------------------


# --- NUEVO (3) mutacion_barajar() ---------------------------------------------
def mutacion_barajar(lista_genes: list, posiciones: tuple[int, int] | None = None
                     ) -> list:
    """Aleatoriza el orden del bloque entre dos posiciones.

    El único de los cuatro cuya salida no está determinada por las dos posiciones
    que extrajo - y el único que puede devolver el individuo sin cambios, porque
    una permutación aleatoria de un bloque corto a menudo es la identidad. Eso
    se mide a continuación en lugar de afirmarlo.

    Args:
        lista_genes, posiciones.

    Returns:
        list.

    Example:
        Devuelve el individuo intacto 271 de 2000 veces (13.6%).
    """
    mutante = copy.deepcopy(lista_genes)
    if posiciones is None:
        posiciones = random.sample(range(len(mutante)), 2)
    bajo, alto = sorted(posiciones)
    bloque = mutante[bajo:alto + 1]
    random.shuffle(bloque)
    mutante[bajo:alto + 1] = bloque
    return mutante
# ------------------------------------------------------------------------------


def reporte_reordenamiento(lista_genes: list, operador, intentos: int = INTENTOS) -> dict:
    """Aplica un operador `intentos` veces y describe el daño.

    Args:
        lista_genes, operador, intentos.

    Returns:
        dict.

    Example:
        Por posiciones: desplazamiento 4.663 > inversión 4.230 > barajar 3.708 > intercambio 2.000.
    """
    random.seed(SEMILLA)
    cambiados, distancias, intactos, ilegales = [], [], 0, 0
    for _ in range(intentos):
        mutante = operador(lista_genes)
        if not es_permutacion(mutante, lista_genes):
            ilegales += 1
            continue
        conteo = posiciones_cambiadas(lista_genes, mutante)
        cambiados.append(conteo)
        distancias.append(viaje(lista_genes, mutante))
        if conteo == 0:
            intactos += 1
    return {"intentos": intentos,
            "ilegales": ilegales,
            "media_cambiados": statistics.fmean(cambiados) if cambiados else 0.0,
            "max_cambiados": max(cambiados) if cambiados else 0,
            "media_viaje": statistics.fmean(distancias) if distancias else 0.0,
            "max_viaje": max(distancias) if distancias else 0.0,
            "intactos": intactos,
            "cambiados": cambiados}


# --- NUEVO (4) comparar_operadores() ------------------------------------------
OPERADORES = (("intercambio", mutacion_intercambio),
             ("inversion", mutacion_inversion),
             ("desplazamiento", mutacion_desplazamiento),
             ("barajar", mutacion_barajar))


def comparar_operadores(lista_genes: list, intentos: int = INTENTOS) -> list[dict]:
    """Cada operador legal, mismo individuo, misma semilla, mismo instrumento.

    Esta es la contraparte del barrido de régimen del paso 4 en el primer ejemplo: las
    filas son comparables porque lo único que difiere entre ellas es
    qué operador fue llamado.

    Args:
        lista_genes, intentos.

    Returns:
        list[dict].

    Example:
        Por posiciones: desplazamiento 4.663 > inversión 4.230 > barajar 3.708 > intercambio 2.000.
    """
    filas = []
    for nombre, operador in OPERADORES:
        reporte = reporte_reordenamiento(lista_genes, operador, intentos)
        reporte["nombre"] = nombre
        filas.append(reporte)
    return filas


def viaje_sobre_pares(lista_genes: list, operador) -> dict[tuple[int, int], float]:
    """Cada par ordenado de posiciones distintas, alimentado a un operador.

    Cero aleatoriedad: esto pregunta qué HACE el operador, no qué hace en
    promedio. Solo tiene sentido para los tres operadores cuya salida está fijada por
    el par, razón por la cual barajar queda fuera de esto.

    Args:
        lista_genes, operador.

    Returns:
        dict[tuple[int, int], float].

    Example:
        Por posiciones: desplazamiento 4.663 > inversión 4.230 > barajar 3.708 > intercambio 2.000.
    """
    longitud = len(lista_genes)
    return {(i, j): viaje(lista_genes, operador(lista_genes, (i, j)))
            for i in range(longitud) for j in range(longitud) if i != j}
# ------------------------------------------------------------------------------


viaje_ruta = list(range(1, LONGITUD_CROMOSOMA + 1))
print(f"El mismo cromosoma que el paso 1:\n    {viaje_ruta}\n")

print("Un ejemplo de cada uno, desde la misma semilla:")
for nombre, operador in OPERADORES:
    random.seed(SEMILLA)
    print(f"    {nombre:<14} -> {operador(viaje_ruta)}")

filas = comparar_operadores(viaje_ruta)
formato_fila = " {:<14} | {:>21} | {:>11} | {:>11} | {:>11} | {:>11}"
encabezado = formato_fila.format("operador", "posiciones cambiadas", "max",
                           "media viaje", "max", "sin cambios")
print(f"\nCada operador aplicado {INTENTOS} veces a ese individuo:\n")
print(encabezado)
print("-" * len(encabezado))
for r in filas:
    print(formato_fila.format(r["nombre"], f"{r['media_cambiados']:.3f}",
                            r["max_cambiados"], f"{r['media_viaje']:.3f}",
                            f"{r['max_viaje']:.3f}",
                            f"{r['intactos']} / {r['intentos']}"))

ilegales = sum(r["ilegales"] for r in filas)
print(f"\nResultados ilegales en las {len(filas) * INTENTOS} mutaciones: {ilegales}.")
print("Cada uno de estos operadores solo reordena, así que la validez es estructural")
print("en lugar de estar comprobada - no hay ningún paso de reparación en ningún lugar de este archivo.")

# Los dos ordenamientos, y su desacuerdo.
por_posiciones = sorted(filas, key=lambda r: -r["media_cambiados"])
por_viaje = sorted(filas, key=lambda r: -r["media_viaje"])
print("\nClasificados por cuánto del cromosoma perturban:")
print("    " + "  >  ".join(f"{r['nombre']} ({r['media_cambiados']:.2f})"
                            for r in por_posiciones))
print("Clasificados por qué tan lejos mueven realmente un símbolo:")
print("    " + "  >  ".join(f"{r['nombre']} ({r['media_viaje']:.2f})"
                            for r in por_viaje))
if [r["nombre"] for r in por_posiciones] != [r["nombre"] for r in por_viaje]:
    mas_ancho = por_posiciones[0]
    mas_lejos = por_viaje[0]
    print(f"\nLos dos ordenamientos no concuerdan. {mas_ancho['nombre'].capitalize()} toca la "
          f"mayor cantidad de posiciones")
    print(f"({mas_ancho['media_cambiados']:.3f} de {LONGITUD_CROMOSOMA}) y mueve "
          f"los símbolos {mas_ancho['media_viaje']:.3f} lugares en promedio;")
    print(f"{mas_lejos['nombre']} toca {mas_lejos['media_cambiados']:.3f} y los mueve "
          f"{mas_lejos['media_viaje']:.3f}.")
    print("El alcance en una permutación no es una sola cantidad. 'Cuánto es perturbado'")
    print("y 'qué tan lejos llega algo' son preguntas diferentes con respuestas")
    print("diferentes, y una diapositiva que muestra solo la primera es engañosa.")
else:
    print("\nLos dos ordenamientos concuerdan, así que en este cromosoma bastaría con un número.")

# La afirmación exacta detrás de las dos filas de desplazamiento e intercambio.
viaje_intercambio = viaje_sobre_pares(viaje_ruta, mutacion_intercambio)
viaje_desplazamiento = viaje_sobre_pares(viaje_ruta, mutacion_desplazamiento)
identicos = sum(1 for par in viaje_intercambio
                if abs(viaje_intercambio[par] - viaje_desplazamiento[par]) < 1e-12)
print(f"\nPor qué el intercambio y el desplazamiento comparten un viaje medio, exactamente y no por suerte:")
print(f"sobre todos los {len(viaje_intercambio)} pares ordenados de posiciones distintas, los "
      f"dos operadores")
print(f"producen el mismo viaje total en {identicos} de ellos.")
print("Intercambiar las posiciones i y j mueve dos símbolos |i - j| lugares cada uno;")
print("desplazar de i a j mueve un símbolo |i - j| lugares y cada uno de los")
print("símbolos entre ellos exactamente un lugar. Ambos resultan en 2|i - j|.")
print("Misma distancia cubierta, repartida en 2 símbolos o en |i - j| + 1 de ellos.")

# El único operador que puede no hacer nada.
perezoso = max(filas, key=lambda r: r["intactos"])
print(f"\nY la última columna: {perezoso['nombre']} devolvió el individuo sin cambios "
      f"{perezoso['intactos']} veces")
print(f"en {perezoso['intentos']} ({100 * perezoso['intactos'] / perezoso['intentos']:.1f}%). "
      "Extrae dos posiciones y aleatoriza el")
print("bloque entre ellas; cuando ese bloque es corto, un orden aleatorio de él es")
print("a menudo el orden que ya tenía. El operador que suena más violento en")
print("la lista es el único con una posibilidad real de no hacer nada.")

print("\n--- ambas mitades de la lección, en una oración ------------------------")
print("El alcance es el único dial que tiene la mutación, sea cual sea la codificación: sigma para")
print("un gen real, el tamaño del bloque reordenado para una permutación. Y en")
print("ambas mitades la configuración más ancha no fue la mejor. La tabla del paso 4")
print("dice 96 de 100 ejecuciones con sigma 1.0 contra 82 con sigma 3.0; esta tabla")
print("dice que el operador que perturba más posiciones es el que")
print("menos mueve los símbolos. Más grande es una dirección, no una mejora.")

# La imagen: las dos mediciones de alcance, lado a lado, por operador.
nombres = [r["nombre"] for r in filas]
posiciones = np.arange(len(nombres))
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.bar(posiciones - 0.2, [r["media_cambiados"] for r in filas], width=0.4,
       color="tab:blue", label="posiciones cambiadas (de 10)")
ax.bar(posiciones + 0.2, [r["media_viaje"] for r in filas], width=0.4,
       color="tab:orange", label="viaje medio por símbolo (posiciones)")
for indice, r in enumerate(filas):
    if r["intactos"]:
        ax.annotate(f"no hace nada\n{100 * r['intactos'] / r['intentos']:.1f}% "
                    f"del tiempo",
                    (indice, r["media_cambiados"]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8, color="tab:red")
ax.set_xticks(posiciones)
ax.set_xticklabels(nombres)
ax.set_ylabel("posiciones")
ax.set_title(f"Dos mediciones de alcance, {INTENTOS} mutaciones de una "
             f"permutación de {LONGITUD_CROMOSOMA} símbolos")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.legend(fontsize=9)
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "mutacion_permutacion_02_reordenamientos.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/mutacion_permutacion_02_reordenamientos.png")

