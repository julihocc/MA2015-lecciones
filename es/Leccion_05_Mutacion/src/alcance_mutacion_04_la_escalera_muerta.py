"""
Lección 05 - Paso 4: La ejecución que la Lección 02 dio por muerta
====================================================
NUEVO EN ESTE PASO: el ciclo generacional de la Lección 02 y su paisaje de escalera, una
medición de qué tan anchos son realmente los escalones de esa escalera, y un barrido de
cada régimen de mutación sobre 100 ejecuciones independientes.

Una versión anterior del paso 5 de la Lección 02 terminó con una ejecución que se detuvo a las seis
generaciones, a 0.2000 por debajo del óptimo de la escalera, casi sin dispersión genética
restante, y apuntó hacia aquí: la mutación "es el único operador que podría restaurarlo". Este paso es donde esa promesa
venció - y no venció de la forma
en que estaba escrita. Lo que la investigación encontró en su lugar envió una corrección de vuelta a
la Lección 02, que ya no hace esa afirmación.

Lo que el barrido encuentra: ningún régimen de mutación rescata esa ejecución. Ni un sigma más grande,
ni una tasa más alta, ni ninguna de las ocho combinaciones probadas, sobre 100 semillas cada una.
La mutación multiplica la dispersión genética sobreviviente varias veces y compra
exactamente cero de aptitud.

Así que hubo que hacer la segunda pregunta - ¿por qué la cima de esa escalera
es inalcanzable? - y la respuesta no tiene nada que ver con la mutación. Con la altura de la tienda
escrita como 10.0, el escalón superior de esa escalera es un solo punto: tiene ANCHO
CERO. La cuadrícula de fuerza bruta que declaró el óptimo +2.0000 lo encontró solo
porque x = 4.0 resulta ser uno de sus 20001 puntos de muestra; ninguna ejecución podría jamás
aterrizar en él, con ningún operador. El déficit era inalcanzable por construcción, por lo
que la "ejecución muerta" que describió no estaba muerta - había llegado.

Levanta el paisaje por medio escalón - una constante - y el escalón superior se vuelve de
0.5 unidades de ancho. Entonces el mismo régimen de mutación, completamente sin cambios, lo alcanza
en 96 de 100 ejecuciones.

Es por eso que la Lección 02 ahora usa una tienda de 10.5 y reporta una hipótesis refutada
en lugar de un déficit. La geometría rota se mantiene aquí a propósito, como la
cosa siendo diagnosticada: este paso es el diagnóstico, no el bug.

La lección vale más que la que estaba planeada. Antes de preguntar qué
operador está fallando, comprueba que el objetivo exista.

CAMBIOS RESPECTO A alcance_mutacion_03_guiada_por_aptitud.py
Introdúcelos en este orden:
    1. ejecutar()             el ciclo generacional completo de la Lección 02, traído sin cambios
    2. escalera()             el tercer paisaje de la Lección 02, en el que murió su ejecución
    3. anchos_escalones()     medir qué tan ancho es realmente cada escalón de ese paisaje
    4. ALTURA_PICO_REPARADA   la reparación que demanda la medida: medio escalón de margen
    5. barrido()              cada régimen de mutación a través de 100 ejecuciones independientes, en ambos paisajes

Ejecútalo:  python alcance_mutacion_04_la_escalera_muerta.py

Ocho regímenes x 100 corridas: el escalón superior se alcanza 0 veces en 800.
Reparada (altura 10.5, ancho 0.500), el régimen de la Lección 02 llega 96 de 100.
"""
import random
import statistics
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 52
GEN_MIN, GEN_MAX = -10.0, 10.0
MUTACION_MU = 0.0
# Ajustes de la Lección 02, al dígito. Nada aquí está afinado para esta lección.
TAMANO_POBLACION = 10
PROBABILIDAD_CRUZA = 0.8
MAX_GENERACIONES = 60
TAMANO_TORNEO, ALFA_MEZCLA = 3, 1.0
PACIENCIA, MEJORA_MINIMA = 5, 1e-4
# La geometría de la escalera, tal como la Lección 02 la tenía primero. ALTURA_PICO es el valor
# que la tienda alcanza en su cúspide antes de que floor() lo convierta en escalones; estaba escrito
# en línea como el literal 10.0 allí, y se nombra aquí porque todo el paso
# gira en torno a él. La Lección 02 ha sido corregida desde entonces a 10.5 - debido a lo que este
# paso encontró - así que el 10.0 abajo es deliberadamente el valor ANTIGUO: es el
# espécimen bajo el microscopio, no una copia que se desactualizó.
PICO = 4.0
PENDIENTE = 2.0
ALTURA_PICO = 10.0
ESCALON, ELEVACION = 1.0, 0.2
# Ocho regímenes de mutación: (probabilidad de que un individuo mute, sigma). Con
# un gen por individuo esta probabilidad ES la p por gen del paso 2 - las dos
# monedas coinciden - por lo que la cuadrícula de regímenes es de los mismos dos diales que antes.
REGIMENES = ((0.1, 1.0), (0.1, 3.0), (0.3, 1.0), (0.3, 3.0),
           (0.6, 1.0), (0.6, 3.0), (1.0, 1.0), (1.0, 3.0))
REGIMEN_LECCION_02 = (0.1, 1.0)
EJECUCIONES = 100
FIGURAS = Path(__file__).resolve().parent.parent / "figures"

Aptitud = Callable[[float], float]


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


# --- NUEVO (2) escalera() -----------------------------------------------------
def escalera(x: float, altura: float = ALTURA_PICO) -> float:
    """El tercer paisaje de la Lección 02: el progreso llega en saltos, no en una pendiente.

    Una tienda de altura `altura` centrada en PICO, pasada por floor() de modo que
    cada punto en un escalón puntúa idéntico. Dentro de un escalón, la selección no tiene
    nada que comparar y la cruza nada que combinar; el único operador que
    puede cruzar un límite de escalón es la mutación. Es por eso que la Lección 02 le entregó el
    problema a esta lección.

    `altura` es un parámetro en lugar del literal 10.0 de la Lección 02 porque
    anchos_escalones() abajo está a punto de mostrar que el literal es todo el problema.

    Args:
        x, altura.

    Returns:
        float.

    Example:
        Con altura 10.0 el escalón superior tiene ancho 0.000; con 10.5, 0.500.
    """
    return float(np.floor((altura - PENDIENTE * abs(x - PICO)) / ESCALON) * ELEVACION)
# ------------------------------------------------------------------------------


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


# --- NUEVO (1) ejecutar() -----------------------------------------------------
# Todo en esta banda pertenece a la Lección 02, trasplantado sin cambios: el
# individuo, la selección por torneo, la cruza por mezcla, la mutación gaussiana, la
# generación de cinco fases y la regla de paciencia. Es nuevo en este ARCHIVO, no para
# los estudiantes. Los dos diales de mutación son argumentos de ejecutar() en lugar de constantes
# de módulo, que es la única edición, y es lo que hace posible un barrido.
#
# mutar() mantiene la forma exacta de la Lección 02 - una moneda para el individuo, luego una
# extracción gaussiana - en lugar de llamar a mutar_desviacion_aleatoria() del paso 3. Con un
# cromosoma de un solo gen, los dos son el mismo operador, pero consumen el
# flujo aleatorio de manera diferente, y reproducir la ejecución de la Lección 02 al dígito
# requiere su flujo.
class Individuo:
    """Una solución candidata, que lleva la función que la juzga.

    Args:
        argumentos del constructor.

    Example:
        Ocho regímenes x 100 corridas: el escalón superior se alcanza 0 veces en 800.
    """

    def __init__(self, lista_genes: list[float], funcion_aptitud: Aptitud) -> None:
        self.lista_genes = lista_genes
        self.funcion_aptitud = funcion_aptitud
        self.aptitud = funcion_aptitud(lista_genes[0])

    @property
    def gen(self) -> float:
        return self.lista_genes[0]

    def __repr__(self) -> str:
        return f"x={self.gen:+.3f} f={self.aptitud:+.3f}"


def seleccion_torneo(poblacion: list[Individuo], tamano: int) -> list[Individuo]:
    """seleccion_torneo.

    Args:
        poblacion, tamano.

    Returns:
        list[Individuo].

    Example:
        Ocho regímenes x 100 corridas: el escalón superior se alcanza 0 veces en 800.
    """
    return [max([random.choice(poblacion) for _ in range(tamano)],
                key=lambda i: i.aptitud) for _ in range(len(poblacion))]


def cruza(p1: Individuo, p2: Individuo) -> tuple[Individuo, Individuo]:
    """cruza.

    Args:
        p1, p2.

    Returns:
        tuple[Individuo, Individuo].

    Example:
        Ocho regímenes x 100 corridas: el escalón superior se alcanza 0 veces en 800.
    """
    desplazamiento = (1 + 2 * ALFA_MEZCLA) * random.random() - ALFA_MEZCLA
    g1 = acotar((1 - desplazamiento) * p1.gen + desplazamiento * p2.gen)
    g2 = acotar(desplazamiento * p1.gen + (1 - desplazamiento) * p2.gen)
    return (Individuo([g1], p1.funcion_aptitud),
            Individuo([g2], p1.funcion_aptitud))


def mutar(ind: Individuo, sigma: float) -> Individuo:
    """mutar.

    Args:
        ind, sigma.

    Returns:
        Individuo.

    Example:
        Ocho regímenes x 100 corridas: el escalón superior se alcanza 0 veces en 800.
    """
    return Individuo([acotar(ind.gen + random.gauss(MUTACION_MU, sigma))],
                      ind.funcion_aptitud)


def evolucionar_una_generacion(poblacion: list[Individuo], probabilidad_mutacion: float,
                          sigma: float) -> list[Individuo]:
    """SELECCIONAR -> CRUZAR -> MUTAR -> reemplazar, una vez.

    Args:
        poblacion, probabilidad_mutacion, sigma.

    Returns:
        list[Individuo].

    Example:
        Ocho regímenes x 100 corridas: el escalón superior se alcanza 0 veces en 800.
    """
    seleccionados = seleccion_torneo(poblacion, TAMANO_TORNEO)
    cruzados: list[Individuo] = []
    for p1, p2 in zip(seleccionados[::2], seleccionados[1::2]):
        if random.random() < PROBABILIDAD_CRUZA:
            cruzados.extend(cruza(p1, p2))
        else:
            cruzados.extend([p1, p2])
    return [mutar(ind, sigma) if random.random() < probabilidad_mutacion else ind
            for ind in cruzados]


def ejecutar(funcion_aptitud: Aptitud, probabilidad_mutacion: float, sigma: float,
        semilla: int = SEMILLA, paciencia: int = PACIENCIA) -> list[list[Individuo]]:
    """Inicializar, luego evolucionar hasta que las mejoras se sequen.

    Args:
        funcion_aptitud, probabilidad_mutacion, sigma, semilla, paciencia.

    Returns:
        list[list[Individuo]].

    Example:
        Ocho regímenes x 100 corridas: el escalón superior se alcanza 0 veces en 800.
    """
    random.seed(semilla)
    poblacion = [Individuo([random.uniform(GEN_MIN, GEN_MAX)], funcion_aptitud)
                  for _ in range(TAMANO_POBLACION)]
    generaciones = [poblacion]
    for generacion in range(1, MAX_GENERACIONES + 1):
        poblacion = evolucionar_una_generacion(poblacion, probabilidad_mutacion, sigma)
        generaciones.append(poblacion)
        mejores = [max(i.aptitud for i in pop) for pop in generaciones]
        ultima_mejora = 0
        for g in range(1, len(mejores)):
            if mejores[g] > max(mejores[:g]) + MEJORA_MINIMA:
                ultima_mejora = g
        if paciencia and generacion - ultima_mejora >= paciencia:
            break
    return generaciones


def mejor_historico(generaciones: list[list[Individuo]]) -> Individuo:
    """mejor_historico.

    Args:
        generaciones.

    Returns:
        Individuo.

    Example:
        Ocho regímenes x 100 corridas: el escalón superior se alcanza 0 veces en 800.
    """
    return max((ind for pop in generaciones for ind in pop), key=lambda i: i.aptitud)


def dispersion_genetica(poblacion: list[Individuo]) -> float:
    """dispersion_genetica.

    Args:
        poblacion.

    Returns:
        float.

    Example:
        Ocho regímenes x 100 corridas: el escalón superior se alcanza 0 veces en 800.
    """
    return float(np.std([ind.gen for ind in poblacion]))
# ------------------------------------------------------------------------------


# --- NUEVO (3) anchos_escalones() ---------------------------------------------
def anchos_escalones(altura: float, niveles: int = 4) -> list[tuple[float, float]]:
    """¿Qué tan ancho, en x, es cada uno de los `niveles` superiores de escalones de la escalera?

    Resuelto en lugar de muestreado, lo cual es el punto: una búsqueda en cuadrícula no puede ver un
    escalón más estrecho que su propio espaciado, y ese es precisamente el error que esta
    función existe para atrapar.

    El escalón k cubre los x con floor(altura - PENDIENTE*|x - PICO|) == k, eso es
    (altura - k - 1) / PENDIENTE < |x - PICO| <= (altura - k) / PENDIENTE, por lo que su ancho
    es el doble de la longitud de ese intervalo en |x - PICO|, recortado a cero.

    Args:
        altura, niveles.

    Returns:
        list[tuple[float, float]].

    Example:
        Ocho regímenes x 100 corridas: el escalón superior se alcanza 0 veces en 800.
    """
    tope = int(np.floor(altura))
    anchos = []
    for k in range(tope, tope - niveles, -1):
        exterior = (altura - k) / PENDIENTE
        interior = max(0.0, (altura - k - 1) / PENDIENTE)
        anchos.append((k * ELEVACION, 2 * max(0.0, exterior - interior)))
    return anchos
# ------------------------------------------------------------------------------


# --- NUEVO (4) ALTURA_PICO_REPARADA -------------------------------------------
# Medio escalón de margen. La tienda ahora alcanza su pico en 10.5 en lugar de 10.0, por lo que la
# banda superior de floor() ya no es el punto único donde la tienda toca un
# entero - es un intervalo. Nada más sobre el paisaje cambia: el
# número de escalones, su altura, la pendiente y la posición del pico son todos
# como la Lección 02 los escribió, y el escalón superior sigue siendo interior.
ALTURA_PICO_REPARADA = 10.5
# ------------------------------------------------------------------------------


# --- NUEVO (5) barrido() ------------------------------------------------------
def barrido(altura: float, ejecuciones: int = EJECUCIONES) -> list[dict]:
    """Cada régimen en REGIMENES, durante `ejecuciones` semillas independientes, en un paisaje.

    Una semilla no prueba nada sobre una búsqueda estocástica: la conclusión de la Lección 02
    se basaba en una sola ejecución. `ejecuciones` semillas separadas por régimen es lo que permite
    que una fila de esta tabla se lea como una tasa en lugar de una anécdota.

    Args:
        altura, ejecuciones.

    Returns:
        list[dict].

    Example:
        0 de 800 en la escalera muerta; 96 de 100 con la altura reparada.
    """
    def paisaje(x: float) -> float:
        return escalera(x, altura)

    optimo = max(nivel for nivel, _ in anchos_escalones(altura))
    filas = []
    for probabilidad, sigma in REGIMENES:
        mejores, dispersiones, longitudes = [], [], []
        for semilla in range(ejecuciones):
            generaciones = ejecutar(paisaje, probabilidad, sigma, semilla=semilla)
            mejores.append(mejor_historico(generaciones).aptitud)
            dispersiones.append(dispersion_genetica(generaciones[-1]))
            longitudes.append(len(generaciones) - 1)
        filas.append({"probabilidad": probabilidad, "sigma": sigma,
                     "alcanzaron": sum(1 for b in mejores if b >= optimo - 1e-9),
                     "ejecuciones": ejecuciones,
                     "media_mejor": statistics.fmean(mejores),
                     "media_dispersion": statistics.fmean(dispersiones),
                     "media_generaciones": statistics.fmean(longitudes),
                     "optimo": optimo})
    return filas
# ------------------------------------------------------------------------------


def imprimir_barrido(filas: list[dict]) -> None:
    """imprimir_barrido.

    Args:
        filas.

    Returns:
        None. Imprime o guarda figura.

    Example:
        Ocho regímenes x 100 corridas: el escalón superior se alcanza 0 veces en 800.
    """
    formato_fila = " {:>5} | {:>5} | {:>14} | {:>11} | {:>16} | {:>18}"
    encabezado = formato_fila.format("p", "sigma", "alcanzaron cima", "media mejor",
                               "media dispersion", "media generaciones")
    print(encabezado)
    print("-" * len(encabezado))
    for r in filas:
        print(formato_fila.format(f"{r['probabilidad']:.1f}", f"{r['sigma']:.1f}",
                                f"{r['alcanzaron']} de {r['ejecuciones']}",
                                f"{r['media_mejor']:.4f}",
                                f"{r['media_dispersion']:.3f}",
                                f"{r['media_generaciones']:.2f}"))


# ===== Parte 1: reproducir la ejecución que la Lección 02 dio por perdida =======
probabilidad, sigma = REGIMEN_LECCION_02
generaciones = ejecutar(lambda x: escalera(x), probabilidad, sigma)
campeon = mejor_historico(generaciones)
optimo = max(nivel for nivel, _ in anchos_escalones(ALTURA_PICO))
print("Paso 5 de la Lección 02, reproducido aquí con su propia semilla y sus propios "
      "ajustes:")
print(f"    régimen: probabilidad de mutación {probabilidad}, sigma {sigma}, "
      f"SEMILLA = {SEMILLA}")
print(f"    se detuvo después de {len(generaciones) - 1} generaciones de "
      f"{MAX_GENERACIONES} permitidas")
print(f"    mejor encontrado: {campeon}")
print(f"    óptimo de la escalera: {optimo:+.4f}")
print(f"    déficit: {optimo - campeon.aptitud:+.4f}")
print(f"    dispersión genética en la generación final: "
      f"{dispersion_genetica(generaciones[-1]):.3f}")
print("La Lección 02 reportó el mismo déficit y pasó el problema aquí, con el")
print("argumento de que una población sin dispersión restante necesita mutación para restaurar")
print("algo. Así que: dale más mutación.")

# ===== Parte 2: el barrido que supuestamente lo arreglaría ====================
print(f"\nCada régimen, {EJECUCIONES} ejecuciones independientes cada uno, en la escalera de la Lección 02 "
      f"(altura del pico {ALTURA_PICO}):\n")
antes = barrido(ALTURA_PICO)
imprimir_barrido(antes)

dispersiones = [r["media_dispersion"] for r in antes]
mejores = [r["media_mejor"] for r in antes]
total_alcanzado = sum(r["alcanzaron"] for r in antes)
print(f"\nLa mutación le hace a la dispersión lo que se supone que debe hacer: de "
      f"{min(dispersiones):.3f} a {max(dispersiones):.3f},")
print(f"un factor de {max(dispersiones) / min(dispersiones):.1f} a través de la cuadrícula. Y la "
      f"columna de mejor-aptitud no se mueve:")
print(f"{min(mejores):.4f} a {max(mejores):.4f}, todos ellos por debajo de "
      f"{antes[0]['optimo']:+.4f}.")
print(f"El escalón superior fue alcanzado {total_alcanzado} veces en "
      f"{len(antes) * EJECUCIONES} ejecuciones.")
print("\nEse es un rechazo rotundo, y es demasiado rotundo para tratarse de afinación. Cuando ninguna")
print("configuración de un operador cambia en absoluto un resultado, sospecha de la pregunta.")

# ===== Parte 3: el diagnóstico ================================================
print(f"\n¿Qué tan anchos son los escalones de esta escalera? Resuelto exactamente, no muestreado:")
print("    aptitud del escalón | ancho en x")
print("    --------------------+-----------")
for nivel, ancho in anchos_escalones(ALTURA_PICO):
    print(f"       {nivel:+16.4f} | {ancho:10.3f}")
nivel_superior, ancho_superior = anchos_escalones(ALTURA_PICO)[0]
print(f"\nEl escalón superior tiene ancho {ancho_superior:.3f}. Es el punto único "
      f"x = {PICO:.1f},")
print("donde la tienda toca un entero exactamente. Un gen extraído de una distribución")
print("continua aterriza en él con probabilidad cero, así que ningún régimen de mutación,")
print("ninguna cruza y ninguna cantidad de paciencia pueden jamás producirlo.")
print(f"\nLa comprobación de fuerza bruta de la Lección 02 evaluó el paisaje en "
      f"numpy.linspace({GEN_MIN:.0f}, {GEN_MAX:.0f}, 20001),")
print(f"cuyo espaciado es {20.0 / 20000:.3f} - y {PICO:.1f} es uno de sus "
      f"puntos de muestra. Esa es la única")
print(f"razón por la que el óptimo se reportó como {nivel_superior:+.4f}. La cuadrícula vio un pico "
      "que la búsqueda")
print("no puede alcanzar, y el déficit de 0.2000 que la Lección 02 publicó es el ancho")
print("de esa ilusión.")

# ===== Parte 4: la reparación, y el barrido que importa =======================
print(f"\nLa reparación es una constante: altura del pico {ALTURA_PICO} -> "
      f"{ALTURA_PICO_REPARADA}.")
print("    aptitud del escalón | ancho en x")
print("    --------------------+-----------")
for nivel, ancho in anchos_escalones(ALTURA_PICO_REPARADA):
    print(f"       {nivel:+16.4f} | {ancho:10.3f}")
superior_reparado, ancho_reparado = anchos_escalones(ALTURA_PICO_REPARADA)[0]
print(f"\nMismo número de escalones, misma elevación, misma pendiente, mismo pico interior - y "
      f"un escalón")
print(f"superior de {ancho_reparado:.3f} unidades de ancho, con valor {superior_reparado:+.4f}, que "
      "es lo que el antiguo paisaje")
print("afirmó valer todo el tiempo.")

print(f"\nCada régimen otra vez, {EJECUCIONES} ejecuciones cada uno, en la escalera reparada:\n")
despues = barrido(ALTURA_PICO_REPARADA)
imprimir_barrido(despues)

fila_leccion_02 = next(r for r in despues
                     if (r["probabilidad"], r["sigma"]) == REGIMEN_LECCION_02)
mejor_fila = max(despues, key=lambda r: r["alcanzaron"])
peor_fila = min(despues, key=lambda r: r["alcanzaron"])
print(f"\nEl propio régimen de la Lección 02 - p = {fila_leccion_02['probabilidad']}, "
      f"sigma = {fila_leccion_02['sigma']}, el que llamó insuficiente -")
print(f"alcanza la cima en {fila_leccion_02['alcanzaron']} de "
      f"{fila_leccion_02['ejecuciones']} ejecuciones. Nada sobre la mutación fue cambiado para")
print("obtener eso. El paisaje sí lo fue.")
print(f"\nEl mejor régimen en la tabla es p = {mejor_fila['probabilidad']}, "
      f"sigma = {mejor_fila['sigma']} con {mejor_fila['alcanzaron']} de "
      f"{mejor_fila['ejecuciones']};")
print(f"el peor es p = {peor_fila['probabilidad']}, sigma = "
      f"{peor_fila['sigma']} con {peor_fila['alcanzaron']}. Nota cuál es ese:")
if peor_fila["sigma"] > mejor_fila["sigma"]:
    print("el sigma MAYOR es el peor. El paso 1 encontró el mismo intercambio en un")
    print("solo gen - un paso largo es bueno para llegar y malo para quedarse -")
    print("y aquí cuesta ejecuciones completas: una mutación lo suficientemente grande para encontrar el escalón")
    print("superior también es lo suficientemente grande para derribar a su descubridor de nuevo.")
    pares = [(r["probabilidad"],
              next(s["alcanzaron"] for s in despues
                   if s["probabilidad"] == r["probabilidad"] and s["sigma"] == 1.0),
              next(s["alcanzaron"] for s in despues
                   if s["probabilidad"] == r["probabilidad"] and s["sigma"] == 3.0))
             for r in despues if r["sigma"] == 1.0]
    perdidas = sum(1 for _, at1, at3 in pares if at3 < at1)
    print(f"Se mantiene a cada tasa en la tabla: en {perdidas} de {len(pares)} "
          "pares compartiendo un p,")
    print("    " + ",  ".join(f"p={q}: {at1} -> {at3}" for q, at1, at3 in pares))
    print("    elevar sigma de 1.0 a 3.0 cuesta ejecuciones.")
else:
    print("el sigma mayor es el mejor aquí, lo cual no es lo que predijo el intercambio")
    print("del paso 1; la escalera es lo suficientemente ancha para perdonar pasos largos.")

print("\n--- lo que este paso realmente enseñó ----------------------------------")
print("La Lección 02 prometió que la mutación podría revivir su ejecución muerta. No pudo,")
print("y medir ocho regímenes sobre 100 ejecuciones cada uno es lo que lo demostró. El")
print("fallo estaba en el paisaje, no en el operador: el objetivo tenía un")
print("ancho de cero. Antes de afinar un operador, comprueba que lo que estás pidiendo")
print("exista - y luego mide sobre muchas semillas, porque una ejecución no puede distinguir")
print("una búsqueda muerta de una imposible.")

# La imagen: los dos paisajes en el pico, y las tasas de éxito.
fig, (ax_land, ax_rate) = plt.subplots(1, 2, figsize=(11.5, 4.0))
fino = np.linspace(PICO - 2.0, PICO + 2.0, 4001)
ax_land.plot(fino, [escalera(float(x), ALTURA_PICO) for x in fino],
             color="tab:red", linewidth=1.6,
             label=f"altura pico {ALTURA_PICO} (escalón superior de ancho {ancho_superior:.2f})")
ax_land.plot(fino, [escalera(float(x), ALTURA_PICO_REPARADA) for x in fino],
             color="tab:green", linewidth=1.6, linestyle="--",
             label=f"altura pico {ALTURA_PICO_REPARADA} "
                   f"(escalón superior de ancho {ancho_reparado:.2f})")
ax_land.plot([PICO], [nivel_superior], "*", color="tab:red", markersize=13)
ax_land.set_title("La cima de la escalera, magnificada", fontsize=10)
ax_land.set_xlabel("x")
ax_land.set_ylabel("aptitud")
ax_land.grid(True, linestyle=":", alpha=0.5)
ax_land.legend(fontsize=8, loc="lower center")

etiquetas = [f"{r['probabilidad']}/{r['sigma']}" for r in antes]
posiciones = np.arange(len(etiquetas))
ax_rate.bar(posiciones - 0.2, [100 * r["alcanzaron"] / r["ejecuciones"] for r in antes],
            width=0.4, color="tab:red", label=f"altura pico {ALTURA_PICO}")
ax_rate.bar(posiciones + 0.2, [100 * r["alcanzaron"] / r["ejecuciones"] for r in despues],
            width=0.4, color="tab:green",
            label=f"altura pico {ALTURA_PICO_REPARADA}")
ax_rate.set_xticks(posiciones)
ax_rate.set_xticklabels(etiquetas, fontsize=8)
ax_rate.set_xlabel("régimen de mutación (probabilidad / sigma)")
ax_rate.set_ylabel(f"% de {EJECUCIONES} ejecuciones que alcanzan escalón superior")
ax_rate.set_title("Ningún régimen rescata el primer paisaje; "
                  "cada régimen resuelve el segundo", fontsize=10)
ax_rate.grid(True, axis="y", linestyle=":", alpha=0.5)
ax_rate.legend(fontsize=8)
fig.suptitle("La escalera muerta de la Lección 02 no era un problema de mutación")
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "alcance_mutacion_04_la_escalera_muerta.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/alcance_mutacion_04_la_escalera_muerta.png")

