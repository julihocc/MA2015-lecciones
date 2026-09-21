"""
Lección 13 - Paso 5: Evaluando una generación en un pool de procesos
==============================================================
NUEVO EN ESTE PASO: evaluación por lotes, y un pool de multiprocesamiento detrás.

Esta es la sección 13.5 del libro, y la primera técnica en esta lección que
no elimina ningún trabajo en absoluto. La caché elimina evaluaciones. Un pool no elimina
ninguna: ejecuta las mismas evaluaciones en otro lugar, al mismo tiempo.

Para despachar el trabajo, primero debes tenerlo en un solo lugar, y los pasos 2 a 4 lo tenían
en el peor lugar posible - dentro de un constructor, un individuo a la vez.
Así que el contenido real de este paso no es `Pool`; es la reestructuración que
hace posible un pool. La reproducción ahora devuelve *genomas*, toda una generación de
ellos se evalúa en una sola llamada, y solo entonces se construyen los individuos.
Los sobrevivientes que no fueron ni cruzados ni mutados se mantienen como los mismos
objetos, de modo que el ahorro de la sección 13.1 sobrevive a la reescritura intacto.

El agrupamiento resultó en un ahorro que este paso no estaba construido para ahorrar, y el
informe lo dice en lugar de ocultarlo. En los pasos 2 a 4, un hijo que fue cruzado
y luego mutado se evaluaba dos veces: una como el producto de la cruza, otra como
el mutante, y el primer valor se desechaba sin usarse. Posponer la evaluación
hasta que la reproducción haya terminado evalúa solo lo que sobrevive hasta el final. La
misma ejecución cae de 409 evaluaciones a 291 - un ahorro del 28.9% que no tiene nada que
ver con el paralelismo y todo que ver con evaluar tarde.

Dos cosas que este paso mide, y una a la que se niega.

  * Mide que ambas rutas den la misma respuesta, generación por generación.
  * Mide los contadores del proceso padre después de una ejecución con pool - y los encuentra
    en cero, lo cual es un hecho sobre los procesos, no sobre la aritmética.
  * Imprime una lectura de reloj, y la marca como lo que es: una medición de
    la máquina que ejecutó el script, no una propiedad del algoritmo. Las tasas de aciertos
    del paso 3 son las mismas en cada computadora de la habitación. Los segundos a continuación
    no lo son.

CAMBIOS RESPECTO A turnos_04_instantanea.py
Introdúcelos en este orden:
    1. Individuo             el constructor toma una aptitud; deja de evaluar
    2. evaluar_lote()        una llamada evalúa una generación entera, con o sin pool
    3. materializar()        los sobrevivientes y los nuevos genomas se convierten en una lista de individuos
    4. operacion_cruza()     devuelve genomas en bruto para los hijos
    5. operacion_mutacion()  lo mismo, y pasa los individuos intactos a través
    6. ejecutar()            toma un pool, y evalúa cada generación en un solo lote
    7. el_reporte            ambas rutas, la misma respuesta, y un reloj acotado

Ejecútalo:  python turnos_05_evaluacion_paralela.py

Solo agrupar reduce de 409 a 291 evaluaciones (−28.9%). El padre, tras
el pool, lee 0 evaluaciones y 0 unidades de trabajo: el trabajo se fue a
otro proceso, no desapareció.
"""
import json
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt

SEMILLA = 3
EMPLEADOS = 5
DIAS = 7
TURNOS_POR_DIA = 3
TURNOS = DIAS * TURNOS_POR_DIA
LONGITUD_GENOMA = EMPLEADOS * TURNOS
TAMANO_POBLACION = 30
TAMANO_ELITE = 2
PROBABILIDAD_CRUZA = 0.8
PUNTOS_CRUZA = 3
PROBABILIDAD_MUTACION = 0.5
MAX_GENERACIONES = 10
# (mínimo, máximo) de personal para el turno de mañana, día y noche.
LIMITES_TURNO = ((1, 4), (2, 5), (1, 2))
# Cuántos turnos siguientes debe descansar un empleado después de cada tipo de turno.
DESCANSO_DESPUES = (1, 1, 3)
PENALIZACION_TURNO_VACIO = 100
PESO_DESCANSO = 5
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"

# Una instantánea es un archivo ordinario, por lo que pertenece a un lugar donde un estudiante pueda abrirlo
# y leer los genes como texto.
INSTANTANEAS = Path(__file__).resolve().parent.parent / "snapshots"
# Lo suficientemente pequeño como para portarse bien en cualquier máquina de enseñanza; el script imprime cuántas
# CPUs encontró en realidad.
TRABAJADORES = min(4, os.cpu_count() or 1)
# Una variante deliberadamente cara de la misma aptitud, utilizada una vez al final para
# mostrar que el veredicto del pool depende enteramente del costo de una llamada.
REPETICIONES_PESADAS = 20

# El medidor. No se afirma nada en esta lección sin leerlo.
COSTO: Dict[str, int] = {"llamadas": 0, "trabajo": 0, "construidos": 0}


def desviacion_turno(genoma: List[int]) -> int:
    """Penalización por cada turno con personal fuera de su banda permitida.

    Recorre todas las celdas EMPLEADOS x TURNOS, que es de donde viene el costo de una
    evaluación y por qué el contador de trabajo vive en el bucle interior.
    """
    penalizacion = 0
    for t in range(TURNOS):
        en_servicio = 0
        for e in range(EMPLEADOS):
            COSTO["trabajo"] += 1
            en_servicio += genoma[e * TURNOS + t]
        bajo, alto = LIMITES_TURNO[t % TURNOS_POR_DIA]
        penalizacion += max(bajo - en_servicio, 0) + max(en_servicio - alto, 0)
        if en_servicio == 0:
            penalizacion += PENALIZACION_TURNO_VACIO
    return penalizacion


def violaciones_descanso(genoma: List[int]) -> int:
    """Cuenta las veces que un empleado es puesto de nuevo en servicio mientras aún descansa."""
    violaciones = 0
    for e in range(EMPLEADOS):
        descansando = 0
        for t in range(TURNOS):
            COSTO["trabajo"] += 1
            if genoma[e * TURNOS + t] == 1:
                if descansando > 0:
                    violaciones += 1
                descansando = DESCANSO_DESPUES[t % TURNOS_POR_DIA]
            else:
                descansando = max(0, descansando - 1)
    return violaciones


# Una entrada por cada genoma distinto jamás construido. Un acierto es una evaluación que nunca
# sucede; el contador es lo que permite que el informe de abajo indique un ahorro en lugar de
# reclamar uno.
CACHE: Dict[Tuple[int, ...], int] = {}
COSTO["aciertos"] = 0


def aptitud_turnos(genoma: List[int]) -> int:
    """El objetivo, a MAXIMIZAR. Una asignación perfecta puntúa 0."""
    COSTO["llamadas"] += 1
    return -(desviacion_turno(genoma) + PESO_DESCANSO * violaciones_descanso(genoma))


def aptitud_en_cache(genoma: List[int]) -> int:
    """Responde desde la tabla si este genoma exacto ha sido evaluado antes.

    La clave tiene que ser hasheable y tiene que ser *todo* el genoma: dos asignaciones
    que difieren en un bit son dos asignaciones diferentes.
    """
    clave = tuple(genoma)
    if clave in CACHE:
        COSTO["aciertos"] += 1
        return CACHE[clave]
    valor = aptitud_turnos(genoma)
    CACHE[clave] = valor
    return valor


# --- NUEVO (1) Individuo ------------------------------------------------------
class Individuo:
    """Una asignación candidata y su aptitud, que ahora llega desde afuera.

    Evaluar dentro del constructor era lo correcto en el paso 2 y
    es lo incorrecto aquí: una evaluación enterrada en el nacimiento de un objeto
    no puede ser agrupada, y lo que no puede ser agrupado no puede ser despachado. La
    aptitud todavía se calcula exactamente una vez por genoma - solo que no aquí.
    """

    def __init__(self, lista_genes: List[int], aptitud: int) -> None:
        self.lista_genes = list(lista_genes)
        self.aptitud = aptitud
        COSTO["construidos"] += 1
# ------------------------------------------------------------------------------


# --- NUEVO (2) evaluar_lote() -------------------------------------------------
def evaluar_lote(genomas: Sequence[List[int]], pool: Optional[Pool] = None) -> List[int]:
    """Evalúa toda una generación. Con un pool, en tantos procesos como tenga.

    ``pool.map`` conserva el orden, así que la ruta paralela devuelve la
    misma lista que la secuencial.

    Args:
        genomas: lista de cromosomas de 105 bits.
        pool: un ``multiprocessing.Pool`` o None (secuencial).

    Returns:
        Aptitudes en el mismo orden que ``genomas``.
    """
    if pool is None:
        return [aptitud_en_cache(genoma) for genoma in genomas]
    return list(pool.map(aptitud_en_cache, genomas))
# ------------------------------------------------------------------------------


# --- NUEVO (3) materializar() -------------------------------------------------
Criado = Union[Individuo, List[int]]


def materializar(elementos: List[Criado], pool: Optional[Pool] = None) -> List[Individuo]:
    """Convierte sobrevivientes y genomas nuevos en individuos.

    Los sobrevivientes pasan intactos y no cuestan nada (sección 13.1).
    Solo los genomas crudos se evalúan.

    Args:
        elementos: mezcla de ``Individuo`` (sobrevivientes) y listas de bits.
        pool: opcional, se pasa a ``evaluar_lote``.

    Returns:
        Una población de ``Individuo``, mismo orden.
    """
    pendientes = [i for i, elemento in enumerate(elementos) if not isinstance(elemento, Individuo)]
    valores = evaluar_lote([elementos[i] for i in pendientes], pool)
    resultado: List[Individuo] = list(elementos)  # type: ignore[arg-type]
    for i, valor in zip(pendientes, valores):
        resultado[i] = Individuo(elementos[i], valor)  # type: ignore[arg-type]
    return resultado
# ------------------------------------------------------------------------------


def aptitud_de(ind: Individuo) -> int:
    """Leer una aptitud ahora es una búsqueda. Ningún sitio de llamada tuvo que cambiar."""
    return ind.aptitud


def volcar_poblacion(poblacion: List[Individuo], ruta: Path) -> None:
    """Escribe solo los genes. Esa es la totalidad de la instantánea del libro."""
    ruta.parent.mkdir(exist_ok=True)
    with ruta.open("w") as handle:
        json.dump([ind.lista_genes for ind in poblacion], handle)


def restaurar_poblacion(ruta: Path) -> List[Individuo]:
    """Reconstruye los individuos. Cada llamada al constructor es una evaluación."""
    with ruta.open() as handle:
        return [Individuo(lista_genes) for lista_genes in json.load(handle)]


def volcar_cache(ruta: Path) -> None:
    """Escribe la tabla de aptitud también. JSON no tiene tuplas, así que las claves van como listas."""
    ruta.parent.mkdir(exist_ok=True)
    with ruta.open("w") as handle:
        json.dump([[list(clave), valor] for clave, valor in CACHE.items()], handle)


def restaurar_cache(ruta: Path) -> None:
    """Devuelve la tabla antes de que se reconstruya nada, o no se usará."""
    with ruta.open() as handle:
        for clave, valor in json.load(handle):
            CACHE[tuple(clave)] = valor


def crear_genoma_aleatorio() -> List[int]:
    return [random.choice([0, 1]) for _ in range(LONGITUD_GENOMA)]


def seleccion_rango_con_elite(poblacion: List[Individuo]) -> List[Individuo]:
    """Selección por rango, élites primero. El ordenamiento solo cuesta una evaluación cada uno."""
    ordenados = sorted(poblacion, key=aptitud_de, reverse=True)
    paso = 1 / len(ordenados)
    rangos = [1 - i * paso for i in range(len(ordenados))]
    total = sum(rangos)
    seleccionados = ordenados[:TAMANO_ELITE]
    for _ in range(len(ordenados) - TAMANO_ELITE):
        corte = random.random() * total
        acumulado = 0.0
        for i, rango in enumerate(rangos):
            acumulado += rango
            if acumulado > corte:
                seleccionados.append(ordenados[i])
                break
    return seleccionados


def cruza_n_puntos(p1: List[int], p2: List[int], n: int) -> Tuple[List[int], List[int]]:
    cortes = sorted(random.sample(range(1, len(p1) - 1), n) + [0, len(p1)])
    c1, c2 = list(p1), list(p2)
    for i in range(1, n + 1, 2):
        c1[cortes[i]:cortes[i + 1]] = p2[cortes[i]:cortes[i + 1]]
        c2[cortes[i]:cortes[i + 1]] = p1[cortes[i]:cortes[i + 1]]
    return c1, c2


def mutacion_volteo_bit(genoma: List[int]) -> List[int]:
    mutante = list(genoma)
    pos = random.randint(0, len(genoma) - 1)
    mutante[pos] = 1 - mutante[pos]
    return mutante


# --- NUEVO (4) operacion_cruza() ----------------------------------------------
def operacion_cruza(poblacion: List[Individuo]) -> List["Criado"]:
    """Los hijos salen de aquí como listas de genes. Aún no se evalúa nada."""
    descendencia: List[Criado] = []
    for ind1, ind2 in zip(poblacion[::2], poblacion[1::2]):
        if random.random() < PROBABILIDAD_CRUZA:
            g1, g2 = cruza_n_puntos(ind1.lista_genes, ind2.lista_genes, PUNTOS_CRUZA)
            descendencia.extend([g1, g2])
        else:
            descendencia.extend([ind1, ind2])
    return descendencia
# ------------------------------------------------------------------------------


# --- NUEVO (5) operacion_mutacion() -------------------------------------------
def operacion_mutacion(poblacion: List["Criado"]) -> List["Criado"]:
    """Un mutante es una lista de genes; lo que no se toca sigue siendo el objeto que era."""
    descendencia: List[Criado] = []
    for elemento in poblacion:
        genoma = elemento.lista_genes if isinstance(elemento, Individuo) else elemento
        if random.random() < PROBABILIDAD_MUTACION:
            descendencia.append(mutacion_volteo_bit(genoma))
        else:
            descendencia.append(elemento)
    return descendencia
# ------------------------------------------------------------------------------


def estadisticas_generacion(poblacion: List[Individuo], mejor: Individuo) -> Tuple[Individuo, float]:
    """Contabilidad. Cuenta cuántas veces tiene que leer un valor de aptitud."""
    campeon = max(poblacion, key=aptitud_de)
    if aptitud_de(mejor) < aptitud_de(campeon):
        mejor = campeon
    media = sum(aptitud_de(ind) for ind in poblacion) / len(poblacion)
    return mejor, media


# --- NUEVO (6) ejecutar() -----------------------------------------------------
def ejecutar(pool: Optional[Pool] = None) -> Tuple[Individuo, List[Dict[str, int]]]:
    """El mismo algoritmo, con una evaluación por lote por generación."""
    random.seed(SEMILLA)
    poblacion = materializar(
        [crear_genoma_aleatorio() for _ in range(TAMANO_POBLACION)], pool)
    mejor = poblacion[0]
    historial: List[Dict[str, int]] = []
    for gen in range(1, MAX_GENERACIONES + 1):
        c0, b0 = COSTO["llamadas"], COSTO["construidos"]
        seleccionados = seleccion_rango_con_elite(poblacion)
        c1 = COSTO["llamadas"]
        poblacion = materializar(
            operacion_mutacion(operacion_cruza(seleccionados)), pool)
        c2 = COSTO["llamadas"]
        mejor, media = estadisticas_generacion(poblacion, mejor)
        mejor_aptitud = aptitud_de(mejor)
        historial.append({
            "gen": gen,
            "seleccion": c1 - c0,
            "reproduccion": c2 - c1,
            "estadisticas": COSTO["llamadas"] - c2,
            "construidos": COSTO["construidos"] - b0,
            "mejor": mejor_aptitud,
            "media": round(media, 1),
        })
    return mejor, historial
# ------------------------------------------------------------------------------


# --- NUEVO (7) el_reporte -----------------------------------------------------
def aptitud_pesada(genoma: List[int]) -> int:
    """El mismo objetivo, REPETICIONES_PESADAS veces. Misma respuesta, más segundos."""
    valor = 0
    for _ in range(REPETICIONES_PESADAS):
        valor = aptitud_turnos(genoma)
    return valor
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    # Windows spawn reimporta este archivo; un Pool al importar se bloquea.
    mejor_individuo, historial = ejecutar()

    llamadas_secuencial = COSTO["llamadas"]
    trabajo_secuencial = COSTO["trabajo"]
    aciertos_secuencial = COSTO["aciertos"]
    mejor_secuencial = [fila["mejor"] for fila in historial]

    print("Lección 13, paso 5 - evaluando una generación en un pool de procesos")
    print("=============================================================")
    print(f"Esta máquina reporta {os.cpu_count()} CPUs; el pool a continuación usa {TRABAJADORES} "
          "procesos trabajadores.\n")
    PASO_3_LLAMADAS = 409
    print(f"Ejecución secuencial: {llamadas_secuencial} evaluaciones, {aciertos_secuencial} aciertos en caché,")
    print(f"                      {trabajo_secuencial} unidades de trabajo, mejor aptitud "
          f"{aptitud_de(mejor_individuo)}.")

    print(f"\nEl paso 3 corrió esta misma búsqueda por {PASO_3_LLAMADAS} evaluaciones. Agrupar la redujo")
    print(f"a {llamadas_secuencial}, un ahorro de {100 * (PASO_3_LLAMADAS - llamadas_secuencial) / PASO_3_LLAMADAS:.1f}%, y no se inició ni un solo proceso para lograrlo: un "
          "hijo")
    print("que se cruza y luego se muta solía evaluarse dos veces, y el "
          "primer")
    print("valor se descartaba sin leer. Evaluar al final de la reproducción en lugar de "
          "en")
    print("medio elimina eso, y es la misma idea que en el paso 2 - no "
          "pagues")
    print("por un número hasta que algo realmente lo lea.")

    CACHE.clear()
    COSTO.update(llamadas=0, trabajo=0, construidos=0, aciertos=0)
    with Pool(processes=TRABAJADORES) as pool:
        mejor_agrupado, historial_agrupado = ejecutar(pool)
    serie_mejor_agrupado = [fila["mejor"] for fila in historial_agrupado]

    print(f"\nEjecución con pool:   {COSTO['llamadas']} evaluaciones, {COSTO['aciertos']} aciertos en caché,")
    print(f"                      {COSTO['trabajo']} unidades de trabajo, mejor aptitud "
          f"{aptitud_de(mejor_agrupado)}.")

    misma_respuesta = aptitud_de(mejor_agrupado) == aptitud_de(mejor_individuo)
    misma_ruta = serie_mejor_agrupado == mejor_secuencial
    print(f"\nMisma respuesta final: {misma_respuesta}. Mismo mejor-hasta-ahora en cada una de las "
          f"{len(historial)}")
    print(f"generaciones: {misma_ruta}. `pool.map` preserva el orden, así que la ruta con pool "
          "recibe")
    print("exactamente la lista que la ruta secuencial habría recibido. El paralelismo aquí "
          "es un")
    print("detalle de implementación de la aritmética, no un cambio a la búsqueda.")

    print(f"\nY sin embargo el proceso padre contó {COSTO['llamadas']} evaluaciones y "
          f"{COSTO['trabajo']} unidades de trabajo para una ejecución")
    print("que claramente hizo el trabajo. Los contadores, la caché y cualquier otro global "
          "viven")
    print("en la memoria propia de los trabajadores: un proceso bifurcado obtiene una copia, y nada de lo que "
          "escribe")
    print("regresa. Ese es el precio real de un pool, y el paso 6 trata sobre pagarlo")
    print("adecuadamente.")

    muestra = [crear_genoma_aleatorio() for _ in range(512)]
    tiempos = {}
    for etiqueta, funcion in (("barata", aptitud_en_cache), ("pesada", aptitud_pesada)):
        CACHE.clear()
        inicio = time.perf_counter()
        [funcion(genoma) for genoma in muestra]
        segundos_secuencial = time.perf_counter() - inicio
        CACHE.clear()
        with Pool(processes=TRABAJADORES) as pool:
            inicio = time.perf_counter()
            pool.map(funcion, muestra)
            segundos_agrupado = time.perf_counter() - inicio
        tiempos[etiqueta] = (segundos_secuencial, segundos_agrupado)

    print(f"\nEvaluando los mismos {len(muestra)} genomas, medidos con un reloj:\n")
    print("  aptitud                        | secuencial | con pool  | ratio")
    print("  -------------------------------+------------+-----------+------")
    for etiqueta, (segundos_secuencial, segundos_agrupado) in tiempos.items():
        peso = 1 if etiqueta == "barata" else REPETICIONES_PESADAS
        print(f"  {etiqueta:<6} aptitud ({peso:2d} x 210 unids) | {segundos_secuencial:9.3f}s |"
              f" {segundos_agrupado:8.3f}s | {segundos_secuencial / segundos_agrupado:5.2f}")

    print("\nLEE ESA TABLA CON CUIDADO. Esos segundos miden la máquina que ejecutó este")
    print("script, esta vez - sus núcleos, su carga, su Python. No son una "
          "propiedad del")
    print("algoritmo genético y no se reproducirán en otro lugar. Todos los demás "
          "números")
    print("en esta lección sí lo harán.")
    ratio_barato = tiempos["barata"][0] / tiempos["barata"][1]
    ratio_pesado = tiempos["pesada"][0] / tiempos["pesada"][1]
    print(f"\nEn esta ejecución el mismo pool anotó {ratio_barato:.2f} en la aptitud barata y "
          f"{ratio_pesado:.2f} en una aptitud")
    print(f"{REPETICIONES_PESADAS} veces más pesada, sobre datos idénticos.")
    if ratio_pesado > ratio_barato:
        print("La aptitud más pesada le sacó más provecho al pool, que es la dirección que se")
        print("espera: enviar un genoma a otro proceso cuesta lo mismo en ambos casos, "
              "así que")
        print("solo cambia el trabajo que espera en el otro extremo.")
    else:
        print("El orden no salió de esa manera esta vez, que es como se ve una "
              "medición")
        print("de unas pocas centésimas de segundo en una máquina compartida. Ese es "
              "el")
        print("punto de la advertencia anterior, no una excepción a ella.")
    print("Vale la pena tener un pool cuando una evaluación es lo suficientemente cara como para eclipsar "
          "el")
    print("costo de enviarla, y la única forma de saberlo es medir, en la "
          "máquina")
    print("que ejecutará el trabajo.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(1, len(historial) + 1), mejor_secuencial, "o-", label="secuencial")
    axes[0].plot(range(1, len(historial) + 1), serie_mejor_agrupado, "x--", label="con pool")
    axes[0].set_title("Dos rutas, una búsqueda")
    axes[0].set_xlabel("generación")
    axes[0].set_ylabel("mejor aptitud")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.5)
    etiquetas = ["barata", "pesada"]
    axes[1].bar([i - 0.2 for i in range(2)], [tiempos[l][0] for l in etiquetas],
                width=0.4, label="secuencial", color="tab:orange")
    axes[1].bar([i + 0.2 for i in range(2)], [tiempos[l][1] for l in etiquetas],
                width=0.4, label="con pool", color="tab:blue")
    axes[1].set_xticks(range(2))
    axes[1].set_xticklabels(["barata (1x)", f"pesada ({REPETICIONES_PESADAS}x)"])
    axes[1].set_ylabel("segundos en ESTA máquina")
    axes[1].set_title("Dependiente de la máquina, y mostrado como tal")
    axes[1].legend()
    fig.suptitle("Paso 5: un pool mueve el trabajo; no elimina ninguno")
    fig.tight_layout()
    FIGURAS.mkdir(exist_ok=True)
    fig.savefig(FIGURAS / "turnos_05_evaluacion_paralela.png", dpi=130)
    plt.close(fig)
    print("\nFigura guardada en figuras/turnos_05_evaluacion_paralela.png")
    # ------------------------------------------------------------------------------

