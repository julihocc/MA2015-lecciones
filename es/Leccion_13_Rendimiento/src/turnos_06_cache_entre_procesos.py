"""
Lección 13 - Paso 6: Llevando la caché y el medidor de vuelta a través de la frontera
==============================================================================
NUEVO EN ESTE PASO: el padre mantiene la caché, y los trabajadores reportan su costo.

El paso 5 terminó con una ejecución agrupada cuyos contadores leían cero evaluaciones y cero
unidades de trabajo, para una ejecución que claramente había hecho ambas cosas. Los procesos trabajadores tienen
memoria separada tanto con ``spawn`` de Windows como con ``fork`` (copia al escribir) de Unix:
la caché y los contadores de cada proceso son privados y desaparecen al cerrar el pool.

Ese no es solo un problema de medición. Cada trabajador construye una caché privada, el
mismo genoma repetido es respondido una vez por trabajador que casualmente lo recibe, y
ninguna de esas respuestas sobrevive al pool - por lo que la caché del paso 3 dejó de
funcionar silenciosamente en el momento en que se introdujo el pool.

Ambos problemas tienen una solución, y es una regla que vale la pena recordar más allá de esta
lección: *el padre es dueño del estado, los trabajadores son dueños solo de la aritmética.* Busca
el genoma aquí, despacha solo los fallos, y haz que cada trabajador devuelva
lo que costó su llamada para que el medidor pueda sumarse en casa.

CAMBIOS RESPECTO A turnos_05_evaluacion_paralela.py
Introdúcelos en este orden:
    1. evaluar_con_costo()  un trabajador devuelve la aptitud Y el trabajo que tomó
    2. evaluar_lote()       los aciertos se responden aquí; solo los fallos se despachan
                            (aptitud_en_cache() del paso 3 se absorbe en ella)
    3. el_reporte           la ejecución agrupada y la ejecución secuencial, lado a lado

Ejecútalo:  python turnos_06_cache_entre_procesos.py

La caché del padre no viaja al pool: hay que mandarla. El script mide
cuánto cuesta reconstruirla frente a mandarla.
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


# --- NUEVO (1) evaluar_con_costo() --------------------------------------------
def evaluar_con_costo(genoma: List[int]) -> Tuple[int, int]:
    """Evalúa un genoma y reporta lo que costó.

    Esto es lo que se ejecuta en un trabajador. Lee la propia copia de COSTO del trabajador para
    medirse a sí mismo, luego devuelve el delta con la aptitud: un trabajador no puede
    actualizar el medidor del padre, pero puede decirle al padre qué sumar.
    """
    antes = COSTO["trabajo"]
    valor = aptitud_turnos(genoma)
    return valor, COSTO["trabajo"] - antes
# ------------------------------------------------------------------------------


# --- NUEVO (2) evaluar_lote() -------------------------------------------------
def evaluar_lote(genomas: Sequence[List[int]], pool: Optional[Pool] = None) -> List[int]:
    """Responde desde la caché del padre; despacha solo lo que es genuinamente nuevo.

    El orden se preserva por construcción: cada genoma mantiene su propio espacio, y
    los fallos se escriben de nuevo en los espacios de donde vinieron.
    """
    valores: List[Optional[int]] = []
    fallos: List[int] = []
    for i, genoma in enumerate(genomas):
        clave = tuple(genoma)
        if clave in CACHE:
            COSTO["aciertos"] += 1
            valores.append(CACHE[clave])
        else:
            valores.append(None)
            fallos.append(i)

    if fallos:
        pendientes = [genomas[i] for i in fallos]
        if pool is None:
            # Mismo proceso: los contadores ya se han incrementado en su lugar.
            resultados = [evaluar_con_costo(genoma) for genoma in pendientes]
        else:
            resultados = list(pool.map(evaluar_con_costo, pendientes))
            # Diferentes procesos: sus contadores murieron con ellos, así que suma los
            # totales que reportaron a los nuestros.
            COSTO["llamadas"] += len(resultados)
            COSTO["trabajo"] += sum(trabajo for _, trabajo in resultados)
        for i, (valor, _) in zip(fallos, resultados):
            CACHE[tuple(genomas[i])] = valor
            valores[i] = valor
    return [valor for valor in valores if valor is not None]
# ------------------------------------------------------------------------------


Criado = Union[Individuo, List[int]]


def materializar(elementos: List[Criado], pool: Optional[Pool] = None) -> List[Individuo]:
    """Convierte una generación de sobrevivientes y nuevos genomas en individuos.

    Los sobrevivientes pasan intactos y no cuestan nada - esa es la sección 13.1
    todavía haciendo su trabajo. Solo los genomas en bruto se envían para ser evaluados.
    """
    pendientes = [i for i, elemento in enumerate(elementos) if not isinstance(elemento, Individuo)]
    valores = evaluar_lote([elementos[i] for i in pendientes], pool)
    resultado: List[Individuo] = list(elementos)  # type: ignore[arg-type]
    for i, valor in zip(pendientes, valores):
        resultado[i] = Individuo(elementos[i], valor)  # type: ignore[arg-type]
    return resultado


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


def estadisticas_generacion(poblacion: List[Individuo], mejor: Individuo) -> Tuple[Individuo, float]:
    """Contabilidad. Cuenta cuántas veces tiene que leer un valor de aptitud."""
    campeon = max(poblacion, key=aptitud_de)
    if aptitud_de(mejor) < aptitud_de(campeon):
        mejor = campeon
    media = sum(aptitud_de(ind) for ind in poblacion) / len(poblacion)
    return mejor, media


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


if __name__ == "__main__":
    # Windows spawn reimporta este archivo; un Pool al importar se bloquea.
    mejor_individuo, historial = ejecutar()


    # --- NUEVO (3) el_reporte -----------------------------------------------------
    PASO_5_AGRUPADO = (0, 0, 0)

    secuencial = (COSTO["llamadas"], COSTO["aciertos"], COSTO["trabajo"])
    mejor_secuencial = [fila["mejor"] for fila in historial]

    print("Lección 13, paso 6 - la caché y el medidor cruzan la frontera")
    print("==============================================================")
    print(f"Esta máquina reporta {os.cpu_count()} CPUs; el pool a continuación usa {TRABAJADORES} "
          "procesos trabajadores.\n")

    CACHE.clear()
    COSTO.update(llamadas=0, trabajo=0, construidos=0, aciertos=0)
    with Pool(processes=TRABAJADORES) as pool:
        mejor_agrupado, historial_agrupado = ejecutar(pool)
    agrupado = (COSTO["llamadas"], COSTO["aciertos"], COSTO["trabajo"])

    print("  ejecución                  | evaluaciones | aciertos caché | unids trabajo")
    print("  ---------------------------+--------------+----------------+--------------")
    print(f"  secuencial                 | {secuencial[0]:12d} | {secuencial[1]:14d} |"
          f" {secuencial[2]:13d}")
    print(f"  con pool, paso 5           | {PASO_5_AGRUPADO[0]:12d} | {PASO_5_AGRUPADO[1]:14d} |"
          f" {PASO_5_AGRUPADO[2]:13d}")
    print(f"  con pool, paso 6           | {agrupado[0]:12d} | {agrupado[1]:14d} |"
          f" {agrupado[2]:13d}")

    if agrupado == secuencial:
        print(f"\nLa ejecución agrupada y la ejecución secuencial ahora coinciden en los tres números.")
        print("El trabajo se hizo en otros procesos y la cuenta del mismo llegó a casa.")
    else:
        print(f"\nLas dos ejecuciones no coinciden: {secuencial} frente a {agrupado}. Algo se "
              "sigue")
        print("contando en un proceso que luego descarta la cuenta.")

    print(f"\nLa ejecución con pool del paso 5 reportó {PASO_5_AGRUPADO[0]} evaluaciones y "
          f"{PASO_5_AGRUPADO[2]} unidades de trabajo para exactamente esta")
    print("búsqueda. Nada sobre la búsqueda cambió entre entonces y ahora; lo que "
          "cambió es")
    print(f"que alguien está llevando los libros. Y los {agrupado[1]} aciertos de caché son la "
          "otra mitad de")
    print("la reparación: en el paso 5 cada trabajador mantenía una caché privada que moría con la")
    print("generación, por lo que esas repeticiones se pagaban de nuevo. Aquí la búsqueda ocurre "
          "en")
    print("el padre, antes de que nada se despache, y un acierto no cuesta ningún mensaje en "
          "absoluto.")

    misma_respuesta = aptitud_de(mejor_agrupado) == aptitud_de(mejor_individuo)
    misma_ruta = [fila["mejor"] for fila in historial_agrupado] == mejor_secuencial
    print(f"\nMejor aptitud: {aptitud_de(mejor_individuo)} secuencial, "
          f"{aptitud_de(mejor_agrupado)} con pool. Respuesta idéntica: {misma_respuesta};")
    print(f"idéntico mejor-hasta-ahora en las {len(historial)} generaciones: {misma_ruta}.")
    print("\nLa regla de la que realmente trata este paso: el padre es dueño del estado, los "
          "trabajadores")
    print("poseen solo la aritmética. Cada global que un trabajador toca es una copia, y "
          "cada")
    print("copia se desecha - lo cual es un error cuando es una caché, y una mentira cuando "
          "es")
    print("un contador.")

    etiquetas = ["secuencial", "con pool\n(paso 5)", "con pool\n(paso 6)"]
    valores_trabajo = [secuencial[2], PASO_5_AGRUPADO[2], agrupado[2]]
    valores_aciertos = [secuencial[1], PASO_5_AGRUPADO[1], agrupado[1]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(etiquetas, valores_trabajo, color=["tab:orange", "tab:red", "tab:green"])
    axes[0].set_title("Unidades de trabajo que el padre puede contabilizar")
    axes[0].set_ylabel("unidades de trabajo")
    for i, v in enumerate(valores_trabajo):
        axes[0].text(i, v, str(v), ha="center", va="bottom")
    axes[1].bar(etiquetas, valores_aciertos, color=["tab:orange", "tab:red", "tab:green"])
    axes[1].set_title("Aciertos de caché que sobrevivieron a la ejecución")
    axes[1].set_ylabel("aciertos")
    for i, v in enumerate(valores_aciertos):
        axes[1].text(i, v, str(v), ha="center", va="bottom")
    fig.suptitle("Paso 6: el padre es dueño del estado, los trabajadores de la aritmética")
    fig.tight_layout()
    FIGURAS.mkdir(exist_ok=True)
    fig.savefig(FIGURAS / "turnos_06_cache_entre_procesos.png", dpi=130)
    plt.close(fig)
    print("\nFigura guardada en figuras/turnos_06_cache_entre_procesos.png")
    # ------------------------------------------------------------------------------

