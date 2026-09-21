"""
Lección 13 - Paso 7: Una aptitud gruesa, y lo que cuesta ser barato
====================================================================
NUEVO EN ESTE PASO: un objetivo más barato, y un interruptor que elige uno.

Cada optimización hasta ahora ha sido gratuita. La sección 13.1 dejó de recalcular un
número conocido; la caché dejó de recalcular un genoma conocido; el agrupamiento dejó de
evaluar hijos que estaban a punto de ser descartados; el pool movió el trabajo
sin cambiarlo. Ninguna de ellas alteró un solo valor de aptitud, y cada
una se verificó contra la respuesta del paso anterior para demostrarlo.

Esta última técnica no es así. La sección 13.3 del libro evalúa una
versión *gruesa* del problema: más barata, aproximada, y ya no es la cosa
que realmente quieres maximizar. Aquí la regla de descanso de la asignación - la mitad cara
del objetivo, y la mitad que recorre empleados en lugar de turnos -
simplemente se elimina. Una llamada cuesta 105 unidades de trabajo en lugar de 210.

A mitad de precio. La pregunta que este paso existe para hacer es qué estaba
comprando la otra mitad, y la única forma honesta de responderla es tomar la asignación que
produjo la búsqueda barata y puntuarla en el objetivo real. Eso es lo que hace el
reporte, y el número que obtiene es la razón por la que este paso es el último.

CAMBIOS RESPECTO A turnos_06_cache_entre_procesos.py
Introdúcelos en este orden:
    1. aptitud_gruesa()      el objetivo con su mitad cara eliminada
    2. evaluar_con_costo()   el evaluador lee un interruptor: qué objetivo usar
    3. el_reporte            ejecuta ambos, cotiza ambos, y juzga ambos en el real

Ejecútalo:  python turnos_07_aptitud_gruesa.py

La aptitud gruesa es 50% más barata por llamada y es la única técnica
de esta lección que cambia la pregunta, no solo la factura.
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


# --- NUEVO (1) aptitud_gruesa() -----------------------------------------------
def aptitud_gruesa(genoma: List[int]) -> int:
    """La mitad del objetivo: bandas de personal, nada de descanso.

    Quitar ``violaciones_descanso`` reduce a la mitad el costo de una
    llamada — y olvida que un empleado no puede hacer tres turnos
    seguidos. Es la única técnica de la lección que cambia la pregunta.

    Args:
        genoma: 105 bits.

    Returns:
        ``-desviacion_turno(genoma)``. Hay que juzgarla contra el
        objetivo *real*, no contra sí misma.
    """
    COSTO["llamadas"] += 1
    return -desviacion_turno(genoma)
# ------------------------------------------------------------------------------


# --- NUEVO (2) evaluar_con_costo() --------------------------------------------
# Qué objetivo está maximizando realmente la búsqueda. Es un nombre a nivel de módulo
# para que los trabajadores lo hereden cuando el pool se bifurca, y para que haya
# exactamente un lugar donde mirar cuando las respuestas salen extrañas.
OBJETIVO = aptitud_turnos


def _heredar_objetivo(objetivo) -> None:
    """Windows spawn no bifurca; hay que decirle a cada trabajador qué objetivo usar."""
    global OBJETIVO
    OBJETIVO = objetivo


def evaluar_con_costo(genoma: List[int]) -> Tuple[int, int]:
    """Evalúa un genoma con el objetivo actual y reporta lo que costó.

    Args:
        genoma: 105 bits.

    Returns:
        ``(valor, unidades_de_trabajo)`` de esa sola llamada.
    """
    antes = COSTO["trabajo"]
    valor = OBJETIVO(genoma)
    return valor, COSTO["trabajo"] - antes
# ------------------------------------------------------------------------------


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
    llamadas_exacto, trabajo_exacto = COSTO["llamadas"], COSTO["trabajo"]
    aptitud_mejor_exacto = aptitud_de(mejor_individuo)
    genoma_campeon_exacto = list(mejor_individuo.lista_genes)

    print("Lección 13, paso 7 - una aptitud gruesa, y lo que cuesta ser barato")
    print("===================================================================")
    print(f"Objetivo completo: {llamadas_exacto} evaluaciones, {trabajo_exacto} unidades de trabajo, "
          f"{trabajo_exacto // llamadas_exacto} unidades por llamada.")

    OBJETIVO = aptitud_gruesa
    CACHE.clear()
    COSTO.update(llamadas=0, trabajo=0, construidos=0, aciertos=0)
    with Pool(processes=TRABAJADORES, initializer=_heredar_objetivo,
              initargs=(OBJETIVO,)) as pool:
        mejor_grueso, historial_grueso = ejecutar(pool)
    llamadas_grueso, trabajo_grueso = COSTO["llamadas"], COSTO["trabajo"]
    print(f"Objetivo grueso:   {llamadas_grueso} evaluaciones, {trabajo_grueso} unidades de trabajo, "
          f"{trabajo_grueso // llamadas_grueso} unidades por llamada.")
    print(f"\nEl costo por llamada cayó en un {100 * (1 - (trabajo_grueso / llamadas_grueso) / (trabajo_exacto / llamadas_exacto)):.0f}%, exactamente como se diseñó: la llamada gruesa "
          "recorre la")
    print("asignación una vez en lugar de dos veces.")

    # La única comparación que significa algo es en el objetivo que es real.
    COSTO.update(llamadas=0, trabajo=0)
    campeon_grueso_exacto = aptitud_turnos(mejor_grueso.lista_genes)
    campeon_exacto_exacto = aptitud_turnos(genoma_campeon_exacto)
    violaciones_grueso = violaciones_descanso(mejor_grueso.lista_genes)
    violaciones_exacto = violaciones_descanso(genoma_campeon_exacto)

    print("\n  búsqueda guiada por | su propia punt.| puntuado en el objetivo COMPLETO |"
          " violaciones descanso")
    print("  --------------------+----------------+----------------------------------+"
          "---------------------")
    print(f"  el objetivo completo| {aptitud_de(mejor_individuo):14d} |"
          f" {campeon_exacto_exacto:32d} | {violaciones_exacto:20d}")
    print(f"  el grueso           | {aptitud_de(mejor_grueso):14d} |"
          f" {campeon_grueso_exacto:32d} | {violaciones_grueso:20d}")

    brecha = campeon_exacto_exacto - campeon_grueso_exacto
    ahorro = 100 * (trabajo_exacto - trabajo_grueso) / trabajo_exacto
    print(f"\nLa búsqueda gruesa gastó un {ahorro:.1f}% menos de trabajo y su asignación puntúa "
          f"{campeon_grueso_exacto} en el")
    print(f"objetivo que importa, frente a {campeon_exacto_exacto} para la búsqueda que "
          f"pagó el precio completo -")
    if brecha > 0:
        print(f"una brecha de {brecha}. La asignación gruesa rompe la regla de descanso "
              f"{violaciones_grueso} veces, porque nada")
        print("en el objetivo que estaba optimizando había oído hablar jamás de esa regla. No es "
              "una")
        print("búsqueda peor; es una búsqueda de algo diferente, y el algo diferente "
              "fue")
        print("elegido porque era barato.")
    elif brecha == 0:
        print("ninguna brecha en absoluto. En esta instancia el término eliminado no cambió dónde "
              "la")
        print("búsqueda terminó - lo cual es un resultado sobre esta asignación, no una licencia para")
        print("eliminar términos en general.")
    else:
        print(f"una brecha de {-brecha} a favor de la búsqueda gruesa. Eso vale la pena afirmarlo "
              "claramente:")
        print("el objetivo más barato encontró la mejor asignación aquí, lo que sucede cuando "
              "el")
        print("término eliminado estaba mayormente estorbando al principio.")

    print("\nEsa es la nota con la que termina este curso. Cada optimización antes de esta "
          "fue")
    print("gratuita, y cada una fue verificada contra la respuesta del paso anterior; esta es")
    print("más barata que todas ellas y es la única que tuvo que ser juzgada en lugar "
          "de")
    print("verificada. Una aproximación no es una forma más rápida de obtener la misma "
          "respuesta. Es una")
    print("pregunta diferente, que se hace porque es más barato hacerla, y si fue o no "
          "un")
    print("buen trato es una medición - la impresa arriba - y nunca una")
    print("suposición.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["objetivo completo", "objetivo grueso"], [trabajo_exacto, trabajo_grueso],
                color=["tab:blue", "tab:orange"])
    axes[0].set_title("Unidades de trabajo gastadas en la búsqueda")
    axes[0].set_ylabel("unidades de trabajo")
    for i, v in enumerate([trabajo_exacto, trabajo_grueso]):
        axes[0].text(i, v, str(v), ha="center", va="bottom")
    axes[1].bar(["objetivo completo", "objetivo grueso"],
                [campeon_exacto_exacto, campeon_grueso_exacto],
                color=["tab:blue", "tab:orange"])
    axes[1].set_title("Ambos campeones, puntuados en el objetivo COMPLETO")
    axes[1].set_ylabel("aptitud verdadera (0 es perfecto)")
    for i, v in enumerate([campeon_exacto_exacto, campeon_grueso_exacto]):
        axes[1].text(i, v, str(v), ha="center", va="top")
    fig.suptitle("Paso 7: la única optimización en esta lección que no es gratis")
    fig.tight_layout()
    FIGURAS.mkdir(exist_ok=True)
    fig.savefig(FIGURAS / "turnos_07_aptitud_gruesa.png", dpi=130)
    plt.close(fig)
    print("\nFigura guardada en figuras/turnos_07_aptitud_gruesa.png")
    # ------------------------------------------------------------------------------

