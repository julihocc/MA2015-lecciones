"""
Lección 13 - Paso 3: Una caché para genomas que regresan
=======================================================
NUEVO EN ESTE PASO: un diccionario de genoma a aptitud, y un contador de aciertos.

El paso 2 puso la factura en una evaluación por individuo construido. Ese es un piso
solo para genomas *nuevos*. Un algoritmo genético no produce genomas nuevos todo
el tiempo: las élites se copian hacia adelante, la selección elige a los mismos padres dos veces,
la cruza entre dos copias de una asignación devuelve esa asignación sin cambios, y
una mutación puede deshacer una anterior. Cada uno de esos es un genoma que el programa
ya ha cotizado.

La sección 13.2 del libro mantiene un diccionario de genoma a aptitud. Un acierto es una
evaluación que nunca ocurre, por lo que un acierto cuesta 0 unidades de trabajo en lugar de 210.
La caché no cambia nada sobre la búsqueda: devuelve exactamente el valor que la
función de aptitud habría devuelto, por lo que la respuesta debe salir idéntica a la del
paso 2, y el script verifica eso en lugar de asumirlo.

El precio es la memoria. La caché contiene una entrada por cada genoma distinto jamás visto,
y nada en este diseño desaloja ninguno.

CAMBIOS RESPECTO A turnos_02_evaluar_una_vez.py
Introdúcelos en este orden:
    1. CACHE             un diccionario de genoma a aptitud, y un contador de aciertos
    2. aptitud_en_cache() buscar el genoma antes de pagar por él
    3. Individuo         el constructor llama a aptitud_en_cache() - una línea cambiada
    4. el_reporte        contar aciertos, y verificar que la caché no alteró la respuesta
    5. escaneo_tasa_aciertos() ¿es una pequeña tasa de aciertos culpa de la caché, o de la búsqueda?

Ejecútalo:  python turnos_03_la_cache.py

19 aciertos en 428 construcciones, 409 genomas distintos, 4.4% menos
trabajo, respuesta −87. El escaneo: 14.4% de aciertos con mutación 0.10 y
3.2% con 0.75 — la tasa es de la búsqueda, no de la caché.
"""
import random
from pathlib import Path
from typing import Dict, List, Tuple

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


# --- NUEVO (1) CACHE ----------------------------------------------------------
# Una entrada por genoma distinto jamás construido. Un acierto es una evaluación que nunca
# sucede; el contador es lo que permite que el informe de abajo indique un ahorro en lugar de
# reclamar uno.
CACHE: Dict[Tuple[int, ...], int] = {}
COSTO["aciertos"] = 0
# ------------------------------------------------------------------------------


def aptitud_turnos(genoma: List[int]) -> int:
    """El objetivo, a MAXIMIZAR. Una asignación perfecta puntúa 0."""
    COSTO["llamadas"] += 1
    return -(desviacion_turno(genoma) + PESO_DESCANSO * violaciones_descanso(genoma))


# --- NUEVO (2) aptitud_en_cache() ---------------------------------------------
def aptitud_en_cache(genoma: List[int]) -> int:
    """Responde desde la tabla si este genoma exacto ya se cotizó.

    La clave tiene que ser hasheable y *todo* el genoma: un bit de
    diferencia es otra asignación.

    Args:
        genoma: 105 bits.

    Returns:
        La aptitud, de la caché o recién calculada. Un acierto cuesta 0
        unidades de trabajo en lugar de 210.

    Example:
        19 aciertos en 428 construcciones a mutación 0.50.
    """
    clave = tuple(genoma)
    if clave in CACHE:
        COSTO["aciertos"] += 1
        return CACHE[clave]
    valor = aptitud_turnos(genoma)
    CACHE[clave] = valor
    return valor
# ------------------------------------------------------------------------------


class Individuo:
    """Una asignación candidata: 105 bits, primero empleados, evaluado exactamente una vez.

    Los genes nunca cambian después de la construcción, por lo que la aptitud tampoco puede
    cambiar. Calcularla aquí y almacenarla es la totalidad de la sección 13.1:
    tres líneas, sin aproximación, sin ceder nada.
    """

    def __init__(self, lista_genes: List[int]) -> None:
        self.lista_genes = list(lista_genes)
        self.aptitud = aptitud_en_cache(self.lista_genes)   # --- CAMBIADO --- era aptitud_turnos()
        COSTO["construidos"] += 1


def aptitud_de(ind: Individuo) -> int:
    """Leer una aptitud ahora es una búsqueda. Ningún sitio de llamada tuvo que cambiar."""
    return ind.aptitud


def crear_aleatorio() -> Individuo:
    return Individuo([random.choice([0, 1]) for _ in range(LONGITUD_GENOMA)])


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


def operacion_cruza(poblacion: List[Individuo]) -> List[Individuo]:
    descendencia: List[Individuo] = []
    for ind1, ind2 in zip(poblacion[::2], poblacion[1::2]):
        if random.random() < PROBABILIDAD_CRUZA:
            g1, g2 = cruza_n_puntos(ind1.lista_genes, ind2.lista_genes, PUNTOS_CRUZA)
            descendencia.extend([Individuo(g1), Individuo(g2)])
        else:
            descendencia.extend([ind1, ind2])
    return descendencia


def operacion_mutacion(poblacion: List[Individuo]) -> List[Individuo]:
    descendencia: List[Individuo] = []
    for ind in poblacion:
        if random.random() < PROBABILIDAD_MUTACION:
            descendencia.append(Individuo(mutacion_volteo_bit(ind.lista_genes)))
        else:
            descendencia.append(ind)
    return descendencia


def estadisticas_generacion(poblacion: List[Individuo], mejor: Individuo) -> Tuple[Individuo, float]:
    """Contabilidad. Cuenta cuántas veces tiene que leer un valor de aptitud."""
    campeon = max(poblacion, key=aptitud_de)
    if aptitud_de(mejor) < aptitud_de(campeon):
        mejor = campeon
    media = sum(aptitud_de(ind) for ind in poblacion) / len(poblacion)
    return mejor, media


def ejecutar() -> Tuple[Individuo, List[Dict[str, int]]]:
    random.seed(SEMILLA)
    poblacion = [crear_aleatorio() for _ in range(TAMANO_POBLACION)]
    mejor = poblacion[0]
    historial: List[Dict[str, int]] = []
    for gen in range(1, MAX_GENERACIONES + 1):
        c0, b0 = COSTO["llamadas"], COSTO["construidos"]
        seleccionados = seleccion_rango_con_elite(poblacion)
        c1 = COSTO["llamadas"]
        poblacion = operacion_mutacion(operacion_cruza(seleccionados))
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


mejor_individuo, historial = ejecutar()


# --- NUEVO (4) el_reporte -----------------------------------------------------
PASO_2_LLAMADAS = 428
PASO_2_TRABAJO = 89880
PASO_2_MEJOR = -87
PASO_1_TRABAJO_REF = 195300
TRABAJO_POR_LLAMADA = 2 * EMPLEADOS * TURNOS

print("Lección 13, paso 3 - una caché para genomas que regresan")
print("======================================================")
print("REPROD todavía construye los mismos individuos; la columna ahora cuenta solo los")
print("que tuvo que evaluar.\n")
print(" gen | individuos nuevos | evaluados en REPROD| mejor|   media")
print("-----+-------------------+--------------------+------+-------")
for fila in historial:
    print(f" {fila['gen']:3d} | {fila['construidos']:17d} | {fila['reproduccion']:18d} |"
          f" {fila['mejor']:4d} | {fila['media']:6.1f}")

construidos = COSTO["construidos"]
llamadas = COSTO["llamadas"]
aciertos = COSTO["aciertos"]
print(f"\nSe construyeron {construidos} individuos. {llamadas} de ellos necesitaron una evaluación y "
      f"{aciertos} fueron")
print(f"respondidos desde la caché, que contiene {len(CACHE)} genomas distintos.")
print("\n                     paso 2     paso 3     ahorrado")
print(f"  llamadas aptitud {PASO_2_LLAMADAS:8d}   {llamadas:8d}   {100 * (PASO_2_LLAMADAS - llamadas) / PASO_2_LLAMADAS:5.1f}%")
print(f"  unidades trabajo {PASO_2_TRABAJO:8d}   {COSTO['trabajo']:8d}   {100 * (PASO_2_TRABAJO - COSTO['trabajo']) / PASO_2_TRABAJO:5.1f}%")
print(f"\nCada acierto ahorró {TRABAJO_POR_LLAMADA} unidades de trabajo, por lo que la caché ahorró "
      f"{aciertos * TRABAJO_POR_LLAMADA} de ellas en total.")
if aciertos * TRABAJO_POR_LLAMADA == PASO_2_TRABAJO - COSTO["trabajo"]:
    print("Eso explica toda la diferencia: nada más en la ejecución cambió.")

mejor_aptitud = aptitud_de(mejor_individuo)
print(f"\nMejor asignación encontrada: aptitud {mejor_aptitud}.")
if mejor_aptitud == PASO_2_MEJOR:
    print(f"El paso 2 terminó en {PASO_2_MEJOR} también. Una caché devuelve el valor que la "
          "función de aptitud")
    print("habría devuelto, por lo que una respuesta idéntica es el resultado esperado - y "
          "una")
    print("diferente significaría que la clave era incorrecta, no que la caché fuera inteligente.")
else:
    print(f"El paso 2 terminó en {PASO_2_MEJOR}. La respuesta se movió, por lo que la clave de la caché es "
          "incorrecta.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(["paso 1", "paso 2", "paso 3"],
            [PASO_1_TRABAJO_REF, PASO_2_TRABAJO, COSTO["trabajo"]],
            color=["tab:red", "tab:orange", "tab:green"])
axes[0].set_title("Unidades de trabajo para la misma ejecución")
axes[0].set_ylabel("unidades de trabajo")
for i, v in enumerate([PASO_1_TRABAJO_REF, PASO_2_TRABAJO, COSTO["trabajo"]]):
    axes[0].text(i, v, str(v), ha="center", va="bottom")
axes[1].plot([r["gen"] for r in historial], [r["construidos"] for r in historial],
             "ko--", label="individuos construidos")
axes[1].plot([r["gen"] for r in historial], [r["reproduccion"] for r in historial],
             "o-", color="tab:green", label="realmente evaluados")
axes[1].set_title("La brecha entre las dos líneas es la caché")
axes[1].set_xlabel("generación")
axes[1].set_ylabel("cuenta")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.5)
fig.suptitle("Paso 3: un acierto es una evaluación que nunca sucede")
fig.tight_layout()
FIGURAS.mkdir(exist_ok=True)
fig.savefig(FIGURAS / "turnos_03_la_cache.png", dpi=130)
plt.close(fig)
print("\nFigura guardada en figuras/turnos_03_la_cache.png")
# ------------------------------------------------------------------------------


# --- NUEVO (5) escaneo_tasa_aciertos() ----------------------------------------
def escaneo_tasa_aciertos(probabilidades: List[float]) -> List[Tuple[float, int, int]]:
    """Vuelve a ejecutar todo el AG a varias tasas de mutación y cuenta los aciertos en caché.

    19 aciertos en 428 construcciones es un rendimiento pequeño, y la pregunta honesta es si
    eso es un hecho sobre el almacenamiento en caché o un hecho sobre esta búsqueda. Solo una segunda
    medición puede responderla, así que aquí hay una.
    """
    global PROBABILIDAD_MUTACION
    filas = []
    for probabilidad in probabilidades:
        CACHE.clear()
        COSTO.update(llamadas=0, trabajo=0, construidos=0, aciertos=0)
        PROBABILIDAD_MUTACION = probabilidad
        ejecutar()
        filas.append((probabilidad, COSTO["construidos"], COSTO["aciertos"]))
    return filas


print("\nLa misma ejecución, a cuatro probabilidades de mutación. Todo lo demás se mantiene")
print("fijo, incluida la semilla.\n")
print("  p de mutación | individuos const. | aciertos caché | tasa de aciertos")
print("  --------------+-------------------+----------------+-----------------")
escaneo = escaneo_tasa_aciertos([0.1, 0.25, 0.5, 0.75])
for probabilidad, n_construidos, n_aciertos in escaneo:
    print(f"  {probabilidad:13.2f} | {n_construidos:17d} | {n_aciertos:14d} |"
          f" {100 * n_aciertos / n_construidos:15.1f}%")
p_baja, construidos_baja, aciertos_baja = escaneo[0]
p_alta, construidos_alta, aciertos_alta = escaneo[-1]
print(f"\nA p={p_baja} la caché responde al {100 * aciertos_baja / construidos_baja:.1f}% de las construcciones; a "
      f"p={p_alta} responde al {100 * aciertos_alta / construidos_alta:.1f}%.")
print("El diccionario es el mismo diccionario. Lo que cambia es con qué frecuencia la "
      "búsqueda")
print("regresa a un genoma que ya ha visto, y esa es una propiedad de "
      "los")
print("operadores, no de la caché. Una caché vale lo que las repeticiones de la búsqueda "
      "valen")
print("- por lo que tiene que ser medida en el problema que nos ocupa.")
# ------------------------------------------------------------------------------

