"""
Lección 13 - Paso 4: Instantáneas, y lo que olvida una instantánea
===========================================================
NUEVO EN ESTE PASO: volcar y restaurar una población, y volcar la caché.

Las ejecuciones largas se interrumpen: una laptop se suspende, un trabajo en cola alcanza su tiempo límite, un
estudio de afinación quiere ramificarse de una ejecución que ya costó una hora. La respuesta del libro
(sección 13.4) es escribir los genes de la población en un archivo y leerlos
de nuevo más tarde.

Eso funciona, y son tres líneas. Pero una población no son solo sus genes. Sus
valores de aptitud son la parte cara, y JSON no los transporta: reconstruir
un individuo a partir de una lista de genes ejecuta el constructor, y el constructor
evalúa. Así que un reinicio desde una instantánea simple paga de nuevo por toda la
población antes de que haya hecho algún trabajo nuevo.

La caché del paso 3 es la solución, y es el punto de poner estas dos
técnicas una al lado de la otra: vuelca la caché junto con los genes y la
población restaurada no cuesta nada en absoluto. Este script reinicia la misma
población dos veces, en frío y en caliente, y cuenta ambas.

CAMBIOS RESPECTO A turnos_03_la_cache.py
Introdúcelos en este orden:
    1. INSTANTANEAS       a dónde van los archivos
    2. volcar_poblacion() los genes de una población, una lista cada uno, como JSON
    3. restaurar_poblacion() reconstruye los individuos desde el archivo
    4. volcar_cache()     la caché también es estado, y es la mitad cara
    5. restaurar_cache()  devuelve la tabla antes de reconstruir cualquier cosa
    6. el_reporte         reinicio en frío, reinicio en caliente, y cuenta lo que cuesta cada uno

Ejecútalo:  python turnos_04_instantanea.py

30 asignaciones se restauran por 30 evaluaciones / 6,300 unidades de trabajo
a partir de genes solos, y por 0 / 0 cuando la caché se restaura primero. El
archivo de caché es 14.0× el de los genes.
"""
import json
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

# --- NUEVO (1) INSTANTANEAS ---------------------------------------------------
# Una instantánea es un archivo ordinario, por lo que pertenece a un lugar donde un estudiante pueda abrirlo
# y leer los genes como texto.
INSTANTANEAS = Path(__file__).resolve().parent.parent / "snapshots"
# ------------------------------------------------------------------------------

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


class Individuo:
    """Una asignación candidata: 105 bits, primero empleados, evaluado exactamente una vez.

    Los genes nunca cambian después de la construcción, por lo que la aptitud tampoco puede
    cambiar. Calcularla aquí y almacenarla es la totalidad de la sección 13.1:
    tres líneas, sin aproximación, sin ceder nada.
    """

    def __init__(self, lista_genes: List[int]) -> None:
        self.lista_genes = list(lista_genes)
        self.aptitud = aptitud_en_cache(self.lista_genes)
        COSTO["construidos"] += 1


def aptitud_de(ind: Individuo) -> int:
    """Leer una aptitud ahora es una búsqueda. Ningún sitio de llamada tuvo que cambiar."""
    return ind.aptitud


# --- NUEVO (2) volcar_poblacion() ---------------------------------------------
def volcar_poblacion(poblacion: List[Individuo], ruta: Path) -> None:
    """Escribe solo los genes. Esa es la totalidad de la instantánea del libro."""
    ruta.parent.mkdir(exist_ok=True)
    with ruta.open("w") as handle:
        json.dump([ind.lista_genes for ind in poblacion], handle)
# ------------------------------------------------------------------------------


# --- NUEVO (3) restaurar_poblacion() ------------------------------------------
def restaurar_poblacion(ruta: Path) -> List[Individuo]:
    """Reconstruye los individuos. Cada llamada al constructor es una evaluación."""
    with ruta.open() as handle:
        return [Individuo(lista_genes) for lista_genes in json.load(handle)]
# ------------------------------------------------------------------------------


# --- NUEVO (4) volcar_cache() -------------------------------------------------
def volcar_cache(ruta: Path) -> None:
    """Escribe la tabla de aptitud también. JSON no tiene tuplas, así que las claves van como listas."""
    ruta.parent.mkdir(exist_ok=True)
    with ruta.open("w") as handle:
        json.dump([[list(clave), valor] for clave, valor in CACHE.items()], handle)
# ------------------------------------------------------------------------------


# --- NUEVO (5) restaurar_cache() ----------------------------------------------
def restaurar_cache(ruta: Path) -> None:
    """Devuelve la tabla antes de que se reconstruya nada, o no se usará."""
    with ruta.open() as handle:
        for clave, valor in json.load(handle):
            CACHE[tuple(clave)] = valor
# ------------------------------------------------------------------------------


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


# --- NUEVO (6) el_reporte -----------------------------------------------------
llamadas_ejecucion, trabajo_ejecucion = COSTO["llamadas"], COSTO["trabajo"]
TRABAJO_POR_LLAMADA = 2 * EMPLEADOS * TURNOS

print("Lección 13, paso 4 - instantáneas, y lo que olvida una instantánea")
print("==========================================================")
print(f"La ejecución costó {llamadas_ejecucion} evaluaciones y {trabajo_ejecucion} unidades de trabajo, y terminó "
      f"con aptitud {aptitud_de(mejor_individuo)}.")

# La población a guardar es la que dejó la última generación, y ejecutar()
# devuelve solo el campeón. Así que la ejecución se repite aquí, en línea. Con la
# caché de la primera ejecución aún caliente, la repetición es gratis, y la cuenta abajo
# lo dice.
repeticion_antes = COSTO["llamadas"]
random.seed(SEMILLA)
poblacion = [crear_aleatorio() for _ in range(TAMANO_POBLACION)]
for _ in range(MAX_GENERACIONES):
    poblacion = operacion_mutacion(operacion_cruza(
        seleccion_rango_con_elite(poblacion)))

llamadas_repeticion = COSTO["llamadas"] - repeticion_antes
print(f"Repetir esa ejecución para recuperar su última población costó {llamadas_repeticion} "
      "evaluaciones:")
print("misma semilla, mismos genomas, y la caché ya los tenía todos.")

ruta_genes = INSTANTANEAS / "turnos_poblacion.json"
ruta_cache = INSTANTANEAS / "turnos_cache.json"
volcar_poblacion(poblacion, ruta_genes)
volcar_cache(ruta_cache)
print(f"\nGuardadas {len(poblacion)} asignaciones en snapshots/{ruta_genes.name} "
      f"({ruta_genes.stat().st_size} bytes)")
print(f"Guardados {len(CACHE)} valores de aptitud en caché en snapshots/{ruta_cache.name} "
      f"({ruta_cache.stat().st_size} bytes)")

original = [ind.aptitud for ind in poblacion]

# Reinicio 1: un proceso fresco comenzaría con una caché vacía. Simula eso.
CACHE.clear()
COSTO.update(llamadas=0, trabajo=0, aciertos=0)
frio = restaurar_poblacion(ruta_genes)
llamadas_frio, trabajo_frio = COSTO["llamadas"], COSTO["trabajo"]

# Reinicio 2: mismo archivo, pero la caché se restaura primero.
CACHE.clear()
COSTO.update(llamadas=0, trabajo=0, aciertos=0)
restaurar_cache(ruta_cache)
caliente = restaurar_poblacion(ruta_genes)
llamadas_caliente, trabajo_caliente = COSTO["llamadas"], COSTO["trabajo"]

print("\n  reinicio         | evaluaciones | unids trabajo")
print("  -----------------+--------------+--------------")
print(f"  solo genes       | {llamadas_frio:12d} | {trabajo_frio:13d}")
print(f"  genes + cache    | {llamadas_caliente:12d} | {trabajo_caliente:13d}")

ratio = ruta_cache.stat().st_size / ruta_genes.stat().st_size
print(f"\nEl archivo de caché es {ratio:.1f} veces el tamaño del archivo de genes, y lo que "
      f"compró fue")
print(f"{llamadas_frio - llamadas_caliente} evaluaciones. A {TRABAJO_POR_LLAMADA} unidades de trabajo cada una, es un mal "
      "trato; en una función de")
print("aptitud que toma un minuto por llamada, es excelente. La técnica "
      "es")
print("la misma en cualquier caso - solo cambia el tipo de cambio.")

print(f"\nUna instantánea de {len(poblacion)} asignaciones cuesta {llamadas_frio} evaluaciones para "
      "volver a la vida,")
print("porque JSON almacena genes y el constructor es lo que convierte genes en una")
print(f"aptitud. Restaurar la caché primero reduce eso a {llamadas_caliente}: cada asignación en "
      "el")
print("archivo es un genoma que la tabla ya conoce.")

exacto = all(a == b for a, b in zip(original, [ind.aptitud for ind in frio]))
exacto_caliente = all(a == b for a, b in zip(original, [ind.aptitud for ind in caliente]))
print(f"\nAmbas poblaciones restauradas coinciden con la guardada valor por valor: "
      f"{exacto and exacto_caliente}.")
print("Una instantánea que cambiara una aptitud sería peor que ninguna instantánea en absoluto, "
      "por lo que")
print("esto se verifica en lugar de asumirse.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(["solo genes", "genes + cache"], [trabajo_frio, trabajo_caliente],
            color=["tab:red", "tab:green"])
axes[0].set_title(f"Costo de restaurar {len(poblacion)} asignaciones")
axes[0].set_ylabel("unidades de trabajo")
for i, v in enumerate([trabajo_frio, trabajo_caliente]):
    axes[0].text(i, v, str(v), ha="center", va="bottom")
axes[1].plot(range(1, len(original) + 1), sorted(original), "o-", label="guardada")
axes[1].plot(range(1, len(original) + 1),
             sorted(ind.aptitud for ind in caliente), "x--", label="restaurada")
axes[1].set_title("Valores de aptitud restaurados, ordenados")
axes[1].set_xlabel("asignación")
axes[1].set_ylabel("aptitud")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.5)
fig.suptitle("Paso 4: los genes son baratos de almacenar; la aptitud es lo que cuesta")
fig.tight_layout()
FIGURAS.mkdir(exist_ok=True)
fig.savefig(FIGURAS / "turnos_04_instantanea.png", dpi=130)
plt.close(fig)
print("\nFigura guardada en figuras/turnos_04_instantanea.png")
# ------------------------------------------------------------------------------

