"""
Lección 13 - Paso 2: Evaluar la función de aptitud una vez
========================================================
NUEVO EN ESTE PASO: la aptitud se calcula en el constructor y se almacena.

Esta es la sección 13.1 del libro, y es el mejor trato de todo el curso.
Los genes de un individuo no cambian después de que se construye, por lo que su aptitud tampoco puede
cambiar. A pesar de esto, el paso 1 lo recalculaba cada vez que se necesitaba un valor
- una vez por individuo dentro del ordenamiento, dos veces más por individuo en las
estadísticas - y pagaba una evaluación completa de 210 celdas cada vez.

Mueve la evaluación a `__init__` y el programa deja de pagar por lo que
ya sabe. Nada más cambia: ni la semilla, ni los operadores, ni el
orden de las extracciones aleatorias, ni la respuesta. Solo la factura.

La Lección 02, paso 1, contó la otra mitad de esta misma idea. Su ejecución costó 107
evaluaciones de aptitud en lugar de las evidentes 110, porque un individuo que es
seleccionado pero ni cruzado ni mutado es el *mismo objeto* y nunca es
reconstruido. La Lección 02 tiene la cuenta; esta es la técnica que hace que la cuenta sea
la verdad - una vez que la aptitud vive en el objeto, "no reconstruido" y "no
reevaluado" son la misma oración.

CAMBIOS RESPECTO A turnos_01_el_costo_de_una_ejecucion.py
Introdúcelos en este orden:
    1. Individuo         evaluar una vez, en el constructor, y almacenar el valor
    2. aptitud_de()      leer una aptitud se convierte en una búsqueda; ningún sitio de llamada cambia
    3. el_reporte        cuenta la misma ejecución de nuevo y verifica que la respuesta no se movió

Ejecútalo:  python turnos_02_evaluar_una_vez.py

428 llamadas, 89,880 unidades de trabajo: 54.0% menos, con SELECT y STATS
en 0. La mejor aptitud sigue siendo −87: el ahorro no costó nada.
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

    Recorre todas las celdas EMPLEADOS × TURNOS. Una evaluación completa
    son 210 unidades de trabajo.

    Args:
        genoma: 105 bits, empleado mayor.

    Returns:
        Penalización de personal, no negativa.
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
    """Cuenta las veces que un empleado vuelve al servicio todavía descansando.

    Args:
        genoma: 105 bits, empleado mayor.

    Returns:
        Violaciones de descanso. En el objetivo pesa 5.
    """
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


def aptitud_turnos(genoma: List[int]) -> int:
    """El objetivo, a MAXIMIZAR. Una asignación perfecta puntúa 0.

    Args:
        genoma: 105 bits.

    Returns:
        ``-(personal + 5 * descanso)``. Semilla 3: −87.
    """
    COSTO["llamadas"] += 1
    return -(desviacion_turno(genoma) + PESO_DESCANSO * violaciones_descanso(genoma))


# --- NUEVO (1) Individuo ------------------------------------------------------
class Individuo:
    """Una asignación candidata: 105 bits, primero empleados, evaluado exactamente una vez.

    Los genes nunca cambian después de la construcción, por lo que la aptitud tampoco puede
    cambiar. Calcularla aquí y almacenarla es la totalidad de la sección 13.1:
    tres líneas, sin aproximación, sin ceder nada.
    """

    def __init__(self, lista_genes: List[int]) -> None:
        self.lista_genes = list(lista_genes)
        self.aptitud = aptitud_turnos(self.lista_genes)
        COSTO["construidos"] += 1
# ------------------------------------------------------------------------------


# --- NUEVO (2) aptitud_de() ---------------------------------------------------
def aptitud_de(ind: Individuo) -> int:
    """Leer una aptitud ahora es una búsqueda. Ningún sitio de llamada tuvo que cambiar."""
    return ind.aptitud
# ------------------------------------------------------------------------------


def crear_aleatorio() -> Individuo:
    """Una asignación aleatoria de 105 bits. Ahora paga una sola evaluación al nacer."""
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

# --- NUEVO (3) el_reporte -----------------------------------------------------
# El paso 1 imprimió estos tres números. Se citan aquí, no se recalculan, así
# que este script puede decir si el cambio fue gratis y si fue seguro.
PASO_1_LLAMADAS = 930
PASO_1_TRABAJO = 195300
PASO_1_MEJOR = -87

print("Lección 13, paso 2 - calcular la función de aptitud una vez")
print("=======================================================")
print("Llamadas de aptitud, divididas por la fase que las hizo.\n")
print(" gen | individuos nuevos | SELEC  | REPROD| ESTAD | mejor| media")
print("-----+-------------------+--------+-------+-------+------+-------")
for fila in historial:
    print(f" {fila['gen']:3d} | {fila['construidos']:17d} | {fila['seleccion']:6d} |"
          f" {fila['reproduccion']:5d} | {fila['estadisticas']:5d} | {fila['mejor']:4d} |"
          f" {fila['media']:6.1f}")

construidos = COSTO["construidos"]
llamadas = COSTO["llamadas"]
total_seleccion = sum(r["seleccion"] for r in historial)
total_reproduccion = sum(r["reproduccion"] for r in historial)
total_estadisticas = sum(r["estadisticas"] for r in historial)
print(f"\nLa factura se ha movido a REPROD, que es donde están los nuevos genomas. "
      f"SELEC ahora cuesta")
print(f"{total_seleccion} llamadas y ESTAD {total_estadisticas}: leer un valor almacenado no es una "
      "evaluación, por lo que ambas")
print(f"columnas son cero hasta abajo, y REPROD paga {total_reproduccion} - una llamada por "
      "individuo")
print("que construye.")
print("\n                     paso 1     paso 2     ahorrado")
print(f"  llamadas aptitud {PASO_1_LLAMADAS:8d}   {llamadas:8d}   {100 * (PASO_1_LLAMADAS - llamadas) / PASO_1_LLAMADAS:5.1f}%")
print(f"  unidades trabajo {PASO_1_TRABAJO:8d}   {COSTO['trabajo']:8d}   {100 * (PASO_1_TRABAJO - COSTO['trabajo']) / PASO_1_TRABAJO:5.1f}%")
print(f"\nLa ejecución todavía construye {construidos} individuos - los mismos {construidos} - y ahora llama "
      f"a la función de")
print(f"aptitud {llamadas} veces, exactamente una vez cada uno. Ese es el piso para este diseño: "
      "ningún")
print("esquema en esta lección evalúa un genoma completamente nuevo menos de una vez.")

mejor_aptitud = aptitud_de(mejor_individuo)
print(f"\nMejor asignación encontrada: aptitud {mejor_aptitud}.")
if mejor_aptitud == PASO_1_MEJOR:
    print(f"El paso 1 terminó en {PASO_1_MEJOR} también. Misma semilla, misma respuesta: este ahorro se "
          "pagó con")
    print("nada en absoluto, que es la razón por la que pertenece al comienzo de la lección y no "
          "al final.")
else:
    print(f"El paso 1 terminó en {PASO_1_MEJOR}. La respuesta se movió, así que esta refactorización NO "
          "conservó el comportamiento.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(["paso 1", "paso 2"], [PASO_1_LLAMADAS, llamadas], color=["tab:red", "tab:green"])
axes[0].set_title("Llamadas de aptitud en la misma ejecución")
axes[0].set_ylabel("llamadas")
for i, v in enumerate([PASO_1_LLAMADAS, llamadas]):
    axes[0].text(i, v, str(v), ha="center", va="bottom")
axes[1].plot([r["gen"] for r in historial], [r["mejor"] for r in historial], "o-", label="mejor")
axes[1].plot([r["gen"] for r in historial], [r["media"] for r in historial], "s--", label="media")
axes[1].set_title("La búsqueda en sí está intacta")
axes[1].set_xlabel("generación")
axes[1].set_ylabel("aptitud")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.5)
fig.suptitle("Paso 2: la misma búsqueda, a una evaluación por individuo")
fig.tight_layout()
FIGURAS.mkdir(exist_ok=True)
fig.savefig(FIGURAS / "turnos_02_evaluar_una_vez.png", dpi=130)
plt.close(fig)
print("\nFigura guardada en figuras/turnos_02_evaluar_una_vez.png")
# ------------------------------------------------------------------------------

