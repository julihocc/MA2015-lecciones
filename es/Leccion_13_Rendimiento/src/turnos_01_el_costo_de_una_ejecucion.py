"""
Lección 13 - Paso 1: El costo real de una ejecución
==============================================
NUEVO EN ESTE PASO: el problema de asignación de turnos, un algoritmo genético simple, y dos
contadores que lo miden - llamadas a la aptitud y unidades de trabajo.

Esta es la última lección del curso, y es la única sobre el costo de
ejecutar un algoritmo genético en lugar de sobre el algoritmo en sí. Todo lo
que sigue es un intento de hacer que esta ejecución sea más barata. Así que lo primero que hay que construir
no es una optimización: es el medidor.

Dos contadores se ejecutan en todos los scripts de esta lección.

  * COSTO["llamadas"] cuenta las llamadas a la función de aptitud.
  * COSTO["trabajo"]  cuenta los pasos elementales que esas llamadas toman - un incremento
    por celda (empleado, turno) que la evaluación examina.

Las unidades de trabajo importan más que los segundos. Un segundo es un hecho sobre la laptop que
ejecutó el script; una unidad de trabajo es un hecho sobre el algoritmo, y es idéntica
en cada máquina, cada Python y cada ejecución con esta semilla. Solo un script en
esta lección imprime una lectura de reloj, y dice fuertemente lo que vale esa lectura.

El código de abajo está escrito de la manera en que el problema se escribe usualmente primero: la
función de aptitud se llama dondequiera que se necesite un valor de aptitud. Ese es el
defecto con el que abre esta lección, y el paso 2 lo elimina.

Ejecútalo:  python turnos_01_el_costo_de_una_ejecucion.py

La ejecución construye 428 individuos y llama a la aptitud 930 veces — 2.17
evaluaciones por individuo, repartidas SELECT 300 / CRUZA 0 / STATS 630.
195,300 unidades de trabajo, mejor aptitud −87.
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

    Recorre todas las celdas EMPLEADOS × TURNOS, que es de donde viene el
    costo de una evaluación y por qué el contador de trabajo vive en el
    bucle interior.

    Args:
        genoma: 105 bits, empleado mayor.

    Returns:
        Penalización de personal, no negativa. Un turno vacío cuesta 100.

    Example:
        Junto con las violaciones de descanso, una llamada completa cuesta
        210 unidades de trabajo.
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
        Número de violaciones de descanso. En el objetivo pesa 5.

    Example:
        El campeón de esta semilla tiene 17 violaciones y puntúa −87.
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
        genoma: 105 bits, empleado mayor.

    Returns:
        ``-(personal + 5 * descanso)``. La semilla 3 llega a −87.

    Example:
        Los pasos 1 a 6 reportan todos esa misma aptitud −87.
    """
    COSTO["llamadas"] += 1
    return -(desviacion_turno(genoma) + PESO_DESCANSO * violaciones_descanso(genoma))


class Individuo:
    """Una asignación candidata: 105 bits, primero empleados.

    Nota lo que esta clase NO hace: no evalúa nada. La aptitud se
    obtiene pidiéndola, y pedir cuesta una evaluación completa cada vez.
    """

    def __init__(self, lista_genes: List[int]) -> None:
        self.lista_genes = list(lista_genes)
        COSTO["construidos"] += 1

    def aptitud(self) -> int:
        """Recalcula el objetivo desde cero. Cada llamada es una evaluación pagada."""
        return aptitud_turnos(self.lista_genes)


def aptitud_de(ind: Individuo) -> int:
    """El único lugar donde el resto del programa lee un valor de aptitud.

    Args:
        ind: una asignación candidata.

    Returns:
        La aptitud pedida. El paso 1 la recalcula; los posteriores la
        consultan.
    """
    return ind.aptitud()


def crear_aleatorio() -> Individuo:
    """Una asignación aleatoria de 105 bits."""
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
    """Cruza de n puntos sobre bits. Copia segmentos; no inventa un 1 que no estuviera."""

    cortes = sorted(random.sample(range(1, len(p1) - 1), n) + [0, len(p1)])
    c1, c2 = list(p1), list(p2)
    for i in range(1, n + 1, 2):
        c1[cortes[i]:cortes[i + 1]] = p2[cortes[i]:cortes[i + 1]]
        c2[cortes[i]:cortes[i + 1]] = p1[cortes[i]:cortes[i + 1]]
    return c1, c2


def mutacion_volteo_bit(genoma: List[int]) -> List[int]:
    """Invierte un bit al azar. Un cambio, una celda distinta."""

    mutante = list(genoma)
    pos = random.randint(0, len(genoma) - 1)
    mutante[pos] = 1 - mutante[pos]
    return mutante


def operacion_cruza(poblacion: List[Individuo]) -> List[Individuo]:
    """Aplica cruza por pares. En este paso los hijos no se evalúan todavía."""

    descendencia: List[Individuo] = []
    for ind1, ind2 in zip(poblacion[::2], poblacion[1::2]):
        if random.random() < PROBABILIDAD_CRUZA:
            g1, g2 = cruza_n_puntos(ind1.lista_genes, ind2.lista_genes, PUNTOS_CRUZA)
            descendencia.extend([Individuo(g1), Individuo(g2)])
        else:
            descendencia.extend([ind1, ind2])
    return descendencia


def operacion_mutacion(poblacion: List[Individuo]) -> List[Individuo]:
    """Aplica el volteo de bit. Tampoco evalúa: el costo aparece en SELECT y STATS."""
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

print("Lección 13, paso 1 - el medidor, antes de cualquier optimización")
print("=====================================================")
print(f"Problema de turnos: {EMPLEADOS} empleados x {DIAS} días x {TURNOS_POR_DIA} "
      f"turnos = {LONGITUD_GENOMA} bits.")
print(f"Una llamada de aptitud recorre {2 * EMPLEADOS * TURNOS} celdas, así que una llamada "
      f"cuesta {2 * EMPLEADOS * TURNOS} unidades de trabajo.\n")
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
print(f"\nREPROD cuesta {total_reproduccion} llamadas: construir un individuo evalúa "
      "nada aquí, porque")
print("esta versión no tiene lugar para guardar el valor. SELEC y ESTAD pagan "
      f"en cambio - {total_seleccion} y")
print(f"{total_estadisticas} llamadas - por números que el programa podría haber anotado una vez.")
print(f"\nLa ejecución creó {construidos} individuos y llamó a la función de aptitud "
      f"{llamadas} veces.")
print(f"Eso es {llamadas / construidos:.2f} evaluaciones por individuo construido, y "
      "cada uno de")
print("ellos después del primero recalcula un número que el programa ya sabía.")
print(f"Costo total: {COSTO['trabajo']} unidades de trabajo.")
print(f"\nUn límite inferior es una evaluación por individuo: {construidos} llamadas, "
      f"{construidos * 2 * EMPLEADOS * TURNOS} unidades de trabajo.")
print(f"La brecha entre {llamadas} y {construidos} es puro desperdicio, y el paso 2 recoge "
      "todo.")
print(f"\nMejor asignación encontrada: aptitud {aptitud_de(mejor_individuo)} "
      f"(0 sería una asignación sin ninguna violación).")

gens = [fila["gen"] for fila in historial]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(gens, [r["seleccion"] for r in historial], label="SELECCIÓN", color="tab:orange")
axes[0].bar(gens, [r["estadisticas"] for r in historial],
            bottom=[r["seleccion"] for r in historial], label="ESTADÍSTICAS", color="tab:red")
axes[0].plot(gens, [r["construidos"] for r in historial], "ko--", label="individuos construidos")
axes[0].set_title("A dónde van las llamadas de aptitud")
axes[0].set_xlabel("generación")
axes[0].set_ylabel("llamadas de aptitud")
axes[0].legend()
axes[1].plot(gens, [r["mejor"] for r in historial], "o-", label="mejor")
axes[1].plot(gens, [r["media"] for r in historial], "s--", label="media")
axes[1].set_title("La búsqueda en sí")
axes[1].set_xlabel("generación")
axes[1].set_ylabel("aptitud")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.5)
fig.suptitle("Paso 1: la ejecución cuesta muchas más evaluaciones que individuos tiene")
fig.tight_layout()
FIGURAS.mkdir(exist_ok=True)
fig.savefig(FIGURAS / "turnos_01_el_costo_de_una_ejecucion.png", dpi=130)
plt.close(fig)
print(f"\nFigura guardada en figuras/turnos_01_el_costo_de_una_ejecucion.png")

