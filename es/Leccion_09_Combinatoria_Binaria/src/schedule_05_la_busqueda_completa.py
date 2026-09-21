"""
Lección 09 - Horarios 5: La búsqueda completa
=============================================
NUEVO EN ESTE PASO: una_generacion(), ejecutar(), y una comparación honesta con línea base.

CAMBIOS RESPECTO A schedule_04_operadores.py
Introdúzcalos en este orden:
    1. una_generacion()      mantiene el mejor horario y produce hijos reparados
    2. ejecutar()            repite el proceso bajo un presupuesto de evaluación fijo
    3. el reporte base       compara el resultado con horarios aleatorios reparados

Ejecútalo:  python schedule_05_la_busqueda_completa.py

El AG con costo 41 vence al mejor de 5,100 horarios aleatorios reparados, con costo 52, al mismo conteo de evaluaciones. La legalidad se mantiene en (0, 0).
"""
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEMILLA = 15
DIAS = ("Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom")
TURNOS = ("mañana", "día", "tarde")
EMPLEADOS = tuple(f"E{i}" for i in range(1, 9))
DEMANDA = (2, 2, 1)
MAX_TURNOS = 5
TAMANO_POBLACION = 100
GENERACIONES = 50
FIGURES = Path(__file__).resolve().parent.parent / "figures"
PREFERENCIA = tuple(1 + ((empleado * 7 + dia * 3 + turno * 5) % 5)
                   for empleado in range(len(EMPLEADOS))
                   for dia in range(len(DIAS)) for turno in range(len(TURNOS)))


def indice_gen(empleado: int, dia: int, turno: int) -> int:
    """Posición del bit (empleado, día, turno) en el cromosoma de 168 bits.

    Args:
        empleado: índice 0..7.
        dia: índice 0..6.
        turno: índice 0..2.

    Returns:
        Posición 0..167.

    Example:
        El campeón de costo 41 es un vector de 168 bits indexado así.
    """
    return empleado * len(DIAS) * len(TURNOS) + dia * len(TURNOS) + turno


def horario_aleatorio() -> List[int]:
    """168 monedas. La línea base repara 5,100 de estos, no los compara crudos.

    Returns:
        168 bits crudos.

    Example:
        El mejor de 5,100 reparados vale 52. El AG, con las mismas
        evaluaciones, vale 41.
    """
    probabilidad = sum(DEMANDA) / (len(EMPLEADOS) * len(TURNOS))
    return [int(random.random() < probabilidad) for _ in PREFERENCIA]


def violaciones(bits: Sequence[int]) -> Tuple[int, int]:
    """Cobertura mal cubierta más sobrecarga. El campeón debe imprimir (0, 0).

    Args:
        bits: cromosoma de 168 bits.

    Returns:
        ``(fallos de cobertura, exceso de carga)``.

    Example:
        El horario final del AG reporta cobertura=0 y carga=0 junto al
        costo 41.
    """
    cobertura = sum(abs(sum(bits[indice_gen(e, dia, turno)] for e in range(len(EMPLEADOS))) - requerido)
                   for dia in range(len(DIAS)) for turno, requerido in enumerate(DEMANDA))
    cargas_trabajo = [sum(bits[indice_gen(e, dia, turno)] for dia in range(len(DIAS))
                     for turno in range(len(TURNOS))) for e in range(len(EMPLEADOS))]
    return cobertura, sum(max(0, trabajo - MAX_TURNOS) for trabajo in cargas_trabajo)


def costo_preferencia(bits: Sequence[int]) -> int:
    """Calidad del horario legal. El AG y la línea base se comparan aquí.

    Args:
        bits: cromosoma de 168 bits.

    Returns:
        Suma de preferencias asignadas. Menor es mejor.

    Example:
        AG 41 contra mejor aleatorio reparado 52, mismo presupuesto de
        5,100 evaluaciones.
    """
    return sum(bit * costo for bit, costo in zip(bits, PREFERENCIA))


def reparar(origen: Sequence[int]) -> List[int]:
    """Legaliza cada hijo y cada muestra de la línea base. Sin esto, 41 no es comparable.

    Args:
        origen: 168 bits, posiblemente ilegales.

    Returns:
        Un horario con violaciones (0, 0).

    Example:
        AG y aleatorio reparado usan la misma regla. La ventaja 41 vs 52
        no es "el AG ignora restricciones".
    """
    bits = list(origen)
    cargas_trabajo = [sum(bits[indice_gen(e, dia, turno)] for dia in range(len(DIAS))
                     for turno in range(len(TURNOS))) for e in range(len(EMPLEADOS))]
    for dia in range(len(DIAS)):
        for turno, requerido in enumerate(DEMANDA):
            asignados = [e for e in range(len(EMPLEADOS)) if bits[indice_gen(e, dia, turno)]]
            while len(asignados) > requerido:
                e = max(asignados, key=lambda x: (PREFERENCIA[indice_gen(x, dia, turno)], cargas_trabajo[x]))
                bits[indice_gen(e, dia, turno)] = 0; cargas_trabajo[e] -= 1; asignados.remove(e)
    for e in range(len(EMPLEADOS)):
        while cargas_trabajo[e] > MAX_TURNOS:
            posiciones = [(dia, turno) for dia in range(len(DIAS)) for turno in range(len(TURNOS))
                         if bits[indice_gen(e, dia, turno)]]
            dia, turno = max(posiciones, key=lambda dt: PREFERENCIA[indice_gen(e, *dt)])
            bits[indice_gen(e, dia, turno)] = 0; cargas_trabajo[e] -= 1
    for dia in range(len(DIAS)):
        for turno, requerido in enumerate(DEMANDA):
            asignados = sum(bits[indice_gen(e, dia, turno)] for e in range(len(EMPLEADOS)))
            while asignados < requerido:
                opciones = [e for e in range(len(EMPLEADOS)) if cargas_trabajo[e] < MAX_TURNOS
                           and not bits[indice_gen(e, dia, turno)]]
                e = min(opciones, key=lambda x: (PREFERENCIA[indice_gen(x, dia, turno)], cargas_trabajo[x]))
                bits[indice_gen(e, dia, turno)] = 1; cargas_trabajo[e] += 1; asignados += 1
    return bits


def seleccion_torneo(poblacion: List[List[int]], tamano: int = 3) -> List[int]:
    """Minimiza costo entre tres legales.

    Args:
        poblacion: horarios ya reparados.
        tamano: contendientes; 3.

    Returns:
        El de menor costo de preferencia.

    Example:
        50 generaciones con esta presión llegan a costo 41.
    """
    return min(random.sample(poblacion, tamano), key=costo_preferencia)


def cruza(primero: Sequence[int], segundo: Sequence[int]) -> List[int]:
    """Corta entre empleados. El hijo se muta y repara antes de almacenarse.

    Args:
        primero: padre legal.
        segundo: padre legal.

    Returns:
        Un hijo crudo.

    Example:
        50 generaciones de este corte, más mutación y reparación, bajan
        el costo a 41.
    """
    corte = random.randrange(1, len(EMPLEADOS)) * len(DIAS) * len(TURNOS)
    return list(primero[:corte]) + list(segundo[corte:])


def mutar(origen: Sequence[int]) -> List[int]:
    """Mueve un turno de un empleado. ``reparar`` cierra las violaciones que abre.

    Args:
        origen: horario, usualmente legal.

    Returns:
        168 bits, posiblemente ilegales.

    Example:
        Combinada con elitismo, 50 generaciones alcanzan costo 41.
    """
    bits = list(origen)
    empleado = random.randrange(len(EMPLEADOS))
    asignados = [indice_gen(empleado, dia, turno) for dia in range(len(DIAS))
                for turno in range(len(TURNOS)) if bits[indice_gen(empleado, dia, turno)]]
    no_asignados = [indice_gen(empleado, dia, turno) for dia in range(len(DIAS))
                  for turno in range(len(TURNOS)) if not bits[indice_gen(empleado, dia, turno)]]
    bits[random.choice(asignados)] = 0; bits[random.choice(no_asignados)] = 1
    return bits


# --- NUEVO (1) una_generacion() -----------------------------------------------
def una_generacion(poblacion: List[List[int]], elite: List[int]) -> List[List[int]]:
    """Cruza, muta y repara, copiando al elite para no perder el 41 si ya está.

    Args:
        poblacion: generación actual.
        elite: mejor de la historia.

    Returns:
        Nueva población de 100 horarios legales.

    Example:
        50 vueltas × 100 = 5,100 evaluaciones, las mismas que la línea
        base aleatoria.
    """
    hijos = [elite]
    while len(hijos) < TAMANO_POBLACION:
        hijo = cruza(seleccion_torneo(poblacion), seleccion_torneo(poblacion))
        hijos.append(reparar(mutar(hijo)))
    return hijos
# ------------------------------------------------------------------------------


# --- NUEVO (2) ejecutar() -----------------------------------------------------
def ejecutar() -> Tuple[List[int], List[int]]:
    """50 generaciones desde SEMILLA = 15. Devuelve el campeón y su historia.

    Returns:
        ``(mejor horario, historia de costos)``.

    Example:
        Costo 41, violaciones (0, 0). La línea base de 5,100 reparados
        se queda en 52.
    """
    random.seed(SEMILLA)
    poblacion = [reparar(horario_aleatorio()) for _ in range(TAMANO_POBLACION)]
    mejor = min(poblacion, key=costo_preferencia)
    historia = [costo_preferencia(mejor)]
    for _ in range(GENERACIONES):
        poblacion = una_generacion(poblacion, mejor)
        mejor = min((mejor, min(poblacion, key=costo_preferencia)), key=costo_preferencia)
        historia.append(costo_preferencia(mejor))
    return mejor, historia
# ------------------------------------------------------------------------------


# --- NUEVO (3) el reporte base ------------------------------------------------
mejor, historia = ejecutar()
random.seed(SEMILLA + 1)
linea_base = [costo_preferencia(reparar(horario_aleatorio()))
            for _ in range(TAMANO_POBLACION * (GENERACIONES + 1))]
# ------------------------------------------------------------------------------

print("Lección 09 - Horarios 5: la búsqueda completa")
print(f"Mejor costo de preferencia del AG: {costo_preferencia(mejor)}")
print(f"Mejor de {len(linea_base):,} horarios aleatorios reparados: {min(linea_base)}")
print(f"Violaciones del horario final: cobertura={violaciones(mejor)[0]}, carga_trabajo={violaciones(mejor)[1]}")
print(f"Evaluaciones de horarios: {TAMANO_POBLACION * (GENERACIONES + 1):,}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.step(range(len(historia)), historia, where="post", label="mejor costo AG")
ax.axhline(min(linea_base), color="black", linestyle="--", label="mejor aleatorio reparado")
ax.set(xlabel="generación", ylabel="costo de preferencia",
       title="Calidad de la programación en un límite de factibilidad visible")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "schedule_05_la_busqueda_completa.png", dpi=160)
plt.close(fig)

