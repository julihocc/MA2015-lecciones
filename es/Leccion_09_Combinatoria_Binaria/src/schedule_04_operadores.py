"""
Lección 09 - Horarios 4: Operadores bajo reparación
===================================================
NUEVO EN ESTE PASO: seleccion_torneo(), cruza(), y mutar().

CAMBIOS RESPECTO A schedule_03_reparacion.py
Introdúzcalos en este orden:
    1. seleccion_torneo()    prefiere el menor costo entre los horarios ya legales
    2. cruza()               intercambia horarios enteros de empleados, no bits arbitrarios
    3. mutar()               intercambia la asignación de un empleado antes de reparar

Ejecútalo:  python schedule_04_operadores.py

Los 100 hijos son factibles. El costo medio cae de 77.8 a 61.2: los operadores mueven la calidad y la reparación conserva la legalidad.
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
        La cruza corta en un múltiplo de 21: un empleado entero, no un
        bit a mitad de su semana.
    """
    return empleado * len(DIAS) * len(TURNOS) + dia * len(TURNOS) + turno


def horario_aleatorio() -> List[int]:
    """168 monedas. La población se repara antes de seleccionar.

    Returns:
        168 bits crudos.

    Example:
        100 padres reparados, costo medio 77.8.
    """
    probabilidad = sum(DEMANDA) / (len(EMPLEADOS) * len(TURNOS))
    return [int(random.random() < probabilidad) for _ in PREFERENCIA]


def violaciones(bits: Sequence[int]) -> Tuple[int, int]:
    """Cobertura mal cubierta más sobrecarga. Tras reparar, ambos son 0.

    Args:
        bits: cromosoma de 168 bits.

    Returns:
        ``(fallos de cobertura, exceso de carga)``.

    Example:
        100/100 hijos factibles. El operador no se fía de la cruza:
        repara después.
    """
    cobertura = sum(abs(sum(bits[indice_gen(e, dia, turno)] for e in range(len(EMPLEADOS))) - requerido)
                   for dia in range(len(DIAS)) for turno, requerido in enumerate(DEMANDA))
    cargas_trabajo = [sum(bits[indice_gen(e, dia, turno)] for dia in range(len(DIAS))
                     for turno in range(len(TURNOS))) for e in range(len(EMPLEADOS))]
    return cobertura, sum(max(0, trabajo - MAX_TURNOS) for trabajo in cargas_trabajo)


def costo_preferencia(bits: Sequence[int]) -> int:
    """Calidad del horario legal. El torneo minimiza este número.

    Args:
        bits: cromosoma de 168 bits.

    Returns:
        Suma de preferencias asignadas. Menor es mejor.

    Example:
        Padres 77.8, hijos 61.2. La presión de selección se lee aquí,
        no en las violaciones.
    """
    return sum(bit * costo for bit, costo in zip(bits, PREFERENCIA))


def reparar(origen: Sequence[int]) -> List[int]:
    """Legaliza tras cruza y mutación. Sin esto, los hijos no se pueden comparar.

    Args:
        origen: 168 bits, posiblemente ilegales.

    Returns:
        Un horario con violaciones (0, 0).

    Example:
        100/100 hijos factibles. El costo medio 61.2 es de horarios
        legales, no de infactibles baratos.
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


# --- NUEVO (1) seleccion_torneo() ---------------------------------------------
def seleccion_torneo(poblacion: List[List[int]], tamano: int = 3) -> List[int]:
    """Minimiza costo entre tres legales. La presión solo es honesta si todos caben.

    Args:
        poblacion: horarios ya reparados.
        tamano: contendientes; 3.

    Returns:
        El de menor ``costo_preferencia``.

    Example:
        Una generación con esta presión baja el costo medio de 77.8 a
        61.2.
    """
    return min(random.sample(poblacion, tamano), key=costo_preferencia)
# ------------------------------------------------------------------------------


# --- NUEVO (2) cruza() --------------------------------------------------------
def cruza(primero: Sequence[int], segundo: Sequence[int]) -> List[int]:
    """Corta entre empleados, no entre bits. Un empleado no se parte a mitad de semana.

    Un corte a bit arbitrario mezcla las 21 celdas de un empleado y
    destruye la única estructura que ``reparar`` puede reconstruir barato.

    Args:
        primero: padre legal.
        segundo: padre legal.

    Returns:
        Un hijo crudo, a menudo ilegal, que se repara después.

    Example:
        100 hijos, todos factibles tras reparar. El corte por empleado
        es la razón de que la reparación no empiece de cero.
    """
    corte_empleado = random.randrange(1, len(EMPLEADOS))
    corte = corte_empleado * len(DIAS) * len(TURNOS)
    return list(primero[:corte]) + list(segundo[corte:])
# ------------------------------------------------------------------------------


# --- NUEVO (3) mutar() --------------------------------------------------------
def mutar(origen: Sequence[int]) -> List[int]:
    """Mueve un turno de un empleado: apaga un 1 y enciende un 0 del mismo bloque.

    Un flip de bit suelto suele violar cobertura y carga a la vez. Un
    intercambio interno deja la carga del empleado igual y le pide a
    ``reparar`` solo el ajuste de cobertura.

    Args:
        origen: horario, usualmente legal.

    Returns:
        168 bits, posiblemente ilegales.

    Example:
        Tras reparar, 100/100 hijos factibles y costo medio 61.2.
    """
    bits = list(origen)
    empleado = random.randrange(len(EMPLEADOS))
    asignados = [indice_gen(empleado, dia, turno) for dia in range(len(DIAS))
                for turno in range(len(TURNOS)) if bits[indice_gen(empleado, dia, turno)]]
    no_asignados = [indice_gen(empleado, dia, turno) for dia in range(len(DIAS))
                  for turno in range(len(TURNOS)) if not bits[indice_gen(empleado, dia, turno)]]
    bits[random.choice(asignados)] = 0
    bits[random.choice(no_asignados)] = 1
    return bits
# ------------------------------------------------------------------------------


random.seed(SEMILLA)
poblacion = [reparar(horario_aleatorio()) for _ in range(TAMANO_POBLACION)]
hijos = [reparar(mutar(cruza(seleccion_torneo(poblacion), seleccion_torneo(poblacion))))
            for _ in range(TAMANO_POBLACION)]

print("Lección 09 - Horarios 4: operadores")
print(f"Hijos factibles después de la reparación: {sum(sum(violaciones(c)) == 0 for c in hijos)}/{TAMANO_POBLACION}")
print(f"Costo medio de los padres: {sum(map(costo_preferencia, poblacion)) / TAMANO_POBLACION:.1f}")
print(f"Costo medio de los hijos:  {sum(map(costo_preferencia, hijos)) / TAMANO_POBLACION:.1f}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot([[costo_preferencia(bits) for bits in poblacion],
            [costo_preferencia(bits) for bits in hijos]], tick_labels=["padres", "hijos"])
ax.set(ylabel="costo de preferencia", title="Los operadores cambian la calidad; la reparación preserva la legalidad")
fig.tight_layout()
fig.savefig(FIGURES / "schedule_04_operadores.png", dpi=160)
plt.close(fig)

