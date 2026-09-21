"""
Lección 09 - Horarios 3: Reparar las restricciones
==================================================
NUEVO EN ESTE PASO: costo_preferencia(), reparar(), y un censo antes/después.

CAMBIOS RESPECTO A schedule_02_horarios_aleatorios.py
Introdúzcalos en este orden:
    1. costo_preferencia()   separa la calidad del horario de la legalidad
    2. reparar()             elimina el exceso de trabajo, luego llena cada escasez
    3. la comparación        demuestra que la población almacenada es factible

Ejecútalo:  python schedule_03_reparacion.py

La reparación hace factibles a 500/500 horarios. Los costos reparados van desde 58: la legalidad ya no es el objetivo, el costo de preferencia sí.
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
TAMANO_POBLACION = 500
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
        ``reparar`` usa este mapa para quitar un 1 de más o poner un 1 de
        menos sin recorrer el vector a ciegas.
    """
    return empleado * len(DIAS) * len(TURNOS) + dia * len(TURNOS) + turno


def horario_aleatorio() -> List[int]:
    """168 monedas independientes. Casi todos nacen ilegales; ``reparar`` los legaliza.

    Returns:
        168 bits crudos.

    Example:
        0/500 factibles antes de reparar; 500/500 después.
    """
    probabilidad = sum(DEMANDA) / (len(EMPLEADOS) * len(TURNOS))
    return [int(random.random() < probabilidad) for _ in PREFERENCIA]


def violaciones(bits: Sequence[int]) -> Tuple[int, int]:
    """Cobertura mal cubierta más sobrecarga. Tras ``reparar`` ambos deben ser 0.

    Args:
        bits: cromosoma de 168 bits.

    Returns:
        ``(fallos de cobertura, exceso de carga)``.

    Example:
        0/500 ceros antes, 500/500 ceros después. El censo no mira el
        costo: mira la legalidad.
    """
    cobertura = 0
    for dia in range(len(DIAS)):
        for turno, requerido in enumerate(DEMANDA):
            asignados = sum(bits[indice_gen(empleado, dia, turno)]
                           for empleado in range(len(EMPLEADOS)))
            cobertura += abs(asignados - requerido)
    cargas_trabajo = [sum(bits[indice_gen(empleado, dia, turno)]
                     for dia in range(len(DIAS)) for turno in range(len(TURNOS)))
                 for empleado in range(len(EMPLEADOS))]
    return cobertura, sum(max(0, trabajo - MAX_TURNOS) for trabajo in cargas_trabajo)


# --- NUEVO (1) costo_preferencia() --------------------------------------------
def costo_preferencia(bits: Sequence[int]) -> int:
    """Menor es mejor; las restricciones están deliberadamente ausentes de este valor.

    Si el costo mezclara violaciones, un infactible barato ganaría a un
    legal caro. Por eso la legalidad vive en ``violaciones`` y la calidad
    aquí.

    Args:
        bits: cromosoma de 168 bits.

    Returns:
        Suma de preferencias de los turnos asignados.

    Example:
        Tras reparar, el mínimo es 58. Ese 58 es un horario legal, no
        un infactible disfrazado.
    """
    return sum(bit * costo for bit, costo in zip(bits, PREFERENCIA))
# ------------------------------------------------------------------------------


# --- NUEVO (2) reparar() ------------------------------------------------------
def reparar(origen: Sequence[int]) -> List[int]:
    """Quita excesos, recorta sobrecargas y llena escaseces. Legaliza, no optimiza.

    El orden importa: primero se recorta lo de más (cobertura y carga),
    luego se rellena lo de menos eligiendo al empleado más barato con
    cupo. El sesgo de preferencia queda en la muestra: por eso los
    costos reparados empiezan en 58, no en un aleatorio puro.

    Args:
        origen: 168 bits crudos.

    Returns:
        Un horario con violaciones (0, 0).

    Example:
        500/500 factibles. Costos desde 58. La selección del paso 4
        hereda esta distribución, no el cubo {0,1}^168.
    """
    bits = list(origen)
    cargas_trabajo = [sum(bits[indice_gen(empleado, dia, turno)]
                     for dia in range(len(DIAS)) for turno in range(len(TURNOS)))
                 for empleado in range(len(EMPLEADOS))]
    for dia in range(len(DIAS)):
        for turno, requerido in enumerate(DEMANDA):
            asignados = [e for e in range(len(EMPLEADOS)) if bits[indice_gen(e, dia, turno)]]
            while len(asignados) > requerido:
                empleado = max(asignados, key=lambda e: (PREFERENCIA[indice_gen(e, dia, turno)], cargas_trabajo[e]))
                bits[indice_gen(empleado, dia, turno)] = 0
                cargas_trabajo[empleado] -= 1
                asignados.remove(empleado)
    for empleado in range(len(EMPLEADOS)):
        while cargas_trabajo[empleado] > MAX_TURNOS:
            posiciones = [(dia, turno) for dia in range(len(DIAS)) for turno in range(len(TURNOS))
                         if bits[indice_gen(empleado, dia, turno)]]
            dia, turno = max(posiciones, key=lambda dt: PREFERENCIA[indice_gen(empleado, *dt)])
            bits[indice_gen(empleado, dia, turno)] = 0
            cargas_trabajo[empleado] -= 1
    for dia in range(len(DIAS)):
        for turno, requerido in enumerate(DEMANDA):
            asignados = sum(bits[indice_gen(e, dia, turno)] for e in range(len(EMPLEADOS)))
            while asignados < requerido:
                opciones = [e for e in range(len(EMPLEADOS))
                           if cargas_trabajo[e] < MAX_TURNOS and not bits[indice_gen(e, dia, turno)]]
                empleado = min(opciones, key=lambda e: (PREFERENCIA[indice_gen(e, dia, turno)], cargas_trabajo[e]))
                bits[indice_gen(empleado, dia, turno)] = 1
                cargas_trabajo[empleado] += 1
                asignados += 1
    return bits
# ------------------------------------------------------------------------------


# --- NUEVO (3) la comparación -------------------------------------------------
random.seed(SEMILLA)
crudos = [horario_aleatorio() for _ in range(TAMANO_POBLACION)]
reparados = [reparar(bits) for bits in crudos]
antes = [sum(violaciones(bits)) for bits in crudos]
despues = [sum(violaciones(bits)) for bits in reparados]
# ------------------------------------------------------------------------------

print("Lección 09 - Horarios 3: reparación")
print(f"Factibles antes de la reparación: {sum(v == 0 for v in antes)}/{TAMANO_POBLACION}")
print(f"Factibles después de la reparación:  {sum(v == 0 for v in despues)}/{TAMANO_POBLACION}")
print(f"Costo de preferencia después de la reparación: min={min(map(costo_preferencia, reparados))}, "
      f"media={sum(map(costo_preferencia, reparados)) / TAMANO_POBLACION:.1f}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist([costo_preferencia(bits) for bits in reparados], color="tab:green")
ax.set(xlabel="costo de preferencia", ylabel="horarios factibles",
       title="La reparación hace de la legalidad un invariante; el costo queda por optimizar")
fig.tight_layout()
fig.savefig(FIGURES / "schedule_03_reparacion.png", dpi=160)
plt.close(fig)

