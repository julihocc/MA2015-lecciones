"""
Lección 09 - Horarios 2: Horarios aleatorios
============================================
NUEVO EN ESTE PASO: horario_aleatorio(), violaciones(), y un censo de factibilidad.

CAMBIOS RESPECTO A schedule_01_los_requisitos.py
Introdúzcalos en este orden:
    1. horario_aleatorio()     muestrea cada asignación independientemente
    2. violaciones()           mantiene visibles los fallos de cobertura y carga de trabajo
    3. el censo                mide el fallo antes de asignar una aptitud

Ejecútalo:  python schedule_02_horarios_aleatorios.py

0 de 500 horarios aleatorios son factibles. El mejor todavía tiene 14 violaciones: la selección no tiene a quién copiar.
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


def indice_gen(empleado: int, dia: int, turno: int) -> int:
    """Posición del bit (empleado, día, turno) en el cromosoma de 168 bits.

    Args:
        empleado: índice 0..7.
        dia: índice 0..6.
        turno: índice 0..2.

    Returns:
        Posición 0..167.

    Example:
        ``violaciones`` recorre las 21 celdas (día, turno) sumando por
        esta función. Sin el mapa, el censo no sabría qué bits son un turno.
    """
    return empleado * len(DIAS) * len(TURNOS) + dia * len(TURNOS) + turno


# --- NUEVO (1) horario_aleatorio() --------------------------------------------
def horario_aleatorio() -> List[int]:
    """Cada bit es una moneda. La probabilidad media respeta la demanda, no las restricciones.

    La moneda se calibra a 5/24 para que el número esperado de 1 coincida
    con las 35 asignaciones. Eso no coordina cobertura ni carga: 0 de 500
    salen legales.

    Returns:
        168 bits independientes.

    Example:
        SEMILLA = 15, 500 sorteos: 0 factibles, mejor con 14 violaciones.
    """
    probabilidad = sum(DEMANDA) / (len(EMPLEADOS) * len(TURNOS))
    return [int(random.random() < probabilidad)
            for _ in range(len(EMPLEADOS) * len(DIAS) * len(TURNOS))]
# ------------------------------------------------------------------------------


# --- NUEVO (2) violaciones() --------------------------------------------------
def violaciones(bits: Sequence[int]) -> Tuple[int, int]:
    """Cobertura mal cubierta más sobrecarga de empleados. Cero solo si es legal.

    Dos cuentas, no una: un horario puede cubrir la demanda sobrecargando
    a alguien, o respetar las cargas y dejar un turno vacío. Mezclarlas
    en un solo número escondería cuál restricción falla.

    Args:
        bits: cromosoma de 168 bits.

    Returns:
        ``(fallos de cobertura, exceso de carga)``. Factible si ambos son 0.

    Example:
        0/500 factibles. El mejor aleatorio aún suma 14 violaciones.
    """
    cobertura = 0
    for dia in range(len(DIAS)):
        for turno, requerido in enumerate(DEMANDA):
            asignados = sum(bits[indice_gen(empleado, dia, turno)]
                           for empleado in range(len(EMPLEADOS)))
            cobertura += abs(asignados - requerido)
    carga_trabajo = 0
    for empleado in range(len(EMPLEADOS)):
        asignados = sum(bits[indice_gen(empleado, dia, turno)]
                       for dia in range(len(DIAS)) for turno in range(len(TURNOS)))
        carga_trabajo += max(0, asignados - MAX_TURNOS)
    return cobertura, carga_trabajo
# ------------------------------------------------------------------------------


# --- NUEVO (3) el censo -------------------------------------------------------
random.seed(SEMILLA)
poblacion = [horario_aleatorio() for _ in range(TAMANO_POBLACION)]
conteos = [sum(violaciones(bits)) for bits in poblacion]
factibles = sum(conteo == 0 for conteo in conteos)
# ------------------------------------------------------------------------------

print("Lección 09 - Horarios 2: horarios aleatorios")
print(f"Horarios aleatorios factibles: {factibles}/{TAMANO_POBLACION}")
print(f"Mediana de violaciones: {sorted(conteos)[len(conteos) // 2]}")
print(f"Mejor conteo de violaciones aleatorio: {min(conteos)}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(conteos, bins=range(min(conteos), max(conteos) + 2), color="tab:blue")
ax.set(xlabel="violaciones de cobertura más carga de trabajo", ylabel="horarios",
       title="Los bits independientes raramente forman un horario legal")
fig.tight_layout()
fig.savefig(FIGURES / "schedule_02_horarios_aleatorios.png", dpi=160)
plt.close(fig)

