"""
Lección 09 - Horarios 1: Los requisitos
=======================================
NUEVO EN ESTE PASO: un horario binario, demanda explícita, y capacidad de empleados.

Un bit asigna a un empleado a un turno. La instancia necesita 35 asignaciones
en 21 turnos, mientras que ocho empleados pueden cubrir como máximo cinco turnos cada uno.

Ejecútalo:  python schedule_01_los_requisitos.py

168 bits codifican 35 asignaciones requeridas bajo la capacidad 40. El margen es de 5 turnos: hay holgura, pero un bit aleatorio no la encuentra.
"""
from pathlib import Path

import matplotlib.pyplot as plt

SEMILLA = 15
DIAS = ("Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom")
TURNOS = ("mañana", "día", "tarde")
EMPLEADOS = tuple(f"E{i}" for i in range(1, 9))
DEMANDA = (2, 2, 1)
MAX_TURNOS = 5
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def indice_gen(empleado: int, dia: int, turno: int) -> int:
    """Mapea una coordenada del horario a una posición del cromosoma binario.

    El orden es empleado, luego día, luego turno. Un corte entre empleados
    (paso 4) cae en un múltiplo de 21 porque cada empleado ocupa
    7 × 3 bits.

    Args:
        empleado: índice 0..7.
        dia: índice 0..6.
        turno: índice 0..2.

    Returns:
        Posición 0..167.

    Example:
        8 × 7 × 3 = 168 bits. Las 35 asignaciones requeridas viven en
        esas 168 celdas, no en un vector más corto.
    """
    return empleado * len(DIAS) * len(TURNOS) + dia * len(TURNOS) + turno


longitud_cromosoma = len(EMPLEADOS) * len(DIAS) * len(TURNOS)
asignaciones_requeridas = len(DIAS) * sum(DEMANDA)
capacidad_disponible = len(EMPLEADOS) * MAX_TURNOS

print("Lección 09 - Horarios 1: los requisitos")
print(f"Cromosoma: {len(EMPLEADOS)} empleados x {len(DIAS)} días x "
      f"{len(TURNOS)} turnos = {longitud_cromosoma} bits")
print(f"Asignaciones requeridas: {asignaciones_requeridas}")
print(f"Capacidad disponible:    {capacidad_disponible}")
print(f"Margen de capacidad:       {capacidad_disponible - asignaciones_requeridas}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
matriz = [list(DEMANDA) for _ in DIAS]
imagen = ax.imshow(matriz, cmap="Blues", vmin=0, vmax=max(DEMANDA))
ax.set_xticks(range(len(TURNOS)), TURNOS)
ax.set_yticks(range(len(DIAS)), DIAS)
for dia in range(len(DIAS)):
    for turno in range(len(TURNOS)):
        ax.text(turno, dia, DEMANDA[turno], ha="center", va="center")
ax.set_title("Empleados requeridos en cada turno")
fig.colorbar(imagen, ax=ax, label="empleados")
fig.tight_layout()
fig.savefig(FIGURES / "schedule_01_los_requisitos.png", dpi=160)
plt.close(fig)

