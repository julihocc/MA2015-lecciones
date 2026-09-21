"""
Lección 10 - Equipo 2: Roles legales aleatorios
=======================================
NUEVO EN ESTE PASO: equipo_aleatorio(), metricas_equipo(), y un censo de presupuesto.

CAMBIOS RESPECTO A equipo_01_la_plantilla.py
Introdúcelos en este orden:
    1. equipo_aleatorio()        llenar cada bloque de rol ordenado sin duplicados
    2. metricas_equipo()         mantener la habilidad y el precio como evidencia separada
    3. el censo                  medir factibilidad de presupuesto antes de la optimización

Ejecútalo:  python equipo_02_plantillas_aleatorias.py

Las 1,000 plantillas contienen once jugadores distintos. Solo 440 cumplen el presupuesto: legalidad de rol no es factibilidad de precio.
"""
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

SEMILLA = 3
DATOS = Path(__file__).with_name("players_20.csv")
FIGURAS = Path(__file__).resolve().parent.parent / "figures"
CONTEOS_ROLES = {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}
GRUPO_POR_ROL = 12
PRESUPUESTO_EUR = 600_000_000
TAMANO_POBLACION = 1000


def rol_principal(posiciones: str) -> str:
    """Primera posición FIFA → GK/DEF/MID/FWD.

    Args:
        posiciones: campo del CSV.

    Returns:
        El bloque de rol.

    Example:
        ``equipo_aleatorio`` muestrea 1+4+3+3 usando estos bloques.
    """
    primera = posiciones.split(",")[0].strip()
    if primera == "GK": return "GK"
    if primera in {"CB", "LB", "RB", "LWB", "RWB"}: return "DEF"
    if primera in {"CM", "CDM", "CAM", "LM", "RM"}: return "MID"
    return "FWD"


def cargar_grupo(ruta: Path = DATOS) -> pd.DataFrame:
    """48 candidatos, 12 por rol.

    Args:
        ruta: ``players_20.csv``.

    Returns:
        DataFrame con columna ``role``.

    Example:
        De 1,000 plantillas sorteadas aquí, 440 caben en €600m.
    """
    columnas = ["sofifa_id", "short_name", "overall", "value_eur", "player_positions"]
    frame = pd.read_csv(ruta, usecols=columnas).dropna(subset=["value_eur", "player_positions"])
    frame["role"] = frame.player_positions.map(rol_principal)
    return pd.concat([frame[frame.role == rol].sort_values(
        ["overall", "value_eur"], ascending=[False, True]).head(GRUPO_POR_ROL)
        for rol in CONTEOS_ROLES], ignore_index=True)


# --- NUEVO (1) equipo_aleatorio() ---------------------------------------------
def equipo_aleatorio(grupo: pd.DataFrame) -> List[int]:
    """Llena cada bloque sin duplicados. El precio no entra: por eso 440/1000 caben.

    Args:
        grupo: los 48 candidatos.

    Returns:
        Once índices distintos, ordenados por rol.

    Example:
        1,000/1,000 tienen once jugadores distintos. Solo 440 cumplen
        el presupuesto.
    """
    equipo: List[int] = []
    for rol, conteo in CONTEOS_ROLES.items():
        candidatos = grupo.index[grupo.role == rol].tolist()
        equipo.extend(random.sample(candidatos, conteo))
    return equipo
# ------------------------------------------------------------------------------


# --- NUEVO (2) metricas_equipo() ----------------------------------------------
def metricas_equipo(grupo: pd.DataFrame, equipo: Sequence[int]) -> Tuple[int, int]:
    """Habilidad y precio por separado. Mezclarlos escondería el techo de €600m.

    Args:
        grupo: candidatos.
        equipo: once índices.

    Returns:
        ``(overall total, valor_eur total)``.

    Example:
        440 equipos con valor ≤ 600m. El mejor overall de esos 440 se
        imprime como referencia, no como óptimo.
    """
    elegido = grupo.loc[list(equipo)]
    return int(elegido.overall.sum()), int(elegido.value_eur.sum())
# ------------------------------------------------------------------------------


# --- NUEVO (3) el censo -------------------------------------------------------
random.seed(SEMILLA)
grupo = cargar_grupo()
poblacion = [equipo_aleatorio(grupo) for _ in range(TAMANO_POBLACION)]
metricas = [metricas_equipo(grupo, equipo) for equipo in poblacion]
factibles = [(habilidad, costo) for habilidad, costo in metricas if costo <= PRESUPUESTO_EUR]
# ------------------------------------------------------------------------------

print("Lección 10 - Equipo 2: roles legales aleatorios")
print(f"Equipos con once jugadores distintos: {sum(len(set(t)) == 11 for t in poblacion)}/{TAMANO_POBLACION}")
print(f"Equipos dentro del presupuesto: {len(factibles)}/{TAMANO_POBLACION}")
print(f"Mejor habilidad total asequible: {max(habilidad for habilidad, _ in factibles)}")

FIGURAS.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter([costo / 1e6 for _, costo in metricas], [habilidad for habilidad, _ in metricas], alpha=0.3)
ax.axvline(PRESUPUESTO_EUR / 1e6, color="tab:red", label="presupuesto")
ax.set(xlabel="valor del equipo (€m)", ylabel="calificación general total",
       title="La legalidad de roles no garantiza factibilidad de presupuesto")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "equipo_02_plantillas_aleatorias.png", dpi=160)
plt.close(fig)

