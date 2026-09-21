"""
Lección 10 - Equipo 3: Reparación de presupuesto
==================================
NUEVO EN ESTE PASO: reparar_presupuesto() y una auditoría antes/después.

CAMBIOS RESPECTO A equipo_02_plantillas_aleatorias.py
Introdúcelos en este orden:
    1. reparar_presupuesto() reemplazar jugadores costosos dentro del mismo bloque de rol
    2. la auditoria          probar cambios de precio sin duplicados ni desviación de rol

Ejecútalo:  python equipo_03_reparacion_presupuesto.py

La reparación hace asequibles a 1,000/1,000 equipos y preserva la unicidad. El precio se impone sin romper los bloques de rol.
"""
from pathlib import Path
import random
from typing import List, Sequence, Tuple

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
        ``reparar`` solo sustituye dentro del mismo bloque: un GK no
        paga el presupuesto de un delantero.
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
        1,000 reparaciones caben en €600m sin salir de este grupo.
    """
    columnas = ["sofifa_id", "short_name", "overall", "value_eur", "player_positions"]
    frame = pd.read_csv(ruta, usecols=columnas).dropna(subset=["value_eur", "player_positions"])
    frame["role"] = frame.player_positions.map(rol_principal)
    return pd.concat([frame[frame.role == rol].sort_values(
        ["overall", "value_eur"], ascending=[False, True]).head(GRUPO_POR_ROL)
        for rol in CONTEOS_ROLES], ignore_index=True)


def equipo_aleatorio(grupo: pd.DataFrame) -> List[int]:
    """Once distintos, un bloque legal. El precio suele pasarse de €600m.

    Args:
        grupo: los 48 candidatos.

    Returns:
        Once índices distintos.

    Example:
        Antes de reparar, solo una fracción cabe. Después, 1,000/1,000.
    """
    equipo: List[int] = []
    for rol, conteo in CONTEOS_ROLES.items():
        equipo.extend(random.sample(grupo.index[grupo.role == rol].tolist(), conteo))
    return equipo


def metricas_equipo(grupo: pd.DataFrame, equipo: Sequence[int]) -> Tuple[int, int]:
    """Habilidad y precio. ``reparar`` mira el segundo; la unicidad, el set.

    Args:
        grupo: candidatos.
        equipo: once índices.

    Returns:
        ``(overall total, valor_eur total)``.

    Example:
        Tras reparar, todos los valores son ≤ 600m y los once siguen
        distintos.
    """
    elegido = grupo.loc[list(equipo)]
    return int(elegido.overall.sum()), int(elegido.value_eur.sum())


# --- NUEVO (1) reparar_presupuesto() ------------------------------------------
def reparar_presupuesto(grupo: pd.DataFrame, fuente: Sequence[int]) -> List[int]:
    """Sustituye al más caro por uno más barato del mismo rol, minimizando pérdida/ahorro.

    No toca la unicidad: el reemplazo se elige fuera del once. No toca
    los bloques: un DEF solo se cambia por otro DEF.

    Args:
        grupo: los 48 candidatos.
        fuente: plantilla, a menudo por encima de €600m.

    Returns:
        Once índices distintos con valor ≤ 600m.

    Example:
        1,000/1,000 asequibles, 1,000/1,000 con once distintos.
    """
    equipo = list(fuente)
    while metricas_equipo(grupo, equipo)[1] > PRESUPUESTO_EUR:
        opciones = []
        for espacio, actual in enumerate(equipo):
            rol = grupo.loc[actual, "role"]
            for reemplazo in grupo.index[(grupo.role == rol) & (~grupo.index.isin(equipo))]:
                ahorro = int(grupo.loc[actual, "value_eur"] - grupo.loc[reemplazo, "value_eur"])
                perdida_habilidad = int(grupo.loc[actual, "overall"] - grupo.loc[reemplazo, "overall"])
                if ahorro > 0:
                    opciones.append((perdida_habilidad / ahorro, -ahorro, espacio, reemplazo))
        if not opciones:
            raise RuntimeError("El grupo de candidatos no puede satisfacer el presupuesto")
        _, _, espacio, reemplazo = min(opciones)
        equipo[espacio] = int(reemplazo)
    return equipo
# ------------------------------------------------------------------------------


# --- NUEVO (2) la auditoria ---------------------------------------------------
random.seed(SEMILLA)
grupo = cargar_grupo()
crudas = [equipo_aleatorio(grupo) for _ in range(TAMANO_POBLACION)]
reparadas = [reparar_presupuesto(grupo, equipo) for equipo in crudas]
metricas_crudas = [metricas_equipo(grupo, equipo) for equipo in crudas]
metricas_reparadas = [metricas_equipo(grupo, equipo) for equipo in reparadas]
# ------------------------------------------------------------------------------

print("Lección 10 - Equipo 3: reparación de presupuesto")
print(f"Dentro del presupuesto antes: {sum(costo <= PRESUPUESTO_EUR for _, costo in metricas_crudas)}/{TAMANO_POBLACION}")
print(f"Dentro del presupuesto después:  {sum(costo <= PRESUPUESTO_EUR for _, costo in metricas_reparadas)}/{TAMANO_POBLACION}")
print(f"Distintos después de reparación: {sum(len(set(equipo)) == 11 for equipo in reparadas)}/{TAMANO_POBLACION}")
print(f"Cambio medio en habilidad: {sum(a[0] - b[0] for a, b in zip(metricas_reparadas, metricas_crudas)) / TAMANO_POBLACION:+.2f}")

FIGURAS.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter([costo / 1e6 for _, costo in metricas_crudas], [habilidad for habilidad, _ in metricas_crudas], alpha=0.2, label="antes")
ax.scatter([costo / 1e6 for _, costo in metricas_reparadas], [habilidad for habilidad, _ in metricas_reparadas], alpha=0.2, label="después")
ax.axvline(PRESUPUESTO_EUR / 1e6, color="tab:red", label="presupuesto")
ax.set(xlabel="valor del equipo (€m)", ylabel="calificación general total", title="La reparación impone el precio mientras preserva espacios por rol")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "equipo_03_reparacion_presupuesto.png", dpi=160)
plt.close(fig)

