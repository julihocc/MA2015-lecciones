"""
Lección 10 - Equipo 1: Once espacios ordenados
=========================================
NUEVO EN ESTE PASO: cargar_grupo(), bloques de rol, y un grupo de candidatos respaldado por la fuente.

El cromosoma es una lista ordenada de once identificadores distintos de jugadores. El orden
de los espacios codifica un portero, cuatro defensas, tres mediocampistas y tres
delanteros, así que tanto la pertenencia como la posición deben sobrevivir a todo operador.

Ejecútalo:  python equipo_01_la_plantilla.py

El grupo tiene 12 candidatos por bloque de rol y un presupuesto de €600m. Once espacios ordenados: pertenencia y posición tienen que sobrevivir a todo operador.
"""
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

SEMILLA = 3
DATOS = Path(__file__).with_name("players_20.csv")
FIGURAS = Path(__file__).resolve().parent.parent / "figures"
CONTEOS_ROLES = {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}
GRUPO_POR_ROL = 12
PRESUPUESTO_EUR = 600_000_000


def rol_principal(posiciones: str) -> str:
    """Traduce la primera posición FIFA a un bloque: GK, DEF, MID o FWD.

    El cromosoma no entiende "CB, LB": entiende cuatro bloques de tamaño
    fijo. Esta función es el contrato entre el CSV y esos bloques.

    Args:
        posiciones: campo ``player_positions`` del CSV.

    Returns:
        Una de ``GK``, ``DEF``, ``MID``, ``FWD``.

    Example:
        12 candidatos por bloque. Un CB y un LB caen en DEF: el espacio
        de defensa no distingue laterales.
    """
    primera = posiciones.split(",")[0].strip()
    if primera == "GK":
        return "GK"
    if primera in {"CB", "LB", "RB", "LWB", "RWB"}:
        return "DEF"
    if primera in {"CM", "CDM", "CAM", "LM", "RM"}:
        return "MID"
    return "FWD"


def cargar_grupo(ruta: Path = DATOS) -> pd.DataFrame:
    """Los 12 mejores por rol, ordenados por overall y luego por precio.

    Args:
        ruta: ``players_20.csv`` junto al script.

    Returns:
        DataFrame de 48 candidatos (12 × 4 roles) con columna ``role``.

    Example:
        Presupuesto €600m. El grupo contiene el trueque precio-habilidad
        que el paso 2 va a chocar contra ese techo.
    """
    columnas = ["sofifa_id", "short_name", "overall", "value_eur", "player_positions"]
    frame = pd.read_csv(ruta, usecols=columnas).dropna(subset=["value_eur", "player_positions"])
    frame["role"] = frame["player_positions"].map(rol_principal)
    piezas = []
    for rol in CONTEOS_ROLES:
        grupo = frame[frame["role"] == rol].sort_values(
            ["overall", "value_eur"], ascending=[False, True]
        ).head(GRUPO_POR_ROL)
        piezas.append(grupo)
    return pd.concat(piezas, ignore_index=True)


grupo = cargar_grupo()
resumen: Dict[str, str] = {}
for rol in CONTEOS_ROLES:
    subgrupo = grupo[grupo.role == rol]
    resumen[rol] = (f"{len(subgrupo)} candidatos, general {subgrupo.overall.min()}..{subgrupo.overall.max()}, "
                     f"valor €{subgrupo.value_eur.min() / 1e6:.1f}m..€{subgrupo.value_eur.max() / 1e6:.1f}m")

print("Lección 10 - Equipo 1: once espacios ordenados")
print(f"Espacios en la plantilla: {CONTEOS_ROLES}; total={sum(CONTEOS_ROLES.values())}")
print(f"Presupuesto: €{PRESUPUESTO_EUR / 1e6:.0f}m")
for rol, texto in resumen.items():
    print(f"{rol}: {texto}")

FIGURAS.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5))
for rol in CONTEOS_ROLES:
    subgrupo = grupo[grupo.role == rol]
    ax.scatter(subgrupo.value_eur / 1e6, subgrupo.overall, label=rol, s=55)
ax.set(xlabel="valor de mercado (€m)", ylabel="calificación general",
       title="El grupo de candidatos contiene una compensación precio-habilidad")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "equipo_01_la_plantilla.png", dpi=160)
plt.close(fig)

