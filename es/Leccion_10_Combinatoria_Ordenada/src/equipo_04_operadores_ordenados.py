"""
Lección 10 - Equipo 4: Operadores ordenados
======================================
NUEVO EN ESTE PASO: cruza_roles(), mutar(), y una generación seleccionada.

CAMBIOS RESPECTO A equipo_03_reparacion_presupuesto.py
Introdúcelos en este orden:
    1. cruza_roles()         recombinar dentro de los bloques sin jugadores duplicados
    2. mutar()               reemplazar a un jugador con un jugador no utilizado del mismo rol
    3. una generacion        medir la calidad mientras la reparación preserva el presupuesto

Ejecútalo:  python equipo_04_operadores_ordenados.py

Los 100 hijos siguen siendo legales. La habilidad media sube de 979.6 a 985.3: los operadores conscientes del rol mueven la calidad sin romper la plantilla.
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
TAMANO_POBLACION = 100


def rol_principal(posiciones: str) -> str:
    """Primera posición FIFA → GK/DEF/MID/FWD.

    Args:
        posiciones: campo del CSV.

    Returns:
        El bloque de rol.

    Example:
        ``cruza_roles`` recombina dentro de estos bloques.
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
        100 padres y 100 hijos se miden sobre este grupo.
    """
    cols = ["sofifa_id", "short_name", "overall", "value_eur", "player_positions"]
    frame = pd.read_csv(ruta, usecols=cols).dropna(subset=["value_eur", "player_positions"])
    frame["role"] = frame.player_positions.map(rol_principal)
    return pd.concat([frame[frame.role == rol].sort_values(["overall", "value_eur"], ascending=[False, True]).head(GRUPO_POR_ROL)
                      for rol in CONTEOS_ROLES], ignore_index=True)


def equipo_aleatorio(grupo: pd.DataFrame) -> List[int]:
    """Once distintos, un bloque legal. Se repara antes de seleccionar.

    Args:
        grupo: los 48 candidatos.

    Returns:
        Once índices.

    Example:
        100 padres reparados, habilidad media 979.6.
    """
    equipo: List[int] = []
    for rol, conteo in CONTEOS_ROLES.items(): equipo.extend(random.sample(grupo.index[grupo.role == rol].tolist(), conteo))
    return equipo


def metricas_equipo(grupo: pd.DataFrame, equipo: Sequence[int]) -> Tuple[int, int]:
    """Habilidad y precio. El torneo maximiza la primera; reparar recorta la segunda.

    Args:
        grupo: candidatos.
        equipo: once índices.

    Returns:
        ``(overall total, valor_eur total)``.

    Example:
        Habilidad media 979.6 → 985.3. Precio ≤ 600m en los 100 hijos.
    """
    elegido = grupo.loc[list(equipo)]; return int(elegido.overall.sum()), int(elegido.value_eur.sum())


def reparar_presupuesto(grupo: pd.DataFrame, fuente: Sequence[int]) -> List[int]:
    """Vuelve a €600m sin duplicar ni cambiar de rol.

    Args:
        grupo: candidatos.
        fuente: plantilla, posiblemente cara tras cruza o mutación.

    Returns:
        Once índices asequibles y distintos.

    Example:
        100/100 hijos legales (unicidad y presupuesto).
    """
    equipo = list(fuente)
    while metricas_equipo(grupo, equipo)[1] > PRESUPUESTO_EUR:
        opciones = []
        for espacio, actual in enumerate(equipo):
            rol = grupo.loc[actual, "role"]
            for reemplazo in grupo.index[(grupo.role == rol) & (~grupo.index.isin(equipo))]:
                ahorro = int(grupo.loc[actual, "value_eur"] - grupo.loc[reemplazo, "value_eur"])
                perdida = int(grupo.loc[actual, "overall"] - grupo.loc[reemplazo, "overall"])
                if ahorro > 0: opciones.append((perdida / ahorro, -ahorro, espacio, reemplazo))
        _, _, espacio, reemplazo = min(opciones); equipo[espacio] = int(reemplazo)
    return equipo


# --- NUEVO (1) cruza_roles() --------------------------------------------------
def cruza_roles(grupo: pd.DataFrame, primero: Sequence[int], segundo: Sequence[int]) -> List[int]:
    """Recombina dentro de cada bloque. Un corte global duplicaría un GK o perdería un DEF.

    Args:
        grupo: candidatos, para rellenar si faltan.
        primero: padre legal.
        segundo: padre legal.

    Returns:
        Once índices, un jugador por espacio, sin duplicados de rol.

    Example:
        Tras mutar y reparar, 100 hijos legales y habilidad media 985.3.
    """
    hijo: List[int] = []; offset = 0
    for rol, conteo in CONTEOS_ROLES.items():
        candidatos = list(dict.fromkeys(list(primero[offset:offset + conteo]) + list(segundo[offset:offset + conteo])))
        candidatos += [int(i) for i in grupo.index[grupo.role == rol] if i not in candidatos]
        hijo.extend(candidatos[:conteo]); offset += conteo
    return hijo
# ------------------------------------------------------------------------------


# --- NUEVO (2) mutar() --------------------------------------------------------
def mutar(grupo: pd.DataFrame, fuente: Sequence[int]) -> List[int]:
    """Sustituye a un jugador por otro no usado del mismo rol.

    Args:
        grupo: candidatos.
        fuente: plantilla de 11.

    Returns:
        Once índices distintos, mismo patrón de roles.

    Example:
        No introduce duplicados. ``reparar`` solo se ocupa del precio.
    """
    equipo = list(fuente); espacio = random.randrange(len(equipo)); rol = grupo.loc[equipo[espacio], "role"]
    opciones = [int(i) for i in grupo.index[grupo.role == rol] if i not in equipo]
    equipo[espacio] = random.choice(opciones); return equipo
# ------------------------------------------------------------------------------


# --- NUEVO (3) una generacion -------------------------------------------------
random.seed(SEMILLA); grupo = cargar_grupo()
padres = [reparar_presupuesto(grupo, equipo_aleatorio(grupo)) for _ in range(TAMANO_POBLACION)]
seleccionar = lambda: max(random.sample(padres, 3), key=lambda equipo: metricas_equipo(grupo, equipo)[0])
hijos = [reparar_presupuesto(grupo, mutar(grupo, cruza_roles(grupo, seleccionar(), seleccionar()))) for _ in range(TAMANO_POBLACION)]
# ------------------------------------------------------------------------------

print("Lección 10 - Equipo 4: operadores ordenados")
print(f"Hijos legales: {sum(len(set(t)) == 11 and metricas_equipo(grupo, t)[1] <= PRESUPUESTO_EUR for t in hijos)}/{TAMANO_POBLACION}")
print(f"Habilidad media padres: {sum(metricas_equipo(grupo, t)[0] for t in padres) / TAMANO_POBLACION:.1f}")
print(f"Habilidad media hijos:  {sum(metricas_equipo(grupo, t)[0] for t in hijos) / TAMANO_POBLACION:.1f}")

FIGURAS.mkdir(exist_ok=True); fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot([[metricas_equipo(grupo, t)[0] for t in padres], [metricas_equipo(grupo, t)[0] for t in hijos]], tick_labels=["padres", "hijos"])
ax.set(ylabel="calificación general total", title="Los operadores conscientes del rol preservan una plantilla ordenada legal")
fig.tight_layout(); fig.savefig(FIGURAS / "equipo_04_operadores_ordenados.png", dpi=160); plt.close(fig)

