"""
Lección 10 - Equipo 5: La búsqueda completa de plantilla
===========================================
NUEVO EN ESTE PASO: ejecutar(), una línea base de igual presupuesto, y la plantilla seleccionada.

CAMBIOS RESPECTO A equipo_04_operadores_ordenados.py
Introdúcelos en este orden:
    1. ejecutar()            repetir cruza consciente del rol, mutación, reparación y elitismo
    2. la linea base         comparar contra el mismo número de equipos aleatorios reparados
    3. la plantilla          reportar nombres, roles, habilidad y precio de la respuesta final

Ejecútalo:  python equipo_05_la_busqueda_completa.py

El AG alcanza habilidad 991 con €597.5m y empata a la mejor plantilla aleatoria reparada a iguales evaluaciones. El empate es la lección, no una victoria a reescribir.
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
GENERACIONES = 50
GENERAL: List[int] = []
VALOR: List[int] = []
ROL: List[str] = []
INDICE_ROL: dict[str, List[int]] = {}


def rol_principal(posiciones: str) -> str:
    """Primera posición FIFA → GK/DEF/MID/FWD.

    Args:
        posiciones: campo del CSV.

    Returns:
        El bloque de rol.

    Example:
        La plantilla final se imprime con estos roles junto a overall y
        precio.
    """
    primera = posiciones.split(",")[0].strip()
    if primera == "GK": return "GK"
    if primera in {"CB", "LB", "RB", "LWB", "RWB"}: return "DEF"
    if primera in {"CM", "CDM", "CAM", "LM", "RM"}: return "MID"
    return "FWD"


def cargar_grupo(ruta: Path = DATOS) -> pd.DataFrame:
    """48 candidatos. También llena los arreglos GENERAL, VALOR, ROL e INDICE_ROL.

    Args:
        ruta: ``players_20.csv``.

    Returns:
        DataFrame de 48 filas. Los arreglos globales aceleran las 5,100
        evaluaciones.

    Example:
        AG 991 / €597.5m contra la mejor aleatoria reparada al mismo
        presupuesto de evaluaciones.
    """
    cols = ["sofifa_id", "short_name", "overall", "value_eur", "player_positions"]
    frame = pd.read_csv(ruta, usecols=cols).dropna(subset=["value_eur", "player_positions"])
    frame["role"] = frame.player_positions.map(rol_principal)
    grupo = pd.concat([frame[frame.role == rol].sort_values(["overall", "value_eur"], ascending=[False, True]).head(GRUPO_POR_ROL)
                      for rol in CONTEOS_ROLES], ignore_index=True)
    GENERAL[:] = [int(valor) for valor in grupo.overall]
    VALOR[:] = [int(valor) for valor in grupo.value_eur]
    ROL[:] = grupo.role.tolist()
    INDICE_ROL.clear()
    INDICE_ROL.update({rol: [i for i, rol_item in enumerate(ROL) if rol_item == rol]
                       for rol in CONTEOS_ROLES})
    return grupo


def equipo_aleatorio(grupo: pd.DataFrame) -> List[int]:
    """Once distintos. AG y línea base sortean desde aquí.

    Args:
        grupo: candidatos (el cuerpo usa INDICE_ROL).

    Returns:
        Once índices.

    Example:
        5,100 aleatorios reparados empatan al AG en habilidad 991.
    """
    equipo: List[int] = []
    for rol, conteo in CONTEOS_ROLES.items():
        equipo.extend(random.sample(INDICE_ROL[rol], conteo))
    return equipo


def metricas_equipo(grupo: pd.DataFrame, equipo: Sequence[int]) -> Tuple[int, int]:
    """Habilidad y precio desde los arreglos, no desde el DataFrame.

    Args:
        grupo: no se lee; está para no cambiar la firma.
        equipo: once índices.

    Returns:
        ``(overall total, valor_eur total)``.

    Example:
        AG: 991 y €597.5m. Línea base: misma habilidad a iguales
        evaluaciones.
    """
    return sum(GENERAL[i] for i in equipo), sum(VALOR[i] for i in equipo)


def reparar_presupuesto(grupo: pd.DataFrame, fuente: Sequence[int]) -> List[int]:
    """Vuelve a €600m. AG y línea base usan la misma regla.

    Args:
        grupo: no se lee; la firma se conserva.
        fuente: plantilla, posiblemente cara.

    Returns:
        Once índices asequibles.

    Example:
        El empate 991 vs 991 no es "el AG ignora el presupuesto".
    """
    equipo = list(fuente)
    while metricas_equipo(grupo, equipo)[1] > PRESUPUESTO_EUR:
        opciones = []
        for espacio, actual in enumerate(equipo):
            for reemplazo in INDICE_ROL[ROL[actual]]:
                if reemplazo in equipo:
                    continue
                ahorro = VALOR[actual] - VALOR[reemplazo]
                perdida = GENERAL[actual] - GENERAL[reemplazo]
                if ahorro > 0:
                    opciones.append((perdida / ahorro, -ahorro, espacio, reemplazo))
        _, _, espacio, reemplazo = min(opciones); equipo[espacio] = int(reemplazo)
    return equipo


def cruza_roles(grupo: pd.DataFrame, primero: Sequence[int], segundo: Sequence[int]) -> List[int]:
    """Recombina por bloque de rol.

    Args:
        grupo: no se lee.
        primero: padre.
        segundo: padre.

    Returns:
        Once índices, un patrón de roles legal.

    Example:
        50 generaciones de esta cruza empatan a 5,100 aleatorios
        reparados.
    """
    hijo: List[int] = []; offset = 0
    for rol, conteo in CONTEOS_ROLES.items():
        candidatos = list(dict.fromkeys(list(primero[offset:offset + conteo]) + list(segundo[offset:offset + conteo])))
        candidatos += [i for i in INDICE_ROL[rol] if i not in candidatos]
        hijo.extend(candidatos[:conteo]); offset += conteo
    return hijo


def mutar(grupo: pd.DataFrame, fuente: Sequence[int]) -> List[int]:
    """Sustituye a un jugador por otro del mismo rol, probabilidad 0.3 por hijo.

    Args:
        grupo: no se lee.
        fuente: plantilla de 11.

    Returns:
        Once índices distintos.

    Example:
        No basta para despegarse de la línea base: habilidad 991 vs 991.
    """
    equipo = list(fuente); espacio = random.randrange(len(equipo)); rol = ROL[equipo[espacio]]
    equipo[espacio] = random.choice([i for i in INDICE_ROL[rol] if i not in equipo]); return equipo


# --- NUEVO (1) ejecutar() -----------------------------------------------------
def ejecutar(grupo: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """50 generaciones, población 100, SEMILLA = 3.

    Args:
        grupo: candidatos (y arreglos globales ya llenos).

    Returns:
        ``(mejor plantilla, historia de overall)``.

    Example:
        Habilidad 991, €597.5m, empate con la mejor aleatoria reparada
        a 5,100 evaluaciones.
    """
    random.seed(SEMILLA); poblacion = [reparar_presupuesto(grupo, equipo_aleatorio(grupo)) for _ in range(TAMANO_POBLACION)]
    mejor = max(poblacion, key=lambda t: metricas_equipo(grupo, t)[0]); historial = [metricas_equipo(grupo, mejor)[0]]
    for _ in range(GENERACIONES):
        hijos = [mejor]
        while len(hijos) < TAMANO_POBLACION:
            seleccionar = lambda: max(random.sample(poblacion, 3), key=lambda t: metricas_equipo(grupo, t)[0])
            hijo = cruza_roles(grupo, seleccionar(), seleccionar())
            if random.random() < 0.3: hijo = mutar(grupo, hijo)
            hijos.append(reparar_presupuesto(grupo, hijo))
        poblacion = hijos; mejor = max(poblacion, key=lambda t: metricas_equipo(grupo, t)[0]); historial.append(metricas_equipo(grupo, mejor)[0])
    return mejor, historial
# ------------------------------------------------------------------------------


# --- NUEVO (2) la linea base --------------------------------------------------
grupo = cargar_grupo(); mejor, historial = ejecutar(grupo); evaluaciones = TAMANO_POBLACION * (GENERACIONES + 1)
random.seed(SEMILLA + 1); linea_base = max((reparar_presupuesto(grupo, equipo_aleatorio(grupo)) for _ in range(evaluaciones)), key=lambda t: metricas_equipo(grupo, t)[0])
# ------------------------------------------------------------------------------


# --- NUEVO (3) la plantilla ---------------------------------------------------
habilidad, costo = metricas_equipo(grupo, mejor); habilidad_linea_base, _ = metricas_equipo(grupo, linea_base)
seleccionados = grupo.loc[mejor, ["short_name", "role", "overall", "value_eur"]]
# ------------------------------------------------------------------------------

print("Lección 10 - Equipo 5: la búsqueda completa de plantilla")
print(f"Habilidad AG={habilidad}, costo=€{costo / 1e6:.1f}m; habilidad aleatoria mismo presupuesto={habilidad_linea_base}")
print(f"Mejora del AG sobre aleatorio: {habilidad - habilidad_linea_base:+d} puntos de calificación")
print(f"Jugadores distintos: {len(set(mejor))}/11; dentro del presupuesto: {costo <= PRESUPUESTO_EUR}")
print(seleccionados.to_string(index=False))

FIGURAS.mkdir(exist_ok=True); fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(historial, label="mejor habilidad AG"); ax.axhline(habilidad_linea_base, color="black", linestyle="--", label="mejor aleatorio mismo presupuesto")
ax.set(xlabel="generación", ylabel="calificación general total", title="Búsqueda de plantilla contra una línea base de iguales evaluaciones"); ax.legend()
fig.tight_layout(); fig.savefig(FIGURAS / "equipo_05_la_busqueda_completa.png", dpi=160); plt.close(fig)

