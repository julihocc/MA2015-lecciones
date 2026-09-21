"""
Lección 12 - Paso 2: Encontrando el estancamiento
======================================
NUEVO EN ESTE PASO: promedio(), esta_mejorando(), la tendencia de estancamiento, sombrear_estancamientos().

El paso 1 gastó todo su presupuesto en una sola configuración. Este paso pregunta a dónde fue ese presupuesto.
El algoritmo no puede ver qué tan lejos está del óptimo -- no tiene óptimo para comparar -- pero puede
ver su propia historia reciente, y eso es suficiente para distinguir una generación que todavía está haciendo
progreso de una que no. La señal es la del libro: compare la media de la población de esta generación
con la media de las diez anteriores. Si no ha mejorado al menos en una décima de porcentaje, la generación está estancada.

La señal es deliberadamente cruda y es el único sensor que el algoritmo adaptativo de los pasos 3 y 4 tendrá.

CAMBIOS RESPECTO A ag_adaptativo_01_parametros_fijos.py
Introdúzcalos en este orden:
    1. promedio()                la media móvil de una serie, sobre una ventana
    2. esta_mejorando()          la señal de estancamiento, un booleano por generación
    3. tendencia_estancamiento   ejecutar() registra ese booleano junto con sus otras historias
    4. sombrear_estancamientos() pinta las generaciones estancadas en una gráfica

Ejecútalo:  python ag_adaptativo_02_el_estancamiento.py

La señal llama estancadas a 14 de 99 generaciones, la primera en la 74; el 14% del presupuesto se gasta ahí. La primera mitad del presupuesto compra el 85% de la mejora total.
"""
from math import dist, isnan, nan
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt

SEMILLA = 1
DATOS = Path(__file__).with_name("att48_xy.txt")
FIGURAS = Path(__file__).resolve().parent.parent / "figures"

PRESUPUESTO = 12_000            # evaluaciones de ruta, la única moneda que es justa
TAMANO_POBLACION = 120
PROBABILIDAD_CRUZA = 0.9
PROBABILIDAD_MUTACION = 0.25
TAMANO_TORNEO = 3
PERIODO_TENDENCIA = 10          # cuántas generaciones pasadas mira la señal
BRECHA_TENDENCIA = 0.001        # la mejora que demanda: una décima de un porcentaje

Ruta = List[int]
Puntuado = Tuple[Ruta, float]


def cargar_puntos(ruta_archivo: Path = DATOS) -> List[Tuple[int, int]]:
    """Las 48 capitales, como coordenadas enteras, una ciudad por línea."""
    return [tuple(map(int, linea.split())) for linea in ruta_archivo.read_text().splitlines() if linea.strip()]


def longitud_ruta(puntos: Sequence[Tuple[int, int]], ruta: Sequence[int]) -> float:
    """Longitud del recorrido cerrado. Una llamada a esta función es una evaluación."""
    return sum(dist(puntos[a], puntos[b]) for a, b in zip(ruta, tuple(ruta[1:]) + tuple(ruta[:1])))


def ruta_aleatoria(tamano: int) -> Ruta:
    ruta = list(range(tamano))
    random.shuffle(ruta)
    return ruta


def vecino_mas_cercano(puntos: Sequence[Tuple[int, int]], inicio: int = 0) -> Ruta:
    """La línea base constructiva barata de la Lección 10; el AG tiene que responder a ella."""
    ruta, no_vistos = [inicio], set(range(len(puntos))) - {inicio}
    while no_vistos:
        actual = ruta[-1]
        mas_cercano = min(no_vistos, key=lambda ciudad: dist(puntos[actual], puntos[ciudad]))
        ruta.append(mas_cercano)
        no_vistos.remove(mas_cercano)
    return ruta


def cruza_ordenada(primero: Sequence[int], segundo: Sequence[int]) -> Ruta:
    """Conserva un fragmento de un padre y llena los huecos en el orden del otro padre."""
    izq, der = sorted(random.sample(range(len(primero)), 2))
    hijo = [-1] * len(primero)
    hijo[izq:der] = primero[izq:der]
    restantes = [ciudad for ciudad in segundo if ciudad not in hijo]
    huecos = list(range(der, len(primero))) + list(range(izq))
    for indice, ciudad in zip(huecos, restantes):
        hijo[indice] = ciudad
    return hijo


def mutacion_inversion(fuente: Sequence[int]) -> Ruta:
    """Invierte un segmento: una permutación legal sigue siendo una permutación legal."""
    ruta = list(fuente)
    izq, der = sorted(random.sample(range(len(ruta)), 2))
    ruta[izq:der] = reversed(ruta[izq:der])
    return ruta


def seleccion_torneo(poblacion: List[Puntuado]) -> Puntuado:
    return min(random.sample(poblacion, TAMANO_TORNEO), key=lambda puntuado: puntuado[1])


# --- NUEVO (1) promedio() --------------------------------------------------------
def promedio(serie: Sequence[float], periodo: int) -> float:
    """Media de los valores de `periodo` antes del último, o nan si no hay suficientes.

    El último valor se excluye a propósito: quien llama lo compara contra este.

    Args:
        serie: medias de población, una por generación.
        periodo: 10.

    Returns:
        Media móvil, o nan al inicio.

    Example:
        14 de 99 generaciones caen debajo del umbral. El sensor no ve
        el óptimo: solo se ve a sí mismo.
    """
    if len(serie) < periodo + 1:
        return nan
    return sum(serie[-periodo - 1:-1]) / periodo
# ------------------------------------------------------------------------------


# --- NUEVO (2) esta_mejorando() ---------------------------------------------------
def esta_mejorando(serie: Sequence[float], periodo: int = PERIODO_TENDENCIA, brecha: float = BRECHA_TENDENCIA) -> bool:
    """Verdadero mientras la serie sigue cayendo más rápido que `brecha` contra su propio pasado.

    La longitud de la ruta se minimiza, así que mejora significa que el valor más nuevo está DEBAJO del
    promedio móvil. Antes de que haya suficiente historia la respuesta es Verdadero: una ejecución
    que acaba de empezar no está estancada, es joven.

    Args:
        serie: medias de población.
        periodo: 10.
        brecha: 0.001 (una décima de porcentaje).

    Returns:
        True si sigue mejorando.

    Example:
        Primera generación estancada: 74. 14 de 99. El 85% de la mejora
        ya ocurrió en la primera mitad del presupuesto.
    """
    referencia = promedio(serie, periodo)
    if isnan(referencia) or referencia == 0:
        return True
    return serie[-1] < referencia * (1 - brecha)
# ------------------------------------------------------------------------------


def ejecutar(puntos: Sequence[Tuple[int, int]], semilla: int) -> Dict[str, object]:
    """Una ejecución del AG de parámetros fijos, detenida por el presupuesto de evaluaciones.

    Solo se inicia una generación si el presupuesto puede pagarla completa, para que ninguna ejecución
    se corte a la mitad y cada ejecución reporte las evaluaciones que realmente gastó.
    """
    random.seed(semilla)
    evaluaciones = 0
    poblacion: List[Puntuado] = []
    for _ in range(TAMANO_POBLACION):
        ruta = ruta_aleatoria(len(puntos))
        poblacion.append((ruta, longitud_ruta(puntos, ruta)))
        evaluaciones += 1

    mejor = min(poblacion, key=lambda puntuado: puntuado[1])
    historial_mejores = [mejor[1]]
    historial_medias = [sum(puntuado[1] for puntuado in poblacion) / len(poblacion)]
    historial_gastado = [evaluaciones]
    # --- NUEVO (3) tendencia_estancamiento --------------------------------------------------
    tendencia_estancamiento: List[bool] = []       # una entrada por generación: Verdadero mientras mejora
    # --------------------------------------------------------------------------

    while evaluaciones + len(poblacion) - 1 <= PRESUPUESTO:
        hijos = [mejor]                                  # elitismo: trasladado, no reevaluado
        while len(hijos) < len(poblacion):
            padre_uno, padre_dos = seleccion_torneo(poblacion), seleccion_torneo(poblacion)
            if random.random() < PROBABILIDAD_CRUZA:
                ruta = cruza_ordenada(padre_uno[0], padre_dos[0])
            else:
                ruta = list(padre_uno[0])
            if random.random() < PROBABILIDAD_MUTACION:
                ruta = mutacion_inversion(ruta)
            hijos.append((ruta, longitud_ruta(puntos, ruta)))
            evaluaciones += 1
        poblacion = hijos
        mejor = min(poblacion, key=lambda puntuado: puntuado[1])
        historial_mejores.append(mejor[1])
        historial_medias.append(sum(puntuado[1] for puntuado in poblacion) / len(poblacion))
        historial_gastado.append(evaluaciones)
        tendencia_estancamiento.append(esta_mejorando(historial_medias))     # (3) leer el sensor cada generación

    return {
        "ruta": mejor[0],
        "longitud": mejor[1],
        "historial_mejores": historial_mejores,
        "historial_medias": historial_medias,
        "historial_gastado": historial_gastado,
        "tendencia_estancamiento": tendencia_estancamiento,
        "evaluaciones": evaluaciones,
        "generaciones": len(historial_mejores) - 1,
    }


# --- NUEVO (4) sombrear_estancamientos() ---------------------------------------------------
def sombrear_estancamientos(eje: plt.Axes, tendencia_estancamiento: Sequence[bool]) -> None:
    """Pintar una banda roja sobre cada generación que la señal llama estancada.

    Args:
        eje: Axes de matplotlib.
        tendencia_estancamiento: un booleano por generación.

    Returns:
        None.

    Example:
        14 bandas rojas, la primera en la generación 74.
    """
    for indice, mejorando in enumerate(tendencia_estancamiento):
        if not mejorando:
            eje.axvspan(indice + 1, indice + 2, color="red", alpha=0.12, linewidth=0)
# ------------------------------------------------------------------------------


puntos = cargar_puntos()
linea_base = longitud_ruta(puntos, vecino_mas_cercano(puntos))
registro = ejecutar(puntos, SEMILLA)

tendencia = registro["tendencia_estancamiento"]
historial_mejores = registro["historial_mejores"]
historial_gastado = registro["historial_gastado"]
estancados = [indice for indice, mejorando in enumerate(tendencia) if not mejorando]
calentamiento = min(PERIODO_TENDENCIA, len(tendencia))          # generaciones antes de que la señal tenga historia
por_generacion = [historial_gastado[i + 1] - historial_gastado[i] for i in range(len(tendencia))]
evaluaciones_estancadas = sum(por_generacion[i] for i in estancados)

mitad = registro["evaluaciones"] / 2
cruce = next(i for i, gastado in enumerate(historial_gastado) if gastado >= mitad)
ganancia_total = historial_mejores[0] - historial_mejores[-1]
ganancia_primera_mitad = historial_mejores[0] - historial_mejores[cruce]

print("Lección 12 - Paso 2: encontrando el estancamiento")
print(f"Mejor ruta AG (semilla {SEMILLA}):    {registro['longitud']:,.0f}  ({registro['evaluaciones']:,} evaluaciones)")
print(f"Calentamiento de la señal:          las primeras {calentamiento} generaciones responden Verdadero por construcción")
print(f"Generaciones estancadas:            {len(estancados)} de {len(tendencia)}")
print(f"Primer estancamiento:               generación {estancados[0] + 1}" if estancados else "Primer estancamiento: ninguno")
print(f"Evaluaciones gastadas estancado:    {evaluaciones_estancadas:,} de {registro['evaluaciones']:,} "
      f"({evaluaciones_estancadas / registro['evaluaciones']:.0%})")
print(f"Mejora total:                       {ganancia_total:,.0f} de longitud de ruta")
print(f"Comprado por la primera mitad:      {ganancia_primera_mitad:,.0f} ({ganancia_primera_mitad / ganancia_total:.0%} de eso) "
      f"en {historial_gastado[cruce]:,} evaluaciones")
print(f"Comprado por la segunda mitad:      {ganancia_total - ganancia_primera_mitad:,.0f} "
      f"({1 - ganancia_primera_mitad / ganancia_total:.0%})")
print("La configuración que compró la primera mitad sigue vigente para la segunda.")

FIGURAS.mkdir(exist_ok=True)
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
ejes[0].plot(registro["historial_medias"], label="media de población")
ejes[0].plot(historial_mejores, label="mejor de la ejecución")
sombrear_estancamientos(ejes[0], tendencia)
ejes[0].axhline(linea_base, color="black", linestyle="--", label="vecino más cercano")
ejes[0].set(xlabel="generación", ylabel="longitud de ruta",
            title=f"Generaciones estancadas sombreadas ({len(estancados)} de {len(tendencia)})")
ejes[0].legend()
ejes[1].plot(historial_gastado, historial_mejores)
ejes[1].axvline(historial_gastado[cruce], color="grey", linestyle=":")
ejes[1].annotate(f"la mitad del presupuesto\nha comprado {ganancia_primera_mitad / ganancia_total:.0%}\nde la mejora",
                 xy=(historial_gastado[cruce], historial_mejores[cruce]),
                 xytext=(0.45, 0.65), textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color="grey"))
ejes[1].set(xlabel="evaluaciones gastadas", ylabel="mejor longitud de ruta", title="Lo que compra el presupuesto")
fig.tight_layout()
fig.savefig(FIGURAS / "ag_adaptativo_02_el_estancamiento.png", dpi=160)
plt.close(fig)

