"""
Lección 12 - Paso 3: Probabilidades que se mueven
============================================
NUEVO EN ESTE PASO: adaptar_probabilidades(), y un ejecutar() que puede usarlo.

El sensor del paso 2 dice, en cada generación, si la población sigue
mejorando. Este paso lo conecta a las dos probabilidades: cuando la ejecución se estanca,
súbalas -- más recombinación, más mutación, más alcance; mientras la ejecución
mejora, bájelas hacia un piso y deje que la selección refine lo que tiene.
La regla es la del libro, y es deliberadamente contundente: multiplicar por 1.1 cuando
se estanca, por 0.99 cuando mejora, y acotar.

Observe cuál de las dos ramas se dispara realmente. El script las cuenta, y en
esta semilla la respuesta no es la que la historia lleva a esperar: la rama estancada
nunca se ejecuta en absoluto, por lo que lo que se anuncia como adaptación se comporta como un
programa de decaimiento unidireccional. Aún así supera a la ejecución fija aquí -- lo cual dice más
sobre los valores fijos que sobre la regla.

El script ejecuta ambos regímenes en la misma semilla para que las dos curvas puedan ponerse lado
a lado. Una semilla es una anécdota, no evidencia -- la Lección 06 dijo por qué -- y
el paso 5 hace la medición.

CAMBIOS RESPECTO A ag_adaptativo_02_el_estancamiento.py
Introdúzcalos en este orden:
    1. constantes_adaptacion        los pisos, el techo y los dos multiplicadores
    2. adaptar_probabilidades()     la regla en sí, sobre dos flotantes simples
    3. ejecutar(adaptativo)         las constantes se vuelven variables por ejecución que la regla mueve

Ejecútalo:  python ag_adaptativo_03_probabilidades_adaptativas.py

Misma semilla, mismo presupuesto: adaptativo 48,830 contra fijo 57,076 (−14.4%). La rama estancada se dispara 0 de 99 veces. La cruza baja 0.90 → 0.33 y la mutación 0.25 → 0.09 y nunca suben.
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

# --- NUEVO (1) constantes_adaptacion ---------------------------------------------
MIN_PROBABILIDAD_CRUZA = 0.1       # nunca dejar de recombinar por completo
MIN_PROBABILIDAD_MUTACION = 0.05   # nunca dejar de inventar por completo
MAX_PROBABILIDAD = 1.0
FACTOR_ESTANCAMIENTO = 1.1         # estancado: abrir los operadores, rápido
FACTOR_MEJORA = 0.99               # mejorando: cerrarlos, lentamente
# ------------------------------------------------------------------------------

Ruta = List[int]
Puntuado = Tuple[Ruta, float]


def cargar_puntos(ruta_archivo: Path = DATOS) -> List[Tuple[int, int]]:
    """Las 48 capitales, coordenadas enteras, una ciudad por línea.

    Args:
        ruta_archivo: ``att48_xy.txt`` junto a este script.

    Returns:
        48 pares (x, y).
    """
    return [tuple(map(int, linea.split())) for linea in ruta_archivo.read_text().splitlines() if linea.strip()]


def longitud_ruta(puntos: Sequence[Tuple[int, int]], ruta: Sequence[int]) -> float:
    """Longitud del recorrido cerrado. Una llamada es una evaluación.

    Args:
        puntos: las 48 ciudades.
        ruta: permutación de 0..47.

    Returns:
        Distancia euclidiana del ciclo. El vecino más cercano da 40,526;
        la semilla 1 del paso 1 da 57,076.
    """
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
    """Conserva un fragmento de un padre y llena huecos en el orden del otro.

    Args:
        primero, segundo: permutaciones de las 48 ciudades.

    Returns:
        Un hijo legal: cada ciudad una vez. Un corte-y-copia ordinario
        duplicaría ciudades; este no.
    """
    izq, der = sorted(random.sample(range(len(primero)), 2))
    hijo = [-1] * len(primero)
    hijo[izq:der] = primero[izq:der]
    restantes = [ciudad for ciudad in segundo if ciudad not in hijo]
    huecos = list(range(der, len(primero))) + list(range(izq))
    for indice, ciudad in zip(huecos, restantes):
        hijo[indice] = ciudad
    return hijo


def mutacion_inversion(fuente: Sequence[int]) -> Ruta:
    """Invierte un segmento: una permutación legal sigue siendo legal.

    Args:
        fuente: una ruta.

    Returns:
        La misma ciudades, un tramo al revés.
    """
    ruta = list(fuente)
    izq, der = sorted(random.sample(range(len(ruta)), 2))
    ruta[izq:der] = reversed(ruta[izq:der])
    return ruta


def seleccion_torneo(poblacion: List[Puntuado]) -> Puntuado:
    return min(random.sample(poblacion, TAMANO_TORNEO), key=lambda puntuado: puntuado[1])


def promedio(serie: Sequence[float], periodo: int) -> float:
    """Media de los valores de `periodo` antes del último, o nan si no hay suficientes."""
    if len(serie) < periodo + 1:
        return nan
    return sum(serie[-periodo - 1:-1]) / periodo


def esta_mejorando(serie: Sequence[float], periodo: int = PERIODO_TENDENCIA, brecha: float = BRECHA_TENDENCIA) -> bool:
    """Verdadero mientras la serie sigue cayendo más rápido que `brecha` contra su propio pasado."""
    referencia = promedio(serie, periodo)
    if isnan(referencia) or referencia == 0:
        return True
    return serie[-1] < referencia * (1 - brecha)


# --- NUEVO (2) adaptar_probabilidades() --------------------------------------------
def adaptar_probabilidades(cruza: float, mutacion: float, mejorando: bool) -> Tuple[float, float]:
    """Mueve ambas probabilidades un escalón, en la dirección que apunta el sensor.

    Abrir es más rápido que cerrar (1.1 contra 0.99) a propósito: de un
    estancamiento se tiene que escapar rápido, mientras que una ejecución que mejora debe dejarse sola
    por el tiempo que siga mejorando.

    Args:
        cruza: probabilidad actual.
        mutacion: probabilidad actual.
        mejorando: lo que dijo el sensor.

    Returns:
        ``(cruza, mutacion)`` acotadas.

    Example:
        Rama estancada: 0 de 99. Cruza 0.90 → 0.33, mutación 0.25 → 0.09.
        Es un decaimiento, no adaptación.
    """
    if mejorando:
        return (max(cruza * FACTOR_MEJORA, MIN_PROBABILIDAD_CRUZA),
                max(mutacion * FACTOR_MEJORA, MIN_PROBABILIDAD_MUTACION))
    return (min(cruza * FACTOR_ESTANCAMIENTO, MAX_PROBABILIDAD),
            min(mutacion * FACTOR_ESTANCAMIENTO, MAX_PROBABILIDAD))
# ------------------------------------------------------------------------------


def ejecutar(puntos: Sequence[Tuple[int, int]], semilla: int, adaptativo: bool = False) -> Dict[str, object]:
    """Una ejecución, detenida por el presupuesto de evaluaciones; `adaptativo` enciende la regla."""
    random.seed(semilla)
    evaluaciones = 0
    # --- NUEVO (3) ejecutar(adaptativo) ------------------------------------------------
    probabilidad_cruza = PROBABILIDAD_CRUZA         # las constantes se vuelven estado:
    probabilidad_mutacion = PROBABILIDAD_MUTACION   # ambas ejecuciones empiezan donde lo hizo el paso 1
    historial_cruza = [probabilidad_cruza]
    historial_mutacion = [probabilidad_mutacion]
    # --------------------------------------------------------------------------
    poblacion: List[Puntuado] = []
    for _ in range(TAMANO_POBLACION):
        ruta = ruta_aleatoria(len(puntos))
        poblacion.append((ruta, longitud_ruta(puntos, ruta)))
        evaluaciones += 1

    mejor = min(poblacion, key=lambda puntuado: puntuado[1])
    historial_mejores = [mejor[1]]
    historial_medias = [sum(puntuado[1] for puntuado in poblacion) / len(poblacion)]
    historial_gastado = [evaluaciones]
    tendencia_estancamiento: List[bool] = []       # una entrada por generación: Verdadero mientras mejora

    while evaluaciones + len(poblacion) - 1 <= PRESUPUESTO:
        hijos = [mejor]                                  # elitismo: trasladado, no reevaluado
        while len(hijos) < len(poblacion):
            padre_uno, padre_dos = seleccion_torneo(poblacion), seleccion_torneo(poblacion)
            if random.random() < probabilidad_cruza:       # (3) la variable, no la constante
                ruta = cruza_ordenada(padre_uno[0], padre_dos[0])
            else:
                ruta = list(padre_uno[0])
            if random.random() < probabilidad_mutacion:    # (3) la variable, no la constante
                ruta = mutacion_inversion(ruta)
            hijos.append((ruta, longitud_ruta(puntos, ruta)))
            evaluaciones += 1
        poblacion = hijos
        mejor = min(poblacion, key=lambda puntuado: puntuado[1])
        historial_mejores.append(mejor[1])
        historial_medias.append(sum(puntuado[1] for puntuado in poblacion) / len(poblacion))
        historial_gastado.append(evaluaciones)
        mejorando = esta_mejorando(historial_medias)
        tendencia_estancamiento.append(mejorando)
        if adaptativo:                                       # (3) el sensor mueve las perillas
            probabilidad_cruza, probabilidad_mutacion = adaptar_probabilidades(
                probabilidad_cruza, probabilidad_mutacion, mejorando)
        historial_cruza.append(probabilidad_cruza)
        historial_mutacion.append(probabilidad_mutacion)

    return {
        "ruta": mejor[0],
        "longitud": mejor[1],
        "historial_mejores": historial_mejores,
        "historial_medias": historial_medias,
        "historial_gastado": historial_gastado,
        "tendencia_estancamiento": tendencia_estancamiento,
        "historial_cruza": historial_cruza,
        "historial_mutacion": historial_mutacion,
        "evaluaciones": evaluaciones,
        "generaciones": len(historial_mejores) - 1,
    }


def sombrear_estancamientos(eje: plt.Axes, tendencia_estancamiento: Sequence[bool]) -> None:
    """Pintar una banda roja sobre cada generación que la señal llama estancada."""
    for indice, mejorando in enumerate(tendencia_estancamiento):
        if not mejorando:
            eje.axvspan(indice + 1, indice + 2, color="red", alpha=0.12, linewidth=0)


puntos = cargar_puntos()
fijo = ejecutar(puntos, SEMILLA, adaptativo=False)
adaptativo = ejecutar(puntos, SEMILLA, adaptativo=True)

pista_cruza = adaptativo["historial_cruza"]
pista_mutacion = adaptativo["historial_mutacion"]
estancamientos = adaptativo["tendencia_estancamiento"].count(False)

print("Lección 12 - Paso 3: probabilidades que se mueven")
print(f"Ambas ejecuciones usan la semilla {SEMILLA} y el mismo presupuesto de {PRESUPUESTO:,} evaluaciones.")
print(f"Fija     : {fijo['longitud']:>9,.0f}   cruza {PROBABILIDAD_CRUZA:.2f} "
      f"mutación {PROBABILIDAD_MUTACION:.2f}   ({fijo['evaluaciones']:,} evaluaciones, "
      f"{fijo['generaciones']} generaciones)")
print(f"Adaptativa: {adaptativo['longitud']:>9,.0f}   cruza "
      f"{min(pista_cruza):.2f}-{max(pista_cruza):.2f} mutación "
      f"{min(pista_mutacion):.2f}-{max(pista_mutacion):.2f}   "
      f"({adaptativo['evaluaciones']:,} evaluaciones, {adaptativo['generaciones']} generaciones)")
print(f"Diferencia:{fijo['longitud'] - adaptativo['longitud']:>9,.0f} más corta para la ejecución adaptativa "
      f"({(adaptativo['longitud'] - fijo['longitud']) / fijo['longitud']:+.1%})")
print(f"Rama estancada se disparó:  {estancamientos:>3} de {len(adaptativo['tendencia_estancamiento'])} generaciones de la ejecución adaptativa")
print(f"Rama de mejora se disparó:  {len(adaptativo['tendencia_estancamiento']) - estancamientos:>3} de {len(adaptativo['tendencia_estancamiento'])}"
      f"   (la ejecución fija en la misma semilla se estancó {fijo['tendencia_estancamiento'].count(False)} veces)")
print(f"La cruza fue {PROBABILIDAD_CRUZA:.2f} -> {pista_cruza[-1]:.2f}, "
      f"mutación {PROBABILIDAD_MUTACION:.2f} -> {pista_mutacion[-1]:.2f}, sin volver a subir.")
print("Entonces la regla no adaptó aquí: mantener tranquilos a los operadores mantuvo mejorando a la media,")
print("lo cual mantuvo feliz al sensor, lo que mantuvo cerrando a los operadores. Es un programa de decaimiento.")
print("Esta es UNA semilla. Muestra que el mecanismo se ejecuta, no que ayuda: el paso 5 mide.")

FIGURAS.mkdir(exist_ok=True)
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
ejes[0].plot(pista_cruza, label="probabilidad cruza")
ejes[0].plot(pista_mutacion, label="probabilidad mutación")
sombrear_estancamientos(ejes[0], adaptativo["tendencia_estancamiento"])
ejes[0].axhline(PROBABILIDAD_CRUZA, color="tab:blue", linestyle="--", alpha=0.4)
ejes[0].axhline(PROBABILIDAD_MUTACION, color="tab:orange", linestyle="--", alpha=0.4)
ejes[0].set(xlabel="generación", ylabel="probabilidad", ylim=(0, 1.05),
            title="A dónde fueron las perillas (punteada = los valores fijos;\nsin banda roja porque ninguna generación se estancó)")
ejes[0].legend()
ejes[1].plot(fijo["historial_gastado"], fijo["historial_mejores"], label=f"fija: {fijo['longitud']:,.0f}")
ejes[1].plot(adaptativo["historial_gastado"], adaptativo["historial_mejores"],
             label=f"adaptativa: {adaptativo['longitud']:,.0f}")
ejes[1].set(xlabel="evaluaciones gastadas", ylabel="mejor longitud de ruta",
            title=f"Misma semilla, mismo presupuesto")
ejes[1].legend()
fig.tight_layout()
fig.savefig(FIGURAS / "ag_adaptativo_03_probabilidades_adaptativas.png", dpi=160)
plt.close(fig)

