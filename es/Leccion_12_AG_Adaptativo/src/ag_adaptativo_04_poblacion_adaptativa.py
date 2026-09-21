"""
Lección 12 - Paso 4: Una población que se redimensiona sola
=====================================================
NUEVO EN ESTE PASO: redimensionar_poblacion(), y un ejecutar() que puede usarlo.

La segunda mitad del algoritmo adaptativo del libro cambia la población misma.
Mientras la ejecución mejora, elimina al peor individuo, de modo que cada generación es
más barata y el presupuesto compra más de ellas. Cuando la ejecución se estanca, inyecta rutas
aleatorias frescas -- inmigrantes -- para que haya material nuevo que recombinar.

Ambos mecanismos están en el mismo sensor que el paso 3, así que se aplica la misma pregunta:
¿qué rama se dispara realmente? El script ejecuta tres regímenes y cuenta. El redimensionamiento
resulta traer de vuelta los estancamientos que el paso 3 había hecho desaparecer -- una población
más pequeña mejora a sacudidas, y las sacudidas activan el sensor -- así que aquí, a diferencia
del paso 3, se ejercen ambas ramas de ambas reglas.

CAMBIOS RESPECTO A ag_adaptativo_03_probabilidades_adaptativas.py
Introdúzcalos en este orden:
    1. constantes_poblacion         el piso, el techo y cuántos inmigrantes compra un estancamiento
    2. redimensionar_poblacion()    descartar al peor mientras mejora, inmigrar mientras está estancado
    3. ejecutar(redimensionar)      un segundo interruptor, y el tamaño de la población se convierte en un historial

Ejecútalo:  python ag_adaptativo_04_poblacion_adaptativa.py

Solo redimensionamiento: 35 estancamientos, población 68–120, 49,374. Con las probabilidades también: 41 estancamientos, población 40, 52,295 — peor en esta semilla que el redimensionamiento solo.
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

MIN_PROBABILIDAD_CRUZA = 0.1       # nunca dejar de recombinar por completo
MIN_PROBABILIDAD_MUTACION = 0.05   # nunca dejar de inventar por completo
MAX_PROBABILIDAD = 1.0
FACTOR_ESTANCAMIENTO = 1.1         # estancado: abrir los operadores, rápido
FACTOR_MEJORA = 0.99               # mejorando: cerrarlos, lentamente

# --- NUEVO (1) constantes_poblacion ---------------------------------------------
MIN_TAMANO_POBLACION = 40          # por debajo de esto el torneo no tiene de dónde elegir
MAX_TAMANO_POBLACION = 240         # por encima de esto una generación cuesta más de lo que devuelve
INMIGRANTES_POR_ESTANCAMIENTO = 2  # rutas aleatorias frescas inyectadas cuando la ejecución está atascada
# ------------------------------------------------------------------------------

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


def adaptar_probabilidades(cruza: float, mutacion: float, mejorando: bool) -> Tuple[float, float]:
    """Mueve ambas probabilidades un escalón, en la dirección que apunta el sensor.

    Abrir es más rápido que cerrar (1.1 contra 0.99) a propósito: de un
    estancamiento se tiene que escapar rápido, mientras que una ejecución que mejora debe dejarse sola
    por el tiempo que siga mejorando.
    """
    if mejorando:
        return (max(cruza * FACTOR_MEJORA, MIN_PROBABILIDAD_CRUZA),
                max(mutacion * FACTOR_MEJORA, MIN_PROBABILIDAD_MUTACION))
    return (min(cruza * FACTOR_ESTANCAMIENTO, MAX_PROBABILIDAD),
            min(mutacion * FACTOR_ESTANCAMIENTO, MAX_PROBABILIDAD))


# --- NUEVO (2) redimensionar_poblacion() ----------------------------------------------
def redimensionar_poblacion(puntos: Sequence[Tuple[int, int]], poblacion: List[Puntuado],
                            mejorando: bool) -> Tuple[List[Puntuado], int]:
    """Encogerse mientras funciona, crecer con sangre fresca mientras no lo hace.

    Devuelve la nueva población y las evaluaciones que costó: descartar es gratis,
    cada inmigrante tiene que ser medido antes de que pueda competir.

    Args:
        puntos: las 48 ciudades.
        poblacion: generación actual.
        mejorando: lo que dijo el sensor.

    Returns:
        ``(nueva población, evaluaciones de inmigrantes)``.

    Example:
        Solo redimensionamiento: 35 estancamientos, 70 inmigrantes, 103
        descartes, población 68–120.
    """
    if mejorando:
        if len(poblacion) > MIN_TAMANO_POBLACION:
            poblacion = poblacion[:]
            poblacion.remove(max(poblacion, key=lambda puntuado: puntuado[1]))
        return poblacion, 0
    agregados = 0
    poblacion = poblacion[:]
    while len(poblacion) < MAX_TAMANO_POBLACION and agregados < INMIGRANTES_POR_ESTANCAMIENTO:
        ruta = ruta_aleatoria(len(puntos))
        poblacion.append((ruta, longitud_ruta(puntos, ruta)))
        agregados += 1
    return poblacion, agregados
# ------------------------------------------------------------------------------


def ejecutar(puntos: Sequence[Tuple[int, int]], semilla: int, adaptativo: bool = False,
             redimensionar: bool = False) -> Dict[str, object]:
    """Una ejecución, detenida por el presupuesto; dos interruptores adaptativos independientes."""
    random.seed(semilla)
    evaluaciones = 0
    probabilidad_cruza = PROBABILIDAD_CRUZA         # las constantes se volvieron estado en el paso 3
    probabilidad_mutacion = PROBABILIDAD_MUTACION
    historial_cruza = [probabilidad_cruza]
    historial_mutacion = [probabilidad_mutacion]
    # --- NUEVO (3) ejecutar(redimensionar) --------------------------------------------------
    historial_tamano = [TAMANO_POBLACION]           # la población también es estado ahora
    inmigrantes = 0                                 # con qué frecuencia la rama estancada compró alcance
    descartes = 0                                   # con qué frecuencia la rama de mejora compró velocidad
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
            if random.random() < probabilidad_cruza:
                ruta = cruza_ordenada(padre_uno[0], padre_dos[0])
            else:
                ruta = list(padre_uno[0])
            if random.random() < probabilidad_mutacion:
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
        if adaptativo:
            probabilidad_cruza, probabilidad_mutacion = adaptar_probabilidades(
                probabilidad_cruza, probabilidad_mutacion, mejorando)
        if redimensionar:                                  # (3) el mismo sensor, un segundo actuador
            antes = len(poblacion)
            poblacion, costo = redimensionar_poblacion(puntos, poblacion, mejorando)
            evaluaciones += costo
            inmigrantes += costo
            descartes += 1 if len(poblacion) < antes else 0
        historial_cruza.append(probabilidad_cruza)
        historial_mutacion.append(probabilidad_mutacion)
        historial_tamano.append(len(poblacion))

    return {
        "ruta": mejor[0],
        "longitud": mejor[1],
        "historial_mejores": historial_mejores,
        "historial_medias": historial_medias,
        "historial_gastado": historial_gastado,
        "tendencia_estancamiento": tendencia_estancamiento,
        "historial_cruza": historial_cruza,
        "historial_mutacion": historial_mutacion,
        "historial_tamano": historial_tamano,
        "inmigrantes": inmigrantes,
        "descartes": descartes,
        "evaluaciones": evaluaciones,
        "generaciones": len(historial_mejores) - 1,
    }


def sombrear_estancamientos(eje: plt.Axes, tendencia_estancamiento: Sequence[bool]) -> None:
    """Pintar una banda roja sobre cada generación que la señal llama estancada."""
    for indice, mejorando in enumerate(tendencia_estancamiento):
        if not mejorando:
            eje.axvspan(indice + 1, indice + 2, color="red", alpha=0.12, linewidth=0)


puntos = cargar_puntos()
fija = ejecutar(puntos, SEMILLA)
redimensionada = ejecutar(puntos, SEMILLA, redimensionar=True)
completa = ejecutar(puntos, SEMILLA, adaptativo=True, redimensionar=True)

print("Lección 12 - Paso 4: una población que se redimensiona sola")
print(f"Las tres ejecuciones usan la semilla {SEMILLA} y el mismo presupuesto de {PRESUPUESTO:,} evaluaciones.")
encabezado = f"{'régimen':<26}{'longitud':>10}{'evals':>9}{'gens':>7}{'pob min-max':>14}{'inmigrantes':>12}{'descart':>7}"
print(encabezado)
for nombre, registro in (("fija", fija), ("solo redimensión", redimensionada), ("redimensión + probab.", completa)):
    tamanos = registro["historial_tamano"]
    print(f"{nombre:<26}{registro['longitud']:>10,.0f}{registro['evaluaciones']:>9,}"
          f"{registro['generaciones']:>7}{f'{min(tamanos)}-{max(tamanos)}':>14}"
          f"{registro['inmigrantes']:>12}{registro['descartes']:>7}")

print(f"Solo redimensión: el sensor se estancó {redimensionada['tendencia_estancamiento'].count(False)} veces y compró "
      f"{redimensionada['inmigrantes']} inmigrantes; descartar al peor {redimensionada['descartes']} veces")
print(f"  hizo a las generaciones más baratas, por lo que el mismo presupuesto compró "
      f"{redimensionada['generaciones'] - fija['generaciones']} más de ellas "
      f"({redimensionada['generaciones']} contra {fija['generaciones']}).")
print(f"Redimensión + probab.: {completa['tendencia_estancamiento'].count(False)} estancamientos contra "
      f"{fija['tendencia_estancamiento'].count(False)} de la ejecución fija y 0 en el paso 3 --")
print(f"  una población que se encoge mejora a sacudidas, por lo que la regla de probabilidad finalmente toma su "
      f"rama estancada también (la cruza oscila {min(completa['historial_cruza']):.2f}-"
      f"{max(completa['historial_cruza']):.2f} en lugar de decaer directamente hacia abajo).")
print(f"En esta semilla las dos mitades juntas ({completa['longitud']:,.0f}) lo hacen peor que solo "
      f"redimensionar ({redimensionada['longitud']:,.0f}).")
print("Sigue siendo una semilla. El paso 5 ejecuta todo esto doce veces.")

FIGURAS.mkdir(exist_ok=True)
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
ejes[0].plot(redimensionada["historial_tamano"], label="solo redimensión")
ejes[0].plot(completa["historial_tamano"], label="redimensión + probab.")
ejes[0].axhline(TAMANO_POBLACION, color="black", linestyle="--", alpha=0.5, label="fija")
sombrear_estancamientos(ejes[0], redimensionada["tendencia_estancamiento"])
ejes[0].set(xlabel="generación", ylabel="individuos",
            title="Tamaño de la población (bandas rojas: estanc. de la ej. de solo redimensión)")
ejes[0].legend()
for nombre, registro in (("fija", fija), ("solo redimensión", redimensionada), ("redimensión + probab.", completa)):
    ejes[1].plot(registro["historial_gastado"], registro["historial_mejores"],
                 label=f"{nombre}: {registro['longitud']:,.0f}")
ejes[1].set(xlabel="evaluaciones gastadas", ylabel="mejor longitud de ruta", title="Misma semilla, mismo presupuesto")
ejes[1].legend()
fig.tight_layout()
fig.savefig(FIGURAS / "ag_adaptativo_04_poblacion_adaptativa.png", dpi=160)
plt.close(fig)

