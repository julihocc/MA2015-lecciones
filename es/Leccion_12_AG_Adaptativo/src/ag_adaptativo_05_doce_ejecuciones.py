"""
Lección 12 - Paso 5: Doce ejecuciones de cada uno
========================================
NUEVO EN ESTE PASO: EJECUCIONES, medir(), resumen_emparejado(), y una tabla en lugar de una ejecución.

Los pasos 3 y 4 mostraron cada uno una semilla, y una semilla no decide nada: la Lección 06
midió la dispersión de este tipo de resultado y la Lección 07 mostró una clasificación de
parámetros volteándose de semilla en semilla. Así que cada régimen se ejecuta ahora doce veces,
la ejecución i en la semilla i, lo que hace que la comparación sea emparejada -- las mismas doce poblaciones
iniciales enfrentan a los cuatro regímenes, y la diferencia se puede leer ejecución por ejecución.

Los cuatro regímenes son el algoritmo fijo del paso 1, cada mitad de la maquinaria adaptativa
por sí sola, y ambas mitades juntas. Los cuatro se detienen en el mismo presupuesto
de evaluaciones, y la tabla imprime las evaluaciones que cada uno realmente gastó
para que el lector pueda comprobar que los presupuestos son realmente comparables.

CAMBIOS RESPECTO A ag_adaptativo_04_poblacion_adaptativa.py
Introdúzcalos en este orden:
    1. EJECUCIONES          doce ejecuciones emparejadas, ejecución i en semilla i
    2. medir()              un régimen sobre todas las semillas, devolviendo lo que encontró
    3. resumen_emparejado() la diferencia contra el régimen fijo, ejecución por ejecución
    4. la_comparacion       la tabla y el diagrama de caja reemplazan al reporte de una sola ejecución

Ejecútalo:  python ag_adaptativo_05_doce_ejecuciones.py

Sobre 12 ejecuciones emparejadas: fijo 58,195; probabilidades 50,145 (−8,050 ± 1,022, 12/12); redimensionar 53,555 (10/12); ambos 52,622 (11/12). Cada régimen gasta 11,901–11,946 evaluaciones.
"""
from math import dist, isnan, nan, sqrt
from pathlib import Path
import random
from statistics import fmean, stdev
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

MIN_TAMANO_POBLACION = 40          # por debajo de esto el torneo no tiene de dónde elegir
MAX_TAMANO_POBLACION = 240         # por encima de esto una generación cuesta más de lo que devuelve
INMIGRANTES_POR_ESTANCAMIENTO = 2  # rutas aleatorias frescas inyectadas cuando la ejecución está atascada

# --- NUEVO (1) EJECUCIONES -------------------------------------------------------------
EJECUCIONES = 12                   # la ejecución i usa la semilla i, así que cada régimen enfrenta los mismos dados
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


def redimensionar_poblacion(puntos: Sequence[Tuple[int, int]], poblacion: List[Puntuado],
                            mejorando: bool) -> Tuple[List[Puntuado], int]:
    """Encogerse mientras funciona, crecer con sangre fresca mientras no lo hace.

    Devuelve la nueva población y las evaluaciones que costó: descartar es gratis,
    cada inmigrante tiene que ser medido antes de que pueda competir.
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


def ejecutar(puntos: Sequence[Tuple[int, int]], semilla: int, adaptativo: bool = False,
             redimensionar: bool = False) -> Dict[str, object]:
    """Una ejecución, detenida por el presupuesto; dos interruptores adaptativos independientes."""
    random.seed(semilla)
    evaluaciones = 0
    probabilidad_cruza = PROBABILIDAD_CRUZA         # las constantes se volvieron estado en el paso 3
    probabilidad_mutacion = PROBABILIDAD_MUTACION
    historial_cruza = [probabilidad_cruza]
    historial_mutacion = [probabilidad_mutacion]
    historial_tamano = [TAMANO_POBLACION]           # la población se volvió estado en el paso 4
    inmigrantes = 0
    descartes = 0
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
        if redimensionar:
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


# --- NUEVO (2) medir() --------------------------------------------------------
def medir(puntos: Sequence[Tuple[int, int]], adaptativo: bool, redimensionar: bool) -> Dict[str, List[float]]:
    """Ejecutar un régimen una vez por semilla y guardar lo que cada ejecución encontró y gastó.

    Args:
        puntos: las 48 ciudades.
        adaptativo: enciende ``adaptar_probabilidades``.
        redimensionar: enciende ``redimensionar_poblacion``.

    Returns:
        Dict con 12 longitudes, evaluaciones y generaciones.

    Example:
        Fijo 58,195; probabilidades 50,145 (12/12); redimensionar 53,555
        (10/12); ambos 52,622 (11/12).
    """
    longitudes, gastados, generaciones = [], [], []
    for semilla in range(1, EJECUCIONES + 1):
        registro = ejecutar(puntos, semilla, adaptativo=adaptativo, redimensionar=redimensionar)
        longitudes.append(registro["longitud"])
        gastados.append(registro["evaluaciones"])
        generaciones.append(registro["generaciones"])
    return {"longitudes": longitudes, "evaluaciones": gastados, "generaciones": generaciones}
# ------------------------------------------------------------------------------


# --- NUEVO (3) resumen_emparejado() -------------------------------------------------
def resumen_emparejado(variante: Sequence[float], referencia: Sequence[float]) -> Dict[str, float]:
    """Compara dos regímenes ejecución por ejecución, no media contra media.

    Ambos regímenes vieron las mismas semillas, así que la estadística honesta es la media de las
    diferencias por ejecución y su error estándar. La media más o menos dos errores
    estándar se reporta como aproximación descriptiva, no como prueba de significancia.

    Args:
        variante: 12 longitudes del régimen bajo prueba.
        referencia: 12 longitudes del fijo, mismas semillas.

    Returns:
        Victorias, diferencia media, error estándar y si el intervalo aproximado
        de dos errores estándar excluye cero.

    Example:
        Probabilidades vs fijo: −8,050 ± 1,022, 12/12, decisivo.
        El paso 6 contra 0.60/0.05 no lo es: +1,274 ± 727.
    """
    diferencias = [bueno - base for bueno, base in zip(variante, referencia)]
    diferencia_media = fmean(diferencias)
    error = stdev(diferencias) / sqrt(len(diferencias)) if len(diferencias) > 1 else 0.0
    return {"victorias": sum(1 for valor in diferencias if valor < 0),
            "diferencia_media": diferencia_media,
            "error_estandar": error,
            "decisivo": abs(diferencia_media) > 2 * error}
# ------------------------------------------------------------------------------


def sombrear_estancamientos(eje: plt.Axes, tendencia_estancamiento: Sequence[bool]) -> None:
    """Pintar una banda roja sobre cada generación que la señal llama estancada."""
    for indice, mejorando in enumerate(tendencia_estancamiento):
        if not mejorando:
            eje.axvspan(indice + 1, indice + 2, color="red", alpha=0.12, linewidth=0)


# --- NUEVO (4) la_comparacion ---------------------------------------------------
puntos = cargar_puntos()
regimenes = [("fijo", False, False),
             ("probabilidades", True, False),
             ("redimensionar", False, True),
             ("ambos", True, True)]
resultados = {nombre: medir(puntos, adaptativo, redimensionar) for nombre, adaptativo, redimensionar in regimenes}
referencia = resultados["fijo"]["longitudes"]

print(f"Lección 12 - Paso 5: {EJECUCIONES} ejecuciones de cada régimen, semillas 1-{EJECUCIONES}, "
      f"presupuesto {PRESUPUESTO:,} evaluaciones")
print(f"{'régimen':<16}{'media':>9}{'de':>8}{'mejor':>9}{'evals':>9}{'gens':>7}"
      f"{'vict.':>7}{'diferencia media vs fijo':>28}")
for nombre, _, _ in regimenes:
    longitudes = resultados[nombre]["longitudes"]
    resumen = resumen_emparejado(longitudes, referencia)
    if nombre == "fijo":
        diferencia = "referencia"
        victorias = "-"
    else:
        diferencia = (f"{resumen['diferencia_media']:+,.0f} +/- "
                      f"{resumen['error_estandar']:,.0f}")
        victorias = f"{resumen['victorias']}/{EJECUCIONES}"
    print(f"{nombre:<16}{fmean(longitudes):>9,.0f}{stdev(longitudes):>8,.0f}{min(longitudes):>9,.0f}"
          f"{fmean(resultados[nombre]['evaluaciones']):>9,.0f}"
          f"{fmean(resultados[nombre]['generaciones']):>7,.0f}{victorias:>7}{diferencia:>28}")

print(f"(victorias = ejecuciones en las que ese régimen devolvió una ruta más corta que el régimen fijo "
      f"en la misma semilla)")
mejor_nombre = min(resultados, key=lambda nombre: fmean(resultados[nombre]["longitudes"]))
for nombre, _, _ in regimenes[1:]:
    resumen = resumen_emparejado(resultados[nombre]["longitudes"], referencia)
    veredicto = ("el intervalo aproximado de dos EE excluye cero"
                 if resumen["decisivo"] else "el intervalo aproximado de dos EE incluye cero")
    print(f"{nombre:<14} vs fijo: {resumen['diferencia_media']:+,.0f} "
          f"({resumen['diferencia_media'] / fmean(referencia):+.1%}), "
          f"{resumen['error_estandar']:,.0f} error estándar -- {veredicto}")
print(f"Ruta media más corta: {mejor_nombre} con {fmean(resultados[mejor_nombre]['longitudes']):,.0f}.")
print(f"Cada régimen gastó entre {min(fmean(resultados[n]['evaluaciones']) for n in resultados):,.0f} "
      f"y {max(fmean(resultados[n]['evaluaciones']) for n in resultados):,.0f} evaluaciones en promedio, "
      f"así que la comparación es a presupuesto igual.")
print("Los regímenes adaptativos superan a ESTA configuración fija. El paso 6 pregunta si superan a una afinada.")

FIGURAS.mkdir(exist_ok=True)
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
ejes[0].boxplot([resultados[nombre]["longitudes"] for nombre, _, _ in regimenes])
ejes[0].set_xticks(range(1, len(regimenes) + 1), [nombre for nombre, _, _ in regimenes])
ejes[0].set(ylabel="longitud de ruta", title=f"{EJECUCIONES} ejec. de cada régimen, presupuesto igual")
ejes[0].tick_params(axis="x", rotation=15)
for nombre, _, _ in regimenes[1:]:
    diferencias = [variante - base for variante, base in zip(resultados[nombre]["longitudes"], referencia)]
    ejes[1].plot(range(1, EJECUCIONES + 1), diferencias, "o-", label=nombre)
ejes[1].axhline(0, color="black", linewidth=1)
ejes[1].set(xlabel="ejecución (= semilla)", ylabel="longitud de ruta menos el régimen fijo",
            title="Diferencias emparejadas: por debajo de cero es mejor")
ejes[1].legend()
fig.tight_layout()
fig.savefig(FIGURAS / "ag_adaptativo_05_doce_ejecuciones.png", dpi=160)
plt.close(fig)
# ------------------------------------------------------------------------------

