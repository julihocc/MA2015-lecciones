"""
Lección 12 - Paso 6: El rival afinado
====================================
NUEVO EN ESTE PASO: ejecutar() toma sus tasas iniciales como argumentos, CONFIGURACIONES_FIJAS,
y la comparación que decide la lección.

El paso 5 mostró a los tres regímenes adaptativos superando al algoritmo fijo, el mejor de
ellos por ocho mil unidades de longitud de ruta. Pero el algoritmo fijo que superaron nunca fue afinado: sus
probabilidades de cruza y mutación se heredaron de la Lección 10, donde
se eligieron para un presupuesto diferente y nunca se cuestionaron. La Lección 07 dedicó una sesión
entera a la forma correcta de elegirlas, y aún no se le ha pedido al esquema adaptativo
que supere eso.

Así que este paso hace el trabajo de la Lección 07 en este problema -- un pequeño barrido de configuraciones
fijas, cada una medida sobre las mismas doce semillas en el mismo presupuesto -- y pone
a la mejor de ellas contra el régimen adaptativo. Cuatro configuraciones es una cuadrícula delgada,
elegida para que quepa en una clase; el punto no es encontrar el óptimo del barrido sino
averiguar si la maquinaria adaptativa sobrevive al contacto con un oponente que
haya sido afinado en absoluto.

Las dos últimas líneas de la salida son la lección.

CAMBIOS RESPECTO A ag_adaptativo_05_doce_ejecuciones.py
Introdúzcalos en este orden:
    1. ejecutar(tasas)         las probabilidades iniciales se vuelven argumentos, no constantes
    2. CONFIGURACIONES_FIJAS   cuatro configuraciones fijas, siendo la primera la del paso 1
    3. el_veredicto            cada configuración contra el régimen adaptativo, emparejado, a igual presupuesto

Ejecútalo:  python ag_adaptativo_06_un_rival_afinado.py

Contra cuatro fijas: 0.60/0.05 vale 48,870 y el adaptativo gana 4/12 (+1,274 ± 727). El régimen adaptativo queda detrás de 3 de 4 en la media y no se distingue de la mejor.
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

EJECUCIONES = 12                   # la ejecución i usa la semilla i, así que cada régimen enfrenta los mismos dados

# --- NUEVO (2) CONFIGURACIONES_FIJAS ---------------------------------------------------
# Cuatro configuraciones fijas (cruza, mutación), al estilo de la cuadrícula de la Lección 07.
# La primera es la configuración que los pasos 1-5 heredaron de la Lección 10; las otras tres
# son esquinas del mismo barrido, más silenciosas en un operador u otro.
CONFIGURACIONES_FIJAS = [(0.9, 0.25), (0.6, 0.05), (0.4, 0.15), (0.2, 0.30)]
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


# --- NUEVO (1) ejecutar(tasas) -------------------------------------------------------
def ejecutar(puntos: Sequence[Tuple[int, int]], semilla: int, adaptativo: bool = False,
             redimensionar: bool = False, cruza: float = PROBABILIDAD_CRUZA,
             mutacion: float = PROBABILIDAD_MUTACION) -> Dict[str, object]:
    """Una ejecución, detenida por el presupuesto; las tasas iniciales ahora son argumentos.

    Nada más cambia: con los argumentos por defecto esto es exactamente el ejecutar() del
    paso 5. Hacer que las tasas sean argumentos es lo que permite que un script pruebe cuatro configuraciones
    sin cuatro copias de las constantes.

    Args:
        puntos: las 48 ciudades.
        semilla: 1..12.
        adaptativo: enciende la regla de probabilidades.
        redimensionar: enciende el redimensionamiento.
        cruza: tasa inicial fija o de partida.
        mutacion: tasa inicial fija o de partida.

    Returns:
        Dict de la ejecución.

    Example:
        0.60/0.05 fijo: media 48,870. Adaptativo gana 4/12.
    """
    random.seed(semilla)
    evaluaciones = 0
    probabilidad_cruza = cruza
    probabilidad_mutacion = mutacion
# ------------------------------------------------------------------------------
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


def medir(puntos: Sequence[Tuple[int, int]], adaptativo: bool, redimensionar: bool,
          cruza: float = PROBABILIDAD_CRUZA,
          mutacion: float = PROBABILIDAD_MUTACION) -> Dict[str, List[float]]:
    """Ejecutar un régimen una vez por semilla y guardar lo que cada ejecución encontró y gastó."""
    longitudes, gastados, generaciones = [], [], []
    medias_cruza, medias_mutacion = [], []
    for semilla in range(1, EJECUCIONES + 1):
        registro = ejecutar(puntos, semilla, adaptativo=adaptativo, redimensionar=redimensionar,
                            cruza=cruza, mutacion=mutacion)   # (1) las tasas viajan a través
        longitudes.append(registro["longitud"])
        gastados.append(registro["evaluaciones"])
        generaciones.append(registro["generaciones"])
        medias_cruza.append(fmean(registro["historial_cruza"]))
        medias_mutacion.append(fmean(registro["historial_mutacion"]))
    return {"longitudes": longitudes, "evaluaciones": gastados, "generaciones": generaciones,
            "cruza": medias_cruza, "mutacion": medias_mutacion}


def resumen_emparejado(variante: Sequence[float], referencia: Sequence[float]) -> Dict[str, float]:
    """Compara dos regímenes ejecución por ejecución, no media contra media.

    Ambos regímenes vieron las mismas semillas, así que la estadística honesta es la media de las
    diferencias por ejecución y su error estándar. La media más o menos dos errores
    estándar se reporta como aproximación descriptiva, no como prueba de significancia.
    """
    diferencias = [bueno - base for bueno, base in zip(variante, referencia)]
    diferencia_media = fmean(diferencias)
    error = stdev(diferencias) / sqrt(len(diferencias)) if len(diferencias) > 1 else 0.0
    return {"victorias": sum(1 for valor in diferencias if valor < 0),
            "diferencia_media": diferencia_media,
            "error_estandar": error,
            "decisivo": abs(diferencia_media) > 2 * error}


def sombrear_estancamientos(eje: plt.Axes, tendencia_estancamiento: Sequence[bool]) -> None:
    """Pintar una banda roja sobre cada generación que la señal llama estancada."""
    for indice, mejorando in enumerate(tendencia_estancamiento):
        if not mejorando:
            eje.axvspan(indice + 1, indice + 2, color="red", alpha=0.12, linewidth=0)


# --- NUEVO (3) el_veredicto ------------------------------------------------------
puntos = cargar_puntos()
resultado_adaptativo = medir(puntos, adaptativo=True, redimensionar=False)
longitudes_adaptativas = resultado_adaptativo["longitudes"]
resultados_fijos = {tasas: medir(puntos, adaptativo=False, redimensionar=False,
                                 cruza=tasas[0], mutacion=tasas[1])
                    for tasas in CONFIGURACIONES_FIJAS}

print(f"Lección 12 - Paso 6: el rival afinado, {EJECUCIONES} ejecuciones cada uno, presupuesto {PRESUPUESTO:,} evaluaciones")
print(f"{'régimen':<28}{'media':>9}{'de':>8}{'mejor':>9}{'evals':>9}"
      f"{'victorias adapt.':>16}{'adaptativo menos fijo':>25}")
print(f"{'adaptativo (probabilidades)':<28}{fmean(longitudes_adaptativas):>9,.0f}"
      f"{stdev(longitudes_adaptativas):>8,.0f}{min(longitudes_adaptativas):>9,.0f}"
      f"{fmean(resultado_adaptativo['evaluaciones']):>9,.0f}{'referencia':>16}{'':>25}")
for tasas in CONFIGURACIONES_FIJAS:
    longitudes = resultados_fijos[tasas]["longitudes"]
    resumen = resumen_emparejado(longitudes_adaptativas, longitudes)
    etiqueta = f"fijo {tasas[0]:.2f} / {tasas[1]:.2f}"
    if tasas == (PROBABILIDAD_CRUZA, PROBABILIDAD_MUTACION):
        etiqueta += " (paso 1)"
    victorias = f"{resumen['victorias']}/{EJECUCIONES}"
    diferencia = f"{resumen['diferencia_media']:+,.0f} +/- {resumen['error_estandar']:,.0f}"
    print(f"{etiqueta:<28}{fmean(longitudes):>9,.0f}{stdev(longitudes):>8,.0f}{min(longitudes):>9,.0f}"
          f"{fmean(resultados_fijos[tasas]['evaluaciones']):>9,.0f}{victorias:>16}{diferencia:>25}")
print("(victorias adapt. = ejecuciones emparejadas en las que el régimen adaptativo devolvió la ruta más corta;")
print(" adaptativo menos fijo: un número positivo significa que el régimen adaptativo salió más largo)")

mejores_tasas = min(CONFIGURACIONES_FIJAS, key=lambda tasas: fmean(resultados_fijos[tasas]["longitudes"]))
mejores_longitudes = resultados_fijos[mejores_tasas]["longitudes"]
duelo = resumen_emparejado(longitudes_adaptativas, mejores_longitudes)
nota_intervalo = ("el intervalo aproximado de dos EE excluye cero"
                  if duelo["decisivo"] else "el intervalo aproximado de dos EE incluye cero")
print(f"Mejor configuración fija entre cuatro en estas mismas doce semillas: "
      f"cruza {mejores_tasas[0]:.2f}, mutación {mejores_tasas[1]:.2f}, "
      f"media {fmean(mejores_longitudes):,.0f}.")
print(f"Media adaptativa {fmean(longitudes_adaptativas):,.0f} frente a esa configuración "
      f"({duelo['diferencia_media']:+,.0f} +/- {duelo['error_estandar']:,.0f}, "
      f"ganando {duelo['victorias']} de {EJECUCIONES} ejecuciones emparejadas); {nota_intervalo}.")
detras = sum(1 for tasas in CONFIGURACIONES_FIJAS
             if fmean(resultados_fijos[tasas]["longitudes"]) < fmean(longitudes_adaptativas))
print(f"En la media está por detrás de {detras} de las {len(CONFIGURACIONES_FIJAS)} configuraciones fijas y por delante de "
      f"{len(CONFIGURACIONES_FIJAS) - detras}.")
print(f"El programa pasa la ejecución promediando cruza {fmean(resultado_adaptativo['cruza']):.2f} "
      f"y mutación {fmean(resultado_adaptativo['mutacion']):.2f}; la mejor configuración fija es "
      f"cruza {mejores_tasas[0]:.2f}, mutación {mejores_tasas[1]:.2f}.")
print("Esta comparación exploratoria seleccionó y evaluó las configuraciones fijas con las mismas semillas;")
print("describe estas ejecuciones, pero no es validación independiente ni una prueba de significancia.")

FIGURAS.mkdir(exist_ok=True)
etiquetas = ["adaptativo"] + [f"{tasas[0]:.2f}/{tasas[1]:.2f}" for tasas in CONFIGURACIONES_FIJAS]
medias = [fmean(longitudes_adaptativas)] + [fmean(resultados_fijos[tasas]["longitudes"]) for tasas in CONFIGURACIONES_FIJAS]
errores = [stdev(longitudes_adaptativas) / sqrt(EJECUCIONES)] + [
    stdev(resultados_fijos[tasas]["longitudes"]) / sqrt(EJECUCIONES) for tasas in CONFIGURACIONES_FIJAS]
colores = ["tab:orange"] + ["tab:blue"] * len(CONFIGURACIONES_FIJAS)
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
ejes[0].bar(etiquetas, medias, yerr=errores, capsize=4, color=colores)
ejes[0].set(ylabel="longitud de ruta media", ylim=(0, max(medias) * 1.2),
            title=f"Adaptativo contra cuatro configuraciones fijas ({EJECUCIONES} ejecuciones cada uno)")
ejes[0].tick_params(axis="x", rotation=15)
ejes[1].boxplot([longitudes_adaptativas] + [resultados_fijos[tasas]["longitudes"] for tasas in CONFIGURACIONES_FIJAS])
ejes[1].set_xticks(range(1, len(etiquetas) + 1), etiquetas)
ejes[1].set(ylabel="longitud de ruta", title="Mismos datos, ejecución por ejecución")
ejes[1].tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(FIGURAS / "ag_adaptativo_06_un_rival_afinado.png", dpi=160)
plt.close(fig)
# ------------------------------------------------------------------------------

