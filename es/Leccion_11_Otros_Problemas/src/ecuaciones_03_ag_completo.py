"""
Lección 11 - Ecuaciones 3: El algoritmo genético lo resuelve
============================================================
NUEVO EN ESTE PASO: Individuo, los tres operadores, y ejecutar().

Nada aquí es maquinaria nueva: la selección por rango con elitismo vino de la lección 03,
la cruza blend de la lección 04, la desviación gaussiana de la lección 05. Lo que es nuevo
es la regla de parada. Todas las demás lecciones se detuvieron después de un presupuesto fijo porque
no podían decir si habían ganado. Esta se detiene cuando el residual es cero,
porque eso es una prueba.

CAMBIOS RESPECTO A ecuaciones_02_poblacion_aleatoria.py
Introdúcelos en este orden:
    1. Individuo                     un candidato que lleva su propio residual
    2. seleccion_rango_con_elite()   selección por rango, dos élites conservadas
    3. cruza_blend()                 mezcla de valores reales, luego acotados a enteros
    4. mutacion_desviacion_aleatoria()  empujón gaussiano en cada gen con probabilidad 0.5
    5. ejecutar()                    el bucle, deteniéndose en el momento en que el residual es 0

Ejecútalo:  python ecuaciones_03_ag_completo.py

La semilla 3 alcanza residual 0 en la generación 5 tras 2,794 evaluaciones, en x = -6, y = 2, z = 3. f, g y w son cada una 0: eso es una prueba, no una opinión.
"""
from math import factorial
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEMILLA = 3
FIGURAS = Path(__file__).resolve().parent.parent / "figures"
TAMANO_POBLACION = 400
PROBABILIDAD_CRUZA = 0.8
PROBABILIDAD_MUTACION = 0.4
MAX_GENERACIONES = 100
TAMANO_ELITE = 2

CAJA_BAJA, CAJA_ALTA = -20, 20


def f(x: int, y: int, z: int) -> int:
    """Primera ecuación del sistema; una solución la hace cero."""
    return (x * y + 2) ** 2 + (x + y) ** (abs(z - 2) ** 3 + 2) + x * y * z


def g(x: int, y: int, z: int) -> int:
    """Segunda ecuación. El factorial es la razón por la que z debe mantenerse pequeña."""
    return x * (y * z + 10) - factorial(abs(z - 3)) + y ** abs(x) + 11 * z


def w(x: int, y: int, z: int) -> int:
    """Tercera ecuación."""
    return (x + 7 * y) ** abs(z + x) - (z + 16) ** 2 - 151


def error_total(x: int, y: int, z: int) -> int:
    """Suma de residuales absolutos. Exactamente cero significa una solución exacta."""
    try:
        return abs(f(x, y, z)) + abs(g(x, y, z)) + abs(w(x, y, z))
    except (ValueError, ZeroDivisionError):
        return 10 ** 100


def digitos(error: int) -> int:
    """Longitud decimal del residual, calculada a partir de la longitud en bits porque
    estos enteros son rutinariamente demasiado largos para que str() los convierta."""
    if error == 0:
        return 1
    estimacion = int(error.bit_length() * 0.30103) + 1
    while 10 ** (estimacion - 1) > error:
        estimacion -= 1
    while 10 ** estimacion <= error:
        estimacion += 1
    return estimacion


def acotar(valor: float) -> int:
    """Los genes son enteros dentro de la caja; la cruza y la mutación no lo son."""
    return max(CAJA_BAJA, min(CAJA_ALTA, round(valor)))


def tripleta_aleatoria() -> Tuple[int, int, int]:
    """Una extracción uniforme de la caja. Esto es todo lo que es la 'búsqueda aleatoria'.

    Returns:
        (x, y, z) en [-20, 20]^3.

    Example:
        La población inicial de 2,794 evaluaciones parte de aquí, no
        de la raíz.
    """
    return tuple(random.randint(CAJA_BAJA, CAJA_ALTA) for _ in range(3))


# --- NUEVO (1) Individuo ------------------------------------------------------
class Individuo:
    """Una tripleta candidata. La aptitud es el residual negado, así que mayor es mejor
    y el valor objetivo 0 es la mayor aptitud que el problema admite.

    Args:
        genes: tres reales; se acotan a enteros en [-20, 20].

    Returns:
        Un Individuo con ``aptitud = -error_total``.

    Example:
        Semilla 3: x = -6, y = 2, z = 3, aptitud 0 en la generación 5.
    """

    def __init__(self, genes: Sequence[float]) -> None:
        self.genes: List[int] = [acotar(gen) for gen in genes]
        self.aptitud: int = -error_total(*self.genes)

    def __str__(self) -> str:
        return f"x={self.genes[0]}, y={self.genes[1]}, z={self.genes[2]}"
# ------------------------------------------------------------------------------


# --- NUEVO (2) seleccion_rango_con_elite() ------------------------------------
def seleccion_rango_con_elite(individuos: List[Individuo],
                              tamano_elite: int = 0) -> List[Individuo]:
    """Selección por rango: el peso de muestreo depende de la posición, no de la aptitud.
    Eso es lo único que funciona aquí - los valores de aptitud difieren por
    miles de órdenes de magnitud, por lo que la selección proporcional le daría
    toda la población a cualquier individuo que casualmente lidere.

    Args:
        individuos: generación actual.
        tamano_elite: copias garantizadas del frente; aquí 2.

    Returns:
        Nueva población del mismo tamaño.

    Example:
        Semilla 3 llega a residual 0 en 5 generaciones. El rango, no la
        ruleta, es lo que hace posible esa presión.
    """
    ordenada = sorted(individuos, key=lambda ind: ind.aptitud, reverse=True)
    distancia_rango = 1.0 / len(individuos)
    rangos = [1.0 - indice * distancia_rango for indice in range(len(individuos))]
    suma_rangos = sum(rangos)
    seleccionados = ordenada[:tamano_elite]
    for _ in range(len(ordenada) - tamano_elite):
        umbral = random.random() * suma_rangos
        acumulado = 0.0
        for indice, rango in enumerate(rangos):
            acumulado += rango
            if acumulado > umbral:
                seleccionados.append(ordenada[indice])
                break
    return seleccionados
# ------------------------------------------------------------------------------


# --- NUEVO (3) cruza_blend() --------------------------------------------------
def cruza_blend(primero: Sequence[int], segundo: Sequence[int],
                    alpha: float = 0.8) -> Tuple[List[float], List[float]]:
    """Muestrea cada gen de un hijo a partir de un intervalo extendido más allá de los padres, para que
    el par pueda alcanzar valores fuera del segmento que los separa.

    Args:
        primero: genes del primer padre.
        segundo: genes del segundo padre.
        alpha: 0.8; ``acotar`` redondea después.

    Returns:
        Dos listas de reales, aún sin acotar.

    Example:
        El salto de una muestra aleatoria (residual 453) a la raíz
        (-6, 2, 3) pasa por estos intervalos.
    """
    hijo_uno, hijo_dos = list(primero), list(segundo)
    for indice in range(len(primero)):
        tramo = abs(hijo_dos[indice] - hijo_uno[indice])
        bajo = min(hijo_uno[indice], hijo_dos[indice]) - alpha * tramo
        alto = max(hijo_uno[indice], hijo_dos[indice]) + alpha * tramo
        hijo_uno[indice] = bajo + random.random() * (alto - bajo)
        hijo_dos[indice] = bajo + random.random() * (alto - bajo)
    return hijo_uno, hijo_dos
# ------------------------------------------------------------------------------


# --- NUEVO (4) mutacion_desviacion_aleatoria() --------------------------------
def mutacion_desviacion_aleatoria(genes: Sequence[int], sigma: float = 3.0,
                               probabilidad: float = 0.5) -> List[float]:
    """Un empujón gaussiano. Redondear de nuevo a enteros es lo que lo hace un salto.

    Args:
        genes: tripleta actual.
        sigma: 3.0; un paso de varios enteros.
        probabilidad: 0.5 por gen.

    Returns:
        Tres reales, aún sin acotar.

    Example:
        2,794 evaluaciones hasta residual 0. Cada mutación es un salto
        en la malla, no un desliz.
    """
    mutante = list(genes)
    for indice in range(len(mutante)):
        if random.random() < probabilidad:
            mutante[indice] += random.gauss(0.0, sigma)
    return mutante
# ------------------------------------------------------------------------------


# --- NUEVO (5) ejecutar() -----------------------------------------------------
def ejecutar() -> Tuple[Individuo, int, int, List[int]]:
    """Devuelve el mejor individuo, la generación alcanzada, el número de
    evaluaciones de residual gastadas, y el mejor residual después de cada generación.

    Returns:
        ``(campeón, generación, evaluaciones, historia)``. Se detiene al
        residual 0.

    Example:
        Semilla 3: generación 5, 2,794 evaluaciones, (-6, 2, 3).
    """
    random.seed(SEMILLA)
    poblacion = [Individuo(tripleta_aleatoria()) for _ in range(TAMANO_POBLACION)]
    evaluaciones = TAMANO_POBLACION
    mejor = max(poblacion, key=lambda ind: ind.aptitud)
    historia = [-mejor.aptitud]
    generacion = 0
    while generacion < MAX_GENERACIONES and mejor.aptitud != 0:
        generacion += 1
        padres = seleccion_rango_con_elite(poblacion, TAMANO_ELITE)
        cruzados: List[Individuo] = []
        for uno, dos in zip(padres[::2], padres[1::2]):
            if random.random() < PROBABILIDAD_CRUZA:
                genes_uno, genes_dos = cruza_blend(uno.genes, dos.genes)
                cruzados += [Individuo(genes_uno), Individuo(genes_dos)]
                evaluaciones += 2
            else:
                cruzados += [uno, dos]
        poblacion = []
        for candidato in cruzados:
            if random.random() < PROBABILIDAD_MUTACION:
                poblacion.append(Individuo(mutacion_desviacion_aleatoria(candidato.genes)))
                evaluaciones += 1
            else:
                poblacion.append(candidato)
        campeon = max(poblacion, key=lambda ind: ind.aptitud)
        if campeon.aptitud > mejor.aptitud:
            mejor = campeon
        historia.append(-mejor.aptitud)
    return mejor, generacion, evaluaciones, historia
# ------------------------------------------------------------------------------


mejor, generaciones, evaluaciones, historia = ejecutar()
lado = CAJA_ALTA - CAJA_BAJA + 1
x, y, z = mejor.genes

print("Lección 11 - Ecuaciones 3: el algoritmo genético lo resuelve")
print(f"Semilla {SEMILLA}, población {TAMANO_POBLACION}, límite de generaciones {MAX_GENERACIONES}")
print(f"Se detuvo en la generación:   {generaciones}")
print(f"Evaluaciones del residual:    {evaluaciones:,}")
print(f"Mejor candidato:          {mejor}")
print(f"Residual:                {-mejor.aptitud}")
print("Verificando las tres ecuaciones una por una, que es el punto de este problema:")
print(f"  f({x}, {y}, {z}) = {f(x, y, z)}")
print(f"  g({x}, {y}, {z}) = {g(x, y, z)}")
print(f"  w({x}, {y}, {z}) = {w(x, y, z)}")
print(f"Las tres son cero:      {f(x, y, z) == g(x, y, z) == w(x, y, z) == 0}")
print(f"Evaluaciones gastadas vs el tamaño de la caja completa: "
      f"{evaluaciones:,} de {lado ** 3:,} ({evaluaciones / lado ** 3:.0%})")
print("Esa última línea es la pregunta que el paso 4 responde.")

FIGURAS.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(len(historia)), [digitos(error) for error in historia], "o-")
ax.set(xlabel="generación", ylabel="dígitos decimales del mejor residual",
       title=f"Semilla {SEMILLA}: el residual llega a cero exactamente en la generación {generaciones}")
ax.axhline(1, color="crimson", linestyle="--", label="residual = 0")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "ecuaciones_03_ag_completo.png", dpi=160)
plt.close(fig)

