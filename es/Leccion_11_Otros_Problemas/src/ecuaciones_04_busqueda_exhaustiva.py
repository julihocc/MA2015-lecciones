"""
Lección 11 - Ecuaciones 4: El método que no necesita un algoritmo genético
==========================================================================
NUEVO EN ESTE PASO: enumerar_caja(), un ejecutar() con semilla, y la comparación honesta.

Los genes son enteros y la caja es finita, así que todo el espacio de búsqueda puede
ser recorrido. Este es el paso donde el curso finalmente hace la pregunta que ha estado
evadiendo: dado que un algoritmo genético es una heurística, ¿cuándo es la herramienta
incorrecta? La respuesta aquí no es la que sugieren los conteos de evaluación, y el
paso está escrito alrededor de lo que el código realmente mide.

CAMBIOS RESPECTO A ecuaciones_03_ag_completo.py
Introdúcelos en este orden:
    1. enumerar_caja()   recorre cada tripleta entera en la caja y conserva cada raíz
    2. ejecutar()        toma la semilla como argumento, para que el AG pueda repetirse
    3. la comparación    costo, confiabilidad, y lo que cada método puede afirmar

Ejecútalo:  python ecuaciones_04_busqueda_exhaustiva.py

Las 5 semillas encuentran la misma raíz, en 5 a 28 generaciones. Enumerar la caja cuesta 68,921 evaluaciones y prueba que hay exactamente una raíz. El AG es más barato; la enumeración afirma más.
"""
from math import factorial
from pathlib import Path
import random
import time
from typing import Dict, List, Sequence, Tuple

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
    """Una extracción uniforme de la caja. Esto es todo lo que es la 'búsqueda aleatoria'."""
    return tuple(random.randint(CAJA_BAJA, CAJA_ALTA) for _ in range(3))


class Individuo:
    """Una tripleta candidata. La aptitud es el residual negado, así que mayor es mejor
    y el valor objetivo 0 es la mayor aptitud que el problema admite."""

    def __init__(self, genes: Sequence[float]) -> None:
        self.genes: List[int] = [acotar(gen) for gen in genes]
        self.aptitud: int = -error_total(*self.genes)

    def __str__(self) -> str:
        return f"x={self.genes[0]}, y={self.genes[1]}, z={self.genes[2]}"


def seleccion_rango_con_elite(individuos: List[Individuo],
                              tamano_elite: int = 0) -> List[Individuo]:
    """Selección por rango: el peso de muestreo depende de la posición, no de la aptitud.
    Eso es lo único que funciona aquí - los valores de aptitud difieren por
    miles de órdenes de magnitud, por lo que la selección proporcional le daría
    toda la población a cualquier individuo que casualmente lidere."""
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


def cruza_blend(primero: Sequence[int], segundo: Sequence[int],
                    alpha: float = 0.8) -> Tuple[List[float], List[float]]:
    """Muestrea cada gen de un hijo a partir de un intervalo extendido más allá de los padres, para que
    el par pueda alcanzar valores fuera del segmento que los separa."""
    hijo_uno, hijo_dos = list(primero), list(segundo)
    for indice in range(len(primero)):
        tramo = abs(hijo_dos[indice] - hijo_uno[indice])
        bajo = min(hijo_uno[indice], hijo_dos[indice]) - alpha * tramo
        alto = max(hijo_uno[indice], hijo_dos[indice]) + alpha * tramo
        hijo_uno[indice] = bajo + random.random() * (alto - bajo)
        hijo_dos[indice] = bajo + random.random() * (alto - bajo)
    return hijo_uno, hijo_dos


def mutacion_desviacion_aleatoria(genes: Sequence[int], sigma: float = 3.0,
                               probabilidad: float = 0.5) -> List[float]:
    """Un empujón gaussiano. Redondear de nuevo a enteros es lo que lo hace un salto."""
    mutante = list(genes)
    for indice in range(len(mutante)):
        if random.random() < probabilidad:
            mutante[indice] += random.gauss(0.0, sigma)
    return mutante


def ejecutar(semilla: int) -> Tuple[Individuo, int, int, List[int]]:   # --- CAMBIADO --- (2) ejecutar() toma la semilla
    """Devuelve el mejor individuo, la generación alcanzada, el número de
    evaluaciones de residual gastadas, y el mejor residual después de cada generación.

    Args:
        semilla: la ejecución i usa la semilla i.

    Returns:
        ``(campeón, generación, evaluaciones, historia)``.

    Example:
        Cinco semillas, misma raíz, 5 a 28 generaciones, 2,794 a 13,764
        evaluaciones.
    """
    random.seed(semilla)
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


# --- NUEVO (1) enumerar_caja() ------------------------------------------------
def enumerar_caja() -> Tuple[List[Tuple[int, int, int]], int, Dict[int, int]]:
    """Evalúa cada tripleta entera en la caja. Devuelve cada raíz exacta, el
    número de evaluaciones, y el recuento de dígitos del residual más pequeño por corte de z.

    Returns:
        ``(raíces, evaluaciones, dígitos_por_z)``.

    Example:
        68,921 evaluaciones, exactamente una raíz. El AG es ~10× más
        barato; esto prueba unicidad, que el AG no puede afirmar.
    """
    raices: List[Tuple[int, int, int]] = []
    por_corte: Dict[int, int] = {}
    evaluaciones = 0
    for z in range(CAJA_BAJA, CAJA_ALTA + 1):
        mas_pequeno = None
        for x in range(CAJA_BAJA, CAJA_ALTA + 1):
            for y in range(CAJA_BAJA, CAJA_ALTA + 1):
                error = error_total(x, y, z)
                evaluaciones += 1
                tamano = digitos(error) if error else 0
                if mas_pequeno is None or tamano < mas_pequeno:
                    mas_pequeno = tamano
                if error == 0:
                    raices.append((x, y, z))
        por_corte[z] = mas_pequeno
    return raices, evaluaciones, por_corte
# ------------------------------------------------------------------------------


# --- NUEVO (3) la comparación -------------------------------------------------
SEMILLAS = [3, 1, 7, 16, 42]
lado = CAJA_BAJA * 0 + (CAJA_ALTA - CAJA_BAJA + 1)

inicio_ag = time.perf_counter()
resultados_ag = [ejecutar(semilla) for semilla in SEMILLAS]
segundos_ag = time.perf_counter() - inicio_ag
resueltos = [resultado for resultado in resultados_ag if resultado[0].aptitud == 0]
evaluaciones_ag = [resultado[2] for resultado in resultados_ag]
generaciones_ag = [resultado[1] for resultado in resultados_ag]

inicio_exhaustivo = time.perf_counter()
raices, evaluaciones_exhaustivas, por_corte = enumerar_caja()
segundos_exhaustivo = time.perf_counter() - inicio_exhaustivo
# ------------------------------------------------------------------------------

mejor, generaciones, evaluaciones, historia = resultados_ag[0]

print("Lección 11 - Ecuaciones 4: el método que no necesita un algoritmo genético")
print(f"Caja de búsqueda: {lado ** 3:,} tripletas enteras")
print()
print(f"Algoritmo genético, semillas {SEMILLAS}:")
print(f"  encontró una raíz exacta:   {len(resueltos)} de {len(SEMILLAS)} ejecuciones")
print(f"  generaciones necesarias:    {min(generaciones_ag)} a {max(generaciones_ag)}")
print(f"  evaluaciones de residual:  {min(evaluaciones_ag):,} a {max(evaluaciones_ag):,}")
print(f"  tiempo total de pared:       {segundos_ag:.2f} s para {len(SEMILLAS)} ejecuciones")
print(f"  raíces reportadas:        {len({tuple(resultado[0].genes) for resultado in resueltos})} distintas")
print()
print("Enumeración exhaustiva de la misma caja:")
print(f"  evaluaciones de residual:  {evaluaciones_exhaustivas:,}")
print(f"  tiempo de pared:             {segundos_exhaustivo:.2f} s")
print(f"  raíces encontradas:           {len(raices)} -> {raices}")
print()
proporcion = evaluaciones_exhaustivas / (sum(evaluaciones_ag) / len(evaluaciones_ag))
print(f"El algoritmo genético es más barato: gasta en promedio "
      f"{sum(evaluaciones_ag) / len(evaluaciones_ag):,.0f} evaluaciones, "
      f"{proporcion:.0f} veces menos que la enumeración.")
print("Aún así, es la opción más débil aquí, y la razón no es el costo:")
print(f"  - la enumeración prueba que la caja contiene exactamente {len(raices)} raíz; el AG "
      f"no puede decir eso,")
print("  - la enumeración no tiene semilla, tamaño de población ni tasa de cruza,")
print(f"  - la enumeración siempre devuelve la misma respuesta; el AG devolvió una aquí "
      f"en {len(resueltos)} de {len(SEMILLAS)} ejecuciones, lo cual es una medición, no una garantía.")
mas_ancha = 201
print(f"Lo que cambiaría el veredicto es la caja. Cuesta lado^3, así que ensancharla "
      f"a [-100, 100] significa {mas_ancha ** 3:,} tripletas, "
      f"{mas_ancha ** 3 / lado ** 3:.0f} veces esta ejecución.")
print("Un algoritmo genético se gana su lugar cuando el espacio deja de ser recorrible,")
print("no cuando resulta vencer a un bucle que podrías haber escrito en cuatro líneas.")

FIGURAS.mkdir(exist_ok=True)
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
ejes[0].bar(["AG (media de\n%d semillas)" % len(SEMILLAS), "exhaustiva"],
            [sum(evaluaciones_ag) / len(evaluaciones_ag), evaluaciones_exhaustivas],
            color=["#4c72b0", "#c44e52"])
ejes[0].set(ylabel="evaluaciones de residual", title="El costo no es el argumento")
ejes[0].set_yscale("log")
zs = sorted(por_corte)
ejes[1].plot(zs, [por_corte[z] for z in zs], "o-")
ejes[1].set(xlabel="z", ylabel="dígitos del residual más pequeño en el corte",
            title="La enumeración también mapea la caja: el único cero está en z = %d" % raices[0][2])
fig.tight_layout()
fig.savefig(FIGURAS / "ecuaciones_04_busqueda_exhaustiva.png", dpi=160)
plt.close(fig)

