"""
Lección 12 - Paso 1: Una ejecución con parámetros fijos
==================================================
NUEVO EN ESTE PASO: el programa completo.

Hasta ahora, cada lección ha escrito los parámetros en la parte superior del archivo como
constantes; la Lección 07 luego los afinó por búsqueda en cuadrícula y reportó qué valores fijos
eran mejores. Esta lección hace una pregunta diferente: ¿deberían ser fijos
en absoluto? Una ejecución tiene una fase temprana, cuando la población está dispersa y la
búsqueda necesita alcance, y una fase tardía, cuando la población es casi idéntica
y la búsqueda necesita refinamiento. Una constante tiene que servir a ambas.

El problema es la instancia del agente viajero de la Lección 10 -- las mismas 48 capitales
estadounidenses, la misma cruza ordenada y mutación por inversión, reescritas aquí para
que este archivo se ejecute por su cuenta. Se usa una búsqueda de permutación porque es lo suficientemente
larga para que esas dos fases sean visibles.

Una cosa sí cambia respecto a la Lección 10: la ejecución se detiene cuando ha gastado un
presupuesto de evaluaciones, no después de un número fijo de generaciones. La Lección 07
estableció las evaluaciones como la moneda en la que se comparan dos configuraciones, y
el algoritmo adaptativo de los pasos 3 y 4 cambia su propio tamaño de población, lo cual
hace que un conteo de generaciones no tenga sentido como presupuesto.

Ejecútalo:  python ag_adaptativo_01_parametros_fijos.py

La semilla 1 gasta 11,901 evaluaciones en 99 generaciones y devuelve una ruta legal de 57,076, un 40.8% más larga que la línea base del vecino más cercano de 40,526.
"""
from math import dist
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

Ruta = List[int]
Puntuado = Tuple[Ruta, float]


def cargar_puntos(ruta_archivo: Path = DATOS) -> List[Tuple[int, int]]:
    """Las 48 capitales, como coordenadas enteras, una ciudad por línea.

    Args:
        ruta_archivo: ``att48_xy.txt`` de esta lección, no importado de la 10.

    Returns:
        48 pares (x, y).

    Example:
        Vecino más cercano 40,526. El AG fijo de semilla 1 llega a 57,076.
    """
    return [tuple(map(int, linea.split())) for linea in ruta_archivo.read_text().splitlines() if linea.strip()]


def longitud_ruta(puntos: Sequence[Tuple[int, int]], ruta: Sequence[int]) -> float:
    """Longitud del recorrido cerrado. Una llamada a esta función es una evaluación.

    Args:
        puntos: las 48 ciudades.
        ruta: permutación legal.

    Returns:
        Longitud euclidiana. Cada llamada cuenta 1 en el presupuesto de 12,000.

    Example:
        Semilla 1 gasta 11,901 de estas llamadas. Generaciones ya no son
        la moneda: el paso 4 redimensiona la población.
    """
    return sum(dist(puntos[a], puntos[b]) for a, b in zip(ruta, tuple(ruta[1:]) + tuple(ruta[:1])))


def ruta_aleatoria(tamano: int) -> Ruta:
    """Una permutación legal. La población inicial y los inmigrantes del paso 4 salen de aquí.

    Args:
        tamano: 48.

    Returns:
        Una permutación de 0..47.

    Example:
        120 rutas iniciales, semilla 1. El AG fijo parte de ellas hacia
        57,076.
    """
    ruta = list(range(tamano))
    random.shuffle(ruta)
    return ruta


def vecino_mas_cercano(puntos: Sequence[Tuple[int, int]], inicio: int = 0) -> Ruta:
    """La línea base constructiva barata de la Lección 10; el AG tiene que responder a ella.

    Args:
        puntos: las 48 ciudades.
        inicio: ciudad 0.

    Returns:
        Una permutación legal de longitud 40,526.

    Example:
        El AG fijo de semilla 1 queda 40.8% más largo. Esa es la línea
        que el adaptativo del paso 3 también tiene que mirar.
    """
    ruta, no_vistos = [inicio], set(range(len(puntos))) - {inicio}
    while no_vistos:
        actual = ruta[-1]
        mas_cercano = min(no_vistos, key=lambda ciudad: dist(puntos[actual], puntos[ciudad]))
        ruta.append(mas_cercano)
        no_vistos.remove(mas_cercano)
    return ruta


def cruza_ordenada(primero: Sequence[int], segundo: Sequence[int]) -> Ruta:
    """Conserva un fragmento de un padre y llena los huecos en el orden del otro padre.

    Args:
        primero: padre legal.
        segundo: padre legal.

    Returns:
        Una permutación. Probabilidad fija 0.90 en este paso.

    Example:
        El paso 3 baja esa 0.90 a 0.33 y nunca la sube. Aquí es constante.
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
    """Invierte un segmento: una permutación legal sigue siendo una permutación legal.

    Args:
        fuente: ruta legal.

    Returns:
        Una permutación. Probabilidad fija 0.25 en este paso.

    Example:
        El paso 3 la baja a 0.09. El rival afinado del paso 6 usa 0.05.
    """
    ruta = list(fuente)
    izq, der = sorted(random.sample(range(len(ruta)), 2))
    ruta[izq:der] = reversed(ruta[izq:der])
    return ruta


def seleccion_torneo(poblacion: List[Puntuado]) -> Puntuado:
    """Torneo de 3 sobre longitud. No se adapta: la presión es un entero fijo.

    Args:
        poblacion: pares (ruta, longitud).

    Returns:
        El más corto de tres.

    Example:
        99 generaciones de esta presión, 11,901 evaluaciones, 57,076.
    """
    return min(random.sample(poblacion, TAMANO_TORNEO), key=lambda puntuado: puntuado[1])


def ejecutar(puntos: Sequence[Tuple[int, int]], semilla: int) -> Dict[str, object]:
    """Una ejecución del AG de parámetros fijos, detenida por el presupuesto de evaluaciones.

    Solo se inicia una generación si el presupuesto puede pagarla completa, para que ninguna ejecución
    se corte a la mitad y cada ejecución reporte las evaluaciones que realmente gastó.

    Args:
        puntos: las 48 ciudades.
        semilla: 1 en este paso; la ejecución i usa i en el 5.

    Returns:
        Dict con mejor ruta, longitudes, evaluaciones y generaciones.

    Example:
        Semilla 1: 11,901 evaluaciones, 99 generaciones, 57,076.
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

    return {
        "ruta": mejor[0],
        "longitud": mejor[1],
        "historial_mejores": historial_mejores,
        "historial_medias": historial_medias,
        "historial_gastado": historial_gastado,
        "evaluaciones": evaluaciones,
        "generaciones": len(historial_mejores) - 1,
    }


puntos = cargar_puntos()
linea_base = longitud_ruta(puntos, vecino_mas_cercano(puntos))
registro = ejecutar(puntos, SEMILLA)

print("Lección 12 - Paso 1: una ejecución con parámetros fijos")
print(f"Ciudades:                   {len(puntos)}")
print(f"Cruza / mutación:           {PROBABILIDAD_CRUZA} / {PROBABILIDAD_MUTACION} (constante toda la ejecución)")
print(f"Población:                  {TAMANO_POBLACION} (constante toda la ejecución)")
print(f"Presupuesto:                {PRESUPUESTO:,} evaluaciones")
print(f"Gastado:                    {registro['evaluaciones']:,} en {registro['generaciones']} generaciones")
print(f"Línea base vecino-cercano:  {linea_base:,.0f}")
print(f"Mejor ruta AG (semilla {SEMILLA}):    {registro['longitud']:,.0f}")
print(f"AG vs línea base:           {(registro['longitud'] - linea_base) / linea_base:+.1%}")
print(f"Ruta final legal:           {sorted(registro['ruta']) == list(range(len(puntos)))}")

FIGURAS.mkdir(exist_ok=True)
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
ejes[0].plot(registro["historial_mejores"], label="mejor de la ejecución")
ejes[0].plot(registro["historial_medias"], label="media de población")
ejes[0].axhline(linea_base, color="black", linestyle="--", label="vecino más cercano")
ejes[0].set(xlabel="generación", ylabel="longitud de ruta",
            title=f"Parámetros fijos, semilla {SEMILLA}")
ejes[0].legend()
cerrado = registro["ruta"] + registro["ruta"][:1]
ejes[1].plot([puntos[ciudad][0] for ciudad in cerrado], [puntos[ciudad][1] for ciudad in cerrado], "o-", markersize=3)
ejes[1].set(title=f"Mejor ruta: {registro['longitud']:,.0f}", aspect="equal")
ejes[1].set_xticks([])
ejes[1].set_yticks([])
fig.tight_layout()
fig.savefig(FIGURAS / "ag_adaptativo_01_parametros_fijos.png", dpi=160)
plt.close(fig)

