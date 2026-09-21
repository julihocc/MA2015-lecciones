"""
Lección 04 - Paso 1: Dos padres, y cómo medir lo que un operador les hace
=================================================================================
La cruza es el operador que toma dos cromosomas y devuelve dos nuevos.
Nada más en un algoritmo genético crea una combinación que no estaba ahí
antes, por lo que toda la lección gira en torno a una pregunta: DADOS ESTOS DOS PADRES,
¿CUÁL ES EL CONJUNTO DE HIJOS QUE UN OPERADOR PUEDE PRODUCIR?

Este paso no construye nada más que el instrumento que lo responde, y lo aplica al
operador más barato posible: el que devuelve a los padres tal cual.
Cada uno de los siete operadores en los pasos 2 a 7 se mide contra esa fila.

Los padres son el propio par de Gridin para el capítulo 4 (SEMILLA = 3, seis genes extraídos
uniformemente de la caja), por lo que los números aquí se alinean con los del libro.

Ejecútalo:  python descendencia_01_dos_padres.py

La fila cero: 2 hijos, 0.00% genes nuevos, 100.00% clones, 100.00% legales.
Elegir libremente entre los padres permite 2^6 = 64 cromosomas.
"""
import random
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt


SEMILLA = 3                      # Semilla de Gridin para el capítulo 4; cada medición se restablece a ella
CANTIDAD_GENES = 6               # un cromosoma son seis números reales
GEN_MIN, GEN_MAX = 0.0, 10.0     # la caja dentro de la cual un gen tiene que mantenerse
ENSAYOS = 2000                   # extracciones por medición, para porcentajes estables
FIGURAS = Path(__file__).resolve().parent.parent / "figures"

Cromosoma = list[float]
Operador = Callable[[list, list], tuple[list, list]]
Legalidad = Callable[[list], bool]


def crear_padres() -> tuple[Cromosoma, Cromosoma]:
    """Los dos padres que se entregan a cada operador en esta lección.

    Son los padres del capítulo 4 de Gridin (misma semilla, mismos seis genes)
    por lo que cualquier cosa impresa aquí se puede comparar directamente con el libro.
    random.seed() se llama dentro de la función en lugar de a nivel de módulo para
    que cada medición comience desde el mismo par, independientemente de lo que se haya ejecutado antes.

    Args:
        (sin argumentos).

    Returns:
        tuple[Cromosoma, Cromosoma].

    Example:
        SEMILLA = 3 produce [2.38 5.44 3.70 6.04 6.26 0.66] y [0.13 8.37 2.59 2.34 9.96 4.70].
    """
    random.seed(SEMILLA)
    p1 = [round(random.uniform(GEN_MIN, GEN_MAX), 2) for _ in range(CANTIDAD_GENES)]
    p2 = [round(random.uniform(GEN_MIN, GEN_MAX), 2) for _ in range(CANTIDAD_GENES)]
    return p1, p2


def es_legal_real(cromosoma: Cromosoma) -> bool:
    """Un cromosoma de valores reales es legal cuando cada gen está dentro de la caja.

    Esa es la restricción completa en esta representación, y es lo que permite
    que las tablas a continuación alguna vez llamen a un hijo *ilegal*: un operador que sale
    de la caja ha producido algo que el problema no puede evaluar.

    Args:
        cromosoma.

    Returns:
        bool.

    Example:
        La fila cero: 2 hijos, 0.00% genes nuevos, 100.00% clones, 100.00% legales.
    """
    return all(GEN_MIN <= gen <= GEN_MAX for gen in cromosoma)


def mostrar(cromosoma) -> str:
    """Un cromosoma en una línea, para que padres e hijos se alineen en columnas.

    Args:
        cromosoma.

    Returns:
        str.

    Example:
        La fila cero: 2 hijos, 0.00% genes nuevos, 100.00% clones, 100.00% legales.
    """
    partes = [f"{gen:6.2f}" if isinstance(gen, float) else f"{gen:>6d}"
             for gen in cromosoma]
    return "[" + " ".join(partes) + "]"


def cruza_clon(padre1: list, padre2: list) -> tuple[list, list]:
    """La línea base de alcance cero: los hijos SON los padres.

    Nadie usaría esto. Es la vara de medir: cada fila de cada tabla a continuación
    es interesante exactamente en la medida en que difiere de esta.

    Args:
        padre1, padre2.

    Returns:
        tuple[list, list].

    Example:
        2 hijos, 0.00% genes nuevos, 100.00% clones, 100.00% legales.
    """
    return list(padre1), list(padre2)


def reporte_descendencia(etiqueta: str, operador: Operador,
                         padre1: list, padre2: list,
                         es_legal: Legalidad, ensayos: int = ENSAYOS) -> dict:
    """Ejecuta un operador `ensayos` veces en el mismo par y describe la nube.

    Cuatro números, y toda la lección se lee de ellos:

      hijos     cuántos cromosomas DIFERENTES salieron. Para los operadores de cortar
                y cambiar, este es el conjunto alcanzable completo del operador, lo
                suficientemente pequeño para ser listado; para los aritméticos es solo el
                número de extracciones, que es el punto.
      nuevos    % de ranuras de genes que tienen un valor que NINGUNO de los padres tenía en esa
                ranura - valores que el operador calculó en lugar de copiar.
      clones    % de hijos que son una copia exacta de un padre: el operador
                se ejecutó y no pasó nada.
      legales   % de hijos que el problema realmente puede leer.

    Siempre re-sembrado de SEMILLA, para que las filas de una tabla sean comparables por
    construcción en lugar de por esperanza.

    Args:
        etiqueta, operador, padre1, padre2, es_legal, ensayos.

    Returns:
        dict.

    Example:
        2000 extracciones desde SEMILLA = 3; un punto alcanza 10 hijos.
    """
    random.seed(SEMILLA)
    vistos: set[tuple] = set()
    ranuras = valores_nuevos = clones = legales = producidos = 0
    for _ in range(ensayos):
        for hijo in operador(padre1, padre2):
            producidos += 1
            vistos.add(tuple(hijo))
            if list(hijo) == list(padre1) or list(hijo) == list(padre2):
                clones += 1
            if es_legal(hijo):
                legales += 1
            for indice, gen in enumerate(hijo):
                ranuras += 1
                if gen != padre1[indice] and gen != padre2[indice]:
                    valores_nuevos += 1
    return {"etiqueta": etiqueta, "hijos": len(vistos), "nuevos": 100 * valores_nuevos / ranuras,
            "clones": 100 * clones / producidos, "legales": 100 * legales / producidos,
            "alcanzables": vistos}


def imprimir_encabezado_reporte() -> None:
    """Un encabezado para cada tabla de comparación en los pasos 1 a 7.

    Args:
        (sin argumentos).

    Returns:
        None. Imprime o guarda figura.

    Example:
        La fila cero: 2 hijos, 0.00% genes nuevos, 100.00% clones, 100.00% legales.
    """
    print(f"{'operador':28s} {'hijos':>8} {'nuevos':>7} {'clones':>7} {'legales':>7}")
    print(f"{'':28s} {'alcanz.':>8} {'genes':>7} {'de un':>7} {'':>7}")
    print(f"{'':28s} {'':>8} {'':>7} {'padre':>7} {'':>7}")


def imprimir_fila_reporte(fila: dict) -> None:
    """imprimir_fila_reporte.

    Args:
        fila.

    Returns:
        None. Imprime o guarda figura.

    Example:
        La fila cero: 2 hijos, 0.00% genes nuevos, 100.00% clones, 100.00% legales.
    """
    print(f"{fila['etiqueta']:28s} {fila['hijos']:8d} {fila['nuevos']:6.2f}% "
          f"{fila['clones']:6.2f}% {fila['legales']:6.2f}%")


padre1, padre2 = crear_padres()

print("Los dos padres, gen por gen:")
print(f"    padre 1    {mostrar(padre1)}")
print(f"    padre 2    {mostrar(padre2)}")
print(f"    distancia  {mostrar([round(abs(a - b), 2) for a, b in zip(padre1, padre2)])}")
print("\nCada operador en esta lección recibe exactamente estos dos cromosomas")
print("y nada más. Así que hay una pregunta para hacer a cada uno de ellos: ¿cuál")
print("es el conjunto de hijos que puede producir a partir de este par?")

print(f"\nLa línea base, {ENSAYOS} extracciones:")
base = reporte_descendencia("clon (sin cruza)", cruza_clon,
                            padre1, padre2, es_legal_real)
imprimir_encabezado_reporte()
imprimir_fila_reporte(base)
print(f"\nLee la fila: salieron {base['hijos']} cromosomas diferentes,")
print(f"el {base['nuevos']:.2f}% de los espacios de genes tenía un valor que ningún padre")
print(f"tenía ahí, el {base['clones']:.2f}% de los hijos era un padre, y")
print(f"el {base['legales']:.2f}% de ellos eran legales. Así es como se ve el CERO")
print("en este instrumento. Cada fila en los pasos 2 a 7 es una desviación de esto.")

esquinas = 2 ** CANTIDAD_GENES
print(f"\nUn número a tener en cuenta. Un hijo construido solo ELIGIENDO, gen por")
print(f"gen, entre los dos padres puede ser uno de {esquinas} cromosomas -")
print(f"2 opciones en cada una de {CANTIDAD_GENES} posiciones. Los pasos 2 y 3 preguntan cuánto")
print("de ese conjunto puede alcanzar realmente cada operador de corte y cambio. El paso 4 pregunta")
print("qué sucede cuando a un operador se le permite calcular un valor en lugar de")
print("elegir uno, y la respuesta es un tipo de número completamente diferente.")


fig, ax = plt.subplots(figsize=(9, 4))
posiciones = range(1, CANTIDAD_GENES + 1)
ax.fill_between(posiciones, padre1, padre2, color="tab:blue", alpha=0.15,
                label="entre los padres")
ax.plot(posiciones, padre1, "o-", color="tab:blue", label="padre 1")
ax.plot(posiciones, padre2, "s-", color="tab:red", label="padre 2")
ax.axhline(GEN_MIN, color="grey", linewidth=0.8, linestyle="--")
ax.axhline(GEN_MAX, color="grey", linewidth=0.8, linestyle="--")
ax.set_xticks(list(posiciones))
ax.set_xlabel("posición del gen")
ax.set_ylabel("valor del gen")
ax.set_ylim(GEN_MIN - 1.5, GEN_MAX + 1.5)
ax.set_title("Los dos padres y la región entre ellos")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, linestyle=":", alpha=0.5)

FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "descendencia_01_dos_padres.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/descendencia_01_dos_padres.png")

