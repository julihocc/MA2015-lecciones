"""
Lección 04 - Paso 3: Más puntos de corte, y lo que la familia nunca puede hacer
=================================================================================
NUEVO EN ESTE PASO: cruza_n_puntos(), cruza_uniforme(), y la tabla de la familia.

Si un corte es bueno, ¿son tres mejor? Medido en este par la respuesta es no,
y la razón vale más que la respuesta. Lo que separa a los miembros de esta
familia no es cuántos cortes hacen sino cuántas de las 64 combinaciones de los
genes de los padres pueden realmente alcanzar: la cruza uniforme llega a todas,
los operadores de n puntos llegan a unas pocas.

También hay algo que ninguno de ellos puede hacer, y la columna `genes nuevos`
lo dice en cada fila.

CAMBIOS RESPECTO A descendencia_02_un_punto.py
Introdúcelos en este orden:
    1. cruza_n_puntos()          más cortes, la misma idea
    2. cruza_uniforme()          una decisión de corte por gen, el límite de 'más cortes'
    3. la tabla de alcance       toda la familia con el instrumento del paso 1
    4. la figura de patrones     cada patrón de herencia alcanzable, como cuadrícula

Ejecútalo: python descendencia_03_mas_cortes.py

El alcance no es monótono: un punto 10, 2 puntos 12, 3 puntos 8, uniforme
64 de 64. El precio de la uniforme: 3.20% de sus hijos son un padre.
"""
import random
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt


SEMILLA = 3                         # semilla de Gridin para el capítulo 4; cada medición la reinicia
CANTIDAD_GENES = 6                  # un cromosoma tiene seis números reales
GEN_MIN, GEN_MAX = 0.0, 10.0       # la caja dentro de la cual debe permanecer un gen
ENSAYOS = 2000                      # extracciones por medición, para porcentajes estables
FIGURAS = Path(__file__).resolve().parent.parent / "figures"

Cromosoma = list[float]
Operador = Callable[[list, list], tuple[list, list]]
Legalidad = Callable[[list], bool]


def crear_padres() -> tuple[Cromosoma, Cromosoma]:
    """Los dos padres que recibe cada operador de esta lección.

    Son los padres del capítulo 4 de Gridin - misma semilla, mismos seis genes -
    de modo que cualquier cosa impresa aquí puede compararse directamente con el libro.
    random.seed() se llama dentro de la función en lugar de a nivel de módulo
    para que cada medición empiece desde el mismo par, independientemente de lo que se ejecutó antes.

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

    Esa es la única restricción sobre esta representación, y es lo que permite
    que las tablas de abajo llamen a un hijo *ilegal*: un operador que sale de
    la caja ha producido algo que el problema no puede evaluar.

    Args:
        cromosoma.

    Returns:
        bool.

    Example:
        El alcance no es monótono: un punto 10, 2 puntos 12, 3 puntos 8, uniforme
    """
    return all(GEN_MIN <= gen <= GEN_MAX for gen in cromosoma)


def mostrar(cromosoma) -> str:
    """Un cromosoma en una línea, para que padres e hijos queden alineados en columnas.

    Args:
        cromosoma.

    Returns:
        str.

    Example:
        El alcance no es monótono: un punto 10, 2 puntos 12, 3 puntos 8, uniforme
    """
    partes = [f"{gen:6.2f}" if isinstance(gen, float) else f"{gen:>6d}"
              for gen in cromosoma]
    return "[" + " ".join(partes) + "]"


def cruza_clon(padre1: list, padre2: list) -> tuple[list, list]:
    """La base de cero alcance: los hijos SON los padres.

    Nadie usaría esto. Es el patrón de referencia: cada fila de cada tabla
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
    """Ejecuta un operador `ensayos` veces sobre el mismo par y describe la nube.

    Cuatro números, y toda la lección se lee de ellos:

      hijos     cuántos cromosomas DIFERENTES salieron. Para los operadores de
                cortar-e-intercambiar, éste es el conjunto completo alcanzable,
                suficientemente pequeño para listarse; para los aritméticos es
                simplemente el número de extracciones, que es el punto.
      nuevos    % de posiciones de gen que tienen un valor que NINGÚN padre tenía
                en esa posición - valores que el operador calculó en lugar de copiar.
      clon      % de hijos que son una copia exacta de un padre: el operador
                se ejecutó y no pasó nada.
      legal     % de hijos que el problema puede evaluar.

    Siempre reiniciado desde SEMILLA, para que las filas de una tabla sean comparables
    por construcción y no por esperanza.

    Args:
        etiqueta, operador, padre1, padre2, es_legal, ensayos.

    Returns:
        dict.

    Example:
        2000 extracciones desde SEMILLA = 3; un punto alcanza 10 hijos.
    """
    random.seed(SEMILLA)
    vistos: set[tuple] = set()
    posiciones = valores_nuevos = clones = legales = producidos = 0
    for _ in range(ensayos):
        for hijo in operador(padre1, padre2):
            producidos += 1
            vistos.add(tuple(hijo))
            if list(hijo) == list(padre1) or list(hijo) == list(padre2):
                clones += 1
            if es_legal(hijo):
                legales += 1
            for indice, gen in enumerate(hijo):
                posiciones += 1
                if gen != padre1[indice] and gen != padre2[indice]:
                    valores_nuevos += 1
    return {"etiqueta": etiqueta, "hijos": len(vistos), "nuevos": 100 * valores_nuevos / posiciones,
            "clon": 100 * clones / producidos, "legal": 100 * legales / producidos,
            "alcanzables": vistos}


def imprimir_encabezado() -> None:
    """Un encabezado para cada tabla de comparación de los pasos 1 al 7.

    Args:
        (sin argumentos).

    Returns:
        None. Imprime o guarda figura.

    Example:
        El alcance no es monótono: un punto 10, 2 puntos 12, 3 puntos 8, uniforme
    """
    print(f"{'operador':28s} {'hijos':>8} {'genes':>7} {'clon':>7} {'legal':>7}")
    print(f"{'':28s} {'alcanz.':>8} {'nuevos':>7} {'de un':>7} {'':>7}")
    print(f"{'':28s} {'':>8} {'':>7} {'padre':>7} {'':>7}")


def imprimir_fila(fila: dict) -> None:
    """imprimir_fila.

    Args:
        fila.

    Returns:
        None. Imprime o guarda figura.

    Example:
        El alcance no es monótono: un punto 10, 2 puntos 12, 3 puntos 8, uniforme
    """
    print(f"{fila['etiqueta']:28s} {fila['hijos']:8d} {fila['nuevos']:6.2f}% "
          f"{fila['clon']:6.2f}% {fila['legal']:6.2f}%")


def cruza_un_punto(padre1: list, padre2: list) -> tuple[list, list]:
    """Corta ambos padres en el mismo punto e intercambia las colas.

    La cruza más antigua que existe, y la que dibuja cada libro de texto. Nota lo que
    puede y no puede hacer: el gen i de un hijo es el gen i de uno de los padres,
    nunca nada más. El operador elige QUÉ padre, nunca QUÉ valor.

    Args:
        padre1, padre2.

    Returns:
        tuple[list, list].

    Example:
        2000 extracciones producen 10 hijos distintos; 0.00% genes nuevos.
    """
    punto = random.randint(1, len(padre1) - 1)
    hijo1, hijo2 = list(padre1), list(padre2)
    hijo1[punto:], hijo2[punto:] = padre2[punto:], padre1[punto:]
    return hijo1, hijo2


# --- NUEVO (1) cruza_n_puntos() -----------------------------------------------
N_PUNTOS = (2, 3)                # cuántos cortes hace el operador de n puntos


def cruza_n_puntos(padre1: list, padre2: list, n: int) -> tuple[list, list]:
    """Corta n veces e intercambia segmentos alternos.

    Misma idea que un punto, más cortes. Las posiciones de corte se toman de
    range(1, len - 1), que es la elección de Gridin: el último gen nunca puede
    iniciar un segmento. Tenlo en mente al leer los conteos alcanzables - son
    conteos para ESTA implementación, no para la idea en abstracto.

    Args:
        padre1, padre2, n.

    Returns:
        tuple[list, list].

    Example:
        2 puntos alcanzan 12 hijos; 3 puntos alcanzan 8.
    """
    # Gridin extrae de range(1, len-1): el último gen nunca abre un segmento.
    puntos = sorted(random.sample(range(1, len(padre1) - 1), n) + [0, len(padre1)])
    hijo1, hijo2 = list(padre1), list(padre2)
    for i in range(n + 1):
        if i % 2 == 0:
            continue
        hijo1[puntos[i]:puntos[i + 1]] = padre2[puntos[i]:puntos[i + 1]]
        hijo2[puntos[i]:puntos[i + 1]] = padre1[puntos[i]:puntos[i + 1]]
    return hijo1, hijo2
# ------------------------------------------------------------------------------


# --- NUEVO (2) cruza_uniforme() -----------------------------------------------
TASA_UNIFORME = 0.5              # probabilidad de intercambiar cada gen de forma independiente


def cruza_uniforme(padre1: list, padre2: list,
                   tasa: float = TASA_UNIFORME) -> tuple[list, list]:
    """Decide gen a gen de qué padre proviene.

    Este es el límite de "más cortes": una decisión de corte en cada posición. Es
    el único miembro de la familia que puede alcanzar TODAS las combinaciones de los
    genes de los padres, y la tabla de abajo lo muestra haciendo exactamente eso.

    Args:
        padre1, padre2, tasa.

    Returns:
        tuple[list, list].

    Example:
        Alcanza 64 de 64 combinaciones; 3.20% de los hijos son un padre.
    """
    hijo1, hijo2 = list(padre1), list(padre2)
    for i in range(len(padre1)):
        if random.random() < tasa:
            hijo1[i], hijo2[i] = padre2[i], padre1[i]
    return hijo1, hijo2
# ------------------------------------------------------------------------------


padre1, padre2 = crear_padres()
print(f"padre 1   {mostrar(padre1)}")
print(f"padre 2   {mostrar(padre2)}")


# --- NUEVO (3) la tabla de alcance --------------------------------------------
print(f"\nLa familia de cortar-e-intercambiar, {ENSAYOS} extracciones cada una:")
filas = [reporte_descendencia("clon (sin cruza)", cruza_clon,
                              padre1, padre2, es_legal_real),
         reporte_descendencia("un punto", cruza_un_punto,
                              padre1, padre2, es_legal_real)]
for n in N_PUNTOS:
    filas.append(reporte_descendencia(f"{n} puntos",
                                      lambda a, b, n=n: cruza_n_puntos(a, b, n),
                                      padre1, padre2, es_legal_real))
filas.append(reporte_descendencia(f"uniforme (tasa {TASA_UNIFORME})", cruza_uniforme,
                                  padre1, padre2, es_legal_real))
imprimir_encabezado()
for fila in filas:
    imprimir_fila(fila)

por_etiqueta = {fila["etiqueta"]: fila for fila in filas}
esquinas = 2 ** CANTIDAD_GENES
un_punto = por_etiqueta["un punto"]
uniforme = por_etiqueta[f"uniforme (tasa {TASA_UNIFORME})"]
cortes = [por_etiqueta[f"{n} puntos"] for n in N_PUNTOS]

print(f"\nLa columna `genes nuevos` es {max(fila['nuevos'] for fila in filas):.2f}% en")
print("cada fila. Ningún miembro de esta familia calcula nunca un valor de gen; la")
print("familia entera no hace más que elegir un padre por posición. Esa es la")
print("definición de la familia, y la razón por la que es segura en cualquier")
print("representación cuya legalidad es una propiedad de cada gen por sí solo.")

print("\nLa columna `hijos alcanzados` es donde difieren, y no hace lo que")
print("la frase 'más puntos de cruza' sugiere:")
for fila in [un_punto] + cortes + [uniforme]:
    print(f"    {fila['etiqueta']:24s} alcanza {fila['hijos']:3d} de {esquinas}")
peor = min([un_punto] + cortes, key=lambda fila: fila["hijos"])
print(f"\nMás cortes no es mayor alcance. {peor['etiqueta']} alcanza solo")
print(f"{peor['hijos']}, menos que los {un_punto['hijos']} de un punto,")
print(f"porque n puntos de corte tomados de las {CANTIDAD_GENES - 2} posiciones interiores")
print("admiten menos patrones de segmentos distintos que un único corte libre.")
print("El conteo depende de la longitud del cromosoma, así que no lo memorices")
print("- la lección es que 'número de cortes' no es la cantidad que importa.")

print(f"\nLa cruza uniforme alcanza {uniforme['hijos']}, que es exactamente")
print(f"{esquinas}: cada combinación de los genes de los padres, el conjunto completo.")
print("Es el límite de 'más cortes' - una decisión de corte en cada posición. El")
print(f"precio se imprime a su lado: {uniforme['clon']:.2f}% de sus hijos son un")
print("padre devuelto tal cual, porque 'intercambiar nada' e 'intercambiar todo'")
print(f"son 2 de los {esquinas} patrones de los que extrae. Conserva ese número; el paso 6")
print("lo convierte en la fila más sorprendente de la lección.")
# ------------------------------------------------------------------------------


# --- NUEVO (4) la figura de patrones ------------------------------------------
fig, ejes = plt.subplots(1, 3, figsize=(12, 4.5))
for eje, fila in zip(ejes, [un_punto, cortes[-1], uniforme]):
    cuadricula = [[0 if gen == padre1[i] else 1 for i, gen in enumerate(hijo)]
                  for hijo in sorted(fila["alcanzables"])]
    eje.imshow(cuadricula, aspect="auto", cmap="coolwarm", vmin=0, vmax=1,
               interpolation="nearest")
    eje.set_title(f"{fila['etiqueta']}\n{len(cuadricula)} de {2 ** CANTIDAD_GENES} patrones",
                  fontsize=10)
    eje.set_xticks(range(CANTIDAD_GENES))
    eje.set_xticklabels(range(1, CANTIDAD_GENES + 1))
    eje.set_xlabel("posición del gen")
    eje.set_yticks([])
ejes[0].set_ylabel("un hijo alcanzable por fila")
fig.suptitle("De qué padre viene cada gen: azul = padre 1, rojo = padre 2",
             fontsize=10)

FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "descendencia_03_mas_cortes.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/descendencia_03_mas_cortes.png")
# ------------------------------------------------------------------------------

