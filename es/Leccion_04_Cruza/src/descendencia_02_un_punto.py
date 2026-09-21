"""
Lección 04 - Paso 2: Cruza de un punto, y el tamaño de su alcance
==================================================================
NUEVO EN ESTE PASO: cruza_un_punto(), y la enumeración de sus hijos.

El operador de libro de texto: cortar a ambos padres en el mismo punto, intercambiar las colas. Vale
la pena comenzar aquí no porque sea el mejor sino porque su conjunto alcanzable es lo
suficientemente pequeño como para escribirlo completo - que es la única manera de ver lo que el
operador puede y no puede hacer.

CAMBIOS RESPECTO A descendencia_01_dos_padres.py
Introdúcelos en este orden:
    1. cruza_un_punto()          cortar una vez, intercambiar colas - el operador más antiguo que hay
    2. el conjunto alcanzable    enumerar cada hijo que este operador puede producir de este par
    3. la figura de cortes       los cinco hijos dibujados sobre los padres de los que provienen

Ejecútalo:  python descendencia_02_un_punto.py

2000 extracciones producen 10 hijos distintos; los 5 cortes dan los mismos 10.
0.00% genes nuevos: elige qué padre, nunca qué valor.
"""
import random
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt


SEMILLA = 3                         # Semilla de Gridin para el capítulo 4; cada medición se restablece a ella
CANTIDAD_GENES = 6                   # un cromosoma son seis números reales
GEN_MIN, GEN_MAX = 0.0, 10.0   # la caja dentro de la cual un gen tiene que mantenerse
ENSAYOS = 2000                    # extracciones por medición, para porcentajes estables
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
        2000 extracciones producen 10 hijos distintos; los 5 cortes dan los mismos 10.
    """
    return all(GEN_MIN <= gen <= GEN_MAX for gen in cromosoma)


def mostrar(cromosoma) -> str:
    """Un cromosoma en una línea, para que padres e hijos se alineen en columnas.

    Args:
        cromosoma.

    Returns:
        str.

    Example:
        2000 extracciones producen 10 hijos distintos; los 5 cortes dan los mismos 10.
    """
    parts = [f"{gen:6.2f}" if isinstance(gen, float) else f"{gen:>6d}"
             for gen in cromosoma]
    return "[" + " ".join(parts) + "]"


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
        2000 extracciones producen 10 hijos distintos; los 5 cortes dan los mismos 10.
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
        2000 extracciones producen 10 hijos distintos; los 5 cortes dan los mismos 10.
    """
    print(f"{fila['etiqueta']:28s} {fila['hijos']:8d} {fila['nuevos']:6.2f}% "
          f"{fila['clones']:6.2f}% {fila['legales']:6.2f}%")


# --- NUEVO (1) cruza_un_punto() --------------------------------------------
def cruza_un_punto(padre1: list, padre2: list) -> tuple[list, list]:
    """Corta a ambos padres en el mismo punto y cambia las colas.

    La cruza más antigua que existe, y la que todo libro de texto dibuja. Nota lo
    que puede y no puede hacer: el gen i de un hijo es el gen i de un padre o el gen i
    del otro, nunca otra cosa. El operador elige QUÉ padre, nunca QUÉ valor.

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
# ------------------------------------------------------------------------------


padre1, padre2 = crear_padres()
print(f"padre 1   {mostrar(padre1)}")
print(f"padre 2   {mostrar(padre2)}")


# --- NUEVO (2) el conjunto alcanzable ------------------------------------------------
print(f"\nCruza de un punto, {ENSAYOS} extracciones:")
base = reporte_descendencia("clon (sin cruza)", cruza_clon,
                            padre1, padre2, es_legal_real)
un_punto = reporte_descendencia("un-punto", cruza_un_punto,
                             padre1, padre2, es_legal_real)
imprimir_encabezado_reporte()
for fila in (base, un_punto):
    imprimir_fila_reporte(fila)

# El conjunto alcanzable es lo suficientemente pequeño como para escribirlo a mano:
# el punto de corte es un entero de 1 a CANTIDAD_GENES - 1, y nada más es aleatorio.
enumerados = set()
print(f"\nSolo hay {CANTIDAD_GENES - 1} lugares para cortar, así que todo el conjunto")
print("alcanzable se puede listar:")
for punto in range(1, CANTIDAD_GENES):
    hijo_1 = padre1[:punto] + padre2[punto:]
    hijo_2 = padre2[:punto] + padre1[punto:]
    enumerados.update({tuple(hijo_1), tuple(hijo_2)})
    print(f"    corte tras gen {punto}:  {mostrar(hijo_1)}   {mostrar(hijo_2)}")

print(f"\n{ENSAYOS} extracciones produjeron {un_punto['hijos']} hijos distintos;")
print(f"enumerar los puntos de corte da {len(enumerados)}. Mismo conjunto: "
      f"{'sí' if enumerados == un_punto['alcanzables'] else 'NO'}.")
print(f"Así que el alcance completo de este operador es {len(enumerados)} cromosomas, de")
print(f"los {2 ** CANTIDAD_GENES} que elegir libremente entre los padres permite.")
print(f"\nY el {un_punto['nuevos']:.2f}% de las ranuras de genes tenían un valor que ningún padre")
print("tenía en esa ranura. Esa es la lección a extraer: la cruza de un punto")
print("RECOMBINA, no INVENTA. El gen i de un hijo es el gen i de un padre")
print("o el gen i del otro. El operador elige qué padre, nunca qué")
print(f"valor - por lo que también cada hijo es legal ({un_punto['legales']:.2f}%)")
print("aunque el operador no sabe nada sobre la caja.")
# ------------------------------------------------------------------------------


# --- NUEVO (3) la figura de cortes ---------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
posiciones = range(1, CANTIDAD_GENES + 1)
ax.fill_between(posiciones, padre1, padre2, color="tab:blue", alpha=0.12)
for punto in range(1, CANTIDAD_GENES):
    hijo = padre1[:punto] + padre2[punto:]
    ax.plot(posiciones, hijo, "-", color="tab:green", alpha=0.8, linewidth=1.0)
    ax.annotate(f"corte {punto}", (punto + 0.5, hijo[punto]), fontsize=7,
                color="tab:green", textcoords="offset points", xytext=(0, 6))
ax.plot(posiciones, padre1, "o-", color="tab:blue", linewidth=2, label="padre 1")
ax.plot(posiciones, padre2, "s-", color="tab:red", linewidth=2, label="padre 2")
ax.plot([], [], "-", color="tab:green",
        label="los cinco hijos que empiezan como el padre 1")
ax.set_xticks(list(posiciones))
ax.set_xlabel("posición del gen")
ax.set_ylabel("valor del gen")
ax.set_title("Cruza de un punto: cada hijo sigue a un padre y luego al otro")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, linestyle=":", alpha=0.5)

FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "descendencia_02_un_punto.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/descendencia_02_un_punto.png")

# Misma banda, segunda figura: el mecanismo que el perfil no muestra.
CORTE_ESQUEMA = 3
hijo_de_p1 = padre1[:CORTE_ESQUEMA] + padre2[CORTE_ESQUEMA:]
hijo_de_p2 = padre2[:CORTE_ESQUEMA] + padre1[CORTE_ESQUEMA:]
filas = (
    ("padre 1", padre1, "tab:blue"),
    ("padre 2", padre2, "tab:red"),
    ("hijo 1  (corte tras el gen 3)", hijo_de_p1, "tab:green"),
    ("hijo 2  (corte tras el gen 3)", hijo_de_p2, "tab:olive"),
)
fig2, ejes = plt.subplots(4, 1, figsize=(9, 4.8), sharex=True)
for eje, (etiqueta, genes, color) in zip(ejes, filas):
    for indice, gen in enumerate(genes):
        del_padre1 = gen == padre1[indice]
        del_padre2 = gen == padre2[indice]
        cara = "tab:blue" if del_padre1 and not del_padre2 else (
            "tab:red" if del_padre2 and not del_padre1 else color
        )
        eje.bar(indice + 1, 1, width=0.9, color=cara, edgecolor="black", linewidth=0.6)
        eje.text(indice + 1, 0.5, f"{gen:.2f}", ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold")
    eje.axvline(CORTE_ESQUEMA + 0.5, color="black", linestyle="--", linewidth=1.2)
    eje.set_ylabel(etiqueta, rotation=0, ha="right", va="center", fontsize=8)
    eje.set_yticks([])
    eje.set_ylim(0, 1)
    eje.set_xlim(0.4, CANTIDAD_GENES + 0.6)
ejes[-1].set_xticks(list(range(1, CANTIDAD_GENES + 1)))
ejes[-1].set_xlabel("posición del gen")
ejes[0].set_title("Cruza de un punto: corte tras el gen 3, intercambiar las colas")
fig2.tight_layout()
fig2.savefig(FIGURAS / "descendencia_02_un_punto_esquema.png", dpi=150,
             bbox_inches="tight")
plt.close(fig2)
print(f"Esquema guardado en {FIGURAS}/descendencia_02_un_punto_esquema.png")
# ------------------------------------------------------------------------------

