"""
Lección 04 - Paso 4: Aritmética sobre genes, y los primeros hijos ilegales
===========================================================================
NUEVO EN ESTE PASO: acotar(), cruza_mezcla(), cruza_lineal().

Cada operador hasta ahora elegía entre los genes de los padres. Estos dos
calculan nuevos valores - lo cual solo es posible porque un gen aquí es un
NÚMERO, y los números pueden promediarse, interpolarse y extrapolarse. Esa
única suposición sobre la representación es lo que el paso 6 quitará.

Dos consecuencias llegan juntas. La columna `genes nuevos` del instrumento
sale de cero por primera vez, y la columna `legal` sale de 100% por primera
vez: un valor calculado no tiene razón para quedarse dentro de la caja.

El paso 4 de la Lección 01 ya mostró BLX-alfa y el colapso hacia adentro con
alfa = 0. Este paso no lo vuelve a derivar; lo mide, y pone un precio a la cura.

CAMBIOS RESPECTO A descendencia_03_mas_cortes.py
Introdúcelos en este orden:
    1. acotar()                  un gen calculado no tiene razón de respetar la caja
    2. cruza_mezcla()            BLX-alfa: el primer operador que calcula un valor
    3. cruza_lineal()            el hermano determinista - siempre los mismos dos hijos
    4. la tabla de alcance       qué tan lejos más allá de los padres llega cada operador
    5. la figura de mezcla       dónde caen los hijos, alfa 0 contra alfa 0.5

Ejecútalo: python descendencia_04_mezcla.py

Primeros genes calculados (99.57% nuevos) y primeros hijos ilegales.
alfa = 0.5: 49.63% más allá, solo 50.08% legal hasta que acotar() repara.
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
        Primeros genes calculados (99.57% nuevos) y primeros hijos ilegales.
    """
    return all(GEN_MIN <= gen <= GEN_MAX for gen in cromosoma)


def mostrar(cromosoma) -> str:
    """Un cromosoma en una línea, para que padres e hijos queden alineados en columnas.

    Args:
        cromosoma.

    Returns:
        str.

    Example:
        Primeros genes calculados (99.57% nuevos) y primeros hijos ilegales.
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
        Primeros genes calculados (99.57% nuevos) y primeros hijos ilegales.
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
        Primeros genes calculados (99.57% nuevos) y primeros hijos ilegales.
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


N_PUNTOS = (2, 3)                # cuántos cortes hace el operador de n puntos


def cruza_n_puntos(padre1: list, padre2: list, n: int) -> tuple[list, list]:
    """Corta n veces e intercambia segmentos alternos.

    La cruza canónica permite cortes 1..L-1; esta implementación conserva la
    decisión de Gridin y muestrea range(1, L-1), por lo que excluye L-1.

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


TASA_UNIFORME = 0.5              # probabilidad de intercambiar cada gen de forma independiente


def cruza_uniforme(padre1: list, padre2: list,
                   tasa: float = TASA_UNIFORME) -> tuple[list, list]:
    """Decide gen a gen de qué padre proviene.

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


# --- NUEVO (1) acotar() -------------------------------------------------------
def acotar(gen: float, bajo: float = GEN_MIN, alto: float = GEN_MAX) -> float:
    """La aritmética sobre genes puede producir cualquier valor; el espacio de búsqueda tiene bordes.

    La Lección 01 introdujo esto por exactamente la misma razón. Se repite aquí
    porque a partir de este paso los operadores CALCULAN valores de genes en lugar
    de copiarlos, y un valor calculado no tiene razón de respetar la caja.

    Args:
        gen, bajo, alto.

    Returns:
        float.

    Example:
        acotar(11.4) == 10.0 en la caja [0, 10] de esta lección.
    """
    return max(bajo, min(alto, gen))
# ------------------------------------------------------------------------------


# --- NUEVO (2) cruza_mezcla() -------------------------------------------------
ALFAS_MEZCLA = (0.0, 0.5)        # el paso 4 ejecuta el operador con ambos


def cruza_mezcla(padre1: Cromosoma, padre2: Cromosoma, alfa: float,
                 mantener_dentro: bool = True) -> tuple[Cromosoma, Cromosoma]:
    """Cruza de mezcla (BLX-alfa): extrae cada gen hijo de un intervalo ampliado.

    Para el intervalo parental [m, M] de ancho d > 0, cada gen crudo es uniforme en
    [m-alfa*d, M+alfa*d]; su probabilidad exacta de salir del intervalo es
    2*alfa/(1+2*alfa). Una corrida finita mide alcance muestreado, no soporte.

    Para cada posición toma el intervalo entre los genes de los dos padres y
    lo estira un alfa de su propio ancho en ambos extremos, luego extrae
    uniformemente dentro de él. Con alfa = 0 el intervalo es exactamente el
    segmento entre los padres y cada hijo queda entre ellos - la Lección 01
    mostró que la población colapsa hacia adentro por esa razón. Con alfa > 0
    el intervalo llega más allá de ambos padres, lo que hace que este sea el
    primer operador de la lección que puede salir de la caja.

    `mantener_dentro=False` desactiva acotar(), para que el paso 4 pueda
    mostrar con qué frecuencia el operador habría salido de la caja si nadie lo
    hubiera reparado.

    Args:
        padre1, padre2, alfa, mantener_dentro.

    Returns:
        tuple[Cromosoma, Cromosoma].

    Example:
        alfa = 0.5: 99.57% genes nuevos, 50.08% legal hasta acotar().
    """
    hijo1, hijo2 = list(padre1), list(padre2)
    for i in range(len(padre1)):
        distancia = abs(padre2[i] - padre1[i])
        bajo = min(padre1[i], padre2[i]) - alfa * distancia
        alto = max(padre1[i], padre2[i]) + alfa * distancia
        hijo1[i] = round(bajo + random.random() * (alto - bajo), 2)
        hijo2[i] = round(bajo + random.random() * (alto - bajo), 2)
    if mantener_dentro:
        hijo1 = [acotar(gen) for gen in hijo1]
        hijo2 = [acotar(gen) for gen in hijo2]
    return hijo1, hijo2
# ------------------------------------------------------------------------------


# --- NUEVO (3) cruza_lineal() -------------------------------------------------
ALFA_LINEAL = 0.3                # qué tan lejos a lo largo de la línea padre-a-padre avanzar


def cruza_lineal(padre1: Cromosoma, padre2: Cromosoma,
                 alfa: float = ALFA_LINEAL) -> tuple[Cromosoma, Cromosoma]:
    """Avanza una fracción fija alfa a lo largo de la línea que une a los dos padres.

    El hermano determinista de la mezcla: no hay extracción aleatoria en ninguna
    parte, así que el mismo par de padres siempre produce el mismo par de hijos.
    Calcula nuevos valores de gen - y solo puede calcular estos dos.

    Args:
        padre1, padre2, alfa.

    Returns:
        tuple[Cromosoma, Cromosoma].

    Example:
        100.00% genes nuevos y exactamente 2 hijos alcanzables.
    """
    hijo1, hijo2 = list(padre1), list(padre2)
    for i in range(len(padre1)):
        diferencia = padre2[i] - padre1[i]
        hijo1[i] = acotar(round(padre1[i] + alfa * diferencia, 2))
        hijo2[i] = acotar(round(padre2[i] - alfa * diferencia, 2))
    return hijo1, hijo2
# ------------------------------------------------------------------------------


padre1, padre2 = crear_padres()
print(f"padre 1   {mostrar(padre1)}")
print(f"padre 2   {mostrar(padre2)}")


# --- NUEVO (4) la tabla de alcance --------------------------------------------
def proporcion_fuera(operador: Operador) -> float:
    """% de posiciones de gen que tienen un valor fuera del intervalo abarcado por los padres.

    El instrumento del paso 1 cuenta valores NUEVOS. Este cuenta valores
    que están MÁS ALLÁ - la diferencia entre interpolar entre los padres
    y extrapolar más allá de ellos, que es todo el argumento para alfa > 0.

    Args:
        operador.

    Returns:
        float.

    Example:
        Primeros genes calculados (99.57% nuevos) y primeros hijos ilegales.
    """
    random.seed(SEMILLA)
    fuera = posiciones = 0
    for _ in range(ENSAYOS):
        for hijo in operador(padre1, padre2):
            for indice, gen in enumerate(hijo):
                posiciones += 1
                bajo = min(padre1[indice], padre2[indice])
                alto = max(padre1[indice], padre2[indice])
                if gen < bajo or gen > alto:
                    fuera += 1
    return 100 * fuera / posiciones


operadores = [
    ("clon (sin cruza)", cruza_clon),
    (f"uniforme (tasa {TASA_UNIFORME})", cruza_uniforme),
]
for alfa in ALFAS_MEZCLA:
    operadores.append((f"mezcla alfa={alfa}, sin acotar",
                       lambda a, b, alfa=alfa: cruza_mezcla(a, b, alfa,
                                                            mantener_dentro=False)))
operadores.append((f"mezcla alfa={ALFAS_MEZCLA[-1]} + acotar",
                   lambda a, b: cruza_mezcla(a, b, ALFAS_MEZCLA[-1])))
operadores.append((f"lineal alfa={ALFA_LINEAL}", cruza_lineal))

print(f"\nLos operadores aritméticos, {ENSAYOS} extracciones cada uno:")
filas = [reporte_descendencia(etiqueta, operador, padre1, padre2, es_legal_real)
         for etiqueta, operador in operadores]
for fila, (etiqueta, operador) in zip(filas, operadores):
    fila["fuera"] = proporcion_fuera(operador)
imprimir_encabezado()
for fila in filas:
    imprimir_fila(fila)

print(f"\n{'operador':28s} {'genes más allá de los padres':>26}")
for fila in filas:
    print(f"{fila['etiqueta']:28s} {fila['fuera']:25.2f}%")

por_etiqueta = {fila["etiqueta"]: fila for fila in filas}
plano = por_etiqueta[f"mezcla alfa={ALFAS_MEZCLA[0]}, sin acotar"]
amplio = por_etiqueta[f"mezcla alfa={ALFAS_MEZCLA[-1]}, sin acotar"]
reparado = por_etiqueta[f"mezcla alfa={ALFAS_MEZCLA[-1]} + acotar"]
lineal = por_etiqueta[f"lineal alfa={ALFA_LINEAL}"]

print(f"\nLa columna `genes nuevos` ha salido de cero por primera vez:")
print(f"{amplio['nuevos']:.2f}% de las posiciones de gen tienen un valor que ningún padre tenía. No")
print("el 100%, porque una extracción que se redondea a dos decimales puede caer exactamente en")
print("el valor de un padre; pero en la práctica todo gen es calculado, no")
print(f"copiado, y el operador alcanzó {amplio['hijos']} hijos distintos en")
print(f"{ENSAYOS} extracciones - eso es 2 por extracción, todos diferentes. El")
print("conjunto alcanzable ya no es algo que puedas enumerar.")

print(f"\nalfa = {ALFAS_MEZCLA[0]} y alfa = {ALFAS_MEZCLA[-1]} difieren en un")
print("número y en un comportamiento:")
print(f"    alfa = {ALFAS_MEZCLA[0]}:  {plano['fuera']:5.2f}% de genes más allá de "
      f"los padres, {plano['legal']:6.2f}% legal")
print(f"    alfa = {ALFAS_MEZCLA[-1]}:  {amplio['fuera']:5.2f}% de genes más allá de "
      f"los padres, {amplio['legal']:6.2f}% legal")
print(f"Con alfa = {ALFAS_MEZCLA[0]} los hijos siempre están entre los")
print("padres. Ese es el colapso que mostró la Lección 01: una población criada así")
print("solo puede encogerse hacia su propio centro, y cada hijo es legal")
print("precisamente porque nunca va a ningún lugar nuevo.")
print(f"Con alfa = {ALFAS_MEZCLA[-1]} el operador finalmente explora - y")
print(f"rompe la caja de inmediato: solo {amplio['legal']:.2f}% de sus hijos")
print("son legales. Exploración y legalidad son el mismo intercambio aquí, que es")
print(f"la razón por la que existe acotar(). La fila reparada es legal {reparado['legal']:.2f}% de")
print(f"las veces y aun así pone {reparado['fuera']:.2f}% de genes más allá de los")
print("padres, así que la reparación casi no cuesta nada de alcance.")

print(f"\nLa cruza lineal es la extraña: {lineal['nuevos']:.2f}% de genes nuevos,")
print(f"ninguno de ellos más allá de los padres ({lineal['fuera']:.2f}%), y exactamente")
print(f"{lineal['hijos']} hijos alcanzables. Calcula valores - y solo puede")
print("calcular estos dos, porque no hay ninguna extracción aleatoria en ella.")
print("Nuevo no es lo mismo que variado.")
# ------------------------------------------------------------------------------


# --- NUEVO (5) la figura de mezcla -------------------------------------------
posicion = max(range(CANTIDAD_GENES), key=lambda i: abs(padre1[i] - padre2[i]))
fig, ejes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for eje, alfa in zip(ejes, ALFAS_MEZCLA):
    random.seed(SEMILLA)
    valores = []
    for _ in range(ENSAYOS):
        for hijo in cruza_mezcla(padre1, padre2, alfa, mantener_dentro=False):
            valores.append(hijo[posicion])
    eje.hist(valores, bins=40, color="tab:green", alpha=0.75)
    eje.axvline(padre1[posicion], color="tab:blue", linewidth=2, label="padre 1")
    eje.axvline(padre2[posicion], color="tab:red", linewidth=2, label="padre 2")
    eje.axvline(GEN_MIN, color="black", linestyle="--", linewidth=1,
                label="pared de la caja")
    fuera = sum(1 for v in valores
                if v < min(padre1[posicion], padre2[posicion])
                or v > max(padre1[posicion], padre2[posicion]))
    eje.set_title(f"alfa = {alfa}: {100 * fuera / len(valores):.1f}% más allá "
                  f"de los padres", fontsize=10)
    eje.set_xlabel(f"valor extraído para el gen {posicion + 1}")
    eje.grid(True, linestyle=":", alpha=0.5)
ejes[0].set_ylabel("extracciones")
ejes[0].legend(fontsize=8)

FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "descendencia_04_mezcla.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/descendencia_04_mezcla.png")
# ------------------------------------------------------------------------------

