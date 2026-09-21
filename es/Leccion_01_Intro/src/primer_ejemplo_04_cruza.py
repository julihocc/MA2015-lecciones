"""
Lección 01 - Paso 4: Cruza
===========================
NUEVO EN ESTE PASO: cruza_mezcla() y cruza().

La selección por sí sola solo puede copiar. La cruza es el primer operador que
crea valores que antes no estaban en la población, combinando dos progenitores.

CAMBIOS RESPECTO A primer_ejemplo_03_seleccion.py
Introdúcelos en este orden:
    1. acotar()               las propuestas extendidas deben respetar los bordes del espacio
    2. cruza_mezcla()         el operador mismo, trabajando sobre valores crudos de gen
    3. cruza()                envuélvelo para que reciba y devuelva objetos Individuo

Ejecútalo:  python primer_ejemplo_04_cruza.py

Con alfa = 1.0, 14 de 20 hijos caen fuera del intervalo de los
progenitores [-2, +3]. Eso es lo que compra alfa > 0.
"""
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 52
GEN_MIN, GEN_MAX = -10.0, 10.0
ALFA_MEZCLA = 1.0
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


def objetivo(x):
    """La función que queremos MAXIMIZAR.

    Argumentos:
        x: un real o un arreglo de numpy.

    Devuelve:
        sen(x) - 0.2 * |x|. Máximo global cerca de x = +1.372, f = +0.706.

    Ejemplo:
        objetivo(+1.372) ≈ +0.706.
    """
    return np.sin(x) - 0.2 * abs(x)


# --- NUEVO (1) acotar() -------------------------------------------------------
def acotar(gen, minimo=GEN_MIN, maximo=GEN_MAX):
    """Deja un gen dentro del paisaje después de una cruza de mezcla.

    Para progenitores y alfa fijos, la cruza propone valores en un intervalo
    extendido finito. Este paisaje solo existe en [-10, 10], así que toda
    propuesta debe mantenerse dentro del dominio legal.

    Argumentos:
        gen: gen crudo.
        minimo, maximo: extremos cerrados del intervalo.

    Devuelve:
        gen proyectado a [minimo, maximo].

    Ejemplo:
        acotar(11.4) == 10.0; acotar(-2.0) == -2.0.
    """
    return max(minimo, min(maximo, gen))
# ------------------------------------------------------------------------------


class Individuo:
    """Una solución candidata: un cromosoma y su aptitud.

    Argumentos:
        lista_genes: cromosoma; aquí un solo real en [-10, 10].
    """

    def __init__(self, lista_genes):
        self.lista_genes = lista_genes
        self.aptitud = objetivo(lista_genes[0])

    @property
    def gen(self):
        """El único gen: este paisaje es unidimensional."""
        return self.lista_genes[0]

    def __repr__(self):
        return f"x={self.gen:+.3f} f={self.aptitud:+.3f}"


# --- NUEVO (2) cruza_mezcla() -------------------------------------------------
def cruza_mezcla(gen1, gen2, alfa):
    """Variante complementaria de cruza por mezcla para genes reales.

    Se extrae un `desplazamiento` al azar de [-alfa, 1+alfa]:

        hijo1 = (1 - desplazamiento) * gen1 + desplazamiento * gen2
        hijo2 = desplazamiento * gen1 + (1 - desplazamiento) * gen2

    Con alfa = 0 los hijos caen siempre ENTRE los progenitores, así que esta
    cruza no puede expandir ese intervalo parental. La dinámica de la población
    también depende de la selección, la mutación y los emparejamientos
    posteriores. Con alfa > 0, el intervalo finito se extiende hacia afuera.

    Argumentos:
        gen1, gen2: genes crudos de los dos progenitores.
        alfa: extensión más allá del intervalo de los progenitores.

    Devuelve:
        Par de genes acotados a [GEN_MIN, GEN_MAX].

    Ejemplo:
        Progenitores en -2.0 y +3.0, alfa = 1.0: 14 de 20 hijos caen fuera
        de [-2, +3].
    """
    desplazamiento = (1 + 2 * alfa) * random.random() - alfa
    hijo1 = (1 - desplazamiento) * gen1 + desplazamiento * gen2
    hijo2 = desplazamiento * gen1 + (1 - desplazamiento) * gen2
    return acotar(hijo1), acotar(hijo2)
# ------------------------------------------------------------------------------


# --- NUEVO (3) cruza() --------------------------------------------------------
def cruza(progenitor1, progenitor2):
    """Envuelve cruza_mezcla() para objetos Individuo.

    Argumentos:
        progenitor1, progenitor2: padres con un gen real cada uno.

    Devuelve:
        Dos Individuo nuevos, con aptitud recalculada.

    Ejemplo:
        Diez cruzas de x=-2.0 y x=+3.0 producen 14 hijos fuera de [-2, +3].
    """
    g1, g2 = cruza_mezcla(progenitor1.gen, progenitor2.gen, ALFA_MEZCLA)
    return Individuo([g1]), Individuo([g2])
# ------------------------------------------------------------------------------


random.seed(SEMILLA)
madre = Individuo([-2.0])
padre = Individuo([+3.0])

print(f"Progenitores:   {madre}   y   {padre}")
print(f"\nDiez cruzas de los MISMOS dos progenitores (alfa = {ALFA_MEZCLA}):\n")
fuera_total = 0
hijos = []
for i in range(1, 11):
    h1, h2 = cruza(madre, padre)
    hijos.extend([h1, h2])
    for hijo in (h1, h2):
        fuera = not (madre.gen <= hijo.gen <= padre.gen)
        fuera_total += fuera
        print(f"  {i:2d}. {hijo}{'   <- fuera de los progenitores' if fuera else ''}")

print(f"\n{fuera_total} de 20 hijos cayeron fuera del intervalo [-2, +3].")
print("Con alfa = 0 ese número sería cero: esta cruza no podría expandir")
print("el intervalo parental. Compruébalo: pon ALFA_MEZCLA = 0.0")

lo, hi = madre.gen, padre.gen
dentro = [h for h in hijos if lo <= h.gen <= hi]
fuera_hijos = [h for h in hijos if not (lo <= h.gen <= hi)]

x = np.linspace(GEN_MIN, GEN_MAX, 400)
y = objetivo(x)
plt.figure(figsize=(8, 4))
plt.plot(x, y, color="tab:blue", alpha=0.35)
plt.axvspan(lo, hi, color="gray", alpha=0.2, label="intervalo de los progenitores [-2, +3]")
plt.plot([madre.gen, padre.gen], [madre.aptitud, padre.aptitud],
         "D", color="black", label="progenitores", markersize=8, zorder=4)
plt.plot([h.gen for h in dentro], [h.aptitud for h in dentro],
         "o", color="tab:orange", label="hijos dentro", zorder=3)
plt.plot([h.gen for h in fuera_hijos], [h.aptitud for h in fuera_hijos],
         "o", color="tab:red", label="hijos fuera", zorder=3)
plt.title(f"Cruza de mezcla (alfa = {ALFA_MEZCLA})")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.legend(); plt.grid(True, linestyle=":", alpha=0.5)
FIGURAS.mkdir(exist_ok=True)
plt.savefig(FIGURAS / "primer_ejemplo_04_cruza.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigura guardada en {FIGURAS}/primer_ejemplo_04_cruza.png")

