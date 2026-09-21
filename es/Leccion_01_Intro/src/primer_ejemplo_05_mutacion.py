"""
Lección 01 - Paso 5: Mutación
==============================
NUEVO EN ESTE PASO: mutacion_gaussiana() y mutar().

La cruza recombina lo que la población ya tiene. Una vez que todos los
individuos están sobre la misma colina, todos los hijos caen también en esa
colina. La mutación es el único operador que puede abandonarla.

CAMBIOS RESPECTO A primer_ejemplo_04_cruza.py
Introdúcelos en este orden:
    1. mutacion_gaussiana()   el único operador que puede inventar un valor que nadie tenía
    2. mutar()                envuélvelo para objetos Individuo

Ejecútalo:  python primer_ejemplo_05_mutacion.py

Desde una población convergida en x = -4.6, sigma = 1.0 llega a la colina
global 0 veces de 2000; sigma = 3.0 llega 110 veces.
"""
import random
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 52
GEN_MIN, GEN_MAX = -10.0, 10.0
MUTACION_MU, MUTACION_SIGMA = 0.0, 1.0
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


def objetivo(x):
    """La función que queremos MAXIMIZAR.

    Argumentos:
        x: un real o un arreglo de numpy.

    Devuelve:
        sen(x) - 0.2 * |x|. Máximo global cerca de x = +1.372, f = +0.706.

    Ejemplo:
        objetivo(-4.6) ≈ +0.073 (colina local de este experimento).
    """
    return np.sin(x) - 0.2 * abs(x)


def acotar(gen, minimo=GEN_MIN, maximo=GEN_MAX):
    """Proyecta un gen al intervalo cerrado del paisaje.

    Argumentos:
        gen: gen crudo, posiblemente fuera de [-10, 10].
        minimo, maximo: extremos cerrados.

    Devuelve:
        gen proyectado a [minimo, maximo].

    Ejemplo:
        acotar(11.4) == 10.0.
    """
    return max(minimo, min(maximo, gen))


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


# --- NUEVO (1) mutacion_gaussiana() -------------------------------------------
def mutacion_gaussiana(gen, mu, sigma):
    """Suma al gen un ruido extraído de N(mu, sigma).

    sigma es el tamaño típico de paso. Un valor pequeño hace menos frecuentes
    los saltos largos; uno grande también puede dificultar el refinamiento local.

    Argumentos:
        gen: valor actual.
        mu: media del ruido (aquí 0.0).
        sigma: desviación estándar; el alcance del salto.

    Devuelve:
        gen + N(mu, sigma), acotado a [-10, 10].

    Ejemplo:
        Desde x = -4.6, sigma = 1.0 llega a la colina global 0/2000 veces;
        sigma = 3.0 llega 110/2000.
    """
    return acotar(gen + random.gauss(mu, sigma))
# ------------------------------------------------------------------------------


# --- NUEVO (2) mutar() --------------------------------------------------------
def mutar(individuo):
    """Envuelve mutacion_gaussiana() para objetos Individuo.

    Argumentos:
        individuo: candidato a perturbar.

    Devuelve:
        Un Individuo nuevo; el original no se modifica.

    Ejemplo:
        Con sigma = 1.0, una población en x = -4.6 no alcanza x = +1.38.
    """
    return Individuo([mutacion_gaussiana(individuo.gen, MUTACION_MU, MUTACION_SIGMA)])
# ------------------------------------------------------------------------------


random.seed(SEMILLA)

# Una población atrapada en la misma colina local, como ocurre al final de una
# corrida. La colina cerca de x = -4.6 llega a f = +0.073; la colina global está
# en x = +1.38, a seis unidades de distancia, y alcanza f = +0.706.
ATRAPADOS_EN = -4.6
COLINA_GLOBAL = 1.38

atrapados = [Individuo([ATRAPADOS_EN + random.uniform(-0.05, 0.05)]) for _ in range(6)]
print("Una población convergida, toda sobre la misma colina local:")
for ind in atrapados:
    print("   ", ind)

a, b = atrapados[0], atrapados[1]
medio = (a.gen + b.gen) / 2
print(f"\nUn promedio de estos dos individuos se queda en la colina:")
print(f"    un hijo cerca del promedio cae en x={medio:+.3f}, f={objetivo(medio):+.3f}")
print("    Progenitores muy agrupados dan poco alcance a esta cruza.")

# ----- experimento: ¿hasta dónde alcanza una sola mutación? -------------------
print(f"\nLa colina global está a {abs(COLINA_GLOBAL - ATRAPADOS_EN):.1f} unidades.")
print("¿Con qué frecuencia llega hasta allá UNA SOLA mutación?\n")

for sigma in (1.0, 3.0):
    random.seed(SEMILLA)
    llegadas = 0
    intentos = 2000
    for _ in range(intentos):
        m = mutacion_gaussiana(ATRAPADOS_EN, MUTACION_MU, sigma)
        if abs(m - COLINA_GLOBAL) < 1.5:      # radio de la cuenca, no un umbral de aptitud
            llegadas += 1
    print(f"    sigma = {sigma}:  {llegadas:4d} de {intentos} mutaciones "
          f"({100*llegadas/intentos:5.1f}%) caen en la colina global")

print("\nsigma controla el alcance típico de la mutación. En esta muestra finita,")
print("sigma = 1.0 no produjo llegadas; eso no vuelve imposible un salto gaussiano")
print("más largo. Un sigma muy grande también dificulta el refinamiento local.")
print("\nUn intento no determina la corrida completa. El ciclo del paso 6 repite")
print("la variación durante varias generaciones y crea más oportunidades de")
print("observar saltos poco frecuentes.")


def densidad_gaussiana(malla, mu, sigma):
    """Densidad N(mu, sigma) para dibujar el alcance, no para muestrear.

    Argumentos:
        malla: abscisas.
        mu, sigma: parámetros de la gaussiana.

    Devuelve:
        Densidad evaluada en cada punto de la malla.

    Ejemplo:
        En sigma = 1.0 la masa casi no cubre x = +1.38; en sigma = 3.0 sí.
    """
    return np.exp(-0.5 * ((malla - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


x = np.linspace(GEN_MIN, GEN_MAX, 400)
y = objetivo(x)
pdf1 = densidad_gaussiana(x, ATRAPADOS_EN, 1.0)
pdf3 = densidad_gaussiana(x, ATRAPADOS_EN, 3.0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, color="tab:blue", alpha=0.75, label="objetivo f(x)")
ax.plot([ATRAPADOS_EN], [objetivo(ATRAPADOS_EN)], "D", color="black",
        markersize=8, zorder=4, label="atrapados aquí")
ax.axvline(ATRAPADOS_EN, color="gray", linestyle="--", alpha=0.5)
ax.axvspan(COLINA_GLOBAL - 1.5, COLINA_GLOBAL + 1.5, color="tab:green", alpha=0.12)
ax.plot([COLINA_GLOBAL], [objetivo(COLINA_GLOBAL)], "*", color="tab:green",
        markersize=14, zorder=4, label="colina global")

ax2 = ax.twinx()
ax2.fill_between(x, pdf1, color="tab:orange", alpha=0.35, label="sigma = 1.0")
ax2.plot(x, pdf1, color="tab:orange", linewidth=1.5)
ax2.fill_between(x, pdf3, color="tab:red", alpha=0.18, label="sigma = 3.0")
ax2.plot(x, pdf3, color="tab:red", linewidth=1.5)
ax2.set_ylim(0, pdf1.max() * 2.2)
ax2.set_yticks([])

manejadores, etiquetas = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(manejadores + h2, etiquetas + l2, loc="upper left")
ax.set_title("Alcance de la mutación: escapar de un óptimo local")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.grid(True, linestyle=":", alpha=0.5)
FIGURAS.mkdir(exist_ok=True)
plt.savefig(FIGURAS / "primer_ejemplo_05_mutacion.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigura guardada en {FIGURAS}/primer_ejemplo_05_mutacion.png")

