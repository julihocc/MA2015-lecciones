"""
Lección 02 - Paso 4: Sabiendo cuándo detenerse
=========================================
NUEVO EN ESTE PASO: un run() que se detiene cuando ya nada mejora.

Cada ejecución hasta ahora se detuvo después de exactamente diez generaciones, porque
diez estaba escrito en la parte superior del archivo. El libro enumera tres formas de
terminar una ejecución: se encuentra una solución aceptable, se alcanza un conteo de
generaciones, o las mejoras se agotan. El paso 3 midió exactamente cuándo se agotan
las mejoras, por lo que la tercera condición ahora está disponible gratis.

El conteo fijo no desaparece; deja de ser el plan y se convierte en una red de
seguridad, por lo que se puede elevar a 60. Dos detalles en la implementación valen
el tiempo que toman en clase. La paciencia se cuenta contra la mejor aptitud
VISTA HASTA AHORA, no contra la generación anterior, porque este algoritmo no tiene
elitismo y su mejor puede empeorar. Y por la misma razón, la respuesta que devuelve
una ejecución no es simplemente el mejor de la última generación - la tabla del paso 3
ya muestra la mejor aptitud disminuyendo entre generaciones.

CAMBIOS RESPECTO A flujo_ag_03_metricas_poblacion.py
Introdúcelos en este orden:
    1. MAX_GENERATIONS        CAMBIADO 10 -> 60; un tope es una red, no un plan
    2. PATIENCE, MIN_IMPROVEMENT  qué cuenta como progreso y cuánto esperamos
    3. detenerse por estancamiento dentro de run(): romper cuando se agota la paciencia
    4. las ejecuciones de control volver a ejecutar con la regla apagada, y valorar lo que costó

Ejecútalo:  python flujo_ag_04_condicion_parada.py

La regla ahorra de 49 a 51 de las 60 generaciones permitidas y cuesta
menos que MIN_IMPROVEMENT en ambos problemas. En 1 de 2, el campeón ya
no está en la última generación.
"""
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 52
POPULATION_SIZE = 10
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.1
MAX_GENERATIONS = 60   # --- CAMBIADO --- 10 -> 60: una red de seguridad, no el plan
GENE_MIN, GENE_MAX = -10.0, 10.0
TOURNAMENT_SIZE, BLEND_ALPHA = 3, 1.0
MUTATION_MU, MUTATION_SIGMA = 0.0, 1.0
TARGET = 4.2
# --- NUEVO (2) PATIENCE, MIN_IMPROVEMENT --------------------------------------
# Cuánto esperamos y qué estamos dispuestos a llamar progreso. Una regla sin
# MIN_IMPROVEMENT se deja engañar por el quinto decimal y nunca se dispara; una
# regla sin PATIENCE se dispara en la primera generación desafortunada.
# PATIENCE = 0 apaga toda la regla, que es como se hacen las ejecuciones de control abajo.
PATIENCE = 5
MIN_IMPROVEMENT = 1e-4
# ------------------------------------------------------------------------------
FIGURES = Path(__file__).resolve().parent.parent / "figures"

Fitness = Callable[[float], float]


def sine_landscape(x: float) -> float:
    """El objetivo de la Lección 01, ahora solo un problema entre otros.

    Args:
        x: un real en [-10, 10].

    Returns:
        sen(x) - 0.2 * |x|. Máximo global cerca de x = +1.372, f = +0.706.
    """
    return np.sin(x) - 0.2 * abs(x)


def closeness_to_target(x: float) -> float:
    """MINIMIZAR la distancia a TARGET - escrita como una maximización.

    Un algoritmo genético siempre maximiza; no hay variante minimizadora. A
    un problema que naturalmente es una minimización se le inyecta con su signo
    invertido, y ese es todo el truco. La mejor aptitud alcanzable aquí es 0.

    Args:
        x: un real en [-10, 10].

    Returns:
        -(x - 4.2)^2. Cero solo en x = 4.2.
    """
    return -(x - TARGET) ** 2


def acotar(g: float, low: float = GENE_MIN, high: float = GENE_MAX) -> float:
    """Proyecta un gen al intervalo cerrado del paisaje.

    Args:
        g: gen crudo.
        low, high: extremos cerrados.

    Returns:
        g proyectado a [low, high].
    """
    return max(low, min(high, g))


def population_metrics(population: List["Individuo"]) -> Dict[str, float]:
    """Los cuatro números que describen a una población en lugar de a su campeón.

    `spread` es la desviación estándar de los genes. No está en el libro, y
    es el único de los cuatro que puede distinguir una población convergida de una
    simplemente afortunada: dos poblaciones pueden compartir un mejor y un promedio y no
    tener nada más en común.

    Args:
        population: generación a resumir.

    Returns:
        Dict con best_gene, best_fitness, average_fitness y spread.
    """
    genes = [ind.gen for ind in population]
    fitnesses = [ind.aptitud for ind in population]
    best = max(population, key=lambda i: i.aptitud)
    return {"best_gene": best.gen,
            "best_fitness": best.aptitud,
            "average_fitness": sum(fitnesses) / len(fitnesses),
            "spread": float(np.std(genes))}


class Individuo:
    """Una solución candidata: un cromosoma, su juez y el veredicto.

    Mantener `fitness_function` en el individuo significa que los operadores no
    necesitan que se les pase: un hijo es juzgado por lo que haya juzgado a su
    padre. Soldar la función aquí, como hizo el paso 1, es lo que hizo que todo
    el archivo fuera de un solo propósito.

    Args:
        gene_list: cromosoma; aquí un solo real.
        fitness_function: juez inyectado.
    """

    def __init__(self, gene_list: List[float], fitness_function: Fitness) -> None:
        self.gene_list = gene_list
        self.fitness_function = fitness_function
        self.aptitud = fitness_function(gene_list[0])

    @property
    def gen(self) -> float:
        """El único gen: este paisaje es unidimensional."""
        return self.gene_list[0]

    def __repr__(self) -> str:
        return f"x={self.gen:+.3f} f={self.aptitud:+.3f}"


def seleccion_torneo(population: List[Individuo], size: int) -> List[Individuo]:
    """Sin cambios, y vale la pena decir por qué: la selección compara valores
    de aptitud que ya existen. Nunca necesita saber de dónde vinieron.

    Args:
        population: lista de Individuo.
        size: competidores por torneo.

    Returns:
        Lista de la misma longitud, hecha de copias.
    """
    return [max([random.choice(population) for _ in range(size)],
                key=lambda i: i.aptitud) for _ in range(len(population))]


def create_random(fitness_function: Fitness) -> Individuo:
    """Extrae un individuo y le pega el juez que usará toda su descendencia.

    Args:
        fitness_function: problema a resolver.

    Returns:
        Individuo uniforme en [-10, 10].
    """
    return Individuo([random.uniform(GENE_MIN, GENE_MAX)], fitness_function)


def cruza(p1: Individuo, p2: Individuo) -> Tuple[Individuo, Individuo]:
    """Cruza de mezcla; el hijo hereda el juez del primer padre.

    Args:
        p1, p2: progenitores.

    Returns:
        Dos Individuo nuevos con `p1.fitness_function`.
    """
    shift = (1 + 2 * BLEND_ALPHA) * random.random() - BLEND_ALPHA
    g1 = acotar((1 - shift) * p1.gen + shift * p2.gen)
    g2 = acotar(shift * p1.gen + (1 - shift) * p2.gen)
    return (Individuo([g1], p1.fitness_function),
            Individuo([g2], p1.fitness_function))


def mutar(ind: Individuo) -> Individuo:
    """Suma ruido gaussiano y conserva el juez del padre.

    Args:
        ind: candidato a perturbar.

    Returns:
        Un Individuo nuevo con `ind.fitness_function`.
    """
    return Individuo([acotar(ind.gen + random.gauss(MUTATION_MU, MUTATION_SIGMA))],
                     ind.fitness_function)


def evolve_one_generation(
        population: List[Individuo]) -> Tuple[List[Individuo], Dict[str, int]]:
    """SELECCIONAR -> CRUZAR -> MUTAR -> reemplazar, una vez.

    Args:
        population: generación actual.

    Returns:
        (nueva población, censo).
    """
    census = {"pairs": 0, "crossed": 0, "mutated": 0, "created": 0}

    selected = seleccion_torneo(population, TOURNAMENT_SIZE)             # SELECCIONAR

    crossed: List[Individuo] = []                                        # CRUZAR
    # Pares consecutivos: el torneo ya barajó el orden al copiar.
    for p1, p2 in zip(selected[::2], selected[1::2]):
        census["pairs"] += 1
        if random.random() < CROSSOVER_PROBABILITY:
            crossed.extend(cruza(p1, p2))
            census["crossed"] += 1
            census["created"] += 2
        else:
            crossed.extend([p1, p2])

    mutated: List[Individuo] = []                                        # MUTAR
    for ind in crossed:
        if random.random() < MUTATION_PROBABILITY:
            mutated.append(mutar(ind))
            census["mutated"] += 1
            census["created"] += 1
        else:
            mutated.append(ind)

    return mutated, census                                             # reemplazar


def run(fitness_function: Fitness,
        patience: int = PATIENCE) -> Tuple[List[List[Individuo]],
                                           List[Dict[str, int]]]:
    """INICIALIZAR, luego evolucionar hasta que la condición de parada diga detenerse.

    Lee el cuerpo y nota lo que falta: sin seno, sin objetivo, sin mención de
    qué se está optimizando. Esa ausencia es el punto del paso 2.

    Args:
        fitness_function: el problema.
        patience: generaciones sin mejora antes de cortar; 0 apaga la regla.

    Returns:
        (lista de generaciones, lista de censos). Puede ser más corta que 60.

    Example:
        Ahorra de 49 a 51 de las 60 generaciones y cuesta menos que 1e-4.
    """
    random.seed(SEMILLA)
    population = [create_random(fitness_function) for _ in range(POPULATION_SIZE)]

    generations = [population]
    censuses: List[Dict[str, int]] = []
    for generation in range(1, MAX_GENERATIONS + 1):
        population, census = evolve_one_generation(population)
        generations.append(population)
        censuses.append(census)
        # --- NUEVO (3) detenerse por estancamiento ----------------------------
        # Contra el máximo histórico, no contra la generación previa: sin
        # elitismo el campeón puede empeorar y eso no es "progreso".
        bests = [max(i.aptitud for i in pop) for pop in generations]
        last_improvement = 0
        for g in range(1, len(bests)):
            if bests[g] > max(bests[:g]) + MIN_IMPROVEMENT:
                last_improvement = g
        if patience and generation - last_improvement >= patience:
            break
    return generations, censuses
# ------------------------------------------------------------------------------


PROBLEMAS = [("paisaje de seno", sine_landscape),
            ("cercania al objetivo", closeness_to_target)]

results = []
for name, fitness_function in PROBLEMAS:
    generations, censuses = run(fitness_function)
    results.append((name, fitness_function, generations))

for name, fitness_function, generations in results:
    history = [population_metrics(pop) for pop in generations]

    print(f"=== {name} " + "=" * (58 - len(name)))
    print(" gen | mejor gen | mejor aptitud | aptitud promedio | disp. genetica")
    print("-----+-----------+---------------+------------------+---------------")
    for g, m in enumerate(history):
        print(f" {g:3d} |  {m['best_gene']:+8.3f} |"
              f"      {m['best_fitness']:+8.4f} |"
              f"         {m['average_fitness']:+8.4f} |"
              f"        {m['spread']:7.3f}")

    first, last = history[0], history[-1]
    improved = [g for g in range(1, len(history))
                if history[g]["best_fitness"] > history[g - 1]["best_fitness"]]
    gap_first = first["best_fitness"] - first["average_fitness"]
    gap_last = last["best_fitness"] - last["average_fitness"]
    spreads = [m["spread"] for m in history]
    thinnest = min(range(len(spreads)), key=lambda g: spreads[g])
    after = [g for g in improved if g > thinnest]

    print(f"\nLa mejor aptitud fue de {first['best_fitness']:+.4f} a "
          f"{last['best_fitness']:+.4f}, mejorando en "
          f"{len(improved)} de las {len(history) - 1} generaciones;")
    print(f"la última mejora fue en la generación "
          f"{max(improved) if improved else 0}.")
    print(f"La aptitud promedio fue de {first['average_fitness']:+.4f} a "
          f"{last['average_fitness']:+.4f}, por lo que la brecha entre el")
    print(f"campeón y la multitud se cerró de {gap_first:.4f} a {gap_last:.4f}.")
    print(f"La dispersión genética comenzó en {spreads[0]:.3f}, tocó fondo en "
          f"{spreads[thinnest]:.3f} en la generación {thinnest},")
    print(f"y terminó en {spreads[-1]:.3f}.")
    print(f"Después de la generación {thinnest} la mejor aptitud mejoró "
          f"{len(after)} vez/veces más.")
    print("Lee la columna de dispersión por su cuenta: la selección y la cruza la gastan,")
    print("la mutación es el único operador que repone algo, y los dos no están")
    print("equilibrados. Una población cuya dispersión está cerca de cero ha dejado")
    print("de buscar - es un solo individuo, copiado, y a la selección no le queda")
    print("nada por comparar. La Lección 03 mide la presión que la gasta, y la")
    print("Lección 05 trata sobre el operador que puede reponerla.")
    print()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, fitness_function, generations) in zip(axes, results):
    history = [population_metrics(pop) for pop in generations]
    gens = range(len(history))
    ax.plot(gens, [m["best_fitness"] for m in history], "o-",
            color="tab:green", label="mejor aptitud")
    ax.plot(gens, [m["average_fitness"] for m in history], "o-",
            color="tab:orange", label="aptitud promedio")
    ax.set_title(name)
    ax.set_xlabel("generación")
    ax.set_ylabel("aptitud")
    ax.grid(True, linestyle=":", alpha=0.5)
    twin = ax.twinx()
    twin.plot(gens, [m["spread"] for m in history], "s--",
              color="tab:grey", label="disp. genética")
    twin.set_ylabel("disp. genética (desv. est.)")
    twin.set_ylim(bottom=0)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
fig.suptitle("El promedio alcanza al mejor, mientras la dispersión colapsa - y se recupera en parte")
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "flujo_ag_04_condicion_parada.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figura guardada en {FIGURES}/flujo_ag_04_condicion_parada.png")


# --- NUEVO (4) las ejecuciones de control -------------------------------------
# Una regla de parada solo vale la pena tenerla si es barata. Ponle precio: ejecuta cada
# problema nuevamente con la regla apagada, hasta el límite.
#
# Ambas columnas reportan el mejor individuo JAMÁS VISTO, no el mejor en la última
# generación. Sin elitismo esos son números diferentes, y la diferencia se
# reporta abajo - un flujo que tira a su campeón tiene que recordarlo
# en algún lugar fuera de la población.
def best_ever(generations: List[List[Individuo]]) -> Individuo:
    """El mejor individuo JAMÁS VISTO, no el de la última generación.

    Sin elitismo esos dos números difieren: el flujo puede tirar a su
    campeón y hay que recordarlo fuera de la población.

    Args:
        generations: historial completo, generación 0 incluida.

    Returns:
        El Individuo de mayor aptitud en todo el historial.

    Example:
        En 1 de 2 problemas la última generación ya no lo contiene.
    """
    return max((ind for pop in generations for ind in pop), key=lambda i: i.aptitud)


print(f"Fijando precio a la regla de parada: PATIENCE = {PATIENCE}, "
      f"MIN_IMPROVEMENT = {MIN_IMPROVEMENT}, tope = {MAX_GENERATIONS}.\n")
print("               problema | regla encendida  | regla apagada    | costo de parada")
print("------------------------+------------------+------------------+----------------")
priced = []
for name, fitness_function in PROBLEMAS:
    patient, _ = run(fitness_function, patience=PATIENCE)
    full, _ = run(fitness_function, patience=0)
    on, off = best_ever(patient), best_ever(full)
    priced.append((name, patient, full, on, off))
    print(f"{name:>23s} | gen {len(patient) - 1:2d}  {on.aptitud:+.4f} |"
          f" gen {len(full) - 1:2d}  {off.aptitud:+.4f} |"
          f" {off.aptitud - on.aptitud:+.4f}")

savings = [MAX_GENERATIONS - (len(p) - 1) for _, p, _, _, _ in priced]
costs = [off.aptitud - on.aptitud for _, _, _, on, off in priced]
print(f"\nLa regla ahorró de {min(savings)} a {max(savings)} generaciones de las "
      f"{MAX_GENERATIONS} permitidas.")
if max(costs) < MIN_IMPROVEMENT:
    print(f"Costó menos que MIN_IMPROVEMENT ({MIN_IMPROVEMENT}) en ambos problemas:")
    print("las mejoras realmente se habían agotado, y el resto del límite era")
    print("desperdicio. En estos dos paisajes la regla es gratis.")
else:
    print(f"Costó hasta {max(costs):+.4f}: en al menos uno de estos problemas la")
    print("ejecución todavía iba a alguna parte cuando la regla la detuvo.")

lost = [(name, best_ever(p).aptitud, max(p[-1], key=lambda i: i.aptitud).aptitud)
        for name, p, _, _, _ in priced]
dropped = [(n, e, f) for n, e, f in lost if e - f > MIN_IMPROVEMENT]
if dropped:
    print(f"\nY el defecto: en {len(dropped)} de {len(lost)} problemas la última")
    for n, e, f in dropped:
        print(f"generación no contiene al campeón - {n} terminó con")
        print(f"{f:+.4f} después de haber visto {e:+.4f}. Nada en este flujo protege")
        print("al mejor individuo; lecciones posteriores añaden elitismo exactamente para esto.")
else:
    print("\nEn ambos problemas la última generación aún contiene al campeón,")
    print("lo cual es suerte más que diseño: nada en este flujo lo protege.")
print("\nUna regla de parada es una apuesta a que una meseta es el final de la búsqueda.")
print("El paso 5 prueba esa apuesta en un paisaje donde el progreso viene a saltos,")
print("y descubre si la suavidad estaba haciendo el trabajo aquí.")
# ------------------------------------------------------------------------------

