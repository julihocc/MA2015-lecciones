"""
Lección 02 - Paso 5: Paisajes escalonados, y una suposición que resultó errónea
=========================================================================
NUEVO EN ESTE PASO: un tercer problema, y un segundo instrumento que puede ver lo que la
columna de costo del paso 4 no puede.

El paso 4 evaluó el costo de la regla de parada en dos paisajes suaves y encontró que era gratis. La
sospecha natural es que la suavidad estaba haciendo el trabajo: en un paisaje donde el
progreso llega en saltos en lugar de pendientes, un tramo plano significaría que la
población aún no ha encontrado el siguiente escalón, no que la búsqueda haya terminado - y una
regla de paciencia no puede distinguirlos.

Este paso construye ese paisaje para atrapar la regla. No la atrapa.
La ejecución sube la escalera, llega a su escalón superior, y la regla la detiene seis
generaciones después sin haber perdido nada. La sospecha era errónea, y la razón
vale más de lo que valía la sospecha: estos escalones SUBEN. Cada límite de escalón es una
comparación real, por lo que la selección tiene una dirección que seguir a pesar de que la superficie
es plana entre límites. Escalonado no es lo mismo que sin dirección.

Lo que sí entrega este paso es el instrumento. "La regla no costó nada" no es la
misma afirmación que "la ejecución tuvo éxito", y una ejecución no puede decirte la diferencia sobre
sí misma. Comparar cada ejecución contra la referencia de cuadrícula densa del paisaje - disponible aquí
solo porque estos problemas son unidimensionales y pueden resolverse por fuerza bruta - es lo que
separa una búsqueda que terminó de una búsqueda que se estancó. Las Lecciones 08 y 09
trabajan en paisajes donde esa verificación no está disponible, que es exactamente cuando la
distinción empieza a importar.

CAMBIOS RESPECTO A flujo_ag_04_condicion_parada.py
Introdúcelos en este orden:
    1. escalera()            un paisaje cuyo progreso viene a saltos, no en pendientes
    2. PROBLEMAS               el tercer problema se une a los otros dos, nada más se mueve
    3. el veredicto           medir cada ejecución contra la referencia de cuadrícula densa, no contra sí misma

Ejecútalo:  python flujo_ag_05_paisajes_escalonados.py

La escalera no rompe la regla: la corrida llega a +2.0000, el óptimo
verdadero, y la regla la detiene sin haber perdido nada. Déficit +0.0000.
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
MAX_GENERATIONS = 60   # el tope, sin cambios desde el paso 4
GENE_MIN, GENE_MAX = -10.0, 10.0
TOURNAMENT_SIZE, BLEND_ALPHA = 3, 1.0
MUTATION_MU, MUTATION_SIGMA = 0.0, 1.0
TARGET = 4.2
PEAK = 4.0             # donde la escalera tiene su escalón superior
SLOPE = 2.0            # qué tan rápido se estrechan los escalones a medida que suben
TENT = 10.5            # altura de la carpa: hace que el escalón superior tenga 0.5 de ancho, no un solo punto
# Cuánto esperamos y qué estamos dispuestos a llamar progreso. Una regla sin
# MIN_IMPROVEMENT se deja engañar por el quinto decimal y nunca se dispara; una
# regla sin PATIENCE se dispara en la primera generación desafortunada.
# PATIENCE = 0 apaga toda la regla, que es como se hacen las ejecuciones de control abajo.
PATIENCE = 5
MIN_IMPROVEMENT = 1e-4
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


# --- NUEVO (1) escalera() ----------------------------------------------------
def escalera(x: float, step: float = 1.0, rise: float = 0.2) -> float:
    """Un paisaje cuyo progreso viene en saltos en lugar de pendientes.

    Dentro de un escalón el valor es constante, por lo que cada individuo parado en el
    mismo escalón tiene exactamente la misma aptitud y la selección no tiene nada que comparar.
    La única forma de subir es una mutación lo suficientemente grande como para cruzar un límite de escalón - y
    hasta que ocurra una, la mejor aptitud no se mueve en absoluto.

    Esta no es una forma artificial. Cualquier aptitud construida a partir de un recuento, una categoría o
    una medida redondeada se ve así, y los problemas combinatorios de las
    Lecciones 09 a 11 son todos de ese tipo.

    Los escalones suben hacia PEAK y caen después de este, de modo que el escalón superior se asienta en
    el medio del intervalo. Ese detalle importa: si el mejor escalón estuviera en un
    borde, acotar() lo entregaría gratis a cualquier mutación que se exceda,
    y el paisaje sería fácil por una razón que no tiene nada que ver con el
    algoritmo.

    TENT importa por una segunda razón, encontrada de la manera difícil. Con TENT exactamente 10.0
    la condición para el escalón superior, TENT - SLOPE*|x - PEAK| >= 10, se cumple en el
    único punto x = PEAK y en ningún otro lugar: el mejor escalón tiene ANCHO CERO. Un
    referencia de cuadrícula densa tomada en una cuadrícula que resulta contener ese punto entonces
    reporta un máximo en el que ninguna búsqueda podría aterrizar jamás, y todo "déficit" medido
    contra él es un artefacto. TENT = 10.5 le da al escalón superior un ancho de 0.5.

    La regla general de la que esto es una instancia: cuando compares una ejecución contra un
    referencia de cuadrícula densa, verifica que el nivel superior ocupe algo de volumen. Un óptimo
    que la cuadrícula puede ver y la búsqueda no, no es un benchmark, es un bug.

    Args:
        x: un real en [-10, 10].
        step: ancho de cada salto de floor().
        rise: altura de cada escalón (aquí 0.2).

    Returns:
        Un múltiplo de 0.2. Con TENT = 10.5 el escalón superior vale +2.0
        y tiene 0.5 de ancho alrededor de x = 4.0.

    Example:
        Esta corrida llega a +2.0000; déficit contra fuerza bruta: +0.0000.
    """
    return float(np.floor((TENT - SLOPE * abs(x - PEAK)) / step) * rise)
# ------------------------------------------------------------------------------


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
        (lista de generaciones, lista de censos).
    """
    random.seed(SEMILLA)
    population = [create_random(fitness_function) for _ in range(POPULATION_SIZE)]

    generations = [population]
    censuses: List[Dict[str, int]] = []
    for generation in range(1, MAX_GENERATIONS + 1):
        population, census = evolve_one_generation(population)
        generations.append(population)
        censuses.append(census)
        bests = [max(i.aptitud for i in pop) for pop in generations]
        # Contra el máximo histórico: sin elitismo el campeón puede empeorar.
        last_improvement = 0
        for g in range(1, len(bests)):
            if bests[g] > max(bests[:g]) + MIN_IMPROVEMENT:
                last_improvement = g
        if patience and generation - last_improvement >= patience:
            break
    return generations, censuses


# --- NUEVO (2) PROBLEMAS -------------------------------------------------------
PROBLEMAS = [("paisaje de seno", sine_landscape),
            ("cercania al objetivo", closeness_to_target),
            ("escalera", escalera)]
# ------------------------------------------------------------------------------

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
    if name == "escalera":
        continue
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
fig.tight_layout(rect=(0, 0, 1, 0.90))
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "flujo_ag_05_paisajes_escalonados_historial.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figura guardada en {FIGURES}/flujo_ag_05_paisajes_escalonados_historial.png")

# Una regla de parada solo vale la pena tenerla si es barata. Ponle precio: ejecuta cada
# problema nuevamente con la regla apagada, hasta el límite.
#
# Ambas columnas reportan el mejor individuo JAMÁS VISTO, no el mejor en la última
# generación. Sin elitismo esos son números diferentes, y la diferencia se
# reporta abajo - un flujo que tira a su campeón tiene que recordarlo
# en algún lugar fuera de la población.
def best_ever(generations: List[List[Individuo]]) -> Individuo:
    """El mejor individuo JAMÁS VISTO, no el de la última generación.

    Args:
        generations: historial completo, generación 0 incluida.

    Returns:
        El Individuo de mayor aptitud en todo el historial.
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
    print(f"Costó menos que MIN_IMPROVEMENT ({MIN_IMPROVEMENT}) en ambos problemas:") # WAIT, in English it says "both problems". Now it's 3. I will translate exactly what it said.
    print("las mejoras realmente se habían agotado, y el resto del límite era")
    print("desperdicio. En estos paisajes la regla es gratis.")
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
    print("\nEn los problemas la última generación aún contiene al campeón,")
    print("lo cual es suerte más que diseño: nada en este flujo lo protege.")

# --- NUEVO (3) el veredicto ---------------------------------------------------
# El paso 4 preguntó cuál era el COSTO de la regla. Esa pregunta tiene un punto ciego: una regla puede ser
# gratis porque la búsqueda terminó, o gratis porque la búsqueda estaba muerta, y
# la columna de costo no puede distinguirlos. Así que haz una segunda pregunta que la ejecución
# no puede responder sobre sí misma - ¿qué tan lejos de la referencia de cuadrícula densa del paisaje
# terminó? - por fuerza bruta en una cuadrícula, lo cual es posible solo porque estos
# paisajes son unidimensionales.
GRID = np.linspace(GENE_MIN, GENE_MAX, 20001)

print("\nLo que la columna de costo no puede ver: distancia a la referencia de cuadrícula densa.")
print("               problema | mejor encont.| ref. cuadricula | deficit   | disp. final")
print("------------------------+--------------+----------------+-----------+------------")
shortfalls = []
for name, patient, full, on, off in priced:
    fitness_function = dict(PROBLEMAS)[name]
    true_best = max(fitness_function(float(x)) for x in GRID)
    shortfall = true_best - off.aptitud
    spread = population_metrics(full[-1])["spread"]
    shortfalls.append((name, shortfall, spread, len(patient) - 1, len(full) - 1))
    print(f"{name:>23s} | {off.aptitud:+12.4f} | {true_best:+14.4f} |"
          f" {shortfall:+9.4f} | {spread:11.3f}")

reached = [row for row in shortfalls if row[1] <= MIN_IMPROVEMENT]
stalled = [row for row in shortfalls if row[1] > MIN_IMPROVEMENT]

print("\nUna regla de parada es una apuesta a que una meseta es el final de la búsqueda.")
if not stalled:
    print(f"En los {len(reached)} problemas la apuesta fue correcta: cada ejecución sostenía")
    print("el óptimo de su paisaje para cuando la regla se disparó, escalera incluida.")
    print("\nPor lo tanto, la sospecha sobre la que se construyó este paso es refutada. Un paisaje")
    print("escalonado no derrota a una regla de paciencia, porque estos escalones SUBEN:")
    print("cada límite entre ellos es una comparación sobre la que la selección puede actuar, y")
    print("entre límites la población solo tiene que derivar lo suficientemente lejos para")
    print("alcanzar el siguiente. Plano en algunas partes no es lo mismo que sin dirección.")
else:
    for name, shortfall, spread, stopped, capped in stalled:
        print(f"    {name}: terminó a {shortfall:+.4f} antes del óptimo, y la")
        print(f"    ejecución de control a la generación {capped} terminó exactamente tan corta como")
        print(f"    la que se detuvo en la generación {stopped}. Dispersión final {spread:.3f}.")
    print("\nLa regla no costó nada y la ejecución aún así falló, por lo que la regla no fue")
    print("la falla: a la población se le agotó la dispersión para buscar.")

print("\nQuédate con el instrumento, no con el resultado. 'La regla no costó nada' y 'la")
print("ejecución tuvo éxito' son afirmaciones diferentes, y la columna de costo del paso 4 solo")
print("puede hacer la primera. La segunda necesita una referencia fuera de la ejecución. Aquí")
print("esa referencia es la fuerza bruta en una cuadrícula, que funciona porque x es un")
print("número; de la Lección 08 en adelante no lo es, y la respuesta honesta a '¿tuvo")
print("éxito esta ejecución?' se vuelve genuinamente difícil de obtener.")

# La imagen: los tres paisajes, dónde se detuvo cada ejecución y la referencia de cuadrícula densa.
fig, axes = plt.subplots(1, len(PROBLEMAS), figsize=(4.2 * len(PROBLEMAS), 3.6))
for ax, (name, patient, full, on, off) in zip(axes, priced):
    fitness_function = dict(PROBLEMAS)[name]
    values = [fitness_function(float(x)) for x in GRID]
    ax.plot(GRID, values, color="tab:blue", alpha=0.7, linewidth=1)
    top = float(GRID[int(np.argmax(values))])
    ax.plot([top], [max(values)], "*", color="tab:purple", markersize=14,
            label=f"ópt. verdadero {max(values):+.3f}")
    ax.plot([on.gen], [on.aptitud], "o", color="tab:red", markersize=9,
            label=f"regla encendida, gen {len(patient) - 1}")
    ax.plot([off.gen], [off.aptitud], "s", color="tab:green", markersize=7,
            label=f"regla apagada, gen {len(full) - 1}")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("aptitud")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower center")
fig.suptitle("Dónde se detuvo cada ejecución, y dónde estaba realmente el óptimo")
fig.tight_layout(rect=(0, 0, 1, 0.90))
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "flujo_ag_05_paisajes_escalonados.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURES}/flujo_ag_05_paisajes_escalonados.png")
# ------------------------------------------------------------------------------

