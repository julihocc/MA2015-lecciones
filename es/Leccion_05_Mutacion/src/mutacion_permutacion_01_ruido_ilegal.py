"""
Lección 05 - Paso 1 del segundo ejemplo: cuando el ruido no es un movimiento legal
=========================================================================
Cada operador hasta ahora ha sumado un número a un gen. Eso funciona porque un gen
era un valor real y cualquier valor real en [-10, +10] era un gen legal.

Cambia la codificación y toda la idea colapsa. Si un cromosoma es un ORDEN -
la secuencia en la que se visitan diez lugares, se ejecutan diez trabajos, se
empacan diez artículos - entonces sus genes no son valores libres. Son una permutación: cada uno
de los diez símbolos aparece exactamente una vez. Suma una gaussiana a uno de ellos y el
resultado no es una solución peor, no es una solución en absoluto.

Este no es un caso especial. Es la mitad de los operadores de mutación en el Capítulo 5 de Gridin,
y es la codificación de cada problema en las Lecciones 09 a 11.

La idea de medición del primer ejemplo sobrevive al cambio de codificación, y
es por eso que esta mitad de la lección puede leerse contra la otra. El alcance allí
era distancia en unidades de genes. El alcance aquí tiene dos componentes, y resultan no
estar de acuerdo: cuántas posiciones perturba la mutación, y qué tan lejos viajan realmente
los símbolos.

Ejecútalo:  python mutacion_permutacion_01_ruido_ilegal.py

La desviación aleatoria produce 1935 mutantes inválidos de 2000. La única
salida legal es la intacta; la inversión de bit cambia 1 posición, el intercambio 2.
"""
import copy
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

SEMILLA = 52
LONGITUD_CROMOSOMA = 10
INTENTOS = 2000
MUTACION_MU, MUTACION_SIGMA = 0.0, 1.0
FIGURAS = Path(__file__).resolve().parent.parent / "figures"


def es_permutacion(lista_genes: list, alfabeto: list) -> bool:
    """Un cromosoma es legal solo si es un reordenamiento del alfabeto.

    Args:
        lista_genes, alfabeto.

    Returns:
        bool.

    Example:
        1935 de 2000 mutantes con ruido gaussiano fallan esta prueba.
    """
    return sorted(lista_genes) == sorted(alfabeto)


def posiciones_cambiadas(original: list, mutante: list) -> int:
    """Cuántas posiciones contienen un símbolo diferente después de la mutación.

    Args:
        original, mutante.

    Returns:
        int.

    Example:
        La desviación aleatoria produce 1935 mutantes inválidos de 2000. La única
    """
    return sum(1 for a, b in zip(original, mutante) if a != b)


def viaje(original: list, mutante: list) -> float:
    """Distancia media, en posiciones, que se movió un símbolo.

    La contraparte de |dx| en el primer ejemplo: `posiciones_cambiadas` dice
    cuánto del cromosoma fue perturbado, `viaje` dice qué tan lejos la perturbación
    llevó algo. Dos operadores pueden puntuar lo mismo en uno y no en el otro,
    que es el hallazgo sobre el que se construye el siguiente paso.

    Args:
        original, mutante.

    Returns:
        float.

    Example:
        La desviación aleatoria produce 1935 mutantes inválidos de 2000. La única
    """
    donde = {simbolo: indice for indice, simbolo in enumerate(mutante)}
    return statistics.fmean(abs(donde[simbolo] - indice)
                            for indice, simbolo in enumerate(original))


def mutar_desviacion_aleatoria(lista_genes: list[float], mu: float, sigma: float,
                            p: float) -> list[float]:
    """El operador del paso 3, sin cambios, a punto de ser apuntado a la codificación incorrecta.

    Args:
        lista_genes, mu, sigma, p.

    Returns:
        list[float].

    Example:
        En una permutación produce 1935 inválidos de 2000.
    """
    mutante = copy.deepcopy(lista_genes)
    for i in range(len(mutante)):
        if random.random() < p:
            mutante[i] = mutante[i] + random.gauss(mu, sigma)
    return mutante


def mutacion_inversion_bit(lista_genes: list[int]) -> list[int]:
    """El operador de Gridin para un cromosoma BINARIO: elige una posición, inviértela.

    Vale la pena detenerse aquí. En un gen binario no existe una perturbación
    pequeña - 0 y 1 son adyacentes y no hay nada entre ellos - por lo
    que el operador no tiene sigma y ningún tamaño de paso en absoluto. El único dial que queda es
    cuántos bits invertir, que es el dial de tasa y nada más.

    Args:
        lista_genes.

    Returns:
        list[int].

    Example:
        La desviación aleatoria produce 1935 mutantes inválidos de 2000. La única
    """
    mutante = copy.deepcopy(lista_genes)
    posicion = random.randint(0, len(mutante) - 1)
    mutante[posicion] = (mutante[posicion] + 1) % 2
    return mutante


def mutacion_intercambio(lista_genes: list, posiciones: tuple[int, int] | None = None
                      ) -> list:
    """El operador de Gridin para una PERMUTACIÓN: intercambia los símbolos en dos posiciones.

    Este es el movimiento legal mínimo. Cualquier mutación de una permutación tiene que ser un
    reordenamiento de lo que ya está ahí, y un intercambio es el reordenamiento más
    pequeño que cambia algo en absoluto.

    `posiciones` es opcional para que el siguiente paso pueda preguntar qué hace un operador
    a un par DADO de posiciones en lugar de a uno aleatorio. Dejado como None
    se comporta exactamente como la versión del libro.

    Args:
        lista_genes, posiciones.

    Returns:
        list.

    Example:
        Cambia exactamente 2 posiciones; viaje idéntico al desplazamiento en 90 de 90 pares.
    """
    mutante = copy.deepcopy(lista_genes)
    if posiciones is None:
        posiciones = random.sample(range(len(mutante)), 2)
    i, j = posiciones
    mutante[i], mutante[j] = mutante[j], mutante[i]
    return mutante


def reporte_reordenamiento(lista_genes: list, operador, intentos: int = INTENTOS) -> dict:
    """Aplica un operador `intentos` veces y describe el daño.

    El instrumento del primer ejemplo, reajustado para ordenamientos: re-semilla, dispara el
    operador muchas veces al mismo individuo, y reporta distribuciones en lugar
    de un solo ejemplo.

    Args:
        lista_genes, operador, intentos.

    Returns:
        dict.

    Example:
        La desviación aleatoria produce 1935 mutantes inválidos de 2000. La única
    """
    random.seed(SEMILLA)
    cambiados, distancias, intactos, ilegales = [], [], 0, 0
    for _ in range(intentos):
        mutante = operador(lista_genes)
        if not es_permutacion(mutante, lista_genes):
            ilegales += 1
            continue
        conteo = posiciones_cambiadas(lista_genes, mutante)
        cambiados.append(conteo)
        distancias.append(viaje(lista_genes, mutante))
        if conteo == 0:
            intactos += 1
    return {"intentos": intentos,
            "ilegales": ilegales,
            "legales": len(cambiados),
            "media_cambiados": statistics.fmean(cambiados) if cambiados else 0.0,
            "max_cambiados": max(cambiados) if cambiados else 0,
            "media_viaje": statistics.fmean(distancias) if distancias else 0.0,
            "intactos": intactos,
            "cambiados": cambiados}


viaje_ruta = list(range(1, LONGITUD_CROMOSOMA + 1))
print(f"Un cromosoma que es un ORDEN, no un vector de valores:\n    {viaje_ruta}")
print("Diez lugares, visitados en esta secuencia. Cada símbolo aparece exactamente una vez,")
print("y esa restricción es parte de lo que ES una solución.")

# ----- el operador ilegal -----------------------------------------------------
print(f"\nAplica la desviación aleatoria del paso 3 a él, {INTENTOS} veces, en p = 0.3:\n")
random.seed(SEMILLA)
legales, legales_y_tocados, ejemplos = 0, 0, []
for _ in range(INTENTOS):
    mutante = mutar_desviacion_aleatoria(viaje_ruta, MUTACION_MU, MUTACION_SIGMA, 0.3)
    if len(ejemplos) < 2 and mutante != viaje_ruta:
        ejemplos.append(mutante)
    if es_permutacion(mutante, viaje_ruta):
        legales += 1
        if mutante != viaje_ruta:
            legales_y_tocados += 1
print(f"    {viaje_ruta}")
for mutante in ejemplos:
    print("    " + str([round(g, 2) for g in mutante]))
print(f"\n    {INTENTOS - legales} de {INTENTOS} mutantes "
      f"({100 * (INTENTOS - legales) / INTENTOS:.1f}%) no eran permutaciones.")
print(f"    De los {legales} que lo eran, {legales_y_tocados} tuvieron algún gen tocado "
      f"en absoluto.")
print("    Así que la única salida legal del operador es aquella en la que la moneda nunca")
print("    cayó: no es una mutación débil en esta codificación, no es ninguna")
print("    mutación en esta codificación.")

# La tasa de supervivencia es exactamente la probabilidad de no tocar nada.
print(f"\nTasa de supervivencia contra p, {INTENTOS} extracciones cada uno:\n")
print("       p | sigue siendo una permutación | predicha (1-p)^n")
print("    -----+------------------------------+-----------------")
supervivencia = []
for porcentaje in range(0, 101, 20):
    p = porcentaje / 100
    random.seed(SEMILLA)
    validos = sum(1 for _ in range(INTENTOS)
                if es_permutacion(mutar_desviacion_aleatoria(viaje_ruta, MUTACION_MU,
                                                          MUTACION_SIGMA, p),
                                  viaje_ruta))
    predicho = (1 - p) ** LONGITUD_CROMOSOMA
    supervivencia.append((p, validos / INTENTOS, predicho))
    print(f"    {p:4.1f} |     {validos:5d} de {INTENTOS}        |"
          f"       {predicho:12.4f}")
peor = max(abs(tasa - predicho) for _, tasa, predicho in supervivencia)
print(f"\n    Medidos y predichos nunca difieren en más de {peor:.4f}.")
print("    Una permutación no puede ser perturbada. Solo puede ser reordenada.")

# ----- los dos operadores legales ---------------------------------------------
print("\nDos codificaciones, dos respuestas.\n")
random.seed(SEMILLA)
bits = [random.randint(0, 1) for _ in range(LONGITUD_CROMOSOMA)]
random.seed(SEMILLA)
invertidos = mutacion_inversion_bit(bits)
print(f"    cromosoma binario: {bits}")
print(f"    después de inversión: {invertidos}")
random.seed(SEMILLA)
cambios_inversion = [posiciones_cambiadas(bits, mutacion_inversion_bit(bits))
                for _ in range(INTENTOS)]
print(f"    sobre {INTENTOS} inversiones: siempre exactamente "
      f"{min(cambios_inversion)} posición cambiada "
      f"(min {min(cambios_inversion)}, max {max(cambios_inversion)}).")
print("    No hay tamaño de paso que elegir. En un gen binario, el cambio más")
print("    pequeño posible es el cambio más grande posible.")

random.seed(SEMILLA)
intercambiados = mutacion_intercambio(viaje_ruta)
print(f"\n    permutación:               {viaje_ruta}")
print(f"    después de intercambio:    {intercambiados}")
reporte = reporte_reordenamiento(viaje_ruta, mutacion_intercambio)
print(f"    sobre {reporte['intentos']} intercambios: {reporte['ilegales']} resultados "
      f"ilegales,")
print(f"    siempre exactamente {reporte['media_cambiados']:.3f} posiciones cambiadas "
      f"(max {reporte['max_cambiados']}),")
print(f"    viaje medio {reporte['media_viaje']:.3f} posiciones por símbolo.")
print("    El intercambio es a una permutación lo que el sigma más pequeño es a un gen")
print("    real: el movimiento local, el que refina en lugar de explorar.")

print("\nEl dial de alcance tiene que ser reconstruido desde cero para esta codificación, y")
print("el siguiente paso lo construye: tres operadores que perturban más del")
print("cromosoma que lo que hace un intercambio, y un instrumento que dice por cuánto.")

# La imagen: qué hace el operador ilegal, y qué hace el legal.
fig, (ax_valido, ax_intercambio) = plt.subplots(1, 2, figsize=(11, 3.8))
ax_valido.plot([p for p, _, _ in supervivencia], [100 * r for _, r, _ in supervivencia],
              "o-", color="tab:red", label=f"medido sobre {INTENTOS} extracciones")
ax_valido.plot([p for p, _, _ in supervivencia], [100 * q for _, _, q in supervivencia],
              "--", color="tab:blue",
              label=f"(1 - p) ^ {LONGITUD_CROMOSOMA}")
ax_valido.set_xlabel("probabilidad de mutación por gen p")
ax_valido.set_ylabel("% de mutantes que siguen siendo una permutación")
ax_valido.set_title("Desviación aleatoria en una permutación:\nla única salida legal "
                   "es la intacta", fontsize=10)
ax_valido.grid(True, linestyle=":", alpha=0.5)
ax_valido.legend(fontsize=8)

ax_intercambio.hist(reporte["cambiados"], bins=range(0, LONGITUD_CROMOSOMA + 2),
             align="left", rwidth=0.8, color="tab:green")
ax_intercambio.set_xticks(range(0, LONGITUD_CROMOSOMA + 1))
ax_intercambio.set_xlabel("posiciones cambiadas por un intercambio")
ax_intercambio.set_ylabel(f"conteo de {reporte['intentos']} mutaciones")
ax_intercambio.set_title("Intercambio: el movimiento legal mínimo,\ny es mínimo cada "
                  "vez", fontsize=10)
ax_intercambio.grid(True, axis="y", linestyle=":", alpha=0.5)

fig.suptitle("Cambiar la codificación cambia qué mutaciones existen")
FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "mutacion_permutacion_01_ruido_ilegal.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/mutacion_permutacion_01_ruido_ilegal.png")

