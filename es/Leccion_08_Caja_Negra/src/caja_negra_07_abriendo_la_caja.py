"""
Lección 08 - Paso 7: Abriendo la caja
=====================================
NUEVO EN ESTE PASO: denominator(), el polo, y la escalera.

Seis pasos han tratado complicated_one() como una caja negra, la manera en que el libro
la presenta. Las búsquedas están hechas; ahora leemos la única línea que estaba haciendo
todo el daño. Cada término de la suma es un numerador sobre un denominador, y
el denominador es

    div = ((n + 1) ** 2) * (1 + a + b) * (120 - x ** 2) * resid + 1 / 2

Todo en él es dócil excepto el producto (120 - x ** 2) * resid, que
se puede hacer tan cercano a -1/(2 * (n+1)^2 * (1+a+b)) como lo permita x - y allí
el denominador cruza por cero. La caja no contiene un máximo.
Contiene un polo.

Para los genes del campeón del paso 5 el cruce se encuentra por bisección en
x* = -10.9535836, a un pelo de distancia de -sqrt(120) = -10.9544512 (el +1/2
lo desplaza). Luego la escalera: evaluar f en x* + 10^-k para k creciente. La
"aptitud" reportada se multiplica por 100 cada vez que la distancia se reduce por 100
- 2.2e-14, 2.2e-12, 2.2e-10, 2.2e-08 - sin techo a la vista. ¿Y el
campeón del que el GA estaba tan orgulloso? Está estacionado a 9.6e-08 del polo, y su
2.278e-09 es exactamente el escalón de la escalera a esa distancia. El algoritmo
no encontró un máximo. Midió su propia precisión de estacionamiento frente a una
singularidad.

La moraleja es la del capítulo: con una caja negra, el reporte del algoritmo y la
verdad son documentos diferentes. Juzgar la respuesta requiere un conocimiento que la ejecución
no puede proveer - así que antes de confiar en una búsqueda, abre la caja.

DESAPARECE EN ESTE PASO: todo el algoritmo genético. La búsqueda terminó en
el paso 6; el campeón se cita a continuación como un literal, con su proveniencia.

CAMBIOS RESPECTO A caja_negra_06_el_veredicto.py
Introdúcelos en este orden:
    1. denominator()   la subexpresión peligrosa, aislada y legible
    2. el polo         el denominador cambia de signo: la bisección encuentra dónde
    3. la escalera     f en x* + 10^-k: lo que el GA realmente estaba escalando

Ejecútalo:  python caja_negra_07_abriendo_la_caja.py

El óptimo aparente es un polo cerca de x = -10.9535836. La aptitud 2.278e-09 del campeón mide su distancia a la singularidad, no un máximo.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# El campeón del paso 5, literal: SEMILLA = 19, 100 generaciones, 20,204
# evaluaciones. Doce ejecuciones independientes en el paso 6 se estacionaron dentro de 0.008 de
# este mismo x.
CHAMPION = [0.3259749523591883, 0.47589136714114383, -10.953583467984679,
            3, "cos"]
CHAMPION_FITNESS = 2.278347670913726e-09


def complicated_one(a: float, b: float, x: float, n: int, fun_name: str) -> float:
    """La caja, abierta por fin: mismo código de siempre, ahora lo estamos leyendo.

    Args:
        a: real acotado a [0, 1]; se fija en el campeón del paso 5.
        b: real acotado a [0, 1]; se fija en el campeón del paso 5.
        x: el único gen que se mueve en este paso.
        n: entero; el campeón guarda 3.
        fun_name: etiqueta; el campeón guarda ``cos``.

    Returns:
        Un flotante. Cerca del polo crece como 1/|x - x*|.

    Example:
        En x* + 10^{-k} la lectura sube 2.2e-14, 2.2e-12, 2.2e-10, 2.2e-08.
        El campeón, a 9.6e-08 del polo, vale 2.278e-09: un escalón de esa
        escalera.
    """
    total = 0.0
    for _ in range(10, 10 + n + 1):
        if fun_name == "cos":
            trig = math.cos(math.log2(b ** n + 1) * math.pi * x) / (n + 1) + math.sin(x) ** n
        elif fun_name == "sin":
            trig = math.sin(math.log2(b ** n + 1) * math.pi * x) / (n + 1) + math.cos(x) ** n
        else:
            raise ValueError(f"Función desconocida: {fun_name}")
        resid = trig - math.log2(n + 1)
        div = ((n + 1) ** 2) * (1 + a + b) * (120 - x ** 2) * resid + 1 / 2
        total += ((x * n + math.log(n + 1)) / div) / (10 ** 15)
    return total


# --- NUEVO (1) denominator() ----------------------------------------------------
# La única subexpresión que importaba, aislada. El numerador es aburrido -
# en los genes del campeón es alrededor de -126 y varía lentamente. Todo lo que las
# búsquedas "descubrieron" vive en este denominador.
def denominator(x: float, a: float, b: float, n: int, fun_name: str) -> float:
    """El divisor de cada término, para un fijo (a, b, n, fun_name).

    El numerador es aburrido; todo lo que las búsquedas "descubrieron" vive
    aquí. Un cero de este divisor es un polo de la caja, no un máximo.

    Args:
        x: el gen que se barre.
        a: valor fijo del campeón.
        b: valor fijo del campeón.
        n: valor fijo del campeón.
        fun_name: etiqueta fija del campeón.

    Returns:
        El divisor. Cambia de signo entre -11.0 y -10.9.

    Example:
        La bisección encuentra el cero en x* = -10.9535836, a un pelo de
        -sqrt(120). El campeón se estacionó a 9.6e-08 de ese polo.
    """
    if fun_name == "cos":
        trig = math.cos(math.log2(b ** n + 1) * math.pi * x) / (n + 1) + math.sin(x) ** n
    else:
        trig = math.sin(math.log2(b ** n + 1) * math.pi * x) / (n + 1) + math.cos(x) ** n
    resid = trig - math.log2(n + 1)
    return ((n + 1) ** 2) * (1 + a + b) * (120 - x ** 2) * resid + 1 / 2
# ------------------------------------------------------------------------------

A, B, X_CHAMP, N, FUN = CHAMPION

print("=" * 78)
print("Lección 08 - Paso 7: abriendo la caja")
print("=" * 78)
print(f"\nLos genes permanecen fijos en el campeón del paso 5: a={A:.4f}, b={B:.4f}, "
      f"n={N}, fun={FUN}")
print("Solo x se mueve ahora.")

# --- NUEVO (2) el polo ---------------------------------------------------------
# div(-11.0) = +26.59 y div(-10.9) = -31.38: el denominador cambia de signo
# entre ellos. Una función continua que cambia de signo tiene un cero en
# medio, y la bisección lo encuentra a precisión de máquina en 60 divisiones a la mitad.
low, high = -11.0, -10.9
print("\nEl polo")
print("-" * 78)
print(f"  denominator(-11.0) = {denominator(low, A, B, N, FUN):+.4f}")
print(f"  denominator(-10.9) = {denominator(high, A, B, N, FUN):+.4f}")
for _ in range(60):
    mid = (low + high) / 2
    if denominator(low, A, B, N, FUN) * denominator(mid, A, B, N, FUN) <= 0:
        high = mid
    else:
        low = mid
X_STAR = (low + high) / 2
print(f"  cambio de signo acotado: polo en x* = {X_STAR:.10f}")
print(f"  compara -sqrt(120) = {-math.sqrt(120):.10f}  "
      f"(el +1/2 desplaza el cero del lugar obvio)")
print(f"  el campeón se asienta en x* {X_CHAMP - X_STAR:+.2e}  "
      f"- estacionado a {abs(X_CHAMP - X_STAR):.1e} del polo")
# ------------------------------------------------------------------------------

# --- NUEVO (3) la escalera -------------------------------------------------------
# Caminar hacia el polo en potencias de diez y leer f en cada escalón. Si f estuviera
# acotada cerca de x*, los escalones se asentarían. No lo hacen: cada acercamiento de 100x
# multiplica la lectura por 100. Esa es la firma de 1/(x - x*), y
# es exactamente la ley con la que la cuadrícula del paso 2 se topó (2x, 21x, 1387x) y el GA
# de los pasos 5-6 escaló (12 campeones, 21,504x separados, todos estacionados aquí).
print("\nLa escalera: f en x* + 10^-k")
print("-" * 78)
print(f"  {'k':>3}  {'distancia':>10}  {'f(x* + 10^-k)':>14}")
for k in (1, 2, 4, 6, 8):
    distance = 10.0 ** (-k)
    value = complicated_one(A, B, X_STAR + distance, N, FUN)
    print(f"  {k:>3}  {distance:>10.0e}  {value:>14.3e}")
print(f"\n  el escalón del campeón mismo: distancia {abs(X_CHAMP - X_STAR):.1e}, "
      f"f = {CHAMPION_FITNESS:.3e}  (se asienta exactamente en esta escalera)")
print("\n  No hay máximo. La escalera no tiene escalón superior; la cuadrícula del paso 2")
print("  no pudo ver uno porque no hay ninguno que ver, y el GA de los pasos")
print("  5-6 'mejoró' estacionándose más cerca de una singularidad. El reporte del")
print("  algoritmo y la verdad eran documentos diferentes - y solo al abrir")
print("  la caja se reconcilian.")
# ------------------------------------------------------------------------------

fig, (ax_div, ax_ladder) = plt.subplots(1, 2, figsize=(11.5, 4.0))
xs = np.linspace(-11.05, -10.85, 500)
ax_div.plot(xs, [denominator(v, A, B, N, FUN) for v in xs],
            color="tab:blue", linewidth=1.4)
ax_div.axhline(0.0, color="tab:gray", linewidth=0.8)
ax_div.axvline(X_STAR, color="tab:red", linewidth=1.0, linestyle="--",
               label=f"polo en x* = {X_STAR:.6f}")
ax_div.set_title("El denominador cruza el cero", fontsize=10)
ax_div.set_xlabel("x")
ax_div.set_ylabel("denominator(x)")
ax_div.legend(fontsize=8)
ax_div.grid(True, linestyle=":", alpha=0.5)

distances = np.logspace(-9, -1, 200)
ax_ladder.loglog(distances,
                 [abs(complicated_one(A, B, X_STAR + d, N, FUN))
                  for d in distances],
                 color="tab:blue", linewidth=1.4,
                 label="|f(x* + d)|: la escalera")
ax_ladder.scatter([abs(X_CHAMP - X_STAR)], [CHAMPION_FITNESS], s=70,
                  color="tab:red", zorder=3,
                  label="campeón del paso 5, estacionado en la escalera")
ax_ladder.set_title("Lo que la búsqueda escalaba: sin escalón superior", fontsize=10)
ax_ladder.set_xlabel("distancia desde el polo")
ax_ladder.set_ylabel("|f|  (escala log)")
ax_ladder.legend(fontsize=8)
ax_ladder.grid(True, which="both", linestyle=":", alpha=0.5)

fig.suptitle("Abriendo la caja: un polo, no un máximo")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "caja_negra_07_abriendo_la_caja.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURES}/caja_negra_07_abriendo_la_caja.png")

