"""
Lección 08 - Paso 1: La función que no puedes mirar
===================================================
NUEVO EN ESTE PASO: complicated_one(), slice_along_x(), y el descubrimiento de que
"el paisaje" no es algo que se pueda dibujar.

Cada lección hasta ahora ha optimizado f(x) = sin(x) - 0.2*|x| o un pariente cercano:
un gen real, una curva, una imagen en la diapositiva y - crucialmente - un
óptimo por fuerza bruta en una cuadrícula para comprobar la ejecución. El Paso 5 de la Lección 02 lo dijo
con esas palabras y advirtió que la respuesta honesta a "¿tuvo éxito esta ejecución?"
se vuelve difícil una vez que x deja de ser un solo número.

Aquí es donde deja de ser un solo número. La función a continuación llega de
otro lugar. Nadie la derivó, nadie la diferenciará, y sus cinco
argumentos son de tres tipos diferentes:

    a, b      reales, acotados a [0, 1]
    x         real, acotado a [-100, 100]
    n         un ENTERO, 0 a 20 - no hay f(a, b, x, 3.5, ...)
    fun_name  una ETIQUETA, 'sin' o 'cos' - no hay punto medio entre ellas

Así que el dominio no es una región del plano. Son 21 x 2 = 42 hojas tridimensionales
separadas, y un gráfico es una línea dibujada a través de una de ellas.

Este paso dibuja seis de esas líneas. El plan era mostrar que no están de acuerdo en
nada; no lo hacen, y la verdad resultó ser más útil. Los cinco cortes
vivos ponen su mejor x en el MISMO lugar, a la resolución en que fueron
dibujados; sus mejores valores difieren por casi un factor de diez; y una hoja entera
es idénticamente cero. Una imagen que está de acuerdo consigo misma sobre la ubicación
y no sobre el valor no ha resuelto lo que está sucediendo allí - y nada
dentro de una imagen puede decirte eso. A partir de aquí el gráfico es decoración, y
cada paso posterior tiene que reemplazar lo que el gráfico solía hacer por nosotros.

Ejecútalo: python caja_negra_01_sin_imagen.py

Cinco cortes vivos coinciden en x = -10.950, pero sus máximos difieren por 9.23× y una hoja entera es idénticamente cero. La imagen no ha resuelto el valor.
"""
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# El dominio declarado. Lee esto como un contrato de quien nos entregó la caja:
# nos dicen qué entradas son legales, y no dicen absolutamente nada sobre la salida.
A_MIN, A_MAX = 0.0, 1.0
B_MIN, B_MAX = 0.0, 1.0
X_MIN, X_MAX = -100.0, 100.0
N_SET = range(0, 21)             # gen entero: una malla, no un intervalo
FUN_SET = ("sin", "cos")         # gen categórico: un conjunto, no un rango
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def complicated_one(a: float, b: float, x: float, n: int, fun_name: str) -> float:
    """La caja negra.

    Reproducida del Capítulo 8 de Gridin exactamente como la da el libro. No la leas
    para encontrarle sentido - el punto de un problema de caja negra es que no se
    supone que lo hagas. Está aquí para que la lección funcione; el único hecho sobre ella que
    se nos permite usar es que toma cinco argumentos y devuelve un float.

    (El paso 7 rompe esa regla a propósito, una vez que cada método legítimo ha sido
    probado y ha fallado en responder la pregunta.)

    Args:
        a: real acotado a [0, 1].
        b: real acotado a [0, 1].
        x: real acotado a [-100, 100].
        n: entero legal 0..20; no hay f(..., 3.5, ...).
        fun_name: etiqueta ``sin`` o ``cos``; no hay punto medio.

    Returns:
        Un flotante. Con n = 0 la caja es idénticamente 0.0.

    Example:
        Los cinco cortes vivos coinciden en x = -10.950 y sus máximos
        difieren por 9.23×. Eso es todo lo que un gráfico puede decir aquí.
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


def slice_along_x(a: float, b: float, n: int, fun_name: str,
                  points: int = 4001) -> Tuple[np.ndarray, np.ndarray]:
    """Una línea a través de la caja: variar x, mantener los otros cuatro genes quietos.

    Este es el único tipo de imagen que admite una función de cinco argumentos, y es
    por eso que las imágenes en las Lecciones 01-06 estaban haciendo más trabajo del que
    parecía. Allí, el corte ERA el paisaje. Aquí es una línea de una familia incontable.

    Args:
        a: valor fijo de a.
        b: valor fijo de b.
        n: valor fijo de n.
        fun_name: etiqueta fija ``sin`` o ``cos``.
        points: muestras de x en [-100, 100]; 4001 da un espaciado de 0.05.

    Returns:
        ``(xs, ys)``: el eje x y los valores de la caja.

    Example:
        El corte (a, b, n, fun) = (0.5, 0.5, 3, ``cos``) pone su máximo en
        x = -10.950, el mismo lugar que los otros cortes vivos.
    """
    xs = np.linspace(X_MIN, X_MAX, points)
    ys = np.array([complicated_one(a, b, float(x), n, fun_name) for x in xs])
    return xs, ys


print("=" * 78)
print("Lección 08 - Paso 1: la función que no puedes mirar")
print("=" * 78)

print("\nEl dominio declarado")
print("-" * 78)
print(f"  a         real, [{A_MIN}, {A_MAX}]")
print(f"  b         real, [{B_MIN}, {B_MAX}]")
print(f"  x         real, [{X_MIN}, {X_MAX}]")
print(f"  n         entero, {min(N_SET)}..{max(N_SET)}   ({len(N_SET)} valores legales)")
print(f"  fun_name  etiqueta, uno de {list(FUN_SET)}   ({len(FUN_SET)} valores legales)")
sheets = len(N_SET) * len(FUN_SET)
print(f"\n  Ejes continuos: 3.  Combinaciones discretas: {len(N_SET)} x {len(FUN_SET)} = {sheets}.")
print(f"  Así que el dominio son {sheets} volúmenes 3-D separados, no una sola superficie. No hay")
print("  un solo objeto para graficar, y no hay camino continuo de n=3 a n=4.")

# Cuatro cortes a lo largo de x, todos en los mismos (a, b), difiriendo solo en los dos
# genes discretos. Si los genes discretos fueran un detalle, estos se verían iguales.
CASES: List[Tuple[float, float, int, str]] = [
    (0.5, 0.5, 0, "cos"),     # los genes discretos se mueven, a y b se quedan quietos ...
    (0.5, 0.5, 3, "cos"),
    (0.5, 0.5, 3, "sin"),
    (0.5, 0.5, 11, "cos"),
    (0.0, 0.0, 3, "cos"),     # ... y ahora a y b se mueven, los genes discretos se quedan
    (0.0, 0.0, 3, "sin"),
]

print("\nSeis cortes a lo largo de x a través de la misma caja")
print("-" * 78)
print(f"  {'a':>4} {'b':>4} {'n':>3} {'fun':>4} | {'max en corte':>15} {'en x':>10} "
      f"{'min en corte':>15} {'cambios signo':>13}")
slices = []
for a, b, n, fun_name in CASES:
    xs, ys = slice_along_x(a, b, n, fun_name)
    slices.append((a, b, n, fun_name, xs, ys))
    top = int(np.argmax(ys))
    crossings = int(np.count_nonzero(np.sign(ys[:-1]) != np.sign(ys[1:])))
    print(f"  {a:4.1f} {b:4.1f} {n:3d} {fun_name:>4} | {ys[top]:15.6e} {xs[top]:10.3f} "
          f"{ys.min():15.6e} {crossings:13d}")

# Cada afirmación a continuación se calcula a partir de las cuatro filas recién impresas.
live = [(a, b, n, fn, xs, ys) for a, b, n, fn, xs, ys in slices if ys.max() > 0.0]
argmaxes = [float(xs[int(np.argmax(ys))]) for _, _, _, _, xs, ys in live]
maxima = [float(ys.max()) for _, _, _, _, _, ys in live]
dead = len(slices) - len(live)
argmax_span = max(argmaxes) - min(argmaxes)
value_ratio = max(maxima) / min(maxima)

print(f"\n  {dead} de los {len(slices)} cortes nunca suben por encima de cero en absoluto.")
print(f"  Los otros {len(live)} ponen su mejor x dentro de {argmax_span:.3f} uno del otro")
print(f"  (todos ellos en x = {min(argmaxes):.3f}), así que los cortes están de acuerdo DÓNDE.")
print(f"  Sus mejores VALORES difieren por un factor de {value_ratio:.3g}.")
print("\n  Esa combinación es la trampa. El acuerdo en la ubicación se lee como evidencia")
print(f"  de que la imagen está diciendo la verdad, mientras que el factor de {value_ratio:.3g} en el")
print("  valor dice que no ha resuelto lo que sea que esté sucediendo allí realmente.")
spacing = (X_MAX - X_MIN) / (len(slices[0][4]) - 1)
print(f"  Estas curvas se muestrearon en {len(slices[0][4])} puntos, espaciado {spacing:.4f}. Nada")
print("  dentro de un gráfico te dice si eso fue lo suficientemente fino. El paso 2 le hace a la cuadrícula")
print("  esa pregunta directamente, y obtiene una respuesta que nadie quiere.")

# Uno de los 21 valores enteros es especial, y vale la pena nombrarlo: con n = 0 el
# numerador (x*n + log(n+1)) es 0 + log(1) = 0 para todo x, a, b.
zero_sheet = [n for n in N_SET
              if max(abs(complicated_one(0.3, 0.7, x, n, "cos")) for x in (-7.0, 0.0, 7.0, 40.0)) == 0.0]
print(f"\n  Valores enteros de n para los que la caja devuelve exactamente 0.0 en todas partes: {zero_sheet}")
print(f"  Eso es {len(zero_sheet)} de las {len(N_SET)} hojas, muertas por construcción. Una búsqueda que")
print("  gasta su población allí no está buscando; está inactiva. Nada en la firma de")
print("  la función nos advirtió, y ningún gráfico a lo largo de x lo habría mostrado.")

print("\nQué se ha ido, y qué tiene que reemplazarlo")
print("-" * 78)
print("  Se ha ido: la imagen. No hay una curva que señalar en una diapositiva, y no hay manera")
print("             de ver el óptimo antes de ejecutar nada.")
print("  Se ha ido: la comprobación por fuerza bruta. El paso 2 lo valora y lo encuentra inasequible.")
print("  Queda:     los diagnósticos que este curso ya ha construido - mejor-vs-promedio,")
print("             dispersión de genes, el censo de evaluaciones. A partir de aquí no son")
print("             crédito extra; son los únicos instrumentos que hay.")

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, (a, b, n, fun_name, xs, ys) in zip(axes.ravel(), slices):
    ax.plot(xs, ys, color="tab:blue", linewidth=0.9)
    ax.set_title(f"a={a}, b={b}, n={n}, fun='{fun_name}'", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("f")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
fig.suptitle("Seis cortes a través de la misma caja negra: misma forma, escala salvajemente diferente")
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "caja_negra_01_sin_imagen.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURES}/caja_negra_01_sin_imagen.png")

