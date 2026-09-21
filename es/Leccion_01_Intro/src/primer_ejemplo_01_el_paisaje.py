"""
Lección 01 - Paso 1: El problema
=================================
Antes de escribir cualquier algoritmo, veamos qué estamos buscando.

    f(x) = sen(x) - 0.2 * |x|,   x en [-10, 10]

Ejecútalo:  python primer_ejemplo_01_el_paisaje.py

Cuatro cimas en un eje. El máximo global no es el más cercano al origen:
cualquier método que solo suba por la pendiente se detiene donde empezó.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

GEN_MIN, GEN_MAX = -10.0, 10.0
FIGURAS = Path(__file__).resolve().parent.parent / "figuras"


def objetivo(x):
    """La función que queremos MAXIMIZAR.

    Cuatro cimas en [-10, 10]. El AG no necesita derivadas: solo este
    número. El máximo global no es la cima más cercana al origen.

    Argumentos:
        x: un real o un arreglo de numpy.

    Devuelve:
        sen(x) - 0.2 * |x|. En la malla de 400 puntos de este archivo,
        el máximo cae cerca de x = +1.372, f = +0.706.

    Ejemplo:
        objetivo(+1.372) ≈ +0.706 (colina global).
        objetivo(-4.417) ≈ +0.073 (colina local del paso 7).
    """
    return np.sin(x) - 0.2 * abs(x)


x = np.linspace(GEN_MIN, GEN_MAX, 400)
y = objetivo(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, color="tab:blue")
plt.title("El espacio de búsqueda")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, linestyle=":", alpha=0.5)
FIGURAS.mkdir(exist_ok=True)
plt.savefig(FIGURAS / "primer_ejemplo_01_el_paisaje.png", dpi=150, bbox_inches="tight")
plt.close()

# argmax sobre la malla, no un óptimo analítico: el AG tampoco tendrá más.
mejor_x = x[np.argmax(y)]
print(f"El máximo global está cerca de x = {mejor_x:+.3f}, f(x) = {objetivo(mejor_x):+.3f}")
print("Observa las otras tres cimas. Cualquier método que solo suba por la pendiente")
print("se detendrá en aquella donde haya empezado por casualidad.")
print(f"\nFigura guardada en {FIGURAS}/primer_ejemplo_01_el_paisaje.png")

