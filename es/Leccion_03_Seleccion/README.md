# Lección 03 — Selección

Cinco métodos de selección, construidos uno sobre el otro sobre los mismos diez individuos, y medidos con el mismo instrumento — para que la lección termine no con una lista de cinco técnicas sino con una sola escala en la que los cinco se sitúan, y un compromiso del que nadie escapa.

- **Fuente en el libro:** Gridin, *Learning Genetic Algorithms with Python*, Capítulo 3 (*Selection*) — 5 secciones, ~2,600 palabras, 6 figuras, 225 líneas en 15 archivos.
- **Presupuesto:** 48 min (el Capítulo 3 es el 7.4% del libro).
- **Requisitos:** Lección 01. Esta lección usa la *misma* población — mismo objetivo, misma `SEMILLA = 52` — de manera que los números continúan en lugar de reiniciar, y la selección por torneo en el paso 6 reproduce el paso 3 de la Lección 01 exactamente.

## Modelo matemático

Para pesos no negativos \(w_i\), la selección proporcional usa
\(p_i=w_i/\sum_jw_j\) y el individuo \(i\) recibe \(Np_i\) copias en
esperanza. Un piso libre cambia esas probabilidades. La selección por rango usa
pesos de orden; SUS conserva la esperanza y reduce la dispersión. Con reemplazo
y rango (r=1) para el mejor, un torneo de tamaño (k) selecciona el rango
(r) con probabilidad
( ((N-r+1)/N)^k-((N-r)/N)^k ); la suma sobre los rangos es uno. El elitismo requiere aptitud determinista y
estacionaria y una copia sin cambios para conservar al mejor.

## Qué se lleva el estudiante

1. La selección no inventa nada. Toma N individuos y devuelve N, todos copias de lo que ya estaba allí — por lo que lo único que hay que medir es la contabilidad: quién fue copiado, cuántas veces, quién desapareció.
2. **La selección proporcional oculta un parámetro.** La aptitud de la población es negativa en casi todas partes, un sector de ruleta no puede ser negativo, y la reparación habitual — desplazar todo hacia arriba — introduce silenciosamente un piso que controla la presión más fuertemente que los valores de aptitud.
3. La selección por rango elimina esa dependencia de la escala; el elitismo elimina el riesgo de que una generación sea peor que su padre; SUS elimina el ruido de muestreo; el torneo pone la presión en un número entero que se establece a propósito.
4. **Presión y diversidad son un solo intercambio, no dos perillas.** A través de 13 configuraciones: presión vs aptitud media se correlaciona **+0.95**, presión vs dispersión genética **−0.92**. Ningún método aquí compra aptitud sin pagar en diversidad.

## El ejemplo en curso

Los diez individuos de la Lección 01 en `f(x) = sin(x) - 0.2·|x|`, `x ∈ [-10, 10]`, y una función de medición (`reporte_presion`) aplicada a cada método en **2,000 muestras** a partir de una semilla fija — para que las filas de las tablas de comparación sean comparables por construcción en lugar de por esperanza.

El prefijo es `presion_seleccion_`, porque la presión es de lo que realmente trata la lección.

## Los seis pasos

| # | Script | Qué añade | Qué prueba su salida |
|---|---|---|---|
| 1 | `presion_seleccion_01_la_poblacion.py` | el aparato de medición y la línea base de no hacer nada | Copiar a todos una vez: 0 extintos, diversidad 6.665, 1 copia del mejor — el cero de cada medición posterior. La aptitud es negativa casi en todas partes, por lo que aún no se puede construir ninguna ruleta |
| 2 | `presion_seleccion_02_proporcional.py` | `desplazar_a_positivo()`, `seleccion_proporcional()`, `reporte_presion()` | El piso *es* la presión: el mejor espera **2.35** copias en el piso 0.001 pero **1.30** en el piso 3.0 — misma población, misma ruleta |
| 3 | `presion_seleccion_03_rango.py` | `seleccion_rango()` | El orden por sí solo fija la presión en un valor definido, inmune a la reescala — pero el mejor individuo todavía se pierde en el **13.6%** de las muestras |
| 4 | `presion_seleccion_04_elitismo.py` | `seleccion_rango_con_elite()` | Una línea reduce la pérdida del mejor de 13.6% a **0.0%**, aumenta la presión (1.80 → 2.64 copias) y cuesta diversidad (5.448 → 5.210) |
| 5 | `presion_seleccion_05_sus.py` | `seleccion_muestreo_universal_estocastico()` | Misma ruleta, misma expectativa, la dispersión colapsa. Y **no** arregla el piso: 2.35 vs 1.31 copias, la misma división que encontró el paso 2 |
| 6 | `presion_seleccion_06_torneo.py` | `seleccion_torneo()`, y toda la comparación | Un entero establece la presión: k=3 → 2.74 copias, k=5 → 4.13, k=10 → 6.52. En las 13 configuraciones, presión vs calidad **+0.95**, presión vs diversidad **−0.92** |

**Estos números se verifican contra salidas reales.** Cualquier diapositiva, material o traducción que mencione una cifra debe usar una de estas. Vuelva a ejecutar el script en lugar de confiar en esta tabla si el código ha cambiado.

## Cómo está marcado el código

Dos mecanismos, obligatorios en todo el curso:

- **La receta** — cada script a partir del segundo abre con un bloque `CAMBIOS RESPECTO A …` listando, en orden, los cambios que convierten al script anterior en este. Ese orden es el orden a teclear en clase y el orden que siguen las diapositivas.
- **Las bandas** — cada ítem de la receta tiene su propia banda `# --- NUEVO (n) nombre ---` en el sitio del cambio, numerada y nombrada para coincidir. Todo lo que esté fuera de una banda es código que los estudiantes ya tienen.

## Ejecutándolo

```bash
uv run es/Leccion_03_Seleccion/src/presion_seleccion_01_la_poblacion.py
```

Cada script es independiente, no recibe argumentos, y guarda su figura en `../figures/` bajo su propio nombre. Toda la secuencia se ejecuta en aproximadamente **16 s** (2,000 muestras por medición es lo que compra los promedios estables).

## Rutas y estado, 25 de septiembre de 2026

- Scripts compartidos: [`src/`](src/), 6 pasos; figuras reproducibles en [`figures/`](figures/).
- Estudio autónomo: [`estudiante/leccion_03.pdf`](estudiante/leccion_03.pdf) y [`estudiante/leccion_03_presion_seleccion.ipynb`](estudiante/leccion_03_presion_seleccion.ipynb), con la fuente TeX junto al PDF.
- Uso docente: [`docente/presentacion/leccion_03_compacta.pdf`](docente/presentacion/leccion_03_compacta.pdf) y [`docente/notebooks/leccion_03_compacta.ipynb`](docente/notebooks/leccion_03_compacta.ipynb). La presentación compacta tiene 9 láminas y la libreta contiene comentarios de guía junto al código.

Las piezas extensas conservan la progresión completa; las compactas son autónomas para una ruta de unos 40 minutos. Los ejemplos son sintéticos. La aplicación al reto de Planta Física es una decisión de modelación, no un resultado validado por el socio. Decisión docente del 25/09/2026: se omite la verificación de derechos de Gridin para estos materiales. No se afirma una licencia ni una autorización acreditada; esta verificación no se registra como pendiente ni como bloqueo. No se modificó Canvas ni se probó Colab alojado.
