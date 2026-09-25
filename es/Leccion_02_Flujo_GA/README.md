# Lección 02 — El flujo del algoritmo genético

La Lección 01 terminó con un algoritmo genético que funcionaba pero cuyo cuerpo del ciclo era una serie de bloques de comentarios sin nombre. Esta lección convierte ese ciclo en un flujo del que se puede hablar: fases con nombre, un problema que llega como argumento en lugar de estar soldado, cuatro números por generación en lugar de uno, y una regla para saber cuándo detenerse — la cual el último paso luego sorprende diciendo una mentira cómoda.

- **Fuente en el libro:** Gridin, *Learning Genetic Algorithms with Python*, Capítulo 2 (*Genetic Algorithm Flow*) — 7 secciones, ~2,350 palabras, 8 figuras, 188 líneas en 8 archivos (`individual.py`, `fitness.py`, `selection.py`, `crossover.py`, `mutate.py`, `population.py`, `settings.py`, `genetic_algorithm_flow.py`). La división de archivos del capítulo es la que siguen los pasos.
- **Presupuesto:** 44 min (El Capítulo 2 es el 6.7% del libro).
- **Prerrequisitos:** Lección 01. El Paso 1 reproduce el paso 6 de la Lección 01 al dígito con la misma semilla, y lo dice en su propia salida — esa identidad es la prueba de que nombrar el flujo no cambió nada.

## Modelo matemático

Sea \(P_t\) la población en la generación \(t\). Una generación es
\(P_{t+1}=R(M(C(S(P_t))))\), una transición aleatoria. El máximo, promedio y
dispersión describen la población finita; el mejor histórico es un estado
separado. Una regla de paciencia detecta \(b_t-b_{t-k}<\varepsilon\); no
establece convergencia ni optimalidad. La escalera es
(r\lfloor(T-s|x-x_0|)/h\rfloor): cada banda ordinaria tiene ancho unilateral
(h/s=0.5); con (T=10.5,h=1,s=2), la cima es (|x-4|\le0.25), de ancho 0.5.

## Lo que el estudiante se lleva

1. Un algoritmo genético tiene cinco fases con nombre, y una generación de evolución es una función. No se puede discutir un flujo que no tiene límite.
2. **El problema es un argumento, no parte del algoritmo.** Una vez que se inyecta la función de aptitud, el mismo controlador resuelve un problema diferente sin que nada más cambie — y una minimización es solo una maximización con el signo invertido.
3. El mejor individuo es el número sobre el cual un AG es menos honesto. La aptitud promedio y la **dispersión genética (gene spread)** son las que indican si una población todavía está buscando.
4. Una ejecución puede terminar de tres maneras; "las mejoras se agotaron" es la única que necesita ser medida, y la medición es barata.
5. **Una regla de parada barata no es una buena noticia por sí sola.** Informa que una ejecución dejó de mejorar — no si la ejecución terminó o simplemente se atascó.

## Los ejemplos en ejecución

El prefijo es `flujo_ag_`. Tres funciones de aptitud se entregan a un controlador sin cambios:

- `sine_landscape` — El `sin(x) − 0.2·|x|` de la Lección 01, ahora solo un problema entre otros.
- `closeness_to_target` — `−(x − 4.2)²`, una minimización escrita como una maximización.
- `escalera` — escalones planos que suben hacia x = 4, agregados en el paso 5. Dos detalles son deliberados. Su escalón superior es **interior**: en un borde, `acotar()` se lo entregaría a cualquier mutación que se exceda. Y la carpa de la que está cortada es 10.5 en lugar de 10.0, por lo que el escalón superior es **0.5 de ancho en lugar de un solo punto** — ver la corrección abajo.

## Los cinco pasos

| # | Script | Qué agrega | Qué prueba su salida |
|---|---|---|---|
| 1 | `flujo_ag_01_las_cinco_fases.py` | `evolve_one_generation()`, `run()`, un censo por fase | Misma semilla, misma respuesta que la Lección 01: `x=+1.372 f=+0.706`. La ejecución costó **107** evaluaciones de aptitud, no las 110 obvias — un individuo seleccionado pero no cruzado ni mutado es el mismo objeto y nunca se vuelve a evaluar |
| 2 | `flujo_ag_02_aptitud_inyectada.py` | la función de aptitud se convierte en un argumento | Un controlador, dos problemas: la ejecución del seno reproduce exactamente el paso 1, la ejecución del objetivo aterriza a **0.0038** de +4.200. La única diferencia entre las ejecuciones es qué función se pasó |
| 3 | `flujo_ag_03_metricas_poblacion.py` | `population_metrics()`, la tabla de historial, su figura | En el problema del seno, la aptitud promedio sube de −1.5588 a +0.6228 mientras que la dispersión genética cae de **6.665 a 0.064** para la generación 8. El campeón se queda quieto mientras la población colapsa debajo de él |
| 4 | `flujo_ag_04_condicion_parada.py` | `PATIENCE`, `MIN_IMPROVEMENT`, detenerse por estancamiento, ejecuciones de control | La regla ahorra de **49 a 51** de las 60 generaciones permitidas y cuesta menos que `MIN_IMPROVEMENT` en ambos problemas. También expone un defecto: en 1 de 2 problemas, la última generación ya no contiene al campeón — nada aquí lo protege |
| 5 | `flujo_ag_05_paisajes_escalonados.py` | `escalera()`, el tercer problema, el veredicto frente a la referencia de cuadrícula densa | La escalera **no** rompe la regla: la ejecución llega a `+2.0000`, la referencia de cuadrícula densa del paisaje, y la regla la detiene sin haber perdido nada. Cada uno de los tres problemas coincide con su referencia de cuadrícula, déficit `+0.0000` |

**Estos números se verifican contra la salida real.** Cualquier diapositiva, folleto o traducción que indique una cifra debe indicar una de estas. Vuelve a ejecutar el script en lugar de confiar en esta tabla si el código ha cambiado.

### Una hipótesis refutada y una corrección

El paso 5 fue diseñado para atrapar la regla de parada. El plan: un paisaje escalonado en el cual una regla de paciencia confundiría "aún no ha encontrado el siguiente escalón" con "ha terminado", saldría temprano, y perdería una aptitud que una ejecución más larga habría encontrado.

**No ocurre.** La ejecución sube la escalera, llega a su escalón superior, y la regla la detiene seis generaciones después sin haber perdido nada — en cada altura de carpa y ancho de escalón probado. La hipótesis es refutada, y la razón vale más de lo que valía la hipótesis: estos escalones *suben*, así que cada límite de escalón es una comparación sobre la que la selección puede actuar. Plano en algunas partes no es lo mismo que sin dirección. El paso ahora reporta eso, y mantiene el instrumento que produjo la investigación — la comparación de cada ejecución contra una referencia de cuadrícula densa, que es lo que distingue "la regla no costó nada" de "la ejecución tuvo éxito".

**La corrección.** Una primera versión de este paso reportaba que la escalera terminaba `+0.2000` antes de un óptimo de `+2.0000`. Ese número era un artefacto. Con la carpa en exactamente 10.0, la condición para el escalón superior se cumple en el único punto x = 4.0 y en ningún otro lugar — un óptimo de **ancho cero**, que la cuadrícula de fuerza bruta podía ver (su espaciado de 0.001 resulta contener el 4.0) y en el que ninguna búsqueda podría aterrizar jamás. El defecto fue encontrado al construir la Lección 05, cuyo paso 4 trató de reproducir el déficit y no pudo. La carpa es ahora 10.5, el escalón superior tiene 0.5 de ancho, y el déficit que se inventó para explicar no existe. La regla general está registrada en el docstring de `escalera()`: *un óptimo que la cuadrícula puede ver y la búsqueda no, no es un benchmark, es un bug.*

## Cómo está marcado el código

Dos dispositivos, obligatorios para todo el curso:

- **La receta** — cada script después del primero se abre con un bloque `CAMBIOS RESPECTO A …` que lista, en orden, los cambios que convierten el script anterior en este.
- **Las bandas** — cada elemento de la receta tiene su propia banda `# --- NUEVO (n) nombre ---` en el sitio del cambio. Todo lo que está fuera de una banda es código que los estudiantes ya tienen. El primer elemento del paso 4 es una constante modificada, por lo que marca la línea en su lugar: `MAX_GENERATIONS = 60   # --- CAMBIADO ---`.

## Ejecutándolo

```bash
uv run es/Leccion_02_Flujo_GA/src/flujo_ag_01_las_cinco_fases.py
```

Autónomo, sin argumentos, figuras guardadas en `../figures/`. Toda la secuencia se ejecuta en aproximadamente **8 s**.

## Rutas y estado, 25 de septiembre de 2026

- Scripts compartidos: [`src/`](src/), 5 pasos; figuras reproducibles en [`figures/`](figures/).
- Estudio autónomo: [`estudiante/leccion_02.pdf`](estudiante/leccion_02.pdf) y [`estudiante/leccion_02_flujo_ag.ipynb`](estudiante/leccion_02_flujo_ag.ipynb), con la fuente TeX junto al PDF.
- Uso docente: [`docente/presentacion/leccion_02_compacta.pdf`](docente/presentacion/leccion_02_compacta.pdf) y [`docente/notebooks/leccion_02_compacta.ipynb`](docente/notebooks/leccion_02_compacta.ipynb). La presentación compacta tiene 9 láminas y la libreta contiene comentarios de guía junto al código.

Las piezas extensas conservan la progresión completa; las compactas son autónomas para una ruta de unos 40 minutos. Los ejemplos son sintéticos. La aplicación al reto de Planta Física es una decisión de modelación, no un resultado validado por el socio. La distribución estudiantil sigue bloqueada hasta acreditar derechos del material derivado de Gridin. No se modificó Canvas ni se probó Colab alojado.
