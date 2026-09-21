# Lección 06 — Midiendo la Efectividad

Cada lección hasta ahora terminó leyendo una sola corrida; esta construye la
maquinaria para leer mil. Un algoritmo genético es una variable aleatoria,
por lo que se debe definir una tasa de éxito (¿qué cuenta como éxito?), medirse (¿cuántas
corridas?) y cotizarse (¿cuánto costaron las evaluaciones?) — y cada una de las
tres preguntas tiene su propio instrumento y su propia sorpresa.

- **Fuente en el libro:** Gridin, *Learning Genetic Algorithms con Python*,
  Capítulo 6 (*Efectividad de Algoritmos Genéticos*) — 4 secciones, ~2,100
  palabras, 8 figuras, 301 líneas de código en 5 archivos
  (`03-referencias/…/Chapter06/`).
- **Qué le hace la refactorización:** los tres experimentos del capítulo (estadísticas de una sola corrida,
  el histograma de 1000 corridas, la dispersión población-vs-evaluaciones)
  se convierten en la columna vertebral de **cuatro pasos numerados**. El Paso 2 inserta un paso que el libro
  no tiene — definir el "éxito" antes de contarlo — porque el propio paisaje 2-D del libro hace que la definición ingenua carezca de sentido de forma medible.
- **Duración objetivo:** 39 min por peso del libro (El Capítulo 6 es el 6.0% del texto).
  Cuatro pasos en lugar de los habituales cinco a ocho: cada diferencia es una
  idea del tamaño de la Lección 01, y el presupuesto no puede contener más. Marcado como una
  elección deliberada, no un descuido.
- **Requisitos previos:** Lecciones 01–05 (el algoritmo se reutiliza sin cambios);
  Lección 02 paso 5 (la cuadrícula de fuerza bruta, reutilizada como el instrumento del veredicto);
  Lección 05 (el hallazgo de "óptimo sin volumen", que regresa en dos
  dimensiones).

## Modelo matemático

Para \(n\) corridas independientes y \(s\) éxitos, \(\hat p=s/n\) y el error
estándar por sustitución es \(\sqrt{\hat p(1-\hat p)/n}\). Es una desviación
estándar estimada, no un intervalo de confianza; los scripts también reportan
un intervalo de Wilson al 95%, no degenerado con 0 o \(n\) éxitos. El óptimo y
la proporción objetivo son aproximaciones en cuadrícula. Si una extracción ciega
acierta con probabilidad \(q\), un presupuesto \(E\) acierta con
\(1-(1-q)^E\); presupuestos variables se promedian corrida por corrida.

## Qué se lleva el estudiante

1. Un AG es una **variable aleatoria**: la misma configuración en 8 semillas esparce
   sus respuestas más amplio (0.8946) que la mejora completa de una corrida (0.5822).
2. Una tasa de éxito necesita un **veredicto de fuera de la corrida** — una referencia de cuadrícula muestreada y una tolerancia declarada — y el veredicto solo es significativo si el
   objetivo tiene **volumen**: una parte medible del espacio de búsqueda.
3. Una tasa medida en 8 corridas tiene un error estándar por sustitución de
   ±17 puntos; en 1000 corridas es ±1.3. Es una desviación estándar estimada,
   no un intervalo de confianza; la precisión mejora solo con la raíz cuadrada.
4. La efectividad y la **eficiencia** tiran en direcciones opuestas: la población más
   confiable es la menos rentable por evaluación.
5. En un objetivo amplio, el AG apenas supera a la búsqueda a ciegas con el mismo presupuesto — una
   propiedad del problema, no un defecto del algoritmo.

## Los ejemplos en ejecución

Dos paisajes, a propósito:

- **La función 2-D del libro** `f(x, y) = sin(x)cos(x) - (|(x+50)(y-10)|/10)^0.1`
  en `[-100, 100]²` — pasos 1 y 2. Mantenido como el *espécimen*: su óptimo es
  real (+0.500000) pero no tiene volumen, que es exactamente lo que lo hace
  el contraejemplo perfecto para definir el éxito.
- **El paisaje de seno de la Lección 01** `f(x) = sin(x) - 0.2|x|` en `[-10, 10]` —
  pasos 2, 3 y 4. Elegido porque su óptimo es un pico suave con un
  ancho medible (1.43% de la caja), por lo que una tasa de éxito allí significa
  algo — y porque conecta la lección nuevamente con la falla del
  óptimo local de la Lección 01, que el paso 3 mide a escala.

Los operadores son los propios del curso, sin cambios: torneo de 3, cruza
mezclada con alfa = 1.0, mutación gaussiana detrás de una moneda por individuo.
Cada número en esta lección continúa las lecciones anteriores en lugar de
reiniciarlas.

## Los cuatro pasos

| # | Script | Qué añade | Qué prueba su salida |
|---|---|---|---|
| 1 | `tasa_exito_01_una_corrida.py` | el AG del curso en el paisaje 2-D del libro, ejecutado 8 veces | La dispersión entre semillas es 0.8946 — 1.5× toda la ganancia de la corrida rastreada (+0.5822); la semilla 3 reporta -0.0129, la semilla 4 reporta -0.9076, y ambos reportes son verdaderos |
| 2 | `tasa_exito_02_el_veredicto.py` | `optimo_fuerza_bruta()`, `TOLERANCIA`, `veredicto()`, `proporcion_dentro_de_tolerancia()`, el movimiento al seno 1-D | El óptimo 2-D es real (+0.500000) pero ocupa el 0.003197% de la cuadrícula — los 128 puntos ganadores se sitúan exactamente en y = 10 — así que 0 de 8 corridas "tienen éxito"; en seno 1-D el objetivo es el 1.4299% de la caja (0.286 de ancho) y 5 de 8 corridas lo encuentran: 62.5% ± 17.1% |
| 3 | `tasa_exito_03_mil_corridas.py` | `tasa_exito()`, `error_estandar()`, `intervalo_wilson()`, `CORRIDAS = 1000` | La tasa observada es 79.4%, un error estándar por sustitución es 1.3% y el intervalo de Wilson al 95% se reporta aparte; la estimación de 8 corridas difiere por 16.9 puntos y 71/1000 terminan en el pico local x = -4.51 |
| 4 | `tasa_exito_04_el_presupuesto.py` | el contador de evaluaciones, el barrido de población, `exito_ciegas()` | Población 10→30: éxito 80.0%→99.4%, costo 100→300 evaluaciones, el rendimiento cae 8.00→3.31 éxitos por cada 1000 evaluaciones; extracciones a ciegas con igual presupuesto alcanzan 76.2% y 98.7% — el margen del AG es +3.8 y +0.7 puntos |

**Estos números están verificados contra resultados reales.** Cualquier diapositiva, folleto o
traducción que exponga una cifra debe exponer una de estas, no un sustituto plausible.
Vuelve a ejecutar el script en lugar de confiar en esta tabla si el código ha
cambiado desde entonces.

## Cómo está marcado el código

Dos dispositivos, ambos obligatorios a lo largo de todo el curso, ambos aplicados
aquí:

- **La receta** — cada script después del primero se abre con un bloque `CAMBIOS RESPECTO A …`:
  los cambios ordenados que convierten el script anterior en este.
  Los pasos 3 y 4 también llevan una nota `ELIMINADO DE …`, porque cada uno omite
  el código de espécimen de una sola vez del paso anterior; la nota no está numerada a propósito,
  para que la receta mantenga su mecánica de un número por banda.
- **Las bandas** — cada elemento de la receta tiene su propia banda `# --- NUEVO (n) nombre ---`
  en el sitio del cambio. Un valor cambiado está marcado en la línea en su lugar:
  `CORRIDAS = 1000   # --- CAMBIADO ---` en el paso 3, `CORRIDAS = 500` en el paso 4.

La corrida i utiliza la semilla i en toda la lección, por lo que cualquier subconjunto de corridas es
reproducible por sí mismo — y las primeras 8 de las mil corridas del paso 3 *son*
el experimento del paso 2, semilla por semilla, lo cual hace que la comparación sea
honesta.

## Ejecutándolo

Desde la raíz del repositorio, con `uv` manejando el entorno:

```bash
uv run es/Leccion_06_Efectividad/src/tasa_exito_01_una_corrida.py
```

Cada script es autónomo: sin importaciones a través de los pasos, sin importaciones a través de las
lecciones, sin argumentos de línea de comandos. Las figuras se **guardan** en `../figures/`,
nunca se muestran, por lo que toda la secuencia se ejecuta desatendida — unos 8 segundos para los
cuatro pasos.

## Contenido de la carpeta

```
Leccion_06_Efectividad/
├── README.md      este archivo — el contrato de la lección
├── src/           los cuatro scripts numerados
├── figures/       gráficos generados (nunca editados a mano, nunca comprometidos a mano)
```

## Estado

| Pieza | Estado |
|---|---|
| Código | Hecho. Cuatro scripts, todos se ejecutan limpios, todas las afirmaciones verificadas contra la salida. |
| Figuras | Hecho. Cada paso guarda una figura, nombrada según su propio script. |
| Diapositivas | No iniciado, y bloqueado en la plantilla Beamer del curso. |
| Espejo español | Hecho. |

## Cuadernos guiados

Entrada para estudiantes: [`notebooks/README.md`](notebooks/README.md). Esta lección tiene 1 secuencia ejecutable de forma independiente: [`leccion_06_tasa_exito.ipynb`](notebooks/leccion_06_tasa_exito.ipynb). Las salidas retenidas se validaron el date omitted contra los scripts congelados en tres modos de ruta con kernels nuevos; el README de cuadernos registra el entorno y la brecha explícita de Colab alojado.

