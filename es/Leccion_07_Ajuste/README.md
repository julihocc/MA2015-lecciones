# Lección 07 — Ajuste de Parámetros

El capítulo de laboratorio del libro, reconstruido sobre los instrumentos de la Lección 06. Las tres
perillas — probabilidad de cruza, probabilidad de mutación, tamaño de población — se
giran una a la vez y luego juntas, cada configuración se mide a lo largo de cientos
de corridas y se compara con la búsqueda ciega con igual presupuesto. El hallazgo sobre el que
se construye la lección: en este paisaje, cada perilla es secretamente una perilla de presupuesto,
y la curva de campana que prometen los libros de texto existe solo en el margen
sobre la búsqueda ciega, no en la tasa de éxito bruta.

- **Fuente en el libro:** Gridin, *Learning Genetic Algorithms with Python*,
  Capítulo 7 (*Parameter Tuning*) — 3 secciones, ~4,200 palabras, **42 figuras**,
  248 líneas de código en 4 archivos (`03-referencias/…/Chapter07/`).
- **Lo que la refactorización le hace:** el libro ajusta observando una corrida por
  configuración, cada perilla en un paisaje diferente, a lo largo de 30 ventanas
  `plt.show()` que bloquean. La refactorización mantiene el protocolo del libro como el
  *espécimen* (paso 1, y falla en cámara), luego vuelve a hacer las mismas
  preguntas con la maquinaria de la Lección 06: tasas de éxito sobre 500 corridas,
  errores estándar, conteo de evaluaciones, y un punto de referencia de búsqueda ciega. Un
  paisaje para toda la lección, para que las perillas realmente puedan compararse —
  las 42 figuras se convierten en seis.
- **Duración objetivo:** 79 min por peso en el libro (El capítulo 7 es el 12.1% del
  texto, el segundo capítulo más pesado). Seis pasos: los dos pasos de barrido y la
  cuadrícula se llevan el tiempo; los pasos 1 y 5 son rápidos.
- **Prerrequisitos:** Lección 06 (el instrumento de tasa de éxito, el contador de evaluaciones,
  el punto de referencia ciego — todos reusados, ninguno re-derivado); Lecciones
  01–05 (el algoritmo bajo estudio).

## Modelo matemático

Las configuraciones comparten semillas, así que la comparación es emparejada.
Defina \(d_i=I_{i,B}-I_{i,A}\in\{-1,0,1\}\); la brecha estimada es \(\bar d\),
con error estándar \(s_d/\sqrt n\). Esto sustituye un cálculo independiente que
ignora el emparejamiento. Las tasas conservan intervalos de Wilson, incluida la
observación 500/500. El mapa de 36 celdas es exploratorio para este problema y
estas semillas, no un óptimo universal ni un estudio confirmatorio.

## Con qué se va el estudiante

1. Ajustar desde corridas sueltas es **leer ruido**: el mismo protocolo en 8
   semillas separa dos configuraciones por 0.000004 en una semilla y por 0.6284 en
   otra.
2. Una perilla subida aumenta la tasa de éxito **y el presupuesto** — los dos
   efectos están confundidos hasta que se cuentan las evaluaciones.
3. La moneda honesta es el **margen sobre la búsqueda ciega con igual presupuesto**;
   en esa moneda el margen de cruza se reduce a medida que se abre la perilla, y la
   curva de mutación se dobla hacia la campana de los libros (pico cerca de 0.2, negativo en
   ambos extremos).
4. La población es la perilla de presupuesto sin disfraz: una puntuación perfecta del 100%
   simplemente se puede comprar, a costa de un colapso de cinco veces en el rendimiento.
5. Cada veredicto de ajuste es un veredicto **sobre un problema**. Una configuración fija
   es un compromiso — por eso existe el AG adaptativo (Lección 12).

## El ejemplo en ejecución

El paisaje de seno de la Lección 01, `f(x) = sin(x) - 0.2|x|` en `[-10, 10]`, óptimo
+0.705908 conocido por fuerza bruta, éxito significa estar dentro de 0.01 de él. Un
paisaje para toda la lección, a propósito: el libro gira cada perilla en una
función diferente, por lo que su capítulo nunca compara nada con nada. Los
operadores son los del propio curso (torneo 3, mezcla alfa 1.0, mutación
gaussiana), así que cada número continúa de las Lecciones 01–06. El precio de esta
elección se dice en voz alta en el paso 6: el objetivo aquí es *amplio* (1.43% de
la caja), el dinero a ciegas llega lejos, y los veredictos son veredictos sobre este
problema.

## Los seis pasos

| # | Script | Qué añade | Qué prueba su salida |
|---|---|---|---|
| 1 | `afinacion_01_corridas_sueltas.py` | el protocolo del libro, fielmente (pob 16, mutación 0.2, semilla 63) | Una corrida por config corona pc = 0.7 por 0.0021; sobre 8 semillas las victorias van 0/3/5, la división más estrecha es 0.000004 (semilla 3) y la más ancha 0.6284 (semilla 1) — el ranking es inestable |
| 2 | `afinacion_02_el_instrumento.py` | `error_estandar()`, resultados emparejados en `medir()`, perillas como argumentos, CORRIDAS = 500 | La tasa observada sube 55.6% → 83.0%; la brecha de 27.4 puntos se reporta con el error estándar de diferencias 0/1 emparejadas, mientras aún falta la columna de costo |
| 3 | `afinacion_03_la_perilla_del_presupuesto.py` | el contador de evaluaciones, `exito_ciego()`, la columna de margen | Cruza 0.0 gasta 20 evals/corrida, 1.0 gasta 120; el margen sobre ciego cae +30.6 → +0.8 puntos; las 100 evaluaciones extra le compran al AG 27.4 puntos donde 100 extracciones ciegas compran 57.2 — la perilla es de presupuesto |
| 4 | `afinacion_04_mutacion.py` | la segunda perilla (7 configuraciones, cruza fija 0.8) | La tasa bruta es monótona hasta mutación 1.0 (61.8% → 89.8%) — no hay curva de campana; el margen se dobla: −10.9% en 0.0 (peor que dados), pico +5.9% cerca de 0.2, −3.7% en 1.0 — la curva vive en el margen |
| 5 | `afinacion_05_poblacion.py` | la tercera perilla (los tamaños 6, 10, 20, 50 del libro) | Población 50 registra 500/500 éxitos (intervalo de Wilson al 95%: 99.2%–100%) a 550 evals/corrida; el rendimiento colapsa 9.25 → 1.82 éxitos por 1000 evals; el margen hace pico en población 10 (+5.9%) y desaparece en 50 — "¿cuál es mejor?" no tiene respuesta sin moneda |
| 6 | `afinacion_06_la_cuadricula.py` | cuadrícula 6×6, ambas perillas, CORRIDAS = 100 | La esquina (0, 0) recupera la búsqueda ciega (9.0% ± 2.9% vs 13.4% para 10 extra ciegos — el instrumento está calibrado); la mejor celda (1.0, 0.5) puntúa 90.0% a 160 evals con margen de −0.01%; solo 20 de 36 celdas superan su presupuesto — los operadores se ganan el sueldo en las baratas |

**Estos números se verifican contra salida real.** Cualquier diapositiva, apunte o
traducción que indique una cifra debe indicar una de estas, no un sustituto
plausible. Reejecuta el script en lugar de confiar en esta tabla si el código ha
cambiado desde entonces.

## Cómo está marcado el código

Dos recursos, ambos obligatorios para todo el curso, ambos aplicados
aquí:

- **La receta** — todo script después del primero abre con un bloque `CAMBIOS RESPECTO A …`.
  Los pasos 4 y 5 son recetas de un ítem a propósito: el instrumento se
  construye una vez (pasos 2–3) y cada paso posterior solo lo re-apunta, marcado con un
  único bloque `# --- CAMBIADO ---` en las constantes de diseño de estudio.
- **Las bandas** — cada ítem de receta tiene su propia banda `# --- NUEVO (n) nombre ---`
  en el sitio del cambio. El contador del paso 3 tiene dos sitios (la clase `Individuo`
  y las líneas de colección dentro de `medir()`); la banda se sienta en la
  clase y las líneas de colección llevan un comentario de puntero.

La corrida i usa la semilla i en todo, por lo que cada configuración y cada celda enfrenta los
mismos dados y todas las comparaciones están emparejadas.

## Ejecutándolo

Desde la raíz del repositorio, con `uv` manejando el entorno:

```bash
uv run es/Leccion_07_Ajuste/src/afinacion_01_corridas_sueltas.py
```

Cada script es autónomo: sin importaciones entre pasos, sin importaciones entre
lecciones, sin argumentos de línea de comandos. Las figuras se **guardan** en `../figuras/`,
nunca se muestran, por lo que toda la secuencia corre desatendida — unos 15 segundos para
los seis pasos.

## Contenido de la carpeta

```
Leccion_07_Ajuste/
├── README.md      este archivo — el contrato de la lección
├── src/           los seis scripts numerados
├── figuras/       gráficos generados (nunca editados ni commiteados a mano)
```

## Estado

| Pieza | Estado |
|---|---|
| Código | Hecho. Seis scripts, corren limpios, afirmaciones verificadas. |
| Figuras | Hecho. Cada paso guarda una figura, nombrada según su script. |
| Diapositivas | No iniciado, y bloqueado por la plantilla Beamer. |
| Espejo español | Hecho. Reemplaza el monolito previo a la refactorización. |

## Cuadernos guiados

Entrada para estudiantes: [`notebooks/README.md`](notebooks/README.md). Esta lección tiene 1 secuencia ejecutable de forma independiente: [`leccion_07_afinacion.ipynb`](notebooks/leccion_07_afinacion.ipynb). Las salidas retenidas se validaron el date omitted contra los scripts congelados en tres modos de ruta con kernels nuevos; el README de cuadernos registra el entorno y la brecha explícita de Colab alojado.

