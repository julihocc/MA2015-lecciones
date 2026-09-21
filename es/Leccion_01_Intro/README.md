# Lección 01 — Tu primer algoritmo genético

Se construye un algoritmo genético frente al grupo, un operador a la vez, hasta
que resuelve un problema de maximización de una variable — y después se corre
el mismo código con otra semilla y fracasa, que es justamente a lo que responde
el resto del curso.

- **Origen en el libro:** Gridin, *Learning Genetic Algorithms with Python*,
  Capítulo 1 (*Introduction*) — 4 secciones, ~1,900 palabras, 8 figuras, un
  script de 91 líneas (`03-referencias/…/Chapter01/your_first_genetic_algorithm.py`).
- **Qué le hace la refactorización:** ese script único se parte en **siete pasos
  numerados**, de modo que cada operador tiene su propio archivo ejecutable y su
  propia diapositiva.
- **Duración objetivo:** un bloque de 50 minutos (el promedio del curso). Los
  pasos 1–2 son rápidos; los pasos 3–5 cargan la lección; los pasos 6–7 son la
  recompensa y el gancho para la siguiente.
- **Requisitos previos:** ninguno. Es la primera lección técnica del curso.

## Modelo matemático

El problema es \(\max_{x\in[-10,10]} f(x)\), con
\(f(x)=\sin(x)-0.2|x|\). Un individuo guarda un candidato \(x\) y su aptitud
\(f(x)\). Una población es una muestra finita: su mejor miembro es solo el
**mejor observado**. Una cuadrícula densa da una referencia muestreada, no una
prueba exacta del maximizador continuo. Dos corridas con semilla muestran
comportamientos posibles, no una probabilidad de éxito.

## Con qué se va el estudiante

1. Un problema de optimización se puede atacar sin derivadas y sin saber nada de
   la función más allá de cómo evaluarla.
2. Un algoritmo genético trabaja sobre una **población**, no sobre un punto.
3. Los tres operadores hacen trabajos distintos: la selección **repondera**
   candidatos, la cruza **recombina** y la mutación **propone perturbaciones**.
   Cambiar cualquiera modifica el balance entre explotación y exploración.
4. Un algoritmo genético **no** garantiza encontrar el óptimo. El mismo código,
   los mismos parámetros y otra población inicial se quedan atorados.

## El ejemplo que recorre la lección

```
f(x) = sen(x) - 0.2 * |x|,   x en [-10, 10]      (maximizar)
```

Elegida porque es unidimensional (todo se dibuja sobre un solo eje), multimodal
(cuatro cimas, así que atorarse es un riesgo real y no un cuento) y su máximo
global no es la cima más cercana al origen.

## Los siete pasos

| # | Script | Qué agrega | Qué demuestra su salida |
|---|---|---|---|
| 1 | `primer_ejemplo_01_el_paisaje.py` | `objetivo()`, la gráfica | Hay cuatro cimas; subir por la pendiente se detiene en aquella donde empezó |
| 2 | `primer_ejemplo_02_poblacion_aleatoria.py` | `Individuo`, `crear_aleatorio()`, la población | Diez suposiciones a ciegas no caen cerca del óptimo |
| 3 | `primer_ejemplo_03_seleccion.py` | `seleccion_torneo()` | No aparece ningún valor nuevo de `x` — la selección solo copia; 4 de 10 individuos se extinguen |
| 4 | `primer_ejemplo_04_cruza.py` | `acotar()`, `cruza_mezcla()`, `cruza()` | 14 de 20 hijos caen fuera del intervalo de los progenitores — eso es lo que compra `alfa > 0` |
| 5 | `primer_ejemplo_05_mutacion.py` | `mutacion_gaussiana()`, `mutar()` | En esta muestra, sigma = 1.0 alcanza la colina global 0 veces de 2000 y sigma = 3.0 la alcanza 110 veces; cero eventos observados no prueba probabilidad cero |
| 6 | `primer_ejemplo_06_el_ciclo_completo.py` | el ciclo generacional, la gráfica de convergencia | Diez generaciones encuentran `x = +1.372, f = +0.706`, cerca de la cima más alta; una malla de 400 puntos reporta `x = +1.378, f = +0.706` |
| 7 | `primer_ejemplo_07_optimo_local.py` | nada; `SEMILLA` pasa de 52 a 16 | El mismo algoritmo se asienta en `x = -4.417, f = +0.073` y nunca sale |

**Estos números están verificados contra la salida real**, y son idénticos a los
de la versión en inglés: mismas semillas, misma lógica, solo cambia el idioma.
Cualquier diapositiva o material que cite una cifra debe citar una de estas. Si
el código cambió, vuelve a correrlo en lugar de confiar en esta tabla.

## Cómo está marcado el código

Dos recursos, obligatorios en todo el curso, ya aplicados
aquí:

- **La receta** — cada script después del primero abre con un bloque
  `CAMBIOS RESPECTO A …` en su docstring: la lista ordenada de los cambios que
  convierten el script anterior en este. Ese orden es el orden en que se teclea
  en clase, y el orden que deben seguir las diapositivas.
- **Las bandas** — cada punto de la receta tiene su propia banda
  `# --- NUEVO (n) nombre ---` en el cuerpo, en el lugar exacto del cambio,
  numerada y nombrada igual que la receta. Todo lo que queda fuera de una banda
  es código que los estudiantes ya tienen. El paso 7 cambia un valor en lugar de
  agregar código, así que marca la línea: `SEMILLA = 16   # --- CAMBIADO ---`.

Así, la diapositiva del paso *n* es mecánica de escribir: es la receta del
script *n*, en orden, con el código de las bandas como listados.

## Cómo ejecutarlo

Desde la raíz del repositorio, con `uv` administrando el entorno:

```bash
uv run es/Leccion_01_Intro/src/primer_ejemplo_01_el_paisaje.py
```

Cada script es autocontenido: no importa nada de otros pasos ni de otras
lecciones, y no recibe argumentos de línea de comandos. Las figuras se
**guardan** en `../figuras/`, nunca se muestran, para que toda la secuencia corra
sin supervisión.

Para trabajar la misma secuencia como una sola experiencia estudiantil, abre
[la libreta compacta](notebooks/leccion_01_intro_compacta.ipynb). Integra los
slides compactos con incrementos ejecutables del código, de modo que el
estudiante avance de forma lineal sin alternar entre siete archivos. La
[libreta completa](notebooks/leccion_01_intro.ipynb) conserva la lectura
detallada y los experimentos opcionales; los scripts siguen siendo la
referencia autónoma de cada paso.

## Contenido de la carpeta

```
Leccion_01_Intro/
├── README.md         este archivo — el contrato de la lección
├── src/              los siete scripts numerados
├── figuras/          gráficas generadas (nunca se editan ni se agregan a mano)
├── notebooks/        libretas completa y compacta, y guía de apertura
└── slides/           decks Beamer completo y compacto
```

## Estado

| Pieza | Estado |
|---|---|
| Código | Listo. Siete scripts, todos corren limpio, cifras idénticas a las de inglés. |
| Figuras | Listas. Los siete pasos guardan una figura y fueron revisadas visualmente. |
| Libreta completa | Lista. 104 celdas, 34 de código ejecutadas sin errores y siete figuras embebidas; paridad exacta con los scripts. |
| Libreta compacta | Lista. 35 celdas, 9 de código ejecutadas sin errores y siete figuras embebidas; misma paridad numérica. |
| Diapositivas completas | Listas. Los rangos de código corresponden a las bandas numeradas. |
| Diapositivas compactas | Listas. 17 páginas, compiladas dos veces sin desbordamientos y revisadas visualmente; los listados salen de los scripts canónicos. |

