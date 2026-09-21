# Lección 08 — Funciones de Caja Negra

Esta lección elimina las garantías visuales y de búsqueda exhaustiva utilizadas en las
lecciones anteriores. Los estudiantes primero tratan una función de cinco genes como un sistema opaco,
construyen un cromosoma legal de tipo mixto y un GA alrededor de él, luego abren la caja y
descubren que el aparente óptimo es un polo en lugar de un máximo.

- **Fuente en el libro:** Grid Gridin, *Learning Genetic Algorithms with Python*,
    Capítulo 8 (*Black-Box Function*) — 3 secciones, 2,781 palabras, 4 figuras y
  251 líneas distribuidas en 7 archivos.
- **Duración objetivo:** 51 minutos por peso del libro.
- **Prerrequisitos:** Lecciones 02–07, especialmente diagnósticos de población,
  selección, reparación de mutación y evidencia de ejecuciones repetidas.

## Modelo matemático

El espacio legal es un producto cartesiano de intervalos reales, un conjunto
entero y uno categórico. Reparar es proyectar al dominio mixto y cambia la
distribución de los operadores. El denominador expuesto es
(D=(n+1)^2(1+a+b)(120-x^2)r_\phi(x;b,n)+\tfrac12), donde (r_\phi) es el
residuo trigonométrico implementado. Si (D(x)) cambia de signo de
forma continua, la bisección localiza un cero y el objetivo tiene un polo. Los
valores crecientes cerca del polo no identifican un máximo finito: el problema
es no acotado o está mal planteado si el dominio contiene la singularidad.

## Con qué se queda el estudiante

1. Una interfaz de caja negra define entradas legales, pero no dice nada sobre la
   forma, continuidad o acotamiento de su salida.
2. Los genes mixtos reales, enteros y categóricos necesitan reglas explícitas de legalidad.
3. La reparación silenciosa puede hacer que una mutación configurada sea mucho más débil de lo que sugiere su
   probabilidad nominal.
4. El acuerdo entre búsquedas repetidas no prueba que el valor reportado sea
   un óptimo.
5. El conocimiento del dominio sigue siendo necesario cuando un optimizador reporta éxito.

## El ejemplo en curso

`complicated_one(a, b, x, n, fun_name)` combina tres genes reales acotados, un
gen entero y un gen categórico. Fue elegido porque el mismo ejemplo
soporta todo el argumento: la graficación falla, la fuerza bruta se vuelve poco confiable,
los operadores de genes mixtos necesitan reparación y el GA finalmente expone una singularidad.

## Los siete pasos

| # | Script | Qué añade | Qué prueba su salida |
|---|---|---|---|
| 1 | `caja_negra_01_sin_imagen.py` | el dominio declarado y seis cortes unidimensionales | Cinco cortes vivos coinciden en `x = -10.950`, pero sus máximos difieren por `9.23×`; la imagen no ha resuelto el valor |
| 2 | `caja_negra_02_la_cuadricula.py` | un temporizador de evaluación, la cuadrícula del libro y cuatro refinamientos | La cuadrícula del libro cuesta 105,525,000 llamadas; el refinamiento hace que el máximo reportado crezca 61,099 veces en lugar de converger |
| 3 | `caja_negra_03_el_cromosoma.py` | reparación de legalidad y un `Individuo` de tipo mixto | El constructor repara los cinco ejemplos ilegales; la población inicial muestrea solo el 10% del rango declarado de `x` |
| 4 | `caja_negra_04_los_operadores.py` | cruza por gen, mutación y un censo de reparación | 41.5% de las mutaciones enteras forzadas vuelven a su estado original sin cambios; mallas gruesas pueden silenciar la mutación por completo |
| 5 | `caja_negra_05_la_busqueda.py` | selección por rango, el ciclo de generaciones y diagnósticos | La mejor aptitud alcanza `2.278e-09`, pero la curva sigue subiendo y el 97.5% de la población final ha salido del intervalo inicial de `x` |
| 6 | `caja_negra_06_el_veredicto.py` | doce búsquedas independientes | Los campeones ocupan un intervalo estrecho de `x` mientras que su aptitud difiere en 21,504×, lo cual es inconsistente con la convergencia a un máximo finito |
| 7 | `caja_negra_07_abriendo_la_caja.py` | aislamiento del denominador, bisección y una escalera de distancias | El óptimo aparente es un polo cerca de `x = -10.9535836`; la aptitud reportada mide la distancia a la singularidad |

Las afirmaciones numéricas anteriores se volvieron a ejecutar el date omitted. El tiempo en el paso 2
depende de la máquina; los scripts y las diapositivas deben citar el valor medido de la
ejecución actual en lugar de congelarlo como una constante universal.

## Cómo está marcado el código

Cada script después del primero comienza con una receta `CAMBIOS RESPECTO A ...`. Cada
elemento de la receta tiene una banda numerada `NUEVO` correspondiente en el cuerpo del programa, o un
marcador `CAMBIADO` cuando el paso cambia un valor. El verificador del curso confirma que las
cuentas coinciden para los siete scripts.

## Ejecutándolo

Desde `MA2015 lessons/`:

```bash
uv run es/Leccion_08_Caja_Negra/src/caja_negra_01_sin_imagen.py
```

Cada script numerado es autocontenido y guarda sus figuras en
`figures/` sin abrir una ventana interactiva.

## Estado

| Pieza | Estado |
|---|---|
| Código | Terminado. Siete scripts se ejecutan limpios y pasan el verificador de recetas/bandas. |
| Figuras | Terminado. Se generan siete figuras, una por cada script numerado. |

## Cuadernos guiados

Entrada para estudiantes: [`notebooks/README.md`](notebooks/README.md). Esta lección tiene 1 secuencia ejecutable de forma independiente: [`leccion_08_caja_negra.ipynb`](notebooks/leccion_08_caja_negra.ipynb). Las salidas retenidas se validaron el date omitted contra los scripts congelados en tres modos de ruta con kernels nuevos; el README de cuadernos registra el entorno y la brecha explícita de Colab alojado.

