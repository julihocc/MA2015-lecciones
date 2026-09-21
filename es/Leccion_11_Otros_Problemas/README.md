# Lección 11 — Otros Problemas Comunes

Esta lección desarrolla dos pequeños problemas que tiran en direcciones opuestas. Un
sistema de ecuaciones tiene una respuesta que puede ser *verificada*, y un espacio de búsqueda
lo suficientemente pequeño para ser recorrido. El coloreado de grafos no tiene un objetivo separado en absoluto — la
restricción es el objetivo — y un espacio de 205 billones de coloreados. Juntos
son la ocasión del curso para hacer la pregunta que hasta ahora ha evitado: ¿cuándo
es un algoritmo genético la herramienta equivocada?

- **Fuente en el libro:** Grid Gridin, *Learning Genetic Algorithms with Python*,
  Capítulo 11 — 2 secciones, 1,429 palabras, 8 figuras y 317 líneas a lo largo de 6 archivos.
  El capítulo más ligero del libro.
- **Duración objetivo:** 27 minutos por peso del libro. Ocho pasos, cuatro por problema.
- **Prerrequisitos:** Lecciones 03, 04, 05 y 08. La lección 08 en particular: es
  la lección que estableció que usualmente no puedes decir si una ejecución tuvo éxito,
  y ambos problemas aquí son casos donde sí puedes.

El informe pide de cinco a ocho pasos por ejemplo. Esta lección tiene cuatro por
ejemplo, deliberadamente: 27 minutos no soportarán diez pasos, y ninguno de los problemas
tiene diez hitos de enseñanza en él. Rellenar cualquiera de las secuencias habría significado
inventar contenido que el capítulo no contiene.

## Modelo matemático

La búsqueda de ecuaciones minimiza
\(R(x,y,z)=|f|+|g|+|w|\) en
\([-20,20]^3\cap\mathbb Z^3\), un espacio de \(41^3=68{,}921\) ternas. Un
residuo cero es un testigo verificable; enumerar la caja prueba unicidad allí.
El grafo de 30 vértices tiene (3^{30}=205{,}891{,}132{,}094{,}649)
asignaciones de tres colores. El coloreado minimiza
(C(c)=\sum_{(u,v)\in E}I[c_u=c_v]\). Un coloreado
uniforme e independiente con tres colores cumple \(\mathbb E[C]=|E|/3\). Cero
conflictos prueba factibilidad; el retroceso exacto, no el AG, prueba que dos
colores son imposibles.

## Lo que el estudiante se lleva

1. Un objetivo verificable cambia lo que puede ser una regla de parada. Ambas secuencias
   se detienen en una prueba (`residual == 0`, `conflictos == 0`), no en un presupuesto.
2. Cuando la restricción *es* el objetivo, el paisaje cerca de la respuesta es una
   meseta: un conflicto y dos conflictos se ven casi iguales para la búsqueda.
3. Operadores basados en aptitud — hijos que deben vencer a sus padres, mutaciones
   que se conservan solo cuando ayudan — compran un mejor promedio y cuestan evaluaciones extra.
   Ambas mitades de ese intercambio se miden aquí, no se afirman.
4. Una caja finita de enteros puede simplemente enumerarse, y la enumeración responde
   preguntas que un algoritmo genético no puede: cuántas soluciones existen, y si
   la instancia tiene solución en absoluto.
5. Un algoritmo genético se gana su lugar cuando el espacio deja de ser recorrible —
   no cuando resulta vencer a un bucle que podrías haber escrito en cuatro líneas.

## Los ocho pasos

### Sistema de ecuaciones — la respuesta puede verificarse

El ejemplo es el sistema entero no lineal de Gridin en `x, y, z`. Es elegido
porque es el raro problema en este curso cuya respuesta es verificable por
sustitución, y cuyo espacio de búsqueda es finito y pequeño.

| # | Script | Resultado verificado |
|---|---|---|
| 1 | `ecuaciones_01_el_sistema.py` | Los genes son enteros en `[-20, 20]`, así que el espacio completo es de 68,921 tripletas. Los residuales son enteros, no flotantes: a lo largo de la línea `(y, z) = (1, 1)` van de 3 dígitos en `x = 2` a 31 dígitos en `x = 20` |
| 2 | `ecuaciones_02_poblacion_aleatoria.py` | 400 extracciones uniformes (0.58% de la caja) encuentran 0 soluciones; el mejor es `(0, 1, 2)` con residual 453, mientras que la extracción mediana tiene un residual de 957 dígitos y la peor 14,187 dígitos |
| 3 | `ecuaciones_03_ag_completo.py` | La semilla 3 alcanza un residual de 0 en la generación 5 después de 2,794 evaluaciones, en `x = -6, y = 2, z = 3`; `f`, `g` y `w` son cada una individualmente 0 |
| 4 | `ecuaciones_04_busqueda_exhaustiva.py` | Las 5 semillas probadas encuentran la misma raíz, en 5–28 generaciones y 2,794–13,764 evaluaciones. Enumerar la caja cuesta 68,921 evaluaciones (alrededor de 10× más, aproximadamente 3.7 s) y devuelve estrictamente más: hay **exactamente una** raíz en la caja |

El paso 4 no termina de la forma en que se planeó que terminara. La expectativa era que
la búsqueda exhaustiva expusiera al algoritmo genético como un desperdicio. Hace lo
contrario — el AG es aproximadamente diez veces más barato en evaluaciones y confiable en
cada semilla probada. El argumento en su contra aquí, por lo tanto, no es el costo sino
la *afirmación*: la enumeración prueba la unicidad, no tiene semilla, no tiene tamaño de población y no
tiene tasa de cruza, y siempre devuelve la misma respuesta. El veredicto se invierte con la
caja, que cuesta `lado³`: ensancharla a `[-100, 100]` es 8,120,601 tripletas,
118 veces esta ejecución.

### Coloreado de grafos — la restricción es el objetivo

La instancia es la de Gridin, mantenida literalmente: 30 vértices, 64 aristas, grados de 3 a 6.
La aptitud es el conteo negado de aristas cuyos extremos comparten un color, así que un coloreado legal
y un coloreado óptimo son el mismo objeto.

| # | Script | Resultado verificado |
|---|---|---|
| 1 | `coloreado_01_el_grafo.py` | 30 vértices, 64 aristas distintas, grado medio 4.27, y 3³⁰ = 205,891,132,094,649 coloreados. Un coloreado aleatorio en la semilla 7 rompe 26 de las 64 aristas |
| 2 | `coloreado_02_poblacion_aleatoria.py` | 500 coloreados aleatorios producen 0 legales; la media de la muestra es 21.51 conflictos frente a los previstos `\|E\|/3 = 21.33`, y la mejor extracción aún rompe 11 aristas (17.2%) |
| 3 | `coloreado_03_operadores_basados_en_aptitud.py` | La cruza sencilla de dos puntos mueve la media de la población en +0.03 aristas; la versión basada en aptitud en −2.32. El recoloreado codicioso mueve la media en −1.02 pero deja intactos 206 de los 500 individuos, y su mejor individuo (11) es *peor* que el mejor de un recoloreado ciego (9) |
| 4 | `coloreado_04_busqueda_completa.py` | **1 de 4 semillas** alcanza un coloreado legal (semilla 11, generación 15). Las semillas 7 y 16 se estancan en 2 conflictos y la semilla 1 en 1 conflicto, todas aún atascadas después de las 200 generaciones completas. El backtracking encuentra un 3-coloreado adecuado en 40 nodos de búsqueda y prueba que no existe ningún 2-coloreado en 4 nodos — 0.69 ms para ambos |

Este fracaso es el punto de la lección y no debe reescribirse como un éxito.
Gridin reporta que el algoritmo genético resuelve esta instancia; medido en cuatro
semillas lo hace una vez. La instancia realmente es 3-coloreable — el paso de
backtracking lo prueba — así que la ejecución es un fallo genuino, no un problema imposible. Y
a partir de la ejecución por sí sola, no se podría distinguir entre esos dos casos, lo cual es exactamente
el problema de la lección 08 apareciendo en un caso donde una sola llamada a función lo resuelve.

El paso 3 también conlleva un pequeño e honesto detalle que vale un minuto en clase: la codicia
mejora el promedio y renuncia a la cola afortunada.

## Cómo está marcado el código

Cada script después del primero en cada secuencia abre su docstring con una
**receta** — la lista numerada y ordenada de cambios que convierten el script anterior
en este. Cada elemento de la receta tiene exactamente una **banda** correspondiente en el cuerpo,
que lleva el mismo número y el mismo nombre:

```python
# --- NUEVO (3) coloreado_exacto() ---------------------------------------------
def coloreado_exacto(k: int) -> Tuple[List[int], int]:
    ...
# ------------------------------------------------------------------------------
```

Todo lo que está fuera de una banda es código que los estudiantes ya tienen. Las bandas nunca se
arrastran hacia adelante: marcan lo que es nuevo solo en *este* paso. Un paso que cambia un valor
o una firma en lugar de agregar código marca la línea en sí con
`# --- CAMBIADO ---`; `ecuaciones_04` hace esto donde `ejecutar()` comienza a tomar una semilla.

## Ejecutándolo

Desde `MA2015 lessons/`:

```bash
uv run es/Leccion_11_Otros_Problemas/src/ecuaciones_01_el_sistema.py
```

Los ocho scripts se ejecutan de principio a fin en unos 25 segundos, siendo los más lentos
`coloreado_04_busqueda_completa.py` (cuatro ejecuciones de 200 generaciones, ~7.5 s) y
`ecuaciones_04_busqueda_exhaustiva.py` (~4 s). Las figuras siempre se guardan en
`figures/`, nunca se muestran.

**Esta lección es la única en el curso que necesita un paquete más allá de numpy,
matplotlib y pandas:** `python-igraph`, utilizado por `coloreado_01` y
`coloreado_04` para el diseño del grafo. Está declarado en `pyproject.toml`. Consulta

## Contenido de la carpeta

| Ruta | Contenido |
|---|---|
| `src/` | Ocho scripts numerados: `ecuaciones_01`–`ecuaciones_04`, `coloreado_01`–`coloreado_04` |
| `figures/` | Ocho PNGs, uno por script, cada uno nombrado según el script que lo produce |

## Estado

| Pieza | Estado |
|---|---|
| Código | Hecho. Ocho scripts numerados se ejecutan limpios y pasan el verificador de recetas/bandas. |
| Figuras | Hecho. Una figura generada por cada script numerado. |
| Espejo en español | Hecho. |

## Cuadernos guiados

Entrada para estudiantes: [`notebooks/README.md`](notebooks/README.md). Esta lección tiene 2 secuencias ejecutables de forma independiente: [`leccion_11_ecuaciones.ipynb`](notebooks/leccion_11_ecuaciones.ipynb), [`leccion_11_coloreado_grafos.ipynb`](notebooks/leccion_11_coloreado_grafos.ipynb). Las salidas retenidas se validaron el date omitted contra los scripts congelados en tres modos de ruta con kernels nuevos; el README de cuadernos registra el entorno y la brecha explícita de Colab alojado.

