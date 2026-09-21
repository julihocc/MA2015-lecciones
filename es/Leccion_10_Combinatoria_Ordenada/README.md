# Lección 10 — Optimización Combinatoria Ordenada

Esta lección desarrolla dos problemas en los cuales el orden y la unicidad en el cromosoma son parte de la correctitud. La secuencia del TSP preserva una permutación de 48 ciudades; la secuencia del equipo preserva once jugadores distintos en espacios de plantilla específicos por rol.

- **Fuente en el libro:** Grid Gridin, *Learning Genetic Algorithms with Python*,
  Capítulo 10 — 2 secciones, 2,883 palabras, 12 figuras y 523 líneas en 9 archivos.
- **Duración objetivo:** 53 minutos por peso del libro. Utilizar el TSP como la lección principal y la plantilla como una aplicación o continuación.
- **Prerrequisitos:** Lecciones 04, 05 y 09.

## Modelo matemático

Una ruta legal es una permutación \(\pi\) con longitud cerrada
\(L(\pi)=\sum_i d(\pi_i,\pi_{i+1})\), uniendo la última ciudad con la primera.
OX1 e inversión conservan pertenencia; la cruza ordinaria de un punto no. Una
plantilla legal contiene once identificadores distintos en posiciones por rol,
costo dentro del presupuesto y aptitud igual a la habilidad total. Los
experimentos certifican legalidad y comparan líneas base; no certifican óptimos
globales.

## Lo que el estudiante se lleva

1. Un cromosoma ordenado legal contiene cada miembro requerido exactamente una vez.
2. La cruza ordinaria de un punto usualmente duplica y pierde miembros.
3. La cruza ordenada, el intercambio y la inversión preservan la legalidad pero se mueven diferente.
4. Un resultado de AG válido aún necesita una línea base.
5. La posición, la unicidad, el presupuesto y la calidad deberían seguir siendo auditables por separado.

## Los diez pasos

### Agente viajero (TSP)

| # | Script | Resultado verificado |
|---|---|---|
| 1 | `tsp_01_la_ruta.py` | La instancia proporcionada contiene 48 ciudades; la ruta identidad es legal pero larga |
| 2 | `tsp_02_poblacion_aleatoria.py` | Todas las 500 rutas barajadas son legales, con una amplia distribución de longitudes |
| 3 | `tsp_03_cruza_ordenada.py` | La cruza de un punto produce 0/1,000 hijos legales; la cruza ordenada produce 1,000/1,000 |
| 4 | `tsp_04_mutacion.py` | El intercambio y la inversión preservan cada ciudad; sus cambios medios de longitud son +451.0 y +238.9 |
| 5 | `tsp_05_la_busqueda_completa.py` | La ruta del AG es legal pero 57.2% más larga que el vecino más cercano después de 10,100 evaluaciones |

### Plantilla de fútbol

| # | Script | Resultado verificado |
|---|---|---|
| 1 | `equipo_01_la_plantilla.py` | El grupo respaldado por la fuente contiene 12 candidatos para cada bloque de rol y un presupuesto explícito de €600m |
| 2 | `equipo_02_plantillas_aleatorias.py` | Todas las 1,000 plantillas contienen once jugadores distintos; solo 440 cumplen con el presupuesto |
| 3 | `equipo_03_reparacion_presupuesto.py` | La reparación hace que 1,000/1,000 equipos sean asequibles y preserva la unicidad |
| 4 | `equipo_04_operadores_ordenados.py` | Todos los 100 hijos siguen siendo legales; la habilidad media sube de 979.6 a 985.3 |
| 5 | `equipo_05_la_busqueda_completa.py` | El AG alcanza una habilidad de 991 con €597.5m y empata a la mejor plantilla aleatoria reparada a iguales evaluaciones |

El fracaso del TSP y el empate de la plantilla son parte de la lección. Ningún resultado debería reescribirse como una victoria algorítmica.

## Ejecución

Desde `MA2015 lessons/`:

```bash
uv run es/Leccion_10_Combinatoria_Ordenada/src/tsp_01_la_ruta.py
```

`att48_xy.txt` y `players_20.csv` permanecen junto a los scripts. La Lección 12 podría reusar los conceptos y nombres del TSP, pero sus scripts deben redeclarar lo que necesitan localmente en lugar de importar entre lecciones.

## Estado

| Pieza | Estado |
|---|---|
| Código | Hecho. Diez scripts numerados se ejecutan limpiamente y pasan el verificador de recetas/bandas. |
| Figuras | Hecho. Una figura generada por cada script numerado. |

## Cuadernos guiados

Entrada para estudiantes: [`notebooks/README.md`](notebooks/README.md). Esta lección tiene 2 secuencias ejecutables de forma independiente: [`leccion_10_tsp.ipynb`](notebooks/leccion_10_tsp.ipynb), [`leccion_10_equipo.ipynb`](notebooks/leccion_10_equipo.ipynb). Las salidas retenidas se validaron el date omitted contra los scripts congelados en tres modos de ruta con kernels nuevos; el README de cuadernos registra el entorno y la brecha explícita de Colab alojado.

