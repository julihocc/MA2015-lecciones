# Lección 09 — Optimización Combinatoria Binaria

Esta lección desarrolla tres problemas binarios en paralelo: seleccionar artículos,
asignar empleados a turnos y ubicar radares. Cada ejemplo recibe su propia
secuencia de cinco pasos para que los estudiantes puedan ver cómo la representación, la factibilidad y el
objetivo cambian mientras los operadores binarios siguen siendo reconocibles.

- **Fuente en el libro:** Grid Gridin, *Learning Genetic Algorithms with Python*,
  Capítulo 9 — 3 secciones, 6,157 palabras, 38 figuras y 1,283 líneas en 21 archivos.
- **Duración objetivo:** 114 minutos por el peso del libro. Enséñelo como tres bloques o
  divídalo en sesiones; quince pasos no caben en una clase ordinaria.
- **Prerrequisitos:** Lecciones 02–07.

## Modelo matemático

Mochila maximiza \(\sum_i v_i x_i\) sujeto a \(\sum_i w_i x_i\le C\). Radar
minimiza \(U(x)(|S|+1)+\sum_i x_i\); como se activan a lo sumo \(|S|\)
radares, reducir en uno el número de objetivos sin cobertura domina cualquier cambio en su
cantidad. Para bits de horario (x_{eds}), se minimiza
(\sum_{eds}c_{eds}x_{eds}), sujeto a
(\sum_e x_{eds}=d_s) y (\sum_{ds}x_{eds}\le5).
Reparar transforma cadenas crudas en factibles y cambia la distribución que el
AG realmente busca.

## Lo que el estudiante se lleva

1. Un cromosoma binario puede codificar una selección, asignación o ubicación, pero el
   significado de un bit proviene del problema.
2. La factibilidad debe permanecer visible. Una penalización, regla de reparación y respuesta exacta de referencia
   responden preguntas diferentes.
3. La selección no puede rescatar a una población cuyos miembros no factibles empatan todos.
4. La reparación cambia la distribución que se está buscando y, por lo tanto, debe ser
   medida en lugar de ser tratada como fontanería gratuita.
5. Las instancias pequeñas deben verificarse de forma exacta; las más grandes necesitan líneas base honestas.

## Los quince pasos

### Mochila (Knapsack)

| # | Script | Resultado verificado |
|---|---|---|
| 1 | `knapsack_01_la_instancia.py` | La instancia de 12 artículos tiene 4,096 cromosomas; el óptimo exacto tiene un valor de 50 con un peso de 20 |
| 2 | `knapsack_02_poblacion_aleatoria.py` | 233 de 1,000 cromosomas aleatorios son factibles; el mejor valor aleatorio es 47 |
| 3 | `knapsack_03_reparacion.py` | La reparación hace factibles a 1,000/1,000 y cambia 767 cromosomas |
| 4 | `knapsack_04_seleccion_y_cruza.py` | Una generación eleva el valor medio de 39.00 a 40.73; todos los hijos almacenados siguen siendo factibles |
| 5 | `knapsack_05_la_busqueda_completa.py` | El AG alcanza el valor exacto 50 con un peso de 20 con brecha cero en 4,060 evaluaciones |

### Horarios (Scheduling)

| # | Script | Resultado verificado |
|---|---|---|
| 1 | `schedule_01_los_requisitos.py` | 168 bits codifican 35 asignaciones requeridas bajo la capacidad 40 |
| 2 | `schedule_02_horarios_aleatorios.py` | 0 de 500 horarios aleatorios son factibles; el mejor todavía tiene 14 violaciones |
| 3 | `schedule_03_reparacion.py` | La reparación hace factibles a 500/500 horarios; los costos reparados van desde 58 en adelante |
| 4 | `schedule_04_operadores.py` | Todos los 100 hijos son factibles; el costo medio cae de 77.8 a 61.2 |
| 5 | `schedule_05_la_busqueda_completa.py` | El AG con un costo de 41 vence al mejor de 5,100 horarios aleatorios reparados, con costo 52, con igual conteo de evaluaciones |

### Ubicación de radares (Radar placement)

| # | Script | Resultado verificado |
|---|---|---|
| 1 | `radar_01_el_paisaje.py` | Nueve sitios definen 512 planes y juntos cubren los 36 objetivos |
| 2 | `radar_02_el_cromosoma.py` | 21 de 500 planes aleatorios cubren todos los objetivos |
| 3 | `radar_03_la_penalizacion.py` | La penalización auditada clasifica a cada plan factible por encima de cada plan no factible |
| 4 | `radar_04_operadores.py` | Una generación reduce el objetivo medio de 86.0 a 42.6 |
| 5 | `radar_05_la_busqueda_completa.py` | El AG y la enumeración exhaustiva encuentran cinco radares con cobertura completa |

## Ejecución

Desde `MA2015 lessons/`:

```bash
uv run es/Leccion_09_Combinatoria_Binaria/src/knapsack_01_la_instancia.py
```

Cada script numerado es independiente, fija su semilla y guarda una figura
en `figures/`. Los scripts después del primero de cada prefijo llevan recetas correspondientes
y bandas de código numeradas.

## Estado

| Pieza | Estado |
|---|---|
| Código | Terminado. Quince scripts numerados se ejecutan limpiamente y pasan el verificador de recetas/bandas. |
| Figuras | Terminado. Una figura generada por script. |

## Cuadernos guiados

Entrada para estudiantes: [`notebooks/README.md`](notebooks/README.md). Esta lección tiene 3 secuencias ejecutables de forma independiente: [`leccion_09_mochila.ipynb`](notebooks/leccion_09_mochila.ipynb), [`leccion_09_horarios.ipynb`](notebooks/leccion_09_horarios.ipynb), [`leccion_09_radar.ipynb`](notebooks/leccion_09_radar.ipynb). Las salidas retenidas se validaron el date omitted contra los scripts congelados en tres modos de ruta con kernels nuevos; el README de cuadernos registra el entorno y la brecha explícita de Colab alojado.

