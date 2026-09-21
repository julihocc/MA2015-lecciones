# Lección 13 — Mejorando el Rendimiento

La última lección del curso, y la única sobre lo que cuesta un algoritmo genético en lugar de sobre lo que encuentra. Un problema de asignación de turnos se lleva a través de siete pasos mientras la búsqueda en sí se mantiene fija: la respuesta es el mismo `-87` en seis de ellos, y todo el contenido de la lección es la factura decreciente debajo de ella. El séptimo paso es la excepción, y por esa razón es el último.

- **Fuente en el libro:** Gridin, *Learning Genetic Algorithms with Python*,
  Capítulo 13 (*Improving Performance*) — 5 secciones, 1,717 palabras, 6 figuras,
  644 líneas en 10 archivos (`caching/`, `coarse/`, `snapshot/`, `parallel/`).
  Se cubren las cinco secciones. La sección 13.1 **faltaba en el material anterior de esta lección** y ahora es el paso 2.
- **Presupuesto:** 32 min (el Capítulo 13 es el 4.9% del libro). Siete pasos es más de lo que 32 minutos pueden contener cómodamente — ver *Si la sesión es corta* abajo.
- **Requisitos previos:** Lecciones 02 (el flujo y su censo de evaluación) y 09
  (cromosomas binarios, y un problema de programación de la misma familia).

## Modelo matemático

El tiempo total separa llamadas al objetivo, trabajo elemental por llamada,
serialización y sobrecosto de procesos. La tasa de aciertos es
(h=H/(H+M)); si domina el trabajo del objetivo, la aceleración ideal por
trabajo es (1/(1-h)), mientras que la observada incluye búsqueda,
serialización y procesos. Una caché solo es válida para un
objetivo determinista indexado por el genoma inmutable completo. Un reinicio
exacto necesita población, aptitudes/caché, contadores de generación y estado
pseudoaleatorio. Los workers tienen memoria separada tanto con spawn de Windows
como con fork y copia al escribir en Unix. Para un sustituto $\tilde f$, el
error es $e(x)=\tilde f(x)-f(x)$; los candidatos elegidos con $\tilde f$
deben evaluarse después con el objetivo real $f$.

## La regla de medición en la que se basa esta lección

Una lección sobre velocidad es el lugar más fácil en un curso para imprimir algo falso.
Un número de reloj de pared es un hecho sobre una laptop en una tarde; no es una propiedad del algoritmo, y una diapositiva que lo cite como tal estará equivocada en la habitación contigua.

Por lo tanto, cada script lleva dos contadores y declara sus resultados en ellos:

- `COSTO["llamadas"]` — llamadas a la función de aptitud.
- `COSTO["trabajo"]` — pasos elementales dentro de esas llamadas, uno por celda (empleado, turno) examinada. Una evaluación completa cuesta **210 unidades de trabajo**.

Ambos son idénticos en cada máquina, cada Python y cada ejecución con esta semilla.
**Exactamente un script imprime segundos** — el paso 5, que tiene que hacerlo porque la pregunta que hace es sobre procesos. Los imprime junto a un párrafo que dice lo que valen, y su argumento se basa en la proporción entre dos mediciones tomadas en la misma máquina en la misma ejecución, nunca en ninguno de los números por sí solo.

## Lo que el estudiante se lleva

1. Evaluar una vez y almacenar el resultado es la optimización real más barata disponible, y es gratis: misma semilla, misma respuesta, 54% menos evaluaciones.
2. Una caché vale lo que valen las repeticiones de la búsqueda — medido, no asumido. El mismo diccionario devuelve 14.4% o 3.2% dependiendo de la tasa de mutación, y nada sobre la caché cambia entre esas dos ejecuciones.
3. Posponer la evaluación hasta que termine la cruza y mutación vale más que el pool de procesos para el que fue escrito.
4. El multiprocesamiento no elimina ningún trabajo en absoluto; lo reubica, y aleja las variables globales del programa — caché y contadores incluidos — en el proceso.
5. Solo la aproximación realmente hace que el problema sea más pequeño, y es la única técnica aquí que tiene que ser juzgada en lugar de verificada.

## El ejemplo en ejecución

El prefijo es `turnos_`. Un solo problema en todo: un problema de asignación de turnos de 5 empleados, 7 días y 3 turnos por día, un cromosoma de 105 bits, maximizando
`−(desviaciones de personal + 5 × violaciones de descanso)`, de modo que una asignación perfecta puntúa 0.
Fue elegido porque su objetivo se divide limpiamente a la mitad — un término recorre la asignación por turno, el otro por empleado — que es lo que hace que la versión gruesa del paso 7 sea honestamente un 50% más barata en lugar de un truco, y porque la Lección 09 ya ha enseñado un problema de programación, por lo que no se gasta tiempo de la lección en el dominio.

`SEMILLA = 3` en todo momento. Con ella, el objetivo completo alcanza una aptitud de **−87**, y cada paso del 2 al 6 reproduce exactamente eso.

## Los siete pasos

| # | Script | Qué añade | Qué demuestra su salida |
|---|---|---|---|
| 1 | `turnos_01_el_costo_de_una_ejecucion.py` | el problema de asignación, un AG simple, y los dos contadores | La ejecución construye **428** individuos y llama a la aptitud **930** veces — **2.17 evaluaciones por individuo**, dividido en SELECCIÓN 300 / REPRODUCCIÓN 0 / ESTADÍSTICAS 630. **195,300** unidades de trabajo, mejor aptitud **−87** |
| 2 | `turnos_02_evaluar_una_vez.py` | **libro §13.1**: evaluar en el constructor, almacenar el valor | **428** llamadas, **89,880** unidades de trabajo: **54.0%** menos, con SELECCIÓN y ESTADÍSTICAS en **0**. La mejor aptitud sigue siendo **−87** — el ahorro no costó nada |
| 3 | `turnos_03_la_cache.py` | **§13.2**: un diccionario genoma→aptitud, más un escaneo de tasa de aciertos | **19** aciertos en 428 construcciones, **409** genomas distintos, 4.4% menos trabajo, respuesta sin cambios en **−87**. El escaneo muestra que la tasa de aciertos es propiedad de la *búsqueda*: **14.4%** con mutación p=0.10, **6.1%** a 0.25, **4.4%** a 0.50, **3.2%** a 0.75 |
| 4 | `turnos_04_instantanea.py` | **§13.4**: volcar y restaurar la población, y la caché | 30 asignaciones se restauran por **30 evaluaciones / 6,300 unidades de trabajo** a partir de genes solos, y por **0 / 0** cuando la caché se restaura primero. El archivo de caché es **14.0×** el tamaño del archivo de genes — un mal intercambio a 210 unidades por llamada, uno excelente a un minuto por llamada |
| 5 | `turnos_05_evaluacion_paralela.py` | **§13.5**: agrupar la generación, luego pasarla a un `Pool` | El solo agrupamiento reduce la ejecución de 409 a **291** evaluaciones (**−28.9%**), porque un hijo cruzado-luego-mutado solía evaluarse dos veces. La ejecución agrupada da la misma respuesta y el mismo mejor-hasta-ahora en las 10 generaciones — mientras que los contadores del padre leen **0 evaluaciones, 0 unidades de trabajo** |
| 6 | `turnos_06_cache_entre_procesos.py` | el padre es dueño de la caché y totaliza los costos de los trabajadores | La ejecución agrupada ahora reporta **291 / 8 aciertos / 61,110**, igualando a la ejecución secuencial en los tres, respuesta sigue en **−87**. Las cachés privadas por trabajador del paso 5 descartaban todos los aciertos |
| 7 | `turnos_07_aptitud_gruesa.py` | **§13.3**: eliminar la mitad cara del objetivo | **105** unidades de trabajo por llamada en lugar de 210, **51.0%** menos trabajo para la ejecución. La búsqueda gruesa llega a **0 en su propio objetivo** — y su asignación puntúa **−160** en el real frente a **−87**, con **32** violaciones de descanso en lugar de 17 |

**Estos números se verifican con salida real.** Cualquier diapositiva, folleto o traducción que cite una cifra debe citar uno de estos. La única cantidad deliberadamente ausente de esta tabla es un tiempo en segundos.

### El resultado que no fue planeado

El paso 5 fue escrito para introducir `multiprocessing`. Su mayor efecto medido no tiene nada que ver con eso: la reestructuración para que toda una generación pueda despacharse a la vez también detiene al programa de evaluar a los hijos intermedios que la mutación reemplaza inmediatamente, y eso por sí solo elimina 118 de 409 evaluaciones. El pool, en esta máquina, no cambió ningún conteo en lo absoluto. El paso ahora reporta primero el ahorro del agrupamiento y en segundo lugar el pool, en ese orden, porque ese es el orden en que la evidencia los pone.

### Dónde se encuentra esta lección con la Lección 02

El paso 1 de la Lección 02 cuenta **107** evaluaciones de aptitud en una ejecución donde 110 parecen obvias, y explica la discrepancia: un individuo que es seleccionado pero ni cruzado ni mutado es el mismo objeto y nunca se vuelve a evaluar. Ese recuento solo es cierto para un programa que ya ha aplicado la §13.1 — cosa que la Lección 02 hace sin nombrarlo. El paso 2 aquí es la técnica detrás de ello, y el paso 5 la amplía: un individuo que es cruzado *y* mutado tampoco necesita evaluarse dos veces.

### Si la sesión es corta

Siete pasos no caben en 32 minutos al ritmo que se enseña el resto del curso.
Corta el **paso 4 (instantáneas)**: es el más autocontenido de los siete, y nada en los pasos 5 al 7 hace uso de él. Sus cuatro funciones sí acompañan en esos scripts, porque cada script es el programa completo tal como estaba — así que si se corta el paso 4, di en una frase para qué están ahí `dump_population` y `dump_cache` en lugar de dejar que el estudiante se lo pregunte. Los pasos 5 y 6 deben enseñarse juntos o no enseñarse en absoluto: el paso 5 termina con un medidor roto que el paso 6 repara.

## Cómo se marca el código

Cada script a partir del segundo se abre con una **receta**: la lista ordenada de cambios que convierte el script anterior en este. Cada elemento de la receta tiene una **banda** en el cuerpo, en el sitio del cambio, con el mismo número y nombre:

```python
# --- NUEVO (1) aptitud_gruesa() -----------------------------------------------
```


## Ejecutándolo

Desde `MA2015 lessons/`:

```bash
uv run es/Leccion_13_Rendimiento/src/turnos_01_el_costo_de_una_ejecucion.py
```

Cada script es independiente, no toma argumentos, fija `SEMILLA = 3` y guarda su figura en `figures/` (o `figuras/`). Los siete juntos se ejecutan en **unos 11 segundos**.
Los pasos 5 al 7 inician un `multiprocessing.Pool` de tamaño `min(4, os.cpu_count())` e imprimen la cuenta de núcleos que encontraron, de modo que su salida en consola difiere por máquina en esa línea; cada resultado contado no lo hace.

## Contenido de la carpeta

| Ruta | Contenido |
|---|---|
| `src/` | los siete scripts numerados |
| `figures/` | un PNG por script, con el nombre del script |
| `snapshots/` | escrito por el paso 4 en tiempo de ejecución (`turnos_poblacion.json`, `turnos_cache.json`) |

## Estado

| Pieza | Estado |
|---|---|
| Código | Hecho. Siete scripts numerados, todos terminan limpio, todos pasan el verificador de bandas. |
| Figuras | Hecho. Una figura generada por script. |
| Espejo en español | Hecho. |

## Cuadernos guiados

Entrada para estudiantes: [`notebooks/README.md`](notebooks/README.md). Esta lección tiene 1 secuencia ejecutable de forma independiente: [`leccion_13_rendimiento.ipynb`](notebooks/leccion_13_rendimiento.ipynb). Las salidas retenidas se validaron el date omitted contra los scripts congelados en tres modos de ruta con kernels nuevos; el README de cuadernos registra el entorno y la brecha explícita de Colab alojado.

