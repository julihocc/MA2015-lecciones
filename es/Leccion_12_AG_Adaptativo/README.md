# Lección 12 — El Algoritmo Genético Adaptativo

Hasta ahora, todas las lecciones han escrito los parámetros en la parte superior del archivo como constantes, y la Lección 07 afinó esas constantes mediante búsqueda en cuadrícula. Esta lección pregunta si deberían ser constantes en absoluto: construye un sensor que le dice a una ejecución si todavía está mejorando, conecta ese sensor a las probabilidades de cruza y mutación y al tamaño de la población, y luego mide — doce ejecuciones emparejadas con el mismo presupuesto de evaluaciones — si vale la pena tener el mecanismo. Contra la configuración no afinada que heredó, gana todas las ejecuciones; contra cuatro configuraciones fijas exploradas con las mismas semillas, **queda por detrás en la media en tres de las cuatro y el intervalo aproximado de dos errores estándar contra la mejor incluye cero**. Ese resultado descriptivo es la lección.

- **Fuente en el libro:** Gridin, *Learning Genetic Algorithms with Python*,
  Capítulo 12 (*Adaptive Genetic Algorithm*) — 6 secciones, ~2,299 palabras, 8
  figuras, 424 líneas en 6 archivos (`03-referencias/…/Chapter12/`).
- **Lo que la refactorización le hace:** el libro demuestra el mecanismo en una semilla y luego compara adaptativo contra "clásico" a lo largo de 100 ejecuciones en las cuales los dos algoritmos **no** gastan el mismo número de evaluaciones. La refactorización mantiene el mecanismo exactamente (el mismo sensor, los mismos multiplicadores 1.1 / 0.99, la misma regla de población de inmigrantes y descartes) y reemplaza la comparación con una emparejada de presupuesto igual en la moneda que estableció la Lección 07. Luego agrega el paso que el libro no tiene: el algoritmo adaptativo contra un algoritmo fijo *afinado*.
- **Duración objetivo:** 43 min por peso del libro (el Capítulo 12 es el 6.6% del texto).
  Seis pasos: 1–4 son una ejecución cada uno y van rápido; los pasos 5 y 6 son la lección.
- **Prerrequisitos:** Lección 10 (el problema del agente viajero, cruza ordenada, mutación por inversión — se reafirman aquí, no se importan), Lección 07 (afinación, y evaluaciones como la moneda de comparación), Lección 06 (por qué una sola ejecución no prueba nada).

## Modelo matemático

Sea \(m_t\) la longitud media de la población y \(\bar m\) la media de las diez
generaciones anteriores. El sensor declara mejora cuando
\(m_t<(1-0.001)\bar m\). Las probabilidades se multiplican por 0.99 durante la
mejora y por 1.1 durante el estancamiento, y luego se recortan; redimensionar
elimina un individuo sin evaluar o agrega inmigrantes evaluados. Las
comparaciones usan diferencias emparejadas. El intervalo de dos errores estándar
es aproximado, no una prueba de significancia, y la mejor configuración fija es
la mejor de cuatro en estas mismas doce semillas.

## Lo que el estudiante se lleva

1. Un parámetro fijo es un compromiso entre dos fases de una ejecución, y el compromiso puede medirse: la mitad del presupuesto de la ejecución del paso 1 compra el 85% de su mejora total, y la configuración que compró esa mitad sigue vigente para la segunda.
2. Una ejecución puede sentir su propio progreso sin conocer el óptimo — comparar la media de la población contra su propio promedio móvil — y esa señal cruda es el único sensor que tiene el algoritmo adaptativo.
3. Una regla adaptativa debe ser *auditada*, no asumida: en el paso 3, la rama de "aumentar las probabilidades cuando se estanca" se dispara **0 veces en 99 generaciones**. Lo que se anuncia como adaptación se ejecuta como un programa de decaimiento unidireccional.
4. La adaptación supera a una configuración no afinada de manera decisiva (−8,050 de longitud de ruta sobre 12 ejecuciones emparejadas, 12 victorias de 12) y **no** supera a una afinada (+1,274 ± 727 contra la mejor de cuatro configuraciones fijas, 4 victorias de 12).
5. Por lo tanto, un esquema adaptativo es un seguro barato contra una constante mal elegida, no un reemplazo para la Lección 07.

## El ejemplo de ejecución

La instancia del problema del agente viajero de la Lección 10: las 48 capitales estatales de EE. UU. de `att48_xy.txt`, la misma cruza ordenada, la misma mutación por inversión, la misma selección por torneo y elitismo. Esta es la elección propia del libro — el capítulo 12 es el único lugar donde Gridin importa el problema de otro capítulo — y se mantiene porque la comparación es entonces contra un problema que los estudiantes ya han visto buscar, con una línea base del vecino más cercano (**40,526**) que ya conocen.
La regla de no importar sigue vigente: todo se reafirma dentro de cada script y el archivo de datos se encuentra en el propio `src/` de esta lección.

Dos cosas cambian respecto a la Lección 10. La probabilidad de cruza se convierte en una perilla explícita (la Lección 10 siempre cruzaba), y **la regla de parada es un presupuesto de evaluaciones de 12,000, no un conteo de generaciones** — el algoritmo adaptativo redimensiona su propia población, por lo que las generaciones dejan de ser unidades comparables, mientras que las evaluaciones siguen siendo comparables. Cada tabla imprime las evaluaciones realmente gastadas para que el lector pueda comprobar que los presupuestos son realmente iguales.

## Los seis pasos

| # | Script | Qué añade | Qué prueba su salida |
|---|---|---|---|
| 1 | `ag_adaptativo_01_parametros_fijos.py` | el programa completo: la búsqueda de la Lección 10 bajo un presupuesto de 12,000 evaluaciones, cruza 0.90 / mutación 0.25, población 120 | La semilla 1 gasta 11,901 evaluaciones en 99 generaciones y devuelve una ruta legal de **57,076**, un 40.8% más larga que la línea base del vecino más cercano de 40,526 |
| 2 | `ag_adaptativo_02_el_estancamiento.py` | `promedio()`, `esta_mejorando()`, la tendencia de estancamiento, `sombrear_estancamientos()` | La señal llama a **14 de 99** generaciones estancadas, la primera en la generación 74, y el 14% del presupuesto se gasta dentro de ellas; la primera mitad del presupuesto compra el **85%** de la mejora total de la ejecución, la segunda mitad el 15% |
| 3 | `ag_adaptativo_03_probabilidades_adaptativas.py` | `adaptar_probabilidades()`, y un `ejecutar()` que puede usarlo | Misma semilla, mismo presupuesto: adaptativo **48,830** contra fijo **57,076** (−14.4%). Pero la rama estancada se dispara **0 de 99** veces — la cruza baja 0.90 → 0.33 y la mutación 0.25 → 0.09 y nunca vuelven a subir. La regla se comporta como un programa de decaimiento, no como adaptación |
| 4 | `ag_adaptativo_04_poblacion_adaptativa.py` | `redimensionar_poblacion()`, y un segundo interruptor en `ejecutar()` | Solo redimensionamiento: 35 estancamientos, 70 inmigrantes, 103 descartes, población 68–120, **138** generaciones para el mismo presupuesto y **49,374**. Con las probabilidades también: 41 estancamientos, población reducida a 40, 155 generaciones, cruza en el rango de 0.39–1.00 — el redimensionamiento trae de vuelta los estancamientos que el paso 3 había hecho desaparecer — y **52,295**, peor en esta semilla que el redimensionamiento solo |
| 5 | `ag_adaptativo_05_doce_ejecuciones.py` | `EJECUCIONES`, `medir()`, `resumen_emparejado()`, la tabla y el diagrama de caja | Sobre 12 ejecuciones emparejadas (ejecución *i* en semilla *i*): fijo **58,195**, probabilidades **50,145** (−8,050 ± 1,022, victorias 12/12), redimensionar **53,555** (−4,640 ± 926, 10/12), ambos **52,622** (−5,573 ± 1,127, 11/12). Cada régimen gasta 11,901–11,946 evaluaciones, así que los presupuestos son iguales. El intervalo aproximado diferencia-media ± dos EE excluye cero en las tres comparaciones con *esta* configuración fija; es descriptivo, no una prueba de significancia |
| 6 | `ag_adaptativo_06_un_rival_afinado.py` | `ejecutar()` toma sus tasas iniciales como argumentos, `CONFIGURACIONES_FIJAS`, el veredicto | Contra cuatro configuraciones fijas con el mismo presupuesto: 0.90/0.25 → 58,195 (adaptativo gana 12/12), **0.60/0.05 → 48,870** (adaptativo gana 4/12, **+1,274 ± 727**), 0.40/0.15 → 49,675 (6/12, +469 ± 480), 0.20/0.30 → 49,626 (5/12, +519 ± 835). El régimen adaptativo queda por detrás de 3 de las 4 en la media; su intervalo aproximado de dos EE contra la mejor incluye cero. La mejor fija significa la mejor entre estas cuatro en las mismas 12 semillas, no una afinación validada de forma independiente |

**Estos números están verificados contra la salida real.** Cualquier diapositiva, folleto o traducción que indique una cifra debe indicar una de estas, no un sustituto plausible. Vuelva a ejecutar el script en lugar de confiar en esta tabla si el código ha cambiado desde entonces.

### El resultado en torno al cual no se planeó la lección


## Cómo está marcado el código

Dos dispositivos, ambos obligatorios para todo el curso:

- **La receta** — cada script después del primero abre con un bloque `CAMBIOS RESPECTO A …`: la lista ordenada de cambios que convierten el script anterior en este. Ese orden es el orden para escribirlos en clase.
- **Las bandas** — cada elemento de la receta tiene su propia banda `# --- NUEVO (n) nombre ---` en el sitio del cambio, numerada y nombrada para coincidir con la receta. Las bandas no se trasladan: el paso 4 muestra solo los tres cambios del paso 4. Donde un cambio tiene un segundo sitio (las dos variables de probabilidad leídas dentro del ciclo de generaciones en el paso 3, las tasas viajando a través de `medir()` en el paso 6), el segundo sitio lleva un comentario de puntero simple `# (n)` en lugar de una segunda banda.

La ejecución *i* usa la semilla *i* durante los pasos 5 y 6, por lo que cada régimen y cada configuración fija enfrenta a las mismas doce poblaciones iniciales y todas las comparaciones son emparejadas. Los pasos 1–4 usan la semilla 1, que es también la ejecución 1 de las tablas.

## Ejecutándolo

Desde la raíz del repositorio, con `uv` manejando el entorno:

```bash
uv run es/Leccion_12_AG_Adaptativo/src/ag_adaptativo_01_parametros_fijos.py
```

Cada script es autónomo: sin importaciones entre pasos, sin importaciones entre lecciones, sin argumentos de línea de comandos. `att48_xy.txt` vive junto a los scripts y es leído por los seis. Las figuras se **guardan** en `../figures/`, nunca se muestran.

**Tiempo de ejecución: alrededor de 42 s para toda la secuencia** — los pasos 1–4 toman menos de 2.5 s cada uno, los pasos 5 y 6 toman alrededor de 16 s cada uno porque son 48 y 60 ejecuciones del AG. Lo que se recortó para llegar allí: el presupuesto es de 12,000 evaluaciones en lugar de la regla de ejecutar hasta que se detenga del libro, el estudio es de 12 ejecuciones
en lugar de las 100 del libro, y el barrido en el paso 6 es de cuatro configuraciones fijas
en lugar de una cuadrícula completa. Si la secuencia todavía no cabe en un espacio, reduzca `EJECUCIONES` en los pasos 5 y 6 de 12 a 8 — pero diga en voz alta que los errores estándar crecen con eso, porque a 8 ejecuciones la comparación del paso 6 no vale nada.

Dos cosas más que tiene el libro y esta lección no: sus operadores impulsados por la aptitud (una cruza que conserva los dos mejores de cuatro candidatos, una mutación que reintenta hasta tres veces) y su regla de parada (ejecutar hasta que la mejor aptitud deje de mejorar). Ambas se eliminaron a propósito. Los operadores esconden evaluaciones adicionales dentro de sí mismos, lo que hace imposible plantear honestamente una comparación de presupuesto igual, y la regla de parada les da a los dos algoritmos presupuestos diferentes — lo cual es exactamente la falla en la que la Lección 07 invirtió su tiempo.

## Contenido de la carpeta

```
Leccion_12_AG_Adaptativo/
├── README.md      este archivo — el contrato de la lección
├── src/           los seis scripts numerados y att48_xy.txt
├── figures/       gráficos generados (nunca editados a mano, nunca confirmados a mano)
```

## Estado

| Pieza | Estado |
|---|---|
| Código | Hecho. Seis scripts, todos se ejecutan limpios, todas las afirmaciones se calculan a partir de su propia salida. |
| Figuras | Hecho. Una figura por script, nombrada según el script que la produce. |
| Diapositivas | No iniciado, y bloqueado por la plantilla de Beamer del curso. |

## Cuadernos guiados

Entrada para estudiantes: [`notebooks/README.md`](notebooks/README.md). Esta lección tiene 1 secuencia ejecutable de forma independiente: [`leccion_12_ag_adaptativo.ipynb`](notebooks/leccion_12_ag_adaptativo.ipynb). Las salidas retenidas se validaron el date omitted contra los scripts congelados en tres modos de ruta con kernels nuevos; el README de cuadernos registra el entorno y la brecha explícita de Colab alojado.

