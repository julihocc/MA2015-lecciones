# Lección 05 — Mutación

La mutación es el único operador que puede producir un valor que la población nunca tuvo,
por lo que la única pregunta que vale la pena hacer al respecto es qué tan lejos puede llegar una aplicación.
Esta lección construye un instrumento que mide el alcance, lo aplica a los siete
operadores de mutación de Gridin en dos codificaciones y lo lleva de vuelta a la ejecución
que la Lección 02 dio por muerta, donde descubre que la mutación nunca fue lo que
faltaba.

- **Fuente en el libro:** Gridin, *Learning Genetic Algorithms with Python*,
  Capítulo 5 (*Mutation*) — 7 secciones, `1,895 palabras, 7 figuras, 113 líneas en
  7 archivos (`bit_flip.py`, `exchange.py`, `inversion.py`, `shift.py`,
  `shuffle.py`, `random_deviation.py`, `fitness_driven.py`). Un archivo por
  operador y ninguna medición de ninguno de ellos.
- **Presupuesto:** 35 min (el Capítulo 5 es el 5.4% del libro — el más ligero de los cuatro
  capítulos de operadores). Seis pasos, y la secuencia encaja: todo se ejecuta en
  aproximadamente **10 s**.
- **Prerrequisitos:** Lecciones 01 y 02. El paso 1 del primer ejemplo reproduce
  el paso 5 de la Lección 01 al dígito, y el paso 4 reproduce el paso 5 de la Lección 02 al
  dígito; ambas identidades son impresas por los propios scripts.

## Modelo matemático

En un cromosoma real de longitud \(L\), cada gen se activa con
\(B_i\sim\operatorname{Bernoulli}(p)\) y recibe
\(\Delta_i\sim\mathcal N(\mu,\sigma^2)\), seguido aquí por una proyección al
intervalo legal. Se esperan \(Lp\) genes activados; \(\sigma\) controla el
desplazamiento. Las mutaciones de permutación conservan pertenencia; se miden
con distancia de Hamming (H(x,x')=\sum_i I[x_i\ne x_i']) y recorrido total
(D(\pi,\pi')=\sum_v|\operatorname{pos}_\pi(v)-\operatorname{pos}_{\pi'}(v)|).
Un escalón superior de ancho cero
tiene probabilidad cero bajo una propuesta continua ideal.

## Lo que el estudiante se lleva

1. **La mutación tiene una idea — el alcance — y dos diales que lo ajustan.** `sigma` dice
   qué tan lejos se mueve un gen tocado; `p` dice cuántos genes son tocados. Su
   producto es el presupuesto de movimiento; `p` por sí solo decide si se gasta como un
   salto largo o como diez saltos cortos, y solo el salto largo puede cruzar un valle.
2. **El alcance no se puede comprar simplemente.** Más grande es una dirección, no una mejora:
   el sigma pequeño encuentra la mayoría de los movimientos cuesta arriba y nunca llega a ninguna parte, el
   sigma grande llega y no puede quedarse.
3. **La mutación guiada por aptitud (fitness-driven) gana por mutación y pierde por evaluación de aptitud.**
   Reintentar hasta mejorar no añade ninguna habilidad nueva — compra más boletos en la
   misma lotería, a tres veces el precio.
4. **La codificación decide qué mutaciones existen.** En una permutación, agregar
   ruido no es una mutación débil, es una mutación inválida; los únicos movimientos legales son
   reordenamientos, y su alcance tiene que ser medido en dos números, no en uno.
5. **Antes de afinar un operador, comprueba que el objetivo exista.** La escalera de la Lección 02
   no pudo ser resuelta por ningún régimen de mutación porque su escalón superior tenía
   ancho cero. Una ejecución no puede distinguir una búsqueda muerta de una imposible.

## Los ejemplos en ejecución

Dos, porque los operadores del Capítulo 5 se dividen limpiamente en dos y forzarlos en una
sola secuencia ocultaría la división que importa.

- **`alcance_mutacion_`** — un gen de valor real en el paisaje de la Lección 01
  `f(x) = sin(x) − 0.2·|x|`, y más tarde la escalera de la Lección 02. Cuatro pasos.
- **`mutacion_permutacion_`** — un cromosoma que es un *orden* de diez símbolos,
  donde un gen no tiene un valor para perturbar. Dos pasos.

Una idea de medición recorre ambos: disparar el operador 2,000 veces a un
individuo fijo desde una semilla fija y reportar la distribución, no un ejemplo. El
libro imprime un par de antes y después por operador; eso es lo que esta lección
reemplaza.

La mitad de la permutación son dos pasos en lugar de cinco porque los cuatro
operadores de permutación del capítulo son 40 líneas entre ellos y rellenarlos costaría el
presupuesto que necesita la primera mitad.

## Los pasos

### Primer ejemplo — `alcance_mutacion_` (genes de valor real)

| # | Script | Lo que añade | Lo que prueba su salida |
|---|---|---|---|
| 1 | `alcance_mutacion_01_el_instrumento.py` | `reporte_alcance()`, la distribución de desplazamiento | Los dos números de la Lección 01 reaparecen exactamente — sigma 1.0 llega **0 de 2000**, sigma 3.0 **110 de 2000**. El viaje promedio es 0.798·sigma hasta que los muros empiezan a comerlo: en sigma 6.0, el **19.4%** de las propuestas son acotadas y la proporción baja a 0.690. El intercambio en una tabla: sigma 0.5 encuentra la mayoría de los movimientos cuesta arriba (**272**) y llega **0** veces |
| 2 | `alcance_mutacion_02_tasa_y_paso.py` | `CANTIDAD_GENES`, `mutar_desviacion_aleatoria()`, `reporte_regimen()` | Los genes tocados nunca difieren de `p·n` por más de **0.015**. Tres regímenes con el mismo `p·sigma = 0.30` viajan 1.904 / 2.179 / 2.304 en total pero su salto individual más largo difiere por un factor de **7.1** (10.253 vs 1.449). En p = 0.1, **718 de 2000** mutaciones no cambian nada en absoluto, contra un predicho (1−p)¹⁰ = 0.3487 |
| 3 | `alcance_mutacion_03_guiada_por_aptitud.py` | `MAX_INTENTOS`, `mutar_guiada_por_aptitud()`, `comparar()` | El filtro convierte un cambio de aptitud promedio de **−1.0337 en +0.0548** y duplica las llegadas (102 → 273) — pero gasta **5,603 evaluaciones de aptitud contra 2,000**, así que por mil llamadas llega **48.7** veces contra **51.0** de la mutación ciega. La aceptación del 20.2% es exactamente 1−(1−q)³ para la tasa ciega hacia arriba q |
| 4 | `alcance_mutacion_04_la_escalera_muerta.py` | `ejecutar()`, `escalera()`, `anchos_escalones()`, `ALTURA_PICO_REPARADA`, `barrido()` | La ejecución de la Lección 02 reproducida (6 generaciones, mejor **+1.8000**, dispersión 0.192, **0.2000** por debajo). Luego ocho regímenes de mutación × 100 ejecuciones: el escalón superior se alcanza **0 veces en 800 ejecuciones**, mientras que la dispersión promedio de los genes va de 0.462 a 3.194. `anchos_escalones()` dice por qué — el escalón superior es **0.000 de ancho**. Reparada (altura del pico 10.0 → 10.5, escalón superior 0.500 de ancho), el propio régimen de la Lección 02 lo alcanza **96 de 100** veces |

### Segundo ejemplo — `mutacion_permutacion_` (un orden, no un vector)

| # | Script | Lo que añade | Lo que prueba su salida |
|---|---|---|---|
| 1 | `mutacion_permutacion_01_ruido_ilegal.py` | `es_permutacion()`, `viaje()`, `mutacion_inversion_bit()`, `mutacion_intercambio()`, `reporte_reordenamiento()` | La desviación aleatoria en una permutación produce **1935 mutantes inválidos de 2000**, y **0** de los 65 sobrevivientes tuvo algún gen tocado — la única salida legal del operador es la intacta, a una tasa medida que coincide con (1−p)¹⁰ a menos de **0.0025**. La inversión de bit cambia exactamente 1 posición cada vez; el intercambio exactamente 2 |
| 2 | `mutacion_permutacion_02_reordenamientos.py` | `mutacion_inversion()`, `mutacion_desplazamiento()`, `mutacion_barajar()`, `comparar_operadores()` | Las dos medidas de alcance clasifican a los operadores de manera diferente: por posiciones perturbadas **desplazamiento (4.663) > inversión (4.230) > barajar (3.708) > intercambio (2.000)**, por distancia viajada **inversion (1.308) > barajar (0.876) > intercambio (0.733) = desplazamiento (0.733)**. El intercambio y el desplazamiento dan idéntico viaje total en **90 de 90** pares de posiciones. Barajar devuelve el individuo sin cambios **271 de 2000** veces (13.6%) |

**Estos números están verificados contra la salida real.** Cualquier diapositiva, apunte o
traducción que exponga una cifra debe exponer una de estas. Vuelve a ejecutar el script
en lugar de confiar en esta tabla si el código ha cambiado.

**Orden de enseñanza sugerido:** `alcance_mutacion_` 01→04, luego
`mutacion_permutacion_` 01→02. El último paso del segundo ejemplo cierra ambos.

## Un hallazgo que vale la pena conservar — y un defecto que expone en la Lección 02

El paso 4 fue diseñado para redimir una promesa. El paso 5 de la Lección 02 terminó con una ejecución que
murió en una escalera, 0.2000 por debajo del óptimo y sin dispersión genética restante, y
dijo que la mutación "es el único operador que podría restaurarlo". El plan era mostrar
un régimen de mutación diferente reviviéndolo.

No lo hace. Ocho regímenes × 100 ejecuciones cada uno alcanzan el escalón superior **cero veces**,
mientras multiplican la dispersión genética sobreviviente por 6.9. Un rechazo tan rotundo no
se trata de afinación, así que el paso hace la otra pregunta — y descubre que la escalera de la Lección 02,

```
floor((10.0 − 2·|x − 4|) / 1.0) · 0.2
```

tiene un **escalón superior de ancho cero**. La tienda alcanza su punto máximo exactamente en 10.0, por lo que la banda de `floor()`
más alta es el punto único `x = 4.0`. La comprobación por fuerza bruta en la Lección 02
se evaluó en `numpy.linspace(-10, 10, 20001)`, cuyo espaciado es 0.001 y el cual
por lo tanto incluye `x = 4.0` exactamente; esa es la única razón por la que el óptimo se
reportó como +2.0000. Un gen extraído de una distribución continua aterriza allí con
probabilidad cero. **El déficit publicado de 0.2000 de la Lección 02 es inalcanzable por
construcción, y ningún operador podría haberlo cerrado jamás.**

Levantar la tienda medio escalón (`ALTURA_PICO_REPARADA = 10.5`) le da al
escalón superior un ancho de 0.5 sin cambiar el número de escalones, la elevación, la pendiente o
el hecho de que el pico es interior. En ese paisaje, el propio régimen de la Lección 02 —
p = 0.1, sigma = 1.0, el que llamó insuficiente — alcanza la cima en **96 de
100 ejecuciones**.

El paso enseña eso en su lugar, lo cual es más de lo que la versión planeada habría hecho.

## Cómo está marcado el código

Dos dispositivos, obligatorios a nivel de curso:

- **La receta** — cada script después del primero de su prefijo comienza con un
  bloque `CAMBIOS RESPECTO A …` listando, en orden, los cambios que convierten el script
  anterior en este. Ese orden es el orden para teclear en clase y el orden
  que siguen las diapositivas.
- **Las bandas** — cada elemento de la receta tiene su propia banda `# --- NUEVO (n) nombre ---` en
  el sitio del cambio, numerada y nombrada para coincidir. Todo lo que está fuera de una banda
  es código que los estudiantes ya tienen.
  `alcance_mutacion_02_tasa_y_paso.py` cambia una línea en lugar de añadir código
  para su segundo elemento, por lo que marca la línea:
  `self.aptitud = …   # --- CAMBIADO --- un vector se califica sumando sobre sus genes`.

es/Leccion_05_Mutacion/src` — cada línea debe decir `OK`, excepto los dos
scripts de primer prefijo, que no tienen receta y dicen `---`.

## Ejecutándolo

```bash
uv run es/Leccion_05_Mutacion/src/alcance_mutacion_01_el_instrumento.py
```

Cada script es autónomo, no toma argumentos y guarda su figura en
`../figures/` bajo su propio nombre. Los seis scripts se ejecutan en aproximadamente **10 s** en total
(1.3 – 2.2 s cada uno); 2,000 extracciones por medición y 100 ejecuciones por régimen son lo que
compra las tasas estables.

## Contenido de la carpeta

```
Leccion_05_Mutacion/
├── README.md      este archivo — el contrato de la lección
├── src/           los seis scripts numerados, dos prefijos
├── figures/       una figura por script
```

## Paquete integral activo — W08, 25/09/2026

Cinco componentes: 6 scripts en src/, presentación y libreta extensas en estudiante/, presentación y libreta compactas en docente/. 6 figuras en figures/. Los dos PDF compilaron dos veces; ambas libretas y los scripts ejecutaron sin errores. El compacto tiene nueve láminas y comentarios en cada celda de código.

Ruta de 40 minutos por lección. En S04: L04 recibe 51 minutos, L05 recibe 35 y las transiciones y cierre reciben 14. Ejemplos sintéticos separados del modelo de Planta Física. Datos externos y ZIP: no aplica. Colab alojado, accesibilidad integral y distribución estudiantil: no comprobados.

## Cuadernos guiados

La libreta extensa activa está en [`estudiante/`](estudiante/). Los cuadernos previos por hitos permanecen en [`notebooks/`](notebooks/) como material complementario.
