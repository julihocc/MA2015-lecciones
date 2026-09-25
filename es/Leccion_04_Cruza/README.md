# Lección 04 — Cruza

Siete operadores de cruza, todos ellos aplicados a los mismos dos padres y
medidos con el mismo instrumento, de modo que la lección termina no con un desfile de
técnicas, sino con una sola tabla que muestra que los operadores se dividen en dos familias
que no pueden intercambiarse. Una cruza elegida para la representación equivocada no
produce una solución peor: produce algo que no es una solución.

- **Fuente en el libro:** Gridin, *Learning Genetic Algorithms with Python*,
  Capítulo 4 (*Crossover*) — 7 secciones, `2,748 palabras, 14 figuras, 157 líneas
  en 7 archivos.
- **Presupuesto:** 51 min (El Capítulo 4 es el 7.8% del libro).
- **Prerrequisitos:** Lección 01 (BLX-alpha y `acotar()` se introdujeron allí,
  y no se vuelven a derivar aquí) y Lección 03 (el método: medir un operador
  muchas veces desde una semilla fija y leer las columnas).

## Modelo matemático

Para padres fijos, el **alcance** es el soporte de la distribución de hijos. Las
cruzas de uno y (n) puntos copian segmentos parentales contiguos alternados;
esta implementación restringe los cortes de (n) puntos a `range(1, L-1)`, no
al conjunto canónico (1,\ldots,L-1). La cruza uniforme tiene \(2^L\) primeros hijos posibles si difieren todos los
pares parentales; la lineal alcanza sus dos hijos afines. BLX-\(\alpha\) muestrea de
\([m-\alpha d,M+\alpha d]\), \(d=M-m\), y sale del intervalo parental con
probabilidad \(2\alpha/(1+2\alpha)\). OX1 conserva pertenencia y unicidad solo
en permutaciones. Una muestra ilustra el soporte; la enumeración establece un
conteo exacto del conjunto alcanzable.

## Con qué se queda el estudiante

1. **La cruza se define por lo que puede alcanzar.** Dados dos padres fijos, el
   conjunto alcanzable de un operador es algo finito y contable. La cruza de
   un punto en un cromosoma de seis genes puede producir exactamente **10** hijos
   diferentes; la cruza uniforme puede producir **64**, que son todos ellos.
2. **"Más puntos de cruza" no es un dial de fuerza.** La cruza de 3 puntos alcanza
   **8** hijos aquí (menos que los 10 de un punto) porque n puntos de corte extraídos
   de las posiciones interiores admiten menos patrones de segmentos que un corte libre.
3. **Los operadores de corte y cambio copian; los operadores aritméticos calculan.**
   La familia de corte pone un valor que los padres no tenían en el **0.00%** de los
   espacios de genes. Mezcla pone uno en el **99.57%**, y en alpha = 0.5 también sale
   del espacio de búsqueda, por lo que solo el **50.08%** de sus hijos son legales
   hasta que `acotar()` los repara.
4. **La cruza guiada por aptitud compra seguridad con movimiento.** Nunca es peor que
   los padres (**0.00%** de las extracciones), y paga por eso devolviendo a los dos
   padres sin cambios en el **16.50%** de las extracciones.
5. **Las dos familias no se superponen.** En una ruta de nueve paradas, un punto,
   n puntos, mezcla y lineal producen un hijo legal el **0.00%** del tiempo; uniforme
   logra el **0.95%**, y cada uno de esos es un padre devuelto directamente.
   La cruza ordenada es legal el 100% del tiempo en las rutas, y en los genes con valores
   reales devuelve silenciosamente a los padres. **Cero** operadores son útiles en ambas.

## El ejemplo en ejecución

Un par fijo de padres, y la nube de hijos que un operador puede crear a partir
de ellos. Esa nube es lo que nombra el prefijo `descendencia_`, y una función
(`reporte_descendencia`) la mide para cada operador en más de **2,000 extracciones** desde una
semilla fija, para que las filas de cada tabla sean comparables por construcción.

Los padres son los mismos de Gridin: `SEMILLA = 3`, seis genes extraídos uniformemente de
`[0, 10]` — `[2.38 5.44 3.70 6.04 6.26 0.66]` y `[0.13 8.37 2.59 2.34 9.96
4.70]`. Desde el paso 6, la misma semilla produce dos rutas a través de nueve paradas de
entrega, que es la otra representación del capítulo y el tamaño que usa Gridin en
`order.py`.

El instrumento informa cuatro números por operador: a cuántos **hijos diferentes**
alcanzó, la proporción de espacios de genes que tienen un valor **nuevo**, la proporción de
hijos que son un **clon** de un padre, y la proporción que son **legales**. Los
primeros tres describen al operador; el último describe el matrimonio entre
el operador y la representación, y ahí es donde aterriza la lección.

## Los siete pasos

| # | Script | Lo que agrega | Lo que prueba su salida |
|---|---|---|---|
| 1 | `descendencia_01_dos_padres.py` | los padres, el instrumento y la base de no hacer nada | La fila cero: 2 hijos alcanzados, 0.00% genes nuevos, 100.00% clones, 100.00% legales. También el número contra el que se mide cada paso posterior: elegir libremente entre los padres permite 2⁶ = **64** cromosomas |
| 2 | `descendencia_02_un_punto.py` | `cruza_un_punto()`, y la enumeración de todo su alcance | 2,000 extracciones producen **10** hijos distintos; enumerar los 5 puntos de corte produce los mismos 10. **0.00%** genes nuevos: el operador elige qué padre, nunca qué valor |
| 3 | `descendencia_03_mas_cortes.py` | `cruza_n_puntos()`, `cruza_uniforme()` | El alcance no es monótono en los cortes: un punto **10**, 2 puntos **12**, 3 puntos **8**, uniforme **64 de 64**. El precio de la uniforme: **3.20%** de sus hijos son un padre devuelto |
| 4 | `descendencia_04_mezcla.py` | `acotar()`, `cruza_mezcla()`, `cruza_lineal()` | Los primeros genes calculados (**99.57%** nuevos) y los primeros hijos ilegales. alpha = 0: **0.00%** de genes más allá de los padres, 100% legal — El colapso interno de la Lección 01, medido. alpha = 0.5: **49.63%** más allá, solo **50.08%** legal; con `acotar()`, 100% legal y aún 49.63% más allá. Lineal: 100.00% genes nuevos y exactamente **2** hijos alcanzables |
| 5 | `descendencia_05_guiada_por_aptitud.py` | `objetivo()`, `cruza_guiada_por_aptitud()` | La mezcla ciega es peor que sus padres en **41.70%** de las extracciones — la cruza no garantiza nada. La regla guiada por aptitud es peor en **0.00%**, y paga por ello: **16.50%** de las extracciones devuelven a los padres sin cambios, y solo el **56.10%** de lo que devuelve es un hijo |
| 6 | `descendencia_06_la_division.py` | las nueve paradas, `longitud_recorrido()`, `es_recorrido_legal()` | Los mismos operadores, una ruta en lugar de un punto. **8 de 8** puntos de corte dan una ruta que visita una parada dos veces y otra ninguna. Hijos legales: un punto **0.00%**, 3 puntos **0.00%**, mezcla **0.00%**, lineal **0.00%**, uniforme **0.95%** — y el recuento de hijos legales de la uniforme que no eran simplemente un padre es **0** |
| 7 | `descendencia_07_ordenada.py` | `cruza_ordenada()`, y la comparación para la que se construyó la lección | OX1 es legal el **100.00%** en las rutas y mantiene **7.17 de 9** tramos de los padres, frente a **3.71** de una ruta aleatoria. Al recibir los padres con valores reales no falla: alcanza **2** hijos, el **100.00%** de ellos un padre. Tabla final: operadores útiles en ambas representaciones — **0** |

**Estos números están verificados con resultados reales.** Cualquier diapositiva, folleto o
traducción que indique una cifra debe indicar una de estas. Vuelve a ejecutar el script
en lugar de confiar en esta tabla si el código ha cambiado.

## Dos resultados que no estaban planeados

- **El paso 3 se diseñó para mostrar que más puntos de corte significan más mezcla.** No lo
  hacen: 3 puntos alcanza a 8 hijos donde un punto alcanza a 10, porque el
  `n_puntos` de Gridin toma muestras de sus cortes de `range(1, len - 1)` y tres cortes de cuatro
  posiciones interiores solo admiten cuatro patrones de segmentos. El paso se reescribió
  en torno a la medición. El recuento depende de la longitud y el script lo dice;
  lo que se generaliza es que "el número de cortes" no es la cantidad que importa,
  y que la cruza uniforme es el único miembro de la familia que alcanza el
  conjunto completo.
- **El paso 7 se diseñó para terminar con el funcionamiento de la cruza ordenada.** Termina con
  la cruza ordenada *fallando silenciosamente* en cambio. Entregado a dos padres que no comparten
  valores de genes, el bucle de llenado de OX1 nunca omite nada y sobrescribe su propia
  franja, por lo que devuelve a los padres sin cambios: legales, plausibles y completamente
  inertes. Ese es el final más útil: la ruta ilegal del paso 6 es ruidosa y
  se encuentra, y esta no.

  Dos resultados honestos más pequeños se sientan en el mismo paso. Los hijos de OX1 promedian una ruta
  de **52.95** frente a **51.23** de 2,000 rutas aleatorias: en nueve paradas, y con
  dos padres mediocres, la herencia de la estructura es visible (7.17 tramos frente a 3.71) mientras
  que la herencia de la calidad no lo es. La cruza proporciona lo primero; solo la selección
  proporciona lo segundo.

## Cómo está marcado el código

Dos dispositivos, obligatorios en todo el curso:

- **La receta** — cada script después del primero se abre con un bloque `CAMBIOS RESPECTO A …`
  que enumera, en orden, los cambios que convierten el script anterior en este.
  Ese orden es el orden en que se debe escribir en clase y el orden que siguen las diapositivas.
- **Las bandas** — cada elemento de la receta tiene su propia banda `# --- NUEVO (n) nombre ---`
  en el sitio del cambio, numerada y nombrada para que coincida. Todo lo que está fuera de una banda
  es código que los estudiantes ya tienen.

Verificado mecánicamente:

```bash
```

## Ejecutándolo

```bash
uv run es/Leccion_04_Cruza/src/descendencia_01_dos_padres.py
```

Cada script es independiente, no toma argumentos y guarda su figura en
`../figures/` con su propio nombre. Toda la secuencia se ejecuta en aproximadamente **26 s**.

## Ajuste al presupuesto

Siete pasos en 51 minutos es ajustado (la Lección 03 ajusta seis en 48). Los pasos 2 y 3
son el par a fusionar si la sesión se queda corta: comparten una idea (la
familia de cortar y cambiar alcanza un conjunto contable y no calcula nada), y la
enumeración del paso 2 se puede mostrar desde la tabla del paso 3 en lugar de por sí sola. Los pasos 6 y 7
no deben cortarse ni dividirse entre sesiones; la división solo enseña si ambas mitades
aterrizan en la misma hora.

Las 14 figuras del libro no se reproducen. Siete son suficientes: una por script,
cada una mostrando la propia medición de ese script.

## Contenido de la carpeta

```
Leccion_04_Cruza/
├── README.md      este archivo — el contrato de la lección
├── src/           los siete scripts numerados
├── figures/       una figura por script
```

## Paquete integral activo — W08, 25/09/2026

Cinco componentes: 7 scripts en src/, presentación y libreta extensas en estudiante/, presentación y libreta compactas en docente/. 8 figuras en figures/. Los dos PDF compilaron dos veces; ambas libretas y los scripts ejecutaron sin errores. El compacto tiene nueve láminas y comentarios en cada celda de código.

Ruta de 40 minutos por lección. En S04: L04 recibe 51 minutos, L05 recibe 35 y las transiciones y cierre reciben 14. Ejemplos sintéticos separados del modelo de Planta Física. Datos externos y ZIP: no aplica. Colab alojado, accesibilidad integral y distribución estudiantil: no comprobados.

## Cuadernos guiados

La libreta extensa activa está en [`estudiante/`](estudiante/). Los cuadernos previos por hitos permanecen en [`notebooks/`](notebooks/) como material complementario.
