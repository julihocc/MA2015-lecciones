# Lección 01 — Tu primer algoritmo genético

Construye un algoritmo genético en siete incrementos autónomos y compara dos corridas con semilla. Se supone Python básico: variables, condicionales, ciclos y funciones. Objetos, arreglos, comprensiones, emparejamiento y gráficas se explican cuando aparecen.

## Materiales

- [Presentación extensa — estudio](estudiante/slides/leccion_01.pdf), 38 páginas; [fuente LaTeX](estudiante/slides/leccion_01.tex).
- [Notebook extenso — estudio](estudiante/notebooks/leccion_01_intro.ipynb): fundamentos completos, código incremental, resultados, ejercicios y soluciones desplegables.
- [Presentación compacta — exposición docente](docente/slides/leccion_01_compacta.pdf), 12 láminas; [fuente LaTeX](docente/slides/leccion_01_compacta.tex). Explica la teoría con un ejemplo cuadrático independiente.
- [Notebook compacto — exposición docente](docente/notebooks/leccion_01_intro_compacta.ipynb): introduce por sí mismo el caso de seno penalizado, sus siete pasos, resultados y comprobaciones.
- [Siete scripts autónomos](src/) y [figuras generadas](figuras/).
- [Paquete estudiantil](leccion_01_paquete_estudiantil.zip): PDF/notebook extensos, scripts, figuras, instrucciones y dependencias bloqueadas del curso. Los compactos docentes se conservan por separado.
- [Instrucciones de ejecución del notebook estudiantil](estudiante/notebooks/README.md).

## Modelo y evidencia

El modelo sintético y adimensional maximiza `sin(x) - 0.2*abs(x)` en `[-10,10]`, con radianes. Es un ejemplo didáctico, no mediciones de Planta Física. Generar poblaciones y aplicar variación aleatoria son objetivos de aprendizaje; no hay dataset externo ni preparación auxiliar de datos. Los extensos derivan el máximo global analítico `acos(0.2)` y lo distinguen de la malla y del mejor resultado observado del algoritmo.

| Paso | Idea que agrega | Resultado observado y comprobación |
|---|---|---|
| 1 | Objetivo y gráfica de aptitud | La altura de cada punto es `f(x)`; tres cimas interiores y una colina parcial muestran valores mejores que los vecinos. Mejor malla: `x=+1.378`, `f=+0.706`. La malla aproxima. |
| 2 | Individuo y población | Diez candidatos; mejor inicial `x=-0.323`, `f=-0.382` con semilla 52. Verificar aptitud guardada. |
| 3 | Selección por torneo | Cuatro individuos originales quedan sin copias. Cambia multiplicidad, sin crear genes. |
| 4 | Cruza por mezcla y acotación | 14 de 20 hijos salen del intervalo parental; todos respetan el dominio. Las dos propuestas comparten un sorteo. |
| 5 | Mutación gaussiana | Desde -4.6, 0/2000 y 110/2000 llegadas cumplen `abs(x-1.38)<1.5` con sigma 1 y 3. Cero observaciones no implica probabilidad cero. |
| 6 | Ciclo con reemplazo completo | Semilla 52: `x=+1.372`, `f=+0.706` tras diez generaciones. Verificar tamaño, límites y aptitud guardada. |
| 7 | Cambiar a semilla 16 | Final `x=-4.417`, `f=+0.073`. La semilla cambia toda la secuencia aleatoria; dos corridas no estiman una tasa de éxito. |

Las cifras mostradas proceden de los scripts. La comparación usa poblaciones finales ordenadas e historiales completos, sin redondear. Conteos y resultados en el mismo entorno deben coincidir exactamente. Entre entornos, comparar punto flotante con `rtol=atol=1e-12` e investigar trayectorias distintas.

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

Las recetas y bandas guían la construcción de los scripts. La presentación
compacta explica los conceptos sin requerir código ni la libreta; la libreta
compacta muestra después cómo se ejecutan con otra función de prueba inventada.

## Ejecución y procedencia

Desde la raíz del repositorio MA2015:

```sh
uv sync --locked
uv run python 02-lecciones/es/Leccion_01_Intro/src/primer_ejemplo_01_el_paisaje.py
```

Cada script corre sin argumentos ni importaciones de otros pasos, guarda su PNG relativo a su ubicación y cierra las figuras. Para ejecución sin interfaz, usar `MPLBACKEND=Agg`. El notebook guarda las figuras relativas al directorio del kernel y conserva las salidas embebidas.

La fuente académica es Ivan Gridin, *Learning Genetic Algorithms with Python*, capítulo 1. El curso conserva la secuencia algorítmica y agrega incrementos autónomos y explicaciones. Los extensos desarrollan derivaciones, sintaxis y conexión con decisiones de ingeniería.
