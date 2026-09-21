# Lección 01 — Libreta guiada

Abre [leccion_01_intro.ipynb](leccion_01_intro.ipynb) para recorrer de forma
lineal la construcción del primer algoritmo genético. La libreta intercala la
explicación conceptual con incrementos pequeños de código, preguntas de
predicción, comprobaciones y experimentos opcionales. Conserva sus salidas
ejecutadas y siete figuras, de modo que también sirve como respaldo sin red.

La ruta principal toma alrededor de 50 minutos. La lectura detallada y los
experimentos opcionales pueden continuar después de la sesión.

## Jupyter local o VS Code

1. Desde la raíz del repositorio, ejecuta una vez `uv sync --frozen`.
2. Abre `leccion_01_intro.ipynb` en Jupyter o VS Code.
3. En VS Code, elige **Select Kernel** y selecciona el entorno Python `.venv` /
   `ma2015-genetic-algorithms` del proyecto.
4. Reinicia el kernel y elige **Run All**. El proyecto requiere Python 3.11+ y
   ya declara NumPy, Matplotlib e `ipykernel`.

Cuando VS Code expone la ruta local de la libreta, los PNG quedan en
`notebooks/figuras/` junto a ella aunque el kernel arranque en la raíz del
workspace. Otros frontends locales de Jupyter usan su directorio de trabajo;
cuando sea posible, abre la libreta desde esta carpeta.

## Extensión Google Colab en VS Code

1. Instala la [extensión Colab oficial de Google](https://marketplace.visualstudio.com/items?itemName=Google.colab).
2. Abre la libreta local en VS Code.
3. Elige **Select Kernel → Colab → Auto Connect** y completa el inicio de sesión
   con Google cuando se solicite.
4. Elige **Run All**. Basta un servidor estándar de CPU; no necesitas subir el
   repositorio, montar Drive, cargar datos externos ni usar aceleradores.

La libreta permanece local mientras el código se ejecuta en el kernel remoto.
Como ese kernel no ve la ruta del equipo, los PNG se guardan en el directorio
remoto actual, normalmente `/content/figuras`. Puedes recuperarlos desde la
vista **Contents** de la barra de actividad de Colab; las figuras también
permanecen incrustadas en la
libreta local cuando la guardas. Consulta la
[guía vigente de la extensión](https://github.com/googlecolab/colab-vscode/wiki/User-Guide).

## Google Colab en el navegador

1. Descarga la libreta y abre [Google Colab](https://colab.research.google.com/).
2. Elige **Archivo → Subir libreta** y selecciona el archivo descargado.
3. Guarda una copia personal en Drive si quieres conservar cambios.
4. Ejecuta las celdas de arriba hacia abajo con **Shift+Enter**.
5. Si cambias una definición anterior, vuelve a ejecutar las celdas que
   dependen de ella; si el estado resulta confuso, reinicia el entorno y usa
   **Ejecutar todas**.

No se necesita clonar el repositorio, montar Drive, descargar datos ni cargar
las diapositivas. La libreta usa la biblioteca estándar de Python, NumPy y
Matplotlib, incluidos en un entorno estándar de CPU de Colab. El sistema de
archivos remoto es temporal; guarda la libreta para conservar las figuras.

## Relación con la lección

La libreta sigue en el mismo orden las
[diapositivas](../slides/leccion_01.tex) y los
[siete scripts autónomos](../src/):

| Paso | Incremento principal |
|---|---|
| 1 | Función objetivo y paisaje de búsqueda |
| 2 | Individuo, candidato aleatorio y población |
| 3 | Selección por torneo |
| 4 | Acotamiento y cruza de mezcla |
| 5 | Mutación gaussiana |
| 6 | Ciclo generacional e historial de convergencia |
| 7 | Semilla 16 y comparación con un óptimo local |

Las definiciones se acumulan entre celdas y la libreta completa es
autocontenida. La lógica, los parámetros, las semillas y el orden de las
extracciones aleatorias coinciden con los scripts. Los experimentos opcionales
usan variables separadas y no alteran los resultados base. La celda de
comprobaciones valida las cifras documentadas y la reproducibilidad de las
corridas completas.

## Validación

Verificada localmente el 17 de septiembre de 2026 con Python 3.11.16, NumPy
2.4.6 y Matplotlib 3.11.1 del archivo de bloqueo del curso. Las herramientas de
Jupyter (`ipykernel`, `nbformat` y `nbconvert`) están incluidas en el entorno
bloqueado del proyecto.

- Formato `nbformat` válido y 104 celdas ejecutadas desde un kernel nuevo en
  una carpeta aislada que contenía únicamente la libreta.
- Las 34 celdas de código terminaron sin errores y cada una tiene una
  explicación inmediatamente anterior.
- Se conservaron siete figuras embebidas y se revisaron visualmente sus
  títulos, rótulos, leyendas y correspondencia con la evidencia numérica.
- Se reprodujeron 4 de 10 individuos sin copias, 14 de 20 hijos fuera del
  intervalo parental y 0/2000 frente a 110/2000 llegadas por mutación.
- Las semillas 52 y 16 reprodujeron exactamente los historiales y soluciones
  finales de los scripts: `x=+1.372, f=+0.706` y `x=-4.417, f=+0.073`.
- El verificador de recetas y bandas de los siete scripts pasó sin errores.
- Se generó correctamente la versión HTML para comprobar la estructura del
  documento; la política del navegador local impidió abrir automáticamente el
  archivo `file://`, por lo que no se certifica una revisión visual del HTML.

La validación corresponde a Jupyter local. La libreta todavía no se ha
ejecutado dentro del servicio alojado de Google Colab ni mediante su extensión
de VS Code. El manejo de rutas para kernel remoto se validó por separado con
una ruta local no disponible y un directorio de trabajo aislado.

