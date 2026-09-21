# Lecci�n 11 � cuadernos guiados

Estos cuadernos son la ruta autocontenida para estudiantes a trav�s de las secuencias numeradas de scripts de la lecci�n. Usa cada cuaderno de principio a fin; cada celda ejecutable tiene una gu�a de lectura inmediatamente anterior.

## �ndice

- [`leccion_11_ecuaciones.ipynb`](leccion_11_ecuaciones.ipynb) � b�squeda con AG, enumeraci�n y certificados exactos. 33 celdas, 11 salidas sem�nticas y 4 figuras integradas.
- [`leccion_11_coloreado_grafos.ipynb`](leccion_11_coloreado_grafos.ipynb) � coloreado guiado por aptitud y comparaci�n exacta por retroceso. 35 celdas, 13 salidas sem�nticas y 4 figuras integradas.

## Ejecuci�n local

Desde la ra�z del repositorio ejecuta `uv sync --frozen`, selecciona `.venv` como kernel de Jupyter o VS Code, abre el cuaderno y usa **Reiniciar kernel y ejecutar todo**. El cuaderno escribe salidas temporales junto a su copia de ejecuci�n y no necesita importar el repositorio ni leer archivos de datos externos.

## Ejecuci�n en Colab

Sube el cuaderno a Colab en el navegador, o �brelo con la extensi�n de Colab para VS Code, guarda una copia personal y ejecuta todas las celdas en orden. La Lecci�n 11 instala `python-igraph==1.0.0` fijado solo si falla la importaci�n. Las figuras generadas existen �nicamente durante esa sesi�n, salvo que las descargues.

## Alcance y validaci�n

Los scripts congelados siguen siendo los ejemplos compactos can�nicos; los cuadernos agregan lectura guiada, predicciones, interpretaciones, experimentos opcionales, respuestas y aserciones finales. El date omitted cada cuaderno pas� validaci�n con `nbformat`, compilaci�n del c�digo, controles de identificadores �nicos y adyacencia de gu�as, ejecuci�n aislada con solo el cuaderno presente y tres simulaciones con kernels nuevos: local gen�rica, ruta accesible tipo VS Code y ruta inaccesible tipo remota. Las salidas retenidas y los nombres de figuras coincidieron con los scripts, y el cuaderno ingl�s correspondiente produjo la misma evidencia num�rica determinista. El entorno bloqueado us� Python 3.11.16, NumPy 2.4.6, Matplotlib 3.11.1, pandas 3.0.5, nbformat 5.11.1 y nbconvert 7.17.1.

No se probaron sesiones autenticadas de la extensi�n Colab para VS Code ni de Colab en navegador; la simulaci�n de rutas no es una prueba alojada en Colab.
