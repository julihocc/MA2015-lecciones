# Lección 01 — Notebooks

- [Extenso — estudio](leccion_01_intro.ipynb): fundamentos, código incremental completo, experimentos y soluciones razonadas.
- [Compacto — clase de 40 minutos](leccion_01_intro_compacta.ipynb): código preparado, resultados y comprobaciones breves; su tabla inicial coincide con la presentación compacta.

## Ejecución local

1. Desde la raíz de MA2015 o la del paquete estudiantil extraído, ejecutar `uv sync --locked`.
2. Abrir el notebook en Jupyter o VS Code y seleccionar el kernel de ese entorno.
3. Reiniciar el kernel y ejecutar todas las celdas en orden. Se requiere Python 3.11 o posterior.
4. Guardar el notebook para conservar salidas. Los PNG también se guardan en `figuras/`, relativa al directorio del kernel. No se usan rutas absolutas de la computadora.

## Google Colab

Subir el notebook a [Google Colab](https://colab.research.google.com/), elegir un entorno CPU y ejecutar todas las celdas en orden. Guardar una copia personal o descargar el notebook ejecutado. No se necesita dataset externo, clonar el repositorio ni montar Drive. El sistema de archivos remoto es temporal.

## Interpretación y reproducibilidad

Conservar parámetros y semillas base para la primera corrida. El extenso explica cada celda y usa nombres separados para experimentos opcionales. Ambas versiones conservan siete figuras embebidas. Las comprobaciones cubren tamaño, límites, aptitud guardada, conteos e historiales repetidos. La semilla controla todos los sorteos posteriores.

La figura del script de mutación presenta una densidad sin acotar; el notebook muestra un histograma de propuestas muestreadas y acotadas. Ambos usan las mismas muestras y criterio de llegada. Una altura de densidad o una fracción observada no garantiza una probabilidad para otra corrida.

La revisión actual pasó dos ejecuciones locales aisladas por notebook. Ambos notebooks también completaron Ejecutar todo en entornos CPU alojados nuevos de Google Colab el 22 de septiembre de 2026 (Python 3.13.15, NumPy 2.1.3, Matplotlib 3.10.0), incluidas las comprobaciones finales de resultados y reproducibilidad.
