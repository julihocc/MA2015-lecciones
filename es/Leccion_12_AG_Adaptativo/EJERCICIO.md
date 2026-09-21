# Lección 12 — Ejercicio: la cuadrícula completa de cruza × mutación

No es un séptimo script. La secuencia de seis pasos terminó. Esta hoja pide
reutilizar `ag_adaptativo_06_un_rival_afinado.py` sin cambiarlo, salvo por ampliar
una constante.

`CONFIGURACIONES_FIJAS` contiene cuatro celdas para que el paso 6 dure cerca de
16 segundos. Es razonable preguntar si una quinta celda habría superado al
régimen adaptativo por más. Mídelo.

## Qué hacer

1. Copia `src/ag_adaptativo_06_un_rival_afinado.py` (no edites el archivo didáctico).
2. Reemplaza

   ```python
   CONFIGURACIONES_FIJAS = [(0.9, 0.25), (0.6, 0.05), (0.4, 0.15), (0.2, 0.30)]
   ```

   por la cuadrícula 4 × 4 de cruza × mutación

   ```python
   CONFIGURACIONES_FIJAS = [
       (cruza, mutacion)
       for cruza in (0.20, 0.40, 0.60, 0.90)
       for mutacion in (0.05, 0.15, 0.25, 0.30)
   ]
   ```

   Esos cuatro valores de cruza y cuatro de mutación son exactamente los
   extremos ya presentes en el paso 6: las cuatro celdas originales permanecen
   y aparecen doce nuevas a su alrededor.
3. **Predice el tiempo antes de ejecutar.** El paso 6 mide 5 regímenes (el
   adaptativo y cuatro configuraciones fijas) en unos 16 segundos, con 12
   ejecuciones y 12,000 evaluaciones cada una. La cuadrícula mide 17 regímenes
   con el mismo presupuesto. Escribe la predicción y después ejecútala.
4. A partir de la tabla impresa, reporta:

   - cuáles celdas fijas superan al régimen adaptativo **en la media** (menor
     longitud media de ruta);
   - por cuánto (`adaptativo menos fijo`: un número positivo significa que el
     régimen adaptativo produjo una ruta más larga);
   - cuál celda es la mejor de las dieciséis y si el intervalo aproximado
     diferencia-media ± dos errores estándar excluye cero. Trátalo como regla
     descriptiva, no como prueba de significancia.

Conserva `EJECUCIONES = 12` y `PRESUPUESTO = 12_000`. No vuelvas a afinar nada.

## Advertencia de tiempo

No inicies esto en los últimos cinco minutos de una sesión.

Predicción: \(16\,\text{s} \times 17 / 5 \approx 54\,\text{s}\). Reserva un
minuto completo, además del tiempo para leer la tabla.
