# Imagen del API para el deploy (Render, con render.yaml en la raiz del repo; vale en
# cualquier hosting que corra contenedores). Solo el runtime: requirements.txt, sin el
# pipeline de datos (SDV/torch).
FROM python:3.12-slim

# LightGBM necesita OpenMP (libgomp), que la imagen slim no trae.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Sin root dentro del contenedor.
RUN useradd -m -u 1000 user
USER user
# Un hilo por libreria numerica. LightGBM (OpenMP), NumPy (OpenBLAS) y numba lanzan uno
# por cada nucleo que ven, y en el plan gratuito ven todos los de la maquina con una
# decima de CPU de cuota: los que esperan en activo se la pueden comer entera.
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Un solo worker: los diez modelos y sus explicadores SHAP se cargan por proceso, y el
# rate limiting guarda sus contadores en memoria (con varios workers, cada uno llevaria
# los suyos). Los hilos atienden peticiones en paralelo. El hosting fija PORT (Render,
# 10000); 8000 es el de desarrollo. El timeout es para peticiones lentas con la decima de
# CPU; NO cubre el arranque (gunicorn no vigila al worker hasta que la app termina de
# cargar), ahi el unico limite es el de Render, 15 minutos.
ENV PORT=8000
EXPOSE 8000
CMD gunicorn --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:${PORT} app:app
