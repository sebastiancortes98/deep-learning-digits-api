# API optimizada para reconocer dígitos manuscritos

Este proyecto implementa el ciclo completo de un modelo de *deep learning* para reconocer dígitos manuscritos del **0 al 9**: prepara los datos, entrena una red neuronal, evalúa su calidad, la exporta a ONNX, la cuantiza a INT8, compara su rendimiento y finalmente la publica como una API REST construida con FastAPI.

El objetivo no es solamente obtener una predicción correcta. El proyecto también muestra cómo convertir un modelo entrenado con scikit-learn en un artefacto más pequeño y portable, preparado para realizar inferencia en CPU dentro de un contenedor Docker.

- Repositorio: <https://github.com/sebastiancortes98/deep-learning-digits-api>
- Imagen Docker: <https://hub.docker.com/r/scortes98/deep-learning-digits-api>
- Documentación interactiva al ejecutar la API: <http://127.0.0.1:8000/docs>

## ¿Qué hace el proyecto?

El flujo de trabajo es el siguiente:

1. Carga `sklearn.datasets.load_digits`, un conjunto de 1.797 imágenes en escala de grises.
2. Cada imagen de 8×8 se representa como un vector de 64 valores, con intensidades entre 0 y 16.
3. Divide los datos de forma estratificada en entrenamiento, validación y prueba.
4. Entrena un perceptrón multicapa o MLP para clasificar los diez dígitos.
5. Guarda el modelo original de scikit-learn en formato Joblib.
6. Exporta el modelo a ONNX con pesos FP32.
7. Aplica cuantización dinámica para generar una versión ONNX con pesos INT8.
8. Comprueba que la versión optimizada conserve la calidad del modelo original.
9. Mide tamaño, memoria, latencia y rendimiento de los modelos ONNX.
10. Sirve el modelo INT8 mediante una API FastAPI, tanto para predicciones individuales como por lotes.
11. Empaqueta únicamente los componentes necesarios para inferencia en una imagen Docker.

```text
load_digits
    │
    ▼
Preparación y división de datos
    │
    ▼
Entrenamiento MLP ──► digits_mlp.joblib
    │
    ▼
Exportación ONNX FP32 ──► digits_mlp_fp32.onnx
    │
    ▼
Cuantización dinámica INT8 ──► digits_mlp_int8.onnx
    │
    ├──► Pruebas y benchmark
    └──► FastAPI ──► Docker
```

## Modelo y datos

### Conjunto de datos

Se utiliza `load_digits` de scikit-learn. Sus imágenes son matrices de 8×8 píxeles y cada píxel tiene un valor entre 0 y 16. La división configurada en `params.yaml` es:

| Partición | Proporción | Muestras |
| --- | ---: | ---: |
| Entrenamiento | 70 % | 1.257 |
| Validación | 15 % | 270 |
| Prueba | 15 % | 270 |

La semilla aleatoria es `42` y la separación es estratificada, por lo que intenta conservar la proporción de cada dígito en todas las particiones.

### Arquitectura de la red neuronal

El clasificador es un `MLPClassifier` con esta arquitectura:

```text
64 entradas → 128 neuronas → 64 neuronas → 32 neuronas → 10 clases
```

- Las 64 entradas corresponden a los 64 píxeles de la imagen.
- Las tres capas ocultas utilizan activación ReLU.
- Las 10 salidas representan las probabilidades de los dígitos del 0 al 9.
- El entrenamiento utiliza Adam, regularización L2, *early stopping* y un máximo de 250 iteraciones.
- Los principales hiperparámetros están centralizados en `params.yaml`.

### Artefactos del modelo

La carpeta `models/` contiene tres representaciones del mismo clasificador:

| Archivo | Función | Tamaño registrado |
| --- | --- | ---: |
| `digits_mlp.joblib` | Modelo original de scikit-learn y referencia de entrenamiento. | 237.496 bytes |
| `digits_mlp_fp32.onnx` | Modelo ONNX con pesos de 32 bits y referencia portable. | 77.452 bytes |
| `digits_mlp_int8.onnx` | Modelo ONNX cuantizado dinámicamente a 8 bits y usado por la API. | 25.494 bytes |

La versión INT8 ocupa aproximadamente un **67 % menos** que la versión ONNX FP32. La cuantización reduce principalmente el tamaño de los pesos; ONNX Runtime ejecuta el modelo optimizado en CPU.

`models/MODEL_VERSION` guarda una versión derivada del SHA-256 del modelo INT8. La API devuelve esta versión para poder identificar el artefacto exacto utilizado en cada respuesta.

## ¿Cómo funciona una predicción?

1. La API recibe exactamente 64 valores entre 0 y 16.
2. Convierte la entrada a una matriz NumPy de tipo `float32`.
3. Normaliza los píxeles dividiéndolos por 16, dejándolos en el rango de 0 a 1.
4. Ejecuta `digits_mlp_int8.onnx` mediante ONNX Runtime y `CPUExecutionProvider`.
5. Obtiene diez probabilidades, una por cada posible dígito.
6. Selecciona la clase con la probabilidad más alta.
7. Devuelve el dígito, su confianza, todas las probabilidades, la versión del modelo y el tiempo de inferencia.

La sesión de ONNX Runtime se carga una sola vez al iniciar FastAPI. Las solicitudes posteriores reutilizan el modelo en memoria en vez de cargarlo para cada predicción.

## Estructura del proyecto

```text
deep_learning_digits_api/
├── app/                    # API de inferencia
├── ml/                     # Entrenamiento, exportación y cuantización
├── models/                 # Modelos generados y versión desplegada
├── results/                # Métricas y evidencias reproducibles
├── scripts/                # Comandos ejecutables del flujo de ML
├── tests/                  # Pruebas de API y calidad del modelo
├── .github/workflows/      # Integración continua en GitHub Actions
├── Dockerfile              # Imagen de producción en dos etapas
├── docker-compose.yml      # Ejecución local segura con Docker Compose
├── dvc.yaml                # Definición reproducible del pipeline
├── params.yaml             # Datos, hiperparámetros y optimización
├── requirements.txt        # Dependencias mínimas de inferencia
└── requirements-train.txt  # Dependencias de entrenamiento y pruebas
```

### `app/`: servicio de inferencia

- `main.py`: crea la aplicación FastAPI, carga el clasificador al iniciar y define los endpoints de salud, disponibilidad y predicción.
- `inference.py`: encapsula ONNX Runtime, configura la ejecución en CPU, normaliza los píxeles y mide el tiempo de inferencia.
- `schemas.py`: define y valida las solicitudes y respuestas con Pydantic. Rechaza valores fuera del rango 0–16, entradas con una longitud distinta de 64 y campos no declarados.
- `config.py`: resuelve las rutas del modelo INT8 y del archivo de versión.

### `ml/`: lógica de aprendizaje automático

`ml/pipeline.py` concentra la lógica reutilizable del proyecto:

- carga los parámetros desde `params.yaml`;
- obtiene el conjunto integrado `load_digits`;
- normaliza y divide los datos;
- construye y entrena el MLP;
- evalúa *accuracy* y F1 macro;
- exporta el modelo a ONNX con *opset* 17;
- preprocesa y cuantiza los pesos a INT8;
- ejecuta inferencia directa sobre los modelos ONNX;
- calcula hashes SHA-256 y genera las métricas de entrenamiento.

### `models/`: modelos entrenados

Contiene los artefactos Joblib, ONNX FP32 y ONNX INT8. La aplicación usa por defecto `digits_mlp_int8.onnx`, aunque la variable de entorno `MODEL_PATH` permite indicar otro artefacto compatible.

El contenedor de producción copia únicamente el modelo INT8 y `MODEL_VERSION`; los modelos de entrenamiento y FP32 no se incluyen en la imagen final.

### `scripts/`: tareas ejecutables

- `train.py`: entrena el MLP, exporta los modelos, cuantiza el ONNX y actualiza `training_metrics.json`.
- `benchmark.py`: compara FP32 e INT8 con lotes de 1, 16 y 64 elementos. Realiza 20 ejecuciones de calentamiento y 100 mediciones por lote.
- `smoke_test.py`: inicia temporalmente la API, espera que esté disponible y comprueba una predicción real de extremo a extremo.
- `check_gates.py`: falla si la exactitud INT8 es inferior a 95 % o si la latencia p95 para un elemento supera 10 ms.

### `results/`: métricas y evidencia

- `training_metrics.json`: particiones, arquitectura, calidad, tamaños y SHA-256 de los artefactos.
- `benchmark.json`: entorno de ejecución, memoria, latencias p50/p95 y rendimiento por tamaño de lote.
- `smoke_test.json`: evidencia de una predicción completa contra la API.

Los tiempos dependen del procesador, sistema operativo, carga de la máquina y versión de las dependencias. Por eso deben interpretarse junto con el entorno registrado en `benchmark.json`.

### `tests/`: validación automática

Las pruebas verifican que:

- `/health` y `/ready` respondan correctamente;
- una imagen produzca el dígito esperado;
- la inferencia por lotes funcione;
- una entrada inválida sea rechazada con HTTP 422;
- la exactitud del modelo INT8 sea al menos 95 %;
- el desacuerdo entre FP32 e INT8 no supere 2 %.

### `params.yaml` y `dvc.yaml`

`params.yaml` es la fuente central de configuración: semilla, proporciones de datos, arquitectura, regularización, tamaño de lote, tasa de aprendizaje, número máximo de iteraciones, versión ONNX y tipo de cuantización.

`dvc.yaml` describe dos etapas reproducibles:

1. `train_export_quantize`: entrenamiento, exportación y cuantización.
2. `benchmark`: medición de los modelos FP32 e INT8.

## Resultados actuales

La versión incluida `digits-mlp-v1-dc60599d0767` obtuvo:

| Modelo | Exactitud de prueba | F1 macro |
| --- | ---: | ---: |
| scikit-learn | 98,15 % | 98,15 % |
| ONNX FP32 | 98,15 % | 98,15 % |
| ONNX INT8 | 98,15 % | 98,15 % |

En esta ejecución, la cuantización no produjo pérdida de exactitud respecto del modelo FP32. Los resultados pueden variar al cambiar los parámetros, los datos o las versiones de las bibliotecas.

## Endpoints de la API

| Método | Ruta | Descripción |
| --- | --- | --- |
| `GET` | `/health` | Confirma que el proceso web está activo. |
| `GET` | `/ready` | Confirma que el modelo fue cargado y devuelve su versión. |
| `POST` | `/v1/predict` | Clasifica una imagen de 8×8. |
| `POST` | `/v1/predict/batch` | Clasifica entre 1 y 256 imágenes en una solicitud. |

### Ejemplo de predicción individual

```bash
curl -X POST http://127.0.0.1:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pixels": [0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0]
  }'
```

Respuesta resumida:

```json
{
  "model_version": "digits-mlp-v1-dc60599d0767",
  "inference_ms": 0.031,
  "prediction": {
    "digit": 0,
    "confidence": 0.999,
    "probabilities": [0.999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0]
  }
}
```

Los valores exactos de confianza y latencia pueden cambiar entre ejecuciones.

## Ejecución con Docker

### Usar la imagen publicada

```bash
docker pull scortes98/deep-learning-digits-api:latest
docker run --rm -p 8000:8000 scortes98/deep-learning-digits-api:latest
```

Si el puerto 8000 ya está ocupado, se puede usar otro puerto local sin cambiar el contenedor:

```bash
docker run --rm -p 8010:8000 scortes98/deep-learning-digits-api:latest
```

En ese caso la API queda disponible en <http://127.0.0.1:8010/docs>.

### Construir localmente

```bash
docker build -t deep-learning-digits-api:local .
docker run --rm -p 8000:8000 deep-learning-digits-api:local
```

También se puede usar:

```bash
docker compose up --build
```

La imagen usa una construcción en dos etapas, ejecuta la aplicación con un usuario sin privilegios, incluye un `HEALTHCHECK` y contiene solo las dependencias y el modelo necesarios para inferencia. Compose utiliza un sistema de archivos de solo lectura, elimina capacidades de Linux y monta `/tmp` como almacenamiento temporal.

## Instalación y reproducción local

Se recomienda Python 3.12.

### Linux o macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
```

### Reproducir el flujo completo

```bash
python scripts/train.py
python -m pytest -q
python scripts/benchmark.py
python scripts/check_gates.py
python scripts/smoke_test.py
```

### Iniciar la API sin Docker

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Después se puede abrir:

- Swagger UI: <http://127.0.0.1:8000/docs>
- Esquema OpenAPI: <http://127.0.0.1:8000/openapi.json>
- Estado del servicio: <http://127.0.0.1:8000/ready>

## Integración continua

El flujo `.github/workflows/ci.yml` se ejecuta en cada *push* a `main` y en cada *pull request*. GitHub Actions:

1. instala Python 3.12 y las dependencias;
2. reproduce los artefactos del modelo;
3. ejecuta las pruebas;
4. genera el benchmark;
5. aplica las puertas de calidad y latencia;
6. construye y prueba el contenedor;
7. conserva durante 30 días el modelo INT8, su versión y los resultados JSON.

Así, el repositorio comprueba automáticamente que el modelo publicado siga siendo reproducible, preciso y utilizable como servicio.
