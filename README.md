# 🛡️ Credit Card Fraud Detection with Autoencoders

Proyecto de detección de fraude en transacciones con tarjeta de crédito usando PyTorch y Autoencoders.

Este flujo trabaja exclusivamente con el dataset `data/raw/creditcard.csv` y no utiliza datos de Bitcoin ni del conjunto de datos Elliptic.

---

## 📌 1. Fundamento Teórico

1. **Desbalance Extremo**: Las transacciones fraudulentas representan una fracción muy pequeña del total, por lo que un enfoque supervisado clásico puede sesgarse y generar falsos negativos.
2. **Aprendizaje No Supervisado**: El Autoencoder se entrena solo con transacciones normales para aprender la distribución de comportamientos legítimos.
3. **Criterio de Anomalía**: Una transacción se clasifica como anomalía cuando el error de reconstrucción supera un umbral óptimo $\tau$:

$$\mathcal{L}_{\text{MSE}}(\mathbf{x}, \hat{\mathbf{x}}) = \frac{1}{d} \sum_{i=1}^{d} (x_i - \hat{x}_i)^2$$

---

## 🧩 2. Estructura del Repositorio

- `data/raw/creditcard.csv` — Dataset original de transacciones con variables `Time`, `Amount`, `Class` y `V1` a `V28`.
- `data/processed/` — Artefactos procesados y conjuntos listos para entrenamiento y evaluación.
- `models/` — Pesos y checkpoints de modelos entrenados.
- `notebooks/01_eda_and_preprocessing.ipynb` — Exploración, preprocesamiento y partición de datos.
- `notebooks/02_autoencoder_training.ipynb` — Entrenamiento del Autoencoder en transacciones normales.
- `notebooks/03_evaluation_and_threshold.ipynb` — Evaluación final, selección de umbral y métricas.
- `src/model.py` — Definición de la arquitectura del Autoencoder.
- `scripts/hyperparameter_search.py` — Búsqueda y comparación de configuraciones de entrenamiento.

---

## 🚀 3. Flujo de Ejecución

1. **Preprocesamiento** (`notebooks/01_eda_and_preprocessing.ipynb`):
   - Carga `creditcard.csv`.
   - Escala `Time` y `Amount` con `RobustScaler`.
   - Construye `X_train_licit.npy` usando solo transacciones normales (`Class = 0`).
   - Guarda `X_val.npy`, `X_test.npy`, `y_val.npy`, `y_test.npy`, `scaler.joblib` y demás artefactos.

2. **Entrenamiento** (`notebooks/02_autoencoder_training.ipynb`):
   - Carga `data/processed/X_train_licit.npy`.
   - Entrena el Autoencoder minimizando la pérdida MSE.
   - Guarda el checkpoint en `models/autoencoder_creditcard.pth`.

3. **Evaluación y Umbral** (`notebooks/03_evaluation_and_threshold.ipynb`):
   - Carga `X_val.npy`, `X_test.npy`, `y_val.npy`, `y_test.npy` y `scaler.joblib`.
   - Carga el checkpoint `models/autoencoder_creditcard.pth`.
   - Calcula errores de reconstrucción y selecciona el umbral óptimo por `F1` en validación.
   - Evalúa la detección en el conjunto de prueba y guarda métricas y artefactos.

---

## 📌 4. Archivos Clave

- `data/processed/X_train_licit.npy` — Entrenamiento exclusivo con transacciones normales.
- `data/processed/X_val.npy`, `data/processed/y_val.npy` — Validación para selección de umbral.
- `data/processed/X_test.npy`, `data/processed/y_test.npy` — Evaluación final.
- `data/processed/scaler.joblib` — Escalador utilizado en el preprocesamiento.
- `models/autoencoder_creditcard.pth` — Checkpoint final del Autoencoder.
- `data/processed/threshold_config.joblib` — Umbral óptimo y métricas almacenadas.

---

## ✅ Nota

Este proyecto ya no usa datos de Bitcoin ni del dataset Elliptic. Todo el flujo está adaptado al dataset de fraude con tarjeta de crédito `creditcard.csv`.