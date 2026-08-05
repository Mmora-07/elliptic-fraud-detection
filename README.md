# 🛡️ Credit Card Fraud Detection with Autoencoders

Pipeline de detección de anomalías basado en **Error de Reconstrucción ($\mathcal{L}_{\text{MSE}}$)** con **PyTorch** sobre el dataset `creditcard.csv`.

---

## 📌 1. Fundamento Teórico

1. **Desbalance Extremo**: Las transacciones fraudulentas representan apenas el ~0.17% del total, haciendo inviable un enfoque supervisado tradicional sin sesgos severos.
2. **Hipótesis de la Variedad (Manifold Hypothesis)**: Las transacciones legítimas se concentran en un subespacio estadístico continuo. El Autoencoder se entrena **únicamente con datos normales** para memorizar la estructura de dicho subespacio.
3. **Criterio de Anomalía**: Datos atípicos sufren una alta distorsión al ser reconstruidos desde el espacio latente. Una transacción se clasifica como fraude si supera un umbral $\tau$:

$$\mathcal{L}_{\text{MSE}}(\mathbf{x}, \mathbf{\hat{x}}) = \frac{1}{d} \sum_{i=1}^{d} (x_i - \hat{x}_i)^2 > \tau$$

---

## 🏗️ 2. Arquitectura de la Red

Red simétrica modular definida en `src/model.py`:

```text
Entrada [d=29] ──► Encoder [16 ──► 8] ──► Bottleneck [z=4] ──► Decoder [8 ──► 16] ──► Reconstrucción [d=29]
Capas: Transformaciones lineales con LeakyReLU(0.2) y BatchNorm1d.Bottleneck: Compresión en subespacio latente $z \in \mathbb{R}^4$.Loss & Optimizador: $\mathcal{L}_{\text{MSE}}$ optimizado con AdamW + ReduceLROnPlateau.
📂 3. Estructura del Repositorio
├── data/
│   ├── raw/creditcard.csv             # Dataset original
│   └── processed/                     # Sets particionados (train, val, test)
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb # RobustScaler + Split no supervisado
│   ├── 02_autoencoder_training.ipynb  # Bucle de entrenamiento PyTorch
│   └── 03_evaluation_and_threshold.ipynb # Curvas PR-AUC / ROC, selección de τ y métricas
├── src/
│   ├── dataset.py                     # DataLoader PyTorch
│   ├── model.py                       # Encoder, Decoder & Autoencoder Modules
│   └── utils.py                       # Visualización y cálculo de MSE
├── models/best_autoencoder.pth        # Pesos guardados
└── requirements.txt
🚀 4. Workflow de EjecuciónPreprocesamiento (notebooks/01_...):Escalado de Amount y Time con RobustScaler (variables PCA V1-V28 estandarizadas).Train: 100% transacciones normales ($\text{Class} = 0$).Val/Test: Muestras normales residuales + 50%/50% de las anomalías ($\text{Class} = 1$).Entrenamiento (notebooks/02_...):Minimización de $\mathcal{L}_{\text{MSE}}$ sobre datos normales con EarlyStopping.Evaluación y Umbral (notebooks/03_...):Generación de densidad de errores (KDE) para ambas clases.Optimización del umbral $\tau$ maximizando el $F_1\text{-Score}$ en la curva Precision-Recall.