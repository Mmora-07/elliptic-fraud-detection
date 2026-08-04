# 🛡️ Detección de Anomalías Financieras en Bitcoin usando Autoencoders

Este repositorio contiene la implementación e investigación de un enfoque no supervisado para la **detección de transacciones fraudulentas e ilícitas** sobre la red de Bitcoin, utilizando la arquitectura de **Autoencoders Densos** sobre el *Elliptic Bitcoin Dataset*.

---

## 📌 Planteamiento del Problema

En los sistemas financieros, la detección de fraude enfrenta dos obstáculos críticos: el **extremo desbalance de clases** (<2% de transacciones anómalas) y la constante evolución de las tácticas delictivas. 

Bajo la **Hipótesis de la Variedad (Manifold Hypothesis)**, se asume que las transacciones lícitas yacen cerca de una subvariedad de menor dimensión en el espacio de características. Un **Autoencoder** entrenado *exclusivamente* con comportamiento lícito aprenderá a reconstruir este espacio. Por ende, cualquier transacción anómala o fraudulenta presentará una distorsión significativa al ser reconstruida, resultando en un **Error Cuadrático Medio ($\mathcal{L}_{\text{MSE}}$) elevado**.

---

## 📐 Fundamentación Matemática

### 1. Codificación y Decodificación
Dada una transacción $x \in \mathbb{R}^d$ ($d=166$ atributos), el codificador la proyecta a una representación latente restringida $z \in \mathbb{R}^k$ ($k \ll d$):

$$z = f_\theta(x) = \sigma(W_e x + b_e)$$

Posteriormente, el decodificador intenta reconstruir la señal original a partir de $z$:

$$\hat{x} = g_\phi(z) = \sigma(W_d z + b_d)$$

### 2. Función de Pérdida (MSE)
La red optimiza los parámetros $\{\theta, \phi\}$ minimizando el error de reconstrucción sobre el conjunto de entrenamiento lícito:

$$\mathcal{L}_{\text{MSE}}(x, \hat{x}) = \frac{1}{d} \sum_{j=1}^{d} (x_j - \hat{x}_j)^2$$

### 3. Regla de Decisión y Score de Anomalía
Se define la score de anomalía $S(x) = \mathcal{L}_{\text{MSE}}(x, \hat{x})$. La clasificación binaria de la transacción se realiza comparando contra un umbral crítico $\tau$:

$$\hat{y} = \begin{cases} 1 \quad (\text{Fraude}), & \text{si } S(x) > \tau \\ 0 \quad (\text{Lícito}), & \text{si } S(x) \le \tau \end{cases}$$

---

## 📂 Estructura del Repositorio

```text
elliptic-fraud-detection/
├── data/                       <-- Almacenamiento local de datos (Raw / Processed)
├── notebooks/                  <-- Exploración y entrenamiento interactivo
│   ├── 01_eda_and_preprocessing.ipynb
│   └── 02_autoencoder_training.ipynb
├── src/                        <-- Módulos ejecutables en Python
│   ├── data_loader.py          <-- Carga, filtrado y escalado
│   └── model.py                <-- Arquitectura PyTorch del Autoencoder
├── .gitignore
├── requirements.txt
└── README.md