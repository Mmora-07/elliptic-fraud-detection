# 🪙 Detección de Transacciones Ilícitas en Bitcoin: Evaluación de Autoencoders Tabulares vs. Graph Convolutional Networks (GCN)

Este repositorio alberga la investigación e implementación de modelos de aprendizaje automático para la detección de transacciones fraudulentas e ilícitas en la red Bitcoin, utilizando el **Elliptic Bitcoin Dataset**.

El proyecto contrasta un enfoque **no supervisado basado en anomalías tabulares (Autoencoder)** frente a un enfoque de **aprendizaje profundo en grafos (Graph Convolutional Network - GCN)** que incorpora explícitamente la topología de la red de transacciones.

---

## 📌 Resumen del Proyecto

* **Dataset:** Elliptic Bitcoin Dataset (203,769 nodos/transacciones, 234,355 aristas/flujos, 166 características numéricas).
* **Problema:** Alto desbalance de clases (las transacciones ilícitas representan aproximadamente el 10% de los datos etiquetados) y un marcado cambio temporal de la distribución (*covariate shift*).
* **Línea Base (Fase 1):** Autoencoder entrenado exclusivamente con transacciones lícitas (`class2`) para medir anomalías mediante el Error Cuadrático Medio de Reconstrucción ($\mathcal{L}_{\text{MSE}}$).
* **Modelo Avanzado (Fase 2 - Rama `feature/gnn-gcn`):** Red Neuronal Convolucional para Grafos (GCN) semisupervisada construida con `PyTorch Geometric`, combinando atributos de transacción con la matriz de adyacencia del grafo.

---

## 📁 Estructura del Repositorio

```text
elliptic-fraud-detection/
├── data/
│   ├── raw/                        # Datasets crudos de Elliptic (.csv)
│   │   ├── elliptic_txs_classes.csv
│   │   ├── elliptic_txs_edgelist.csv
│   │   └── elliptic_txs_features.csv
│   └── processed/                  # Matrices procesadas, scalers y arrays NumPy
├── models/                         # Pesos guardados (.pth) y configuraciones (.joblib)
│   ├── autoencoder_licit.pth
│   ├── gcn_elliptic.pth
│   └── threshold_config.joblib
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_autoencoder_training.ipynb
│   ├── 03_evaluation_and_threshold.ipynb
│   └── 04_gcn_training_and_evaluation.ipynb
├── src/                            # Módulos Python reutilizables
│   ├── __init__.py
│   ├── model.py                    # Arquitectura PyTorch del Autoencoder Tabular
│   ├── gnn_model.py                # Arquitectura GCN (PyTorch Geometric)
│   └── graph_dataset.py            # Loader y constructor del grafo PyG
├── .gitignore
├── README.md
└── requirements.txt