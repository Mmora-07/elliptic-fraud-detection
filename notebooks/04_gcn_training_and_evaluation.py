import torch
import torch.nn as nn
from sklearn.metrics import classification_report, precision_recall_curve, auc, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('.')

from src.graph_dataset import load_elliptic_graph
from src.gnn_model import GCNClassifier

root = Path.cwd()

# 1. Cargar Datos y Grafo
data, _ = load_elliptic_graph(
    features_path=str(root / 'data' / 'raw' / 'elliptic_txs_features.csv'),
    classes_path=str(root / 'data' / 'raw' / 'elliptic_txs_classes.csv'),
    edgelist_path=str(root / 'data' / 'raw' / 'elliptic_txs_edgelist.csv')
)

# 2. Creación de Máscaras Temporales
train_mask = (data.time_step <= 34) & (data.y != -1)
val_mask = (data.time_step >= 35) & (data.time_step <= 41) & (data.y != -1)
test_mask = (data.time_step >= 42) & (data.y != -1)

if val_mask.sum().item() == 0:
    val_mask = (data.time_step >= 35) & (data.time_step <= 45) & (data.y != -1)
if test_mask.sum().item() == 0:
    test_mask = (data.time_step >= 42) & (data.y != -1)

print(f"Nodos Train: {train_mask.sum().item()}")
print(f"Nodos Val:   {val_mask.sum().item()}")
print(f"Nodos Test:  {test_mask.sum().item()}")

# 3. Manejo de Desbalance de Clases
# Peso inversamente proporcional a la frecuencia

y_train = data.y[train_mask]
num_licit = (y_train == 0).sum().item()
num_illicit = (y_train == 1).sum().item()

if num_illicit == 0:
    raise ValueError('No hay ejemplos ilícitos en el split de entrenamiento; ajusta las máscaras temporales.')

pos_weight = torch.tensor([1.0, num_licit / num_illicit], dtype=torch.float32)
print(f"Pesos de Clase (Lícita: {pos_weight[0]:.2f}, Ilícita: {pos_weight[1]:.2f})")

# 4. Configurar Modelo, Optimizador y Criterio
device = torch.device('cpu')
data = data.to(device)
pos_weight = pos_weight.to(device)

model = GCNClassifier(in_channels=data.num_node_features, hidden_channels=64, out_channels=2, dropout=0.2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=pos_weight)

model_dir = root / 'models'
model_dir.mkdir(exist_ok=True)

# 5. Bucle de Entrenamiento
best_val_prauc = 0.0
epochs = 60

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out_val = model(data.x, data.edge_index)
        probs_val = torch.softmax(out_val[val_mask], dim=1)[:, 1].cpu().numpy()
        y_val_true = data.y[val_mask].cpu().numpy()
        precision, recall, _ = precision_recall_curve(y_val_true, probs_val)
        val_prauc = auc(recall, precision)

    if val_prauc > best_val_prauc:
        best_val_prauc = val_prauc
        torch.save(model.state_dict(), model_dir / 'best_gcn_model.pth')

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss Train: {loss.item():.4f} | Val PR-AUC: {val_prauc:.4f}")

# 6. Evaluación Final en el Test Set
model.load_state_dict(torch.load(model_dir / 'best_gcn_model.pth', map_location=device))
model.eval()

with torch.no_grad():
    logits_test = model(data.x, data.edge_index)[test_mask]
    probs_test = torch.softmax(logits_test, dim=1)[:, 1].cpu().numpy()
    preds_test = torch.argmax(logits_test, dim=1).cpu().numpy()
    y_test_true = data.y[test_mask].cpu().numpy()

precision, recall, _ = precision_recall_curve(y_test_true, probs_test)
test_prauc = auc(recall, precision)

print("\n--- RESULTADOS GCN (TEST SET) ---")
print(f"PR-AUC Test: {test_prauc:.4f}")
print("\nReporte de Clasificación:")
print(classification_report(y_test_true, preds_test, target_names=['Lícita (0)', 'Ilícita (1)']))

cm = confusion_matrix(y_test_true, preds_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Lícita', 'Ilícita'], yticklabels=['Lícita', 'Ilícita'])
plt.title('Matriz de Confusión - GCN Model')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()
