import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


class Data:
    """Mini-implementación compatible de torch_geometric.data.Data sin depender de PyG."""

    def __init__(self, x=None, edge_index=None, y=None, time_step=None, scaler=None):
        self.x = x
        self.edge_index = edge_index
        self.y = y
        self.time_step = time_step
        self.scaler = scaler

    @property
    def num_nodes(self) -> int:
        return self.x.size(0) if self.x is not None else 0

    @property
    def num_edges(self) -> int:
        return self.edge_index.size(1) if self.edge_index is not None else 0

    @property
    def num_node_features(self) -> int:
        return self.x.size(1) if self.x is not None else 0

    def to(self, device):
        return Data(
            x=self.x.to(device) if self.x is not None else None,
            edge_index=self.edge_index.to(device) if self.edge_index is not None else None,
            y=self.y.to(device) if self.y is not None else None,
            time_step=self.time_step.to(device) if self.time_step is not None else None,
            scaler=self.scaler,
        )


def load_elliptic_graph(
    features_path: str,
    classes_path: str,
    edgelist_path: str,
    train_time_step_threshold: int = 34,
) -> tuple[Data, dict]:
    """Carga el dataset Elliptic y construye un objeto de grafo compatible con el notebook."""
    print("Cargando archivos de datos...")

    if features_path.endswith('.parquet'):
        df_features = pd.read_parquet(features_path)
    else:
        df_features = pd.read_csv(features_path, header=None)
        df_features.columns = ['txId', 'time_step'] + [f'feat_{i}' for i in range(df_features.shape[1] - 2)]

    df_classes = pd.read_parquet(classes_path) if classes_path.endswith('.parquet') else pd.read_csv(classes_path)
    df_edges = pd.read_parquet(edgelist_path) if edgelist_path.endswith('.parquet') else pd.read_csv(edgelist_path)

    df_features = df_features.sort_values(by='txId').reset_index(drop=True)
    tx_id_to_idx = {tx_id: idx for idx, tx_id in enumerate(df_features['txId'])}

    class_map = {'1': 1, '2': 0, 'unknown': -1}
    df_classes['target'] = df_classes['class'].map(class_map).fillna(-1).astype(int)

    df_nodes = pd.merge(df_features, df_classes[['txId', 'target']], on='txId', how='left')

    time_steps = torch.tensor(df_nodes['time_step'].values, dtype=torch.long)
    y = torch.tensor(df_nodes['target'].values, dtype=torch.long)

    feature_cols = [col for col in df_nodes.columns if col not in ['txId', 'time_step', 'target', 'class']]
    feature_values = df_nodes[feature_cols].fillna(0.0).to_numpy(dtype='float32')

    train_mask = df_nodes['time_step'].to_numpy() <= train_time_step_threshold
    scaler = StandardScaler()
    scaler.fit(feature_values[train_mask])
    x_scaled = scaler.transform(feature_values).astype('float32')
    x = torch.tensor(x_scaled, dtype=torch.float32)

    valid_edges = df_edges[
        df_edges['txId1'].isin(tx_id_to_idx) & df_edges['txId2'].isin(tx_id_to_idx)
    ]

    src = valid_edges['txId1'].map(tx_id_to_idx).astype(int).to_numpy()
    dst = valid_edges['txId2'].map(tx_id_to_idx).astype(int).to_numpy()
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y, time_step=time_steps, scaler=scaler)

    print("Grafo construido con éxito:")
    print(f"   - Nodos (N): {data.num_nodes}")
    print(f"   - Aristas (E): {data.num_edges}")
    print(f"   - Dimensión de características: {data.num_node_features}")
    print(f"   - Etiquetados (Lícitos/Ilícitos): {(data.y != -1).sum().item()}")
    print(f"   - Escalado aplicado con StandardScaler sobre train_time_step <= {train_time_step_threshold}")

    return data, tx_id_to_idx