import torch
import torch.nn as nn
import torch.nn.functional as F


def build_normalized_adjacency(edge_index: torch.Tensor, num_nodes: int, device: torch.device | None = None) -> torch.Tensor:
    if edge_index is None or edge_index.numel() == 0:
        return None

    if device is None:
        device = edge_index.device

    edge_index = edge_index.to(device)
    edge_index_bidir = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    row, col = edge_index_bidir

    deg = torch.bincount(row, minlength=num_nodes).to(dtype=torch.float32)
    inv_sqrt_deg = deg.pow(-0.5).clamp(min=1e-12)
    weights = inv_sqrt_deg[row] * inv_sqrt_deg[col]

    self_loops = torch.arange(num_nodes, device=device)
    edge_index_full = torch.cat([edge_index_bidir, torch.stack([self_loops, self_loops])], dim=1)
    weights_full = torch.cat([weights, torch.ones(num_nodes, device=device, dtype=torch.float32)])

    return torch.sparse_coo_tensor(
        edge_index_full,
        weights_full,
        (num_nodes, num_nodes),
        device=device,
        dtype=torch.float32,
    )


class GCNConv(nn.Module):
    """GCN layer implementada con operaciones sparse de PyTorch para no depender de torch_geometric."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected node features of shape [num_nodes, num_features], got {tuple(x.shape)}")

        num_nodes = x.size(0)
        if edge_index is None or edge_index.numel() == 0:
            return x @ self.weight + (self.bias if self.bias is not None else 0)

        if adj is None:
            adj = build_normalized_adjacency(edge_index, num_nodes, x.device)

        if adj is not None:
            x_agg = torch.sparse.mm(adj, x)
        else:
            x_agg = x

        out = x_agg @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class GATConv(nn.Module):
    """Versión ligera de GAT basada en atención por aristas para entornos sin torch_geometric."""

    def __init__(self, in_channels: int, out_channels: int, heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.empty(in_channels, out_channels * heads))
        self.attn_src = nn.Parameter(torch.empty(heads, out_channels))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_channels))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return x @ self.weight

        x_proj = self.dropout(x @ self.weight)
        x_proj = x_proj.view(-1, self.heads, self.out_channels)

        src = x_proj[edge_index[0]]
        dst = x_proj[edge_index[1]]
        attn_logits = torch.sum(src * self.attn_src.unsqueeze(0), dim=-1) + torch.sum(dst * self.attn_dst.unsqueeze(0), dim=-1)
        attn_logits = F.leaky_relu(attn_logits, 0.2)

        # Normalización por nodo destino
        node_index = edge_index[1]
        max_logits = torch.zeros((x.size(0), self.heads), device=x.device)
        max_logits[node_index] = attn_logits.max(dim=0).values
        attn_logits = attn_logits - max_logits[node_index]
        attn_weights = torch.softmax(attn_logits, dim=0)

        out = torch.einsum('ehi,eh->hi', dst, attn_weights)
        return out.reshape(-1, self.heads * self.out_channels)


class GCNClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 2, dropout: float = 0.3):
        super(GCNClassifier, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.bn2 = nn.BatchNorm1d(hidden_channels // 2)

        self.classifier = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = dropout
        self._adj_cache = None
        self._adj_cache_num_nodes = None

    def _get_adj(self, edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor | None:
        if self._adj_cache is None or self._adj_cache.device != device or self._adj_cache_num_nodes != num_nodes:
            self._adj_cache = build_normalized_adjacency(edge_index, num_nodes, device)
            self._adj_cache_num_nodes = num_nodes
        return self._adj_cache

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        if adj is None:
            adj = self._get_adj(edge_index, x.size(0), x.device)

        x = self.conv1(x, edge_index, adj=adj)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, adj=adj)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        logits = self.classifier(x)
        return logits


class GATClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 2, heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels // 2, heads=1, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_channels // 2)
        self.classifier = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)