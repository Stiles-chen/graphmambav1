from __future__ import annotations

import warnings
from typing import List
from functools import lru_cache
import hashlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import degree, to_dense_batch

from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE
from mamba_ssm import Mamba

# Keep old behavior for reproducibility.
torch.manual_seed(0)

# Global cache for DFS orders to avoid redundant computation across batches
_dfs_cache_dict = {}


def permute_nodes_within_identity(identities: torch.Tensor) -> torch.Tensor:
    unique_identities = torch.unique(identities)
    node_indices = torch.arange(len(identities), device=identities.device)
    masks = identities.unsqueeze(0) == unique_identities.unsqueeze(1)
    permuted_indices = torch.cat(
        [node_indices[mask][torch.randperm(int(mask.sum()), device=identities.device)] for mask in masks]
    )
    return permuted_indices


def sort_rand_gpu(pop_size: int, num_samples: int, neighbours: torch.Tensor) -> torch.Tensor:
    idx_select = torch.argsort(torch.rand(pop_size, device=neighbours.device))[:num_samples]
    return neighbours[idx_select]


def augment_seq(edge_index: torch.Tensor, batch: torch.Tensor, num_k: int = -1):
    unique_batches = torch.unique(batch)
    permuted_indices = []
    mask = []

    for batch_index in unique_batches:
        indices_in_batch = (batch == batch_index).nonzero().squeeze()
        for k in indices_in_batch:
            neighbours = edge_index[1][edge_index[0] == k]
            if num_k > 0 and len(neighbours) > num_k:
                neighbours = sort_rand_gpu(len(neighbours), num_k, neighbours)
            permuted_indices.append(neighbours)
            mask.append(torch.zeros(neighbours.shape, dtype=torch.bool, device=batch.device))
            permuted_indices.append(torch.tensor([k], device=batch.device))
            mask.append(torch.tensor([1], dtype=torch.bool, device=batch.device))

    permuted_indices = torch.cat(permuted_indices)
    mask = torch.cat(mask)
    return permuted_indices.to(device=batch.device), mask.to(device=batch.device)


def lexsort(keys: List[torch.Tensor], dim: int = -1, descending: bool = False) -> torch.Tensor:
    """Stable indirect sort using multiple keys (last key has highest priority)."""
    assert len(keys) >= 1
    out = keys[0].argsort(dim=dim, descending=descending, stable=True)
    for k in keys[1:]:
        index = k.gather(dim, out)
        index = index.argsort(dim=dim, descending=descending, stable=True)
        out = out.gather(dim, index)
    return out


def permute_within_batch(batch: torch.Tensor) -> torch.Tensor:
    unique_batches = torch.unique(batch)
    permuted_indices = []
    for batch_index in unique_batches:
        indices_in_batch = (batch == batch_index).nonzero().squeeze()
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch), device=batch.device)]
        permuted_indices.append(permuted_indices_in_batch)
    return torch.cat(permuted_indices)


def scatter_mean_fallback(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """scatter_mean over dim=0 with a pure-torch fallback (avoids hard dep on torch_scatter)."""
    try:
        from torch_scatter import scatter_mean  # type: ignore

        return scatter_mean(src, index, dim=0, dim_size=dim_size)
    except Exception:
        if src.dim() == 1:
            src_ = src.unsqueeze(-1)
        else:
            src_ = src
        out = src_.new_zeros((dim_size, src_.size(-1)))
        cnt = src_.new_zeros((dim_size, 1))
        out.index_add_(0, index, src_)
        ones = torch.ones((src_.size(0), 1), device=src_.device, dtype=src_.dtype)
        cnt.index_add_(0, index, ones)
        out = out / cnt.clamp(min=1)
        return out if src.dim() > 1 else out.squeeze(-1)


def _get_graph_structure_hash(edge_index: torch.Tensor, node_batch: torch.Tensor, num_nodes: int) -> str:
    """Generate a hash for graph structure to enable cross-batch caching.

    This enables caching DFS orders for the same graph structure even when they
    come in different batches, significantly reducing redundant computation.
    """
    # Convert to CPU for hashing to ensure consistency
    ei = edge_index.detach().cpu()
    nb = node_batch.detach().cpu()

    # Create a unique hash based on graph topology
    # We use a simple approach: hash the sorted edge list + batch structure
    ei_tuple = (tuple(ei[0].tolist()), tuple(ei[1].tolist()), tuple(nb.tolist()), num_nodes)
    hash_obj = hashlib.md5(str(ei_tuple).encode())
    return hash_obj.hexdigest()


def _clear_dfs_cache():
    """Clear the global DFS cache. Useful for memory management."""
    global _dfs_cache_dict
    _dfs_cache_dict.clear()


class GPSLayer(nn.Module):
    """Local MPNN + global sequence model layer.

    This fork supports Mamba node-scan, edge-scan, and a new parallel mode
    ``scan_target='both'`` that runs node-scan and edge-scan in parallel and
    fuses their node-level outputs.
    """

    def __init__(
        self,
        dim_h,
        local_gnn_type,
        global_model_type,
        num_heads,
        pna_degrees=None,
        equivstable_pe=False,
        dropout=0.0,
        attn_dropout=0.0,
        layer_norm=False,
        batch_norm=True,
        bigbird_cfg=None,
        enable_reverse_mamba: bool = False,
        fusion_mode: str = 'fixed',
        fixed_weight: float = 0.5,
        scan_target: str = 'node',
        edge_dim=None,
    ):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.NUM_BUCKETS = 3

        self.enable_reverse_mamba = enable_reverse_mamba
        self.fusion_mode = fusion_mode
        self.fixed_weight = fixed_weight
        self.concat_proj = None
        self.gate_layer = None

        self.scan_target = scan_target
        if self.scan_target not in ['node', 'edge', 'both']:
            raise ValueError(f"Unsupported scan_target: {self.scan_target}")

        # Edge scan input projection.
        if scan_target in ['edge', 'both']:
            if edge_dim is None:
                edge_dim = dim_h
            self.edge_scan_input_proj = nn.Identity() if edge_dim == dim_h else nn.Linear(edge_dim, dim_h)
        else:
            self.edge_scan_input_proj = None

        # Local message-passing model.
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h), nn.ReLU(), Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(
                in_channels=dim_h,
                out_channels=dim_h // num_heads,
                heads=num_heads,
                edge_dim=dim_h,
            )
        elif local_gnn_type == 'PNA':
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(
                dim_h,
                dim_h,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=16,
                towers=1,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
            )
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(
                dim_h,
                dim_h,
                dropout=dropout,
                residual=True,
                equivstable_pe=equivstable_pe,
            )
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global sequence model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = torch.nn.MultiheadAttention(dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(dim=dim_h, heads=num_heads, dropout=self.attn_dropout, causal=False)
        elif global_model_type == 'BigBird':
            if bigbird_cfg is None:
                raise ValueError("bigbird_cfg must be provided when global_model_type='BigBird'")
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        elif 'Mamba' in global_model_type:
            # Keep the original behavior for the many Mamba_* variants.
            suffix = global_model_type.split('_')[-1]
            if suffix == '2':
                self.self_attn = Mamba(d_model=dim_h, d_state=8, d_conv=4, expand=2)
            elif suffix == '4':
                self.self_attn = Mamba(d_model=dim_h, d_state=4, d_conv=4, expand=4)
            elif suffix == 'SmallConv':
                self.self_attn = Mamba(d_model=dim_h, d_state=16, d_conv=2, expand=1)
            elif suffix == 'SmallState':
                self.self_attn = Mamba(d_model=dim_h, d_state=8, d_conv=4, expand=1)
            else:
                self.self_attn = Mamba(d_model=dim_h, d_state=16, d_conv=4, expand=1)
        else:
            raise ValueError(f"Unsupported global x-former model: {global_model_type}")
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.activation = F.relu
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        if self.layer_norm:
            self.norm2 = pygnn.norm.GraphNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

        # Reverse Mamba.
        if self.enable_reverse_mamba and (self.self_attn is not None) and ('Mamba' in global_model_type):
            self.self_attn_reverse = Mamba(d_model=dim_h, d_state=16, d_conv=4, expand=1)
            if self.fusion_mode == 'gated':
                self.gate_layer = nn.Linear(dim_h * 2, dim_h)
            elif self.fusion_mode == 'concat':
                self.concat_proj = nn.Linear(dim_h * 2, dim_h)
        else:
            self.self_attn_reverse = None

        # Node/edge fusion when scan_target == 'both'.
        self.edge_node_fusion_mode = 'fixed'
        self.edge_node_weight = 0.5
        self.edge_node_gate_layer = None
        self.edge_node_concat_proj = None
        if self.scan_target == 'both':
            # Prefer dedicated keys if they exist; otherwise reuse the existing
            # reverse-fusion hyperparameters so users can control behavior
            # without introducing new config fields.
            try:
                from torch_geometric.graphgym.config import cfg as _cfg  # type: ignore

                _gt = getattr(_cfg, 'gt', _cfg)
                self.edge_node_fusion_mode = getattr(_gt, 'edge_node_fusion_mode', getattr(_gt, 'fusion_mode', 'fixed'))
                self.edge_node_weight = float(getattr(_gt, 'edge_node_weight', getattr(_gt, 'fixed_weight', 0.5)))
            except Exception:
                self.edge_node_fusion_mode = self.fusion_mode
                self.edge_node_weight = float(self.fixed_weight)

            if self.edge_node_fusion_mode == 'gated':
                self.edge_node_gate_layer = nn.Linear(dim_h * 2, dim_h)
            elif self.edge_node_fusion_mode == 'concat':
                self.edge_node_concat_proj = nn.Linear(dim_h * 2, dim_h)

    @staticmethod
    def _is_valid_permutation_1d(order: torch.Tensor, n: int) -> bool:
        """CPU-side validation: True iff `order` is a permutation of [0..n-1]."""
        if not torch.is_tensor(order):
            return False
        if n == 0:
            return order.numel() == 0
        if order.numel() != n:
            return False
        try:
            lst = order.detach().view(-1).cpu().tolist()
        except Exception:
            return False
        used = [False] * n
        for v in lst:
            iv = int(v)
            if iv < 0 or iv >= n or used[iv]:
                return False
            used[iv] = True
        return True

    def forward(self, batch):
        # Defensive sanitize.
        h = torch.nan_to_num(batch.x, nan=0.0, posinf=1e4, neginf=-1e4)
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            batch.edge_attr = torch.nan_to_num(batch.edge_attr, nan=0.0, posinf=1e4, neginf=-1e4)

        h_in1 = h
        h_out_list = []
        
        # Pre-compute orderings once to avoid redundant DFS calculations across layers
        # Use global cache with graph structure hashing to enable cross-batch reuse
        if hasattr(batch, 'edge_index') and batch.edge_index.numel() > 0:
            # Pre-compute edge ordering for edge scan
            if self.scan_target in ['edge', 'both'] and not hasattr(batch, '_cached_edge_order'):
                num_edges = int(batch.edge_index.size(1))
                # Use global cache with graph structure hashing
                cache_key = _get_graph_structure_hash(batch.edge_index, batch.batch, h.size(0))
                
                if cache_key not in _dfs_cache_dict:
                    # Cache miss: compute DFS
                    edge_order = self._dfs_edge_order(batch.edge_index, batch.batch, h.size(0))
                    edge_order = self._sanitize_edge_order(edge_order, num_edges, device=torch.device('cpu'))
                    _dfs_cache_dict[cache_key] = edge_order
                else:
                    # Cache hit: reuse DFS order
                    edge_order = _dfs_cache_dict[cache_key]
                
                batch._cached_edge_order = edge_order.to(batch.edge_index.device, non_blocking=True)

            # Pre-compute node ordering for node scan (Mamba_DFS variants)
            if (self.global_model_type == 'Mamba_DFS' or 
                self.global_model_type == 'Mamba_Hybrid_Degree_Noise' or 
                self.global_model_type == 'Mamba_Hybrid_Degree_Noise_Bucket') and not hasattr(batch, '_cached_node_order'):
                dfs_attr = getattr(batch, 'dfs_node_order', None)
                if dfs_attr is None or not self._is_valid_permutation_1d(dfs_attr, h.size(0)):
                    # Use global cache for node ordering too
                    cache_key = _get_graph_structure_hash(batch.edge_index, batch.batch, h.size(0)) + "_node"
                    
                    if cache_key not in _dfs_cache_dict:
                        node_order = self._dfs_node_order(batch.edge_index, batch.batch, h.size(0))
                        node_order = self._sanitize_node_order(node_order, h.size(0), device=torch.device('cpu'))
                        _dfs_cache_dict[cache_key] = node_order
                    else:
                        node_order = _dfs_cache_dict[cache_key]
                    
                    batch._cached_node_order = node_order.to(h.device, non_blocking=True)

        # Local MPNN.
        if self.local_model is not None:
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = batch.pe_EquivStableLapPE if self.equivstable_pe else None
                local_out = self.local_model(
                    Batch(
                        batch=batch,
                        x=h,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                        pe_EquivStableLapPE=es_data,
                    )
                )
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.equivstable_pe:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr, batch.pe_EquivStableLapPE)
                else:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local

            h_local = torch.nan_to_num(h_local, nan=0.0, posinf=1e4, neginf=-1e4)
            if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
                batch.edge_attr = torch.nan_to_num(batch.edge_attr, nan=0.0, posinf=1e4, neginf=-1e4)

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Global sequence model.
        if self.self_attn is not None:
            if self.scan_target == 'edge':
                if 'Mamba' not in self.global_model_type:
                    raise ValueError("scan_target='edge' currently supports only Mamba-based global_model_type.")
                h_attn = self._edge_mamba_scan(batch, h)
            elif self.scan_target == 'node':
                h_attn = self._node_global_scan(batch, h)
            else:  # 'both'
                if 'Mamba' not in self.global_model_type:
                    raise ValueError("scan_target='both' requires a Mamba-based global_model_type.")
                h_node = self._node_global_scan(batch, h)
                h_edge = self._edge_mamba_scan(batch, h)
                h_attn = self._fuse_edge_node_outputs(h_node, h_edge)

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn
            h_attn = torch.nan_to_num(h_attn, nan=0.0, posinf=1e4, neginf=-1e4)
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local/global.
        if len(h_out_list) == 0:
            h = h_in1
        else:
            h = sum(h_out_list)
        h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)

        # Feed Forward block.
        h = h + self._ff_block(h)
        h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _fuse_edge_node_outputs(self, h_node: torch.Tensor, h_edge: torch.Tensor) -> torch.Tensor:
        if h_node.shape != h_edge.shape:
            raise ValueError(f"Node/Edge scan output shape mismatch: {h_node.shape} vs {h_edge.shape}")

        mode = getattr(self, 'edge_node_fusion_mode', 'fixed')
        w = float(getattr(self, 'edge_node_weight', 0.5))
        w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)

        if mode == 'fixed':
            return w * h_node + (1.0 - w) * h_edge
        if mode == 'gated':
            if self.edge_node_gate_layer is None:
                raise RuntimeError("edge_node_gate_layer is not initialized (mode='gated').")
            combined = torch.cat([h_node, h_edge], dim=-1)
            gate = torch.sigmoid(self.edge_node_gate_layer(combined))
            return gate * h_node + (1.0 - gate) * h_edge
        if mode == 'concat':
            if self.edge_node_concat_proj is None:
                raise RuntimeError("edge_node_concat_proj is not initialized (mode='concat').")
            combined = torch.cat([h_node, h_edge], dim=-1)
            out = self.edge_node_concat_proj(combined)
            return out
        raise ValueError(f"Unsupported edge_node_fusion_mode: {mode}")

    def _node_global_scan(self, batch, h: torch.Tensor) -> torch.Tensor:
        """Compute the global (sequence) model output over node embeddings.

        This retains the large collection of Mamba_* ordering variants used by
        the provided configs.
        """
        if self.global_model_type in ['Transformer', 'Performer', 'BigBird', 'Mamba']:
            h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                return self._sa_block(h_dense, None, ~mask)[mask]
            if self.global_model_type == 'Performer':
                return self.self_attn(h_dense, mask=mask)[mask]
            if self.global_model_type == 'BigBird':
                return self.self_attn(h_dense, attention_mask=mask)
            return self.self_attn(h_dense)[mask]

        if self.global_model_type == 'Mamba_DFS':
            dfs_attr = getattr(batch, 'dfs_node_order', None)
            if dfs_attr is not None and self._is_valid_permutation_1d(dfs_attr, h.size(0)):
                h_ind_perm = dfs_attr
            else:
                # Try to use cached node order first
                cached_order = getattr(batch, '_cached_node_order', None)
                if cached_order is not None and cached_order.numel() == h.size(0):
                    h_ind_perm = cached_order
                else:
                    h_ind_perm = self._dfs_node_order(batch.edge_index, batch.batch, h.size(0))
                    h_ind_perm = self._sanitize_node_order(h_ind_perm, h.size(0), device=torch.device('cpu'))
                    h_ind_perm = h_ind_perm.to(h.device, non_blocking=True)
            h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
            flat_fwd = self.self_attn(h_dense)[mask]

            num_nodes = h.size(0)
            if flat_fwd.size(0) != num_nodes:
                if not hasattr(self, '_warned_bad_dense_mask'):
                    setattr(self, '_warned_bad_dense_mask', True)
                    warnings.warn(
                        f"Mamba_DFS: mask sum ({flat_fwd.size(0)}) != num_nodes ({num_nodes}); falling back.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                h_dense0, mask0 = to_dense_batch(h, batch.batch)
                return self.self_attn(h_dense0)[mask0]

            h_fwd = flat_fwd.new_empty((num_nodes, flat_fwd.size(-1)))
            h_fwd[h_ind_perm] = flat_fwd
            if not self.enable_reverse_mamba:
                return h_fwd

            h_ind_perm_rev = torch.flip(h_ind_perm, dims=[0]).contiguous()
            h_dense_rev, mask_rev = to_dense_batch(h[h_ind_perm_rev], batch.batch[h_ind_perm_rev])
            flat_rev = self.self_attn_reverse(h_dense_rev)[mask_rev]
            if flat_rev.size(0) != num_nodes:
                return h_fwd
            h_rev = flat_rev.new_empty((num_nodes, flat_rev.size(-1)))
            h_rev[h_ind_perm_rev] = flat_rev
            return self._fuse_mamba_outputs(h_fwd, h_rev)

        if self.global_model_type == 'Mamba_Hybrid_Degree_Noise':
            deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
            if batch.split == 'train':
                deg_noise = torch.rand_like(deg)
                h_ind_perm = lexsort([deg + deg_noise, batch.batch])
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                return self._apply_mamba_scan_with_permutation(
                    h, batch.batch, h_ind_perm, h_dense, mask, h_ind_perm_reverse, batch
                )
            mamba_arr = []
            for _ in range(5):
                deg_noise = torch.rand_like(deg)
                h_ind_perm = lexsort([deg + deg_noise, batch.batch])
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                mamba_arr.append(
                    self._apply_mamba_scan_with_permutation(
                        h, batch.batch, h_ind_perm, h_dense, mask, h_ind_perm_reverse, batch
                    )
                )
            return sum(mamba_arr) / 5

        if self.global_model_type == 'Mamba_Hybrid_Degree_Noise_Bucket':
            deg_ = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
            if batch.split == 'train':
                deg_noise = torch.rand_like(deg_)
                deg = deg_ + deg_noise
                indices_arr, emb_arr = [], []
                bucket_assign = torch.randint(0, self.NUM_BUCKETS, (deg.numel(),), device=deg.device)
                for i in range(self.NUM_BUCKETS):
                    ind_i = (bucket_assign == i).nonzero().view(-1)
                    h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                    h_ind_perm_i = ind_i[h_ind_perm_sort]
                    h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                    h_dense = self._apply_mamba_bucket_scan(h_dense, mask, batch)
                    indices_arr.append(h_ind_perm_i)
                    emb_arr.append(h_dense)
                h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                return torch.cat(emb_arr)[h_ind_perm_reverse]
            mamba_arr = []
            for _ in range(5):
                deg_noise = torch.rand_like(deg_)
                deg = deg_ + deg_noise
                indices_arr, emb_arr = [], []
                bucket_assign = torch.randint(0, self.NUM_BUCKETS, (deg.numel(),), device=deg.device)
                for i in range(self.NUM_BUCKETS):
                    ind_i = (bucket_assign == i).nonzero().view(-1)
                    h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                    h_ind_perm_i = ind_i[h_ind_perm_sort]
                    h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                    h_dense = self._apply_mamba_bucket_scan(h_dense, mask, batch)
                    indices_arr.append(h_ind_perm_i)
                    emb_arr.append(h_dense)
                h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                mamba_arr.append(torch.cat(emb_arr)[h_ind_perm_reverse])
            return sum(mamba_arr) / 5

        if 'Mamba' in self.global_model_type:
            h_dense, mask = to_dense_batch(h, batch.batch)
            return self.self_attn(h_dense)[mask]

        raise RuntimeError(f"Unexpected {self.global_model_type}")

    def _sa_block(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return x

    def _ff_block(self, x):
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def _fuse_mamba_outputs(self, h_fwd: torch.Tensor, h_rev: torch.Tensor) -> torch.Tensor:
        if h_fwd.shape != h_rev.shape:
            raise ValueError(f"Forward/Reverse Mamba output shape mismatch: {h_fwd.shape} vs {h_rev.shape}")

        if self.fusion_mode == 'fixed':
            return self.fixed_weight * h_fwd + (1 - self.fixed_weight) * h_rev
        if self.fusion_mode == 'gated':
            if self.gate_layer is None:
                raise RuntimeError("gate_layer is not initialized (mode='gated').")
            combined = torch.cat([h_fwd, h_rev], dim=-1)
            gate = torch.sigmoid(self.gate_layer(combined))
            return gate * h_fwd + (1 - gate) * h_rev
        if self.fusion_mode == 'concat':
            if self.concat_proj is None:
                raise RuntimeError("concat_proj is not initialized (mode='concat').")
            combined = torch.cat([h_fwd, h_rev], dim=-1)
            return self.concat_proj(combined)
        raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

    def _edge_mamba_scan(self, batch, h: torch.Tensor) -> torch.Tensor:
        if 'Mamba' not in self.global_model_type:
            raise ValueError("edge scan currently supports only Mamba-based global_model_type.")
        if not hasattr(batch, 'edge_attr') or batch.edge_attr is None:
            raise ValueError("edge scan requires batch.edge_attr.")
        if self.edge_scan_input_proj is None:
            raise RuntimeError("edge_scan_input_proj is not initialized.")

        edge_index = batch.edge_index
        num_nodes = h.size(0)
        valid_edge_mask = (
            (edge_index[0] >= 0)
            & (edge_index[0] < num_nodes)
            & (edge_index[1] >= 0)
            & (edge_index[1] < num_nodes)
        )
        if not bool(valid_edge_mask.all()):
            edge_index = edge_index[:, valid_edge_mask]
            batch.edge_index = edge_index
            batch.edge_attr = batch.edge_attr[valid_edge_mask]
            if hasattr(batch, 'dfs_edge_order'):
                delattr(batch, 'dfs_edge_order')
            if hasattr(batch, 'dfs_edge_rank'):
                delattr(batch, 'dfs_edge_rank')

        edge_attr = batch.edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if not torch.is_floating_point(edge_attr):
            edge_attr = edge_attr.to(dtype=h.dtype)
        elif edge_attr.dtype != h.dtype:
            edge_attr = edge_attr.to(dtype=h.dtype)

        edge_feat = self.edge_scan_input_proj(edge_attr)
        edge_feat = torch.nan_to_num(edge_feat, nan=0.0, posinf=1e4, neginf=-1e4)

        num_edges = int(edge_index.size(1))
        edge_batch_all = batch.batch[edge_index[0]]

        dfs_edge_rank = getattr(batch, 'dfs_edge_rank', None)
        if torch.is_tensor(dfs_edge_rank) and dfs_edge_rank.numel() == num_edges:
            rank = dfs_edge_rank.to(device=edge_batch_all.device, non_blocking=True).view(-1)
            if rank.dtype != torch.long:
                rank = rank.to(torch.long)
            batch_id = edge_batch_all.to(torch.long)
            stride = (rank.max() + 1).to(torch.long)
            key = batch_id * stride + rank
            edge_order = torch.argsort(key)
        else:
            # Try to use cached edge order (computed once in forward() to avoid redundant DFS)
            dfs_edge_order = getattr(batch, 'dfs_edge_order', None)
            if dfs_edge_order is not None and self._is_valid_permutation_1d(dfs_edge_order, num_edges):
                edge_order = dfs_edge_order
            else:
                # Check for cached edge order first
                cached_order = getattr(batch, '_cached_edge_order', None)
                if cached_order is not None and cached_order.numel() == num_edges:
                    edge_order = cached_order
                else:
                    edge_order = self._dfs_edge_order(edge_index, batch.batch, num_nodes)
                    edge_order = self._sanitize_edge_order(edge_order, num_edges, device=torch.device('cpu'))
                    edge_order = edge_order.to(edge_index.device, non_blocking=True)

        edge_feat_perm = edge_feat[edge_order]
        edge_batch = edge_batch_all[edge_order]
        edge_dense, edge_mask = to_dense_batch(edge_feat_perm, edge_batch)
        edge_out = self.self_attn(edge_dense)[edge_mask]
        edge_out = edge_out[torch.argsort(edge_order)]
        batch.edge_attr = edge_out

        src, dst = edge_index[0], edge_index[1]
        node_msg = scatter_mean_fallback(edge_out, src, dim_size=num_nodes)
        # Accumulate messages from destination nodes as well
        dst_msg = scatter_mean_fallback(edge_out, dst, dim_size=num_nodes)
        node_msg = node_msg + dst_msg
        return torch.nan_to_num(node_msg, nan=0.0, posinf=1e4, neginf=-1e4)

    def _sanitize_edge_order(self, edge_order, num_edges: int, device: torch.device) -> torch.Tensor:
        if num_edges == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if not torch.is_tensor(edge_order):
            edge_order = torch.tensor(edge_order, dtype=torch.long)
        edge_order = edge_order.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        if edge_order.numel() == num_edges:
            in_range = (edge_order >= 0) & (edge_order < num_edges)
            if bool(in_range.all()):
                seen = torch.zeros(num_edges, dtype=torch.bool, device=device)
                seen[edge_order] = True
                if bool(seen.all()):
                    return edge_order

        if not hasattr(self, "_warned_invalid_edge_order"):
            setattr(self, "_warned_invalid_edge_order", True)
            warnings.warn(
                "Invalid edge_order detected; falling back to a sanitized permutation.",
                RuntimeWarning,
                stacklevel=2,
            )
        edge_list = edge_order.detach().cpu().tolist()
        used = [False] * num_edges
        out = []
        for v in edge_list:
            iv = int(v)
            if 0 <= iv < num_edges and not used[iv]:
                used[iv] = True
                out.append(iv)
        if len(out) < num_edges:
            out.extend([i for i, u in enumerate(used) if not u])
        return torch.tensor(out, device=device, dtype=torch.long)

    def _sanitize_node_order(self, node_order, num_nodes: int, device: torch.device) -> torch.Tensor:
        if num_nodes == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if not torch.is_tensor(node_order):
            node_order = torch.tensor(node_order, dtype=torch.long)
        node_order = node_order.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        if node_order.numel() == num_nodes:
            in_range = (node_order >= 0) & (node_order < num_nodes)
            if bool(in_range.all()):
                seen = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                seen[node_order] = True
                if bool(seen.all()):
                    return node_order

        if not hasattr(self, "_warned_invalid_node_order"):
            setattr(self, "_warned_invalid_node_order", True)
            warnings.warn(
                "Invalid node_order detected; falling back to a sanitized permutation.",
                RuntimeWarning,
                stacklevel=2,
            )
        node_list = node_order.detach().cpu().tolist()
        used = [False] * num_nodes
        out = []
        for v in node_list:
            iv = int(v)
            if 0 <= iv < num_nodes and not used[iv]:
                used[iv] = True
                out.append(iv)
        if len(out) < num_nodes:
            out.extend([i for i, u in enumerate(used) if not u])
        return torch.tensor(out, device=device, dtype=torch.long)

    def _dfs_node_order(self, edge_index: torch.Tensor, node_batch: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute DFS node order with optimized torch operations to minimize GPU-CPU transfers."""
        edge_index_cpu = edge_index.detach().cpu()
        node_batch_cpu = node_batch.detach().cpu()
        src = edge_index_cpu[0]
        dst = edge_index_cpu[1]

        # Build adjacency list more efficiently using torch operations
        # Use offsets instead of tolist() for faster access
        adjacency = [[] for _ in range(num_nodes)]
        src_list = src.tolist()  # Only converting once here
        dst_list = dst.tolist()  # Only converting once here
        
        for u, v in zip(src_list, dst_list):
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                adjacency[u].append(v)

        order = []
        visited = set()
        # Use torch operations to get unique graphs instead of converting to list multiple times
        unique_graphs_tensor = torch.unique(node_batch_cpu)

        for gid_tensor in unique_graphs_tensor:
            gid = int(gid_tensor.item())  # Convert once
            nodes_tensor = torch.where(node_batch_cpu == gid)[0]
            nodes = nodes_tensor.tolist()  # Only convert once per graph
            node_set = set(nodes)
            
            for start in nodes:
                if start in visited:
                    continue
                # Iterative DFS to avoid stack overflow and improve performance
                stack = [start]
                traverse_order = []
                
                while stack:
                    cur = stack[-1]
                    if cur in visited:
                        stack.pop()
                        continue
                    visited.add(cur)
                    traverse_order.append(cur)
                    
                    # Add neighbors in reverse order for consistent traversal
                    neighbors = [nxt for nxt in adjacency[cur] 
                                if nxt in node_set and nxt not in visited]
                    if neighbors:
                        stack.extend(reversed(neighbors))
                    else:
                        stack.pop()
                
                order.extend(traverse_order)
            
            # Add unvisited nodes in this graph
            for n in nodes:
                if n not in visited:
                    order.append(n)
                    visited.add(n)
                    
        return torch.tensor(order, device=edge_index.device, dtype=torch.long)

    def _dfs_edge_order(self, edge_index: torch.Tensor, node_batch: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute DFS edge order with optimized torch operations.
        
        Key optimization: Minimize GPU-CPU transfers by converting to list only once per component,
        instead of repeatedly throughout the DFS traversal.
        """
        edge_index_cpu = edge_index.detach().cpu()
        # Convert to lists ONCE to avoid repeated GPU-CPU transfers
        src_list = edge_index_cpu[0].tolist()
        dst_list = edge_index_cpu[1].tolist()
        edge_ids = list(range(edge_index.size(1)))
        
        # Build adjacency list with edge indices for efficient traversal
        adjacency = [[] for _ in range(num_nodes)]
        for eid, (u, v) in enumerate(zip(src_list, dst_list)):
            adjacency[u].append((v, eid))

        order = []
        visited_edges = set()
        node_batch_cpu = node_batch.detach().cpu()
        
        # Get unique graphs once using torch, then convert to list once
        unique_graphs_tensor = torch.unique(node_batch_cpu)
        unique_graphs = unique_graphs_tensor.tolist()
        
        for gid in unique_graphs:
            # Get nodes for this graph - convert once
            nodes_tensor = torch.where(node_batch_cpu == gid)[0]
            nodes = nodes_tensor.tolist()
            
            if not nodes:
                continue
            visited_nodes = set()
            
            for start in nodes:
                if start in visited_nodes:
                    continue
                # Iterative DFS for node traversal, collecting edges
                stack = [start]
                
                while stack:
                    cur = stack[-1]
                    if cur in visited_nodes:
                        stack.pop()
                        continue
                    visited_nodes.add(cur)
                    
                    # Process all edges from current node
                    unvisited_neighbors = []
                    for nxt, eid in adjacency[cur]:
                        if eid not in visited_edges:
                            order.append(eid)
                            visited_edges.add(eid)
                        # Check if next node is unvisited and in the same graph
                        # Use cached node_batch_cpu to avoid .item() calls in loops
                        if nxt not in visited_nodes and node_batch_cpu[nxt].item() == gid:
                            unvisited_neighbors.append(nxt)
                    
                    if unvisited_neighbors:
                        stack.extend(reversed(unvisited_neighbors))
                    else:
                        stack.pop()

        # Add remaining edges that weren't visited
        if len(visited_edges) < len(edge_ids):
            for eid in edge_ids:
                if eid not in visited_edges:
                    order.append(eid)
                    
        return torch.tensor(order, device=edge_index.device, dtype=torch.long)

    def extra_repr(self):
        return (
            f"summary: dim_h={self.dim_h}, local_gnn_type={self.local_gnn_type}, "
            f"global_model_type={self.global_model_type}, heads={self.num_heads}"
        )

    def _apply_mamba_bucket_scan(self, h_dense: torch.Tensor, mask: torch.Tensor, batch) -> torch.Tensor:
        """Apply (optionally bidirectional) Mamba on a dense sequence and return flat masked output."""
        h_fwd = self.self_attn(h_dense)
        if not self.enable_reverse_mamba:
            return h_fwd[mask]
        reverse_mamba = getattr(self, 'self_attn_reverse', None)
        if reverse_mamba is None:
            raise ValueError("enable_reverse_mamba=True requires self_attn_reverse")
        h_rev = torch.flip(h_dense, dims=[1])
        h_rev = reverse_mamba(h_rev)
        h_rev = torch.flip(h_rev, dims=[1])
        h_attn = self._fuse_mamba_outputs(h_fwd, h_rev)
        return h_attn[mask]

    def _apply_mamba_scan_with_permutation(
        self,
        h: torch.Tensor,
        batch_index: torch.Tensor,
        h_ind_perm: torch.Tensor,
        h_dense: torch.Tensor,
        mask: torch.Tensor,
        h_ind_perm_reverse: torch.Tensor,
        batch,
    ) -> torch.Tensor:
        """Run Mamba on a permuted dense sequence (and optionally reverse) and map back."""
        h_fwd = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
        if not self.enable_reverse_mamba:
            return h_fwd
        h_ind_perm_rev = torch.flip(h_ind_perm, dims=[0])
        h_dense_rev, mask_rev = to_dense_batch(h[h_ind_perm_rev], batch_index[h_ind_perm_rev])
        h_ind_perm_rev_reverse = torch.argsort(h_ind_perm_rev)
        reverse_mamba = getattr(self, 'self_attn_reverse', None)
        if reverse_mamba is None:
            raise ValueError("enable_reverse_mamba=True requires self_attn_reverse")
        h_rev = reverse_mamba(h_dense_rev)[mask_rev][h_ind_perm_rev_reverse]
        return self._fuse_mamba_outputs(h_fwd, h_rev)