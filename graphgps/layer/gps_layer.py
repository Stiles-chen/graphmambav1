from __future__ import annotations

import warnings
from typing import List

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
        edge_scan_order: str = 'dfs',
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

        self.edge_scan_order = edge_scan_order
        if self.edge_scan_order not in ['default', 'dfs']:
            raise ValueError(f"Unsupported edge_scan_order: {self.edge_scan_order}")

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
        """Fast validation: True iff `order` is a permutation of [0..n-1].

        Note: This is performance-critical (called inside forward); avoid Python
        loops / `.tolist()`.
        """
        if not torch.is_tensor(order):
            return False
        order = order.view(-1)
        if n == 0:
            return order.numel() == 0
        if order.numel() != n:
            return False
        if order.dtype != torch.long:
            # Still allow int tensors.
            if not order.dtype.is_floating_point:
                order = order.to(torch.long)
            else:
                return False
        # Range check.
        if int(order.min()) < 0 or int(order.max()) >= n:
            return False
        # Uniqueness check.
        # Sort+compare is usually faster than `unique` for this use case.
        sorted_vals = torch.sort(order).values
        target = torch.arange(n, device=sorted_vals.device, dtype=sorted_vals.dtype)
        return bool((sorted_vals == target).all())

    def forward(self, batch):
        # Defensive sanitize.
        h = torch.nan_to_num(batch.x, nan=0.0, posinf=1e4, neginf=-1e4)
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            batch.edge_attr = torch.nan_to_num(batch.edge_attr, nan=0.0, posinf=1e4, neginf=-1e4)

        h_in1 = h
        h_out_list = []

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
        # Be defensive: if NaN/Inf appears, gated fusion can propagate it into
        # subsequent layers and lead to hard-to-debug CUDA asserts later.
        h_node = torch.nan_to_num(h_node, nan=0.0, posinf=1e4, neginf=-1e4)
        h_edge = torch.nan_to_num(h_edge, nan=0.0, posinf=1e4, neginf=-1e4)

        if h_node.shape != h_edge.shape:
            raise ValueError(f"Node/Edge scan output shape mismatch: {h_node.shape} vs {h_edge.shape}")

        mode = getattr(self, 'edge_node_fusion_mode', 'fixed')
        w = float(getattr(self, 'edge_node_weight', 0.5))
        w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)

        # For regression tasks (like peptides-structural), apply stronger numerical stabilization
        # Check if this is likely a regression task by examining tensor magnitudes
        h_node_max = h_node.abs().max()
        h_edge_max = h_edge.abs().max()
        is_regression_like = (h_node_max > 10.0) or (h_edge_max > 10.0)  # Heuristic threshold

        if is_regression_like:
            # Apply gradient clipping before fusion for regression tasks
            clip_value = 5.0  # Conservative clipping
            h_node = torch.clamp(h_node, -clip_value, clip_value)
            h_edge = torch.clamp(h_edge, -clip_value, clip_value)

        # Normalize both outputs to improve numerical stability when fusing
        # This prevents one branch from dominating due to magnitude differences
        h_node_norm = torch.norm(h_node, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        h_edge_norm = torch.norm(h_edge, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        h_node_normalized = h_node / h_node_norm
        h_edge_normalized = h_edge / h_edge_norm
        
        # Use the average norm for reconstruction
        avg_norm = (h_node_norm + h_edge_norm) / 2.0

        if mode == 'fixed':
            fused = w * h_node_normalized + (1.0 - w) * h_edge_normalized
            result = fused * avg_norm
        elif mode == 'gated':
            if self.edge_node_gate_layer is None:
                raise RuntimeError("edge_node_gate_layer is not initialized (mode='gated').")
            combined = torch.cat([h_node_normalized, h_edge_normalized], dim=-1)
            gate = torch.sigmoid(self.edge_node_gate_layer(combined))
            fused = gate * h_node_normalized + (1.0 - gate) * h_edge_normalized
            result = fused * avg_norm
        elif mode == 'concat':
            if self.edge_node_concat_proj is None:
                raise RuntimeError("edge_node_concat_proj is not initialized (mode='concat').")
            combined = torch.cat([h_node_normalized, h_edge_normalized], dim=-1)
            fused = self.edge_node_concat_proj(combined)
            # Restore magnitude after projection
            fused_norm = torch.norm(fused, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            result = fused / fused_norm * avg_norm
        else:
            raise ValueError(f"Unsupported edge_node_fusion_mode: {mode}")

        # Final safety check for regression tasks
        if is_regression_like:
            result = torch.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)

        return result

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
                h_ind_perm = self._dfs_node_order(batch.edge_index, batch.batch, h.size(0))
            h_ind_perm = self._sanitize_node_order(h_ind_perm, h.size(0), device=torch.device('cpu'))
            h_ind_perm = h_ind_perm.to(h.device, non_blocking=True)
            h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
            flat_fwd = self.self_attn(h_dense)[mask]
            flat_fwd = torch.nan_to_num(flat_fwd, nan=0.0, posinf=1e4, neginf=-1e4)

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
            h_fwd = torch.nan_to_num(h_fwd, nan=0.0, posinf=1e4, neginf=-1e4)
            if not self.enable_reverse_mamba:
                return h_fwd

            h_ind_perm_rev = torch.flip(h_ind_perm, dims=[0]).contiguous()
            h_dense_rev, mask_rev = to_dense_batch(h[h_ind_perm_rev], batch.batch[h_ind_perm_rev])
            flat_rev = self.self_attn_reverse(h_dense_rev)[mask_rev]
            flat_rev = torch.nan_to_num(flat_rev, nan=0.0, posinf=1e4, neginf=-1e4)
            if flat_rev.size(0) != num_nodes:
                return h_fwd
            h_rev = flat_rev.new_empty((num_nodes, flat_rev.size(-1)))
            h_rev[h_ind_perm_rev] = flat_rev
            h_rev = torch.nan_to_num(h_rev, nan=0.0, posinf=1e4, neginf=-1e4)
            return self._fuse_mamba_outputs(h_fwd, h_rev)

        if self.global_model_type == 'Mamba_Hybrid_Degree_Noise':
            deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
            if batch.split == 'train':
                deg_noise = torch.rand_like(deg)
                h_ind_perm = lexsort([deg + deg_noise, batch.batch])
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                result = self._apply_mamba_scan_with_permutation(
                    h, batch.batch, h_ind_perm, h_dense, mask, h_ind_perm_reverse, batch
                )
                return torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
            mamba_arr = []
            for _ in range(5):
                deg_noise = torch.rand_like(deg)
                h_ind_perm = lexsort([deg + deg_noise, batch.batch])
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                result = self._apply_mamba_scan_with_permutation(
                    h, batch.batch, h_ind_perm, h_dense, mask, h_ind_perm_reverse, batch
                )
                mamba_arr.append(torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4))
            return torch.stack(mamba_arr).mean(dim=0)

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
                    h_dense = torch.nan_to_num(h_dense, nan=0.0, posinf=1e4, neginf=-1e4)
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
                    h_dense = torch.nan_to_num(h_dense, nan=0.0, posinf=1e4, neginf=-1e4)
                    indices_arr.append(h_ind_perm_i)
                    emb_arr.append(h_dense)
                h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                mamba_arr.append(torch.cat(emb_arr)[h_ind_perm_reverse])
            return torch.stack(mamba_arr).mean(dim=0)

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

        # Ensure basic invariants early to avoid device-side asserts from
        # advanced indexing kernels.
        if edge_index.dtype != torch.long:
            edge_index = edge_index.to(torch.long)
            batch.edge_index = edge_index

        # Keep edge_index and edge_attr consistent if something upstream
        # accidentally changed one but not the other.
        if batch.edge_attr is None:
            raise ValueError("edge scan requires batch.edge_attr.")
        if batch.edge_attr.size(0) != edge_index.size(1):
            min_len = min(int(batch.edge_attr.size(0)), int(edge_index.size(1)))
            if not hasattr(self, '_warned_edge_attr_mismatch'):
                setattr(self, '_warned_edge_attr_mismatch', True)
                warnings.warn(
                    f"edge_attr/edge_index length mismatch: edge_attr={batch.edge_attr.size(0)} vs "
                    f"num_edges={edge_index.size(1)}. Truncating to {min_len}.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            batch.edge_attr = batch.edge_attr[:min_len]
            edge_index = edge_index[:, :min_len]
            batch.edge_index = edge_index
            if hasattr(batch, 'dfs_edge_order'):
                delattr(batch, 'dfs_edge_order')
            if hasattr(batch, 'dfs_edge_rank'):
                delattr(batch, 'dfs_edge_rank')

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
        # After filtering, edge_index must be in range; if not, fall back to
        # filtering again (prevents torch_scatter / indexing device asserts).
        if num_edges > 0:
            src0 = edge_index[0]
            dst0 = edge_index[1]
            in_range = (src0 >= 0) & (src0 < num_nodes) & (dst0 >= 0) & (dst0 < num_nodes)
            if not bool(in_range.all()):
                edge_index = edge_index[:, in_range]
                batch.edge_index = edge_index
                batch.edge_attr = batch.edge_attr[in_range]
                num_edges = int(edge_index.size(1))
                if hasattr(batch, 'dfs_edge_order'):
                    delattr(batch, 'dfs_edge_order')
                if hasattr(batch, 'dfs_edge_rank'):
                    delattr(batch, 'dfs_edge_rank')

        edge_batch_all = batch.batch[edge_index[0]] if num_edges > 0 else batch.batch.new_empty((0,), dtype=torch.long)

        if self.edge_scan_order == 'default':
            edge_order = torch.arange(num_edges, device=edge_index.device)
        else:  # 'dfs'
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
                dfs_edge_order = getattr(batch, 'dfs_edge_order', None)
                if dfs_edge_order is not None and self._is_valid_permutation_1d(dfs_edge_order, num_edges):
                    edge_order = dfs_edge_order
                else:
                    edge_order = self._dfs_edge_order(edge_index, batch.batch, num_nodes)
                # Keep sanitation on-device for performance; fall back internally if invalid.
                edge_order = self._sanitize_edge_order(edge_order, num_edges, device=edge_index.device)

        # Final safety: ensure edge_order is within bounds.
        if num_edges > 0:
            if edge_order.dtype != torch.long:
                edge_order = edge_order.to(torch.long)
            if edge_order.numel() != num_edges or int(edge_order.min()) < 0 or int(edge_order.max()) >= num_edges:
                edge_order = self._sanitize_edge_order(edge_order, num_edges, device=edge_index.device)

        edge_feat_perm = edge_feat[edge_order]
        edge_batch = edge_batch_all[edge_order]
        edge_dense, edge_mask = to_dense_batch(edge_feat_perm, edge_batch)
        edge_out = self.self_attn(edge_dense)[edge_mask]
        edge_out = torch.nan_to_num(edge_out, nan=0.0, posinf=1e4, neginf=-1e4)
        edge_out = edge_out[torch.argsort(edge_order)]
        batch.edge_attr = edge_out

        src, dst = edge_index[0], edge_index[1]
        node_msg = scatter_mean_fallback(edge_out, src, dim_size=num_nodes)
        node_msg = node_msg + scatter_mean_fallback(edge_out, dst, dim_size=num_nodes)
        # Scale down to prevent gradient explosion when combining multiple sources
        node_msg = node_msg / 2.0
        node_msg = torch.nan_to_num(node_msg, nan=0.0, posinf=1e4, neginf=-1e4)
        return node_msg

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
        # Slow-path: build a valid permutation.
        # Prefer tensor ops; avoid Python per-element loops.
        edge_order = edge_order.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        edge_order = edge_order[(edge_order >= 0) & (edge_order < num_edges)]
        if edge_order.numel() == 0:
            return torch.arange(num_edges, device=device, dtype=torch.long)
        # Remove duplicates while preserving first occurrence.
        # Use CPU only when absolutely necessary.
        try:
            uniq, inv = torch.unique(edge_order, return_inverse=True)
            # `torch.unique` does not preserve order; emulate stable unique via
            # first occurrence positions.
            first_pos = torch.full((uniq.numel(),), edge_order.numel(), device=device, dtype=torch.long)
            pos = torch.arange(edge_order.numel(), device=device, dtype=torch.long)
            first_pos.scatter_reduce_(0, inv, pos, reduce='amin', include_self=True)
            keep = torch.argsort(first_pos)
            stable_uniq = uniq[keep]
        except Exception:
            stable_uniq = edge_order.detach().cpu().unique(sorted=False).to(device=device)

        seen = torch.zeros(num_edges, device=device, dtype=torch.bool)
        seen[stable_uniq] = True
        missing = (~seen).nonzero().view(-1)
        return torch.cat([stable_uniq, missing], dim=0)[:num_edges]

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
        node_order = node_order.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
        node_order = node_order[(node_order >= 0) & (node_order < num_nodes)]
        if node_order.numel() == 0:
            return torch.arange(num_nodes, device=device, dtype=torch.long)
        try:
            uniq, inv = torch.unique(node_order, return_inverse=True)
            first_pos = torch.full((uniq.numel(),), node_order.numel(), device=device, dtype=torch.long)
            pos = torch.arange(node_order.numel(), device=device, dtype=torch.long)
            first_pos.scatter_reduce_(0, inv, pos, reduce='amin', include_self=True)
            keep = torch.argsort(first_pos)
            stable_uniq = uniq[keep]
        except Exception:
            stable_uniq = node_order.detach().cpu().unique(sorted=False).to(device=device)

        seen = torch.zeros(num_nodes, device=device, dtype=torch.bool)
        seen[stable_uniq] = True
        missing = (~seen).nonzero().view(-1)
        return torch.cat([stable_uniq, missing], dim=0)[:num_nodes]

    def _dfs_node_order(self, edge_index: torch.Tensor, node_batch: torch.Tensor, num_nodes: int) -> torch.Tensor:
        edge_index_cpu = edge_index.detach().cpu()
        node_batch_cpu = node_batch.detach().cpu()
        src = edge_index_cpu[0].tolist()
        dst = edge_index_cpu[1].tolist()
        adjacency = [[] for _ in range(num_nodes)]
        for u, v in zip(src, dst):
            if 0 <= u < num_nodes and 0 <= v < num_nodes:
                adjacency[u].append(v)

        order = []
        unique_graphs = torch.unique(node_batch_cpu).tolist()
        for gid in unique_graphs:
            nodes = torch.where(node_batch_cpu == gid)[0].tolist()
            visited = set()
            node_set = set(nodes)
            for start in nodes:
                if start in visited:
                    continue
                stack = [start]
                while stack:
                    cur = stack.pop()
                    if cur in visited:
                        continue
                    visited.add(cur)
                    order.append(cur)
                    for nxt in reversed(adjacency[cur]):
                        if nxt in node_set and nxt not in visited:
                            stack.append(nxt)
            for n in nodes:
                if n not in visited:
                    order.append(n)
        return torch.tensor(order, device=edge_index.device, dtype=torch.long)

    def _dfs_edge_order(self, edge_index: torch.Tensor, node_batch: torch.Tensor, num_nodes: int) -> torch.Tensor:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        edge_ids = list(range(edge_index.size(1)))
        adjacency = [[] for _ in range(num_nodes)]
        for eid, (u, v) in enumerate(zip(src, dst)):
            adjacency[u].append((v, eid))

        order = []
        unique_graphs = torch.unique(node_batch).tolist()
        for gid in unique_graphs:
            nodes = torch.where(node_batch == gid)[0].tolist()
            if not nodes:
                continue
            visited_nodes = set()
            used_edges = set()
            for start in nodes:
                if start in visited_nodes:
                    continue
                stack = [start]
                while stack:
                    cur = stack.pop()
                    if cur in visited_nodes:
                        continue
                    visited_nodes.add(cur)
                    for nxt, eid in reversed(adjacency[cur]):
                        if eid not in used_edges:
                            order.append(eid)
                            used_edges.add(eid)
                        if nxt not in visited_nodes and node_batch[nxt].item() == gid:
                            stack.append(nxt)

        if len(order) < len(edge_ids):
            used = set(order)
            order.extend([eid for eid in edge_ids if eid not in used])
        return torch.tensor(order, device=edge_index.device, dtype=torch.long)

    def extra_repr(self):
        return (
            f"summary: dim_h={self.dim_h}, local_gnn_type={self.local_gnn_type}, "
            f"global_model_type={self.global_model_type}, heads={self.num_heads}"
        )

    def _apply_mamba_bucket_scan(self, h_dense: torch.Tensor, mask: torch.Tensor, batch) -> torch.Tensor:
        """Apply (optionally bidirectional) Mamba on a dense sequence and return flat masked output."""
        h_fwd = self.self_attn(h_dense)
        h_fwd = torch.nan_to_num(h_fwd, nan=0.0, posinf=1e4, neginf=-1e4)
        if not self.enable_reverse_mamba:
            return h_fwd[mask]
        reverse_mamba = getattr(self, 'self_attn_reverse', None)
        if reverse_mamba is None:
            raise ValueError("enable_reverse_mamba=True requires self_attn_reverse")
        h_rev = torch.flip(h_dense, dims=[1])
        h_rev = reverse_mamba(h_rev)
        h_rev = torch.flip(h_rev, dims=[1])
        h_rev = torch.nan_to_num(h_rev, nan=0.0, posinf=1e4, neginf=-1e4)
        h_attn = self._fuse_mamba_outputs(h_fwd, h_rev)
        h_attn = torch.nan_to_num(h_attn, nan=0.0, posinf=1e4, neginf=-1e4)
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
        h_fwd = torch.nan_to_num(h_fwd, nan=0.0, posinf=1e4, neginf=-1e4)
        if not self.enable_reverse_mamba:
            return h_fwd

        # IMPORTANT:
        # A naive `torch.flip(h_ind_perm)` makes `batch_index[h_ind_perm_rev]`
        # typically *decreasing* (graphs appear in reverse order). Some PyG
        # versions/utilities assume the `batch` vector is sorted/non-decreasing
        # and can produce an incorrect mask whose number of True entries is
        # smaller than `num_nodes`. That mismatch then triggers CUDA
        # IndexKernel out-of-bounds when we later index with an inverse
        # permutation of length `num_nodes`.
        #
        # Fix: reverse the node order *within each graph* while keeping graphs
        # in increasing batch id order.
        h_ind_perm_rev = self._reverse_perm_within_batch(h_ind_perm, batch_index)

        h_dense_rev, mask_rev = to_dense_batch(h[h_ind_perm_rev], batch_index[h_ind_perm_rev])
        h_ind_perm_rev_reverse = torch.argsort(h_ind_perm_rev)
        reverse_mamba = getattr(self, 'self_attn_reverse', None)
        if reverse_mamba is None:
            raise ValueError("enable_reverse_mamba=True requires self_attn_reverse")
        h_rev = reverse_mamba(h_dense_rev)[mask_rev][h_ind_perm_rev_reverse]
        h_rev = torch.nan_to_num(h_rev, nan=0.0, posinf=1e4, neginf=-1e4)
        result = self._fuse_mamba_outputs(h_fwd, h_rev)
        return torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)

    @staticmethod
    def _reverse_perm_within_batch(h_ind_perm: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Reverse a node permutation within each graph, preserving graph order.

        Args:
            h_ind_perm: A 1D index tensor (ideally a permutation of [0..N-1]).
            batch_index: The `batch.batch` vector mapping each node -> graph id.

        Returns:
            A new permutation where nodes are reversed within each graph.
        """
        if h_ind_perm.numel() == 0:
            return h_ind_perm
        if h_ind_perm.dtype != torch.long:
            h_ind_perm = h_ind_perm.to(torch.long)

        # Fully vectorized GPU implementation.
        device = h_ind_perm.device
        batch_perm = batch_index[h_ind_perm].to(torch.long)

        # Ensure (mostly) non-decreasing batch ids; if not, stably sort by batch.
        if batch_perm.numel() > 1 and not bool((batch_perm[1:] >= batch_perm[:-1]).all()):
            order = torch.argsort(batch_perm, stable=True)
            h_ind_perm = h_ind_perm[order]
            batch_perm = batch_perm[order]

        n = int(h_ind_perm.numel())
        if n == 0:
            return h_ind_perm

        num_graphs = int(batch_perm.max().item()) + 1
        counts = torch.bincount(batch_perm, minlength=num_graphs)
        offsets = torch.cumsum(counts, dim=0)
        starts = offsets - counts

        pos = torch.arange(n, device=device, dtype=torch.long)
        start_pos = starts[batch_perm]
        end_pos = offsets[batch_perm]
        pos_rev = start_pos + (end_pos - 1 - pos)
        return h_ind_perm[pos_rev]
