import numpy as np
import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvESLapPE
from graphgps.layer.bigbird_layer import SingleBigBirdLayer
from mamba_ssm import Mamba

from torch_geometric.utils import degree, sort_edge_index
from typing import List

import numpy as np
import torch
from torch import Tensor

def permute_nodes_within_identity(identities):
    unique_identities, inverse_indices = torch.unique(identities, return_inverse=True)
    node_indices = torch.arange(len(identities), device=identities.device)

    masks = identities.unsqueeze(0) == unique_identities.unsqueeze(1)

    # Generate random indices within each identity group using torch.randint
    permuted_indices = torch.cat([
        node_indices[mask][torch.randperm(mask.sum(), device=identities.device)] for mask in masks
    ])
    return permuted_indices

def sort_rand_gpu(pop_size, num_samples, neighbours):
    # Randomly generate indices and select num_samples in neighbours
    idx_select = torch.argsort(torch.rand(pop_size, device=neighbours.device))[:num_samples]
    neighbours = neighbours[idx_select]
    return neighbours

def augment_seq(edge_index, batch, num_k = -1):
    unique_batches = torch.unique(batch)
    # Initialize list to store permuted indices
    permuted_indices = []
    mask = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()
        for k in indices_in_batch:
            neighbours = edge_index[1][edge_index[0]==k]
            if num_k > 0 and len(neighbours) > num_k:
                neighbours = sort_rand_gpu(len(neighbours), num_k, neighbours)
            permuted_indices.append(neighbours)
            mask.append(torch.zeros(neighbours.shape, dtype=torch.bool, device=batch.device))
            permuted_indices.append(torch.tensor([k], device=batch.device))
            mask.append(torch.tensor([1], dtype=torch.bool, device=batch.device))
    permuted_indices = torch.cat(permuted_indices)
    mask = torch.cat(mask)
    return permuted_indices.to(device=batch.device), mask.to(device=batch.device)

def lexsort(
    keys: List[Tensor],
    dim: int = -1,
    descending: bool = False,
) -> Tensor:
    r"""Performs an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, returns an array of integer indices that
    describe their sort order.
    The last key in the sequence is used for the primary sort order, the
    second-to-last key for the secondary sort order, and so on.

    Args:
        keys ([torch.Tensor]): The :math:`k` different columns to be sorted.
            The last key is the primary sort key.
        dim (int, optional): The dimension to sort along. (default: :obj:`-1`)
        descending (bool, optional): Controls the sorting order (ascending or
            descending). (default: :obj:`False`)
    """
    assert len(keys) >= 1

    out = keys[0].argsort(dim=dim, descending=descending, stable=True)
    for k in keys[1:]:
        index = k.gather(dim, out)
        index = index.argsort(dim=dim, descending=descending, stable=True)
        out = out.gather(dim, index)
    assert out.min() >= 0 and out.max() < keys[0].shape[0], "lexsort returned invalid indices"
    return out


def permute_within_batch(batch):
    # Enumerate over unique batch indices
    unique_batches = torch.unique(batch)

    # Initialize list to store permuted indices
    permuted_indices = []

    for batch_index in unique_batches:
        # Extract indices for the current batch
        indices_in_batch = (batch == batch_index).nonzero().squeeze()

        # Permute indices within the current batch
        permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]

        # Append permuted indices to the list
        permuted_indices.append(permuted_indices_in_batch)

    # Concatenate permuted indices into a single tensor
    permuted_indices = torch.cat(permuted_indices)

    return permuted_indices


def scatter_mean_fallback(src, index, dim_size):
    out = src.new_zeros((dim_size, src.size(-1)))
    out.index_add_(0, index, src)
    count = src.new_zeros(dim_size)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    count = count.clamp_min(1.0).unsqueeze(-1)
    return out / count

class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type, num_heads,
                 pna_degrees=None, equivstable_pe=False, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, enable_reverse_mamba=False, fusion_mode='fixed',
                 fixed_weight=0.5, scan_target='node'):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.NUM_BUCKETS = 3
        self.enable_reverse_mamba = enable_reverse_mamba
        self.self_attn_reverse = None
        self.fusion_mode = fusion_mode
        self.fixed_weight = fixed_weight
        self.concat_proj = None
        self.gate_layer = None
        self.scan_target = scan_target
        self.edge_scan_input_proj = None

        # Local message-passing model.
        if local_gnn_type == 'None':
            self.local_model = None
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   nn.ReLU(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=16, # dim_h,
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=dropout,
                                             residual=True,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.self_attn = None
        elif global_model_type == 'Transformer':
            self.self_attn = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == 'Performer':
            self.self_attn = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        elif global_model_type == "BigBird":
            bigbird_cfg.dim_hidden = dim_h
            bigbird_cfg.n_heads = num_heads
            bigbird_cfg.dropout = dropout
            self.self_attn = SingleBigBirdLayer(bigbird_cfg)
        elif 'Mamba' in global_model_type:
            if global_model_type.split('_')[-1] == '2':
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=8,  # SSM state expansion factor
                        d_conv=4,    # Local convolution width
                        expand=2,    # Block expansion factor
                    )
            elif global_model_type.split('_')[-1] == '4':
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=4,  # SSM state expansion factor
                        d_conv=4,    # Local convolution width
                        expand=4,    # Block expansion factor
                    )
            elif global_model_type.split('_')[-1] == 'Multi':
                self.self_attn = nn.ModuleList()
                for i in range(4):
                    self.self_attn.append(Mamba(d_model=dim_h, # Model dimension d_model
                    d_state=16,  # SSM state expansion factor
                    d_conv=4,    # Local convolution width
                    expand=1,    # Block expansion factor
                    ))
            elif global_model_type.split('_')[-1] == 'SmallConv':
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=16,  # SSM state expansion factor
                        d_conv=2,    # Local convolution width
                        expand=1,    # Block expansion factor
                    )
            elif global_model_type.split('_')[-1] == 'SmallState':
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=8,  # SSM state expansion factor
                        d_conv=4,    # Local convolution width
                        expand=1,    # Block expansion factor
                    )
            else:
                self.self_attn = Mamba(d_model=dim_h, # Model dimension d_model
                        d_state=16,  # SSM state expansion factor
                        d_conv=4,    # Local convolution width
                        expand=1,    # Block expansion factor
                    )
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type

        # Initialize reverse Mamba and gate layer if enabled
        if self.enable_reverse_mamba and 'Mamba' in global_model_type:
            self.self_attn_reverse = Mamba(d_model=dim_h, d_state=16, d_conv=4, expand=1)
            if self.fusion_mode == 'gated':
                self.gate_layer = nn.Linear(dim_h * 2, dim_h)
            elif self.fusion_mode == 'concat':
                self.concat_proj = nn.Linear(dim_h * 2, dim_h)
        else:
            self.self_attn_reverse = None
            self.gate_layer = None
            self.concat_proj = None

        if self.scan_target == 'edge':
            self.edge_scan_input_proj = None

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            # self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
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
            # self.norm2 = pygnn.norm.LayerNorm(dim_h)
            self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection
        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                   x=h,
                                                   edge_index=batch.edge_index,
                                                   edge_attr=batch.edge_attr,
                                                   pe_EquivStableLapPE=es_data))
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.equivstable_pe:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr,
                                               batch.pe_EquivStableLapPE)
                else:
                    h_local = self.local_model(h, batch.edge_index, batch.edge_attr)
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            if self.scan_target not in ['node', 'edge']:
                raise ValueError(f"Unsupported scan_target: {self.scan_target}")
            if self.scan_target == 'edge' and 'Mamba' in self.global_model_type:
                h_attn = self._edge_mamba_scan(batch, h)
            elif self.global_model_type in ['Transformer', 'Performer', 'BigBird', 'Mamba']:
                h_dense, mask = to_dense_batch(h, batch.batch)
            if self.scan_target == 'node' and self.global_model_type == 'Transformer':
                h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.scan_target == 'node' and self.global_model_type == 'Performer':
                h_attn = self.self_attn(h_dense, mask=mask)[mask]
            elif self.scan_target == 'node' and self.global_model_type == 'BigBird':
                h_attn = self.self_attn(h_dense, attention_mask=mask)

            elif self.scan_target == 'node' and self.global_model_type == 'Mamba':
                h_attn = self.self_attn(h_dense)[mask]

            elif self.scan_target == 'node' and self.global_model_type == 'Mamba_Permute':
                h_ind_perm = permute_within_batch(batch.batch)
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif self.scan_target == 'node' and self.global_model_type == 'Mamba_Degree':
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                h_ind_perm = lexsort([deg, batch.batch])
                # 确保 h_ind_perm 有效
                valid_mask = h_ind_perm < len(batch.batch)
                h_ind_perm = h_ind_perm[valid_mask]
                batch_perm = batch.batch[h_ind_perm].contiguous()
                batch_perm_cpu = batch_perm.cpu()
                _, batch_perm_remapped = torch.unique(batch_perm_cpu, return_inverse=True)
                batch_perm_remapped = batch_perm_remapped.to(batch_perm.device)
                h_dense, mask = to_dense_batch(h[h_ind_perm], batch_perm_remapped)
                h_ind_perm_reverse = torch.argsort(h_ind_perm)
                if self.enable_reverse_mamba:
                    h_ind_perm_rev = torch.flip(h_ind_perm, dims=[0]).contiguous()
                    valid_mask_rev = h_ind_perm_rev < len(batch.batch)
                    h_ind_perm_rev = h_ind_perm_rev[valid_mask_rev]
                    batch_perm_rev = batch.batch[h_ind_perm_rev].contiguous()
                    batch_perm_rev_cpu = batch_perm_rev.cpu()
                    _, batch_perm_rev_remapped = torch.unique(batch_perm_rev_cpu, return_inverse=True)
                    batch_perm_rev_remapped = batch_perm_rev_remapped.to(batch_perm_rev.device)
                    h_dense_rev, mask_rev = to_dense_batch(h[h_ind_perm_rev], batch_perm_rev_remapped)
                    h_ind_perm_rev_reverse = torch.argsort(h_ind_perm_rev)
                    h_fwd = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                    h_rev = self.self_attn_reverse(h_dense_rev)[mask_rev][h_ind_perm_rev_reverse]
                    h_attn = self._fuse_mamba_outputs(h_fwd, h_rev, batch)
                else:
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]

            elif self.scan_target == 'node' and self.global_model_type == 'Mamba_Hybrid':
                if batch.split == 'train':
                    h_ind_perm = permute_within_batch(batch.batch)
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        h_ind_perm = permute_within_batch(batch.batch)
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.scan_target == 'node' and 'Mamba_Hybrid_Degree' == self.global_model_type:
                if batch.split == 'train':
                    h_ind_perm = permute_within_batch(batch.batch)
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        h_ind_perm = permute_within_batch(batch.batch)
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.scan_target == 'node' and 'Mamba_Hybrid_Degree_Noise' == self.global_model_type:
                if batch.split == 'train':
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    deg_noise = torch.rand_like(deg).to(deg.device)
                    h_ind_perm = lexsort([deg+deg_noise, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    if self.enable_reverse_mamba:
                        # Reverse permutation
                        h_ind_perm_rev = torch.flip(h_ind_perm, dims=[0])
                        h_dense_rev, _ = to_dense_batch(h[h_ind_perm_rev], batch.batch[h_ind_perm_rev])
                        h_ind_perm_rev_reverse = torch.argsort(h_ind_perm_rev)
                        # Forward Mamba
                        h_fwd = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        # Reverse Mamba
                        h_rev = self.self_attn_reverse(h_dense_rev)[mask][h_ind_perm_rev_reverse]
                        # Fuse
                        h_attn = self._fuse_mamba_outputs(h_fwd, h_rev, batch)
                    else:
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    for i in range(5):
                        deg_noise = torch.rand_like(deg).to(deg.device)
                        h_ind_perm = lexsort([deg+deg_noise, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        if self.enable_reverse_mamba:
                            # Reverse permutation
                            h_ind_perm_rev = torch.flip(h_ind_perm, dims=[0])
                            h_dense_rev, _ = to_dense_batch(h[h_ind_perm_rev], batch.batch[h_ind_perm_rev])
                            h_ind_perm_rev_reverse = torch.argsort(h_ind_perm_rev)
                            # Forward Mamba
                            h_fwd = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                            # Reverse Mamba
                            h_rev = self.self_attn_reverse(h_dense_rev)[mask][h_ind_perm_rev_reverse]
                            # Fuse
                            h_attn = self._fuse_mamba_outputs(h_fwd, h_rev, batch)
                        else:
                            h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.scan_target == 'node' and 'Mamba_Hybrid_Degree_Noise_Bucket' == self.global_model_type:
                if batch.split == 'train':
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                    deg_noise = torch.rand_like(deg).to(deg.device)
                    # deg_noise = torch.randn(deg.shape).to(deg.device)
                    deg = deg + deg_noise
                    indices_arr, emb_arr = [], []
                    bucket_assign = torch.randint_like(deg, 0, self.NUM_BUCKETS).to(deg.device)
                    for i in range(self.NUM_BUCKETS):
                        ind_i = (bucket_assign == i).nonzero().squeeze()
                        h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        h_dense = self._apply_mamba_bucket_scan(h_dense, mask, batch)
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    deg_ = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.float)
                    for i in range(5):
                        # deg_noise = torch.std(deg)*torch.randn(deg.shape).to(deg.device)
                        deg_noise = torch.rand_like(deg_).to(deg_.device)
                        # deg_noise = torch.randn(deg.shape).to(deg.device)
                        deg = deg_ + deg_noise
                        indices_arr, emb_arr = [], []
                        bucket_assign = torch.randint_like(deg, 0, self.NUM_BUCKETS).to(deg.device)
                        for i in range(self.NUM_BUCKETS):
                            ind_i = (bucket_assign == i).nonzero().squeeze()
                            h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            h_dense = self._apply_mamba_bucket_scan(h_dense, mask, batch)
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif 'Mamba_Hybrid_Noise' == self.global_model_type:
                if batch.split == 'train':
                    deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(batch.batch.device)
                    indices_arr, emb_arr = [],[]
                    bucket_assign = torch.randint_like(deg_noise, 0, self.NUM_BUCKETS).to(deg_noise.device)
                    for i in range(self.NUM_BUCKETS):
                        ind_i = (bucket_assign==i).nonzero().squeeze()
                        h_ind_perm_sort = lexsort([deg_noise[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        if self.enable_reverse_mamba:
                            h_ind_perm_rev = torch.flip(h_ind_perm_i, dims=[0])
                            h_dense_rev, mask_rev = to_dense_batch(h[h_ind_perm_rev], batch.batch[h_ind_perm_rev])
                            h_ind_perm_rev_reverse = torch.argsort(h_ind_perm_rev)
                            h_fwd = self.self_attn(h_dense)[mask]
                            h_rev = self.self_attn_reverse(h_dense_rev)[mask_rev][h_ind_perm_rev_reverse]
                            h_dense = self._fuse_mamba_outputs(h_fwd, h_rev, batch)
                        else:
                            h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    deg = batch.batch.to(torch.float)
                    for i in range(5):
                        deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(batch.batch.device)
                        indices_arr, emb_arr = [],[]
                        bucket_assign = torch.randint_like(deg_noise, 0, self.NUM_BUCKETS).to(deg_noise.device)
                        for i in range(self.NUM_BUCKETS):
                            ind_i = (bucket_assign==i).nonzero().squeeze()
                            h_ind_perm_sort = lexsort([deg_noise[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            if self.enable_reverse_mamba:
                                h_ind_perm_rev = torch.flip(h_ind_perm_i, dims=[0])
                                h_dense_rev, mask_rev = to_dense_batch(h[h_ind_perm_rev], batch.batch[h_ind_perm_rev])
                                h_ind_perm_rev_reverse = torch.argsort(h_ind_perm_rev)
                                h_fwd = self.self_attn(h_dense)[mask]
                                h_rev = self.self_attn_reverse(h_dense_rev)[mask_rev][h_ind_perm_rev_reverse]
                                h_dense = self._fuse_mamba_outputs(h_fwd, h_rev, batch)
                            else:
                                h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif 'Mamba_Hybrid_Noise_Bucket' == self.global_model_type:
                if batch.split == 'train':
                    deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(batch.batch.device)
                    h_ind_perm = lexsort([deg_noise, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    deg = batch.batch.to(torch.float)
                    for i in range(5):
                        deg_noise = torch.rand_like(batch.batch.to(torch.float)).to(batch.batch.device)
                        h_ind_perm = lexsort([deg_noise, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_Eigen':
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                centrality = batch.EigCentrality
                if batch.split == 'train':
                    # Shuffle within 1 STD
                    centrality_noise = torch.std(centrality)*torch.rand(centrality.shape).to(centrality.device)
                    # Order by batch, degree, and centrality
                    h_ind_perm = lexsort([centrality+centrality_noise, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        centrality_noise = torch.std(centrality)*torch.rand(centrality.shape).to(centrality.device)
                        h_ind_perm = lexsort([centrality+centrality_noise, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif 'Mamba_Eigen_Bucket' == self.global_model_type:
                centrality = batch.EigCentrality
                if batch.split == 'train':
                    centrality_noise = torch.std(centrality)*torch.rand(centrality.shape).to(centrality.device)
                    indices_arr, emb_arr = [],[]
                    bucket_assign = torch.randint_like(centrality, 0, self.NUM_BUCKETS).to(centrality.device)
                    for i in range(self.NUM_BUCKETS):
                        ind_i = (bucket_assign==i).nonzero().squeeze()
                        h_ind_perm_sort = lexsort([(centrality+centrality_noise)[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        centrality_noise = torch.std(centrality)*torch.rand(centrality.shape).to(centrality.device)
                        indices_arr, emb_arr = [],[]
                        bucket_assign = torch.randint_like(centrality, 0, self.NUM_BUCKETS).to(centrality.device)
                        for i in range(self.NUM_BUCKETS):
                            ind_i = (bucket_assign==i).nonzero().squeeze()
                            h_ind_perm_sort = lexsort([(centrality+centrality_noise)[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_RWSE':
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                RWSE_sum = torch.sum(batch.pestat_RWSE, dim=1)
                if batch.split == 'train':
                    # Shuffle within 1 STD
                    RWSE_noise = torch.std(RWSE_sum)*torch.randn(RWSE_sum.shape).to(RWSE_sum.device)
                    # Sort in descending order
                    # Nodes with more local connections -> larger sum in RWSE
                    # Nodes with more global connections -> smaller sum in RWSE
                    h_ind_perm = lexsort([-RWSE_sum+RWSE_noise, batch.batch])
                    # h_ind_perm = lexsort([-RWSE_sum+RWSE_noise, deg, batch.batch])
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    # Sort in descending order
                    # Nodes with more local connections -> larger sum in RWSE
                    # Nodes with more global connections -> smaller sum in RWSE
                    # h_ind_perm = lexsort([-RWSE_sum, deg, batch.batch])
                    mamba_arr = []
                    for i in range(5):
                        RWSE_noise = torch.std(RWSE_sum)*torch.randn(RWSE_sum.shape).to(RWSE_sum.device)
                        h_ind_perm = lexsort([-RWSE_sum+RWSE_noise, batch.batch])
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_Cluster':
                h_ind_perm = permute_within_batch(batch.batch)
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                if batch.split == 'train':
                    unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                    permuted_louvain = torch.zeros(batch.LouvainCluster.shape).long().to(batch.LouvainCluster.device)
                    random_permute = torch.randperm(unique_cluster_n+1).long().to(batch.LouvainCluster.device)
                    for i in range(unique_cluster_n):
                        indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                        permuted_louvain[indices] = random_permute[i]
                    #h_ind_perm_1 = lexsort([deg[h_ind_perm], permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]])
                    #h_ind_perm_1 = lexsort([permuted_louvain[h_ind_perm], deg[h_ind_perm], batch.batch[h_ind_perm]])
                    h_ind_perm_1 = lexsort([permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]])
                    h_ind_perm = h_ind_perm[h_ind_perm_1]
                    h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                else:
                    #h_ind_perm = lexsort([batch.LouvainCluster, deg, batch.batch])
                    #h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                    #h_ind_perm_reverse = torch.argsort(h_ind_perm)
                    #h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                    mamba_arr = []
                    for i in range(5):
                        unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                        permuted_louvain = torch.zeros(batch.LouvainCluster.shape).long().to(batch.LouvainCluster.device)
                        random_permute = torch.randperm(unique_cluster_n+1).long().to(batch.LouvainCluster.device)
                        for i in range(len(torch.unique(batch.LouvainCluster))):
                            indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                            permuted_louvain[indices] = random_permute[i]
                        # potentially permute it 5 times and average
                        # on the cluster level
                        #h_ind_perm_1 = lexsort([deg[h_ind_perm], permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]])
                        #h_ind_perm_1 = lexsort([permuted_louvain[h_ind_perm], deg[h_ind_perm], batch.batch[h_ind_perm]])
                        h_ind_perm_1 = lexsort([permuted_louvain[h_ind_perm], batch.batch[h_ind_perm]])
                        h_ind_perm = h_ind_perm[h_ind_perm_1]
                        h_dense, mask = to_dense_batch(h[h_ind_perm], batch.batch[h_ind_perm])
                        h_ind_perm_reverse = torch.argsort(h_ind_perm)
                        h_attn = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_Hybrid_Degree_Bucket':
                if batch.split == 'train':
                    h_ind_perm = permute_within_batch(batch.batch)
                    deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                    indices_arr, emb_arr = [],[]
                    for i in range(self.NUM_BUCKETS):
                        ind_i = h_ind_perm[h_ind_perm%self.NUM_BUCKETS==i]
                        h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        h_ind_perm = permute_within_batch(batch.batch)
                        deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                        indices_arr, emb_arr = [],[]
                        for i in range(self.NUM_BUCKETS):
                            ind_i = h_ind_perm[h_ind_perm%self.NUM_BUCKETS==i]
                            h_ind_perm_sort = lexsort([deg[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_Cluster_Bucket':
                h_ind_perm = permute_within_batch(batch.batch)
                deg = degree(batch.edge_index[0], batch.x.shape[0]).to(torch.long)
                if batch.split == 'train':
                    indices_arr, emb_arr = [],[]
                    unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                    permuted_louvain = torch.zeros(batch.LouvainCluster.shape).long().to(batch.LouvainCluster.device)
                    random_permute = torch.randperm(unique_cluster_n+1).long().to(batch.LouvainCluster.device)
                    for i in range(len(torch.unique(batch.LouvainCluster))):
                        indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                        permuted_louvain[indices] = random_permute[i]
                    for i in range(self.NUM_BUCKETS):
                        ind_i = h_ind_perm[h_ind_perm%self.NUM_BUCKETS==i]
                        h_ind_perm_sort = lexsort([permuted_louvain[ind_i], deg[ind_i], batch.batch[ind_i]])
                        h_ind_perm_i = ind_i[h_ind_perm_sort]
                        h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                        h_dense = self.self_attn(h_dense)[mask]
                        indices_arr.append(h_ind_perm_i)
                        emb_arr.append(h_dense)
                    h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                    h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                else:
                    mamba_arr = []
                    for i in range(5):
                        indices_arr, emb_arr = [],[]
                        unique_cluster_n = len(torch.unique(batch.LouvainCluster))
                        permuted_louvain = torch.zeros(batch.LouvainCluster.shape).long().to(batch.LouvainCluster.device)
                        random_permute = torch.randperm(unique_cluster_n+1).long().to(batch.LouvainCluster.device)
                        for i in range(len(torch.unique(batch.LouvainCluster))):
                            indices = torch.nonzero(batch.LouvainCluster == i).squeeze()
                            permuted_louvain[indices] = random_permute[i]
                        for i in range(self.NUM_BUCKETS):
                            ind_i = h_ind_perm[h_ind_perm%self.NUM_BUCKETS==i]
                            h_ind_perm_sort = lexsort([permuted_louvain[ind_i], deg[ind_i], batch.batch[ind_i]])
                            h_ind_perm_i = ind_i[h_ind_perm_sort]
                            h_dense, mask = to_dense_batch(h[h_ind_perm_i], batch.batch[h_ind_perm_i])
                            h_dense = self.self_attn(h_dense)[mask]
                            indices_arr.append(h_ind_perm_i)
                            emb_arr.append(h_dense)
                        h_ind_perm_reverse = torch.argsort(torch.cat(indices_arr))
                        h_attn = torch.cat(emb_arr)[h_ind_perm_reverse]
                        mamba_arr.append(h_attn)
                    h_attn = sum(mamba_arr) / 5

            elif self.global_model_type == 'Mamba_Augment':
                aug_idx, aug_mask = augment_seq(batch.edge_index, batch.batch, 3)
                h_dense, mask = to_dense_batch(h[aug_idx], batch.batch[aug_idx])
                aug_idx_reverse = torch.nonzero(aug_mask).squeeze()
                h_attn = self.self_attn(h_dense)[mask][aug_idx_reverse]
            elif self.scan_target == 'node':
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def _fuse_mamba_outputs(self, h_fwd, h_rev, batch):
        """Fuse forward and reverse Mamba outputs."""
        if h_fwd.shape != h_rev.shape:
            raise ValueError(
                f"Forward/Reverse Mamba output shape mismatch: {h_fwd.shape} vs {h_rev.shape}"
            )

        if self.fusion_mode == 'fixed':
            h_attn = self.fixed_weight * h_fwd + (1 - self.fixed_weight) * h_rev
        elif self.fusion_mode == 'gated':
            combined = torch.cat([h_fwd, h_rev], dim=-1)
            gate = torch.sigmoid(self.gate_layer(combined))
            h_attn = gate * h_fwd + (1 - gate) * h_rev
        elif self.fusion_mode == 'concat':
            combined = torch.cat([h_fwd, h_rev], dim=-1)
            if combined.size(-1) != self.dim_h * 2:
                raise ValueError(
                    f"Concat fusion expected last dim {self.dim_h * 2}, got {combined.size(-1)}"
                )
            h_attn = self.concat_proj(combined)
            if h_attn.size(-1) != self.dim_h:
                raise ValueError(
                    f"Concat projection expected output dim {self.dim_h}, got {h_attn.size(-1)}"
                )
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")
        return h_attn

    def _edge_mamba_scan(self, batch, h):
        if 'Mamba' not in self.global_model_type:
            raise ValueError("edge scan currently supports only Mamba-based global_model_type.")
        if not hasattr(batch, 'edge_attr') or batch.edge_attr is None:
            raise ValueError("edge scan requires batch.edge_attr.")

        edge_attr = batch.edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if self.edge_scan_input_proj is None:
            self.edge_scan_input_proj = nn.Linear(
                edge_attr.size(-1), self.dim_h, device=edge_attr.device
            )
        edge_feat = self.edge_scan_input_proj(edge_attr.float())

        edge_order = self._dfs_edge_order(batch.edge_index, batch.batch, h.size(0))
        edge_feat_perm = edge_feat[edge_order]
        edge_batch = batch.batch[batch.edge_index[0]][edge_order]

        edge_dense, edge_mask = to_dense_batch(edge_feat_perm, edge_batch)
        edge_out = self.self_attn(edge_dense)[edge_mask]
        edge_out = edge_out[torch.argsort(edge_order)]
        batch.edge_attr = edge_out

        src, dst = batch.edge_index[0], batch.edge_index[1]
        node_msg = scatter_mean_fallback(edge_out, src, dim_size=h.size(0))
        node_msg = node_msg + scatter_mean_fallback(edge_out, dst, dim_size=h.size(0))
        return node_msg

    def _dfs_edge_order(self, edge_index, node_batch, num_nodes):
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
                    neigh = adjacency[cur]
                    for nxt, eid in reversed(neigh):
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
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s

    def _apply_mamba_bucket_scan(self, h_dense, mask, batch):
        h_fwd = self.self_attn(h_dense)
        if not self.enable_reverse_mamba:
            return h_fwd[mask]
        h_rev = torch.flip(h_dense, dims=[1])
        reverse_mamba = getattr(self, 'self_attn_reverse', None)
        if reverse_mamba is None:
            raise ValueError("Independent reverse Mamba is required for Mamba_Hybrid_Degree_Noise_Bucket.")
        h_rev = reverse_mamba(h_rev)
        h_rev = torch.flip(h_rev, dims=[1])
        h_attn = self._fuse_mamba_outputs(h_fwd, h_rev, batch)
        return h_attn[mask]

    def _apply_mamba_scan_with_permutation(
            self, h, batch_index, h_ind_perm, h_dense, mask, h_ind_perm_reverse, batch):
        h_fwd = self.self_attn(h_dense)[mask][h_ind_perm_reverse]
        if not self.enable_reverse_mamba:
            return h_fwd
        h_ind_perm_rev = torch.flip(h_ind_perm, dims=[0])
        h_dense_rev, mask_rev = to_dense_batch(h[h_ind_perm_rev], batch_index[h_ind_perm_rev])
        h_ind_perm_rev_reverse = torch.argsort(h_ind_perm_rev)
        reverse_mamba = getattr(self, 'self_attn_reverse', None)
        reverse_mamba = reverse_mamba if reverse_mamba is not None else self.self_attn
        h_rev = reverse_mamba(h_dense_rev)[mask_rev][h_ind_perm_rev_reverse]
        return self._fuse_mamba_outputs(h_fwd, h_rev, batch)
