import logging
import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

"""
=== Description of the VOCSuperpixels dataset === 
Each graph is a tuple (x, edge_attr, edge_index, y)
Shape of x : [num_nodes, 14]
Shape of edge_attr : [num_edges, 1] or [num_edges, 2]
Shape of edge_index : [2, num_edges]
Shape of y : [num_nodes]
"""

VOC_node_input_dim = 14
# VOC_edge_input_dim = 1 or 2; defined in class VOCEdgeEncoder

@register_node_encoder('VOCNode')
class VOCNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.encoder = torch.nn.Linear(VOC_node_input_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # Defensive: some VOC superpixel feature files may contain NaN/Inf.
        # If they propagate into BatchNorm/Mamba, loss can become NaN at epoch 0.
        batch.x = torch.nan_to_num(batch.x, nan=0.0, posinf=1e4, neginf=-1e4)
        batch.x = self.encoder(batch.x)
        batch.x = torch.nan_to_num(batch.x, nan=0.0, posinf=1e4, neginf=-1e4)

        return batch


@register_edge_encoder('VOCEdge')
class VOCEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        VOC_edge_input_dim = 2 if cfg.dataset.name == 'edge_wt_region_boundary' else 1
        self.encoder = torch.nn.Linear(VOC_edge_input_dim, emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        if batch.edge_attr is not None:
            edge_attr = torch.nan_to_num(batch.edge_attr, nan=0.0, posinf=1e4, neginf=-1e4)

            # VOC-specific stabilization:
            # 1) hard-clip extreme raw edge features,
            # 2) standardize per mini-batch,
            # 3) clamp normalized values to a safe range.
            edge_attr = torch.clamp(edge_attr, min=-1e3, max=1e3)
            mean = edge_attr.mean(dim=0, keepdim=True)
            std = edge_attr.std(dim=0, unbiased=False, keepdim=True)
            edge_attr = (edge_attr - mean) / (std + 1e-6)
            edge_attr = torch.clamp(edge_attr, min=-10.0, max=10.0)

            if not hasattr(self, '_voc_edge_stats_logged'):
                self._voc_edge_stats_logged = True
                logging.info(
                    "VOCEdgeEncoder normalized edge_attr stats: min=%.4f max=%.4f mean=%.4f std=%.4f",
                    edge_attr.min().item(),
                    edge_attr.max().item(),
                    edge_attr.mean().item(),
                    edge_attr.std(unbiased=False).item(),
                )

            batch.edge_attr = self.encoder(edge_attr)
            batch.edge_attr = torch.nan_to_num(batch.edge_attr, nan=0.0, posinf=1e4, neginf=-1e4)
        return batch
