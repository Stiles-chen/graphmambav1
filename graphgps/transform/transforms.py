import logging

import torch
from torch_geometric.utils import subgraph
from tqdm import tqdm


def pre_transform_in_memory(dataset, transform_func, show_progress=False):
    """Pre-transform already loaded PyG dataset object.

    Apply transform function to a loaded PyG dataset object so that
    the transformed result is persistent for the lifespan of the object.
    This means the result is not saved to disk, as what PyG's `pre_transform`
    would do, but also the transform is applied only once and not at each
    data access as what PyG's `transform` hook does.

    Implementation is based on torch_geometric.data.in_memory_dataset.copy

    Args:
        dataset: PyG dataset object to modify
        transform_func: transformation function to apply to each data example
        show_progress: show tqdm progress bar
    """
    if transform_func is None:
        return dataset

    data_list = [transform_func(dataset.get(i))
                 for i in tqdm(range(len(dataset)),
                               disable=not show_progress,
                               mininterval=10,
                               miniters=len(dataset)//20)]
    data_list = list(filter(None, data_list))

    dataset._indices = None
    dataset._data_list = data_list
    dataset.data, dataset.slices = dataset.collate(data_list)


def generate_splits(data, g_split):
  n_nodes = len(data.x)
  train_mask = torch.zeros(n_nodes, dtype=bool)
  valid_mask = torch.zeros(n_nodes, dtype=bool)
  test_mask = torch.zeros(n_nodes, dtype=bool)
  idx = torch.randperm(n_nodes)
  val_num = test_num = int(n_nodes * (1 - g_split) / 2)
  train_mask[idx[val_num + test_num:]] = True
  valid_mask[idx[:val_num]] = True
  test_mask[idx[val_num:val_num + test_num]] = True
  data.train_mask = train_mask
  data.val_mask = valid_mask
  data.test_mask = test_mask
  return data


def typecast_x(data, type_str):
    if type_str == 'float':
        data.x = data.x.float()
    elif type_str == 'long':
        data.x = data.x.long()
    else:
        raise ValueError(f"Unexpected type '{type_str}'.")
    return data


def concat_x_and_pos(data):
    data.x = torch.cat((data.x, data.pos), 1)
    return data

def move_node_feat_to_x(data):
    """For ogbn-proteins, move the attribute node_species to attribute x."""
    data.x = data.node_species
    return data


def add_dfs_edge_order(data):
    """Precompute and cache DFS edge traversal order for each graph sample."""
    if not hasattr(data, 'edge_index') or data.edge_index is None:
        return data
    if data.edge_index.numel() == 0:
        data.dfs_edge_order = torch.empty(0, dtype=torch.long)
        return data

    if hasattr(data, 'num_nodes') and data.num_nodes is not None:
        num_nodes = int(data.num_nodes)
    elif hasattr(data, 'x') and data.x is not None:
        num_nodes = int(data.x.size(0))
    else:
        num_nodes = int(data.edge_index.max().item()) + 1

    edge_index = data.edge_index.cpu()
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    adjacency = [[] for _ in range(num_nodes)]
    for eid, (u, v) in enumerate(zip(src, dst)):
        if 0 <= u < num_nodes:
            adjacency[u].append((v, eid))

    order, used_edges = [], set()
    for start in range(num_nodes):
        visited_nodes = set()
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
                if 0 <= nxt < num_nodes and nxt not in visited_nodes:
                    stack.append(nxt)

    if len(order) < data.edge_index.size(1):
        all_edges = set(range(data.edge_index.size(1)))
        order.extend(sorted(list(all_edges - used_edges)))
    data.dfs_edge_order = torch.tensor(order, dtype=torch.long)
    return data


def add_dfs_node_order(data):
    """Precompute and cache DFS node traversal order for each graph sample."""
    if hasattr(data, 'num_nodes') and data.num_nodes is not None:
        num_nodes = int(data.num_nodes)
    elif hasattr(data, 'x') and data.x is not None:
        num_nodes = int(data.x.size(0))
    else:
        return data
    if num_nodes == 0:
        data.dfs_node_order = torch.empty(0, dtype=torch.long)
        return data
    if not hasattr(data, 'edge_index') or data.edge_index is None or data.edge_index.numel() == 0:
        data.dfs_node_order = torch.arange(num_nodes, dtype=torch.long)
        return data

    edge_index = data.edge_index.cpu()
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    adjacency = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            adjacency[u].append(v)

    order, visited = [], set()
    for start in range(num_nodes):
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
                if nxt not in visited:
                    stack.append(nxt)
    if len(order) < num_nodes:
        missing = [n for n in range(num_nodes) if n not in visited]
        order.extend(missing)
    data.dfs_node_order = torch.tensor(order, dtype=torch.long)
    return data

def clip_graphs_to_size(data, size_limit=5000):
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    if N <= size_limit:
        return data
    else:
        logging.info(f'  ...clip to {size_limit} a graph of size: {N}')
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None
        edge_index, edge_attr = subgraph(list(range(size_limit)),
                                         data.edge_index, edge_attr)
        if hasattr(data, 'x'):
            data.x = data.x[:size_limit]
            data.num_nodes = size_limit
        else:
            data.num_nodes = size_limit
        if hasattr(data, 'node_is_attributed'):  # for ogbg-code2 dataset
            data.node_is_attributed = data.node_is_attributed[:size_limit]
            data.node_dfs_order = data.node_dfs_order[:size_limit]
            data.node_depth = data.node_depth[:size_limit]
        data.edge_index = edge_index
        if hasattr(data, 'edge_attr'):
            data.edge_attr = edge_attr
        return data
