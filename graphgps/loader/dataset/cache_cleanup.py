import logging
import os
import os.path as osp
import pickle

import torch


def cleanup_broken_processed_files(processed_dir, processed_file_names):
    """Remove empty or unreadable cache markers left by interrupted processing."""
    if isinstance(processed_file_names, str):
        processed_file_names = [processed_file_names]

    broken_paths = []

    for filename in ['pre_transform.pt', 'pre_filter.pt']:
        path = osp.join(processed_dir, filename)
        if _is_broken_torch_file(path):
            broken_paths.append(path)

    for filename in processed_file_names:
        path = osp.join(processed_dir, filename)
        if osp.exists(path) and osp.getsize(path) == 0:
            broken_paths.append(path)

    for path in broken_paths:
        os.remove(path)

    if broken_paths:
        logging.warning("Removed broken processed cache files: %s",
                        ', '.join(broken_paths))


def _is_broken_torch_file(path):
    if not osp.exists(path):
        return False

    if osp.getsize(path) == 0:
        return True

    try:
        torch.load(path, map_location='cpu')
    except (EOFError, OSError, RuntimeError, pickle.UnpicklingError):
        return True

    return False
