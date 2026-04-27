import argparse
import pickle
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, mean_absolute_error
from tqdm import tqdm
import torch
import yaml
import pandas as pd

PREDICTION_KEYS = (
    'predictions', 'prediction', 'preds', 'pred', 'logits',
    'scores', 'y_pred', 'val_pred', 'test_pred'
)
TARGET_KEYS = (
    'targets', 'target', 'labels', 'label', 'y_true',
    'ground_truth', 'gt', 'val_targets', 'test_targets'
)

SPLIT_KEYS = ('val', 'valid', 'validation', 'test')

MODEL_CKPT_KEYS = {
    'state_dict', 'model_state_dict', 'model_state', 'optimizer_states',
    'optimizer_state_dict', 'lr_schedulers', 'scheduler_state_dict',
    'callbacks', 'epoch', 'global_step'
}

PREDICTION_FILENAMES = (
    'predictions.pt', 'predictions.pth', 'predictions.pkl',
    'preds.pt', 'preds.pth', 'logits.pt', 'scores.pt',
    'val_predictions.pt', 'test_predictions.pt'
)

def parse_args():
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="Fusion and Evaluation Script")
    parser.add_argument('--config', default=None, help='Path to config.yaml file')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument('--idx', default=None, type=int, help='Only for DHG datasets')
    parser.add_argument('--alpha', default=None, type=float, help='Weighted summation coefficient')
    parser.add_argument('--joint-dir', default=None, help='Path to joint feature predictions')
    parser.add_argument('--bone-dir', default=None, help='Path to bone feature predictions')
    parser.add_argument('--joint-motion-dir', default=None, help='Path to joint motion feature predictions')
    parser.add_argument('--bone-motion-dir', default=None, help='Path to bone motion feature predictions')
    parser.add_argument('--node-dir', default=None, help='Path to node feature predictions')
    parser.add_argument('--edge-dir', default=None, help='Path to edge feature predictions')

    # 解析初始命令行参数
    args = parser.parse_args()

    # 如果提供了 config 文件
    if args.config is not None:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file '{args.config}' not found!")

        with open(args.config, 'r') as f:
            config_args = yaml.load(f, yaml.FullLoader)

        # 检查 config 文件中的键是否有效
        valid_keys = vars(args).keys()
        for k in config_args.keys():
            if k not in valid_keys:
                raise ValueError(f"Unknown argument in config file: '{k}'")

        # 将配置文件中的默认值设置到命令行参数解析器
        parser.set_defaults(**config_args)

        # 再次解析参数，优先使用命令行参数
        args = parser.parse_args()


    return args



def load_labels(dataset, idx=1):
    """根据数据集加载标签"""
    if 'COCOSuperpixels' in dataset:
        with open('datasets/COCOSuperpixels/slic_compactness_30/edge_wt_region_boundary/raw/val.pickle', 'rb') as f:
            data_info = pickle.load(f)
            labels = data_info[0][-1]
    elif 'VOCSuperpixels' in dataset:
        with open('datasets/VOCSuperpixels/slic_compactness_30/edge_wt_region_boundary/raw/val.pickle', 'rb') as f:
            data_info = pickle.load(f)
            labels = data_info[0][-1]
    elif 'MalNetTiny' in dataset:
        val_txt = 'datasets/MalNetTiny/raw/split_info_tiny/type/val.txt'
        graph_root = 'datasets/MalNetTiny/raw/malnet-graphs-tiny'
        # 获取所有类别
        category_names = sorted([
            d for d in os.listdir(graph_root)
            if os.path.isdir(os.path.join(graph_root, d))
        ])
        category2idx = {name: idx for idx, name in enumerate(category_names)}
        with open(val_txt, 'r') as f:
            labels = []
            for line in f:
                rel_path = line.strip()
                category = rel_path.split('/')[0]  # 一级目录，即类别
                label = category2idx[category]
                labels.append(label)
    elif 'peptides-functional' in dataset:
        csv_path = 'datasets/peptides-functional/raw/peptide_multi_class_dataset.csv.gz'
        split_path = 'datasets/peptides-functional/splits_random_stratified_peptide.pickle'
        # 读取数据表
        df = pd.read_csv(csv_path)
        # 读取分割索引
        with open(split_path, 'rb') as f:
            splits = pickle.load(f)
        val_indices = splits['val']  # 整型索引
        # 把字符串标签转为numpy数组
        labels_raw = df.iloc[val_indices]['labels'].values  # 这里每个是string
        # 转化为one-hot numpy数组
        val_onehot = np.stack([np.array(eval(l)) for l in labels_raw])
        labels = val_onehot
    elif 'peptides-structural' in dataset:
        csv_path = 'datasets/peptides-structural/raw/peptide_structure_normalized_dataset.csv.gz'
        split_path = 'datasets/peptides-structural/splits_random_stratified_peptide_structure.pickle'
        # 读取数据表
        df = pd.read_csv(csv_path)
        # 读取分割索引
        with open(split_path, 'rb') as f:
            splits = pickle.load(f)
        val_indices = splits['val']  # 整型索引
        # 把字符串标签转为numpy数组
        labels_raw = df.iloc[val_indices]['labels'].values  # 这里每个是string
        # 转化为one-hot numpy数组
        val_onehot = np.stack([np.array(eval(l)) for l in labels_raw])
        labels = val_onehot
    else:
        raise NotImplementedError("Unsupported dataset!")
    return labels


def _is_numeric_scalar(value):
    return isinstance(value, (int, float, np.integer, np.floating))


def _normalize_prediction_item(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.numpy()
    if isinstance(value, np.ndarray):
        return value.item() if value.ndim == 0 else value
    if _is_numeric_scalar(value):
        return float(value)
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value)
        return arr.item() if arr.ndim == 0 else arr
    raise TypeError(f"Unsupported prediction item type: {type(value)}")


def _looks_like_prediction_values(values, sample_size=5):
    if not values:
        return False
    for value in values[:sample_size]:
        if isinstance(value, dict):
            return False
        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            continue
        if _is_numeric_scalar(value):
            continue
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue
            if len(value) == 2 and isinstance(value[0], (str, int, np.integer)):
                if isinstance(value[1], dict):
                    return False
            continue
        return False
    return True


def _load_torch_object(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _extract_predictions(obj, source):
    if isinstance(obj, dict):
        for key in PREDICTION_KEYS:
            if key in obj:
                return _extract_predictions(obj[key], f"{source}['{key}']")

        for key in SPLIT_KEYS:
            if key in obj:
                return _extract_predictions(obj[key], f"{source}['{key}']")

        if any(key in obj for key in MODEL_CKPT_KEYS):
            keys_preview = list(obj.keys())[:10]
            raise ValueError(
                f"{source} looks like a model checkpoint instead of predictions. "
                f"Top-level keys: {keys_preview}"
            )

        values = list(obj.values())
        if _looks_like_prediction_values(values):
            keys = list(obj.keys())
            if keys and all(isinstance(k, (int, np.integer)) for k in keys):
                values = [obj[k] for k in sorted(keys)]
            return [_normalize_prediction_item(value) for value in values]

        keys_preview = list(obj.keys())[:10]
        raise ValueError(
            f"Unable to locate predictions in {source}. Top-level keys: {keys_preview}"
        )

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return []
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in obj):
            return [_normalize_prediction_item(item[1]) for item in obj]
        return [_normalize_prediction_item(item) for item in obj]

    if torch.is_tensor(obj):
        obj = obj.detach().cpu()
        if obj.ndim == 0:
            raise ValueError(f"{source} is a scalar, not a batch of predictions")
        return [_normalize_prediction_item(item) for item in obj]

    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            raise ValueError(f"{source} is a scalar, not a batch of predictions")
        return [_normalize_prediction_item(item) for item in obj]

    raise ValueError(f"Unsupported checkpoint structure in {source}: {type(obj)}")


def _extract_targets(obj, source):
    if isinstance(obj, dict):
        for key in TARGET_KEYS:
            if key in obj:
                return _extract_targets(obj[key], f"{source}['{key}']")

        for key in SPLIT_KEYS:
            if key in obj:
                nested = _extract_targets(obj[key], f"{source}['{key}']")
                if nested is not None:
                    return nested

        return None

    if isinstance(obj, (list, tuple)):
        return [_normalize_prediction_item(item) for item in obj]

    if torch.is_tensor(obj):
        obj = obj.detach().cpu()
        if obj.ndim == 0:
            return [obj.item()]
        return [_normalize_prediction_item(item) for item in obj]

    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return [obj.item()]
        return [_normalize_prediction_item(item) for item in obj]

    return None


def _candidate_prediction_paths(dir_path, filename='predictions.pt'):
    candidates = []
    if os.path.isfile(dir_path):
        candidates.append(dir_path)
    else:
        normalized_dir = os.path.normpath(dir_path)
        base_name = os.path.basename(normalized_dir)

        if base_name == 'pt':
            run_dir = os.path.dirname(normalized_dir)
            for name in PREDICTION_FILENAMES:
                candidates.append(os.path.join(run_dir, name))

        for name in PREDICTION_FILENAMES:
            candidates.append(os.path.join(normalized_dir, name))

        candidates.append(os.path.join(normalized_dir, filename))

    seen = set()
    ordered = []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def load_predictions(dir_path, filename='predictions.pt'):
    """Load sample-level predictions from a checkpoint-like file."""
    errors = []
    for pt_path in _candidate_prediction_paths(dir_path, filename):
        if not os.path.exists(pt_path):
            continue
        try:
            pt = _load_torch_object(pt_path)
            predictions = _extract_predictions(pt, pt_path)
            if len(predictions) == 0:
                raise ValueError(f"No predictions found in {pt_path}")
            return predictions
        except Exception as exc:
            errors.append(f"{pt_path}: {exc}")

    if not errors:
        raise FileNotFoundError(
            f"No checkpoint or prediction file found under '{dir_path}'"
        )

    raise ValueError(
        "Unable to load sample-level predictions. Tried:\n" + "\n".join(errors)
    )


def load_predictions_and_targets(dir_path, filename='predictions.pt'):
    """Load predictions and optional targets from prediction artifacts."""
    errors = []
    for pt_path in _candidate_prediction_paths(dir_path, filename):
        if not os.path.exists(pt_path):
            continue
        try:
            pt = _load_torch_object(pt_path)
            predictions = _extract_predictions(pt, pt_path)
            if len(predictions) == 0:
                raise ValueError(f"No predictions found in {pt_path}")
            targets = _extract_targets(pt, pt_path)
            return predictions, targets
        except Exception as exc:
            errors.append(f"{pt_path}: {exc}")

    if not errors:
        raise FileNotFoundError(
            f"No checkpoint or prediction file found under '{dir_path}'"
        )

    raise ValueError(
        "Unable to load sample-level predictions/targets. Tried:\n" + "\n".join(errors)
    )


def compute_metric(labels, predictions, dataset):
    """根据数据集计算相应指标"""
    predictions = np.array(predictions)

    if dataset in ['COCOSuperpixels', 'VOCSuperpixels']:
        pred_classes = np.argmax(predictions, axis=1)
        f1 = f1_score(labels, pred_classes, average='macro')
        return f1, 0
    elif dataset == 'MalNetTiny':
        pred_classes = np.argmax(predictions, axis=1)
        acc = accuracy_score(labels, pred_classes)
        return acc, 0
    elif dataset == 'peptides-functional':
        ap = average_precision_score(labels, predictions, average='macro')
        return ap, 0
    elif dataset == 'peptides-structural':
        mae = mean_absolute_error(labels, predictions)
        return mae, 0
    else:
        # 默认 Accuracy
        pred_classes = np.argmax(predictions, axis=1)
        acc = accuracy_score(labels, pred_classes)
        return acc, 0


import itertools
from tqdm import tqdm

def main():
    args = parse_args()
    is_minimize = (args.dataset == 'peptides-structural')
    best_metric = float('inf') if is_minimize else 0
    best_alphas = [0.5, 0.5]

    # Load labels
    print(args)
    labels = load_labels(args.dataset, idx=args.idx)

    # Load predictions
    if 'COCOSuperpixels' in args.dataset:
        nodedir = f'results/cocosuperpixels-EX-bi/0'
        edgedir = f'results/cocosuperpixels-EX-bi-edge/0'

        r1, t1 = load_predictions_and_targets(nodedir)
        r2, t2 = load_predictions_and_targets(edgedir)

    elif 'VOCSuperpixels' in args.dataset:
        nodedir = f'results/voc_mhdn/0'
        edgedir = f'results/voc_mhdnb/0'

        r1, t1 = load_predictions_and_targets(nodedir)
        r2, t2 = load_predictions_and_targets(edgedir)

    elif "MalNetTiny" in args.dataset:
        nodedir = f'results/malnettiny-EX-bi/0'
        edgedir = f'results/malnettiny-EX-bi-edge/0'

        r1, t1 = load_predictions_and_targets(nodedir)
        r2, t2 = load_predictions_and_targets(edgedir)

    elif "peptides-functional" in args.dataset:
        nodedir = f'results/peptides-func-EX-bi/0'
        edgedir = f'results/peptides-func-EX-bi-edge/0'

        r1, t1 = load_predictions_and_targets(nodedir)
        r2, t2 = load_predictions_and_targets(edgedir)

    elif "peptides-structural" in args.dataset:
        nodedir = f'results/peptides-struct-EX-bi/0'
        edgedir = f'results/peptides-struct-EX-bi-edge/0'

        r1, t1 = load_predictions_and_targets(nodedir)
        r2, t2 = load_predictions_and_targets(edgedir)

    else:
        # Only load node and edge predictions
        r1, t1 = load_predictions_and_targets(args.node_dir) if args.node_dir else (None, None)
        r2, t2 = load_predictions_and_targets(args.edge_dir) if args.edge_dir else (None, None)

    # Prefer sample-level targets saved together with predictions.
    # This avoids graph-level/node-level mismatch in datasets like VOC/COCO Superpixels.
    if t1 is not None and (r1 is None or len(t1) == len(r1)):
        if len(labels) != len(t1):
            print(
                f"[Info] Replace labels loaded from dataset ({len(labels)}) "
                f"with artifact targets ({len(t1)}) for sample-level alignment."
            )
            labels = np.array(t1)

    if t2 is not None and r2 is not None and len(t2) != len(r2):
        raise ValueError(
            f"Edge predictions/targets length mismatch: pred={len(r2)} target={len(t2)}"
        )

    # Fusion and evaluation: Coarse Search
    coarse_step = 0.2
    fine_step = 0.05
    print(f"length of the labels : {len(labels)}")
    print(f"pred length r1: {len(r1)},r2: {len(r2)} ")
    if 'dhg' not in args.dataset:
        lengths = [len(r) for r in [r1, r2] if r is not None]
        assert all(length == len(labels) for length in lengths)
    alphas_list = list(itertools.product(np.arange(0.0, 1.1, coarse_step), repeat=2))  # 只 node 和 edge
    best_acc = 0
    best_alphas = [0.5, 0.5]  # 初始化为两个权重

    print("Starting coarse search...")
    for alphas in tqdm(alphas_list, desc="Coarse search"):
        if sum(alphas) == 0:  # 避免所有权重为零的情况
            continue

        preds = []
        for i in range(len(labels)):
            r11 = r1[i] if r1 else 0
            r22 = r2[i] if r2 else 0
            fused_scores = r11 * alphas[0] + r22 * alphas[1]
            preds.append(fused_scores)

        # Compute accuracy
        top1_metric, _ = compute_metric(labels, preds, args.dataset)
        if (is_minimize and top1_metric < best_metric) or (not is_minimize and top1_metric > best_metric):
            best_metric = top1_metric
            best_alphas = alphas

    print(f"Coarse search completed. Best Metric: {best_metric}")
    print(f"Coarse Best Alphas: {best_alphas}")

    # Fine Search
    fine_ranges = [np.arange(max(0.0, alpha - coarse_step), min(1.0, alpha + coarse_step) + fine_step, fine_step)
                   for alpha in best_alphas]
    fine_alphas_list = list(itertools.product(*fine_ranges))

    print("Starting fine search...")
    for alphas in tqdm(fine_alphas_list, desc="Fine search"):
        if sum(alphas) == 0:  # 避免所有权重为零的情况
            continue

        preds = []
        for i in range(len(labels)):
            r11 = r1[i] if r1 else 0
            r22 = r2[i] if r2 else 0
            fused_scores = r11 * alphas[0] + r22 * alphas[1]
            preds.append(fused_scores)

        # Compute accuracy
        top1_metric, _ = compute_metric(labels, preds, args.dataset)
        if (is_minimize and top1_metric < best_metric) or (not is_minimize and top1_metric > best_metric):
            best_metric = top1_metric
            best_alphas = alphas

    print(f"Fine search completed. Best Metric: {best_metric}")
    print(f"Fine Best Alphas: {best_alphas}")

    # 计算各个模态的acc:
    npred, epred = [], []
    for i in range(len(labels)):
        r11 = r1[i] if r1 else 0
        r22 = r2[i] if r2 else 0
        npred.append(r11)
        epred.append(r22)

    n_top1_acc, n_top5_acc = compute_metric(labels, npred, args.dataset)
    e_top1_acc, e_top5_acc = compute_metric(labels, epred, args.dataset)
    print(args.dataset)
    print(args.idx)
    print(f'node acc: {n_top1_acc}\nedge acc: {e_top1_acc}\n')
    print(f"best_metric : {best_metric}")


if __name__ == "__main__":
    main()
    # configs/ensemble.yaml
