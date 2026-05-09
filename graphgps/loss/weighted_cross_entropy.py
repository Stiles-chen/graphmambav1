import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('weighted_cross_entropy')
def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    if cfg.model.loss_fun == 'weighted_cross_entropy':
        # Some datasets (e.g., VOCSuperpixels) may contain ignore labels (255)
        # or other out-of-range values. Mask them out to avoid invalid bincount
        # / indexing and to prevent NaNs.
        n_classes = pred.shape[1] if pred.ndim > 1 else 2
        valid = (true >= 0) & (true < n_classes)
        if not bool(valid.all()):
            pred = pred[valid]
            true = true[valid]

        V = true.numel()
        if V == 0:
            # Nothing to supervise in this batch. Return a well-defined zero loss.
            zero = pred.sum() * 0.0
            return zero, pred

        # calculating label weights for weighted loss computation
        label_count = torch.bincount(true, minlength=n_classes).to(pred.device)
        weight = (V - label_count).float() / float(V)
        weight *= (label_count > 0).float()
        # multiclass
        if pred.ndim > 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true, weight=weight), pred
        # binary
        else:
            loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                      weight=weight[true])
            return loss, torch.sigmoid(pred)
