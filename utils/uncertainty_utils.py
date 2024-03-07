import torch

def calculate_entropy(probs, class_dim=1):
    """ Calculates entropy of seg masks. """
    p = probs
    eps = torch.finfo(probs.dtype).eps
    logp = torch.log(p + eps)
    plogp = p * logp
    entropy = -torch.sum(plogp, dim=class_dim)
    return entropy

def calculate_predictive_entropy(predictions):
    """
    Calculates predictive entropy for either: 
        - a batch of segmentation masks.
        - N batches of segmentation masks, where N is the number of MC samples.

    Args:
        predictions, of shape = [N, bs, K, h, w] or [bs, K, h, w]

    Returns:
        per-pixel predictive entropy, pe, of shape = [bs, h, w]
    """
    if len(predictions.shape) == 5:
        predictions = predictions.mean(0)

    pe =  calculate_entropy(predictions, class_dim=1)
    return pe

def calculate_mutual_information(predictions):
    """
    Calculates mutual information for N batches of segmentation masks.
    N is the number of MC samples.

    Args:
        predictions, i.e. N batches of seg masks of shape = [N, bs, K, h, w]

    Returns:
        per-pixel mutual information, mi, of shape = [bs, h, w]
    """

    mean_seg_masks = torch.mean(predictions, dim=0)
    predictive_entropy = calculate_entropy(mean_seg_masks, class_dim=1)

    # computing expectation of entropies
    p = predictions
    eps = torch.finfo(predictions.dtype).eps
    logp = torch.log(p + eps)
    plogp = p * logp
    exp_entropies = torch.mean(-torch.sum(plogp, dim=2), dim=0)

    # Computing mutual information
    mi = predictive_entropy - exp_entropies
    return mi