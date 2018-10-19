import warnings
from sklearn.metrics import (
    accuracy_score,
    fbeta_score,
    hamming_loss,
    jaccard_similarity_score
)


def get_metric_functions(metric_names):
    """
    Get list of metrics from config file
    """
    metric_fns = []
    for metric_name in metric_names:
        if metric_name == 'accuracy':
            metric_fns.append(accuracy)
        elif metric_name == 'f_beta':
            metric_fns.append(f_beta)
        elif metric_name == 'ham_loss':
            metric_fns.append(ham_loss)
        elif metric_name == 'jaccard_similarity':
            metric_fns.append(jaccard_similarity)
        else:
            raise NameError("Metric '{metric}' not found.".format(metric=metric_name))
    return metric_fns


def accuracy(preds, targs, threshold=0.5):
    """
    Exact match accuracy classification score.
    The set of labels predicted for a sample (preds) must exactly match the
    corresponding set of labels (targs)
    Args:
        preds: predicted targets as returned by a model
        targs: ground truth target value
        threshold: threshold, default is 0.5

    """
    return accuracy_score(targs, preds)


def f_beta(preds, targs, threshold=0.5, beta=2):
    """
    F-beta score for multi-label classification.
    Args:
        preds: predicted targets as returned by a model
        targs: ground truth target value
        threshold: threshold, default is 0.5
        beta: determines the weight of precision.
        e.g if beta = 2, it is better to predict (false positive) than (false negative)

    """
    # TODO find more clever way to choose threshold and beta
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fbeta_score(targs, (preds > threshold), beta, average='samples')


def ham_loss(preds, targs, threshold=0.5):
    """
    Hamming loss for multi-label classification
    Args:
        preds: predicted targets as returned by a model
        targs: ground truth target value
        threshold: threshold, default is 0.5

    """
    return hamming_loss(targs, (preds > threshold))


def jaccard_similarity(preds, targs, threshold=0.5):
    """
    Jaccard similarity score for multi-label classification.
    Intersection over union
    Args:
        preds: predicted targets as returned by a model
        targs: ground truth target value
        threshold: threshold, default is 0.5

    """
    return jaccard_similarity_score(targs, (preds > threshold))
