import os
import logging
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def log_model_summary(model, verbose=False):
    logger = logging.getLogger(model.__class__.__name__)
    if verbose:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info('Trainable parameters: {}'.format(params))
    logger.info(model)
