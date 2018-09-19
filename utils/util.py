import os
import glog as log
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def log_model_summary(model, verbose=False):
    if verbose:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        log.info('Trainable parameters: {}'.format(params))
    log.info(model)
