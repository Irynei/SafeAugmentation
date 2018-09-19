from torch import nn


def get_loss_function(loss_function, **kwargs):
    """
    Get loss function instance from config file
    """
    try:
        loss = getattr(nn, loss_function)
    except AttributeError:
        raise AttributeError("Loss '{loss}' not found.".format(loss=loss_function))

    loss_instance = loss(**kwargs)
    return loss_instance
