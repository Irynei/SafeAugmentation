from torch import nn


def get_loss_function(loss_function, **kwargs):
    """
    Get loss function instance from config file
    """
    try:
        loss = eval(loss_function)
    except NameError:
        raise NameError("Loss '{loss}' not found.".format(loss=loss_function))

    loss_instance = loss(**kwargs)

    return loss_instance


def multi_label_soft_margin_loss():
    return nn.MultiLabelSoftMarginLoss()
