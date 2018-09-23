
def get_metric_functions(metric_names):
    """
    Get list of metrics from config file
    """
    try:
        metric_fns = []
    except NameError as e:
        raise NameError("One of metric functions ({metric_names}) not found.".format(metric_names=metric_names))

    return metric_fns

# TODO all metrics here
