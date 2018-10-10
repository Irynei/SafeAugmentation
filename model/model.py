import model.architectures as custom_models
import torchvision.models as torchvision_models


def get_model_instance(model_name, **model_params):
    """
    Get model from config file.
    Supports models from torchvision module.
    Example:
        get_model_instance('resnet101', num_classes=10)
    """
    try:
        model = getattr(custom_models, model_name, None)
        if model is None:
            # get model from torchvision
            model = getattr(torchvision_models, model_name)

    except AttributeError:
        raise AttributeError("Model '{model_name}' not found".format(model_name=model_name))

    model_instance = model(**model_params)
    return model_instance
