import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *


def get_model_instance(model_name, **model_params):
    """
    Get model from config file.
    Supports models from torchvision module.
    Example:
        get_model_instance('resnet101', num_classes=10)
    """
    try:
        model = eval(model_name)
    except NameError:
        raise NameError("Model '{model_name}' not found.".format(model_name=model_name))

    model_instance = model(**model_params)

    return model_instance


class MnistModel(nn.Module):
    def __init__(self, num_classes):
        super(MnistModel, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, self.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
