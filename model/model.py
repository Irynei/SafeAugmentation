from torch import nn
import torch.nn.functional as F
import torchvision.models as models


def get_model_instance(model_name, **model_params):
    """
    Get model from config file.
    Supports models from torchvision module.
    Example:
        get_model_instance('resnet101', num_classes=10)
    """
    try:
        if model_name == 'VGG16_32x32':
            model = VGG16_32x32
        elif model_name == 'MnistModel':
            model = MnistModel
        else:
            # get model from torchvision
            model = getattr(models, model_name)

    except AttributeError:
        raise AttributeError("Model '{model_name}' not found".format(model_name=model_name))

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


class VGG16_32x32(nn.Module):
    """ VGG16 that works with 32x32 input """
    def __init__(self, num_classes):
        super(VGG16_32x32, self).__init__()
        self.layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers(self.layers)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
