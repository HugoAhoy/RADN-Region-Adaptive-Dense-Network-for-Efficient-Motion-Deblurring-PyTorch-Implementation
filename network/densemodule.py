import torch
import torch.nn as nn
from collections import OrderedDict

class Bottleneck(nn.Module):
    def __init__(self, inchannel, growthrate, bn_size):
        super(Bottleneck, self).__init__()
        self.innerchannel = growthrate*bn_size
        self.bn = nn.BatchNorm2d(inchannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inchannel, self.innerchannel, kernel_size=1, bias=False)

    def forward(self, *inputs):
        concat_input = torch.cat(inputs, 1)
        x1 = self.bn(concat_input)
        x2 = self.relu(x1)
        output = self.conv(x2)
        return output

class DenseLayer(nn.Module):
    def __init__(self, inchannel, growthrate):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(inchannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inchannel, growthrate, kernel_size=3, padding = 1, bias=False)

    def forward(self, *inputs):
        if len(inputs) == 1:
            concat_input = inputs[0]
        else:
            concat_input = torch.cat(inputs, 1)
        x1 = self.bn(concat_input)
        x2 = self.relu(x1)
        output = self.conv(x2)
        return output

class DenseLayer_B(nn.Module):
    def __init__(self, num_feature_map, growthrate, bn_size):
        super(DenseLayer_B, self).__init__()
        self.bottleneck=Bottleneck(num_feature_map, growthrate, bn_size)
        self.vanilladenselayer=DenseLayer(growthrate*bn_size, growthrate)

    def forward(self, *inputs):
        x = self.bottleneck(*inputs)
        out = self.vanilladenselayer(x)
        return out

class DenseBlock(nn.Module):
    def __init__(self, num_layer, growthrate, num_input_features, bn_size):
        self.num_layer = num_layer
        super(DenseBlock, self).__init__()
        for i in range(num_layer):
            layer = DenseLayer_B(num_input_features + i*growthrate, growthrate, bn_size)
            self.add_module('denselayer{}'.format(i), layer)


    def forward(self, input):
        features = [input]
        for i in range(self.num_layer):
            output = getattr(self, 'denselayer{}'.format(i))(*features)
            features.append(output)
        return torch.cat(features, 1)

class Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.transition = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False)),
            ('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

    def forward(self, input):
        return self.transition(input)
