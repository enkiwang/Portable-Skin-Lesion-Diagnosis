import torch
import torch.nn as nn
from torchvision import models

from mobilenet import MyMobilenet,extract_feats_mobilenet
from resnet import MyResnet, extract_feats_resnet50


_MODELS = ['resnet-50', 'mobilenet']

_NORM_AND_SIZE = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225], [224, 224]]

def set_model (model_name, num_class, neurons_reducer_block=0, comb_method=None, comb_config=None, pretrained=True,
         freeze_conv=False):

    if pretrained:
        pre_torch = True
    else:
        pre_torch = False

    if model_name not in _MODELS:
        raise Exception("The model {} is not available!".format(model_name))

    model = None
    if model_name == 'resnet-50':
        model = MyResnet(models.resnet50(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    elif model_name == 'mobilenet':
        model = MyMobilenet(models.mobilenet_v2(pretrained=pre_torch), num_class, neurons_reducer_block, freeze_conv,
                         comb_method=comb_method, comb_config=comb_config)

    return model


def get_activation_fn(model_name):
    """Get activations from (pretrained) models.
    model_name: name of a model to extract features.
    Return: activation function.
    """
    if model_name == 'resnet-50':
        activation_fn = extract_feats_resnet50
    elif model_name == 'mobilenet':
        activation_fn = extract_feats_mobilenet
    else:
        raise
    return activation_fn


def get_activations(model_name, model, data, student=True):
    """Get activations from (pretrained) models.
    model_name: name of a model to extract features.
    model: (pretrained) model.
    data: input data to model.
    student: whether this model is a studnet or not (detach for teacher).
    Return: activations, i.e., 7x7, and outputs from adapt_avg_pool, reducer_block, classifier.
    """
    
    activation_fn = get_activation_fn(model_name)
    activations = activation_fn(model, data=data, student=student)

    return activations


class ConvReg(nn.Module):
    """Convolutional regressions"""
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)