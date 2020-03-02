import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import torchvision.models as tvm

from dltranz.metric_learn.ml_models import L2Normalization
from domyshnik.constants import *

class MnistClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        if len(x.size()) == 4:
            x = x[:, -1, :, :].unsqueeze(1) # get augmented image
        else:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
# --------------------------------------------------------------------------------------------------

class MnistMetricLearningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 256)
        self.fc2 = nn.Linear(256, 32)
        self.norm = L2Normalization()

    def forward(self, x):
        x = x.view(-1, 1, x.size(-2), x.size(-1)) # b, augs, x, y -> b*augs, 1, x, y
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.norm(x)
        return x

class MnistMetricLearningNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 256)

    def forward(self, x):
        x = x.view(-1, 1, x.size(-2), x.size(-1)) # b, augs, x, y -> b*augs, 1, x, y
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return x

class MnistMetricLearningNet3(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = tvm.resnet50()
        self.f = nn.ReLU()
        self.fc = nn.Linear(1000, 128)
        self.norm = L2Normalization()

    def forward(self, x):
        x = x.view(-1, 1, x.size(-2), x.size(-1)) # b, augs, x, y -> b*augs, 1, x, y
        x = x.repeat(1, 3, 1, 1)
        x = self.w(x)
        #x = self.f(self.fc(x))
        x = self.norm(self.fc(self.f(x)))
        return x

class Cifar10MetricLearningNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = tvm.resnet50()
        self.f = nn.ReLU()
        self.fc = nn.Linear(1000, 128)
        self.norm = L2Normalization()

    def forward(self, x):
        x = x.view(-1, 3, x.size(-2), x.size(-1)) # b, augs, 3, x, y -> b*augs, 3, x, y
        x = self.w(x)
        x = self.norm(self.fc(self.f(x)))
        return x

class Cifar10MetricLearningNet2(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = tvm.resnet50()
        self.f = nn.ReLU()
        self.fc = nn.Linear(1000, 256)
        self.norm = L2Normalization()

    def forward(self, x):
        x = x.view(-1, 3, x.size(-2), x.size(-1)) # b, augs, 3, x, y -> b*augs, 3, x, y
        x = self.w(x)
        x = self.norm(self.fc(self.f(x)))
        return x

# --------------------------------------------------------------------------------------------------

class MnistClassificationMetricLearningModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.metric_learn_model = get_mnist_metriclearning_model()
        self.metric_learn_model.train()
        for param in self.metric_learn_model.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(32, 10)

    def forward(self, x):
        if len(x.size()) == 4:
            x = x[:, -1, :, :].unsqueeze(1) # get augmented image
        else:
            x = x.unsqueeze(1)
        x = self.metric_learn_model(x)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

class Cifar10ClassificationMetricLearningModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.metric_learn_model = get_cifar10_metriclearning_persample_model()
        self.metric_learn_model.train()
        for param in self.metric_learn_model.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        if len(x.size()) == 5: # b, n_augs, c=3 , w, h
            x = x[:, -1, :, :, :].unsqueeze(1) # get augmented image
        else: # b, c=3 , w, h
            x = x.unsqueeze(1)
        x = self.metric_learn_model(x)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output

# --------------------------------------------------------------------------------------------------

class MnistDomyshnikNetNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.metric_learn_model = get_mnist_metriclearning_model()
        self.metric_learn_model = get_mnist_metriclearning_model_cated()
        #self.metric_learn_model = MnistMetricLearningNet2()
        self.metric_learn_model.train()
        for param in self.metric_learn_model.parameters():
            param.requires_grad = False

        self.fc0 = nn.Linear(256, 512)
        self.fc1_ = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 1, x.size(-2), x.size(-1)) # b, augs, x, y -> b*augs, 1, x, y
        x = self.metric_learn_model(x)
        x = F.relu( self.fc0(x))
        x = self.fc1_(x)
        output = F.log_softmax(x, dim=1)
        return output

class MnistDomyshnikNetNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric_learn_model = MnistMetricLearningNet2()

        self.fc0 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 1, x.size(-2), x.size(-1)) # b, augs, x, y -> b*augs, 1, x, y
        x = self.metric_learn_model(x)
        x = F.relu(x)
        x = self.fc0(x)
        output = F.log_softmax(x, dim=1)
        return output

class MnistDomyshnikNetNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric_learn_model = get_mnist_metriclearning_persample_model()
        self.metric_learn_model.train()
        self.metric_learn_model = nn.Sequential(*list(self.metric_learn_model.children())[:-3])
        for param in self.metric_learn_model.parameters():
            param.requires_grad = False
        self.dropout_base = nn.Dropout(0.25)

        self.fc0 = nn.Linear(1000, 512)
        self.dropout0 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        # ? maybe this is mistake because self,metric_learn_model do this again
        x = x.view(-1, 1, x.size(-2), x.size(-1)) # b, augs, x, y -> b*augs, 1, x, y
        x = x.repeat(1, 3, 1, 1)
        x = self.metric_learn_model(x)
        x = self.dropout_base(x)
        x = F.relu(x)

        x = self.fc0(x)
        x = self.dropout0(x)
        x = F.relu(x)

        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x

class Cifar10DomyshnikNetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.metric_learn_model = get_cifar10_metriclearning_persample_model()
        self.metric_learn_model.train()
        self.metric_learn_model = nn.Sequential(*list(self.metric_learn_model.children())[:-3])
        for param in self.metric_learn_model.parameters():
            param.requires_grad = False
        self.dropout_base = nn.Dropout(0.25)

        self.fc0 = nn.Linear(1000, 512)
        self.dropout0 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 3, x.size(-2), x.size(-1)) # b, augs, 3, x, y -> b*augs, 3, x, y
        x = self.metric_learn_model(x)
        x = self.dropout_base(x)
        x = F.relu(x)

        x = self.fc0(x)
        x = self.dropout0(x)
        x = F.relu(x)

        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x

    def allow_grads(self):
        for param in self.metric_learn_model.parameters():
            param.requires_grad = True

# --------------------------------------------------------------------------------------------------

def save_model_params(model, model_name):
    pth = osp.join(WEIGHTS_PATH, MODEL_POSTFIX + '_' + model_name)
    torch.save(model.state_dict(), pth)
    print(f'model saved to {pth}')

def load_model_params(model, model_name, postfx=None):
    postfix = MODEL_POSTFIX if postfx is None else postfx
    pth = osp.join(WEIGHTS_PATH, postfix + '_' + model_name)
    model.load_state_dict(torch.load(pth))
    model.eval()
    print(f'load model {pth}')
    return model

def get_mnist_metriclearning_persample_model():
    model = MnistMetricLearningNet3()
    return load_model_params(model, 'mnist_metric_learning.w', 'mnist_per_sampl')

def get_mnist_metriclearning_model():
    model = MnistMetricLearningNet()
    return load_model_params(model, 'mnist_metric_learning.w')

def get_mnist_metriclearning_model_cated():
    tmp = get_mnist_metriclearning_model()
    model = MnistMetricLearningNet2()
    model.load_state_dict(tmp.state_dict(), strict=False)
    return model

def get_mnist_domyshnik_model():
    model = MnistDomyshnikNetNet()
    return load_model_params(model, 'mnist_domushnik.w')

def get_cifar10_metriclearning_persample_model():
    model = Cifar10MetricLearningNet2()
    return load_model_params(model, 'cifar10_metric_learning.w', 'cifar10_per_sampl_2')
